use ahash::{AHashMap, AHashSet};
use pyo3::prelude::*;
use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap};
use std::rc::Rc;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub struct Pair(usize, usize);

#[derive(Debug, Clone)]
struct Node {
    token_id: usize,
    list_idx: usize,
    prev: Option<usize>,
    next: Option<usize>,
}

#[derive(Debug, Clone)]
struct PretokenList {
    tail: Option<usize>,
    freq: isize,
}

#[derive(Debug, Eq, PartialEq, Clone)]
struct Rank {
    freq: isize,
    pair_bytes: (Rc<[u8]>, Rc<[u8]>),
    pair_id: Pair,
}

impl Ord for Rank {
    fn cmp(&self, other: &Self) -> Ordering {
        self.freq
            .cmp(&other.freq)
            .then_with(|| self.pair_bytes.cmp(&other.pair_bytes))
    }
}

impl PartialOrd for Rank {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

type VocabBytes = HashMap<usize, Vec<u8>>;
type Occurrences = AHashMap<Pair, isize>;
type PairMap = AHashMap<Pair, AHashSet<usize>>;
type RankQueue = BTreeSet<Rank>;

type TokenToRcId = AHashMap<Rc<[u8]>, usize>; // Internal map

/// Merges pretokens to form a vocabulary.
///
/// # Arguments:
/// * `raw_pretokens`: pretokens with their count. special tokens have been filtered out
/// * `initial_vocab_bytes`: int_id to pretoken mapping. At this point, it's the 256 bytes and special tokens
/// * `max_vocab_size`: the returned vocabulary length will be at most max_vocab_size
#[pyfunction]
pub fn merge(
    raw_pretokens: HashMap<Vec<u8>, isize>,
    initial_vocab_bytes: VocabBytes,
    max_vocab_size: usize,
) -> PyResult<(VocabBytes, Vec<(Vec<u8>, Vec<u8>)>)> {
    let mut merges_as_bytes: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();

    // --- 1. Token Interning ---
    let max_id = match initial_vocab_bytes.keys().max() {
        Some(&id) => id,
        None => {
            // Handle empty initial vocab
            return Ok((HashMap::new(), Vec::new()));
        }
    };

    let dummy_rc: Rc<[u8]> = Rc::from(&b""[..]);
    let mut id_to_token: Vec<Rc<[u8]>> = vec![dummy_rc; max_id + 1];
    let mut token_to_id: TokenToRcId = AHashMap::default();
    let mut byte_to_id_map: [usize; 256] = [0; 256];

    for (id, bytes) in &initial_vocab_bytes {
        let bytes_rc: Rc<[u8]> = bytes.clone().into_boxed_slice().into();

        if bytes_rc.len() == 1 {
            byte_to_id_map[bytes_rc[0] as usize] = *id;
        }
        id_to_token[*id] = bytes_rc.clone();
        token_to_id.insert(bytes_rc, *id);
    }
    let mut next_token_id = max_id + 1;

    // --- 2. Build Linked-List Data Structures ---
    let mut arena: Vec<Node> = Vec::with_capacity(raw_pretokens.len());
    let mut pretoken_lists: Vec<PretokenList> = Vec::with_capacity(raw_pretokens.len());
    let mut occurences: Occurrences = AHashMap::default();
    let mut pair_map: PairMap = AHashMap::default();

    for (word_bytes, freq) in raw_pretokens {
        if word_bytes.is_empty() {
            continue;
        }
        let list_idx = pretoken_lists.len();
        let mut prev_node_idx: Option<usize> = None;
        for &byte in word_bytes.iter() {
            let token_id = byte_to_id_map[byte as usize];
            let node_idx = arena.len();

            arena.push(Node {
                token_id: token_id,
                list_idx: list_idx,
                prev: prev_node_idx,
                next: None,
            });
            if let Some(prev_idx) = prev_node_idx {
                arena[prev_idx].next = Some(node_idx);
                let pair = Pair(arena[prev_idx].token_id, token_id);
                *occurences.entry(pair).or_insert(0) += freq;
                pair_map.entry(pair).or_default().insert(prev_idx);
            }
            prev_node_idx = Some(node_idx);
        }

        pretoken_lists.push(PretokenList {
            tail: prev_node_idx,
            freq,
        });
    }

    // --- 3. Build the Initial Priority Queue ---
    let mut rank_queue: RankQueue = occurences
        .iter()
        .map(|(&pair, &freq)| Rank {
            freq,
            pair_bytes: (id_to_token[pair.0].clone(), id_to_token[pair.1].clone()),
            pair_id: pair,
        })
        .collect();

    // --- 4. Merge ---
    while token_to_id.len() < max_vocab_size {
        // 4.1. Get Best Pair
        let best_rank = match rank_queue.pop_last() {
            Some(rank) => rank,
            None => break,
        };
        let best_pair = best_rank.pair_id; // Pair(A, B)

        // 4.2. Create New Token
        let new_token_bytes = [
            id_to_token[best_pair.0].as_ref(),
            id_to_token[best_pair.1].as_ref(),
        ]
        .concat();
        let new_token_rc: Rc<[u8]> = new_token_bytes.into_boxed_slice().into();

        let new_token_id = next_token_id;
        next_token_id += 1;
        id_to_token.push(new_token_rc.clone());
        token_to_id.insert(new_token_rc, new_token_id);
        merges_as_bytes.push((
            id_to_token[best_pair.0].to_vec(),
            id_to_token[best_pair.1].to_vec(),
        ));

        // 4.3. Get All Occurrences
        let nodes_to_merge = match pair_map.remove(&best_pair) {
            Some(nodes) => nodes,
            None => continue,
        };
        occurences.remove(&best_pair);

        let mut freq_delta: AHashMap<Pair, isize> = AHashMap::default();

        // 4.4. Perform O(1) Merges
        let mut sorted_nodes: Vec<_> = nodes_to_merge.into_iter().collect();
        sorted_nodes.sort_unstable(); // Sort by arena index for determinism

        for a_node_idx in sorted_nodes {
            // --- a. Stale Data Check ---
            let (b_node_idx, pre_a_node_idx, c_node_idx, list_idx, freq) = {
                let a_node = &arena[a_node_idx];
                if a_node.token_id != best_pair.0 {
                    continue;
                }
                let b_node_idx = match a_node.next {
                    Some(idx) => idx,
                    None => continue,
                };
                let b_node = &arena[b_node_idx];
                if b_node.token_id != best_pair.1 {
                    continue;
                }
                // Crucial check: Ensure the back-link is correct. If another merge
                // involving `a_node` or `b_node` happened, this link would be broken.
                if b_node.prev != Some(a_node_idx) {
                    continue; // Stale pair, `b_node` is no longer preceded by `a_node`.
                }

                (
                    b_node_idx,
                    a_node.prev,
                    b_node.next,
                    a_node.list_idx,
                    pretoken_lists[a_node.list_idx].freq,
                )
            };

            // --- b. Decrement token pair before merge ---
            if let Some(pre_a_idx) = pre_a_node_idx {
                let pair = Pair(arena[pre_a_idx].token_id, best_pair.0);
                *freq_delta.entry(pair).or_insert(0) -= freq;
                if let Some(set) = pair_map.get_mut(&pair) {
                    set.remove(&pre_a_idx);
                }
            }
            if let Some(c_idx) = c_node_idx {
                let pair = Pair(best_pair.1, arena[c_idx].token_id);
                *freq_delta.entry(pair).or_insert(0) -= freq;
                if let Some(set) = pair_map.get_mut(&pair) {
                    set.remove(&b_node_idx);
                }
            }

            // --- c. Perform O(1) merge ---
            arena[a_node_idx].token_id = new_token_id;
            arena[a_node_idx].next = c_node_idx;
            if let Some(c_idx) = c_node_idx {
                arena[c_idx].prev = Some(a_node_idx);
            } else {
                pretoken_lists[list_idx].tail = Some(a_node_idx);
            }

            // --- d. Increment token pair after merge ---
            if let Some(pre_a_idx) = pre_a_node_idx {
                let pair = Pair(arena[pre_a_idx].token_id, new_token_id);
                *freq_delta.entry(pair).or_insert(0) += freq;
                pair_map.entry(pair).or_default().insert(pre_a_idx);
            }
            if let Some(c_idx) = c_node_idx {
                let pair = Pair(new_token_id, arena[c_idx].token_id);
                *freq_delta.entry(pair).or_insert(0) += freq;
                pair_map.entry(pair).or_default().insert(a_node_idx);
            }
        }

        // --- 4.5. Apply Batched Updates (same as before) ---
        for (pair, delta) in freq_delta {
            if delta == 0 {
                continue;
            }
            let old_freq = *occurences.entry(pair).or_insert(0);
            let new_freq = old_freq + delta;
            if old_freq > 0 {
                rank_queue.remove(&Rank {
                    freq: old_freq,
                    pair_bytes: (id_to_token[pair.0].clone(), id_to_token[pair.1].clone()),
                    pair_id: pair,
                });
            }
            *occurences.get_mut(&pair).unwrap() = new_freq;
            if new_freq > 0 {
                rank_queue.insert(Rank {
                    freq: new_freq,
                    pair_bytes: (id_to_token[pair.0].clone(), id_to_token[pair.1].clone()),
                    pair_id: pair,
                });
            }
        }
    }

    let final_vocab: HashMap<usize, Vec<u8>> = id_to_token
        .into_iter()
        .enumerate()
        .map(|(id, rc)| (id, rc.to_vec()))
        .collect();
    Ok((final_vocab, merges_as_bytes))
}

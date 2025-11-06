use pyo3::prelude::*;
use pyo3::{FromPyObject, IntoPyObject};
use std::collections::HashMap;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd, IntoPyObject, FromPyObject)]
pub struct Pair(usize, usize);

type Pretoken = Vec<usize>;
type Pretokens = HashMap<Pretoken, isize>;
type VocabBytes = HashMap<usize, Vec<u8>>;

#[pyfunction]
pub fn merge(
    raw_pretokens: HashMap<Vec<u8>, isize>,
    initial_vocab_bytes: VocabBytes,
    max_vocab_size: usize,
) -> PyResult<(VocabBytes, Vec<(Vec<u8>, Vec<u8>)>)> {
    let mut merges_as_bytes: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();

    // --- Token Interning Setup ---
    let mut token_to_id: HashMap<Vec<u8>, usize> = HashMap::new();
    let mut id_to_token: Vec<Vec<u8>> = Vec::new();

    for (id, bytes) in &initial_vocab_bytes {
        if *id >= id_to_token.len() {
            id_to_token.resize(id + 1, Vec::new());
        }
        id_to_token[*id] = bytes.clone();
        token_to_id.insert(bytes.clone(), *id);
    }
    let mut next_token_id = id_to_token.len();

    // Transform the input from HashMap<Vec<u8>, isize> to the format needed for merging.
    // e.g., {b"hello": 5} -> {[b"h", b"e", b"l", b"l", b"o"]: 5}
    let mut pretokens: Pretokens = HashMap::with_capacity(raw_pretokens.len());
    for (word_bytes, freq) in raw_pretokens {
        let sequence: Vec<usize> = word_bytes
            .into_iter()
            .map(|b| *token_to_id.get(&vec![b]).unwrap())
            .collect();
        pretokens.insert(sequence, freq);
    }

    // Let's build our accurate frequency cache
    let mut occurences: HashMap<Pair, isize> = HashMap::new();
    for (pretoken_vec, count) in &pretokens {
        for window in pretoken_vec.windows(2) {
            let pair = Pair(window[0], window[1]);
            *occurences.entry(pair).or_insert(0) += *count;
        }
    }

    while token_to_id.len() < max_vocab_size {
        if pretokens.is_empty() {
            break;
        }

        // To ensure correctness, we find the best pair by iterating through all
        // current occurrences. This perfectly mimics the reference implementation's
        // tie-breaking logic, which is to choose the lexicographically largest pair
        // in case of a frequency tie.
        let best_pair_opt = occurences
            .iter()
            .filter(|(_, count)| **count > 0)
            .max_by(|a, b| {
                a.1.cmp(&b.1).then_with(|| {
                    let pair_a_bytes = (&id_to_token[a.0.0], &id_to_token[a.0.1]);
                    let pair_b_bytes = (&id_to_token[b.0.0], &id_to_token[b.0.1]);
                    pair_a_bytes.cmp(&pair_b_bytes)
                })
            })
            .map(|(pair, _)| *pair);

        let mut freq_delta: HashMap<Pair, isize> = HashMap::new();
        if let Some(best_pair) = best_pair_opt {
            let new_token_bytes = [
                id_to_token[best_pair.0].as_slice(),
                id_to_token[best_pair.1].as_slice(),
            ]
            .concat();
            let new_token_id = next_token_id;
            next_token_id += 1;

            id_to_token.push(new_token_bytes.clone());
            token_to_id.insert(new_token_bytes, new_token_id);

            merges_as_bytes.push((
                id_to_token[best_pair.0].clone(),
                id_to_token[best_pair.1].clone(),
            ));

            let mut new_pretokens = HashMap::with_capacity(pretokens.len());
            for (old_pretoken, count) in pretokens.into_iter() {
                let mut new_pretoken_list: Vec<usize> = Vec::with_capacity(old_pretoken.len());
                let mut i = 0;
                let mut changed = false;
                while i < old_pretoken.len() {
                    if i + 1 < old_pretoken.len()
                        && old_pretoken[i] == best_pair.0
                        && old_pretoken[i + 1] == best_pair.1
                    {
                        new_pretoken_list.push(new_token_id);
                        i += 2;
                        changed = true;
                    } else {
                        new_pretoken_list.push(old_pretoken[i]);
                        i += 1;
                    }
                }

                if changed {
                    // Update frequency deltas only if the sequence was modified
                    for window in old_pretoken.windows(2) {
                        let p = Pair(window[0], window[1]);
                        *freq_delta.entry(p).or_insert(0) -= count;
                    }
                    for window in new_pretoken_list.windows(2) {
                        let p = Pair(window[0], window[1]);
                        *freq_delta.entry(p).or_insert(0) += count;
                    }
                }

                *new_pretokens.entry(new_pretoken_list).or_insert(0) += count;
            }

            pretokens = new_pretokens;
        } else {
            // This can happen if the priority queue is exhausted.
            break;
        }

        for (pair, delta) in freq_delta {
            if delta == 0 {
                continue;
            }
            let count_ref = occurences.entry(pair).or_insert(0);
            *count_ref += delta;

            if *count_ref <= 0 {
                // If count is zero or less, remove it to prevent stale entries
                occurences.remove(&pair);
            }
        }
    }

    let final_vocab = id_to_token.into_iter().enumerate().collect();
    Ok((final_vocab, merges_as_bytes))
}

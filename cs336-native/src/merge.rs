use pyo3::prelude::*;
use pyo3::{FromPyObject, IntoPyObject};
use std::collections::{BTreeMap, BinaryHeap, HashMap};
use std::vec;

#[derive(Debug, Clone, Eq, PartialEq, Hash, Ord, PartialOrd, IntoPyObject, FromPyObject)]
pub struct Pair(Vec<u8>, Vec<u8>);

type Pretoken = Vec<Vec<u8>>;
type Pretokens = BTreeMap<Pretoken, isize>;
type Vocab = HashMap<usize, Vec<u8>>;

#[derive(Eq, PartialEq)]
struct HeapItem {
    priority: isize,
    pair: Pair,
}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Compare by priority first (max-heap).
        // If priorities are equal, break ties by comparing the pairs lexicographically.
        // The pair that is lexicographically larger has higher priority.
        self.priority
            .cmp(&other.priority)
            .then_with(|| self.pair.cmp(&other.pair))
    }
}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[pyfunction]
pub fn merge(
    raw_pretokens: HashMap<Vec<u8>, isize>,
    initial_vocab: Vocab,
    max_vocab_size: usize,
) -> PyResult<(Vocab, Vec<Pair>)> {
    let mut merges: Vec<Pair> = Vec::new();
    let mut vocab = initial_vocab;

    // Transform the input from HashMap<Vec<u8>, isize> to the format needed for merging.
    // e.g., {b"hello": 5} -> {[b"h", b"e", b"l", b"l", b"o"]: 5}
    let mut pretokens: Pretokens = BTreeMap::new();
    for (word_bytes, freq) in raw_pretokens {
        let sequence: Vec<Vec<u8>> = word_bytes.into_iter().map(|b| vec![b]).collect();
        pretokens.insert(sequence, freq);
    }

    // Let's build our accurate frequency cache
    let mut occurences: HashMap<Pair, isize> = HashMap::new();
    for (pretoken_vec, count) in &pretokens {
        for pair_tokens in pretoken_vec.windows(2) {
            let pair = Pair(pair_tokens[0].clone(), pair_tokens[1].clone());
            *occurences.entry(pair).or_insert(0) += *count;
        }
    }
    // Let's build our (approximate) priority queue
    let mut priority_queue: BinaryHeap<HeapItem> = occurences
        .iter()
        .map(|(pair_ref, count)| HeapItem {
            priority: *count,
            pair: Pair(pair_ref.0.to_vec(), pair_ref.1.to_vec()),
        })
        .collect();

    while vocab.len() < max_vocab_size {
        if pretokens.is_empty() {
            break;
        }

        let mut found_match: Option<HeapItem> = None;
        while let Some(priority_pair) = priority_queue.pop() {
            // If the hit isn't stale, we found our match. Otherwise, try again
            if let Some(true_priority) = occurences.get(&priority_pair.pair) {
                if *true_priority == priority_pair.priority {
                    found_match = Some(priority_pair);
                    break;
                }
            }
        }

        let mut freq_delta: HashMap<Pair, isize> = HashMap::new();
        if let Some(pair) = found_match {
            let new_token = [pair.pair.0.as_slice(), pair.pair.1.as_slice()].concat();

            merges.push(pair.pair.clone());
            vocab.insert(vocab.len(), new_token.clone());

            let mut new_pretokens = BTreeMap::new();
            for (old_pretoken, count) in pretokens.into_iter() {
                let old_length = old_pretoken.len();

                // Check if the pair exists in the pretoken without a full scan.
                // This is a fast path for sequences that won't be changed.
                let might_contain_pair = old_pretoken
                    .windows(2)
                    .any(|w| w[0] == pair.pair.0 && w[1] == pair.pair.1);

                if !might_contain_pair {
                    new_pretokens.insert(old_pretoken, count);
                    continue;
                }

                for pair_tokens in old_pretoken.windows(2) {
                    let p = Pair(pair_tokens[0].clone(), pair_tokens[1].clone());
                    *freq_delta.entry(p).or_insert(0) -= count;
                }

                let mut new_pretoken_list: Vec<Vec<u8>> = Vec::with_capacity(old_length);
                let mut i = 0;
                while i < old_length {
                    if i + 1 < old_length
                        && old_pretoken[i] == pair.pair.0
                        && old_pretoken[i + 1] == pair.pair.1
                    {
                        new_pretoken_list.push(new_token.clone());
                        i += 2; // Skip both merged tokens
                    } else {
                        new_pretoken_list.push(old_pretoken[i].clone());
                        i += 1;
                    }
                }

                for pair_tokens in new_pretoken_list.windows(2) {
                    let p = Pair(pair_tokens[0].clone(), pair_tokens[1].clone());
                    *freq_delta.entry(p).or_insert(0) += count;
                }

                *new_pretokens.entry(new_pretoken_list).or_insert(0) += count;
            }
            pretokens = new_pretokens;
        } else {
            println!("Warning: Unexpected: best_pair was None. Stopping train.");
            break;
        }

        for (pair, delta) in freq_delta {
            let priority = {
                let count_ref = occurences.entry(pair.clone()).or_insert(0);
                *count_ref += delta;
                *count_ref
            };
            priority_queue.push(HeapItem { priority, pair });
        }
    }

    Ok((vocab, merges))
}

use pyo3::prelude::*;
use pyo3::{FromPyObject, IntoPyObject};
use std::collections::{BTreeMap, BinaryHeap, HashMap};
use std::vec;

#[derive(Debug, Clone, Eq, PartialEq, Hash, Ord, PartialOrd, IntoPyObject, FromPyObject)]
struct Pair(Vec<u8>, Vec<u8>);

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
fn merge(
    mut pretokens: Pretokens,
    initial_vocab: Vocab,
    max_vocab_size: usize,
) -> PyResult<(Vocab, Vec<Pair>)> {
    let mut merges: Vec<Pair> = Vec::new();
    let mut vocab = initial_vocab;

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

        let mut freq_delta: BTreeMap<Pretoken, isize> = BTreeMap::new();
        if let Some(pair) = found_match {
            let new_token = [pair.pair.0.as_slice(), pair.pair.1.as_slice()].concat();

            merges.push(pair.pair.clone());
            vocab.insert(vocab.len(), new_token.clone());

            let mut new_pretokens = BTreeMap::new();
            for (old_pretoken, count) in pretokens.into_iter() {
                let mut new_pretoken_list: Vec<Vec<u8>> = Vec::new();
                let old_length = old_pretoken.len();
                let mut has_delta = false;

                let mut i = 0;
                while i < old_length {
                    if i < old_length - 1
                        && old_pretoken[i] == pair.pair.0
                        && old_pretoken[i + 1] == pair.pair.1
                    {
                        has_delta = true;
                        new_pretoken_list.push(new_token.clone());
                        i += 2; // Skip both tokens
                    } else {
                        new_pretoken_list.push(old_pretoken[i].clone());
                        i += 1;
                    }
                }

                if has_delta {
                    for pair_tokens in old_pretoken.windows(2) {
                        *freq_delta
                            .entry(vec![pair_tokens[0].clone(), pair_tokens[1].clone()])
                            .or_insert(0) -= count;
                    }

                    for pair_tokens in new_pretoken_list.windows(2) {
                        *freq_delta
                            .entry(vec![pair_tokens[0].clone(), pair_tokens[1].clone()])
                            .or_insert(0) += count;
                    }

                    *new_pretokens.entry(new_pretoken_list).or_insert(0) += count;
                } else {
                    new_pretokens.insert(old_pretoken, count);
                }
            }
            pretokens = new_pretokens;
        } else {
            println!("Warning: Unexpected: best_pair was None. Stopping train.");
            break;
        }

        // Apply the frequency deltas:
        for (pretoken_vec, count) in &freq_delta {
            let pair = Pair(pretoken_vec[0].clone(), pretoken_vec[1].clone());

            let priority = {
                let count_ref = occurences.entry(pair.clone()).or_insert(0);
                *count_ref += *count;
                *count_ref
            };
            priority_queue.push(HeapItem {
                priority: priority,
                pair: pair,
            });
        }
    }

    Ok((vocab, merges))
}

/// Registers the `merge` function with the `cs336_native` module.
#[pymodule]
fn cs336_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(merge, m)?)?;
    Ok(())
}

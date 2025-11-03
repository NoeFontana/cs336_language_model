use pyo3::prelude::*;
use pyo3::{FromPyObject, IntoPyObject};
// BTreeMap is for your 'pretokens' (key is Vec<Vec<u8>>, which is not Hash)
// HashMap is for 'vocab' and 'occurences' (keys are hashable)
use std::collections::{BTreeMap, HashMap};

// --- Helper Structs & Types ---

// A Pair of byte tokens.
#[derive(Clone, Eq, PartialEq, Hash, Ord, PartialOrd, IntoPyObject, FromPyObject)]
struct Pair(Vec<u8>, Vec<u8>);

// A Pair of byte token slices. Used for efficient counting.
#[derive(Clone, Copy, Eq, PartialEq, Hash, Ord, PartialOrd)]
struct PairRef<'a>(&'a [u8], &'a [u8]);

// A "pretoken" is a sequence of tokens (bytes)
type Pretoken = Vec<Vec<u8>>;
// BTreeMap is the Rust equivalent for a Python dict
// where the key is a non-hashable tuple (like tuple[bytes, ...])
type Pretokens = BTreeMap<Pretoken, isize>;
// Vocab is a standard hash map
type Vocab = HashMap<usize, Vec<u8>>;

// --- Private Rust Helper Function ---
// (Not exposed to Python)

/// This is a direct translation of your `single_merge` helper function.
/// It finds the most frequent pair in the current `pretokens`.
fn single_merge_rust(pretokens: &Pretokens) -> Option<Pair> {
    // `occurences: dict[tuple[bytes, bytes], int] = {}`. We use references for keys to avoid clones.
    let mut occurences: HashMap<PairRef, isize> = HashMap::new();

    // `for pretoken, count in pretokens.items():`
    for (pretoken_vec, count) in pretokens {
        // `for pair in zip(pretoken, pretoken[1:], strict=False):`
        for pair_tokens in pretoken_vec.windows(2) {
            // Create a pair of references, avoiding cloning the vectors.
            let pair_ref = PairRef(&pair_tokens[0], &pair_tokens[1]);
            // `occurences[pair] = occurences.get(pair, 0) + count`
            *occurences.entry(pair_ref).or_insert(0) += *count;
        }
    }

    // --- Find best pair ---
    // Implements your tie-breaking logic (max count, then max pair)
    occurences
        .into_iter()
        .max_by(|(pair_a, count_a), (pair_b, count_b)| {
            count_a.cmp(count_b).then_with(|| pair_a.cmp(pair_b))
        })
        // Convert the best PairRef back to an owned Pair for the return value.
        .map(|(pair_ref, _count)| Pair(pair_ref.0.to_vec(), pair_ref.1.to_vec()))
}

// --- Public Python Function ---

/// This is the Rust translation of your main `merge` function.
#[pyfunction]
fn merge(
    // pyo3 handles the copy from Python dict -> Rust BTreeMap
    mut pretokens: Pretokens,
    initial_vocab: Vocab,
    max_vocab_size: usize,
) -> PyResult<(Vocab, Vec<Pair>)> {
    // `merges: list[tuple[bytes, bytes]] = []`
    let mut merges: Vec<Pair> = Vec::new();

    // `vocab = initial_vocab.copy()`
    // We get ownership of initial_vocab, so we can just use it
    let mut vocab = initial_vocab;

    // `while len(vocab) < max_vocab_size:`
    while vocab.len() < max_vocab_size {
        // `if len(pretokens) == 0:`
        if pretokens.is_empty() {
            break;
        }

        // `best_pair = single_merge(pretokens)`
        // We pass a reference to our private helper
        let best_pair = single_merge_rust(&pretokens);

        // `if best_pair:`
        if let Some(pair) = best_pair {
            // `merges.append(best_pair)`
            merges.push(pair.clone());

            // `best_flat = b"".join(best_pair)`
            let best_flat = [pair.0.as_slice(), pair.1.as_slice()].concat();

            // `vocab[len(vocab)] = best_flat`
            vocab.insert(vocab.len(), best_flat.clone());

            // --- Rebuild pretokens ---
            // `new_pretokens: dict[tuple[bytes, ...], int] = {}`
            let mut new_pretokens: Pretokens = BTreeMap::new();

            // `for pretoken, count in pretokens.items():`
            for (pretoken_vec, count) in &pretokens {
                let mut new_pretoken_list: Vec<Vec<u8>> = Vec::new();
                let mut i = 0;
                let pretoken_length = pretoken_vec.len();

                // `while i < pretoken_length:`
                while i < pretoken_length {
                    // `if i < pretoken_length - 1 and (pretoken[i], pretoken[i + 1]) == best_pair:`
                    if i < pretoken_length - 1
                        && pretoken_vec[i] == pair.0
                        && pretoken_vec[i + 1] == pair.1
                    {
                        new_pretoken_list.push(best_flat.clone());
                        i += 2;
                    } else {
                        new_pretoken_list.push(pretoken_vec[i].clone());
                        i += 1;
                    }
                }
                *new_pretokens.entry(new_pretoken_list).or_insert(0) += *count;
            }
            // `pretokens = new_pretokens`
            pretokens = new_pretokens;
        } else {
            // `logging.getLogger...`
            eprintln!("Warning: Unexpected: best_pair was None. Stopping train.");
            break;
        }
    }

    // `return vocab, merges`
    // pyo3 handles copying the Rust HashMap/Vec -> Python dict/list
    Ok((vocab, merges))
}

/// Registers the `merge` function with the `cs336_native` module.
#[pymodule]
fn cs336_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(merge, m)?)?;
    Ok(())
}

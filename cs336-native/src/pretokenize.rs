use ahash::AHashMap as HashMap;
use memmap2::Mmap;
use pcre2::bytes::{Regex, RegexBuilder};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use rayon::prelude::*;
use std::fs::File;

fn pcre2_escape(s: &str) -> String {
    let mut escaped = String::new();
    for c in s.chars() {
        match c {
            '.' | '^' | '$' | '*' | '+' | '?' | '(' | ')' | '[' | '{' | '\\' | '|' => {
                escaped.push('\\');
                escaped.push(c);
            }
            _ => escaped.push(c),
        }
    }
    escaped
}

fn compile_special_regex(special_tokens: &[String]) -> Result<Option<Regex>, pcre2::Error> {
    if special_tokens.is_empty() {
        return Ok(None);
    }
    let mut sorted_specials = special_tokens.to_vec();
    sorted_specials.sort_by_key(|s| s.len());
    sorted_specials.reverse();
    let pattern = sorted_specials
        .iter()
        .map(|s| pcre2_escape(s))
        .collect::<Vec<_>>()
        .join("|");

    RegexBuilder::new()
        .jit_if_available(true)
        .build(&pattern)
        .map(Some)
}

fn compile_main_regex() -> Result<Regex, pcre2::Error> {
    let pattern = r"'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^a-zA-Z0-9\s]+|\s+(?!\S)|\s+";
    RegexBuilder::new().jit_if_available(true).build(pattern)
}

fn process_chunk<'a>(
    main_re: &Regex,
    chunk: &'a [u8],
    counts: &mut HashMap<&'a [u8], u64>,
) -> Result<(), pcre2::Error> {
    for result in main_re.find_iter(chunk) {
        let mat = result?;
        let token = &chunk[mat.start()..mat.end()];
        *counts.entry(token).or_insert(0) += 1;
    }
    Ok(())
}

fn process_segment_with_specials<'a>(
    segment: &'a [u8],
    special_re: Option<&Regex>,
    main_re: &Regex,
    counts: &mut HashMap<&'a [u8], u64>,
) -> Result<(), pcre2::Error> {
    if let Some(sre) = special_re {
        let mut last_end = 0;
        for result in sre.find_iter(segment) {
            let mat = result?;
            // Process content before special token
            if mat.start() > last_end {
                process_chunk(main_re, &segment[last_end..mat.start()], counts)?;
            }
            last_end = mat.end();
        }
        // Process remaining content
        if last_end < segment.len() {
            process_chunk(main_re, &segment[last_end..], counts)?;
        }
    } else {
        process_chunk(main_re, segment, counts)?;
    }
    Ok(())
}

pub fn pretokenize_bytes_impl<'a>(
    data: &'a [u8],
    special_tokens: Vec<String>,
) -> Result<HashMap<&'a [u8], u64>, String> {
    let special_re =
        compile_special_regex(&special_tokens).map_err(|e| format!("Regex error: {}", e))?;
    let main_re = compile_main_regex().map_err(|e| format!("Regex error: {}", e))?;

    let num_threads = rayon::current_num_threads();
    const OVERPROVISION_FACTOR: usize = 8;
    let num_chunks = num_threads * OVERPROVISION_FACTOR;
    let chunk_size = std::cmp::max(data.len() / num_chunks, 1024 * 1024); 
    let estimated_tokens_per_chunk = 32 * 1024; // Heuristic

    let delimiter = "<|endoftext|>";
    let finder = memchr::memmem::Finder::new(delimiter.as_bytes());

    let mut boundaries = Vec::new();
    let mut current = 0;
    while current < data.len() {
        let target_end = std::cmp::min(current + chunk_size, data.len());
        let mut end = target_end;
        if end < data.len() {
            // Try to find the document delimiter first
            if let Some(pos) = finder.find(&data[end..]) {
                end += pos + delimiter.len();
            } else {
                end = data.len();
            }
        }
        boundaries.push((current, end));
        current = end;
    }

    let final_counts = boundaries
        .par_iter()
        .fold(
            || Ok(HashMap::with_capacity(estimated_tokens_per_chunk)),
            |acc: Result<HashMap<&'a [u8], u64>, String>, &(start, end)| {
                let mut local_map = acc?;
                let chunk = &data[start..end];
                process_segment_with_specials(chunk, special_re.as_ref(), &main_re, &mut local_map)
                    .map_err(|e| format!("Match error: {}", e))?;

                Ok(local_map)
            },
        )
        .reduce(
            || Ok(HashMap::new()),
            |m1, m2| {
                let mut m1 = m1?;
                let mut m2 = m2?;
                if m2.len() > m1.len() {
                    for (k, v) in m1 {
                        *m2.entry(k).or_insert(0) += v;
                    }
                    Ok(m2)
                } else {
                    for (k, v) in m2 {
                        *m1.entry(k).or_insert(0) += v;
                    }
                    Ok(m1)
                }
            },
        )?;

    Ok(final_counts)
}

#[pyfunction]
pub fn pretokenize_bytes(
    py: Python,
    data: &[u8],
    special_tokens: Vec<String>,
) -> PyResult<Py<PyAny>> {
    let final_counts = pretokenize_bytes_impl(data, special_tokens)
        .map_err(PyErr::new::<PyValueError, _>)?;

    let dict = PyDict::new(py);
    for (k, v) in final_counts {
        let key = PyBytes::new(py, k);
        dict.set_item(key, v)?;
    }
    Ok(dict.into())
}

#[pyfunction]
pub fn pretokenize_file(
    py: Python,
    path: String,
    special_tokens: Vec<String>,
) -> PyResult<Py<PyAny>> {
    let file = File::open(&path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    pretokenize_bytes(py, &mmap, special_tokens)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pretokenize_basic() {
        let data = b"Hello world!";
        let counts = pretokenize_bytes_impl(data, vec![]).unwrap();

        assert_eq!(counts.get(&b"Hello"[..]), Some(&1));
        assert_eq!(counts.get(&b" world"[..]), Some(&1));
        assert_eq!(counts.get(&b"!"[..]), Some(&1));
    }

    #[test]
    fn test_pretokenize_pcre2_spaces() {
        let data = b"  multiple   spaces  ";
        let counts = pretokenize_bytes_impl(data, vec![]).unwrap();

        assert_eq!(counts.get(&b" "[..]), Some(&1));
        assert_eq!(counts.get(&b" multiple"[..]), Some(&1));
        assert_eq!(counts.get(&b"  "[..]), Some(&2));
        assert_eq!(counts.get(&b" spaces"[..]), Some(&1));
    }

    #[test]
    fn test_pretokenize_pcre2_null_bytes() {
        let data = b"\x00\x00";
        let counts = pretokenize_bytes_impl(data, vec![]).unwrap();
        assert_eq!(counts.get(&b"\x00\x00"[..]), Some(&1));
    }

    #[test]
    fn test_pretokenize_pcre2_specials() {
        let data = b" <|endoftext|>";
        let specs = vec!["<|endoftext|>".to_string()];
        let counts = pretokenize_bytes_impl(data, specs).unwrap();
        // Should be " " only (special token ignored)
        assert_eq!(counts.get(&b" "[..]), Some(&1));
        assert_eq!(counts.get(&b"<|endoftext|>"[..]), None);
    }

    #[test]
    fn test_pretokenize_mixed_specials_and_text() {
        let data = b"Hello <|special|> world!";
        let specs = vec!["<|special|>".to_string()];
        let counts = pretokenize_bytes_impl(data, specs).unwrap();

        assert_eq!(counts.get(&b"Hello"[..]), Some(&1));
        assert_eq!(counts.get(&b" "[..]), Some(&1)); // space before <|special|>
        assert_eq!(counts.get(&b"<|special|>"[..]), None); // ignored
        assert_eq!(counts.get(&b" world"[..]), Some(&1)); // contains space after
        assert_eq!(counts.get(&b"!"[..]), Some(&1));
    }

    #[test]
    fn test_pretokenize_empty() {
        let data = b"";
        let counts = pretokenize_bytes_impl(data, vec![]).unwrap();
        assert!(counts.is_empty());
    }

    #[test]
    fn test_pretokenize_only_specials() {
        let data = b"<|a|><|b|>";
        let specs = vec!["<|a|>".to_string(), "<|b|>".to_string()];
        let counts = pretokenize_bytes_impl(data, specs).unwrap();
        assert!(counts.is_empty());
    }
}

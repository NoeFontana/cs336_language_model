import collections

import cs336_native
import pytest
import regex as re
from hypothesis import given
from hypothesis import strategies as st

from cs336_basics.pretokenization import _batch_count_from_segment

# The pretokenization pattern from pretokenization.py
PRETOKEN_PATTERN_STR = rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
COMPILED_PATTERN = re.compile(PRETOKEN_PATTERN_STR, flags=re.V1)


def python_pretokenize(data: bytes, special_tokens: list[str]) -> collections.Counter:
    # Simplified version of the logic in pretokenization.py for equivalence testing
    if special_tokens:
        special_tokens_pattern = re.compile(
            b"|".join(re.escape(tok.encode("utf-8")) for tok in sorted(special_tokens, key=len, reverse=True)),
            flags=re.V1,
        )
    else:
        special_tokens_pattern = None

    pretokens = collections.Counter()
    last_end = 0
    if special_tokens_pattern is not None:
        for special_match in re.finditer(special_tokens_pattern, data, concurrent=False):
            corpus_segment = data[last_end : special_match.start()]
            pretokens.update(_batch_count_from_segment(memoryview(corpus_segment), COMPILED_PATTERN))
            last_end = special_match.end()

    corpus_segment = data[last_end:]
    if corpus_segment:
        pretokens.update(_batch_count_from_segment(memoryview(corpus_segment), COMPILED_PATTERN))

    return pretokens


@given(data=st.binary(min_size=0, max_size=1000), specials=st.lists(st.text(min_size=1, max_size=10), max_size=5))
def test_pretokenize_equivalence(data, specials):
    # Get Python counts
    py_counts = python_pretokenize(data, specials)

    # Get Rust counts
    # Rust implementation expects a list of strings
    rust_counts_dict = cs336_native.pretokenize(data, specials)
    rust_counts = collections.Counter(rust_counts_dict)

    # Compare
    assert py_counts == rust_counts, (
        f"Mismatch for data={data!r}, specials={specials!r}\nPy: {py_counts}\nRust: {rust_counts}"
    )


@pytest.mark.parametrize(
    "sample",
    [
        b"hello world",
        b"  multiple   spaces  ",
        b"don't you'll i'm",
        b"Numbers 123 and symbols !@#",
        b"Unicode: \xf0\x9f\x98\x8a \xe4\xbd\xa0\xe5\xa5\xbd",
        b"Trailing whitespace  ",
        b"Leading whitespace",
        b"Special token <|endoftext|> mixed",
    ],
)
def test_pretokenize_manual_samples(sample):
    specials = ["<|endoftext|>"]
    py_counts = python_pretokenize(sample, specials)
    rust_counts = collections.Counter(cs336_native.pretokenize(sample, specials))
    assert py_counts == rust_counts

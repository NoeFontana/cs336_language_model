from __future__ import annotations

import json
from pathlib import Path

import pytest

from cs336.tokenizer import Tokenizer


def test_init():
    vocab = {0: b"a", 1: b"b"}
    merges = [(b"a", b"b")]
    special_tokens = ["<|endoftext|>"]
    tokenizer = Tokenizer(vocab, merges, special_tokens)
    assert tokenizer.vocab == vocab
    assert tokenizer.merges == merges
    assert tokenizer.special_tokens == special_tokens


def test_init_no_special_tokens():
    vocab = {0: b"a", 1: b"b"}
    merges = [(b"a", b"b")]
    tokenizer = Tokenizer(vocab, merges)
    assert tokenizer.vocab == vocab
    assert tokenizer.merges == merges
    assert tokenizer.special_tokens is None


def test_from_files(tmp_path: Path):
    vocab = {0: "a", 1: "b"}
    merges = ["a b"]
    special_tokens = ["<|endoftext|>"]

    vocab_file = tmp_path / "vocab.json"
    with open(vocab_file, "w") as f:
        json.dump(vocab, f)

    merges_file = tmp_path / "merges.txt"
    with open(merges_file, "w") as f:
        f.write("\n".join(merges))

    tokenizer = Tokenizer.from_files(str(vocab_file), str(merges_file), special_tokens)

    expected_vocab = {0: b"a", 1: b"b", 2: b"<|endoftext|>"}
    expected_merges = [(b"a", b"b")]

    assert tokenizer.vocab == expected_vocab
    assert tokenizer.merges == expected_merges
    assert tokenizer.special_tokens == special_tokens


def test_init_with_new_special_token():
    vocab = {0: b"a", 1: b"b"}
    merges = [(b"a", b"b")]
    special_tokens = ["<|endoftext|>"]
    tokenizer = Tokenizer(vocab, merges, special_tokens)
    assert tokenizer.vocab == {0: b"a", 1: b"b", 2: b"<|endoftext|>"}


@pytest.mark.skip(reason="Not yet implemented")
def test_tokenizer_encode():
    pass


@pytest.mark.skip(reason="Not yet implemented")
def test_tokenizer_encode_iterable():
    pass


@pytest.mark.skip(reason="Not yet implemented")
def test_tokenizer_decode():
    pass

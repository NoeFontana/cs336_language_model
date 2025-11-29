from __future__ import annotations

import base64
import json
from pathlib import Path

from cs336_basics.tokenizer import Tokenizer


def test_init():
    """Tests the basic initialization of the Tokenizer."""
    vocab = {0: b"a", 1: b"b"}
    vocab_with_special = {0: b"a", 1: b"b", 2: b"<|endoftext|>"}
    merges = [(b"a", b"b")]
    special_tokens = ["<|endoftext|>"]
    tokenizer = Tokenizer(vocab, merges, special_tokens)
    assert tokenizer.vocab == vocab_with_special
    assert tokenizer.merges == {pair: i for i, pair in enumerate(merges)}
    assert tokenizer.special_tokens == special_tokens


def test_init_no_special_tokens():
    """Tests Tokenizer initialization without any special tokens."""
    vocab = {0: b"a", 1: b"b"}
    merges = [(b"a", b"b")]
    tokenizer = Tokenizer(vocab, merges)
    assert tokenizer.vocab == vocab
    assert tokenizer.merges == {pair: i for i, pair in enumerate(merges)}
    assert tokenizer.special_tokens is None


def test_from_files(tmp_path: Path):
    """Tests loading a tokenizer from separate vocab and merges files."""
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
    assert tokenizer.merges == {pair: i for i, pair in enumerate(expected_merges)}
    assert tokenizer.special_tokens == special_tokens


def test_from_files_single_json(tmp_path: Path):
    """Tests loading a tokenizer from a single combined JSON file."""
    vocab = {0: b"a", 1: b"b", 2: b"<|endoftext|>"}
    merges = [(b"a", b"b")]
    special_tokens = ["<|endoftext|>"]

    serializable_vocab = {
        token_id: base64.b64encode(token_bytes).decode("ascii") for token_id, token_bytes in vocab.items()
    }
    serializable_merges = [
        (base64.b64encode(p1).decode("ascii"), base64.b64encode(p2).decode("ascii")) for p1, p2 in merges
    ]

    data_to_save = {"vocab": serializable_vocab, "merges": serializable_merges}

    tokenizer_file = tmp_path / "tokenizer.json"
    with open(tokenizer_file, "w") as f:
        json.dump(data_to_save, f)

    tokenizer = Tokenizer.from_files(str(tokenizer_file), None, special_tokens)

    assert tokenizer.vocab == vocab
    assert tokenizer.merges == {pair: i for i, pair in enumerate(merges)}
    assert tokenizer.special_tokens == special_tokens


def test_init_with_new_special_token():
    """Tests that a new special token is added to the vocabulary on initialization."""
    vocab = {0: b"a", 1: b"b"}
    merges = [(b"a", b"b")]
    special_tokens = ["<|endoftext|>"]
    tokenizer = Tokenizer(vocab, merges, special_tokens)
    assert tokenizer.vocab == {0: b"a", 1: b"b", 2: b"<|endoftext|>"}


def test_tokenizer_encode_simple():
    """Tests a simple encoding case with a single merge rule."""
    vocab = {i: bytes([i]) for i in range(256)}
    vocab[256] = b"ab"
    merges = [(b"a", b"b")]
    tokenizer = Tokenizer(vocab, merges)

    # 'a' is 97, 'b' is 98, 'c' is 99. 'ab' is 256.
    assert tokenizer.encode("abc") == [256, 99]


def test_tokenizer_encode_with_special_tokens():
    """Tests encoding a string that contains special tokens."""
    vocab = {i: bytes([i]) for i in range(256)} | {256: b"<|endoftext|>"}
    merges = []
    special_tokens = ["<|endoftext|>"]
    tokenizer = Tokenizer(vocab, merges, special_tokens)

    assert tokenizer.encode("a<|endoftext|>b") == [97, 256, 98]


def test_tokenizer_decode():
    """Tests simple decoding of a list of token IDs."""
    vocab = {0: b"Hel", 1: b"lo", 2: b" world", 3: b"!"}
    merges = []
    tokenizer = Tokenizer(vocab, merges)
    decoded_text = tokenizer.decode([0, 1, 2, 3])
    assert decoded_text == "Hello world!"


def test_tokenizer_decode_invalid_utf8():
    """Tests that decoding handles invalid UTF-8 bytes gracefully."""
    vocab = {0: b"a", 1: b"\xff", 2: b"b"}
    merges = []
    tokenizer = Tokenizer(vocab, merges)
    # The invalid byte should be replaced by the Unicode replacement character U+FFFD
    assert tokenizer.decode([0, 1, 2]) == "a\ufffdb"


def test_tokenizer_encode_iterable_simple():
    """Tests the memory-efficient encoding of an iterable of strings."""
    vocab = {i: bytes([i]) for i in range(256)}
    merges = []
    tokenizer = Tokenizer(vocab, merges)
    texts = ["abc", "def"]
    encoded = list(tokenizer.encode_iterable(texts))
    assert encoded == [97, 98, 99, 100, 101, 102]


def test_roundtrip_encode_decode():
    """Tests that encoding and then decoding a string returns the original string."""
    vocab = {i: bytes([i]) for i in range(256)} | {
        256: b"He",
        257: b"llo",
        258: b"<|endoftext|>",
    }
    merges = [(b"H", b"e"), (b"l", b"lo")]
    special_tokens = ["<|endoftext|>"]
    tokenizer = Tokenizer(vocab, merges, special_tokens)

    text = "Hello world! <|endoftext|>"
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    assert decoded == text

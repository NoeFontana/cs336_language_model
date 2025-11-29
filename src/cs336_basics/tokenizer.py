from __future__ import annotations

import base64
import json
from collections.abc import Iterable, Iterator

import regex as re


class Tokenizer:
    """A BPE tokenizer that handles encoding and decoding of text."""

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        Initializes the Tokenizer.

        Args:
            vocab (dict[int, bytes]): A mapping from token IDs to token bytes.
            merges (list[tuple[bytes, bytes]]): A list of BPE merges, ordered by priority.
            special_tokens (list[str] | None): A list of special tokens. These tokens
                are added to the vocabulary if not already present and are not split
                during pre-tokenization.
        """
        self.vocab = vocab

        self.reversed_vocab = {v: k for k, v in vocab.items()}

        self.merges = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens = special_tokens
        if special_tokens:
            for token in special_tokens:
                token_bytes = token.encode("utf-8")
                if token_bytes not in self.vocab.values():
                    new_id = len(self.vocab)
                    self.vocab[new_id] = token_bytes
                    self.reversed_vocab[token_bytes] = new_id

        self.special_tokens_pattern: re.Pattern | None = None
        if special_tokens:
            # We sort by decreasing length to be compatible with overlapping patterns.
            # E.g. if we have specials tokens <a>, <b> and <a><b>, we want the last one to match text<a><b>text
            self.special_tokens_pattern = re.compile(
                b"|".join(re.escape(tok.encode("utf-8")) for tok in sorted(special_tokens, key=len, reverse=True)),
                flags=re.V1,
            )

        pretoken_pattern_str = rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.compiled_pretoken_pattern = re.compile(pretoken_pattern_str, flags=re.V1)

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str | None,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        """
        Loads a tokenizer from vocabulary and merges files.

        Supports two formats:
        1. A single JSON file containing both 'vocab' and 'merges' (base64 encoded).
        2. Two separate files: a JSON for vocabulary and a text file for merges.

        Args:
            vocab_filepath (str): Path to the vocabulary file. If it's a single
                .json file with merges, merges_filepath can be None.
            merges_filepath (str | None): Path to the merges file. Can be None if
                vocab_filepath is a combined JSON file.
            special_tokens (list[str] | None): A list of special tokens.

        Returns:
            Tokenizer: An initialized Tokenizer instance.
        """
        if vocab_filepath.endswith(".json") and merges_filepath is None:
            with open(vocab_filepath, encoding="utf-8") as f:
                data = json.load(f)
            vocab_data = data["vocab"]
            merges_data = data["merges"]
            vocab = {int(k): base64.b64decode(v) for k, v in vocab_data.items()}
            merges = [(base64.b64decode(p1), base64.b64decode(p2)) for p1, p2 in merges_data]
        elif merges_filepath is not None:
            if merges_filepath is None:
                raise ValueError("merges_filepath must be provided if vocab_filepath is not a JSON file.")
            with open(vocab_filepath, encoding="utf-8") as f:
                vocab_data = json.load(f)

            vocab = {int(k): v.encode("utf-8") for k, v in vocab_data.items()}

            merges = []
            with open(merges_filepath, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    p1, p2 = line.split()
                    merges.append((p1.encode("utf-8"), p2.encode("utf-8")))
        else:
            raise ValueError("merges_filepath must be provided if vocab_filepath is not a combined JSON file.")

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Encodes a string into a list of token IDs.

        The encoding process involves:
        1. Splitting the text by special tokens.
        2. Applying pre-tokenization to the regular text chunks.
        3. Applying BPE merges to the pre-tokenized chunks.
        4. Converting all tokens (regular and special) to their corresponding IDs.

        Args:
            text (str): The input string to encode.

        Returns:
            list[int]: A list of token IDs.
        """
        text_bytes = text.encode("utf-8")

        def _encode_chunk(chunk: bytes) -> list[int]:
            """Encodes a regular chunk of text (without special tokens)."""
            chunk_tokens: list[int] = []
            for pretoken_match in self.compiled_pretoken_pattern.finditer(chunk):  # type: ignore[no-matching-overload]
                pretoken_list = [bytes([b]) for b in pretoken_match.group()]
                while len(pretoken_list) >= 2:
                    min_rank = float("inf")
                    best_pair_idx = -1
                    for i in range(len(pretoken_list) - 1):
                        pair = (pretoken_list[i], pretoken_list[i + 1])
                        rank = self.merges.get(pair)
                        if rank is not None and rank < min_rank:
                            min_rank = rank
                            best_pair_idx = i

                    if best_pair_idx == -1:
                        break  # No more merges possible

                    p1, p2 = pretoken_list[best_pair_idx], pretoken_list[best_pair_idx + 1]
                    pretoken_list = pretoken_list[:best_pair_idx] + [p1 + p2] + pretoken_list[best_pair_idx + 2 :]
                chunk_tokens.extend(self.reversed_vocab[token] for token in pretoken_list)
            return chunk_tokens

        if self.special_tokens_pattern is None:
            return _encode_chunk(text_bytes)

        encoded_tokens: list[int] = []
        last_end = 0
        for match in self.special_tokens_pattern.finditer(text_bytes):
            encoded_tokens.extend(_encode_chunk(text_bytes[last_end : match.start()]))
            encoded_tokens.append(self.reversed_vocab[match.group()])
            last_end = match.end()
        encoded_tokens.extend(_encode_chunk(text_bytes[last_end:]))
        return encoded_tokens

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Encodes an iterable of strings, yielding token IDs.

        This method is memory-efficient for large corpora as it processes
        the input text line by line.

        Args:
            iterable (Iterable[str]): An iterable of strings (e.g., a file object).

        Yields:
            int: The next token ID in the encoded sequence.
        """
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """
        Decodes a list of token IDs back into a string.

        Args:
            ids (list[int]): A list of token IDs.

        Returns:
            str: The decoded string. Invalid byte sequences are replaced.
        """
        byte_string = b"".join([self.vocab[_id] for _id in ids])
        return byte_string.decode(errors="replace")

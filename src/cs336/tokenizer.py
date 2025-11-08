from __future__ import annotations

import json


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        if special_tokens:
            for token in special_tokens:
                token_bytes = token.encode("utf-8")
                if token_bytes not in self.vocab.values():
                    new_id = len(self.vocab)
                    self.vocab[new_id] = token_bytes

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
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

        return cls(vocab, merges, special_tokens)

#!/usr/bin/env python
from __future__ import annotations

import argparse
import itertools
import logging
import sys
from pathlib import Path

import numpy as np

from cs336.tokenizer import Tokenizer


def main() -> None:
    """
    Tokenizes a dataset using a pre-trained tokenizer and saves the output.

    This script reads a text file, tokenizes its content line by line, and
    writes the resulting token IDs to a binary file as a NumPy array of
    uint16 values. This format is efficient for large datasets.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )

    parser = argparse.ArgumentParser(description="Tokenize a dataset using a pre-trained BPE tokenizer.")
    parser.add_argument(
        "--vocab-file",
        type=Path,
        default=Path("results/owt.json"),
        help="Path to the vocabulary file (JSON format).",
    )
    parser.add_argument(
        "--merges-file",
        type=Path,
        help="Path to the merges file. Optional if merges are in the vocab file.",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=Path("~/datasets/cs336/owt_train.txt").expanduser(),
        help="Path to the input text file to tokenize.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help="Path to save the tokenized output. Defaults to input-file with .bin extension.",
    )
    parser.add_argument(
        "--special-tokens",
        nargs="*",
        default=["<|endoftext|>"],
        help="A list of special tokens to add to the tokenizer.",
    )
    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = args.input_file.with_suffix(".bin")

    logging.info("Loading tokenizer...")
    tokenizer = Tokenizer.from_files(
        vocab_filepath=str(args.vocab_file),
        merges_filepath=str(args.merges_file) if args.merges_file else None,
        special_tokens=args.special_tokens,
    )
    logging.info(f"Tokenizer loaded with {len(tokenizer.vocab)} tokens.")

    if len(tokenizer.vocab) > 65536:
        logging.warning(
            f"Warning: Vocabulary size ({len(tokenizer.vocab)}) exceeds 65536. "
            "Tokens will be stored as uint16, which may cause overflow for larger token IDs.",
        )

    logging.info(f"Tokenizing {args.input_file} and streaming to {args.output_file}...")

    total_tokens = 0
    chunk_size = 8192  # Process 8k tokens at a time

    try:
        with open(args.input_file, encoding="utf-8") as f_in, open(args.output_file, "wb") as f_out:
            token_iterator = tokenizer.encode_iterable(f_in)
            while True:
                chunk = list(itertools.islice(token_iterator, chunk_size))
                if not chunk:
                    break
                token_array = np.array(chunk, dtype=np.uint16)
                token_array.tofile(f_out)
                total_tokens += len(token_array)
    except Exception as e:
        logging.error(f"An error occurred during tokenization: {e}")
        sys.exit(1)

    logging.info(f"Tokenization complete. Wrote {total_tokens} tokens.")


if __name__ == "__main__":
    main()

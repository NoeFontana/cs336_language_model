import argparse
import itertools
import logging
import sys
from pathlib import Path

import numpy as np

from cs336.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


def tokenize_dataset(
    input_file: Path,
    output_file: Path,
    vocab_file: Path,
    merges_file: Path | None = None,
    special_tokens: list[str] | None = None,
    chunk_size: int = 8192,
) -> int:
    """Tokenizes a dataset using a pre-trained tokenizer and saves the output.

    This function reads a text file, tokenizes its content, and writes the
    resulting token IDs to a binary file as a NumPy array of uint16 values.

    Args:
        input_file: Path to the input text file to tokenize.
        output_file: Path to save the tokenized output.
        vocab_file: Path to the vocabulary file (JSON format).
        merges_file: Optional path to the merges file.
        special_tokens: Optional list of special tokens to add.
        chunk_size: The number of tokens to process at a time.

    Returns:
        The total number of tokens written to the output file.

    Raises:
        ValueError: If the tokenizer's vocabulary size exceeds the limit for uint16.
        IOError: If there's an error reading the input or writing the output file.
    """
    logger.info("Loading tokenizer...")
    tokenizer = Tokenizer.from_files(
        vocab_filepath=str(vocab_file),
        merges_filepath=str(merges_file) if merges_file else None,
        special_tokens=special_tokens,
    )
    logger.info(f"Tokenizer loaded with {len(tokenizer.vocab)} tokens.")

    if len(tokenizer.vocab) > 65536:
        raise ValueError(
            f"Vocabulary size ({len(tokenizer.vocab)}) exceeds 65536, which is not supported for uint16 storage."
        )

    logger.info(f"Tokenizing {input_file} and streaming to {output_file}...")

    total_tokens = 0
    try:
        with open(input_file, encoding="utf-8") as f_in, open(output_file, "wb") as f_out:
            token_iterator = tokenizer.encode_iterable(f_in)
            while True:
                chunk = list(itertools.islice(token_iterator, chunk_size))
                if not chunk:
                    break
                token_array = np.array(chunk, dtype=np.uint16)
                token_array.tofile(f_out)
                total_tokens += len(token_array)
    except (OSError, UnicodeDecodeError) as e:
        raise OSError(f"Error during file processing: {e}") from e

    return total_tokens


def main() -> None:
    """
    Command-line interface for tokenizing a dataset.
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
        required=True,
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
        required=True,
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

    output_file = args.output_file or args.input_file.with_suffix(".bin")

    try:
        total_tokens = tokenize_dataset(
            input_file=args.input_file,
            output_file=output_file,
            vocab_file=args.vocab_file,
            merges_file=args.merges_file,
            special_tokens=args.special_tokens,
        )
        logging.info(f"Tokenization complete. Wrote {total_tokens} tokens.")
    except (OSError, ValueError) as e:
        logging.error(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

import argparse
import base64
import json
import logging
from pathlib import Path
from typing import Any

from cs336_basics.adapters import run_train_bpe

logger = logging.getLogger(__name__)


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Trains a BPE tokenizer on a dataset.

    Args:
        input_path: Path to the input corpus file.
        vocab_size: Total vocabulary size.
        special_tokens: list of special tokens to add to the vocabulary.

    Returns:
        A tuple containing the vocabulary and the merge rules.
    """
    logger.info(f"Training BPE on {input_path} with vocab size {vocab_size}")
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )
    return vocab, merges


def save_tokenizer_files(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    output_prefix: str,
) -> None:
    """Saves the vocabulary and merges to a JSON file.

    Args:
        vocab: The vocabulary mapping token IDs to token bytes.
        merges: A list of merge rules.
        output_prefix: Prefix for the output JSON file. The file will be named
            `{output_prefix}.json`.
    """
    output_path = Path(f"{output_prefix}.json")

    # Prepare data for JSON serialization (bytes need to be base64 encoded)
    serializable_vocab = {
        token_id: base64.b64encode(token_bytes).decode("ascii") for token_id, token_bytes in vocab.items()
    }
    serializable_merges = [
        (base64.b64encode(p1).decode("ascii"), base64.b64encode(p2).decode("ascii")) for p1, p2 in merges
    ]

    data_to_save: dict[str, Any] = {
        "vocab": serializable_vocab,
        "merges": serializable_merges,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=2)
    logger.info(f"Saved vocab and merges to {output_path}")


def train_and_save_bpe_tokenizer(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    output_prefix: str | None = None,
) -> None:
    """Trains a BPE tokenizer and saves the vocabulary and merges.

    Args:
        input_path: Path to the input corpus file.
        vocab_size: Total vocabulary size.
        special_tokens: list of special tokens to add to the vocabulary.
        output_prefix: Prefix for output files. If not set, uses the input
            filename stem.
    """
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)

    if output_prefix is None:
        output_prefix = Path(input_path).stem

    save_tokenizer_files(vocab, merges, output_prefix)


def main() -> None:
    """Main function to train a BPE tokenizer from the command line."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(description="Train BPE tokenizer on a dataset.")
    parser.add_argument(
        "--input-path",
        default="/home/noe/datasets/cs336/TinyStoriesV2-GPT4-valid.txt",
        type=str,
        help="Path to input corpus file (e.g., data/en.txt)",
    )
    parser.add_argument(
        "--vocab-size",
        default=10_000,
        type=int,
        help="Total vocabulary size (including special tokens)",
    )
    parser.add_argument(
        "--special-tokens",
        type=str,
        nargs="*",
        default=["<|endoftext|>"],
        help="list of special tokens to add to the vocabulary",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Prefix for output files (vocab and merges). If not set, uses input filename stem.",
    )
    args = parser.parse_args()

    train_and_save_bpe_tokenizer(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
        output_prefix=args.output_prefix,
    )


if __name__ == "__main__":
    main()

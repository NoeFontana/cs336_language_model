import argparse
import logging
from pathlib import Path

from tests.adapters import run_train_bpe


def main() -> None:
    """
    Train a BPE tokenizer on a dataset and save vocab and merges files.
    """
    parser = argparse.ArgumentParser(description="Train BPE tokenizer on a dataset.")
    parser.add_argument(
        "--input-path",
        default="data/TinyStoriesV2-GPT4-valid.txt",
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
        help="List of special tokens to add to the vocabulary",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Prefix for output files (vocab and merges). If not set, uses input filename stem.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Training BPE on {args.input_path} with vocab size {args.vocab_size}")

    vocab, merges = run_train_bpe(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=args.special_tokens,
    )

    if args.output_prefix is not None:
        output_prefix: str = args.output_prefix
        vocab_path = Path(f"{output_prefix}.vocab")
        merges_path = Path(f"{output_prefix}.merges")

        # Save vocab
        with vocab_path.open("wb") as f_vocab:
            for token_id, token_bytes in vocab.items():
                f_vocab.write(f"{token_id}\t{token_bytes!r}\n".encode())
        logger.info(f"Saved vocab to {vocab_path}")

        # Save merges
        with merges_path.open("wb") as f_merges:
            for pair in merges:
                f_merges.write(f"{pair[0]!r}\t{pair[1]!r}\n".encode())
        logger.info(f"Saved merges to {merges_path}")


if __name__ == "__main__":
    main()

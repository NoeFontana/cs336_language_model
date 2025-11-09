import argparse
import base64
import json
import logging
from pathlib import Path

from cs336.adapters import run_train_bpe


def main() -> None:
    """
    Train a BPE tokenizer on a dataset and save vocab and merges files.
    """
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
        output_path = Path(f"{output_prefix}.json")

        # Prepare data for JSON serialization
        serializable_vocab = {
            token_id: base64.b64encode(token_bytes).decode("ascii") for token_id, token_bytes in vocab.items()
        }
        serializable_merges = [
            (base64.b64encode(p1).decode("ascii"), base64.b64encode(p2).decode("ascii")) for p1, p2 in merges
        ]

        data_to_save = {
            "vocab": serializable_vocab,
            "merges": serializable_merges,
        }

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=2)
        logger.info(f"Saved vocab and merges to {output_path}")


if __name__ == "__main__":
    main()

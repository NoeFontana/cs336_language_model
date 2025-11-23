"""
Orchestrates experiments on the TinyStories dataset.

This script provides a command-line interface to run a sequence of tasks
for a language model experiment:
1.  Train a BPE tokenizer.
2.  Tokenize a dataset.
3.  Train a language model.
"""

import argparse
import logging
import sys
from pathlib import Path

from cs336.scripts.tokenize_dataset import tokenize_dataset
from cs336.scripts.train_bpe import train_and_save_bpe_tokenizer
from cs336.scripts.train_lm import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    Trainer,
    TrainerConfig,
)

# Set up a logger for this script
logger = logging.getLogger(__name__)


def main() -> None:
    """Main entry point for the script, handling command-line arguments."""
    DEFAULT_VOCAB_SIZE = 10_000

    parser = argparse.ArgumentParser(
        description="Run experiments on the TinyStories dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Sub-command to run")

    # --- Parser for train-bpe ---
    bpe_parser = subparsers.add_parser(
        "train-bpe", help="Train a BPE tokenizer.", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    bpe_parser.add_argument(
        "--input_path",
        type=str,
        default=Path("~/datasets/cs336/TinyStoriesV2-GPT4-train.txt").expanduser().as_posix(),
        help="Path to input corpus file for tokenizer training.",
    )
    bpe_parser.add_argument(
        "--vocab_size",
        default=DEFAULT_VOCAB_SIZE,
        type=int,
        help="Total vocabulary size for the tokenizer.",
    )
    bpe_parser.add_argument(
        "--special_tokens",
        type=str,
        nargs="*",
        default=["<|endoftext|>"],
        help="List of special tokens to add to the vocabulary.",
    )
    bpe_parser.add_argument(
        "--output_prefix",
        type=str,
        default=Path("~/datasets/cs336/tinystories_tokenizer").expanduser().as_posix(),
        help="Prefix for output tokenizer files (e.g., 'my_tokenizer' -> 'my_tokenizer.json').",
    )

    # --- Parser for tokenize ---
    tokenize_parser = subparsers.add_parser(
        "tokenize", help="Tokenize a dataset.", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    tokenize_parser.add_argument(
        "--vocab_file",
        type=Path,
        default=Path("~/datasets/cs336/tinystories_tokenizer.json").expanduser(),
        help="Path to the vocabulary file (JSON format).",
    )
    tokenize_parser.add_argument(
        "--merges_file",
        type=Path,
        help="Path to the merges file. Optional if merges are in the vocab file.",
    )
    tokenize_parser.add_argument(
        "--input_file",
        type=Path,
        required=True,
        help="Path to the input text file to tokenize (e.g., 'data/train.txt').",
    )
    tokenize_parser.add_argument(
        "--output_folder",
        default=Path("~/datasets/cs336").expanduser(),
        type=Path,
        help="Path to save the tokenized output. Defaults to the directory of the input file.",
    )
    tokenize_parser.add_argument(
        "--special_tokens",
        nargs="*",
        default=["<|endoftext|>"],
        help="A list of special tokens.",
    )

    # --- Parser for train-lm ---
    lm_parser = subparsers.add_parser(
        "train-lm", help="Train a Transformer Language Model.", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Add arguments to the train-lm subparser, which will override TrainingConfig defaults
    lm_parser.add_argument(
        "--train_data_path",
        type=str,
        default=Path("~/datasets/cs336/TinyStoriesV2-GPT4-train.bin").expanduser().as_posix(),
    )
    lm_parser.add_argument(
        "--val_data_path",
        type=str,
        default=Path("~/datasets/cs336/TinyStoriesV2-GPT4-valid.bin").expanduser().as_posix(),
    )
    lm_parser.add_argument("--checkpoint_path", type=str, default="checkpoints/tinystories")
    lm_parser.add_argument("--vocab_size", type=int, default=DEFAULT_VOCAB_SIZE)
    lm_parser.add_argument("--context_length", type=int, default=256)
    lm_parser.add_argument("--d_model", type=int, default=512)
    lm_parser.add_argument("--d_ff", type=int, default=1344)
    lm_parser.add_argument("--num_heads", type=int, default=16)
    lm_parser.add_argument("--num_layers", type=int, default=4)
    lm_parser.add_argument("--batch_size", type=int, default=64)
    lm_parser.add_argument("--learning_rate", type=float, default=1e-2)
    lm_parser.add_argument("--min_learning_rate", type=float, default=3e-5)
    lm_parser.add_argument("--warmup_steps", type=int, default=2000)
    lm_parser.add_argument("--max_steps", type=int, default=20000)
    lm_parser.add_argument("--wandb_project", type=str, default="tiny-stories-experiments")
    lm_parser.add_argument("--wandb_run_name", type=str)
    lm_parser.add_argument("--resume_from_checkpoint", type=str)
    lm_parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

    try:
        if args.command == "train-bpe":
            logger.info("Starting BPE tokenizer training...")
            train_and_save_bpe_tokenizer(
                input_path=args.input_path,
                vocab_size=args.vocab_size,
                special_tokens=args.special_tokens,
                output_prefix=args.output_prefix,
            )
            logger.info("BPE tokenizer training complete.")

        elif args.command == "tokenize":
            logger.info("Starting dataset tokenization...")
            output_file = args.output_folder / args.input_file.with_suffix(".bin").name

            total_tokens = tokenize_dataset(
                input_file=args.input_file,
                output_file=output_file,
                vocab_file=args.vocab_file,
                merges_file=args.merges_file,
                special_tokens=args.special_tokens,
            )
            logger.info(f"Dataset tokenization complete. Wrote {total_tokens} tokens.")

        elif args.command == "train-lm":
            logger.info("Starting language model training...")
            model_config = ModelConfig(
                vocab_size=args.vocab_size,
                context_length=args.context_length,
                d_model=args.d_model,
                d_ff=args.d_ff,
                num_heads=args.num_heads,
                num_layers=args.num_layers,
            )
            optimizer_config = OptimizerConfig(
                learning_rate=args.learning_rate,
                min_learning_rate=args.min_learning_rate,
                warmup_steps=args.warmup_steps,
            )
            data_config = DataConfig(
                train_data_path=args.train_data_path,
                val_data_path=args.val_data_path,
            )
            trainer_config = TrainerConfig(
                batch_size=args.batch_size,
                max_steps=args.max_steps,
                device=args.device,
                checkpoint_path=args.checkpoint_path,
                resume_from_checkpoint=args.resume_from_checkpoint,
                wandb_project=args.wandb_project,
                wandb_run_name=args.wandb_run_name,
            )

            config = ExperimentConfig(
                model=model_config,
                optimizer=optimizer_config,
                data=data_config,
                trainer=trainer_config,
            )

            trainer = Trainer(config)
            trainer.train()
            trainer.close()
            logger.info("Language model training complete.")

    except (OSError, ValueError, FileNotFoundError) as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

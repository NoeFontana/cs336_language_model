import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.optim as optim
import wandb

from cs336.checkpoint import load_checkpoint, save_checkpoint
from cs336.data import get_batch
from cs336.loss import cross_entropy
from cs336.optim.scheduler import lr_cosine_schedule
from cs336.transformer import TransformerLM


def get_args() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a Transformer Language Model.")
    parser.add_argument(
        "--train_data_path",
        type=str,
        default="./results/owt_train.bin",
        help="Path to memory-mapped training data.",
    )
    parser.add_argument(
        "--val_data_path",
        type=str,
        default="./results/owt_valid.bin",
        help="Path to memory-mapped validation data.",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size.")
    parser.add_argument("--context_length", type=int, default=256, help="Context length for the model.")
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension.")
    parser.add_argument("--d_ff", type=int, default=2048, help="Feed-forward dimension.")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--num_layers", type=int, default=6, help="Number of transformer layers.")
    parser.add_argument("--theta", type=float, default=10000.0, help="Theta for RoPE.")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Peak learning rate.")
    parser.add_argument("--min_learning_rate", type=float, default=3e-5, help="Minimum learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay.")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="Number of warmup steps.")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--max_steps", type=int, help="Maximum number of training steps.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to train on.")
    parser.add_argument("--log_period", type=int, default=10, help="Log training progress every N steps.")
    parser.add_argument("--val_period", type=int, default=500, help="Run validation every N steps.")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints", help="Path to save checkpoints.")
    parser.add_argument("--resume_from_checkpoint", type=str, help="Path to a checkpoint to resume training from.")
    parser.add_argument("--wandb_project", type=str, default="cs336-language-model", help="Wandb project name.")
    parser.add_argument("--wandb_run_name", type=str, help="Wandb run name.")
    return parser.parse_args()


@dataclass
class TrainingState:
    """A dataclass to hold the state of the training loop."""

    model: TransformerLM
    optimizer: optim.AdamW
    step: int = 0


def setup_training(
    args: argparse.Namespace,
) -> tuple[np.memmap, np.memmap, torch.nn.Module, torch.optim.Optimizer, int]:
    """
    Set up training by loading data, initializing the model and optimizer,
    and optionally resuming from a checkpoint.

    Args:
        args: Command-line arguments.

    Returns:
        A tuple containing the training data, validation data, and the training state.
    """
    logger = logging.getLogger(__name__)

    # Data loading
    logger.info("Loading data...")
    train_data = np.memmap(args.train_data_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(args.val_data_path, dtype=np.uint16, mode="r")

    # Model and optimizer setup
    logger.info("Setting up model and optimizer...")
    model: torch.nn.Module = TransformerLM(
        vocab_size=args.vocab_size,
        max_seq_len=args.context_length,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        theta=args.theta,
    ).to(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    step = 0

    if args.resume_from_checkpoint:
        logger.info(f"Resuming training from checkpoint {args.resume_from_checkpoint}")
        step = load_checkpoint(args.resume_from_checkpoint, model, optimizer)

    return train_data, val_data, model, optimizer, step


def validate_model(model: torch.nn.Module, val_data: np.memmap, args: argparse.Namespace) -> float:
    """
    Run a validation step and log the validation loss.

    Args:
        model: The model to validate.
        val_data: The validation dataset.
        args: Command-line arguments.

    Returns:
        The validation loss.
    """
    logger = logging.getLogger(__name__)
    logger.info("Running validation...")
    model.eval()
    with torch.inference_mode():
        val_x, val_y = get_batch(val_data, args.batch_size, args.context_length, args.device)
        val_loss = cross_entropy(model(val_x), val_y)
    model.train()
    return val_loss.item()


def main() -> None:
    """
    Main function for training a Transformer Language Model.
    This script handles:
    - Parsing command-line arguments for training configuration.
    - Loading training and validation datasets.
    - Initializing the Transformer model and AdamW optimizer.
    - Optionally resuming training from a checkpoint.
    - Running the training loop, including:
        - Cosine learning rate schedule with warmup.
        - Forward and backward passes.
        - Logging training progress.
        - Periodic validation.
        - Saving checkpoints.
    - Saving a final model checkpoint upon completion.
    """
    args = get_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger = logging.getLogger(__name__)

    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=cast(Any, args))

    Path(args.checkpoint_path).mkdir(exist_ok=True)

    train_data, val_data, model, optimizer, step = setup_training(args)

    # Training loop
    logger.info("Starting training...")
    total_steps = (
        args.max_steps
        if args.max_steps
        else (len(train_data) // (args.batch_size * args.context_length)) * args.num_epochs
    )
    for epoch in range(args.num_epochs):
        for _ in range(0, len(train_data) - 1, args.batch_size):
            if args.max_steps and step >= args.max_steps:
                break

            start_time = time.perf_counter()

            # Learning rate schedule
            lr = lr_cosine_schedule(
                it=step,
                max_learning_rate=args.learning_rate,
                min_learning_rate=args.min_learning_rate,
                tw=args.warmup_steps,
                tc=total_steps,
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # Prepare batch
            x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)

            # Forward pass
            optimizer.zero_grad()
            logits = model(x)
            loss = cross_entropy(logits, y)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            end_time = time.perf_counter()

            # Logging
            if step % args.log_period == 0:
                logger.info(
                    f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}, "
                    f"LR: {lr:.6f}, Time: {(end_time - start_time) * 1000:.2f}ms"
                )
                wandb.log(
                    {"train/loss": loss.item(), "train/lr": lr, "step": step},
                )

            # Validation and checkpointing
            if step % args.val_period == 0 and step > 0:
                val_loss = validate_model(model, val_data, args)
                logger.info(f"Validation Loss: {val_loss:.4f}")
                wandb.log({"val/loss": val_loss, "step": step})

                # Save checkpoint
                checkpoint_file = Path(args.checkpoint_path) / f"step_{step}.pt"
                save_checkpoint(model, optimizer, step, checkpoint_file)
                logger.info(f"Saved checkpoint to {checkpoint_file}")

            step += 1

    # Save final checkpoint
    final_checkpoint_path = Path(args.checkpoint_path) / "final.pt"
    save_checkpoint(model, optimizer, step, final_checkpoint_path)
    logger.info(f"Training finished. Final checkpoint saved to {final_checkpoint_path}")

    wandb.finish()


if __name__ == "__main__":
    main()

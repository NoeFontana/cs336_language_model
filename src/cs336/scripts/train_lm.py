import argparse
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import wandb

from cs336.checkpoint import load_checkpoint, save_checkpoint
from cs336.data import get_batch
from cs336.loss import cross_entropy
from cs336.optim.scheduler import lr_cosine_schedule
from cs336.transformer import TransformerLM

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for training a Transformer Language Model."""

    train_data_path: str = "./results/owt_train.bin"
    val_data_path: str = "./results/owt_valid.bin"
    batch_size: int = 64
    vocab_size: int = 32000
    context_length: int = 256
    d_model: int = 512
    d_ff: int = 2048
    num_heads: int = 8
    num_layers: int = 6
    theta: float = 10000.0
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    num_epochs: int = 1
    max_steps: int | None = None
    device: str = "cuda"
    log_period: int = 10
    val_period: int = 500
    checkpoint_path: str = "checkpoints"
    resume_from_checkpoint: str | None = None
    wandb_project: str = "cs336-language-model"
    wandb_run_name: str | None = None


class Trainer:
    """A class to encapsulate the training process of a TransformerLM."""

    def __init__(self, config: TrainingConfig):
        """Initializes the Trainer.

        Args:
            config: The training configuration.
        """
        self.config = config
        wandb.init(project=config.wandb_project, name=config.wandb_run_name, config=asdict(config))

        # Setup directories
        Path(self.config.checkpoint_path).mkdir(exist_ok=True, parents=True)

        # Setup data, model, and optimizer
        self._setup_training()

    def _setup_training(self) -> None:
        """Sets up data, model, optimizer, and resumes from checkpoint if specified."""
        logger.info("Loading data...")
        self.train_data = np.memmap(self.config.train_data_path, dtype=np.uint16, mode="r")
        self.val_data = np.memmap(self.config.val_data_path, dtype=np.uint16, mode="r")

        logger.info("Setting up model and optimizer...")
        self.model: torch.nn.Module = TransformerLM(
            vocab_size=self.config.vocab_size,
            max_seq_len=self.config.context_length,
            d_model=self.config.d_model,
            d_ff=self.config.d_ff,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            theta=self.config.theta,
        ).to(self.config.device)
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay
        )
        self.step = 0

        if self.config.resume_from_checkpoint:
            logger.info(f"Resuming training from checkpoint {self.config.resume_from_checkpoint}")
            self.step = load_checkpoint(self.config.resume_from_checkpoint, self.model, self.optimizer)

    def _validate(self) -> float:
        """Runs a validation step and returns the loss.

        Returns:
            The validation loss.
        """
        logger.info("Running validation...")
        self.model.eval()
        with torch.inference_mode():
            val_x, val_y = get_batch(
                self.val_data, self.config.batch_size, self.config.context_length, self.config.device
            )
            val_loss = cross_entropy(self.model(val_x), val_y)
        self.model.train()
        return val_loss.item()

    def train(self) -> None:
        """Runs the main training loop."""
        logger.info("Starting training...")
        total_steps = (
            self.config.max_steps
            if self.config.max_steps
            else (len(self.train_data) // (self.config.batch_size * self.config.context_length))
            * self.config.num_epochs
        )

        for epoch in range(self.config.num_epochs):
            for _ in range(0, len(self.train_data) - 1, self.config.batch_size):
                if self.config.max_steps and self.step >= self.config.max_steps:
                    break

                start_time = time.perf_counter()

                lr = lr_cosine_schedule(
                    it=self.step,
                    max_learning_rate=self.config.learning_rate,
                    min_learning_rate=self.config.min_learning_rate,
                    tw=self.config.warmup_steps,
                    tc=total_steps,
                )
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

                x, y = get_batch(
                    self.train_data, self.config.batch_size, self.config.context_length, self.config.device
                )

                self.optimizer.zero_grad()
                logits = self.model(x)
                loss = cross_entropy(logits, y)
                loss.backward()
                self.optimizer.step()

                end_time = time.perf_counter()

                if self.step % self.config.log_period == 0:
                    logger.info(
                        f"Epoch {epoch}, Step {self.step}, Loss: {loss.item():.4f}, "
                        f"LR: {lr:.6f}, Time: {(end_time - start_time) * 1000:.2f}ms"
                    )
                    wandb.log({"train/loss": loss.item(), "train/lr": lr, "step": self.step})

                if self.step > 0 and self.step % self.config.val_period == 0:
                    val_loss = self._validate()
                    logger.info(f"Validation Loss: {val_loss:.4f}")
                    wandb.log({"val/loss": val_loss, "step": self.step})

                    checkpoint_file = Path(self.config.checkpoint_path) / f"step_{self.step}.pt"
                    save_checkpoint(self.model, self.optimizer, self.step, checkpoint_file)
                    logger.info(f"Saved checkpoint to {checkpoint_file}")

                self.step += 1

        final_checkpoint_path = Path(self.config.checkpoint_path) / "final.pt"
        save_checkpoint(self.model, self.optimizer, self.step, final_checkpoint_path)
        logger.info(f"Training finished. Final checkpoint saved to {final_checkpoint_path}")

    def close(self) -> None:
        """Cleans up resources, like finishing the WandB run."""
        wandb.finish()


def get_cli_args() -> argparse.Namespace:
    """Parse and return command-line arguments for the training script."""
    parser = argparse.ArgumentParser(description="Train a Transformer Language Model.")
    # We can add arguments dynamically from the dataclass fields
    # For now, we'll keep it explicit for clarity and consistency with the original
    parser.add_argument("--train_data_path", type=str, help="Path to memory-mapped training data.")
    parser.add_argument("--val_data_path", type=str, help="Path to memory-mapped validation data.")
    parser.add_argument("--batch_size", type=int, help="Batch size for training.")
    parser.add_argument("--vocab_size", type=int, help="Vocabulary size.")
    parser.add_argument("--context_length", type=int, help="Context length for the model.")
    parser.add_argument("--d_model", type=int, help="Model dimension.")
    parser.add_argument("--d_ff", type=int, help="Feed-forward dimension.")
    parser.add_argument("--num_heads", type=int, help="Number of attention heads.")
    parser.add_argument("--num_layers", type=int, help="Number of transformer layers.")
    parser.add_argument("--theta", type=float, help="Theta for RoPE.")
    parser.add_argument("--learning_rate", type=float, help="Peak learning rate.")
    parser.add_argument("--min_learning_rate", type=float, help="Minimum learning rate.")
    parser.add_argument("--weight_decay", type=float, help="Weight decay.")
    parser.add_argument("--warmup_steps", type=int, help="Number of warmup steps.")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs.")
    parser.add_argument("--max_steps", type=int, help="Maximum number of training steps.")
    parser.add_argument("--device", type=str, help="Device to train on.")
    parser.add_argument("--log_period", type=int, help="Log training progress every N steps.")
    parser.add_argument("--val_period", type=int, help="Run validation every N steps.")
    parser.add_argument("--checkpoint_path", type=str, help="Path to save checkpoints.")
    parser.add_argument("--resume_from_checkpoint", type=str, help="Path to a checkpoint to resume training from.")
    parser.add_argument("--wandb_project", type=str, help="Wandb project name.")
    parser.add_argument("--wandb_run_name", type=str, help="Wandb run name.")
    return parser.parse_args()


def main() -> None:
    """
    Main function for training a Transformer Language Model from the command line.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    args = get_cli_args()
    # Filter out None values so we can use dataclass defaults
    cli_config = {k: v for k, v in vars(args).items() if v is not None}
    config = TrainingConfig(**cli_config)

    trainer = Trainer(config)
    try:
        trainer.train()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()

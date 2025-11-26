import logging
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import hydra
import torch
import torch.optim as optim
import wandb
from omegaconf import DictConfig

from cs336.checkpoint import save_checkpoint
from cs336.data import create_train_loader, create_val_loader
from cs336.loss import cross_entropy
from cs336.optim.clip import gradient_clipping
from cs336.optim.scheduler import lr_cosine_schedule
from cs336.transformer import TransformerLM

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 32000
    context_length: int = 256
    d_model: int = 512
    d_ff: int = 2048
    num_heads: int = 8
    num_layers: int = 6
    theta: float = 10000.0


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    warmup_steps: int = 2000


@dataclass(frozen=True)
class DataConfig:
    train_data_path: str = "./results/owt_train.bin"
    val_data_path: str = "./results/owt_valid.bin"
    seed: int = 42


@dataclass(frozen=True)
class TrainerConfig:
    batch_size: int = 64
    num_epochs: int = 1
    max_steps: int = 1_000_000
    device: str = "cuda"
    log_period: int = 10
    val_period: int = 500
    checkpoint_path: str = "checkpoints"
    resume_from_checkpoint: str | None = None
    wandb_project: str = "cs336-language-model"
    wandb_run_name: str | None = None
    use_torch_compile: bool = True
    num_workers: int = 4


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    optimizer: OptimizerConfig
    data: DataConfig
    trainer: TrainerConfig


class Trainer:
    """A class to encapsulate the training process of a TransformerLM."""

    def __init__(self, config: ExperimentConfig):
        """Initializes the Trainer.
        Args:
            config: The training configuration.
        """
        self.config = config
        wandb.init(
            project=config.trainer.wandb_project,
            name=config.trainer.wandb_run_name,
            config=asdict(config),
            settings=wandb.Settings(save_code=True),
        )

        # Setup directories
        Path(self.config.trainer.checkpoint_path).mkdir(exist_ok=True, parents=True)

        # Setup data, model, and optimizer
        self._setup_training()

    def _setup_training(self) -> None:
        """Sets up data, model, optimizer, and resumes from checkpoint if specified."""
        loaded_checkpoint = None
        if self.config.trainer.resume_from_checkpoint:
            logger.info(f"Resuming training from checkpoint {self.config.trainer.resume_from_checkpoint}")
            loaded_checkpoint = torch.load(self.config.trainer.resume_from_checkpoint, map_location="cpu")
            if "config" in loaded_checkpoint:
                logger.info("Found config in checkpoint, updating configuration.")
                loaded_config = loaded_checkpoint["config"]
                self.config = ExperimentConfig(
                    model=ModelConfig(**loaded_config["model"]),
                    optimizer=OptimizerConfig(**loaded_config["optimizer"]),
                    data=DataConfig(**loaded_config["data"]),
                    trainer=TrainerConfig(**loaded_config["trainer"]),
                )

        logger.info("Setting up model and optimizer...")
        self.model: torch.nn.Module = TransformerLM(
            vocab_size=self.config.model.vocab_size,
            max_seq_len=self.config.model.context_length,
            d_model=self.config.model.d_model,
            d_ff=self.config.model.d_ff,
            num_heads=self.config.model.num_heads,
            num_layers=self.config.model.num_layers,
            theta=self.config.model.theta,
        ).to(self.config.trainer.device)

        if self.config.trainer.use_torch_compile:
            logger.info("Compiling model with torch.compile...")
            self.model.compile()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.optimizer.learning_rate,
            weight_decay=self.config.optimizer.weight_decay,
        )
        self.step = 0

        if loaded_checkpoint:
            if not all(k in loaded_checkpoint for k in ["model_state_dict", "optimizer_state_dict", "iteration"]):
                raise ValueError("Checkpoint is missing required keys.")

            self.model.load_state_dict(loaded_checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(loaded_checkpoint["optimizer_state_dict"])
            self.step = loaded_checkpoint["iteration"]

    def _get_infinite_loader(self, loader: torch.utils.data.DataLoader) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
        while True:
            yield from loader

    def _validate(self, val_loader: Iterable[tuple[torch.Tensor, torch.Tensor]]) -> float:
        """Runs a validation step and returns the loss.
        Returns:
            The validation loss.
        """
        logger.info("Running validation...")
        self.model.eval()
        device = self.config.trainer.device
        with torch.inference_mode():
            total_val_loss = 0.0
            num_val_batches = 0
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                val_loss = cross_entropy(self.model(x), y)
                total_val_loss += val_loss.item()
                num_val_batches += 1
        self.model.train()
        return total_val_loss / num_val_batches if num_val_batches > 0 else 0.0

    def train(self) -> None:
        """Runs the main training loop."""
        logger.info("Starting training...")

        train_loader = create_train_loader(
            data_path=self.config.data.train_data_path,
            context_length=self.config.model.context_length,
            batch_size=self.config.trainer.batch_size,
            num_workers=self.config.trainer.num_workers,
            persistent_workers=True,
            seed=self.config.data.seed,
        )
        val_loader = create_val_loader(
            data_path=self.config.data.val_data_path,
            context_length=self.config.model.context_length,
            batch_size=self.config.trainer.batch_size,
            num_workers=self.config.trainer.num_workers,
            persistent_workers=True,
        )

        total_steps = self.config.trainer.max_steps
        if total_steps is None:
            if self.config.trainer.num_epochs is None:
                raise ValueError("Either trainer.max_steps or trainer.num_epochs must be specified.")
            total_steps = int(self.config.trainer.num_epochs * len(train_loader))
            logger.info(
                f"max_steps not specified, training for {total_steps} steps ({self.config.trainer.num_epochs} epochs)."
            )

        train_iter = iter(self._get_infinite_loader(train_loader))
        self.model.train()
        scaler = torch.GradScaler()

        while self.step < total_steps:
            start_time = time.perf_counter()
            x, y = next(train_iter)

            lr = lr_cosine_schedule(
                it=self.step,
                max_learning_rate=self.config.optimizer.learning_rate,
                min_learning_rate=self.config.optimizer.min_learning_rate,
                tw=self.config.optimizer.warmup_steps,
                tc=total_steps,
            )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

            x = x.to(self.config.trainer.device, non_blocking=True)
            y = y.to(self.config.trainer.device, non_blocking=True)
            self.optimizer.zero_grad(True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = self.model(x)
                loss = cross_entropy(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(self.optimizer)

            gradient_clipping(self.model.parameters(), 1.0)

            scaler.step(self.optimizer)
            scaler.update()

            end_time = time.perf_counter()

            if self.step % self.config.trainer.log_period == 0:
                logger.info(
                    f"Step {self.step}, Loss: {loss.item():.4f}, "
                    f"LR: {lr:.6f}, Time: {(end_time - start_time) * 1000:.2f}ms"
                )
                wandb.log({"train/loss": loss.item(), "train/lr": lr, "step": self.step})

            if self.step > 0 and self.step % self.config.trainer.val_period == 0:
                val_loss = self._validate(val_loader)
                logger.info(f"Validation Loss: {val_loss:.4f}")
                wandb.log({"val/loss": val_loss, "step": self.step})

                checkpoint_file = Path(self.config.trainer.checkpoint_path) / f"step_{self.step}.pt"
                save_checkpoint(self.model, self.optimizer, self.step, checkpoint_file, config=asdict(self.config))
                logger.info(f"Saved checkpoint to {checkpoint_file}")

            self.step += 1

        final_checkpoint_path = Path(self.config.trainer.checkpoint_path) / "final.pt"
        save_checkpoint(
            self.model,
            self.optimizer,
            self.step,
            final_checkpoint_path,
            config=asdict(self.config),
        )
        logger.info(f"Training finished. Final checkpoint saved to {final_checkpoint_path}")

    def close(self) -> None:
        """Cleans up resources, like finishing the WandB run."""
        wandb.finish()


@hydra.main(version_base=None, config_path="conf", config_name="experiments")
def main(cfg: DictConfig) -> None:
    """
    Main function for training a Transformer Language Model from the command line.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Instantiate the configuration object from the Hydra config
    config = ExperimentConfig(
        model=ModelConfig(**cfg.model),
        optimizer=OptimizerConfig(**cfg.optimizer),
        data=DataConfig(**cfg.data),
        trainer=TrainerConfig(**cfg.trainer),
    )

    trainer = Trainer(config)
    try:
        trainer.train()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()

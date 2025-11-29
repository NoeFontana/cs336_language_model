import contextlib
import logging
import time
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

import hydra
import torch
import wandb
from omegaconf import DictConfig

from cs336_basics.checkpoint import save_checkpoint
from cs336_basics.config import (
    AdamWConfig,
    BaseOptimizerConfig,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    MuonConfig,
    ProfilerConfig,
    TrainerConfig,
)
from cs336_basics.data import create_train_loader, create_val_loader
from cs336_basics.factory import create_optimizer
from cs336_basics.loss.cross_entropy import CrossEntropyLoss
from cs336_basics.optim.clip import gradient_clipping
from cs336_basics.optim.scheduler import lr_cosine_schedule
from cs336_basics.transformer import TransformerLM

logger = logging.getLogger(__name__)


def get_optimizer_config(cfg: DictConfig) -> BaseOptimizerConfig:
    """
    Parses the Hydra config to return the appropriate OptimizerConfig dataclass.
    """
    opt_cfg_dict = cfg.optimizer
    name = opt_cfg_dict.get("name")

    if name == "muon":
        return MuonConfig(**opt_cfg_dict)
    # Default to AdamW for 'adamw' or fallback
    return AdamWConfig(**opt_cfg_dict)


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

                # Reconstruct optimizer config correctly
                opt_config_dict = loaded_config["optimizer"]
                if opt_config_dict.get("name") == "muon":
                    opt_config = MuonConfig(**opt_config_dict)
                else:
                    opt_config = AdamWConfig(**opt_config_dict)

                # Check if profiler config exists in checkpoint (backward compatibility)
                profiler_cfg = loaded_config.get("profiler", asdict(ProfilerConfig()))

                self.config = ExperimentConfig(
                    model=ModelConfig(**loaded_config["model"]),
                    optimizer=opt_config,
                    data=DataConfig(**loaded_config["data"]),
                    trainer=TrainerConfig(**loaded_config["trainer"]),
                    profiler=ProfilerConfig(**profiler_cfg),
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
            ffn_type=self.config.model.ffn_type,
            qk_norm=self.config.model.qk_norm,
        ).to(self.config.trainer.device)
        self.loss: torch.nn.Module = CrossEntropyLoss().to(self.config.trainer.device)

        if self.config.trainer.use_torch_compile:
            logger.info("Compiling model with torch.compile...")
            self.model.compile(mode="max-autotune")
            self.loss.compile(mode="max-autotune")

        self.optimizer = create_optimizer(self.model, self.config.optimizer)
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
        """Runs a validation step and returns the loss."""
        logger.info("Running validation...")
        self.model.eval()
        device = self.config.trainer.device
        with torch.inference_mode():
            total_val_loss = 0.0
            num_val_batches = 0
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                val_loss = self.loss(self.model(x), y)
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

        profiler_context = (
            torch.profiler.profile(
                schedule=torch.profiler.schedule(
                    wait=self.config.profiler.wait,
                    warmup=self.config.profiler.warmup,
                    active=self.config.profiler.active,
                    repeat=self.config.profiler.repeat,
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(self.config.profiler.dirpath),
                record_shapes=True,
                profile_memory=True,
                with_stack=False,  # DISABLE this to reduce overhead
                with_flops=True,  # ENABLE this to see Model FLOPS Utilization (MFU)
            )
            if self.config.profiler.enabled
            else contextlib.nullcontext()
        )

        with profiler_context as prof:
            while self.step < total_steps:
                start_time = time.perf_counter()
                x, y = next(train_iter)

                base_lr = lr_cosine_schedule(
                    it=self.step,
                    max_learning_rate=self.config.optimizer.learning_rate,
                    min_learning_rate=self.config.optimizer.min_learning_rate,
                    tw=self.config.optimizer.warmup_steps,
                    tc=total_steps,
                )

                # Update Learning Rate (Handling Muon)
                for param_group in self.optimizer.param_groups:
                    if param_group.get("use_muon", False):
                        if isinstance(self.config.optimizer, MuonConfig):
                            param_group["lr"] = self.config.optimizer.muon_learning_rate * (
                                base_lr / self.config.optimizer.learning_rate
                            )
                    else:
                        param_group["lr"] = base_lr

                x = x.to(self.config.trainer.device, non_blocking=True)
                y = y.to(self.config.trainer.device, non_blocking=True)
                self.optimizer.zero_grad(True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = self.model(x)
                    loss = self.loss(logits, y)

                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)

                gradient_clipping(self.model.parameters(), 1.0)

                scaler.step(self.optimizer)
                scaler.update()

                end_time = time.perf_counter()

                if self.step % self.config.trainer.log_period == 0:
                    logger.info(
                        f"Step {self.step}, Loss: {loss.item():.4f}, "
                        f"LR: {base_lr:.6f}, Time: {(end_time - start_time) * 1000:.2f}ms"
                    )
                    wandb.log({"train/loss": loss.item(), "train/lr": base_lr, "step": self.step})

                if self.step > 0 and self.step % self.config.trainer.val_period == 0:
                    val_loss = self._validate(val_loader)
                    logger.info(f"Validation Loss: {val_loss:.4f}")
                    wandb.log({"val/loss": val_loss, "step": self.step})

                    checkpoint_file = Path(self.config.trainer.checkpoint_path) / f"step_{self.step}.pt"
                    save_checkpoint(self.model, self.optimizer, self.step, checkpoint_file, config=asdict(self.config))
                    logger.info(f"Saved checkpoint to {checkpoint_file}")

                self.step += 1

                if self.config.profiler.enabled:
                    prof.step()  # type: ignore[possibly-missing-attribute,union-attr]

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
    # We need to handle missing profiler config if older config format is used,
    # but Hydra should handle defaults if we added them to yaml.
    # Since we don't have a default profiler yaml, we should probably check if it's in cfg.
    # Assuming cfg structure matches the dataclasses.

    profiler_cfg = cfg.get("profiler")
    if profiler_cfg is None:
        profiler_config = ProfilerConfig()
    else:
        profiler_config = ProfilerConfig(**profiler_cfg)

    config = ExperimentConfig(
        model=ModelConfig(**cfg.model),
        optimizer=get_optimizer_config(cfg),
        data=DataConfig(**cfg.data),
        trainer=TrainerConfig(**cfg.trainer),
        profiler=profiler_config,
    )

    trainer = Trainer(config)
    try:
        trainer.train()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()

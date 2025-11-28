"""
Orchestrates experiments.

This script provides a command-line interface to run a sequence of tasks
for a language model experiment:
1.  Train a BPE tokenizer.
2.  Tokenize a dataset.
3.  Train a language model.
"""

import logging
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from cs336.scripts.tokenize_dataset import tokenize_dataset
from cs336.scripts.train_bpe import train_and_save_bpe_tokenizer
from cs336.scripts.train_lm import (
    AdamWConfig,
    BaseOptimizerConfig,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    MuonConfig,
    ProfilerConfig,
    Trainer,
    TrainerConfig,
)

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Set up a logger for this script
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="experiments")
def main(cfg: DictConfig) -> None:
    """Main entry point for the script, handling command-line arguments."""

    # Access the task configuration.
    # We expect 'task' to be present in the configuration.
    if "task" not in cfg:
        logger.error("No task specified in configuration.")
        return

    task_name = cfg.task.name
    logger.info(f"Running task: {task_name}")

    try:
        if task_name == "train_bpe":
            logger.info("Starting BPE tokenizer training...")
            # Expand user path manually since OmegaConf doesn't do it
            input_path = str(Path(cfg.task.input_path).expanduser())
            output_prefix = str(Path(cfg.task.output_prefix).expanduser())

            train_and_save_bpe_tokenizer(
                input_path=input_path,
                vocab_size=cfg.task.vocab_size,
                special_tokens=list(cfg.task.special_tokens),
                output_prefix=output_prefix,
            )
            logger.info("BPE tokenizer training complete.")

        elif task_name == "tokenize":
            logger.info("Starting dataset tokenization...")
            input_file = Path(cfg.task.input_file).expanduser()
            output_folder = Path(cfg.task.output_folder).expanduser()
            vocab_file = Path(cfg.task.vocab_file).expanduser()

            merges_file = None
            if cfg.task.merges_file:
                merges_file = Path(cfg.task.merges_file).expanduser()

            output_file = output_folder / input_file.with_suffix(".bin").name

            # Ensure output directory exists
            output_folder.mkdir(parents=True, exist_ok=True)

            total_tokens = tokenize_dataset(
                input_file=input_file,
                output_file=output_file,
                vocab_file=vocab_file,
                merges_file=merges_file,
                special_tokens=list(cfg.task.special_tokens),
            )
            logger.info(f"Dataset tokenization complete. Wrote {total_tokens} tokens.")

        elif task_name == "train_lm":
            logger.info("Starting language model training...")

            # Helper to expand paths in a dict
            def expand_paths(config_dict, keys):
                for k in keys:
                    if k in config_dict and config_dict[k] and isinstance(config_dict[k], str):
                        config_dict[k] = str(Path(config_dict[k]).expanduser())
                return config_dict

            data_kwargs = OmegaConf.to_container(cfg.data, resolve=True)
            data_kwargs = expand_paths(data_kwargs, ["train_data_path", "val_data_path"])

            trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
            trainer_kwargs = expand_paths(trainer_kwargs, ["checkpoint_path"])

            model_config = ModelConfig(**cfg.model)

            optimizer_config: BaseOptimizerConfig
            opt_dict = cfg.optimizer
            if opt_dict.get("name") == "muon":
                optimizer_config = MuonConfig(**opt_dict)
            else:
                optimizer_config = AdamWConfig(**opt_dict)

            data_config = DataConfig(**data_kwargs)
            trainer_config = TrainerConfig(**trainer_kwargs)
            profiler_config = ProfilerConfig(**cfg.profiler) if "profiler" in cfg else ProfilerConfig()

            config = ExperimentConfig(
                model=model_config,
                optimizer=optimizer_config,
                data=data_config,
                trainer=trainer_config,
                profiler=profiler_config,
            )

            trainer = Trainer(config)
            trainer.train()
            trainer.close()
            logger.info("Language model training complete.")

        else:
            logger.error(f"Unknown task: {task_name}")

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

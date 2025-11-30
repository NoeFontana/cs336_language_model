import contextlib
import logging
import os
import statistics
import timeit

import hydra
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from cs336_basics.loss.cross_entropy import CrossEntropyLoss
from cs336_basics.transformer import TransformerLM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main function for benchmarking the Transformer model.

    Args:
        cfg: The Hydra configuration object containing model and benchmark settings.
    """
    device_name = cfg.benchmark.device
    if device_name == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device_name = "cpu"

    device = torch.device(device_name)
    logger.info(f"Using device: {device}")

    logger.info("Initializing model...")
    model_cfg = cfg.model
    model = TransformerLM(
        vocab_size=model_cfg.vocab_size,
        num_layers=model_cfg.num_layers,
        d_model=model_cfg.d_model,
        num_heads=model_cfg.num_heads,
        d_ff=model_cfg.d_ff,
        max_seq_len=model_cfg.context_length,
        theta=model_cfg.theta,
        ffn_type=model_cfg.ffn_type,
        qk_norm=model_cfg.qk_norm,
    ).to(device)

    if cfg.benchmark.get("mixed_precision", False) and device.type == "cuda":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    logger.info("Generating random data...")
    batch_size = cfg.benchmark.batch_size
    seq_len = model_cfg.context_length
    vocab_size = model_cfg.vocab_size

    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    loss_fn = CrossEntropyLoss().to(device)

    def step_fn() -> None:
        """Performs a single forward (and optionally backward) pass."""
        # Enable autocast for mixed precision on CUDA
        if device.type == "cuda" and cfg.benchmark.get("mixed_precision", False):
            ctx: contextlib.AbstractContextManager = torch.autocast(device_type="cuda", dtype=dtype)
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            logits = model(x)
            if cfg.benchmark.backward:
                loss = loss_fn(logits, y)

        if cfg.benchmark.backward:
            loss.backward()  # type: ignore[reportPossiblyUnboundVariable]

        if device.type == "cuda":
            torch.cuda.synchronize()

    # Warmup
    logger.info(f"Running {cfg.benchmark.warmup_steps} warmup steps...")
    model.train()
    for _ in range(cfg.benchmark.warmup_steps):
        if cfg.benchmark.backward:
            model.zero_grad(set_to_none=True)
        step_fn()

    # Measurement
    logger.info(f"Running {cfg.benchmark.measure_steps} measurement steps...")
    timings = []
    for _ in range(cfg.benchmark.measure_steps):
        if cfg.benchmark.backward:
            model.zero_grad(set_to_none=True)

        start_time = timeit.default_timer()
        step_fn()
        end_time = timeit.default_timer()
        timings.append(end_time - start_time)

    mean_time = statistics.mean(timings)
    stdev_time = statistics.stdev(timings) if len(timings) > 1 else 0.0

    logger.info(f"Forward{' + Backward' if cfg.benchmark.backward else ''} Pass Results:")
    logger.info(f"Mean: {mean_time:.6f} s")
    logger.info(f"Std:  {stdev_time:.6f} s")

    hydra_cfg = HydraConfig.get()
    model_name = hydra_cfg.runtime.choices.get("model", "unknown")

    operation = "Forward + Backward" if cfg.benchmark.backward else "Forward"
    precision = "BF16" if cfg.benchmark.get("mixed_precision", False) and device.type == "cuda" else "FP32"

    results = {
        "Model": model_name,
        "Batch Size": cfg.benchmark.batch_size,
        "Context Length": cfg.model.context_length,
        "Operation": operation,
        "Precision": precision,
        "Mean (s)": mean_time,
        "Std (s)": stdev_time,
    }

    df = pd.DataFrame([results])
    df["Mean (s)"] = df["Mean (s)"].map("{:.6f}".format)
    df["Std (s)"] = df["Std (s)"].map("{:.6f}".format)

    markdown_table = df.to_markdown(index=False)
    print("\n" + markdown_table + "\n")

    output_file = "benchmark_results.md"

    file_exists_and_has_content = os.path.exists(output_file) and os.path.getsize(output_file) > 0

    with open(output_file, "a") as f:
        if not file_exists_and_has_content:
            f.write(markdown_table)
        else:
            # If file already has content, skip the header (first two lines)
            # and append only the data rows, preceded by a newline for separation
            lines = markdown_table.splitlines()
            if len(lines) > 2:  # Ensure there's actual data beyond header/separator
                f.write("\n")
                f.write("\n".join(lines[2:]))

    logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()

import itertools
import logging
import statistics
import time
from dataclasses import dataclass, field

import pandas as pd
import torch

from cs336_basics.layer.attention import scaled_dot_product_attention

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    batch_size: int = 8
    head_dims: list[int] = field(default_factory=lambda: [16, 32, 64, 128])
    seq_lens: list[int] = field(default_factory=lambda: [256, 1024, 4096, 8192, 16384])
    warmup_steps: int = 5
    measure_steps: int = 100
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))


class AttentionBenchmark:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        if self.config.device.type == "cpu":
            logger.warning("CUDA not available. Benchmarking on CPU will be slow and memory stats might be inaccurate.")
        logger.info(f"Device: {self.config.device}")
        logger.info(f"Batch Size: {self.config.batch_size}")

    def _create_tensors(
        self, d_head: int, seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Creates random Q, K, V tensors and a causal mask."""
        q = torch.randn(
            self.config.batch_size,
            seq_len,
            d_head,
            device=self.config.device,
            dtype=torch.float32,
            requires_grad=True,
        )
        k = torch.randn(
            self.config.batch_size,
            seq_len,
            d_head,
            device=self.config.device,
            dtype=torch.float32,
            requires_grad=True,
        )
        v = torch.randn(
            self.config.batch_size,
            seq_len,
            d_head,
            device=self.config.device,
            dtype=torch.float32,
            requires_grad=True,
        )
        mask = torch.tril(torch.ones(seq_len, seq_len, device=self.config.device, dtype=torch.bool)).unsqueeze(0)
        return q, k, v, mask

    def _warmup(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> None:
        """Runs warmup iterations."""
        for _ in range(self.config.warmup_steps):
            out = scaled_dot_product_attention(q, k, v, mask=mask)
            loss = out.sum()
            loss.backward()
            q.grad = None
            k.grad = None
            v.grad = None
        if self.config.device.type == "cuda":
            torch.cuda.synchronize()

    def _measure_forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> float:
        """Measures average forward pass time in seconds."""
        start_fwd_loop = time.perf_counter()
        for _ in range(self.config.measure_steps):
            _ = scaled_dot_product_attention(q, k, v, mask=mask)
            if self.config.device.type == "cuda":
                torch.cuda.synchronize()
        end_fwd_loop = time.perf_counter()
        return (end_fwd_loop - start_fwd_loop) / self.config.measure_steps

    def _measure_memory(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> float:
        """Measures memory usage (MB) before backward pass."""
        # Clear previous gradients/graphs
        q.grad = None
        k.grad = None
        v.grad = None

        if self.config.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # Forward pass to build the graph
        out = scaled_dot_product_attention(q, k, v, mask=mask)
        if self.config.device.type == "cuda":
            torch.cuda.synchronize()

        memory_bytes = 0
        if self.config.device.type == "cuda":
            memory_bytes = torch.cuda.memory_allocated(self.config.device)

        del out
        q.grad = None
        k.grad = None
        v.grad = None
        if self.config.device.type == "cuda":
            torch.cuda.empty_cache()

        return memory_bytes / (1024 * 1024)

    def _measure_backward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> float:
        """Measures average backward pass time in seconds."""
        bwd_times = []
        for _ in range(self.config.measure_steps):
            out = scaled_dot_product_attention(q, k, v, mask=mask)
            loss = out.sum()
            if self.config.device.type == "cuda":
                torch.cuda.synchronize()

            t0 = time.perf_counter()
            loss.backward()
            if self.config.device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            bwd_times.append(t1 - t0)

            q.grad = None
            k.grad = None
            v.grad = None

        return statistics.mean(bwd_times)

    def run(self) -> None:
        logger.info("Starting Attention Benchmark...")
        results = []
        configurations = list(itertools.product(self.config.head_dims, self.config.seq_lens))

        for d_head, seq_len in configurations:
            logger.info(f"Benchmarking: d_head={d_head}, seq_len={seq_len}")
            result = {
                "d_head": d_head,
                "seq_len": seq_len,
                "fwd_time_ms": float("nan"),
                "bwd_time_ms": float("nan"),
                "memory_mb": float("nan"),
            }

            try:
                q, k, v, mask = self._create_tensors(d_head, seq_len)

                self._warmup(q, k, v, mask)

                fwd_time_s = self._measure_forward(q, k, v, mask)
                result["fwd_time_ms"] = fwd_time_s * 1000

                memory_mb = self._measure_memory(q, k, v, mask)
                result["memory_mb"] = memory_mb

                bwd_time_s = self._measure_backward(q, k, v, mask)
                result["bwd_time_ms"] = bwd_time_s * 1000

                # Cleanup
                del q, k, v, mask
                if self.config.device.type == "cuda":
                    torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                logger.error(f"OOM for d_head={d_head}, seq_len={seq_len}")
                if self.config.device.type == "cuda":
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"Error for d_head={d_head}, seq_len={seq_len}: {e}")

            results.append(result)

        self._save_results(results)

    def _save_results(self, results: list[dict]) -> None:
        df = pd.DataFrame(results)

        display_df = df.copy()
        display_df["fwd_time_ms"] = display_df["fwd_time_ms"].map(lambda x: f"{x:.4f}" if pd.notnull(x) else "OOM")
        display_df["bwd_time_ms"] = display_df["bwd_time_ms"].map(lambda x: f"{x:.4f}" if pd.notnull(x) else "OOM")
        display_df["memory_mb"] = display_df["memory_mb"].map(lambda x: f"{x:.2f}" if pd.notnull(x) else "OOM")

        print("\n" + display_df.to_markdown(index=False) + "\n")

        output_file = "attention_benchmark_results.md"
        with open(output_file, "w") as f:
            f.write(display_df.to_markdown(index=False))
        logger.info(f"Results saved to {output_file}")


def main():
    config = BenchmarkConfig()
    benchmark = AttentionBenchmark(config)
    benchmark.run()


if __name__ == "__main__":
    main()

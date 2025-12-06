import itertools
import logging
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import pandas as pd
import torch

from cs336_basics.layer.attention import scaled_dot_product_attention

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Compile the attention function
compiled_scaled_dot_product_attention = torch.compile(scaled_dot_product_attention)


@dataclass
class BenchmarkConfig:
    batch_size: int = 8
    head_dims: list[int] = field(default_factory=lambda: [16, 32, 64, 128])
    seq_lens: list[int] = field(default_factory=lambda: [256, 1024, 4096, 8192, 16384])
    warmup_steps: int = 5
    measure_steps: int = 100
    device: torch.device = field(default_factory=lambda: torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    profile_memory: bool = False


class AttentionBenchmark:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        if self.config.device.type == "cpu":
            logger.warning("CUDA not available. Benchmarking on CPU will be slow and memory stats might be inaccurate.")
        logger.info(f"Device: {self.config.device}")
        logger.info(f"Batch Size: {self.config.batch_size}")

    def _create_tensors(
        self,
        d_head: int,
        seq_len: int,
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

    def _warmup(
        self,
        func: Callable,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        """Runs warmup iterations."""
        for _ in range(self.config.warmup_steps):
            out = func(q, k, v, mask=mask)
            loss = out.sum()
            loss.backward()
            q.grad = None
            k.grad = None
            v.grad = None
        if self.config.device.type == "cuda":
            torch.cuda.synchronize()

    def _measure_forward(
        self,
        func: Callable,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
    ) -> float:
        """Measures average forward pass time in seconds."""
        start_fwd_loop = time.perf_counter()
        for _ in range(self.config.measure_steps):
            _ = func(q, k, v, mask=mask)
            if self.config.device.type == "cuda":
                torch.cuda.synchronize()
        end_fwd_loop = time.perf_counter()
        return (end_fwd_loop - start_fwd_loop) / self.config.measure_steps

    def _measure_memory(
        self,
        func: Callable,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
    ) -> float:
        """Measures memory usage (MB) before backward pass."""
        # Clear previous gradients/graphs
        q.grad = None
        k.grad = None
        v.grad = None

        if self.config.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # Forward pass to build the graph
        out = func(q, k, v, mask=mask)
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

    def _measure_backward(
        self,
        func: Callable,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
    ) -> float:
        """Measures average backward pass time in seconds."""
        bwd_times = []
        for _ in range(self.config.measure_steps):
            out = func(q, k, v, mask=mask)
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

    def _benchmark_single(
        self,
        func: Callable,
        q,
        k,
        v,
        mask,
    ) -> dict:
        """Runs benchmark for a single function variant."""
        res = {"fwd_ms": float("nan"), "bwd_ms": float("nan"), "memory_mb": float("nan")}
        try:
            self._warmup(func, q, k, v, mask)

            # Profile memory if needed (only doing it once per config generally, but here we do it per func)
            if self.config.profile_memory and self.config.device.type == "cuda":
                torch.cuda.memory._record_memory_history(max_entries=1_000_000)

            fwd_s = self._measure_forward(func, q, k, v, mask)
            res["fwd_ms"] = fwd_s * 1000

            mem_mb = self._measure_memory(func, q, k, v, mask)
            res["memory_mb"] = mem_mb

            bwd_s = self._measure_backward(func, q, k, v, mask)
            res["bwd_ms"] = bwd_s * 1000

            if self.config.profile_memory and self.config.device.type == "cuda":
                # We append function name or something to filename if we want distinct files
                # For now, skipping distinct names to avoid complexity in this snippet
                torch.cuda.memory._record_memory_history(enabled=None)

        except torch.cuda.OutOfMemoryError:
            if self.config.device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error benchmarking {func}: {e}")

        return res

    def run(self) -> None:
        logger.info("Starting Attention Benchmark (Uncompiled vs Compiled)...")
        results = []
        configurations = list(itertools.product(self.config.head_dims, self.config.seq_lens))

        for d_head, seq_len in configurations:
            logger.info(f"Benchmarking: d_head={d_head}, seq_len={seq_len}")

            row = {
                "d_head": d_head,
                "seq_len": seq_len,
                "fwd_ms": "OOM",
                "bwd_ms": "OOM",
                "compiled_fwd_ms": "OOM",
                "compiled_bwd_ms": "OOM",
                "memory_mb": "OOM",
            }

            try:
                q, k, v, mask = self._create_tensors(d_head, seq_len)

                # Uncompiled
                unc_res = self._benchmark_single(scaled_dot_product_attention, q, k, v, mask)
                if not pd.isna(unc_res["fwd_ms"]):
                    row["fwd_ms"] = f"{unc_res['fwd_ms']:.4f}"
                    row["bwd_ms"] = f"{unc_res['bwd_ms']:.4f}"
                    row["memory_mb"] = f"{unc_res['memory_mb']:.2f}"

                # Compiled
                # Reset grads and memory before compiled run
                q.grad = None
                k.grad = None
                v.grad = None
                if self.config.device.type == "cuda":
                    torch.cuda.empty_cache()

                com_res = self._benchmark_single(compiled_scaled_dot_product_attention, q, k, v, mask)
                if not pd.isna(com_res["fwd_ms"]):
                    row["compiled_fwd_ms"] = f"{com_res['fwd_ms']:.4f}"
                    row["compiled_bwd_ms"] = f"{com_res['bwd_ms']:.4f}"

                # Cleanup
                del q, k, v, mask
                if self.config.device.type == "cuda":
                    torch.cuda.empty_cache()

            except torch.cuda.OutOfMemoryError:
                logger.error(f"OOM during tensor creation for d_head={d_head}, seq_len={seq_len}")
                if self.config.device.type == "cuda":
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.error(f"Error for d_head={d_head}, seq_len={seq_len}: {e}")

            results.append(row)

        self._save_results(results)

    def _save_results(self, results: list[dict]) -> None:
        df = pd.DataFrame(results)
        print("\n" + df.to_markdown(index=False) + "\n")

        output_file = "attention_benchmark_results.md"
        with open(output_file, "w") as f:
            f.write(df.to_markdown(index=False))
        logger.info(f"Results saved to {output_file}")


def main():
    config = BenchmarkConfig()
    benchmark = AttentionBenchmark(config)
    benchmark.run()


if __name__ == "__main__":
    main()

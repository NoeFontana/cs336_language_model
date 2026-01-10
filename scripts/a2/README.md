# Assignment 2 Benchmarks

This directory contains benchmarking scripts for various components of the language model and the full system.

## Scripts

### 1. `benchmark.py`
Benchmarks the full `TransformerLM` model training step (forward, backward, optimizer).
- **Configuration:** Uses Hydra (conf/config.yaml).
- **Features:** Supports profiling memory (`profile_memory=True`), mixed precision, and different model sizes.
- **Usage:**
  ```bash
  PYTHONPATH=src uv run scripts/a2/benchmark.py benchmark.device=cuda benchmark.batch_size=4
  ```

### 2. `benchmark_attention.py`
Benchmarks the `scaled_dot_product_attention` function, comparing the uncompiled version against `torch.compile`.
- **Metrics:** Forward pass latency, Backward pass latency, Memory usage.
- **Sweeps:** Over various sequence lengths and head dimensions.
- **Usage:**
  ```bash
  PYTHONPATH=src uv run scripts/a2/benchmark_attention.py
  ```

### 3. `benchmark_flash_attention.py`
Benchmarks the Triton Flash Attention implementation against a naive PyTorch baseline using `triton.testing.perf_report`.
- **Metrics:** Latency (ms) for Forward, Backward, and End-to-End.
- **Sweeps:** Sequence lengths (128 to 65536) and head dimensions (16, 32, 64, 128).
- **Usage:**
  ```bash
  PYTHONPATH=src uv run scripts/a2/benchmark_flash_attention.py
  ```
  *Note: Ensure your Triton kernel is imported in place of `flash_attention_stub` for accurate results.*

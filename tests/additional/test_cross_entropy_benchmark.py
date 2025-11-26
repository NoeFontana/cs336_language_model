import pytest
import torch
from pytest_benchmark.fixture import BenchmarkFixture

from cs336.loss.cross_entropy import CrossEntropyLoss, cross_entropy

# OWT parameters (expected size)
OWT_BATCH_SIZE: int = 64
OWT_CONTEXT_LENGTH: int = 512
OWT_VOCAB_SIZE: int = 32000
OWT_TOTAL_TOKENS: int = OWT_BATCH_SIZE * OWT_CONTEXT_LENGTH

# TinyStories parameters (smaller for GPU tests to avoid OOM)
TS_BATCH_SIZE: int = 64
TS_CONTEXT_LENGTH: int = 256
TS_VOCAB_SIZE: int = 10000
TS_TOTAL_TOKENS: int = TS_BATCH_SIZE * TS_CONTEXT_LENGTH


@pytest.fixture(scope="module")
def input_data_owt() -> tuple[torch.Tensor, torch.Tensor]:
    """Generates inputs of OWT size for CPU."""
    torch.manual_seed(42)
    logits = torch.randn(OWT_TOTAL_TOKENS, OWT_VOCAB_SIZE)
    labels = torch.randint(0, OWT_VOCAB_SIZE, (OWT_TOTAL_TOKENS,))
    return logits, labels


@pytest.fixture(scope="module")
def input_data_ts() -> tuple[torch.Tensor, torch.Tensor]:
    """Generates inputs of TinyStories size for CPU."""
    torch.manual_seed(42)
    logits = torch.randn(TS_TOTAL_TOKENS, TS_VOCAB_SIZE)
    labels = torch.randint(0, TS_VOCAB_SIZE, (TS_TOTAL_TOKENS,))
    return logits, labels


@pytest.mark.slow
def test_cross_entropy_cpu_benchmark_owt(
    benchmark: BenchmarkFixture, input_data_owt: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Benchmarks cross_entropy on CPU with OWT sizes."""
    logits, labels = input_data_owt
    benchmark(cross_entropy, logits, labels)


@pytest.mark.slow
def test_cross_entropy_compiled_cpu_benchmark_owt(
    benchmark: BenchmarkFixture, input_data_owt: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Benchmarks compiled CrossEntropyLoss module on CPU with OWT sizes."""
    logits, labels = input_data_owt
    model = CrossEntropyLoss()
    compiled_model = torch.compile(model)
    # Warmup to ensure compilation happens before benchmark
    compiled_model(logits, labels)
    benchmark(compiled_model, logits, labels)


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cross_entropy_gpu_benchmark_ts(
    benchmark: BenchmarkFixture, input_data_ts: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Benchmarks cross_entropy on GPU with TinyStories sizes."""
    logits_cpu, labels_cpu = input_data_ts
    # Move to GPU once
    logits = logits_cpu.to("cuda")
    labels = labels_cpu.to("cuda")

    def run_gpu() -> None:
        cross_entropy(logits, labels)
        torch.cuda.synchronize()

    benchmark(run_gpu)


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cross_entropy_compiled_gpu_benchmark_ts(
    benchmark: BenchmarkFixture, input_data_ts: tuple[torch.Tensor, torch.Tensor]
) -> None:
    """Benchmarks compiled CrossEntropyLoss module on GPU with TinyStories sizes."""
    logits_cpu, labels_cpu = input_data_ts
    # Move to GPU once
    logits = logits_cpu.to("cuda")
    labels = labels_cpu.to("cuda")

    model = CrossEntropyLoss()
    compiled_model = torch.compile(model)
    # Warmup to ensure compilation happens before benchmark
    compiled_model(logits, labels)
    torch.cuda.synchronize()

    def run_gpu() -> None:
        compiled_model(logits, labels)
        torch.cuda.synchronize()

    benchmark(run_gpu)

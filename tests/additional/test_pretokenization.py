from pathlib import Path

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from cs336.pretokenization import chunked_pretokenization


@pytest.mark.slow
@pytest.mark.parametrize("dataset", ["~/datasets/cs336/TinyStoriesV2-GPT4-train.txt", "~/datasets/cs336/owt_train.txt"])
def test_chunked_pretokenization_benchmark(benchmark: BenchmarkFixture, dataset: str) -> None:
    """
    Benchmarks the entire chunked_pretokenization pipeline on the TinyStories validation set
    with a varying number of chunks.
    """
    corpus_path = Path(dataset).expanduser()

    if not corpus_path.exists():
        pytest.skip(f"Dataset not found at {corpus_path}")

    special_tokens = ["<|endoftext|>"]

    result = benchmark(
        chunked_pretokenization,
        corpus_path=corpus_path,
        special_tokens=special_tokens,
        num_chunks=10,
    )

    assert result is not None, "chunked_pretokenization returned None"

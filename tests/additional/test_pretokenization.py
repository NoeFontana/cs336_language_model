import os
from pathlib import Path

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from cs336.pretokenization import chunked_pretokenization, pretokenization, split_on_special_tokens


@pytest.fixture(scope="module")
def corpus_data() -> list[str]:
    """
    Loads and splits the validation dataset for pretokenization tests.
    """
    file_path_str = os.path.expanduser("~/datasets/cs336/TinyStoriesV2-GPT4-valid.txt")
    file_path = Path(file_path_str)

    if not file_path.exists():
        pytest.skip(f"Dataset not found at {file_path_str}")

    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        data = f.read()

    # Using default special tokens from train_bpe.py
    special_tokens = ["<|endoftext|>"]
    return split_on_special_tokens(corpus=data, special_tokens=special_tokens)


def test_pretokenization_benchmark(benchmark: BenchmarkFixture, corpus_data: list[str]) -> None:
    """
    Benchmarks the pretokenization function on the TinyStories validation set.
    """
    if not corpus_data:
        pytest.skip("Corpus data is empty, skipping benchmark.")

    # Benchmark the pretokenization function
    result = benchmark(pretokenization, corpus_data)
    assert result is not None, "Pretokenization returned None"


@pytest.mark.slow
@pytest.mark.parametrize(
    "dataset", ["~/datasets/cs336/TinyStoriesV2-GPT4-valid.txt", "~/datasets/cs336/TinyStoriesV2-GPT4-train.txt"]
)
@pytest.mark.parametrize("num_chunks", [6, 10])
def test_chunked_pretokenization_benchmark(benchmark: BenchmarkFixture, dataset: str, num_chunks: int) -> None:
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
        num_chunks=num_chunks,
    )

    assert result is not None, "chunked_pretokenization returned None"

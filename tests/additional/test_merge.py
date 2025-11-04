import importlib.util
import logging
import random
import string
from typing import Any

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from cs336.merge import merge
from cs336.merge.merge_py import merge as py_merge

NATIVE_EXTENSIONS_AVAILABLE = importlib.util.find_spec("cs336_native") is not None


def test_merge_tie_breaking(initial_vocab_fixture: dict[int, bytes]) -> None:
    """Tests that merge breaks ties by picking the lexicographically larger pair."""
    # The pair (b'c', b'd') from b"cd" is lexicographically greater than (b'a', b'b') from b"ab".
    pretokens = {b"ab": 1, b"cd": 1}
    initial_vocab = initial_vocab_fixture
    max_vocab_size = len(initial_vocab) + 1  # Only allow one merge

    vocab, merges = merge(pretokens, initial_vocab, max_vocab_size)

    # The first and only merge should be the lexicographically larger pair.
    expected_merges = [(b"c", b"d")]
    expected_vocab = initial_vocab.copy()
    expected_vocab[len(initial_vocab)] = b"cd"

    assert merges == expected_merges
    assert vocab == expected_vocab


def test_merge_max_vocab_size_limit(initial_vocab_fixture: dict[int, bytes]) -> None:
    """Tests that merge respects the max_vocab_size limit."""
    pretokens = {b"abc": 2}
    initial_vocab = initial_vocab_fixture
    max_vocab_size = len(initial_vocab) + 1  # Only allow one merge

    vocab, merges = merge(pretokens, initial_vocab, max_vocab_size)

    # With tie-breaking, (b, c) is merged first.
    expected_merges = [(b"b", b"c")]
    expected_vocab = initial_vocab.copy()
    expected_vocab[len(initial_vocab)] = b"bc"

    assert merges == expected_merges
    assert vocab == expected_vocab


def test_merge_no_merges_possible(initial_vocab_fixture: dict[int, bytes], caplog: pytest.LogCaptureFixture) -> None:
    """Tests merge when no pairs can be formed."""
    pretokens: dict[bytes, int] = {b"a": 5, b"b": 3}
    initial_vocab = initial_vocab_fixture
    max_vocab_size = len(initial_vocab) + 5

    with caplog.at_level(logging.ERROR):  # Hide the expected warning
        vocab, merges = merge(pretokens, initial_vocab, max_vocab_size)

    assert merges == []
    assert vocab == initial_vocab


def test_merge_empty_pretokens(initial_vocab_fixture: dict[int, bytes]) -> None:
    """Tests merge with an empty pretokens dictionary."""
    pretokens: dict[bytes, int] = {}
    initial_vocab = initial_vocab_fixture
    max_vocab_size = len(initial_vocab) + 5

    vocab, merges = merge(pretokens, initial_vocab, max_vocab_size)

    assert merges == []
    assert vocab == initial_vocab


def test_merge_complex_replacement(initial_vocab_fixture: dict[int, bytes]) -> None:
    """Tests that merging happens correctly multiple times in one pretoken."""
    pretokens = {b"ababc": 1}
    initial_vocab = initial_vocab_fixture
    max_vocab_size = len(initial_vocab) + 2

    vocab, merges = merge(pretokens, initial_vocab, max_vocab_size)

    # Expected merges:
    # 1. (a, b) is most frequent (count 2). Merge (a, b).
    #    pretokens becomes {(b'ab', b'ab', b'c'): 1}
    # 2. (ab, ab) and (ab, c) are tied with count 1. (ab, c) is lexicographically larger. Merge (ab, c).
    #    pretokens becomes {(b'ab', b'abc'): 1}
    # Loop terminates as vocab size is 5.
    expected_merges = [(b"a", b"b"), (b"ab", b"c")]
    expected_vocab = initial_vocab.copy()
    expected_vocab[len(initial_vocab)] = b"ab"
    expected_vocab[len(initial_vocab) + 1] = b"abc"

    assert merges == expected_merges
    assert vocab == expected_vocab


@pytest.mark.skipif(not NATIVE_EXTENSIONS_AVAILABLE, reason="Rust extension not available")
class TestMergeBenchmark:
    """Groups benchmark tests for merge implementations to compare them."""

    # Class attributes to store results between test runs
    py_result: Any = None
    py_stats: Any = None

    rust_result: Any = None
    rust_stats: Any = None

    @pytest.fixture(scope="function")
    def benchmark_data(self, initial_vocab_fixture: dict[int, bytes]) -> dict[str, Any]:
        """Generate a shared, complex dataset for benchmarking."""
        num_unique_words = 10_000
        repetitions = 5
        random.seed(42)

        words = [
            "".join(random.choices(string.ascii_lowercase, k=random.randint(1, 15))) for _ in range(num_unique_words)
        ]
        pretoken_generator = (word.encode("utf-8") for word in words)

        return {
            "pretokens": dict.fromkeys(pretoken_generator, repetitions),
            "initial_vocab": initial_vocab_fixture,
            "max_vocab_size": 1000,
        }

    @pytest.mark.slow(group="merge")
    def test_py_merge_benchmark(self, benchmark: BenchmarkFixture, benchmark_data: dict[str, Any]) -> None:
        """Benchmarks the pure Python merge implementation."""

        def run_py_merge():
            # The copy is done inside the lambda to ensure each run is independent.
            return py_merge(
                benchmark_data["pretokens"].copy(),
                benchmark_data["initial_vocab"].copy(),
                benchmark_data["max_vocab_size"],
            )

        # Use pedantic to control rounds and get the result for the correctness check.
        result = benchmark.pedantic(
            run_py_merge,
            rounds=3,
            iterations=1,
        )
        TestMergeBenchmark.py_result = result
        TestMergeBenchmark.py_stats = benchmark.stats

    @pytest.mark.slow(group="merge")
    def test_rust_merge_benchmark(self, benchmark: BenchmarkFixture, benchmark_data: dict[str, Any]) -> None:
        """Benchmarks the Rust merge implementation and compares against the Python version."""
        from cs336_native import merge as rust_merge  # type: ignore

        def run_rust_merge():
            return rust_merge(
                benchmark_data["pretokens"].copy(),
                benchmark_data["initial_vocab"].copy(),
                benchmark_data["max_vocab_size"],
            )

        rust_result = benchmark.pedantic(
            run_rust_merge,
            rounds=3,
            iterations=1,
        )
        TestMergeBenchmark.rust_result = rust_result
        TestMergeBenchmark.rust_stats = benchmark.stats

    @pytest.mark.slow(group="merge")
    def test_merge_correctness_and_speedup(self, benchmark: BenchmarkFixture) -> None:
        # 1. Validate that outputs are the same

        assert TestMergeBenchmark.rust_result is not None, "Rust benchmark must run first"
        assert TestMergeBenchmark.py_result is not None, "Python benchmark must run first"

        assert TestMergeBenchmark.rust_result == TestMergeBenchmark.py_result, (
            "Rust and Python implementations produced different results"
        )

        # 2. Compare performance
        py_mean = TestMergeBenchmark.py_stats.get("mean", float("inf"))
        rust_mean = TestMergeBenchmark.rust_stats.get("mean", 0.0)

        speedup = py_mean / rust_mean

        assert 2 < speedup, f"Rust version is not 2x faster. Speedup: {speedup:.2f}x"

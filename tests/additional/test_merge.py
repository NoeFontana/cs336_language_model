import logging
import random
import string
from typing import Any, cast

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

# TODO: Cleanup-imports
from cs336.merge import merge
from cs336.merge.merge_py import merge as py_merge

try:
    from cs336_native import merge as rust_merge
except ImportError:
    rust_merge = None


def test_merge_tie_breaking(initial_vocab_fixture: dict[int, bytes]) -> None:
    """Tests that merge breaks ties by picking the lexicographically larger pair."""
    # Both (b'a', b'b') and (b'c', b'd') appear once.
    # (b'c', b'd') is lexicographically greater than (b'a', b'b').
    pretokens = {(b"a", b"b"): 1, (b"c", b"d"): 1}
    initial_vocab = initial_vocab_fixture
    max_vocab_size = len(initial_vocab) + 1  # Only allow one merge

    vocab, merges = merge(pretokens, initial_vocab, max_vocab_size)

    # The first and only merge should be the lexicographically larger pair.
    expected_merges = [(b"c", b"d")]
    expected_vocab = initial_vocab.copy()
    expected_vocab[len(initial_vocab)] = b"cd"

    assert merges == expected_merges
    assert vocab == expected_vocab


def test_merge_multibyte(initial_vocab_fixture: dict[int, bytes]) -> None:
    """Tests the merge process with multi-byte UTF-8 characters."""
    # Pre-token for "αβγ"
    pretoken = (b"\xce\xb1", b"\xce\xb2", b"\xce\xb3")
    pretokens = {pretoken: 1}
    initial_vocab = initial_vocab_fixture
    max_vocab_size = len(initial_vocab) + 2

    vocab, merges = merge(pretokens, initial_vocab, max_vocab_size)

    # 1. Pairs are (α, β) and (β, γ). Both appear once.
    #    (β, γ) is lexicographically larger, so it's merged first.
    #    pretokens becomes {(b'α', b'βγ'): 1}
    # 2. The only remaining pair is (α, βγ), which is merged next.
    expected_merges = [(b"\xce\xb2", b"\xce\xb3"), (b"\xce\xb1", b"\xce\xb2\xce\xb3")]
    expected_vocab = initial_vocab.copy()
    expected_vocab[len(initial_vocab)] = b"\xce\xb2\xce\xb3"
    expected_vocab[len(initial_vocab) + 1] = b"\xce\xb1\xce\xb2\xce\xb3"

    assert merges == expected_merges
    assert vocab == expected_vocab


def test_merge_max_vocab_size_limit(initial_vocab_fixture: dict[int, bytes]) -> None:
    """Tests that merge respects the max_vocab_size limit."""
    pretokens = {(b"a", b"b", b"c"): 2}
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
    pretokens: dict[tuple[bytes, ...], int] = {(b"a",): 5, (b"b",): 3}
    initial_vocab = initial_vocab_fixture
    max_vocab_size = len(initial_vocab) + 5

    with caplog.at_level(logging.ERROR):  # Hide the expected warning
        vocab, merges = merge(pretokens, initial_vocab, max_vocab_size)

    assert merges == []
    assert vocab == initial_vocab


def test_merge_empty_pretokens(initial_vocab_fixture: dict[int, bytes]) -> None:
    """Tests merge with an empty pretokens dictionary."""
    pretokens: dict[tuple[bytes, ...], int] = {}
    initial_vocab = initial_vocab_fixture
    max_vocab_size = len(initial_vocab) + 5

    vocab, merges = merge(pretokens, initial_vocab, max_vocab_size)

    assert merges == []
    assert vocab == initial_vocab


def test_merge_complex_replacement(initial_vocab_fixture: dict[int, bytes]) -> None:
    """Tests that merging happens correctly multiple times in one pretoken."""
    pretokens = {(b"a", b"b", b"a", b"b", b"c"): 1}
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


@pytest.mark.skipif(rust_merge is None, reason="Rust extension not available")
class TestMergeBenchmark:
    """Groups benchmark tests for merge implementations to compare them."""

    # Class attributes to store results between test runs
    py_result: Any = None
    py_stats: Any = None

    @pytest.fixture(scope="function")
    def benchmark_data(self, initial_vocab_fixture: dict[int, bytes]) -> dict[str, Any]:
        """Generate a shared, complex dataset for benchmarking."""
        random.seed(42)
        words = ["".join(random.choices(string.ascii_lowercase, k=random.randint(1, 15))) for _ in range(10_000)]
        text = " ".join(words) * 5
        pretokens: dict[tuple[bytes, ...], int] = {}
        for word in text.split():
            pretoken = tuple(c.encode("utf-8") for c in word)
            pretokens[pretoken] = pretokens.get(pretoken, 0) + 1

        return {
            "pretokens": pretokens,
            "initial_vocab": initial_vocab_fixture,
            "max_vocab_size": 1000 - len(initial_vocab_fixture),
        }

    @pytest.mark.dependency()
    @pytest.mark.benchmark(group="merge", warmup=True)
    def test_py_merge_benchmark(self, benchmark: BenchmarkFixture, benchmark_data: dict[str, Any]) -> None:
        """Benchmarks the pure Python merge implementation."""

        # Run once to get the result for the correctness check in the next test.
        TestMergeBenchmark.py_result = py_merge(
            benchmark_data["pretokens"].copy(),
            benchmark_data["initial_vocab"].copy(),
            benchmark_data["max_vocab_size"],
        )

        # Benchmark the function for performance comparison.
        benchmark(
            py_merge, benchmark_data["pretokens"], benchmark_data["initial_vocab"], benchmark_data["max_vocab_size"]
        )
        TestMergeBenchmark.py_stats = benchmark.stats

    @pytest.mark.dependency(depends=["TestMergeBenchmark::test_py_merge_benchmark"])
    @pytest.mark.benchmark(group="merge", warmup=True)
    def test_rust_merge_benchmark(self, benchmark: BenchmarkFixture, benchmark_data: dict[str, Any]) -> None:
        """Benchmarks the Rust merge implementation and compares against the Python version."""

        def run_rust_merge():
            return cast(Any, rust_merge)(
                benchmark_data["pretokens"], benchmark_data["initial_vocab"].copy(), benchmark_data["max_vocab_size"]
            )

        rust_result = benchmark.pedantic(run_rust_merge, rounds=10, iterations=5)

        # 1. Validate that outputs are the same
        assert TestMergeBenchmark.py_result is not None, "Python benchmark must run first"
        assert rust_result == TestMergeBenchmark.py_result, "Rust and Python implementations produced different results"

        # 2. Compare performance
        py_mean = TestMergeBenchmark.py_stats.get("mean", float("inf"))
        rust_mean = benchmark.stats.get("mean", 0.0)

        speedup = py_mean / rust_mean

        assert 2 < speedup, f"Rust version is not 2x faster. Speedup: {speedup:.2f}x"

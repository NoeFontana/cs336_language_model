#!/usr/bin/env python

import argparse
import logging
import random
import time
from pathlib import Path

from cs336.tokenizer import Tokenizer


def sample_documents(filepath: Path, num_samples: int, separator: str = "<|endoftext|>") -> list[str]:
    """
    Samples a specified number of documents from a text file.

    Args:
        filepath (Path): The path to the text file.
        num_samples (int): The number of documents to sample.
        separator (str): The string that separates documents.

    Returns:
        list[str]: A list of sampled documents.
    """
    logging.info(f"Sampling {num_samples} documents from {filepath}...")
    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    documents = content.split(separator)
    # Filter out any empty strings that might result from splitting
    documents = [doc for doc in documents if doc.strip()]

    # Ensure we don't try to sample more documents than available
    num_to_sample = min(num_samples, len(documents))

    return random.sample(documents, num_to_sample)


def calculate_compression_ratio(documents: list[str], tokenizer: Tokenizer) -> float:
    """
    Calculates the compression ratio (bytes per token) for a list of documents.

    Args:
        documents (list[str]): A list of strings to encode.
        tokenizer (Tokenizer): The tokenizer to use for encoding.

    Returns:
        float: The average number of bytes per token.
    """
    total_bytes = 0
    total_tokens = 0

    for doc in documents:
        encoded_bytes = doc.encode("utf-8")
        token_ids = tokenizer.encode(doc)

        total_bytes += len(encoded_bytes)
        total_tokens += len(token_ids)

    if total_tokens == 0:
        return 0.0

    return total_bytes / total_tokens


def benchmark_throughput(dataset_path: Path, tokenizer: Tokenizer) -> float:
    """
    Benchmarks the tokenizer's encoding throughput on a given dataset.

    Args:
        dataset_path (Path): Path to the dataset file.
        tokenizer (Tokenizer): The tokenizer to benchmark.

    Returns:
        float: Throughput in megabytes per second (MB/s).
    """
    logging.info(f"Benchmarking throughput on {dataset_path}...")
    file_size_bytes = dataset_path.stat().st_size
    logging.info(f"Dataset size: {file_size_bytes / 1e6:.2f} MB")

    with open(dataset_path, encoding="utf-8") as f:
        content = f.read()

    start_time = time.monotonic()
    tokenizer.encode(content)
    end_time = time.monotonic()

    duration = end_time - start_time
    throughput_bytes_per_sec = file_size_bytes / duration
    throughput_mb_per_sec = throughput_bytes_per_sec / 1e6

    logging.info(f"Encoding took {duration:.2f} seconds.")
    logging.info(f"Throughput: {throughput_mb_per_sec:.2f} MB/s")

    return throughput_mb_per_sec


def main():
    """
    Runs the tokenizer comparison experiments.
    """
    # --- Configuration ---
    parser = argparse.ArgumentParser(description="Run tokenizer comparison experiments.")
    parser.add_argument(
        "--tinys-tokenizer-path",
        type=Path,
        default="results/tiny-stories-train.json",
        help="Path to TinyStories tokenizer.",
    )
    parser.add_argument(
        "--owt-tokenizer-path", type=Path, default="results/owt.json", help="Path to OpenWebText tokenizer."
    )
    parser.add_argument(
        "--tinys-data-path",
        type=Path,
        default=Path("~/datasets/cs336/TinyStoriesV2-GPT4-valid.txt").expanduser(),
        help="Path to TinyStories validation data.",
    )
    parser.add_argument(
        "--owt-data-path",
        type=Path,
        default=Path("~/datasets/cs336/owt_valid.txt").expanduser(),
        help="Path to OpenWebText validation data.",
    )
    parser.add_argument("--num-samples", type=int, default=10, help="Number of documents to sample from each dataset.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # --- File validation ---
    for path in [args.tinys_tokenizer_path, args.owt_tokenizer_path, args.tinys_data_path, args.owt_data_path]:
        if not path.exists():
            logging.error(f"File not found: {path}")
            return

    # --- Load Tokenizers ---
    logging.info("Loading tokenizers...")
    tinys_tokenizer = Tokenizer.from_files(str(args.tinys_tokenizer_path), None)
    owt_tokenizer = Tokenizer.from_files(str(args.owt_tokenizer_path), None)

    # --- Sample Data ---
    tinys_samples = sample_documents(args.tinys_data_path, args.num_samples)
    owt_samples = sample_documents(args.owt_data_path, args.num_samples)

    # --- Experiment (a) ---
    logging.info("--- Running Experiment (a): In-domain tokenization ---")
    tinys_ratio = calculate_compression_ratio(tinys_samples, tinys_tokenizer)
    logging.info(f"TinyStories tokenizer on TinyStories data: {tinys_ratio:.2f} bytes/token")

    owt_ratio = calculate_compression_ratio(owt_samples, owt_tokenizer)
    logging.info(f"OpenWebText tokenizer on OpenWebText data: {owt_ratio:.2f} bytes/token")

    # --- Experiment (b) ---
    logging.info("--- Running Experiment (b): Out-of-domain tokenization ---")
    owt_on_tinys_tokenizer_ratio = calculate_compression_ratio(owt_samples, tinys_tokenizer)
    logging.info(f"TinyStories tokenizer on OpenWebText data: {owt_on_tinys_tokenizer_ratio:.2f} bytes/token")

    # --- Throughput Estimation ---
    logging.info("\n--- Estimating Tokenizer Throughput ---")
    # We use the TinyStories tokenizer and validation set for a quick but realistic benchmark.
    throughput_mb_s = benchmark_throughput(args.owt_data_path, owt_tokenizer)

    if throughput_mb_s > 0:
        pile_size_gb = 825
        pile_size_mb = pile_size_gb * 1024
        time_to_tokenize_seconds = pile_size_mb / throughput_mb_s
        time_to_tokenize_hours = time_to_tokenize_seconds / 3600
        time_to_tokenize_days = time_to_tokenize_hours / 24

        logging.info(f"Estimated time to tokenize the Pile ({pile_size_gb} GB): {time_to_tokenize_days:.2f} days")


if __name__ == "__main__":
    main()

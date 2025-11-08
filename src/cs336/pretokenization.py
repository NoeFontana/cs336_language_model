import logging
import mmap
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial, reduce
from pathlib import Path
from typing import BinaryIO

import regex as re


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """Finds byte offsets in a file to divide it into chunks for parallel processing.

    This function divides a file into a desired number of chunks. To avoid splitting
    in the middle of a multi-byte character or a logical unit of text, it adjusts
    the chunk boundaries to align with the start of the next occurrence of a
    specified special token.

    Args:
        file: The binary file object to be chunked.
        desired_num_chunks: The target number of chunks to divide the file into.
        split_special_token: The byte string of a special token (e.g., b"<|endoftext|>")
            that should not be split. Boundaries are moved to align with this token.

    Returns:
        A sorted list of unique byte offsets representing the chunk boundaries. The
        number of chunks created (`len(list) - 1`) may be less than desired if
        token occurrences are sparse, leading to merged chunks.
    """

    fileno = file.fileno()
    file_size = os.fstat(fileno).st_size

    if file_size == 0:
        return [0]

    chunk_size = file_size // desired_num_chunks

    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    with mmap.mmap(fileno, 0, access=mmap.ACCESS_READ) as mm:
        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]

            found_at = mm.find(split_special_token, initial_position)

            if found_at != -1:
                chunk_boundaries[bi] = found_at
            else:
                # No token found from here to the end of the file.
                final_list = chunk_boundaries[:bi] + [file_size]
                return sorted(set(final_list))

    # Make sure all boundaries are unique and sorted
    return sorted(set(chunk_boundaries))


def _batch_count_from_segment(
    segment: memoryview, compiled_pretoken_pattern: re.Pattern, batch_size: int = 50_000
) -> Counter[bytes]:
    """
    Efficiently counts pretokens from a memoryview segment in batches.
    """
    if not segment:
        return Counter()

    occurences: Counter[bytes] = Counter()
    batch = []

    scanner = compiled_pretoken_pattern.finditer(string=segment, concurrent=False)

    for match_g in scanner:
        batch.append(match_g.group())

        if len(batch) >= batch_size:
            occurences.update(batch)
            batch.clear()

    if batch:
        occurences.update(batch)

    return occurences


def process_chunk(
    chunk_range: tuple[int, int],
    file_path: str,
    special_tokens_pattern: re.Pattern,
    compiled_pretoken_pattern: re.Pattern,
) -> Counter[bytes]:
    """Processes a single file chunk for parallel pre-tokenization.

    This worker function is designed to be called by a `ProcessPoolExecutor`. It
    reads a specific byte range (a chunk) from a file, decodes it, splits it on
    special tokens, and performs pre-tokenization to generate token counts.

    Args:
        chunk_range: A tuple `(start, end)` indicating the byte offsets for the chunk.
        file_path: The path to the file from which the chunk will be read.
        special_tokens_pattern: The regex byte pattern for splitting on special tokens.
        compiled_pretoken_pattern: The compiled regex pattern for pre-tokenization.

    Returns:
        A dictionary of pre-token counts for the processed chunk.
    """
    start, end = chunk_range
    chunk_size = end - start

    if chunk_size == 0:
        return Counter()

    pretokens: Counter[bytes] = Counter()
    with open(file_path, "rb") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        mm.madvise(mmap.MADV_SEQUENTIAL)
        chunk_data = memoryview(mm)[start:end]

        last_end = 0
        for special_match in re.finditer(special_tokens_pattern, chunk_data, concurrent=False):
            corpus_segment = chunk_data[last_end : special_match.start()]

            pretokens.update(_batch_count_from_segment(corpus_segment, compiled_pretoken_pattern, 100_000))
            last_end = special_match.end()
            del special_match
            del corpus_segment

        corpus_segment = chunk_data[last_end:]
        if corpus_segment:
            pretokens.update(_batch_count_from_segment(corpus_segment, compiled_pretoken_pattern, 100_000))

        del corpus_segment
        del chunk_data
    return pretokens


def chunked_pretokenization(corpus_path: Path, special_tokens: list[str], num_chunks: int) -> Counter[bytes]:
    """Performs pre-tokenization on a large file in parallel.

    This function orchestrates the pre-tokenization of a large corpus by first
    dividing the file into chunks, then processing each chunk in a separate
    process, and finally aggregating the results into a single count dictionary.

    Args:
        corpus_path: The path to the text file to be tokenized.
        special_tokens: A list of special tokens to be handled during splitting.
        num_chunks: The desired number of chunks to split the file into for
            parallel processing.

    Returns:
        A dictionary mapping each unique pre-token to its total frequency count
        across the entire corpus.
    """
    with Path(corpus_path).open("rb") as f:
        chunk_boundaries = find_chunk_boundaries(f, desired_num_chunks=num_chunks, split_special_token=b"<|endoftext|>")

    special_tokens_pattern = re.compile(
        b"|".join(re.escape(tok) for tok in [tok.encode("utf-8") for tok in special_tokens]), flags=re.V1
    )

    # Compile the pre-tokenization pattern once in the main process.
    pretoken_pattern_str = rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    compiled_pretoken_pattern = re.compile(pretoken_pattern_str, flags=re.V1)

    # Recompute num_chunks in case, find_chunk_boundaries returned less chunks than desired.
    # This may happen when some boundaries are merged due to low density of tokens.
    num_chunks = len(chunk_boundaries) - 1

    logging.getLogger(__name__).info(f"Chunk boundaries: {chunk_boundaries}")

    with ProcessPoolExecutor(max_workers=num_chunks) as executor:
        worker_func = partial(
            process_chunk,
            file_path=corpus_path.as_posix(),
            special_tokens_pattern=special_tokens_pattern,
            compiled_pretoken_pattern=compiled_pretoken_pattern,
        )
        chunk_ranges = [(chunk_boundaries[i], chunk_boundaries[i + 1]) for i in range(num_chunks)]
        futures = [executor.submit(worker_func, chunk_range) for chunk_range in chunk_ranges]

        # Use reduce with Counter addition for a concise and efficient aggregation.
        initial_counter: Counter[bytes] = Counter()
        return reduce(lambda acc, future: acc + future.result(), as_completed(futures), initial_counter)

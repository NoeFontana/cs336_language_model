import logging
import mmap
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial, reduce
from pathlib import Path
from typing import BinaryIO, Final

import regex as re

BYTE_CACHE: Final = [i.to_bytes(1) for i in range(256)]


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


def split_on_special_tokens(corpus: str, special_tokens: list[str]) -> list[str]:
    """Splits a text corpus by a list of special tokens.

    The special tokens act as delimiters and are removed from the output. This is
    used to isolate segments of text that do not contain any special tokens.

    Args:
        corpus: The input string to be split.
        special_tokens: A list of strings (e.g., ["<|endoftext|>"]) to use as
            delimiters for splitting the corpus.

    Returns:
        A list of substrings. Empty strings resulting from the split are excluded.
    """
    pattern = r"|".join(re.escape(tok) for tok in special_tokens)
    split_corpus = re.split(pattern, corpus, concurrent=False)
    return [s for s in split_corpus if s]


def pretokenization(
    split_corpus: list[str],
    pattern: str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
) -> Counter[bytes]:
    """Performs pre-tokenization on text segments and counts token occurrences.

    This function applies a regex pattern to break down text into "pre-tokens",
    which are the smallest units before BPE merging begins. It then counts the
    frequency of each unique pre-token sequence.

    Args:
        split_corpus: A list of strings, typically the output of `split_on_special_tokens`.
        pattern: The regex pattern used to find pre-tokens. Defaults to the GPT-2 pattern.

    Returns:
        A dictionary mapping each unique pre-token (represented as a tuple of its
        constituent bytes) to its frequency count across the corpus.
    """

    compiled_pattern = re.compile(pattern.encode("utf-8"))

    occurences: Counter[bytes] = Counter()
    for corpus in split_corpus:
        corpus_bytes = corpus.encode("utf-8")
        scanner = compiled_pattern.finditer(string=corpus_bytes, concurrent=False)
        occurences.update(match_g.group() for match_g in scanner)
    return occurences


def process_chunk(chunk_range: tuple[int, int], file_path: str, special_tokens: list[str]) -> Counter[bytes]:
    """Processes a single file chunk for parallel pre-tokenization.

    This worker function is designed to be called by a `ProcessPoolExecutor`. It
    reads a specific byte range (a chunk) from a file, decodes it, splits it on
    special tokens, and performs pre-tokenization to generate token counts.

    Args:
        chunk_range: A tuple `(start, end)` indicating the byte offsets for the chunk.
        file_path: The path to the file from which the chunk will be read.
        special_tokens: A list of special tokens used for splitting the text.

    Returns:
        A dictionary of pre-token counts for the processed chunk.
    """
    start, end = chunk_range
    chunk_size = end - start

    if chunk_size == 0:
        return Counter()

    with open(file_path, "rb") as f:
        f.seek(start)
        data = f.read(chunk_size).decode("utf-8", errors="ignore")
        split_data = split_on_special_tokens(corpus=data, special_tokens=special_tokens)
        pretokens = pretokenization(split_data)

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

    # Recompute num_chunks in case, find_chunk_boundaries returned less chunks than desired.
    # This may happen when some boundaries are merged due to low density of tokens.
    num_chunks = len(chunk_boundaries) - 1

    logging.getLogger(__name__).info(f"Chunk boundaries: {chunk_boundaries}")

    with ProcessPoolExecutor(max_workers=num_chunks) as executor:
        worker_func = partial(process_chunk, file_path=corpus_path.as_posix(), special_tokens=special_tokens)
        chunk_ranges = [(chunk_boundaries[i], chunk_boundaries[i + 1]) for i in range(num_chunks)]
        futures = [executor.submit(worker_func, chunk_range) for chunk_range in chunk_ranges]

        # Use reduce with Counter addition for a concise and efficient aggregation.
        initial_counter: Counter[bytes] = Counter()
        return reduce(lambda acc, future: acc + future.result(), as_completed(futures), initial_counter)

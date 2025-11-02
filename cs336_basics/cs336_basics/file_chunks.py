from collections import Counter
import mmap
import os
from typing import BinaryIO, Final

import regex as re

BYTE_CACHE: Final = [i.to_bytes(1) for i in range(256)]


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bystring"

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
    pattern = r"|".join(re.escape(tok) for tok in special_tokens)
    split_corpus = re.split(pattern, corpus, concurrent=True)
    return [s for s in split_corpus if s]


def pretokenization(
    split_corpus: list[str],
    pattern: str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
) -> dict[tuple[bytes, ...], int]:
    occurences: dict[tuple[bytes, ...], int] = {}

    compiled_pattern = re.compile(pattern)

    occurences = Counter()
    for corpus in split_corpus:
        scanner = compiled_pattern.finditer(string=corpus, concurrent=True)

        occurences.update(tuple(BYTE_CACHE[b] for b in match_g.group().encode("utf-8")) for match_g in scanner)
    return occurences


def process_chunk(
    chunk_range: tuple[int, int], file_path: str, special_tokens: list[str]
) -> dict[tuple[bytes, ...], int]:
    """
    Worker function to read, split on special tokens, and pretokenize a single chunk of the file.
    """
    start, end = chunk_range
    chunk_size = end - start

    if chunk_size == 0:
        return {}

    with open(file_path, "rb") as f:
        f.seek(start)
        data = f.read(chunk_size).decode("utf-8", errors="ignore")
        split_data = split_on_special_tokens(corpus=data, special_tokens=special_tokens)
        pretokens = pretokenization(split_data)

        return pretokens

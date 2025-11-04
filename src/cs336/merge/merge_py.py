import logging


def single_merge(pretokens: dict[tuple[bytes, ...], int]) -> tuple[bytes, bytes] | None:
    """
    Finds the best pair of pretokens to merge based on their frequency.

    This function iterates through all pretoken sequences and counts the occurrences of
    adjacent pairs. It then returns the pair with the highest frequency. If multiple
    pairs have the same highest frequency, it returns the lexicographically largest one.

    Args:
        pretokens: A dictionary where keys are tuples of bytes representing pretoken
                   sequences and values are their frequencies.

    Returns:
        A tuple of two bytes objects representing the best pair to merge, or None if no pairs are found.
    """
    occurences: dict[tuple[bytes, bytes], int] = {}
    for pretoken, count in pretokens.items():
        for pair in zip(pretoken, pretoken[1:], strict=False):
            occurences[pair] = occurences.get(pair, 0) + count

    max_count = 0
    best_pair: tuple[bytes, bytes] | None = None
    for pair, count in occurences.items():
        if count > max_count:
            max_count = count
            best_pair = pair
        elif count == max_count:
            if best_pair is None:
                best_pair = pair
            else:
                best_pair = max(best_pair, pair)
    return best_pair


def merge(
    pretokens: dict[bytes, int], initial_vocab: dict[int, bytes], max_vocab_size: int
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Performs byte pair encoding (BPE) merges to build a vocabulary up to a specified size.

    This function iteratively finds the most frequent adjacent pair of tokens in the
    `pretokens` and merges them into a new token. This process continues until the
    vocabulary reaches `max_vocab_size` or no more merges can be performed.

    Args:
        pretokens: A dictionary where keys are tuples of bytes representing initial
                   pretoken sequences (as raw bytes) and values are their frequencies.
                   This dictionary is transformed and then modified in place during
                   the merging process.
        initial_vocab: A dictionary representing the initial vocabulary, where keys are
                       integer IDs and values are bytes objects. New merged tokens will
                       be added to this vocabulary.
        max_vocab_size: The maximum desired size of the vocabulary. The merging process
                        stops once this size is reached or exceeded.

    Returns:
        A tuple containing:
        - A dictionary representing the final vocabulary (integer ID to bytes object).
        - A list of tuples, where each tuple represents a merge operation (the two bytes objects that were merged).
    """
    merges: list[tuple[bytes, bytes]] = []
    vocab = initial_vocab.copy()

    BYTE_CACHE = tuple(bytes([i]) for i in range(256))

    # bytes -> tuple[bytes] to ease merging
    # {b'hello': 5} -> {(b'h', b'e', b'l', b'l', b'o'): 5}
    word_freqs: dict[tuple[bytes, ...], int] = {
        tuple(BYTE_CACHE[b] for b in word_bytes): freq for word_bytes, freq in pretokens.items()
    }

    while len(vocab) < max_vocab_size:
        if not word_freqs:
            break
        best_pair = single_merge(word_freqs)
        if best_pair:
            merges.append(best_pair)
            best_flat = b"".join(best_pair)
            vocab[len(vocab)] = best_flat

            new_pretokens: dict[tuple[bytes, ...], int] = {}
            for pretoken, count in word_freqs.items():
                new_pretoken_list = []
                i = 0
                pretoken_length = len(pretoken)
                while i < pretoken_length:
                    if i < pretoken_length - 1 and (pretoken[i], pretoken[i + 1]) == best_pair:
                        new_pretoken_list.append(best_flat)
                        i += 2  # If a match, skip the pair
                    else:
                        new_pretoken_list.append(pretoken[i])
                        i += 1

                new_pretoken_tuple = tuple(new_pretoken_list)
                new_pretokens[new_pretoken_tuple] = new_pretokens.get(new_pretoken_tuple, 0) + count
            word_freqs = new_pretokens
        else:
            logging.getLogger(__name__).warning("Unexpected: best_pair was None")
            break
    return vocab, merges

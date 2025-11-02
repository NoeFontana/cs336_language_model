import logging


def single_merge(pretokens: dict[tuple[bytes, ...], int]) -> tuple[bytes, bytes] | None:
    occurences: dict[tuple[bytes, bytes], int] = {}
    for pretoken, count in pretokens.items():
        for pair in zip(pretoken, pretoken[1:], strict=True):
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
    pretokens: dict[tuple[bytes, ...], int], initial_vocab: dict[int, bytes], max_vocab_size: int
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    merges: list[tuple[bytes, bytes]] = []
    vocab = initial_vocab.copy()
    while len(vocab) < max_vocab_size:
        if len(pretokens) == 0:
            break
        best_pair = single_merge(pretokens)
        if best_pair:
            merges.append(best_pair)
            best_flat = b"".join(best_pair)
            vocab[len(vocab)] = best_flat

            new_pretokens: dict[tuple[bytes, ...], int] = {}
            for pretoken, count in pretokens.items():
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
            pretokens = new_pretokens
        else:
            logging.getLogger(__name__).warning("Unexpected: best_pair was None")
    return vocab, merges

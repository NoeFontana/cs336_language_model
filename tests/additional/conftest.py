import pytest


@pytest.fixture
def initial_vocab_fixture() -> dict[int, bytes]:
    """Provides a standard initial vocabulary for BPE tests."""
    base_vocab_size = 256
    special_tokens = ["<|endoftext|>"]
    vocab: dict[int, bytes] = {i: i.to_bytes(1, "little") for i in range(base_vocab_size)}
    for token_id, special_token in enumerate(special_tokens, start=base_vocab_size):
        vocab[token_id] = special_token.encode("utf-8")
    return vocab

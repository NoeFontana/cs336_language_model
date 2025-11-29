import pytest

from cs336_basics.layer.feedforward import FeedForward, FFNReLUSquared, FFNSiLU
from cs336_basics.transformer import TransformerLM


def test_transformer_lm_ffn_selection() -> None:
    """Tests that TransformerLM instantiates the correct FFN type."""
    vocab_size = 100
    num_layers = 2
    d_model = 32
    num_heads = 4
    d_ff = 128
    max_seq_len = 10
    theta = 10000.0

    # Test default (SwiGLU)
    model_default = TransformerLM(vocab_size, num_layers, d_model, num_heads, d_ff, max_seq_len, theta)
    for block in model_default.transformer_blocks:
        assert isinstance(block.ffn, FeedForward)

    # Test ReLU Squared
    model_relu = TransformerLM(
        vocab_size,
        num_layers,
        d_model,
        num_heads,
        d_ff,
        max_seq_len,
        theta,
        ffn_type="relu_squared",
    )
    for block in model_relu.transformer_blocks:
        assert isinstance(block.ffn, FFNReLUSquared)

    # Test SiLU
    model_silu = TransformerLM(vocab_size, num_layers, d_model, num_heads, d_ff, max_seq_len, theta, ffn_type="silu")
    for block in model_silu.transformer_blocks:
        assert isinstance(block.ffn, FFNSiLU)


def test_transformer_lm_unknown_ffn() -> None:
    """Tests that TransformerLM raises ValueError for unknown FFN type."""
    vocab_size = 100
    num_layers = 2
    d_model = 32
    num_heads = 4
    d_ff = 128
    max_seq_len = 10
    theta = 10000.0

    with pytest.raises(ValueError):
        TransformerLM(
            vocab_size,
            num_layers,
            d_model,
            num_heads,
            d_ff,
            max_seq_len,
            theta,
            ffn_type="unknown",
        )

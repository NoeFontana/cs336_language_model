import torch

from cs336_basics.transformer import TransformerLM


def test_qk_norm_enabled_vs_disabled() -> None:
    """
    Tests that enabling qk_norm adds parameters to the model.
    """
    # Model configuration
    vocab_size = 100
    num_layers = 1
    d_model = 32
    num_heads = 4
    d_ff = 64
    max_seq_len = 64
    theta = 10000.0

    # Instantiate model with qk_norm=False
    model_no_norm = TransformerLM(
        vocab_size=vocab_size,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        theta=theta,
        qk_norm=False,
    )

    # Instantiate model with qk_norm=True
    model_norm = TransformerLM(
        vocab_size=vocab_size,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        theta=theta,
        qk_norm=True,
    )

    # Count parameters
    params_no_norm = sum(p.numel() for p in model_no_norm.parameters())
    params_norm = sum(p.numel() for p in model_norm.parameters())

    # Check that model with norm has more parameters
    # Each MHSA has 2 RMSNorms (Q and K). Each RMSNorm has 'd_head' params (gain).
    # d_head = 32 / 4 = 8.
    # 2 * 8 = 16 parameters per MHSA.
    # 1 layer -> 16 extra parameters.
    assert params_norm == params_no_norm + 16

    # Verify forward pass
    x = torch.randint(0, vocab_size, (1, 10))
    y_norm = model_norm(x)
    y_no_norm = model_no_norm(x)

    assert y_norm.shape == (1, 10, vocab_size)
    assert y_no_norm.shape == (1, 10, vocab_size)

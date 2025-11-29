import torch

from cs336_basics.transformer import TransformerLM


def test_transformer_lm_compile() -> None:
    """
    Tests that TransformerLM can be compiled with torch.compile and that the
    output is consistent with the uncompiled model.
    """
    # Model configuration
    vocab_size = 100
    num_layers = 2
    d_model = 32
    num_heads = 4
    d_ff = 64
    max_seq_len = 64
    theta = 10000.0

    # Instantiate the model
    model = TransformerLM(
        vocab_size=vocab_size,
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        theta=theta,
    )
    model.eval()  # Set to evaluation mode

    # Create dummy input
    batch_size = 4
    seq_len = 32
    x = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Get output from uncompiled model
    uncompiled_output = model(x)
    assert uncompiled_output.shape == (batch_size, seq_len, vocab_size)

    # Compile the model
    # Using mode="default" for robust compilation.
    # For faster test runs, mode="reduce-overhead" could be used.
    compiled_model = torch.compile(model, mode="default", fullgraph=True)

    # Get output from compiled model
    compiled_output = compiled_model(x)
    assert compiled_output.shape == (batch_size, seq_len, vocab_size)

    # Check that the outputs are numerically close
    assert torch.allclose(uncompiled_output, compiled_output, atol=1e-5)

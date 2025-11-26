import torch

from cs336.layer.feedforward import FFNReLUSquared


def test_ffn_relu_squared_shape() -> None:
    d_model = 16
    d_ff = 64
    batch_size = 2
    seq_len = 10
    model = FFNReLUSquared(d_model, d_ff)
    x = torch.randn(batch_size, seq_len, d_model)
    out = model(x)
    assert out.shape == (batch_size, seq_len, d_model)


def test_ffn_relu_squared_backward() -> None:
    d_model = 16
    d_ff = 64
    model = FFNReLUSquared(d_model, d_ff)
    x = torch.randn(2, 10, d_model, requires_grad=True)
    out = model(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert model.w1.weights.grad is not None
    assert model.w2.weights.grad is not None

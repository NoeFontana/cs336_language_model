import torch
from torch import nn

from cs336.layer.linear import Linear


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps  # ty: ignore[unresolved-attribute]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rrms = torch.rsqrt(x.to(torch.float32).pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rrms.to(x.dtype) * self.gain


class FeedForward(nn.Module):
    """Implements the SwiGLU feed-forward layer.

    This is a variant of the standard feed-forward layer in a Transformer,
    which uses a gated linear unit (GLU) with the SiLU activation function.
    See: https://arxiv.org/abs/2002.05202
    """

    def __init__(self, d_model: int, d_ff: int) -> None:
        """Initializes the FeedForward layer.

        Args:
            d_model: The dimensionality of the input and output.
            d_ff: The inner dimension of the feed-forward layer. It is usually
                a multiple of d_model.
        """
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass of the SwiGLU layer.

        Args:
            x: The input tensor of shape (..., d_model).
        Returns:
            The output tensor of shape (..., d_model).
        """

        silu_in = self.w1(x)
        silu = silu_in * torch.sigmoid(silu_in)
        swiglu = silu * self.w3(x)
        return self.w2(swiglu)

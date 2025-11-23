import torch
from torch import nn

from .attention import MHSA
from .feedforward import FeedForward
from .normalization import RMSNorm


class TransformerBlock(nn.Module):
    """Implements a (pre-norm) Transformer block."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float | None = None) -> None:
        """Initializes the TransformerBlock.

        Args:
            d_model: The dimensionality of the input and output.
            num_heads: The number of attention heads.
            d_ff: The inner dimension of the feed-forward layer.
            max_seq_len: The maximum sequence length.
        """
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MHSA(d_model, num_heads, max_seq_len, theta)
        self.ln2 = RMSNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """Performs the forward pass of the Transformer block.

        This method applies the pre-norm architecture:
        1. Normalization (RMSNorm) -> Multi-Head Self-Attention -> Residual Connection
        2. Normalization (RMSNorm) -> Feed-Forward Network -> Residual Connection

        Args:
            x: The input tensor of shape (..., seq_len, d_model).
            token_positions: Optional tensor for RoPE, shape (..., seq_len).

        Returns:
            The output tensor of shape (..., seq_len, d_model).
        """
        x = x + self.attn(self.ln1(x), token_positions)
        return x + self.ffn(self.ln2(x))

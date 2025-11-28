import torch
from einops import einsum, rearrange
from torch import nn

from cs336.layer.linear import Linear


class RotaryPositionalEmbedding(nn.Module):
    """Implements Rotary Positional Embeddings (RoPE).

    See: https://arxiv.org/abs/2104.09864
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None) -> None:
        """Initializes the RotaryPositionalEmbedding layer.

        Args:
            theta: The base for the geometric progression of frequencies. A common
                value is 10000.0.
            d_k: The dimensionality of the query/key vectors. Must be even.
            max_seq_len: The maximum sequence length for which to pre-compute
                the rotary embeddings.
            device: The device to create the buffer on.
        """
        super().__init__()

        positions = torch.arange(max_seq_len + 1, device=device, dtype=torch.float)
        exponents = torch.arange(0, d_k, step=2, device=device, dtype=torch.float) / d_k
        thetas_k = 1.0 / torch.pow(theta, exponents)

        freqs = torch.outer(positions, thetas_k)

        self.cos_cached: torch.Tensor
        self.sin_cached: torch.Tensor
        freqs_interleaved = torch.repeat_interleave(freqs, 2, dim=-1)
        self.register_buffer("cos_cached", freqs_interleaved.cos().to(dtype=torch.float32), persistent=False)
        self.register_buffer("sin_cached", freqs_interleaved.sin().to(dtype=torch.float32), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """Applies rotary positional embeddings to the input tensor.

        Args:
            x: The input tensor (queries or keys) of shape (..., seq_len, d_k).
            token_positions: A tensor of shape (..., seq_len) containing the
                absolute positions of tokens in the sequence.

        Returns:
            The input tensor with rotary positional embeddings applied, with the
            same shape as the input.
        """
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        
        return self.apply_rotary_interleaved(x, cos, sin)

    def apply_rotary_interleaved(self, x, cos, sin):
        """
        Manually implements complex multiplication for interleaved pairs:
        (a + ib)(cos + isin) = (a*cos - b*sin) + i(a*sin + b*cos)
        """
        x_pairs = x.view(*x.shape[:-1], -1, 2)
        
        x_real = x_pairs[..., 0]
        x_imag = x_pairs[..., 1]
        
        cos_reshaped = cos.view(*cos.shape[:-1], -1, 2)[..., 0]
        sin_reshaped = sin.view(*sin.shape[:-1], -1, 2)[..., 0]
        
        val_real = x_real * cos_reshaped - x_imag * sin_reshaped
        val_imag = x_real * sin_reshaped + x_imag * cos_reshaped
        
        rotated = torch.stack((val_real, val_imag), dim=-1).flatten(-2)
        
        return rotated


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x = torch.exp(x - x.amax(dim=dim, keepdim=True))
    return x / torch.sum(x, dim=dim, keepdim=True)


def scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Computes the scaled dot product attention.

    Args:
        q: Query tensor of shape (..., q_len, d_k).
        k: Key tensor of shape (..., kv_len, d_k).
        v: Value tensor of shape (..., kv_len, d_v).
        mask: Optional mask tensor of shape (..., seq_len, seq_len).
            True is kept. False is masked out.


    Returns:
        The output tensor of shape (..., seq_len, d_v).

    """
    d_k = q.shape[-1]
    scaling = d_k**-0.5
    att: torch.Tensor = einsum(q, k, "... queries d_k, ... keys d_k -> ... queries keys") * scaling

    if mask is not None:
        att.masked_fill_(~mask, float("-inf"))

    att = softmax(att, dim=-1)
    return att @ v


class MHSA(nn.Module):
    """Implements Multi-Head Self-Attention.

    See: https://arxiv.org/abs/1706.03762
    """

    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float | None = None) -> None:
        """Initializes the MHSA layer.

        Args:
            d_model: The dimensionality of the input and output.
            num_heads: The number of attention heads.
            max_seq_len: The maximum sequence length for which to pre-compute
                the rotary embeddings.
            theta: The base for the geometric progression of frequencies of RoPE.
        """
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.num_heads = num_heads  # ty: ignore[unresolved-attribute]
        self.d_head = d_model // num_heads  # ty: ignore[unresolved-attribute]

        self.qkv_proj = Linear(d_model, 3 * d_model)

        self.out_proj = Linear(d_model, d_model)

        mask = torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)
        self.mask: torch.Tensor
        self.register_buffer("mask", ~mask, persistent=False)

        self.rope: RotaryPositionalEmbedding | None = None
        if theta is not None:
            self.rope = RotaryPositionalEmbedding(theta, self.d_head, max_seq_len)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """Performs the forward pass of the MHSA layer.

        Args:
            x: The input tensor of shape (..., seq_len, d_model).
            token_positions: Optional tensor for RoPE, shape (..., seq_len).

        Returns:
            The output tensor of shape (..., seq_len, d_model).
        """
        qkv: torch.Tensor = self.qkv_proj(x)

        qkv_h = rearrange(qkv, "... s (three h d) -> three ... h s d", three=3, h=self.num_heads, d=self.d_head)
        q_h, k_h, v_h = qkv_h[0], qkv_h[1], qkv_h[2]

        seq_len = x.shape[-2]
        causal_mask = self.mask[:seq_len, :seq_len]

        if token_positions is not None and self.rope is not None:
            q_h = self.rope(q_h, token_positions)
            k_h = self.rope(k_h, token_positions)

        atts = scaled_dot_product_attention(q_h, k_h, v_h, causal_mask)
        atts = rearrange(atts, "... h s d -> ... s (h d)", h=self.num_heads, d=self.d_head)

        out = self.out_proj(atts)

        return out

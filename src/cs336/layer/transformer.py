import torch
from einops import einsum, rearrange
from torch import nn

from cs336.layer.linear import Linear


class RMSNorm(nn.Module):
    """Implements Root Mean Square Layer Normalization.

    This normalization technique is a simplification of standard Layer Normalization.
    It normalizes the activations of a layer by their root mean square, and then
    rescales them with a learnable gain parameter.

    See: https://arxiv.org/abs/1910.07467
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initializes the RMSNorm layer.

        Args:
            d_model: The dimensionality of the input tensor.
            eps: A small value added to the denominator for numerical stability.
            device: The device to create the parameters on.
            dtype: The data type for the parameters.
        """
        super().__init__()

        self.gain = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.eps = eps  # ty: ignore[unresolved-attribute]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass of the RMSNorm layer.

        Args:
            x: The input tensor of shape (..., d_model).

        Returns:
            The normalized tensor, with the same shape as the input.
        """
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

        positions = torch.arange(max_seq_len + 1, device=device)
        exponents = torch.arange(0, d_k, step=2, device=device) / d_k
        thetas_k = 1 / torch.pow(theta, exponents)

        freqs = torch.outer(positions, thetas_k)
        freqs_cos_isine = torch.polar(torch.ones_like(freqs), freqs)

        self.freqs_cos_isine: torch.Tensor
        self.register_buffer("freqs_cos_isine", freqs_cos_isine, persistent=False)

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
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cos_isine = self.freqs_cos_isine[token_positions]

        x_rotated_complex = x_complex * freqs_cos_isine

        return torch.view_as_real(x_rotated_complex).view(x.shape)


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

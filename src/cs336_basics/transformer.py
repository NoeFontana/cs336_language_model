import torch
from torch import nn

import cs336_basics.layer as layer


class TransformerLM(nn.Module):
    """A Transformer-based language model."""

    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        ffn_type: str = "swiglu",
        qk_norm: bool = False,
    ) -> None:
        """Initializes the Transformer Language Model.

        Args:
            vocab_size: The size of the vocabulary.
            num_layers: The number of Transformer blocks to stack.
            d_model: The dimensionality of the model's embeddings and layers.
            num_heads: The number of attention heads in each Transformer block.
            d_ff: The inner dimension of the feed-forward networks.
            max_seq_len: The maximum sequence length for pre-computing positional
                embeddings.
            theta: The base for the rotary positional embeddings (RoPE).
            ffn_type: The type of feed-forward network to use.
            qk_norm: Whether to apply RMSNorm to the queries and keys.
        """
        super().__init__()

        self.d_model = d_model  # ty: ignore[unresolved-attribute]
        self.embedding = layer.Embedding(vocab_size, d_model)
        self.transformer_blocks = nn.ModuleList(
            [
                layer.TransformerBlock(d_model, num_heads, d_ff, max_seq_len, theta, ffn_type=ffn_type, qk_norm=qk_norm)
                for _ in range(num_layers)
            ]
        )
        self.out_norm = layer.RMSNorm(d_model)
        self.out_linear = layer.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass of the language model.

        Args:
            x: The input tensor of token indices, with shape
                (..., seq_len).

        Returns:
            A tensor of output logits for the next token, with shape
            (..., seq_len, vocab_size).
        """
        embed = self.embedding(x)

        token_positions = torch.arange(0, x.shape[-1], device=x.device)
        for block in self.transformer_blocks:
            embed = block(embed, token_positions)
        embed = self.out_norm(embed)
        return self.out_linear(embed)

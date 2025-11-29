import torch
from torch import nn


class Embedding(nn.Module):
    """A simple lookup table that stores embeddings of a fixed dictionary and size."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initializes the Embedding layer.

        Args:
            num_embeddings: The size of the dictionary of embeddings.
            embedding_dim: The size of each embedding vector.
            device: The desired device of the embedding matrix.
            dtype: The desired data type of the embedding matrix.
        """
        super().__init__()

        std = 0.02
        self.embedding_matrix = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(size=(num_embeddings, embedding_dim), device=device, dtype=dtype),
                mean=0,
                std=std,
                a=-3 * std,
                b=3 * std,
            )
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Performs a lookup for each token ID.

        Args:
            token_ids: A tensor of shape (batch, seq_len) containing the indices to be
                looked up in the embedding matrix.

        Returns:
            A tensor of shape (batch, seq_len, embedding_dim) containing the corresponding
            embeddings.
        """
        return self.embedding_matrix[token_ids]

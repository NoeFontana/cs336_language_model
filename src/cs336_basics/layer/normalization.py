import torch
from torch import nn


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

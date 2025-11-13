import torch
from torch import nn


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

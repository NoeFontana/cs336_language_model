import torch
from einops import einsum
from torch.nn import Module, Parameter


class Linear(Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
        super().__init__()

        std = 2 / (in_features + out_features)
        self.weights = Parameter(
            torch.nn.init.trunc_normal_(
                torch.empty(size=(out_features, in_features), device=device, dtype=dtype),
                mean=0,
                std=std,
                a=-3 * std,
                b=3 * std,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weights, "... in_feat, out_feat in_feat -> ... out_feat")

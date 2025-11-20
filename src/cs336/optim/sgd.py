import math
from collections.abc import Callable, Iterable
from typing import Any

import torch


class SGD(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]] | Iterable[tuple[str, torch.Tensor]],
        lr=1e-3,
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(  # type: ignore[override]
        self,
        closure: Callable[[], float] | None = None,
    ) -> float | None:
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]  # Get the learning rate.

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or initial value.

                grad = p.grad.data  # Get the gradient of loss with respect to p.

                # Update weight tensor in-place.
                p.data -= lr / math.sqrt(t + 1) * grad

                state["t"] = t + 1  # Increment iteration number.
        return loss

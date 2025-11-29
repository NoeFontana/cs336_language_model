import math
from collections.abc import Callable, Iterable
from typing import Any, cast

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]] | Iterable[tuple[str, torch.Tensor]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
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
            lr = cast(float, group["lr"])
            beta1, beta2 = cast(tuple[float, float], group["betas"])
            eps = cast(float, group["eps"])
            weight_decay = cast(float, group["weight_decay"])

            params = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []

            current_step = 0

            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["t"] = 0
                    # Exponential moving average of gradient values
                    state["m"] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state["v"] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                state["t"] += 1
                current_step = state["t"]

                params.append(p)
                grads.append(p.grad)
                exp_avgs.append(state["m"])
                exp_avg_sqs.append(state["v"])
            if not params:
                continue

            torch._foreach_mul_(exp_avgs, beta1)
            torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1)

            torch._foreach_mul_(exp_avg_sqs, beta2)
            torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1 - beta2)

            bias_correction1 = 1 - beta1**current_step
            bias_correction2 = 1 - beta2**current_step

            step_size = lr * math.sqrt(bias_correction2) / bias_correction1

            # Grad step
            denom = torch._foreach_sqrt(exp_avg_sqs)
            torch._foreach_add_(denom, eps)
            torch._foreach_addcdiv_(params, exp_avgs, denom, value=-step_size)

            # Weight decay step
            torch._foreach_mul_(params, 1 - lr * weight_decay)

        return loss

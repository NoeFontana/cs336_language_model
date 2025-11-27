import math
from collections.abc import Callable, Iterable, Sequence
from typing import Any, cast

import torch
from torch.optim import Optimizer


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 10, eps: float = 1e-7) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)

    X = G.to(torch.bfloat16)

    X.div_(torch.linalg.norm(X) + eps)

    # Transpose if necessary to ensure X @ X.T is the smaller matrix (rows <= cols)
    if G.size(0) > G.size(1):
        X = X.T

    for _ in range(steps):
        A = X @ X.T
        B = A @ A
        B.mul_(c).add_(A, alpha=b)

        X.addmm_(B, X, alpha=1.0, beta=a)

    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(Optimizer):
    """
    Muon - MomentUm Orthogonalized Optimizer
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.01,
        adamw_params: Iterable[torch.Tensor] | None = None,
        adamw_lr: float = 3e-4,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        adamw_weight_decay: float = 0.01,
    ) -> None:
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "nesterov": nesterov,
            "ns_steps": ns_steps,
            "weight_decay": weight_decay,
            "adamw_lr": adamw_lr,
            "adamw_betas": adamw_betas,
            "adamw_eps": adamw_eps,
            "adamw_weight_decay": adamw_weight_decay,
        }

        params = list(params)
        adamw_params = list(adamw_params) if adamw_params is not None else []

        param_groups = [
            {
                "params": params,
                "use_muon": True,
                "lr": lr,
                "momentum": momentum,
                "nesterov": nesterov,
                "ns_steps": ns_steps,
                "weight_decay": weight_decay,
            },
            {
                "params": adamw_params,
                "use_muon": False,
                "lr": adamw_lr,
                "betas": adamw_betas,
                "eps": adamw_eps,
                "weight_decay": adamw_weight_decay,
            },
        ]

        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:  # type: ignore[override,reportIncompatibleMethodOverride]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group.get("use_muon", False):
                self._step_muon(group)
            else:
                self._step_adamw(group)

        return loss

    def _step_muon(self, group: dict[str, Any]) -> None:
        lr = cast(float, group["lr"])
        momentum = cast(float, group["momentum"])
        nesterov = cast(bool, group["nesterov"])
        ns_steps = cast(int, group["ns_steps"])
        weight_decay = cast(float, group["weight_decay"])

        params = [p for p in group["params"] if p.grad is not None]
        if not params:
            return

        grads = [p.grad for p in params]

        momentum_buffer = []
        for p in params:
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(p, memory_format=torch.preserve_format)
            momentum_buffer.append(state["momentum_buffer"])

        # buf = buf * momentum + grad
        torch._foreach_mul_(momentum_buffer, momentum)
        torch._foreach_add_(momentum_buffer, grads)

        if nesterov:
            # g = grad + momentum * buf
            # We use a temporary list for 'g' to avoid modifying 'grad' in place
            g_list: Sequence[torch.Tensor] = torch._foreach_add(grads, momentum_buffer, alpha=momentum)
        else:
            g_list = momentum_buffer

        # Newton-Schulz Orthogonalization
        for p, g in zip(params, g_list, strict=False):
            if g.ndim >= 2:
                view_shape = g.shape
                g_2d = g.view(g.size(0), -1)
                update_2d = zeropower_via_newtonschulz5(g_2d, steps=ns_steps)
                update = update_2d.view(view_shape)

                rows = g.size(0)
                cols = g.numel() // rows
                scale = max(1, rows / cols) ** 0.5
            else:
                g_2d = g.view(g.size(0), -1)
                update_2d = zeropower_via_newtonschulz5(g_2d, steps=ns_steps)
                update = update_2d.view(g.shape)
                scale = 1.0

            # Gradient Step
            p.add_(update.type_as(p), alpha=-lr * scale)

        # Weight Decay Step
        if weight_decay > 0:
            torch._foreach_mul_(params, 1 - lr * weight_decay)

    def _step_adamw(self, group: dict[str, Any]) -> None:
        lr = cast(float, group["lr"])
        beta1, beta2 = cast(tuple[float, float], group["betas"])
        eps = cast(float, group["eps"])
        weight_decay = cast(float, group["weight_decay"])

        params = []
        grads = []
        exp_avgs = []
        exp_avg_sqs = []
        state_steps = []

        for p in group["params"]:
            if p.grad is None:
                continue

            if p.grad.is_sparse:
                raise RuntimeError("AdamW does not support sparse gradients")

            state = self.state[p]

            # State initialization
            if len(state) == 0:
                state["t"] = 0
                state["m"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["v"] = torch.zeros_like(p, memory_format=torch.preserve_format)

            state["t"] += 1

            params.append(p)
            grads.append(p.grad)
            exp_avgs.append(state["m"])
            exp_avg_sqs.append(state["v"])
            state_steps.append(state["t"])

        if not params:
            return

        # Update first moment (m)
        # m = m * beta1 + grad * (1 - beta1)
        torch._foreach_mul_(exp_avgs, beta1)
        torch._foreach_add_(exp_avgs, grads, alpha=1 - beta1)

        # Update second moment (v)
        # v = v * beta2 + (grad * grad) * (1 - beta2)
        torch._foreach_mul_(exp_avg_sqs, beta2)
        torch._foreach_addcmul_(exp_avg_sqs, grads, grads, value=1 - beta2)

        t = state_steps[0]
        bias_correction1 = 1 - beta1**t
        bias_correction2 = 1 - beta2**t
        step_size = lr * math.sqrt(bias_correction2) / bias_correction1

        # Grad Step
        # p = p - step_size * m / (sqrt(v) + eps)
        denom = torch._foreach_sqrt(exp_avg_sqs)
        torch._foreach_add_(denom, eps)

        torch._foreach_addcdiv_(params, exp_avgs, denom, value=-step_size)

        # Weight Decay Step
        # p = p * (1 - lr * weight_decay)
        if weight_decay > 0:
            torch._foreach_mul_(params, 1 - lr * weight_decay)

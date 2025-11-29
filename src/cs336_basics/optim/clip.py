from collections.abc import Iterable

import torch


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Given a set of parameters, clip their combined gradients to have l2 norm at most max_l2_norm.

    Args:
        parameters (Iterable[torch.nn.Parameter]): collection of trainable parameters.
        max_l2_norm (float): a positive value containing the maximum l2-norm.

    The gradients of the parameters (parameter.grad) should be modified in-place.
    """

    total_norm = sum(parameter.grad.pow(2).sum() for parameter in parameters if parameter.grad is not None) ** 0.5
    if total_norm <= max_l2_norm:
        return
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(max_l2_norm / (total_norm + 1e-6))

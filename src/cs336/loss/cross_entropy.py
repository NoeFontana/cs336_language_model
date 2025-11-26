import torch
import torch.nn as nn


def cross_entropy(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Computes the cross-entropy loss.

    Args:
        logits: A tensor of shape (batch, num_classes) containing the logits
            for each class.
        labels: A tensor of shape (batch)
    """

    logits = logits - torch.amax(logits, dim=-1, keepdim=True)

    loss = -logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1) + torch.log(torch.sum(torch.exp(logits), dim=-1))

    return torch.mean(loss)


class CrossEntropyLoss(nn.Module):
    """Cross-entropy loss module."""

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return cross_entropy(logits, labels)

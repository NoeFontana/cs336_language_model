import os
import typing

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
) -> None:
    """
    Saves the model and optimizer state to a checkpoint file.

    Args:
        model (torch.nn.Module): The PyTorch model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        iteration (int): The current training iteration number.
        out (str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
            The path or file-like object to save the checkpoint to.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """
    Loads the model and optimizer state from a checkpoint file.

    The states are loaded in-place into the provided model and optimizer objects.

    Args:
        src (str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
            The path or file-like object to load the checkpoint from.
        model (torch.nn.Module): The PyTorch model to load the state into.
        optimizer (torch.optim.Optimizer): The optimizer to load the state into.

    Returns:
        int: The iteration number saved in the checkpoint.
    """
    # When loading a checkpoint, it's good practice to map to the CPU first,
    # then move the model to the desired device. This avoids potential issues
    # if the checkpoint was saved on a different device (e.g., a different GPU).
    checkpoint = torch.load(src, map_location="cpu")

    if not all(k in checkpoint for k in ["model_state_dict", "optimizer_state_dict", "iteration"]):
        raise ValueError("Checkpoint is missing required keys.")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    iteration = checkpoint["iteration"]

    return iteration

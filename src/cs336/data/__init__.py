import numpy as np
import torch


def get_batch(
    token: np.ndarray, batch_size: int, context_length: int, device: str | torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a batch of input sequences and their corresponding targets from a tokenized dataset.

    This function randomly samples starting positions from the dataset and extracts
    sequences of a specified context length. The target for each position in a
    sequence is the next token in the dataset.

    Args:
        token (np.ndarray): A 1D NumPy array of integer token IDs.
        batch_size (int): The number of sequences in a batch.
        context_length (int): The length of each sequence.
        device (str | torch.device): The PyTorch device to place the output tensors on.

    Returns:
        A tuple containing two tensors:
        - inputs (torch.Tensor): A tensor of shape (batch_size, context_length)
          containing the input sequences.
        - targets (torch.Tensor): A tensor of shape (batch_size, context_length)
          containing the target sequences, where target[i, j] is the token
          that follows input[i, j].
    """
    start_indices = np.random.randint(0, len(token) - context_length, size=(batch_size,))

    indices = start_indices[:, np.newaxis] + np.arange(context_length)

    inputs_np = token[indices]
    targets_np = token[indices + 1]

    # Convert to PyTorch tensors and move to the specified device
    inputs = torch.from_numpy(inputs_np).to(device, torch.long, non_blocking=True)
    targets = torch.from_numpy(targets_np).to(device, torch.long, non_blocking=True)

    return inputs, targets

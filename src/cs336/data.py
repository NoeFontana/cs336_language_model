import os

import numpy as np
import torch
from torch.utils.data import Dataset


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
        A tuple of (inputs, targets) tensors.
            - inputs: A tensor of shape (batch_size, context_length) containing
              the input sequences.
            - targets: A tensor of shape (batch_size, context_length) containing
              the target sequences, where target[i, j] is the token that
              follows input[i, j].
    """
    start_indices = np.random.randint(0, len(token) - context_length, size=(batch_size,))

    indices = start_indices[:, np.newaxis] + np.arange(context_length)

    inputs_np = token[indices]
    targets_np = token[indices + 1]

    # Convert to PyTorch tensors and move to the specified device
    inputs = torch.from_numpy(inputs_np).to(device, torch.long, non_blocking=True)
    targets = torch.from_numpy(targets_np).to(device, torch.long, non_blocking=True)

    return inputs, targets


class LanguageModelDataset(Dataset):
    """
    A PyTorch Dataset that uses memory mapping for efficient, large-scale
    data access across multiple workers.

    Implements Block Indexing which allows standard shuffling without
    massive memory overhead.
    """

    def __init__(self, bin_file: str, context_length: int):
        """
        Args:
            bin_file (str): Path to the binary file containing token IDs.
            context_length (int): The length of each sequence.
        """
        super().__init__()
        self.bin_file = bin_file
        self.context_length = context_length

        if not os.path.exists(bin_file):
            raise FileNotFoundError(f"Data file not found at {bin_file}")

        file_size_bytes = os.path.getsize(bin_file)
        item_size = np.dtype(np.uint16).itemsize
        self.total_tokens = file_size_bytes // item_size

        if self.total_tokens < context_length + 1:
            raise ValueError(f"Dataset too small: {self.total_tokens} tokens for context {context_length}")

        self.num_samples = (self.total_tokens - 1) // self.context_length

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Each worker process will open its own memory map the first time it fetches a batch.
        if not hasattr(self, "data"):
            self.data = np.memmap(self.bin_file, dtype=np.uint16, mode="r")

        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of bounds")

        start_idx = idx * self.context_length
        end_idx = start_idx + self.context_length + 1

        chunk = self.data[start_idx:end_idx].astype(np.int64)

        inputs = torch.from_numpy(chunk[:-1])
        targets = torch.from_numpy(chunk[1:])

        return inputs, targets


def create_train_loader(
    data_path: str,
    context_length: int,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    seed: int | None = 42,
) -> torch.utils.data.DataLoader:
    dataset = LanguageModelDataset(data_path, context_length)

    persistent_workers = persistent_workers and num_workers > 0

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        generator=generator,
    )


def create_val_loader(
    data_path: str,
    context_length: int,
    batch_size: int,
    num_workers: int = 2,
    pin_memory: bool = True,
    persistent_workers: bool = True,
) -> torch.utils.data.DataLoader:
    dataset = LanguageModelDataset(data_path, context_length)
    persistent_workers = persistent_workers and num_workers > 0

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

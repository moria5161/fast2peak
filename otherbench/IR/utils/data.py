"""Paired NumPy spectrum loader."""

import numpy as np
import torch
from torch.utils.data import Dataset


class NpyPairDataset(Dataset):
    def __init__(self, inputs: np.ndarray, targets: np.ndarray) -> None:
        self.inputs = inputs
        self.targets = targets

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # mmap_mode='r' arrays are intentionally read-only; copy one row so PyTorch
        # never receives a non-writable view.
        return (
            torch.from_numpy(np.array(self.inputs[index], copy=True)),
            torch.from_numpy(np.array(self.targets[index], copy=True)),
        )

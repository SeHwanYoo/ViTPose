"""fastMRI multi-coil dataset with semi-supervised split support."""

from __future__ import annotations

import pathlib
import random
from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import retrospective_mask


@dataclass
class FastMRISample:
    kspace: torch.Tensor  # (coils, H, W, 2)
    mask: torch.Tensor  # (H, W, 1)
    target: torch.Tensor | None  # (H, W)
    is_labeled: bool
    fname: str
    slice_idx: int


class FastMRISemiSupervisedDataset(Dataset):
    """Volume-level labeled/unlabeled split for fastMRI knee multi-coil."""

    def __init__(
        self,
        root: str,
        split: str,
        labeled_ratio: float,
        acceleration: int,
        acs_ratio: float = 0.08,
        seed: int = 42,
        crop_size: int = 320,
    ) -> None:
        if split not in {'train', 'val'}:
            raise ValueError("split must be 'train' or 'val'")
        self.root = pathlib.Path(root)
        self.split = split
        self.labeled_ratio = labeled_ratio
        self.acceleration = acceleration
        self.acs_ratio = acs_ratio
        self.crop_size = crop_size

        files = sorted((self.root / split).glob('*.h5'))
        if not files:
            raise FileNotFoundError(f'No HDF5 files found at {(self.root / split)!s}')

        rng = random.Random(seed)
        labeled_files = set(rng.sample(files, k=max(1, int(round(len(files) * labeled_ratio))))) if split == 'train' else set(files)

        self.index: list[tuple[pathlib.Path, int, bool]] = []
        for f in files:
            with h5py.File(f, 'r') as hf:
                num_slices = int(hf['kspace'].shape[0])
            is_labeled = f in labeled_files
            for s in range(num_slices):
                self.index.append((f, s, is_labeled))

    def __len__(self) -> int:
        return len(self.index)

    def _center_crop(self, arr: np.ndarray) -> np.ndarray:
        h, w = arr.shape[-2:]
        top = max((h - self.crop_size) // 2, 0)
        left = max((w - self.crop_size) // 2, 0)
        return arr[..., top:top + self.crop_size, left:left + self.crop_size]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        path, slice_idx, is_labeled = self.index[idx]
        with h5py.File(path, 'r') as hf:
            kspace = hf['kspace'][slice_idx]  # (coils, H, W, 2) for converted fastMRI
            if kspace.shape[-1] != 2:
                kspace = np.stack([kspace.real, kspace.imag], axis=-1)
            kspace = self._center_crop(kspace.transpose(0, 3, 1, 2)).transpose(0, 2, 3, 1)

            target = None
            if is_labeled and 'reconstruction_rss' in hf:
                target = self._center_crop(hf['reconstruction_rss'][slice_idx])

        coils, h, w, _ = kspace.shape
        mask = retrospective_mask((h, w), self.acceleration, self.acs_ratio)

        return {
            'kspace': torch.from_numpy(kspace).float(),
            'mask': mask.float(),
            'target': None if target is None else torch.from_numpy(target).float(),
            'is_labeled': is_labeled,
            'fname': path.name,
            'slice_idx': slice_idx,
            'num_coils': coils,
        }

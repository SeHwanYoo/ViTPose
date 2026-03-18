"""MRI tensor transforms and complex math helpers."""

from __future__ import annotations

import math
from typing import Tuple

import torch

Tensor = torch.Tensor


def to_complex(x: Tensor) -> Tensor:
    if torch.is_complex(x):
        return x
    if x.shape[-1] != 2:
        raise ValueError('Expected [..., 2] real/imag tensor.')
    return torch.view_as_complex(x.contiguous())


def to_ri(x: Tensor) -> Tensor:
    if torch.is_complex(x):
        return torch.view_as_real(x)
    return x


def fft2c(x: Tensor) -> Tensor:
    xc = to_complex(x)
    out = torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(xc, dim=(-2, -1)), norm='ortho'),
        dim=(-2, -1),
    )
    return to_ri(out)


def ifft2c(x: Tensor) -> Tensor:
    xc = to_complex(x)
    out = torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(xc, dim=(-2, -1)), norm='ortho'),
        dim=(-2, -1),
    )
    return to_ri(out)


def rss_combine(img: Tensor, coil_dim: int = 1, eps: float = 1e-8) -> Tensor:
    """Root-sum-of-squares combine, returns magnitude image."""
    img_c = to_complex(img)
    mag = torch.abs(img_c)
    return torch.sqrt(torch.clamp((mag**2).sum(dim=coil_dim), min=eps))


def normalize_instance(img: Tensor, eps: float = 1e-8) -> Tuple[Tensor, Tensor, Tensor]:
    mean = img.mean(dim=(-2, -1), keepdim=True)
    std = img.std(dim=(-2, -1), keepdim=True).clamp_min(eps)
    return (img - mean) / std, mean, std


def retrospective_mask(
    shape: tuple[int, int],
    acceleration: int,
    acs_ratio: float = 0.08,
    device: torch.device | None = None,
) -> Tensor:
    """Create Cartesian equispaced sampling mask (H, W, 1)."""
    h, w = shape
    mask = torch.zeros(w, device=device)
    center = int(round(w * acs_ratio))
    left = (w - center) // 2
    right = left + center
    mask[left:right] = 1.0
    mask[::acceleration] = 1.0
    return mask.view(1, w, 1).repeat(h, 1, 1)

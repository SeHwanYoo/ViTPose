"""Evaluation metrics for MRI reconstruction."""

from __future__ import annotations

import torch


def nmse(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return ((pred - target) ** 2).sum() / (target.pow(2).sum() + eps)


def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0, eps: float = 1e-8) -> torch.Tensor:
    mse = ((pred - target) ** 2).mean().clamp_min(eps)
    return 20 * torch.log10(torch.tensor(max_val, device=pred.device)) - 10 * torch.log10(mse)


def ssim_global(pred: torch.Tensor, target: torch.Tensor, c1: float = 0.01**2, c2: float = 0.03**2) -> torch.Tensor:
    mu_x = pred.mean()
    mu_y = target.mean()
    sigma_x = ((pred - mu_x) ** 2).mean()
    sigma_y = ((target - mu_y) ** 2).mean()
    sigma_xy = ((pred - mu_x) * (target - mu_y)).mean()
    return ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2))

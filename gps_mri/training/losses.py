"""Losses for GPS MRI training phases."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from gps_mri.models.varnet import E2EVarNet

Tensor = torch.Tensor


class SSIMLoss(nn.Module):
    """1 - SSIM for grayscale MRI magnitude images."""

    def __init__(self, c1: float = 0.01**2, c2: float = 0.03**2):
        super().__init__()
        self.c1 = c1
        self.c2 = c2

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
            target = target.unsqueeze(1)
        mu_x = pred.mean(dim=(-2, -1), keepdim=True)
        mu_y = target.mean(dim=(-2, -1), keepdim=True)
        sigma_x = ((pred - mu_x) ** 2).mean(dim=(-2, -1), keepdim=True)
        sigma_y = ((target - mu_y) ** 2).mean(dim=(-2, -1), keepdim=True)
        sigma_xy = ((pred - mu_x) * (target - mu_y)).mean(dim=(-2, -1), keepdim=True)
        ssim = ((2 * mu_x * mu_y + self.c1) * (2 * sigma_xy + self.c2)) / (
            (mu_x**2 + mu_y**2 + self.c1) * (sigma_x + sigma_y + self.c2)
        )
        return 1.0 - ssim.mean()


class ReconLoss(nn.Module):
    def __init__(self, lambda_ssim: float = 0.5):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.ssim = SSIMLoss()
        self.lambda_ssim = lambda_ssim

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        return self.l1(pred, target) + self.lambda_ssim * self.ssim(pred, target)


def data_consistency_loss(model: E2EVarNet, pred_img_ri: Tensor, smaps: Tensor, y: Tensor, mask: Tensor) -> Tensor:
    pred_y = model.forward_operator(pred_img_ri, smaps, mask)
    return ((pred_y - y) ** 2).mean()


@dataclass
class StudentLossBundle:
    total: Tensor
    l_sup: Tensor
    l_gps: Tensor
    l_cps: Tensor
    l_dc: Tensor


class GPSLossComputer:
    def __init__(
        self,
        lambda_sup: float = 1.0,
        lambda_gps: float = 1.0,
        lambda_cps: float = 0.5,
        lambda_dc: float = 0.1,
        lambda_fb: float = 0.5,
        lambda_ssim: float = 0.5,
    ) -> None:
        self.lambda_sup = lambda_sup
        self.lambda_gps = lambda_gps
        self.lambda_cps = lambda_cps
        self.lambda_dc = lambda_dc
        self.lambda_fb = lambda_fb
        self.recon = ReconLoss(lambda_ssim=lambda_ssim)

    def ramp(self, step: int, ramp_steps: int) -> float:
        if ramp_steps <= 0:
            return 1.0
        p = min(max(step, 0), ramp_steps) / ramp_steps
        return float(math.exp(-5.0 * (1.0 - p) ** 2))

    def student_loss(
        self,
        step: int,
        ramp_steps: int,
        model,
        labeled_batch: dict,
        unlabeled_batch: dict,
    ) -> StudentLossBundle:
        ls1 = model.student1(labeled_batch['y'], labeled_batch['mask'])
        ls2 = model.student2(labeled_batch['y'], labeled_batch['mask'])
        l_sup = 0.5 * (
            self.recon(ls1.image, labeled_batch['target']) +
            self.recon(ls2.image, labeled_batch['target'])
        )

        with torch.no_grad():
            teacher_u = model.teacher(unlabeled_batch['y'], unlabeled_batch['mask'])

        us1 = model.student1(unlabeled_batch['y'], unlabeled_batch['mask'])
        us2 = model.student2(unlabeled_batch['y'], unlabeled_batch['mask'])

        l_gps = 0.5 * (self.recon(us1.image, teacher_u.image.detach()) + self.recon(us2.image, teacher_u.image.detach()))
        l_cps = 0.5 * (self.recon(us1.image, us2.image.detach()) + self.recon(us2.image, us1.image.detach()))

        l_dc = 0.5 * (
            data_consistency_loss(model.student1, us1.image_ri, us1.smaps, unlabeled_batch['y'], unlabeled_batch['mask']) +
            data_consistency_loss(model.student2, us2.image_ri, us2.smaps, unlabeled_batch['y'], unlabeled_batch['mask'])
        )

        r = self.ramp(step, ramp_steps)
        total = self.lambda_sup * l_sup + r * (
            self.lambda_gps * l_gps + self.lambda_cps * l_cps + self.lambda_dc * l_dc
        )
        return StudentLossBundle(total=total, l_sup=l_sup, l_gps=l_gps, l_cps=l_cps, l_dc=l_dc)

    def teacher_feedback_loss(self, model, labeled_batch: dict, unlabeled_batch: dict) -> Tensor:
        t_u = model.teacher(unlabeled_batch['y'], unlabeled_batch['mask'])
        with torch.no_grad():
            s1 = model.student1(unlabeled_batch['y'], unlabeled_batch['mask'])
            s2 = model.student2(unlabeled_batch['y'], unlabeled_batch['mask'])
            x_fb = 0.5 * (s1.image + s2.image)

        t_l = model.teacher(labeled_batch['y'], labeled_batch['mask'])
        l_fb = self.recon(t_u.image, x_fb.detach())
        l_sup = self.recon(t_l.image, labeled_batch['target'])
        return self.lambda_fb * l_fb + 0.5 * self.lambda_sup * l_sup

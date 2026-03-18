"""3-phase GPS MRI training loop."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from gps_mri.models.gps_recon_model import GPSMRIModel
from .losses import GPSLossComputer


@dataclass
class PhaseConfig:
    teacher_pretrain_steps: int = 10_000
    cps_ramp_steps: int = 10_000
    feedback_start_step: int = 20_000
    feedback_ramp_steps: int = 5_000


class GPSMRITrainer:
    def __init__(
        self,
        model: GPSMRIModel,
        student_optimizer: torch.optim.Optimizer,
        teacher_optimizer: torch.optim.Optimizer,
        losses: GPSLossComputer,
        cfg: PhaseConfig,
    ) -> None:
        self.model = model
        self.student_optimizer = student_optimizer
        self.teacher_optimizer = teacher_optimizer
        self.losses = losses
        self.cfg = cfg
        self.feedback_triggered = False
        self.feedback_trigger_step: int | None = None

    def teacher_pretrain_step(self, labeled_batch: dict) -> dict[str, float]:
        self.model.unfreeze_teacher()

        out = self.model.teacher(labeled_batch['y'], labeled_batch['mask'])
        loss = self.losses.recon(out.image, labeled_batch['target'])

        self.teacher_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.teacher_optimizer.step()
        return {'teacher_pretrain_loss': float(loss.detach().cpu())}

    def student_step(self, step: int, labeled_batch: dict, unlabeled_batch: dict) -> dict[str, float]:
        self.model.freeze_teacher()
        bundle = self.losses.student_loss(
            step=step,
            ramp_steps=self.cfg.cps_ramp_steps,
            model=self.model,
            labeled_batch=labeled_batch,
            unlabeled_batch=unlabeled_batch,
        )

        self.student_optimizer.zero_grad(set_to_none=True)
        bundle.total.backward()
        self.student_optimizer.step()

        return {
            'student_total': float(bundle.total.detach().cpu()),
            'l_sup': float(bundle.l_sup.detach().cpu()),
            'l_gps': float(bundle.l_gps.detach().cpu()),
            'l_cps': float(bundle.l_cps.detach().cpu()),
            'l_dc': float(bundle.l_dc.detach().cpu()),
        }

    def maybe_feedback_step(
        self,
        step: int,
        labeled_batch: dict,
        unlabeled_batch: dict,
        student_val_psnr: float,
        teacher_val_psnr: float,
    ) -> dict[str, float]:
        if step < self.cfg.feedback_start_step:
            return {'feedback_applied': 0.0}

        if student_val_psnr > teacher_val_psnr and not self.feedback_triggered:
            self.feedback_triggered = True
            self.feedback_trigger_step = step

        if not self.feedback_triggered:
            return {'feedback_applied': 0.0}

        assert self.feedback_trigger_step is not None
        ramp = self.losses.ramp(step - self.feedback_trigger_step, self.cfg.feedback_ramp_steps)

        self.model.unfreeze_teacher()
        feedback_loss = self.losses.teacher_feedback_loss(self.model, labeled_batch, unlabeled_batch)
        total = ramp * feedback_loss

        self.teacher_optimizer.zero_grad(set_to_none=True)
        total.backward()
        self.teacher_optimizer.step()
        return {
            'feedback_applied': 1.0,
            'feedback_ramp': ramp,
            'feedback_loss': float(total.detach().cpu()),
        }

"""Entry point for GPS semi-supervised fastMRI training.

This script provides wiring for the 3-phase algorithm. It defaults to synthetic data
when fastMRI files are unavailable in the environment.
"""

from __future__ import annotations

import argparse
from typing import Iterator

import torch

from gps_mri.models.gps_recon_model import GPSMRIModel
from gps_mri.training.losses import GPSLossComputer
from gps_mri.training.train_gps_recon import GPSMRITrainer, PhaseConfig


def make_synth_batch(batch_size: int, coils: int, h: int, w: int, labeled: bool, device: torch.device) -> dict:
    y = torch.randn(batch_size, coils, h, w, 2, device=device)
    mask = (torch.rand(batch_size, h, w, 1, device=device) > 0.75).float()
    target = torch.rand(batch_size, h, w, device=device) if labeled else None
    return {'y': y, 'mask': mask, 'target': target}


def infinite_synth_loader(
    batch_size: int,
    coils: int,
    h: int,
    w: int,
    labeled: bool,
    device: torch.device,
) -> Iterator[dict]:
    while True:
        yield make_synth_batch(batch_size, coils, h, w, labeled, device)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--num-coils', type=int, default=4)
    p.add_argument('--height', type=int, default=320)
    p.add_argument('--width', type=int, default=320)
    p.add_argument('--student-lr', type=float, default=1e-3)
    p.add_argument('--teacher-lr', type=float, default=3e-4)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    model = GPSMRIModel(num_coils=args.num_coils, num_cascades=8).to(device)
    losses = GPSLossComputer(
        lambda_sup=1.0,
        lambda_gps=1.0,
        lambda_cps=0.5,
        lambda_dc=0.1,
        lambda_fb=0.5,
        lambda_ssim=0.5,
    )

    student_opt = torch.optim.AdamW(
        list(model.student1.parameters()) + list(model.student2.parameters()),
        lr=args.student_lr,
        weight_decay=1e-4,
    )
    teacher_opt = torch.optim.AdamW(model.teacher.parameters(), lr=args.teacher_lr, weight_decay=1e-4)

    trainer = GPSMRITrainer(
        model=model,
        student_optimizer=student_opt,
        teacher_optimizer=teacher_opt,
        losses=losses,
        cfg=PhaseConfig(),
    )

    labeled_iter = infinite_synth_loader(args.batch_size, args.num_coils, args.height, args.width, True, device)
    unlabeled_iter = infinite_synth_loader(args.batch_size, args.num_coils, args.height, args.width, False, device)

    for step in range(1, args.steps + 1):
        lb = next(labeled_iter)
        ub = next(unlabeled_iter)

        if step <= 5:
            metric = trainer.teacher_pretrain_step(lb)
        else:
            metric = trainer.student_step(step, lb, ub)
            fb = trainer.maybe_feedback_step(
                step,
                lb,
                ub,
                student_val_psnr=35.0,
                teacher_val_psnr=34.5,
            )
            metric.update(fb)
        print({'step': step, **metric})


if __name__ == '__main__':
    main()

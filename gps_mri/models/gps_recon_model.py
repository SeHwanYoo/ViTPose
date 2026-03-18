"""Teacher + students wrapper for GPS MRI reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .varnet import E2EVarNet, VarNetOutput


@dataclass
class GPSReconOutput:
    teacher: VarNetOutput
    student1: VarNetOutput
    student2: VarNetOutput


class GPSMRIModel(nn.Module):
    def __init__(self, num_coils: int = 4, num_cascades: int = 8) -> None:
        super().__init__()
        self.teacher = E2EVarNet(num_coils=num_coils, num_cascades=num_cascades)
        self.student1 = E2EVarNet(num_coils=num_coils, num_cascades=num_cascades)
        self.student2 = E2EVarNet(num_coils=num_coils, num_cascades=num_cascades)

    @torch.no_grad()
    def preload_teacher_to_students(self) -> None:
        t_state = self.teacher.state_dict()
        self.student1.load_state_dict(t_state, strict=True)
        self.student2.load_state_dict(t_state, strict=True)

    def freeze_teacher(self) -> None:
        for p in self.teacher.parameters():
            p.requires_grad = False

    def unfreeze_teacher(self) -> None:
        for p in self.teacher.parameters():
            p.requires_grad = True

    def forward(self, y: torch.Tensor, mask: torch.Tensor) -> GPSReconOutput:
        with torch.set_grad_enabled(any(p.requires_grad for p in self.teacher.parameters())):
            t = self.teacher(y, mask)
        s1 = self.student1(y, mask)
        s2 = self.student2(y, mask)
        return GPSReconOutput(teacher=t, student1=s1, student2=s2)

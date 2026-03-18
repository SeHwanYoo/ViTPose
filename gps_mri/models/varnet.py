"""E2E-VarNet style multi-coil MRI reconstructor."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from gps_mri.data.transforms import fft2c, ifft2c, rss_combine, to_complex, to_ri
from .sensitivity_model import UNet2d, SensitivityModel


Tensor = torch.Tensor


def complex_mul(x: Tensor, y: Tensor) -> Tensor:
    return to_ri(to_complex(x) * to_complex(y))


def complex_conj(x: Tensor) -> Tensor:
    return to_ri(torch.conj(to_complex(x)))


def expand_to_coils(img: Tensor, smaps: Tensor) -> Tensor:
    # img: (B, H, W, 2), smaps: (B, C, H, W, 2)
    return complex_mul(img.unsqueeze(1), smaps)


def reduce_from_coils(coil_img: Tensor, smaps: Tensor) -> Tensor:
    # coil_img: (B, C, H, W, 2), smaps: (B, C, H, W, 2)
    prod = complex_mul(coil_img, complex_conj(smaps))
    return prod.sum(dim=1)


@dataclass
class VarNetOutput:
    image: Tensor  # (B, H, W)
    image_ri: Tensor  # (B, H, W, 2)
    smaps: Tensor  # (B, C, H, W, 2)


class VarNetCascade(nn.Module):
    def __init__(self, base_ch: int = 18, levels: int = 4):
        super().__init__()
        self.refine = UNet2d(in_ch=2, out_ch=2, base_ch=base_ch, levels=levels)
        self.dc_lambda = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, H, W, 2)
        h = x.permute(0, 3, 1, 2)
        h = self.refine(h)
        return x + h.permute(0, 2, 3, 1)


class E2EVarNet(nn.Module):
    """Multi-coil unrolled MRI reconstruction with soft data consistency."""

    def __init__(
        self,
        num_coils: int = 4,
        num_cascades: int = 8,
        sens_base_ch: int = 24,
        recon_base_ch: int = 18,
        levels: int = 4,
    ) -> None:
        super().__init__()
        self.num_coils = num_coils
        self.num_cascades = num_cascades
        self.sens_net = SensitivityModel(in_ch=num_coils * 2, out_ch=num_coils * 2, base_ch=sens_base_ch, levels=levels)
        self.cascades = nn.ModuleList([VarNetCascade(recon_base_ch, levels) for _ in range(num_cascades)])

    def _acs_mask(self, mask: Tensor) -> Tensor:
        # mask: (B, 1/H?, W?, 1)
        w = mask.shape[-2]
        acs = int(round(0.08 * w))
        left = (w - acs) // 2
        right = left + acs
        acs_mask = torch.zeros_like(mask)
        acs_mask[..., left:right, :] = 1.0
        return acs_mask

    def forward_operator(self, img_ri: Tensor, smaps: Tensor, mask: Tensor) -> Tensor:
        coil_img = expand_to_coils(img_ri, smaps)
        kspace = fft2c(coil_img)
        return kspace * mask.unsqueeze(1)

    def adjoint_operator(self, kspace: Tensor, smaps: Tensor, mask: Tensor) -> Tensor:
        coil_img = ifft2c(kspace * mask.unsqueeze(1))
        return reduce_from_coils(coil_img, smaps)

    def estimate_smaps(self, y: Tensor, mask: Tensor) -> Tensor:
        acs_mask = self._acs_mask(mask)
        acs_k = y * acs_mask.unsqueeze(1)
        acs_img = ifft2c(acs_k)
        return self.sens_net(acs_img)

    def forward(self, y: Tensor, mask: Tensor) -> VarNetOutput:
        """Args:
        y: (B, C, H, W, 2), mask: (B, H, W, 1) or (B, 1, H, W, 1)
        """
        if mask.ndim == 5:
            mask = mask[:, 0]
        smaps = self.estimate_smaps(y, mask)
        x = self.adjoint_operator(y, smaps, mask)

        for cascade in self.cascades:
            x_pred = cascade(x)
            ah_y = self.adjoint_operator(y, smaps, mask)
            lam = torch.sigmoid(cascade.dc_lambda)
            x = (1.0 - lam) * x_pred + lam * ah_y

        img = rss_combine(expand_to_coils(x, smaps), coil_dim=1)
        return VarNetOutput(image=img, image_ri=x, smaps=smaps)

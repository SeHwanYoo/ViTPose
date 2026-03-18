"""Sensitivity map estimator for multi-coil MRI."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, base_ch: int, levels: int = 4):
        super().__init__()
        chs = [base_ch * (2**i) for i in range(levels)]
        self.down_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        c_in = in_ch
        for c in chs:
            self.down_blocks.append(ConvBlock(c_in, c))
            self.pools.append(nn.AvgPool2d(2))
            c_in = c

        self.bottom = ConvBlock(chs[-1], chs[-1] * 2)
        self.upconvs = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        c_in = chs[-1] * 2
        for c in reversed(chs):
            self.upconvs.append(nn.ConvTranspose2d(c_in, c, kernel_size=2, stride=2))
            self.up_blocks.append(ConvBlock(c * 2, c))
            c_in = c

        self.out = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for block, pool in zip(self.down_blocks, self.pools):
            x = block(x)
            skips.append(x)
            x = pool(x)

        x = self.bottom(x)

        for upconv, block, skip in zip(self.upconvs, self.up_blocks, reversed(skips)):
            x = upconv(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = block(x)
        return self.out(x)


class SensitivityModel(nn.Module):
    """Estimate sensitivity maps from low-frequency ACS data.

    Input/Output real-imag format: (B, coils, H, W, 2)
    """

    def __init__(self, in_ch: int = 8, out_ch: int = 8, base_ch: int = 24, levels: int = 4):
        super().__init__()
        self.unet = UNet2d(in_ch, out_ch, base_ch, levels)

    def forward(self, acs_images: torch.Tensor) -> torch.Tensor:
        b, c, h, w, ri = acs_images.shape
        if ri != 2:
            raise ValueError('Expected real/imag at last dimension = 2.')
        x = acs_images.permute(0, 1, 4, 2, 3).reshape(b, c * 2, h, w)
        out = self.unet(x)
        out = out.reshape(b, c, 2, h, w).permute(0, 1, 3, 4, 2)

        # Normalize along coil dimension so sum |S|^2 ~= 1 per pixel.
        smaps = torch.view_as_complex(out.contiguous())
        denom = (smaps.abs().pow(2).sum(dim=1, keepdim=True) + 1e-8).sqrt()
        smaps = smaps / denom
        return torch.view_as_real(smaps)

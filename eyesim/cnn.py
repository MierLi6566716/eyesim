"""
PupilNet: the small network that reads an eye photo and outputs the prescription.

Input: the two LED frames (64 x 64 each) plus the rig numbers the photo can't
show (distance, LED offset, pupil size, lens opening). Output: M, J0, J45 in
dioptres. A stem conv, four depthwise-separable conv blocks that each halve the
image, average pooling, then two dense layers. About 150k parameters, small
enough to run on a Pi.
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class _DepthwiseBlock(nn.Module):
    """Depthwise-separable conv + BN + ReLU."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


def normalize_frames(x: torch.Tensor) -> torch.Tensor:
    """
    Remove exposure from a batch of frames (B, C, H, W) in [0, 1].

    Both LED channels of a sample are divided by one shared scale (their joint
    98th percentile), so the brightness RATIO between the two LEDs, which
    carries the astigmatism, is kept. The scale has a floor so that a frame
    with no reflex (the instrument's dark zone) is not blown up into noise.
    """
    flat = x.flatten(1)
    k = max(1, int(0.02 * flat.shape[1]))
    scale = flat.topk(k, dim=1).values[:, -1].clamp_min(0.08)
    return (x / scale[:, None, None, None]).clamp(0.0, 2.0)


class PupilNet(nn.Module):
    """
    Predict (M, J0, J45) in dioptres from a 2-channel photorefraction image.

    Input:  (B, 2, H, W)  where H = W = image_size (default 64)
    Output: (B, 3)        [M, J0, J45] in dioptres

    To convert back to clinical notation:
        from eyesim.optimize import power_vector_to_prescription
        S, C, axis = power_vector_to_prescription(M, J0, J45)

    aux_dim > 0 adds the rig numbers as a second input (see
    eyesim.photorefraction.aux_vector). The photo alone can't tell a strong
    prescription far away from a weak one up close, so the network needs them.
    input_norm applies ``normalize_frames`` inside the model.
    """

    def __init__(self, image_size: int = 64, in_channels: int = 2,
                 aux_dim: int = 0, input_norm: bool = False):
        super().__init__()
        self.aux_dim = aux_dim
        self.input_norm = input_norm
        # Stem: standard conv to avoid depthwise on 2-ch input
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # Four depthwise blocks, each halving spatial resolution
        self.blocks = nn.Sequential(
            _DepthwiseBlock(32,  64,  stride=2),
            _DepthwiseBlock(64,  128, stride=2),
            _DepthwiseBlock(128, 256, stride=2),
            _DepthwiseBlock(256, 256, stride=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        # Regression head
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 + (32 if aux_dim else 0), 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 3),
        )
        if aux_dim:
            self.aux_net = nn.Sequential(nn.Linear(aux_dim, 32), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor, aux: torch.Tensor | None = None) -> torch.Tensor:
        if self.input_norm:
            x = normalize_frames(x)
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        if self.aux_dim:
            if aux is None:
                raise ValueError("this PupilNet was built with aux_dim > 0 and needs the geometry input")
            x = torch.cat([x, self.aux_net(aux)], dim=1)
        return self.head(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

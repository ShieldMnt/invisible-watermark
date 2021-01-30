from random import randint, random

import torch
import torch_dct as dct
from torch import nn


class Crop(nn.Module):
    """
    Randomly crops the two spatial dimensions independently to a new size
    that is between `min_pct` and `max_pct` of the old size.

    Input: (N, 3, L, H, W)
    Output: (N, 3, L, H', W')
    """

    def __init__(self, min_pct=0.8, max_pct=1.0):
        super(Crop, self).__init__()
        self.min_pct = min_pct
        self.max_pct = max_pct

    def _pct(self):
        return self.min_pct + random() * (self.max_pct - self.min_pct)

    def forward(self, frames):
        _, _, _, height, width = frames.size()
        dx = int(self._pct() * width)
        dy = int(self._pct() * height)
        dx, dy = (dx // 4) * 4, (dy // 4) * 4
        x = randint(0, width - dx - 1)
        y = randint(0, height - dy - 1)
        return frames[:, :, :, y:y + dy, x:x + dx]


class Scale(nn.Module):
    """
    Randomly scales the two spatial dimensions independently to a new size
    that is between `min_pct` and `max_pct` of the old size.

    Input: (N, 3, L, H, W)
    Output: (N, 3, L, H, W)
    """

    def __init__(self, min_pct=0.8, max_pct=1.0):
        super(Scale, self).__init__()
        self.min_pct = min_pct
        self.max_pct = max_pct

    def _pct(self):
        return self.min_pct + random() * (self.max_pct - self.min_pct)

    def forward(self, frames):
        percent = self._pct()
        _, _, depth, height, width = frames.size()
        height, width = int(percent * height), int(percent * width)
        height, width = (height // 4) * 4, (width // 4) * 4
        return nn.AdaptiveAvgPool3d((depth, height, width))(frames)


class Compression(nn.Module):
    """
    This uses the DCT to produce a differentiable approximation of JPEG compression.

    Input: (N, 3, L, H, W)
    Output: (N, 3, L, H, W)
    """

    def __init__(self, yuv=False, min_pct=0.0, max_pct=0.5):
        super(Compression, self).__init__()
        self.yuv = yuv
        self.min_pct = min_pct
        self.max_pct = max_pct

    def forward(self, y):
        N, _, L, H, W = y.size()

        L = int(y.size(2) * (random() * (self.max_pct - self.min_pct) + self.min_pct))
        H = int(y.size(3) * (random() * (self.max_pct - self.min_pct) + self.min_pct))
        W = int(y.size(4) * (random() * (self.max_pct - self.min_pct) + self.min_pct))

        if self.yuv:
            y = torch.stack([
                (0.299 * y[:, 2, :, :, :] +
                 0.587 * y[:, 1, :, :, :] +
                 0.114 * y[:, 0, :, :, :]),
                (- 0.168736 * y[:, 2, :, :, :] -
                 0.331264 * y[:, 1, :, :, :] +
                 0.500 * y[:, 0, :, :, :]),
                (0.500 * y[:, 2, :, :, :] -
                 0.418688 * y[:, 1, :, :, :] -
                 0.081312 * y[:, 0, :, :, :]),
            ], dim=1)

        y = dct.dct_3d(y)

        if L > 0:
            y[:, :, -L:, :, :] = 0.0

        if H > 0:
            y[:, :, :, -H:, :] = 0.0

        if W > 0:
            y[:, :, :, :, -W:] = 0.0

        y = dct.idct_3d(y)

        if self.yuv:
            y = torch.stack([
                (1.0 * y[:, 0, :, :, :] +
                 1.772 * y[:, 1, :, :, :] +
                 0.000 * y[:, 2, :, :, :]),
                (1.0 * y[:, 0, :, :, :] -
                 0.344136 * y[:, 1, :, :, :] -
                 0.714136 * y[:, 2, :, :, :]),
                (1.0 * y[:, 0, :, :, :] +
                 0.000 * y[:, 1, :, :, :] +
                 1.402 * y[:, 2, :, :, :]),
            ], dim=1)

        return y

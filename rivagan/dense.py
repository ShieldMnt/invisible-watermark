import torch
from torch import nn


def spatial_repeat(x, data):
    """
    This function takes a 5d tensor (with the same shape and dimension order
    as the input to Conv3d) and a 2d data tensor. For each element in the
    batch, the data vector is replicated spatially/temporally and concatenated
    to the channel dimension.

    Input: (N, C_{in}, L, H, W), (N, D)
    Output: (N, C_{in} + D, L, H, W)
    """
    N, D = data.size()
    N, _, L, H, W = x.size()
    x = torch.cat([
        x,
        data.view(N, D, 1, 1, 1).expand(N, D, L, H, W)
    ], dim=1)
    return x


class DenseEncoder(nn.Module):
    """
    Input: (N, 3, L, H, W), (N, D, )
    Output: (N, 3, L, H, W)
    """

    def __init__(self, data_dim, tie_rgb=False, linf_max=0.016,
                 kernel_size=(1, 5, 5), padding=(0, 2, 2)):
        super(DenseEncoder, self).__init__()
        self.linf_max = linf_max
        self.data_dim = data_dim
        self._conv1 = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=kernel_size, padding=padding, stride=1),
            nn.Tanh(),
            nn.BatchNorm3d(32),
        )
        self._conv2 = nn.Sequential(
            nn.Conv3d(32 + data_dim, 64, kernel_size=kernel_size, padding=padding, stride=1),
            nn.Tanh(),
            nn.BatchNorm3d(64),
        )
        self._conv3 = nn.Sequential(
            nn.Conv3d(64 + data_dim, 128, kernel_size=kernel_size, padding=padding, stride=1),
            nn.Tanh(),
            nn.BatchNorm3d(128),
        )
        self._conv4 = nn.Sequential(
            nn.Conv3d(128, 1 if tie_rgb else 3, kernel_size=kernel_size,
                      padding=padding, stride=1),
            nn.Tanh(),
        )

    def forward(self, frames, data):
        data = data * 2.0 - 1.0
        x = self._conv1(frames)
        x = self._conv2(spatial_repeat(x, data))
        x = self._conv3(spatial_repeat(x, data))
        x = self._conv4(x)
        return frames + self.linf_max * x


class DenseDecoder(nn.Module):
    """
    Input: (N, 3, L, H, W)
    Output: (N, D)
    """

    def __init__(self, data_dim, kernel_size=(1, 5, 5)):
        super(DenseDecoder, self).__init__()
        self.data_dim = data_dim
        self._conv = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=kernel_size),
            nn.Tanh(),
            nn.BatchNorm3d(32),
            nn.Conv3d(32, 64, kernel_size=kernel_size),
            nn.Tanh(),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 128, kernel_size=kernel_size),
            nn.Tanh(),
            nn.BatchNorm3d(128),
        )
        self._pool = nn.MaxPool3d(kernel_size=kernel_size)
        self._1x1 = nn.Conv3d(256, self.data_dim, kernel_size=(1, 1, 1))

    def forward(self, frames):
        frames = self._conv(frames)
        frames = torch.cat([self._pool(frames), -self._pool(-frames)], dim=1)
        frames = self._1x1(frames)
        return torch.mean(frames.view(frames.size(0), self.data_dim, -1), dim=2)

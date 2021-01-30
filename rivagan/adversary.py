import torch
from torch import nn


class Critic(nn.Module):
    """
    The Critic module maps a video to a scalar score. It takes in a batch of N videos - each
    of which is of length L, height H, and width W - and produces a score for each video which
    corresponds to how "realistic" it looks.

    Input: (N, 3, L, H, W)
    Output: (N, 1)
    """

    def __init__(self, kernel_size=(1, 3, 3), padding=(0, 0, 0)):
        super(Critic, self).__init__()
        self._conv = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=kernel_size, padding=padding, stride=2),
            nn.Tanh(),
            nn.BatchNorm3d(16),
            nn.Conv3d(16, 32, kernel_size=kernel_size, padding=padding, stride=2),
            nn.Tanh(),
            nn.BatchNorm3d(32),
            nn.Conv3d(32, 64, kernel_size=kernel_size, padding=padding, stride=2),
        )
        self._linear = nn.Linear(64, 1)

    def forward(self, frames):
        frames = self._conv(frames)
        N, _, L, H, W = frames.size()
        return self._linear(torch.mean(frames.view(N, -1, L * H * W), dim=2))


class Adversary(nn.Module):
    """
    The Adversary module maps a sequence of frames to another sequence of frames
    with a constraint on the maximum distortion of each individual pixel.

    Input: (N, 3, L, H, W)
    Output: (N, 3, L, H, W)
    """

    def __init__(self, l1_max=0.05, kernel_size=(1, 3, 3), padding=(0, 1, 1)):
        super(Adversary, self).__init__()
        self.l1_max = l1_max
        self._conv = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=kernel_size, padding=padding),
            nn.Tanh(),
            nn.BatchNorm3d(16),
            nn.Conv3d(16, 3, kernel_size=kernel_size, padding=padding),
            nn.Tanh(),
        )

    def forward(self, frames):
        x = frames
        x = self._conv(x)
        return frames + self.l1_max * x

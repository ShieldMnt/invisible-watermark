from math import exp
from tempfile import NamedTemporaryFile

import cv2
import torch
from torch.nn.functional import conv2d


def gaussian(window_size, sigma):
    """Gaussian window.

    https://en.wikipedia.org/wiki/Window_function#Gaussian_window
    """
    _exp = [exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)]
    gauss = torch.Tensor(_exp)
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):

    padding_size = window_size // 2

    mu1 = conv2d(img1, window, padding=padding_size, groups=channel)
    mu2 = conv2d(img2, window, padding=padding_size, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = conv2d(img1 * img1, window, padding=padding_size, groups=channel) - mu1_sq
    sigma2_sq = conv2d(img2 * img2, window, padding=padding_size, groups=channel) - mu2_sq
    sigma12 = conv2d(img1 * img2, window, padding=padding_size, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    _ssim_quotient = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))
    _ssim_divident = ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    ssim_map = _ssim_quotient / _ssim_divident

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(2.0 / torch.sqrt(mse))


def mjpeg(x):
    """
    Write each video to disk and re-read it from disk.

    Input: (N, 3, L, H, W)
    Output: (N, 3, L, H, W)
    """
    y = torch.zeros(x.size())
    _, _, _, height, width = x.size()

    for n in range(x.size(0)):
        tempfile = NamedTemporaryFile(suffix=".avi")

        vout = cv2.VideoWriter(tempfile.name, cv2.VideoWriter_fourcc(*'MJPG'),
                               20.0, (width, height))
        for l in range(x.size(2)):
            image = x[n, :, l, :, :]  # (3, H, W)
            image = torch.clamp(image.permute(1, 2, 0), min=-1.0, max=1.0)
            vout.write(((image + 1.0) * 127.5).detach().cpu().numpy().astype("uint8"))
        vout.release()

        vin = cv2.VideoCapture(tempfile.name)
        for l in range(x.size(2)):
            _, frame = vin.read()  # (H, W, 3)
            frame = torch.tensor(frame / 127.5 - 1.0)
            y[n, :, l, :, :] = frame.permute(2, 0, 1)

        tempfile.close()
    return y.to(x.device)

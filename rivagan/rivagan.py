import gc
import json
import os
import random
import time
from itertools import chain

import cv2
import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.nn import functional
from tqdm import tqdm

from .adversary import Adversary, Critic
from .attention import AttentiveDecoder, AttentiveEncoder
from .dataloader import load_train_val
from .dense import DenseDecoder, DenseEncoder
from .noise import Compression, Crop, Scale
from .utils import mjpeg, psnr, ssim


def get_acc(y_true, y_pred):
    assert y_true.size() == y_pred.size()
    return (y_pred >= 0.0).eq(y_true >= 0.5).sum().float().item() / y_pred.numel()


def quantize(frames):
    # [-1.0, 1.0] -> {0, 255} -> [-1.0, 1.0]
    return ((frames + 1.0) * 127.5).int().float() / 127.5 - 1.0


def make_pair(frames, data_dim, use_bit_inverse=True, multiplicity=1):
    # Add multiplicity to further stabilize training.
    frames = torch.cat([frames] * multiplicity, dim=0).cuda()
    data = torch.zeros((frames.size(0), data_dim)).random_(0, 2).cuda()

    # Add the bit-inverse to stabilize training.
    if use_bit_inverse:
        frames = torch.cat([frames, frames], dim=0).cuda()
        data = torch.cat([data, 1.0 - data], dim=0).cuda()

    return frames, data


class RivaGAN(object):
    def __init__(self, model="attention", data_dim=32):
        self.model = model
        self.data_dim = data_dim
        self.adversary = Adversary().cuda()
        self.critic = Critic().cuda()
        if model == "attention":
            self.encoder = AttentiveEncoder(data_dim=data_dim).cuda()
            self.decoder = AttentiveDecoder(self.encoder).cuda()
        elif model == "dense":
            self.encoder = DenseEncoder(data_dim=data_dim).cuda()
            self.decoder = DenseDecoder(data_dim=data_dim).cuda()
        else:
            raise ValueError("Unknown model: %s" % model)

    def fit(self, dataset, log_dir=False,
            seq_len=1, batch_size=12, lr=5e-4,
            use_critic=False, use_adversary=False,
            epochs=300, use_bit_inverse=True, use_noise=True):

        if not log_dir:
            log_dir = "experiments/%s-%s" % (self.model, str(int(time.time())))
        os.makedirs(log_dir, exist_ok=False)

        # Set up the noise layers
        crop = Crop()
        scale = Scale()
        compress = Compression()

        def noise(frames):
            if use_noise:
                if random.random() < 0.5:
                    frames = crop(frames)
                if random.random() < 0.5:
                    frames = scale(frames)
                if random.random() < 0.5:
                    frames = compress(frames)

            return frames

        # Set up the data and optimizers
        train, val = load_train_val(seq_len, batch_size, dataset)
        G_opt = optim.Adam(chain(self.encoder.parameters(), self.decoder.parameters()), lr=lr)
        G_scheduler = optim.lr_scheduler.ReduceLROnPlateau(G_opt)
        D_opt = optim.Adam(chain(self.adversary.parameters(), self.critic.parameters()), lr=lr)
        # D_scheduler = optim.lr_scheduler.ReduceLROnPlateau(D_opt)

        # Set up the log directory
        with open(os.path.join(log_dir, "config.json"), "wt") as fout:
            fout.write(json.dumps({
                "model": self.model,
                "data_dim": self.data_dim,
                "seq_len": seq_len,
                "batch_size": batch_size,
                "dataset": dataset,
                "lr": lr,
                "log_dir": log_dir,
            }, indent=2, default=lambda o: str(o)))

        # Optimize the model
        history = []
        for epoch in range(1, epochs + 1):
            metrics = {
                "train.loss": [],
                "train.raw_acc": [],
                "train.mjpeg_acc": [],
                "train.adv_loss": [],
                "val.ssim": [],
                "val.psnr": [],
                "val.crop_acc": [],
                "val.scale_acc": [],
                "val.mjpeg_acc": [],
            }

            gc.collect()
            self.encoder.train()
            self.decoder.train()

            # Optimize critic-adversary
            if use_critic or use_adversary:
                iterator = tqdm(train, ncols=0)
                for frames in iterator:
                    frames, data = make_pair(frames, self.data_dim,
                                             use_bit_inverse=use_bit_inverse)
                    wm_frames = self.encoder(frames, data)
                    adv_loss = 0.0
                    if use_critic:
                        adv_loss += torch.mean(self.critic(frames) - self.critic(wm_frames))
                    if use_adversary:
                        adv_loss -= functional.binary_cross_entropy_with_logits(
                            self.decoder(self.adversary(wm_frames)), data)
                    D_opt.zero_grad()
                    adv_loss.backward()
                    D_opt.step()
                    for p in self.critic.parameters():
                        p.data.clamp_(-0.1, 0.1)

                    metrics["train.adv_loss"].append(adv_loss.item())
                    iterator.set_description("Adversary | %s" % np.mean(metrics["train.adv_loss"]))

            # Optimize encoder-decoder using critic-adversary
            if use_critic or use_adversary:
                iterator = tqdm(train, ncols=0)
                for frames in iterator:
                    frames, data = make_pair(frames, self.data_dim,
                                             use_bit_inverse=use_bit_inverse)
                    wm_frames = self.encoder(frames, data)
                    loss = 0.0
                    if use_critic:
                        critic_loss = torch.mean(self.critic(wm_frames))
                        loss += 0.1 * critic_loss
                    if use_adversary:
                        adversary_loss = functional.binary_cross_entropy_with_logits(
                            self.decoder(self.adversary(wm_frames)), data)
                        loss += 0.1 * adversary_loss
                    G_opt.zero_grad()
                    loss.backward()
                    G_opt.step()

            # Optimize encoder-decoder
            iterator = tqdm(train, ncols=0)
            for frames in iterator:
                frames, data = make_pair(frames, self.data_dim, use_bit_inverse=use_bit_inverse)

                wm_frames = self.encoder(frames, data)
                wm_raw_data = self.decoder(noise(wm_frames))
                wm_mjpeg_data = self.decoder(mjpeg(wm_frames))

                loss = 0.0
                loss += functional.binary_cross_entropy_with_logits(wm_raw_data, data)
                loss += functional.binary_cross_entropy_with_logits(wm_mjpeg_data, data)
                G_opt.zero_grad()
                loss.backward()
                G_opt.step()

                metrics["train.loss"].append(loss.item())
                metrics["train.raw_acc"].append(get_acc(data, wm_raw_data))
                metrics["train.mjpeg_acc"].append(get_acc(data, wm_mjpeg_data))
                iterator.set_description("%s | Loss %.3f | Raw %.3f | MJPEG %.3f" % (
                    epoch,
                    np.mean(metrics["train.loss"]),
                    np.mean(metrics["train.raw_acc"]),
                    np.mean(metrics["train.mjpeg_acc"]),
                ))

            # Validate
            gc.collect()
            self.encoder.eval()
            self.decoder.eval()
            iterator = tqdm(val, ncols=0)
            with torch.no_grad():
                for frames in iterator:
                    frames = frames.cuda()
                    data = torch.zeros((frames.size(0), self.data_dim)).random_(0, 2).cuda()

                    wm_frames = self.encoder(frames, data)
                    wm_crop_data = self.decoder(mjpeg(crop(wm_frames)))
                    wm_scale_data = self.decoder(mjpeg(scale(wm_frames)))
                    wm_mjpeg_data = self.decoder(mjpeg(wm_frames))

                    metrics["val.ssim"].append(
                        ssim(frames[:, :, 0, :, :], wm_frames[:, :, 0, :, :]).item())
                    metrics["val.psnr"].append(
                        psnr(frames[:, :, 0, :, :], wm_frames[:, :, 0, :, :]).item())
                    metrics["val.crop_acc"].append(get_acc(data, wm_crop_data))
                    metrics["val.scale_acc"].append(get_acc(data, wm_scale_data))
                    metrics["val.mjpeg_acc"].append(get_acc(data, wm_mjpeg_data))

                    iterator.set_description(
                        "%s | SSIM %.3f | PSNR %.3f | Crop %.3f | Scale %.3f | MJPEG %.3f" % (
                            epoch,
                            np.mean(metrics["val.ssim"]),
                            np.mean(metrics["val.psnr"]),
                            np.mean(metrics["val.crop_acc"]),
                            np.mean(metrics["val.scale_acc"]),
                            np.mean(metrics["val.mjpeg_acc"]),
                        )
                    )

            metrics = {
                k: round(np.mean(v), 3) if len(v) > 0 else "NaN"
                for k, v in metrics.items()
            }
            metrics["epoch"] = epoch
            history.append(metrics)
            pd.DataFrame(history).to_csv(
                os.path.join(log_dir, "metrics.tsv"), index=False, sep="\t")
            with open(os.path.join(log_dir, "metrics.json"), "wt") as fout:
                fout.write(json.dumps(metrics, indent=2, default=lambda o: str(o)))

            torch.save(self, os.path.join(log_dir, "model.pt"))
            G_scheduler.step(metrics["train.loss"])

        return history

    def save(self, path_to_model):
        torch.save(self, path_to_model)

    def load(path_to_model):
        return torch.load(path_to_model)

    def encode(self, video_in, data, video_out):
        assert len(data) == self.data_dim

        video_in = cv2.VideoCapture(video_in)
        width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
        length = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))

        data = torch.FloatTensor([data]).cuda()
        video_out = cv2.VideoWriter(
            video_out, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))

        for i in tqdm(range(length)):
            ok, frame = video_in.read()
            frame = torch.FloatTensor([frame]) / 127.5 - 1.0      # (L, H, W, 3)
            frame = frame.permute(3, 0, 1, 2).unsqueeze(0).cuda()  # (1, 3, L, H, W)
            wm_frame = self.encoder(frame, data)                       # (1, 3, L, H, W)
            wm_frame = torch.clamp(wm_frame, min=-1.0, max=1.0)
            wm_frame = (
                (wm_frame[0, :, 0, :, :].permute(1, 2, 0) + 1.0) * 127.5
            ).detach().cpu().numpy().astype("uint8")
            video_out.write(wm_frame)

        video_out.release()

    def decode(self, video_in):
        video_in = cv2.VideoCapture(video_in)
        # width = int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH))
        # height = int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
        length = int(video_in.get(cv2.CAP_PROP_FRAME_COUNT))

        for i in tqdm(range(length)):
            ok, frame = video_in.read()
            frame = torch.FloatTensor([frame]) / 127.5 - 1.0      # (L, H, W, 3)
            frame = frame.permute(3, 0, 1, 2).unsqueeze(0).cuda()  # (1, 3, L, H, W)
            data = self.decoder(frame)[0].detach().cpu().numpy()
            yield data

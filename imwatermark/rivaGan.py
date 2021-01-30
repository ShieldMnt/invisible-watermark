import numpy as np
import copy
import torch
import cv2
import pywt
import math
import os
import time
from rivagan import RivaGAN
import pprint

pp = pprint.PrettyPrinter(indent=2)


class RivaWatermark(object):
    rivaGanModel = None
    def __init__(self, watermarks=[], wmLen=32):
        self._watermarks = watermarks
        if wmLen not in [32]:
            raise RuntimeError('rivaGan only supports 32 bits watermarks now.')
        self._data = torch.FloatTensor([self._watermarks]).cuda()

    @classmethod
    def loadModel(cls):
        if RivaWatermark.rivaGanModel:
            return
        modelDir = os.path.dirname(os.path.abspath(__file__))
        rivaModel = os.path.join(modelDir, 'rivaGan.pt')
        RivaWatermark.rivaGanModel = RivaGAN.load(rivaModel)

    def encode(self, frame):
        if not RivaWatermark.rivaGanModel:
            raise RuntimeError('you need load model first')
        frame = torch.FloatTensor([frame]) / 127.5 - 1.0
        frame = frame.permute(3, 0, 1, 2).unsqueeze(0).cuda()
        wm_frame = RivaWatermark.rivaGanModel.encoder(frame, self._data)
        wm_frame = torch.clamp(wm_frame, min=-1.0, max=1.0)
        wm_frame = (
            (wm_frame[0, :, 0, :, :].permute(1, 2, 0) + 1.0) * 127.5
        ).detach().cpu().numpy().astype("uint8")
        return wm_frame

    def decode(self, frame):
        if not RivaWatermark.rivaGanModel:
            raise RuntimeError('you need load model first')
        frame = torch.FloatTensor([frame]) / 127.5 - 1.0
        frame = frame.permute(3, 0, 1, 2).unsqueeze(0).cuda()
        data = RivaWatermark.rivaGanModel.decoder(frame)[0].detach().cpu().numpy()
        return np.array(data > 0.6, dtype=np.uint8)

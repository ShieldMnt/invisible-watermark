import numpy as np
import torch
import cv2
import os
import time


class RivaWatermark(object):
    encoder = None
    decoder = None

    def __init__(self, watermarks=[], wmLen=32, threshold=0.52):
        self._watermarks = watermarks
        self._threshold = threshold
        if wmLen not in [32]:
            raise RuntimeError('rivaGan only supports 32 bits watermarks now.')
        self._data = torch.from_numpy(np.array([self._watermarks], dtype=np.float32))

    @classmethod
    def loadModel(cls):
        try:
            import onnxruntime
        except ImportError:
            raise ImportError(
                "The `RivaWatermark` class requires onnxruntime to be installed. "
                "You can install it with pip: `pip install onnxruntime`."
            )

        if RivaWatermark.encoder and RivaWatermark.decoder:
            return
        # Define the priority order for the execution providers
        # prefer CUDA Execution Provider over CPU Execution Provider.
        EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        modelDir = os.path.dirname(os.path.abspath(__file__))
        RivaWatermark.encoder = onnxruntime.InferenceSession(
            os.path.join(modelDir, 'rivagan_encoder.onnx'), providers=EP_list)
        RivaWatermark.decoder = onnxruntime.InferenceSession(
            os.path.join(modelDir, 'rivagan_decoder.onnx'), providers=EP_list)

    def encode(self, frame):
        if not RivaWatermark.encoder:
            raise RuntimeError('call loadModel method first')

        frame = torch.from_numpy(np.array([frame], dtype=np.float32)) / 127.5 - 1.0
        frame = frame.permute(3, 0, 1, 2).unsqueeze(0)

        inputs = {
            'frame': frame.detach().cpu().numpy(),
            'data': self._data.detach().cpu().numpy()
        }

        outputs = RivaWatermark.encoder.run(None, inputs)
        wm_frame = outputs[0]
        wm_frame = torch.clamp(torch.from_numpy(wm_frame), min=-1.0, max=1.0)
        wm_frame = (
            (wm_frame[0, :, 0, :, :].permute(1, 2, 0) + 1.0) * 127.5
        ).detach().cpu().numpy().astype('uint8')

        return wm_frame

    def decode(self, frame):
        if not RivaWatermark.decoder:
            raise RuntimeError('you need load model first')

        frame = torch.from_numpy(np.array([frame], dtype=np.float32)) / 127.5 - 1.0
        frame = frame.permute(3, 0, 1, 2).unsqueeze(0)
        inputs = {
            'frame': frame.detach().cpu().numpy(),
        }
        outputs = RivaWatermark.decoder.run(None, inputs)
        data = outputs[0][0]
        return np.array(data > self._threshold, dtype=np.uint8)

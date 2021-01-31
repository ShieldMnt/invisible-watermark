import numpy as np
import torch
import cv2
from rivagan import RivaGAN

torch.nn.Module.dump_patches = True

if __name__ == '__main__':
    bgr = cv2.imread('test_vectors/original.jpg')

    watermarks = np.random.randint(0,2,32)
    data = torch.from_numpy(np.array([watermarks], dtype=np.float32))

    frame = torch.from_numpy(np.array([bgr], dtype=np.float32)) / 127.5 - 1.0
    frame = frame.permute(3, 0, 1, 2).unsqueeze(0)

    rivagan = torch.load('rivaGan.pt', map_location=torch.device('cpu'))

    encoder = rivagan.encoder


    torch.onnx.export(encoder, args=(frame, data), f='rivagan_encoder.onnx',
                      export_params=True, opset_version=10, do_constant_folding=True,
                      input_names = ['frame', 'data'],
                      output_names = ['output'],
                      dynamic_axes={
                                    'frame': {
                                        0:'batch_size',
                                        3:'height',
                                        4:'width'
                                    },
                                    'data': {
                                        0:'batch_size',
                                        1:'wmBits'
                                    },
                                    'output': {
                                        0:'batch_size',
                                        3:'height',
                                        4:'width'
                                    }
                                   })

    decoder = rivagan.decoder

    torch.onnx.export(decoder, args=(frame), f='rivagan_decoder.onnx',
                      export_params=True, opset_version=10, do_constant_folding=True,
                      input_names = ['frame'],
                      output_names = ['output'],
                      dynamic_axes={
                                    'frame': {
                                        0:'batch_size',
                                        3:'height',
                                        4:'width'
                                    },
                                    'output': {
                                        0:'batch_size',
                                        1:'wmBits'
                                    }
                                   })

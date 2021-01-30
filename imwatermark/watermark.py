import struct
import uuid
import copy
import base64
import cv2
import numpy as np
from .maxDct import EmbedMaxDct
from .dwtDctSvd import EmbedDwtDctSvd
from .rivaGan import RivaWatermark
import pprint

pp = pprint.PrettyPrinter(indent=2)

class WatermarkEncoder(object):
    def __init__(self, content=b''):
        seq = np.array([n for n in content], dtype=np.uint8)
        self._watermarks = list(np.unpackbits(seq))
        self._wmLen = len(self._watermarks)
        self._wmType = 'bytes'

    def set_by_ipv4(self, addr):
        bits = []
        ips = addr.split('.')
        for ip in ips:
            bits += list(np.unpackbits(np.array([ip % 255], dtype=np.uint8)))
        self._watermarks = bits
        self._wmLen = len(self._watermarks)
        self._wmType = 'ipv4'
        assert self._wmLen == 32

    def set_by_uuid(self, uid):
        u = uuid.UUID(uid)
        self._wmType = 'uuid'
        seq = np.array([n for n in u.bytes], dtype=np.uint8)
        self._watermarks = list(np.unpackbits(seq))
        self._wmLen = len(self._watermarks)

    def set_by_bytes(self, content):
        self._wmType = 'bytes'
        seq = np.array([n for n in content], dtype=np.uint8)
        self._watermarks = list(np.unpackbits(seq))
        self._wmLen = len(self._watermarks)

    def set_by_b16(self, b16):
        content = base64.b16decode(b16)
        self.set_by_bytes(content)
        self._wmType = 'b16'

    def set_by_bits(self, bits=[]):
        self._watermarks = [int(bit) % 2 for bit in bits]
        self._wmLen = len(self._watermarks)
        self._wmType = 'bits'

    def set_watermark(self, wmType='bytes', content=''):
        if wmType == 'ipv4':
            self.set_by_ipv4(content)
        elif wmType == 'uuid':
            self.set_by_uuid(content)
        elif wmType == 'bits':
            self.set_by_bits(content)
        elif wmType == 'bytes':
            self.set_by_bytes(content)
        elif wmType == 'b16':
            self.set_by_b16(content)
        else:
            raise NameError('%s is not supported' % wmType)

    def get_length(self):
        return self._wmLen

    @classmethod
    def loadModel(cls):
        RivaWatermark.loadModel()

    def encode(self, cv2Image, method='dwtDct', **configs):
        (r, c, channels) = cv2Image.shape
        if r*c < 256*256:
            raise RuntimeError('image too small, should be larger than 256x256')
        if method == 'dwtDct':
            embed = EmbedMaxDct(self._watermarks, wmLen=self._wmLen, **configs)
            return embed.encode(cv2Image)
        elif method == 'dwtDctSvd':
            embed = EmbedDwtDctSvd(self._watermarks, wmLen=self._wmLen, **configs)
            return embed.encode(cv2Image)
        elif method == 'rivaGan':
            embed = RivaWatermark(self._watermarks, self._wmLen)
            return embed.encode(cv2Image)
        else:
            raise NameError('%s is not supported' % method)

class WatermarkDecoder(object):
    def __init__(self, wm_type='bytes', length=0):
        self._wmType = wm_type
        if wm_type == 'ipv4':
            self._wmLen = 32
        elif wm_type == 'uuid':
            self._wmLen = 128
        elif wm_type == 'bytes':
            self._wmLen = length
        elif wm_type == 'bits':
            self._wmLen = length
        elif wm_type == 'b16':
            self._wmLen = length
        else:
            raise NameError('%s is unsupported' % wm_type)

    def reconstruct_ipv4(self, bits):
        ips = [str(ip) for ip in list(np.packbits(bits))]
        return '.'.join(ips)

    def reconstruct_uuid(self, bits):
        nums = np.packbits(bits)
        bstr = b''
        for i in range(16):
            bstr += struct.pack('>B', nums[i])

        return str(uuid.UUID(bytes=bstr))

    def reconstruct_bits(self, bits):
        return ''.join([str(b) for b in bits])

    def reconstruct_b16(self, bits):
        bstr = self.reconstruct_bytes(bits)
        return base64.b16encode(bstr)

    def reconstruct_bytes(self, bits):
        nums = np.packbits(bits)
        bstr = b''
        for i in range(self._wmLen//8):
            bstr += struct.pack('>B', nums[i])
        return bstr

    def reconstruct(self, bits):
        if len(bits) != self._wmLen:
            raise RuntimeError('bits are not matched with watermark length')

        if self._wmType == 'ipv4':
            return self.reconstruct_ipv4(bits)
        elif self._wmType == 'uuid':
            return self.reconstruct_uuid(bits)
        elif self._wmType == 'bits':
            return self.reconstruct_bits(bits)
        elif self._wmType == 'b16':
            return self.reconstruct_b16(bits)
        else:
            return self.reconstruct_bytes(bits)

    def decode(self, cv2Image, method='dwtDct', **configs):
        (r, c, channels) = cv2Image.shape
        if r*c < 256*256:
            raise RuntimeError('image too small, should be larger than 256x256')
        bits = []
        if method == 'dwtDct':
            embed = EmbedMaxDct(watermarks=[], wmLen=self._wmLen, **configs)
            bits = embed.decode(cv2Image)
        elif method == 'dwtDctSvd':
            embed = EmbedDwtDctSvd(watermarks=[], wmLen=self._wmLen, **configs)
            bits = embed.decode(cv2Image)
        elif method == 'rivaGan':
            embed = RivaWatermark(watermarks=[], wmLen=self._wmLen, **configs)
            bits = embed.decode(cv2Image)
        else:
            raise NameError('%s is not supported' % method)
        return self.reconstruct(bits)

    @classmethod
    def loadModel(cls):
        RivaWatermark.loadModel()

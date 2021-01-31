# invisible-watermark
invisible-watermark is a **python** library and command line tool for creating invisible watermark over image.(aka. **blink image watermark**, **digital image watermark**). The algorithm doesn't reply on the original image.

Note that this library is still experimental and it doesn't support GPU acceleration, carefully deploy it on the production environment.


supported algorithms:
* [Discrete wavelet transform](https://en.wikipedia.org/wiki/Discrete_wavelet_transform) + [Discrete cosine transform](https://en.wikipedia.org/wiki/Discrete_cosine_transform) frequency embedding algorithm variants.
* [RivaGAN](https://github.com/DAI-Lab/RivaGAN), a deep-learning model trained from Hollywood2 movie clips dataset.


But only the default method (dwtDCT, ~300ms 1080P image) is suitable for on-the-fly embedding. The other methods are too slow on a CPU only environment.

## Supported Algorithms
* **frequency methods**: [algorithm process and test result](https://github.com/ShieldMnt/invisible-watermark/wiki/Frequency-Methods)
* **rivaGan**: encoder/decoder model with Attention mechanism + embed watermark bits into vector.


## How to install
`pip install invisible-watermark`


## [Library API](https://github.com/ShieldMnt/invisible-watermark/wiki/API)
### Embed watermark

* **example** embed 4 characters (32 bits) watermark

```python
import cv2
from imwatermark import WatermarkEncoder

bgr = cv2.imread('test.png')
wm = 'test'

encoder = WatermarkEncoder()
encoder.set_watermark('bytes', wm.encode('utf-8'))
bgr_encoded = encoder.encode(bgr, 'dwtDct')

cv2.imwrite('test_wm.png', bgr_encoded)
```

### Decode watermark
* **example** decode 4 characters (32 bits) watermark

```python
import cv2
from imwatermark import WatermarkDecoder

bgr = cv2.imread('test_wm.png')

decoder = WatermarkDecoder('bytes', 32)
watermark = decoder.decode(bgr, 'dwtDct')
print(watermark.decode('utf-8'))
```


## CLI Usage

```
embed watermark:  ./invisible-watermark -v -a encode -t bytes -m dwtDct -w 'hello' -o ./test_vectors/wm.png ./test_vectors/original.jpg

decode watermark: ./invisible-watermark -v -a decode -t bytes -m dwtDct -l 40 ./test_vectors/wm.png

positional arguments:
  input                 The path of input

optional arguments:
  -h, --help            show this help message and exit
  -a ACTION, --action ACTION
                        encode|decode (default: None)
  -t TYPE, --type TYPE  bytes|b16|bits|uuid|ipv4 (default: bits)
  -m METHOD, --method METHOD
                        dwtDct|dwtDctSvd|rivaGan (default: maxDct)
  -w WATERMARK, --watermark WATERMARK
                        embedded string (default: )
  -l LENGTH, --length LENGTH
                        watermark bits length, required for bytes|b16|bits
                        watermark (default: 0)
  -o OUTPUT, --output OUTPUT
                        The path of output (default: None)
  -v, --verbose         print info (default: False)
```

## Test Result
 
For better doc reading, we compress all images in this page, but the test is taken on 1920x1080 original image.

Methods are not robust to **resize** or aspect ratio changed **crop** but robust to **noise**, **color filter**, **brightness** and **jpg compress.**

**rivaGan overcomes default method on crop attack.**

**only default method is ready for on-the-fly embedding.**

### Input
> * Input Image: 1960x1080 Image
> * Watermark: 
>   - For freq method, we use 64bits, string expression "qingquan"
>   - For RivaGan method, we use 32bits, string expression "qing"
> * Parameters: only take U frame to keep image quality, ```scale=36```

### Attack Performance


**Watermarked Image**

![wm](https://user-images.githubusercontent.com/1647036/106387712-03c17400-6416-11eb-9490-e5e860b025ad.png)

| Attacks | Image | Freq Method | RivaGan |
| --- | --- | --- | --- |
| JPG Compress | ![wm_jpg](https://user-images.githubusercontent.com/1647036/106387721-0e7c0900-6416-11eb-840c-8eab1cb9d748.jpg) | Pass | Pass |
| Noise | ![wm_noise](https://user-images.githubusercontent.com/1647036/106387874-c90c0b80-6416-11eb-99f3-1716f01f2211.png) | Pass | Pass |
| Brightness | ![wm_darken](https://user-images.githubusercontent.com/1647036/106387718-0cb24580-6416-11eb-83af-7f9e94f13cae.png) | Pass | Pass |
| Overlay | ![wm_overlay](https://user-images.githubusercontent.com/1647036/106387733-13d95380-6416-11eb-8aa4-b3d2acfa8637.png) | Pass | Pass |
| Mask | ![wm_mask_large](https://user-images.githubusercontent.com/1647036/106387726-10de6300-6416-11eb-99c3-4a0f70f99224.png) | Pass | Pass |
| crop 7x5 | ![wm_crop_7x5](https://user-images.githubusercontent.com/1647036/106387713-06bc6480-6416-11eb-8ae0-f64289642450.png) | Fail | Pass |
| Resize 50% | ![wm_resize_half](https://user-images.githubusercontent.com/1647036/106387735-15a31700-6416-11eb-8589-2ffa38df2a9a.png) | Fail | Fail |
| Rotate 30 degress | ![wm_rotate](https://user-images.githubusercontent.com/1647036/106387737-19369e00-6416-11eb-8417-05e53e11b77f.png) | Fail | Fail|



### Running Speed (CPU Only)
| Method | Encoding | Decoding |
| --- | --- | --- |
| dwtDct | 300-350ms | 150ms-200ms |
| dwtDctSvd | 1500ms-2s | ~1s |
| rivaGan | ~5s | 4-5s |

### RivaGAN Experimental
Further, We will deliver the 64bit rivaGan model and test the performance on GPU environment.

Detail: [https://github.com/DAI-Lab/RivaGAN](https://github.com/DAI-Lab/RivaGAN)

Zhang, Kevin Alex and Xu, Lei and Cuesta-Infante, Alfredo and Veeramachaneni, Kalyan. Robust Invisible Video Watermarking with Attention. MIT EECS, September 2019.[[PDF](https://arxiv.org/abs/1909.01285)]

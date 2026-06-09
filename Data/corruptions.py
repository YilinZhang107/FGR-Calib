from io import BytesIO
import random

import cv2
import numpy as np
from PIL import Image, ImageFilter
from skimage.filters import gaussian


def _to_pil(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    return Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB")


def jpeg_compression(image, severity=1):
    qualities = [25, 18, 15, 10, 7]
    image = _to_pil(image)
    output = BytesIO()
    image.save(output, "JPEG", quality=qualities[int(severity) - 1])
    output.seek(0)
    return Image.open(output).convert("RGB")


def fft_lowpass(image, severity=1):
    ratios = [0.8, 0.6, 0.4, 0.3, 0.2]
    ratio = ratios[int(severity) - 1]
    array = np.asarray(_to_pil(image)).astype(np.float32)
    rows, cols, _ = array.shape
    center_row, center_col = rows // 2, cols // 2
    radius = max(min(rows, cols) * ratio / 2.0, 1.0)
    y, x = np.ogrid[:rows, :cols]
    distance = np.sqrt((x - center_col) ** 2 + (y - center_row) ** 2)
    mask = np.exp(-(distance**2) / (2.0 * (radius / 1.5) ** 2))

    result = np.zeros_like(array)
    for channel in range(3):
        dft = np.fft.fft2(array[:, :, channel])
        filtered = np.fft.fftshift(dft) * mask
        result[:, :, channel] = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered)))
    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def gaussian_blur(image, severity=1):
    sigma = [1, 2, 3, 4, 6][int(severity) - 1]
    array = np.asarray(_to_pil(image)).astype(np.float32) / 255.0
    try:
        blurred = gaussian(array, sigma=sigma, channel_axis=-1)
    except TypeError:
        blurred = cv2.GaussianBlur(array, (int(sigma * 2) + 1, int(sigma * 2) + 1), sigma)
    return Image.fromarray(np.clip(blurred * 255.0, 0, 255).astype(np.uint8))


def pixelate(image, severity=1):
    scale = [0.6, 0.5, 0.4, 0.3, 0.25][int(severity) - 1]
    image = _to_pil(image)
    width, height = image.size
    small = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.BOX)
    return small.resize((width, height), Image.Resampling.BOX)


def brightness(image, severity=1):
    delta = [0.1, 0.2, 0.3, 0.4, 0.5][int(severity) - 1]
    array = np.asarray(_to_pil(image)).astype(np.float32) / 255.0
    return Image.fromarray(np.clip((array + delta) * 255.0, 0, 255).astype(np.uint8))


def contrast(image, severity=1):
    factor = [0.4, 0.3, 0.2, 0.1, 0.05][int(severity) - 1]
    array = np.asarray(_to_pil(image)).astype(np.float32) / 255.0
    mean = np.mean(array, axis=(0, 1), keepdims=True)
    return Image.fromarray(np.clip((mean + factor * (array - mean)) * 255.0, 0, 255).astype(np.uint8))


CORRUPTION_FUNCTIONS = {
    "jpeg_compression": jpeg_compression,
    "fft_lowpass": fft_lowpass,
    "gaussian_blur": gaussian_blur,
    "pixelate": pixelate,
    "brightness": brightness,
    "contrast": contrast,
}


def apply_corruption(image, corruption_type, severity=1):
    if corruption_type not in CORRUPTION_FUNCTIONS:
        raise ValueError(f"Unknown corruption type: {corruption_type}")
    severity = int(severity)
    if severity < 1 or severity > 5:
        raise ValueError("severity must be in [1, 5]")
    return CORRUPTION_FUNCTIONS[corruption_type](image, severity)


def random_corruption(corruption_types, severities):
    return random.choice(corruption_types), random.choice(severities)


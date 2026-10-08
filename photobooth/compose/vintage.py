from __future__ import annotations

import cv2
import numpy as np

# Classic sepia transform matrix (applied to BGR order below).
_SEPIA_MATRIX_RGB = np.array(
    [
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131],
    ]
)


def _sepia(image_bgr: np.ndarray, strength: float) -> np.ndarray:
    rgb = image_bgr[:, :, ::-1].astype(np.float32)
    sepia_rgb = rgb @ _SEPIA_MATRIX_RGB.T
    sepia_rgb = np.clip(sepia_rgb, 0, 255)
    sepia_bgr = sepia_rgb[:, :, ::-1]
    blended = image_bgr.astype(np.float32) * (1 - strength) + sepia_bgr * strength
    return np.clip(blended, 0, 255).astype(np.uint8)


def _vignette(image_bgr: np.ndarray, strength: float) -> np.ndarray:
    height, width = image_bgr.shape[:2]
    y, x = np.ogrid[:height, :width]
    center_y, center_x = height / 2, width / 2
    max_dist = np.sqrt(center_x**2 + center_y**2)
    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2) / max_dist
    mask = 1 - strength * np.clip(dist, 0, 1) ** 2
    return np.clip(image_bgr.astype(np.float32) * mask[:, :, None], 0, 255).astype(np.uint8)


def _film_grain(image_bgr: np.ndarray, amount: float, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    noise = rng.normal(0, 255 * amount, image_bgr.shape[:2])
    noisy = image_bgr.astype(np.float32) + noise[:, :, None]
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_vintage(
    image_bgr: np.ndarray,
    sepia_strength: float = 0.5,
    vignette_strength: float = 0.35,
    grain_amount: float = 0.12,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Polaroid-ish treatment: sepia tone, edge vignette, then film grain
    last (grain on top of the darkened vignette reads as print/scan
    texture, not a separate flat noise layer).
    """
    out = _sepia(image_bgr, sepia_strength)
    out = _vignette(out, vignette_strength)
    out = _film_grain(out, grain_amount, rng)
    return out

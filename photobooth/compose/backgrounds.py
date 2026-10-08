from __future__ import annotations

import math
import os
from typing import Callable

import cv2
import numpy as np

from ..config import ComposeConfig
from .framing import cover_resize

# Independent random streams per purpose, so the background for frame i never
# depends on how many frames the reel has (a preview of frame 0 matches the
# real reel's frame 0, and a seed reproduces a reel exactly).
_STREAM_PATHS = 0
_STREAM_FRAME = 1
_STREAM_ANCHOR = 2


def path_sequence(seed: int, pool: list[str], count: int, randomize: bool = True) -> list[str]:
    """Which background file each frame starts from. Random order without
    repeats until the pool runs out, then reshuffled, never the same file
    twice in a row. A prefix of a longer sequence equals the shorter one.
    """
    if not pool:
        raise ValueError("No backgrounds configured")
    if not randomize:
        return [pool[i % len(pool)] for i in range(count)]

    rng = np.random.default_rng([seed, _STREAM_PATHS])
    sequence: list[str] = []
    while len(sequence) < count:
        order = [pool[i] for i in rng.permutation(len(pool))]
        if sequence and len(pool) > 1 and order[0] == sequence[-1]:
            order[0], order[1] = order[1], order[0]
        sequence.extend(order)
    return sequence[:count]


def anchor_for(seed: int, index: int, mode: str, offset: float = 0.18) -> float:
    """Horizontal anchor (0..1 across the frame) for frame `index`. Like the
    backgrounds, a function of (seed, index) only, so it reproduces.
    """
    left, right = 0.5 - offset, 0.5 + offset
    if mode == "left":
        return left
    if mode == "right":
        return right
    if mode == "alternate":
        first_is_left = np.random.default_rng([seed, _STREAM_ANCHOR]).random() < 0.5
        return left if (index % 2 == 0) == first_is_left else right
    if mode == "random":
        return float(np.random.default_rng([seed, _STREAM_ANCHOR, index]).choice([left, 0.5, right]))
    return 0.5


def _flip_allowed(paths: list[str], markers: list[str]) -> bool:
    """Mirroring would reverse lettering (the saloon sign), so files whose
    name contains a marker are never flipped.
    """
    names = [os.path.basename(p).lower() for p in paths]
    return not any(marker.lower() in name for name in names for marker in markers)


def vary(image_bgr: np.ndarray, rng: np.random.Generator, flip_ok: bool, zoom_max: float) -> tuple[np.ndarray, str]:
    """Random zoom/pan window, optional mirror, tone and hue jitter. Keeps
    the aspect ratio. Returns the image and a short description.
    """
    height, width = image_bgr.shape[:2]
    zoom = float(rng.uniform(1.0, max(1.0, zoom_max)))
    crop_w, crop_h = round(width / zoom), round(height / zoom)
    x0 = int(rng.integers(0, width - crop_w + 1))
    y0 = int(rng.integers(0, height - crop_h + 1))
    out = image_bgr[y0 : y0 + crop_h, x0 : x0 + crop_w]

    flipped = flip_ok and rng.random() < 0.5
    if flipped:
        out = out[:, ::-1]

    gamma = float(rng.uniform(0.8, 1.25))
    contrast = float(rng.uniform(0.92, 1.15))
    brightness = float(rng.uniform(-12, 12))
    lut = np.clip(((np.arange(256) / 255.0) ** gamma) * 255.0 * contrast + brightness, 0, 255).astype(np.uint8)
    out = cv2.LUT(np.ascontiguousarray(out), lut)

    hue_shift = int(rng.integers(-12, 13))  # OpenCV hue is 0..179, so +-12 is about +-24 degrees
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
    hsv[..., 0] = (hsv[..., 0].astype(np.int16) + hue_shift) % 180
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    label = f"zoom {zoom:.2f}{', mirrored' if flipped else ''}, hue {hue_shift:+d}"
    return out, label


def morph(a_bgr: np.ndarray, b_bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Blends two backgrounds through a soft random mask (a tilted ramp
    wobbled by low-frequency noise), so one theme melts into another
    instead of cross-fading uniformly.
    """
    height, width = a_bgr.shape[:2]
    b = cover_resize(b_bgr, width, height)

    angle = float(rng.uniform(0, 2 * math.pi))
    # float64 on purpose: float32 grid arithmetic was observed to give run-to-run different
    # results (with numpy 2.5.3), which broke seed reproducibility
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    ramp = ((xx - width / 2) * math.cos(angle) + (yy - height / 2) * math.sin(angle)) / (0.5 * math.hypot(width, height))
    mask = 0.5 + ramp * float(rng.uniform(1.5, 4.0)) + float(rng.uniform(-0.25, 0.25))
    noise = cv2.resize(rng.random((6, 8)).astype(np.float32), (width, height), interpolation=cv2.INTER_CUBIC)
    mask = np.clip(mask + (noise.astype(np.float64) - 0.5) * 0.35, 0.0, 1.0)[..., None]

    # rint, not truncation: float results sitting exactly on an integer boundary would otherwise
    # flip by one level from run to run, breaking seed reproducibility
    return np.rint(a_bgr.astype(np.float64) * (1 - mask) + b.astype(np.float64) * mask).astype(np.uint8)


def make_background(
    seed: int,
    index: int,
    pool: list[str],
    load: Callable[[str], np.ndarray],
    cfg: ComposeConfig,
    sequence: list[str] | None = None,
) -> tuple[np.ndarray, str]:
    """The background for frame `index` of the reel identified by `seed`,
    plus a human readable label. Deterministic in (seed, index, config).
    """
    primary = (sequence or path_sequence(seed, pool, index + 1, cfg.randomize_backgrounds))[index]
    rng = np.random.default_rng([seed, _STREAM_FRAME, index])
    image = load(primary)
    used = [primary]
    notes = [os.path.splitext(os.path.basename(primary))[0]]

    others = [p for p in pool if p != primary]
    if cfg.randomize_backgrounds and others and rng.random() < cfg.morph_probability:
        partner = others[int(rng.integers(0, len(others)))]
        image = morph(image, load(partner), rng)
        used.append(partner)
        notes.append("morphed with " + os.path.splitext(os.path.basename(partner))[0])

    if cfg.vary_backgrounds:
        image, variation = vary(image, rng, _flip_allowed(used, cfg.no_flip_markers), cfg.zoom_max)
        notes.append(variation)

    return image, ", ".join(notes)

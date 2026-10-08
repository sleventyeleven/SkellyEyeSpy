from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..config import StripConfig
from ..printer.raster import PRINT_WIDTH_PX

_HEADER_H = 44
_FOOTER_H = 40
_BORDER_PX = 2


def crop_around_person(
    frame_bgr: np.ndarray, mask: np.ndarray, aspect_w: int, aspect_h: int
) -> tuple[np.ndarray, np.ndarray]:
    """Crops frame and mask to aspect_w:aspect_h, centered on the person's
    horizontal centroid (frame center when nobody is detected), so a person
    standing off to one side still lands inside the frame.
    """
    height, width = frame_bgr.shape[:2]
    target = aspect_w / aspect_h
    if width / height > target:
        crop_w, crop_h = round(height * target), height
    else:
        crop_w, crop_h = width, round(width / target)

    ys, xs = np.nonzero(mask)
    center_x = float(xs.mean()) if xs.size else width / 2
    center_y = float(ys.mean()) if ys.size else height / 2
    x0 = int(np.clip(round(center_x - crop_w / 2), 0, width - crop_w))
    y0 = int(np.clip(round(center_y - crop_h / 2), 0, height - crop_h))
    return frame_bgr[y0 : y0 + crop_h, x0 : x0 + crop_w], mask[y0 : y0 + crop_h, x0 : x0 + crop_w]


def strip_height(cfg: StripConfig, n_shots: int) -> int:
    frames = n_shots * cfg.frame_height + (n_shots - 1) * cfg.margin
    return _HEADER_H + cfg.margin + frames + cfg.margin + _FOOTER_H


def _centered_text(draw: ImageDraw.ImageDraw, y: int, text: str, size: int) -> None:
    font = ImageFont.load_default(size=size)
    left, _top, right, _bottom = draw.textbbox((0, 0), text, font=font)
    draw.text(((PRINT_WIDTH_PX - (right - left)) / 2, y), text, fill=(0, 0, 0), font=font)


def render_strip(shots_bgr: list[np.ndarray], cfg: StripConfig, footer: str) -> Image.Image:
    """Stacks finished shots into one printer-width (384 px) strip. Built at
    exactly the print width so the 1-bit dither runs on real pixels with no
    resampling blur.
    """
    if not shots_bgr:
        raise ValueError("Need at least one shot")
    height = strip_height(cfg, len(shots_bgr))
    canvas = Image.new("RGB", (PRINT_WIDTH_PX, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    _centered_text(draw, 8, cfg.header, 24)
    x = (PRINT_WIDTH_PX - cfg.frame_width) // 2
    y = _HEADER_H + cfg.margin
    for shot in shots_bgr:
        resized = cv2.resize(shot, (cfg.frame_width, cfg.frame_height), interpolation=cv2.INTER_AREA)
        canvas.paste(Image.fromarray(resized[:, :, ::-1]), (x, y))
        draw.rectangle(
            (x - 1, y - 1, x + cfg.frame_width, y + cfg.frame_height), outline=(0, 0, 0), width=_BORDER_PX
        )
        y += cfg.frame_height + cfg.margin
    _centered_text(draw, height - _FOOTER_H + 8, footer, 20)
    return canvas

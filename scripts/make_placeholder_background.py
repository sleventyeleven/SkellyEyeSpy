"""Generates a crude placeholder pirate-ship background so the compositing
pipeline can be smoke-tested before real themed artwork exists. NOT final
art -- swap assets/backgrounds/pirate_ship_placeholder.png for a real
commissioned/sourced background before this ships.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

WIDTH, HEIGHT = 1280, 720


def make_background() -> Image.Image:
    img = Image.new("RGB", (WIDTH, HEIGHT))
    pixels = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)

    horizon = int(HEIGHT * 0.55)
    sky_top = np.array([255, 200, 120])
    sky_bottom = np.array([255, 140, 90])
    for y in range(horizon):
        t = y / horizon
        pixels[y, :] = (sky_top * (1 - t) + sky_bottom * t).astype(np.uint8)

    sea_top = np.array([90, 70, 40])
    sea_bottom = np.array([40, 30, 20])
    for y in range(horizon, HEIGHT):
        t = (y - horizon) / (HEIGHT - horizon)
        pixels[y, :] = (sea_top * (1 - t) + sea_bottom * t).astype(np.uint8)

    img = Image.fromarray(pixels, mode="RGB")
    draw = ImageDraw.Draw(img)

    draw.ellipse((WIDTH - 260, 60, WIDTH - 100, 220), fill=(255, 230, 180))

    hull_left, hull_right = WIDTH * 0.15, WIDTH * 0.55
    hull_top = horizon - 40
    hull_bottom = horizon + 60
    draw.polygon(
        [
            (hull_left, hull_top),
            (hull_right, hull_top),
            (hull_right - 30, hull_bottom),
            (hull_left + 30, hull_bottom),
        ],
        fill=(25, 18, 12),
    )

    mast_x = (hull_left + hull_right) / 2
    draw.line([(mast_x, hull_top), (mast_x, hull_top - 260)], fill=(20, 14, 10), width=8)
    draw.polygon(
        [
            (mast_x, hull_top - 240),
            (mast_x - 120, hull_top - 120),
            (mast_x, hull_top - 60),
        ],
        fill=(230, 220, 200),
    )

    draw.line([(0, horizon), (WIDTH, horizon)], fill=(20, 30, 40), width=3)

    return img


if __name__ == "__main__":
    out_path = "assets/backgrounds/pirate_ship_placeholder.png"
    make_background().save(out_path)
    print(f"Wrote {out_path}")

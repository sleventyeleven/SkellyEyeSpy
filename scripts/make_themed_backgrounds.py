"""Generates six crude 4:3 placeholder backgrounds for strip mode (pirate
ship, shipwreck, saloon, treasure island, Jolly Roger, kraken). NOT final
art -- strong silhouettes so they read in a 1-bit thermal print, to be
replaced by commissioned/sourced illustrations before show time.

    .venv/Scripts/python.exe scripts/make_themed_backgrounds.py
"""

from __future__ import annotations

import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1024, 768
OUT_DIR = "assets/backgrounds"


def _gradient(arr: np.ndarray, y0: int, y1: int, top, bottom) -> None:
    for y in range(y0, y1):
        t = (y - y0) / max(1, y1 - y0 - 1)
        arr[y, :] = (np.array(top) * (1 - t) + np.array(bottom) * t).astype(np.uint8)


def _seascape(horizon: float, sky, sea) -> tuple[Image.Image, ImageDraw.ImageDraw, int]:
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    hy = int(H * horizon)
    _gradient(arr, 0, hy, sky[0], sky[1])
    _gradient(arr, hy, H, sea[0], sea[1])
    img = Image.fromarray(arr)
    return img, ImageDraw.Draw(img), hy


def pirate_ship() -> Image.Image:
    img, d, hy = _seascape(0.62, ((255, 205, 130), (255, 140, 90)), ((95, 75, 45), (35, 28, 18)))
    d.ellipse((W - 250, 70, W - 90, 230), fill=(255, 235, 190))
    d.polygon([(150, hy - 70), (560, hy - 70), (520, hy + 70), (190, hy + 70)], fill=(22, 16, 10))
    for mx, mh in ((290, 330), (430, 280)):
        d.line([(mx, hy - 70), (mx, hy - 70 - mh)], fill=(20, 14, 10), width=10)
        d.polygon([(mx, hy - 60 - mh + 20), (mx + 110, hy - 70 - mh // 2), (mx, hy - 90)], fill=(235, 225, 205))
    d.polygon([(290, hy - 400), (350, hy - 385), (290, hy - 365)], fill=(15, 15, 15))
    return img


def shipwreck() -> Image.Image:
    img, d, hy = _seascape(0.58, ((120, 110, 140), (230, 150, 120)), ((50, 55, 70), (18, 20, 30)))
    d.polygon([(220, hy + 20), (620, hy - 40), (650, hy + 60), (300, hy + 120)], fill=(20, 14, 10))
    d.line([(420, hy - 10), (380, hy - 250)], fill=(25, 18, 12), width=12)  # snapped mast
    d.line([(380, hy - 250), (430, hy - 300)], fill=(25, 18, 12), width=8)
    d.polygon([(385, hy - 230), (470, hy - 180), (390, hy - 120)], fill=(200, 190, 170))  # torn sail
    for pts in ([(0, H), (120, hy + 130), (260, H)], [(W - 300, H), (W - 150, hy + 90), (W, H)]):
        d.polygon(pts, fill=(14, 14, 18))  # rocks
    for gx, gy in ((700, 140), (790, 190), (860, 120)):
        d.arc((gx, gy, gx + 50, gy + 30), 200, 340, fill=(20, 20, 20), width=4)
    return img


def saloon() -> Image.Image:
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    _gradient(arr, 0, H, (150, 100, 55), (90, 55, 28))
    img = Image.fromarray(arr)
    d = ImageDraw.Draw(img)
    for x in range(0, W, 64):
        d.line([(x, 0), (x, H)], fill=(60, 36, 18), width=4)  # planks
    d.rectangle((380, 150, 650, 380), fill=(255, 215, 130), outline=(40, 24, 12), width=12)  # window
    d.line([(515, 150), (515, 380)], fill=(40, 24, 12), width=8)
    d.line([(380, 265), (650, 265)], fill=(40, 24, 12), width=8)
    d.rectangle((300, 40, 730, 125), fill=(45, 28, 14))
    font = ImageFont.load_default(size=72)
    d.text((340, 42), "SALOON", fill=(240, 205, 120), font=font)
    for x0 in (0, W - 210):  # swinging doors
        d.rectangle((x0, 330, x0 + 210, 700), fill=(110, 68, 34), outline=(40, 24, 12), width=8)
        for y in range(350, 690, 28):
            d.line([(x0 + 14, y), (x0 + 196, y)], fill=(60, 36, 18), width=6)
    d.rectangle((0, 640, W, H), fill=(40, 24, 12))  # bar counter
    for bx in range(260, 780, 70):
        d.rectangle((bx, 560, bx + 26, 640), fill=(30, 70, 40))
        d.rectangle((bx + 8, 538, bx + 18, 560), fill=(30, 70, 40))
    return img


def treasure_island() -> Image.Image:
    img, d, hy = _seascape(0.55, ((140, 205, 235), (255, 235, 190)), ((40, 130, 160), (15, 60, 85)))
    d.ellipse((150, hy - 60, 880, hy + 220), fill=(230, 200, 130))
    for tx, lean in ((330, -40), (700, 45)):
        d.line([(tx, hy + 40), (tx + lean, hy - 230)], fill=(70, 45, 20), width=16)
        for ang in range(-80, 100, 40):
            ex = tx + lean + 130 * math.sin(math.radians(ang))
            ey = hy - 230 + 70 * math.cos(math.radians(ang)) + 40
            d.line([(tx + lean, hy - 230), (ex, ey)], fill=(20, 110, 45), width=12)
    d.rectangle((420, hy + 100, 600, hy + 180), fill=(100, 60, 25), outline=(30, 18, 8), width=6)  # chest
    d.line([(450, hy + 112), (570, hy + 170)], fill=(245, 210, 70), width=8)
    d.line([(570, hy + 112), (450, hy + 170)], fill=(245, 210, 70), width=8)
    return img


def jolly_roger() -> Image.Image:
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    _gradient(arr, 0, H, (10, 14, 34), (60, 40, 70))
    img = Image.fromarray(arr)
    d = ImageDraw.Draw(img)
    d.ellipse((W - 220, 60, W - 100, 180), fill=(240, 235, 210))
    d.line([(150, 40), (150, H)], fill=(60, 40, 20), width=16)
    d.polygon([(158, 70), (900, 90), (880, 420), (158, 440)], fill=(12, 12, 12))  # flag
    cx, cy = 520, 220
    d.ellipse((cx - 90, cy - 90, cx + 90, cy + 70), fill=(240, 240, 235))
    d.rectangle((cx - 50, cy + 40, cx + 50, cy + 100), fill=(240, 240, 235))
    for ex in (cx - 40, cx + 40):
        d.ellipse((ex - 22, cy - 30, ex + 22, cy + 14), fill=(12, 12, 12))
    d.polygon([(cx, cy + 5), (cx - 12, cy + 35), (cx + 12, cy + 35)], fill=(12, 12, 12))
    d.line([(cx - 160, cy + 140), (cx + 160, cy + 330)], fill=(240, 240, 235), width=22)
    d.line([(cx + 160, cy + 140), (cx - 160, cy + 330)], fill=(240, 240, 235), width=22)
    return img


def kraken() -> Image.Image:
    img, d, hy = _seascape(0.5, ((30, 40, 60), (110, 120, 130)), ((25, 70, 80), (6, 22, 30)))
    for x0, sway in ((170, 90), (420, -70), (700, 110), (880, -60)):
        pts = [(x0 + sway * math.sin(t / 5.0), H - t * 38 - (0 if t else 0)) for t in range(0, 15)]
        d.line(pts, fill=(60, 20, 70), width=46)
        d.line(pts, fill=(95, 40, 105), width=22)
        d.ellipse((pts[-1][0] - 28, pts[-1][1] - 28, pts[-1][0] + 28, pts[-1][1] + 28), fill=(60, 20, 70))
    d.polygon([(440, hy - 8), (560, hy - 8), (545, hy + 22), (455, hy + 22)], fill=(15, 12, 10))  # tiny ship
    d.line([(500, hy - 8), (500, hy - 70)], fill=(15, 12, 10), width=5)
    return img


THEMES = {
    "pirate_ship": pirate_ship,
    "shipwreck": shipwreck,
    "saloon": saloon,
    "treasure_island": treasure_island,
    "jolly_roger": jolly_roger,
    "kraken": kraken,
}

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, make in THEMES.items():
        path = os.path.join(OUT_DIR, f"{name}_43_placeholder.png")
        make().save(path)
        print("Wrote", path)

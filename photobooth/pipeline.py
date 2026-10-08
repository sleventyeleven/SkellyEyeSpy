from __future__ import annotations

import cv2
import numpy as np

from .compose import apply_vintage, composite_on_background
from .compose.framing import place_person
from .config import ComposeConfig
from .segmentation import SelfieSegmenter


def load_background(path: str) -> np.ndarray:
    background = cv2.imread(path)
    if background is None:
        raise FileNotFoundError(f"Background image not found: {path}")
    return background


def compose_photo(
    frame_bgr: np.ndarray,
    person_mask: np.ndarray,
    background_bgr: np.ndarray,
    compose_config: ComposeConfig,
    vintage: bool | None = None,
) -> np.ndarray:
    composite = composite_on_background(frame_bgr, person_mask, background_bgr)
    if not (compose_config.vintage if vintage is None else vintage):
        return composite
    return apply_vintage(
        composite,
        sepia_strength=compose_config.sepia_strength,
        vignette_strength=compose_config.vignette_strength,
        grain_amount=compose_config.grain_amount,
    )


def compose_normalized(
    frame_bgr: np.ndarray,
    person_mask: np.ndarray,
    background_bgr: np.ndarray,
    out_size: tuple[int, int],
    compose_config: ComposeConfig,
    vintage: bool | None = None,
    anchor_x: float = 0.5,
) -> np.ndarray:
    """Person placed at a normalized size on the background (see
    compose.framing.place_person), then the vintage treatment.
    """
    placed = place_person(
        frame_bgr,
        person_mask,
        background_bgr,
        out_size,
        height_ratio=compose_config.person_height_ratio,
        scale_limits=(compose_config.person_scale_min, compose_config.person_scale_max),
        outline_px=compose_config.outline_px,
        edge_mode=compose_config.edge_mode,
        edge_strength=compose_config.edge_strength,
        person_pop=compose_config.person_pop,
        soften=compose_config.background_soften,
        anchor_x=anchor_x,
        group_min_area_ratio=compose_config.group_min_area_ratio,
    )
    if not (compose_config.vintage if vintage is None else vintage):
        return placed
    return apply_vintage(
        placed,
        sepia_strength=compose_config.sepia_strength,
        vignette_strength=compose_config.vignette_strength,
        grain_amount=compose_config.grain_amount,
    )


def render_photo(
    frame_bgr: np.ndarray,
    segmenter: SelfieSegmenter,
    background_bgr: np.ndarray,
    compose_config: ComposeConfig,
) -> np.ndarray:
    """capture -> segment -> composite -> vintage treatment, in one shot.
    A snapshot-triggered pipeline, not continuous per-frame processing --
    latency of a second or so is fine for a photo-booth-style capture.
    """
    return compose_photo(frame_bgr, segmenter.person_mask(frame_bgr), background_bgr, compose_config)

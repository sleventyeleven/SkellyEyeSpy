from __future__ import annotations

import cv2
import numpy as np

_MASK_FEATHER_PX = 9


def composite_on_background(
    frame_bgr: np.ndarray, person_mask: np.ndarray, background_bgr: np.ndarray
) -> np.ndarray:
    """Cuts the person out of frame_bgr (using person_mask, 0/255) and
    composites them onto background_bgr, resized to match frame_bgr.

    The mask is feathered (Gaussian-blurred) before use rather than used
    as a hard cutout -- this softens segmentation edge errors into a
    blend instead of a jagged boundary, which the vintage/grain treatment
    downstream hides even further.
    """
    height, width = frame_bgr.shape[:2]
    background = cv2.resize(background_bgr, (width, height))

    alpha = cv2.GaussianBlur(person_mask, (_MASK_FEATHER_PX, _MASK_FEATHER_PX), 0)
    alpha = alpha.astype(np.float32) / 255.0
    alpha = alpha[:, :, None]

    composite = frame_bgr.astype(np.float32) * alpha + background.astype(np.float32) * (1 - alpha)
    return np.rint(composite).astype(np.uint8)

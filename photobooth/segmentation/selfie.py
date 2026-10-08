from __future__ import annotations

import os

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

_DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "selfie_segmenter_landscape.tflite")


class SelfieSegmenter:
    """Wraps MediaPipe's Selfie Segmentation model (Tasks API -- the
    legacy `mp.solutions` API was removed in current mediapipe releases)
    -- purpose-built for person-cutout/background-replacement, and far
    lighter than an instance-segmentation model like YOLOv8-seg for this
    exact task.

    Requires the model file from scripts/download_models.py.
    """

    def __init__(self, model_path: str = _DEFAULT_MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_path} not found -- run scripts/download_models.py first")

        options = vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.IMAGE,
            output_confidence_masks=True,
        )
        self._segmenter = vision.ImageSegmenter.create_from_options(options)

    def person_mask(self, frame_bgr: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Returns a uint8 mask (0/255) the same height/width as frame_bgr,
        255 where the model's person probability exceeds `threshold`.

        Uses the confidence mask, not the category mask: this model's
        category mask is 0 = person / 255 = background (not 0/1), which a
        `== 1` test silently turns into an always-empty mask.
        """
        rgb = frame_bgr[:, :, ::-1]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._segmenter.segment(mp_image)
        person_probability = np.squeeze(result.confidence_masks[0].numpy_view())
        return (person_probability > threshold).astype(np.uint8) * 255

    def close(self) -> None:
        self._segmenter.close()

    def __enter__(self) -> "SelfieSegmenter":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

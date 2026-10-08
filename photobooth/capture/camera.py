from __future__ import annotations

import sys
import time

import cv2
import numpy as np

from ..config import CaptureConfig

_OPEN_ATTEMPTS = 2
_OPEN_RETRY_DELAY_S = 0.5


class Camera:
    """OpenCV capture wrapper, shared by the dev-laptop webcam and the
    Zero 3W's CSI camera (which appears as a plain V4L2 /dev/videoN once
    its overlay is loaded -- same code path, just a different index).
    """

    def __init__(self, config: CaptureConfig):
        self._config = config
        source: int | str = int(config.source) if config.source.isdigit() else config.source
        # Force V4L2 for plain device indexes and /dev paths (e.g. the stable
        # /dev/v4l/by-id/... symlinks) on Linux: OpenCV's auto-select can pick a
        # GStreamer backend that intermittently fails to start on these boards.
        is_device = isinstance(source, int) or str(source).startswith("/dev/")
        backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") and is_device else cv2.CAP_ANY

        self._cap: cv2.VideoCapture | None = None
        for attempt in range(_OPEN_ATTEMPTS):
            cap = cv2.VideoCapture(source, backend)
            if cap.isOpened():
                self._cap = cap
                break
            cap.release()
            if attempt + 1 < _OPEN_ATTEMPTS:
                time.sleep(_OPEN_RETRY_DELAY_S)

        if self._cap is None:
            raise RuntimeError(f"Could not open capture source: {config.source!r}")

        if backend == cv2.CAP_V4L2:
            # Many UVC webcams (e.g. EMEET C950) offer raw YUYV only at 640x480;
            # 720p/1080p are MJPEG-only, so ask for it before setting the size.
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)

    def read(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        return frame if ok else None

    def release(self) -> None:
        self._cap.release()

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *_exc) -> None:
        self.release()

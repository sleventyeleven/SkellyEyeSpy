"""Downloads the MediaPipe model assets this project needs. Not committed
to git (binary, and easy to re-fetch) -- run this once after cloning:

    .venv/Scripts/python.exe scripts/download_models.py
"""

from __future__ import annotations

import os
import urllib.request

_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "photobooth", "segmentation", "models")

# Landscape variant: wider field of view, better suited to a photo booth's
# fuller-body framing than the default (closer-subject) selfie_segmenter.
_SELFIE_SEGMENTER_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite"
)


def main() -> None:
    os.makedirs(_MODELS_DIR, exist_ok=True)
    dest = os.path.join(_MODELS_DIR, "selfie_segmenter_landscape.tflite")
    if os.path.exists(dest):
        print(f"Already present: {dest}")
        return
    print(f"Downloading {_SELFIE_SEGMENTER_URL}")
    urllib.request.urlretrieve(_SELFIE_SEGMENTER_URL, dest)
    print(f"Wrote {dest}")


if __name__ == "__main__":
    main()

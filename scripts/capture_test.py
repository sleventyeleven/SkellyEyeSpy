"""Dev-loop smoke test: take one photo through the real booth service (camera,
segmentation, backgrounds, compositing) and, if the config enables printing,
send it to the printer (or write the dry-run payload). Run from the repo root:

    .venv/Scripts/python.exe scripts/capture_test.py configs/dev_laptop.yaml

Output goes to scratch/booth/. For interactive use, run scripts/run_booth.py
and open the studio page instead.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from photobooth.config import PhotoBoothConfig
from photobooth.service import PhotoBoothService


def main(config_path: str) -> None:
    config = PhotoBoothConfig.from_yaml(config_path)
    service = PhotoBoothService(config, output_dir="scratch/booth")
    try:
        result = service.snap()
    finally:
        service.close()

    print(f"Wrote {result.image_path}")
    print(f"      {result.debug_path}  (raw frame | person mask | result)")
    print(f"      {result.preview_path}  (the exact 1-bit print)")
    print(f"seed {result.seed}: {', '.join(result.backgrounds)} ({result.seconds:.1f}s)")
    if config.printer.enabled:
        if result.printed and config.printer.dry_run_path:
            size = os.path.getsize(config.printer.dry_run_path)
            print(f"Dry-run ESC/POS payload written to {config.printer.dry_run_path} ({size} bytes)")
        elif result.printed:
            print("Printed.")
        else:
            print(f"Not printed: {result.error}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "configs/dev_laptop.yaml")

"""Standalone booth server for bench testing (no GPIO needed):

    .venv/Scripts/python.exe scripts/run_booth.py configs/dev_laptop.yaml
    curl -X POST http://localhost:5001/api/photobooth/snap
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask

from photobooth.config import PhotoBoothConfig
from photobooth.interaction.factory import build_interaction
from photobooth.service import PhotoBoothService
from photobooth.web import create_blueprint

if __name__ == "__main__":
    config = PhotoBoothConfig.from_yaml(sys.argv[1] if len(sys.argv) > 1 else "configs/dev_laptop.yaml")
    app = Flask(__name__)
    service = PhotoBoothService(config)
    interaction = build_interaction(service, config)  # None unless interaction.enabled
    if interaction is not None:
        print(f"Booth voice: {len(interaction.bank)} lines, audio {'simulated' if interaction.simulated_audio else 'on'}, "
              f"eyes/mouth {'simulated' if interaction.simulated_gpio else 'LIVE'}")
    app.register_blueprint(create_blueprint(service, interaction))
    app.run(host="0.0.0.0", port=5001)

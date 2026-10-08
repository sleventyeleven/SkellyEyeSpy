from __future__ import annotations

from dataclasses import dataclass, field

import yaml


@dataclass
class CaptureConfig:
    source: str = "0"
    width: int = 1280
    height: int = 720


@dataclass
class ComposeConfig:
    background: str = "assets/backgrounds/pirate_ship.png"
    grain_amount: float = 0.12
    vignette_strength: float = 0.35
    sepia_strength: float = 0.5
    working_width: int = 640
    # Normalize the person's visible height to this fraction of the output frame so
    # tall/short/close/far people all leave room for the background to show.
    # Each reel gets a random seed: random distinct backgrounds in random order, each frame
    # zoomed/panned/mirrored/tone-shifted, and some frames morphed from two themes.
    # `backgrounds` entries may be glob patterns, so dropping art into a folder is enough.
    randomize_backgrounds: bool = True
    vary_backgrounds: bool = True
    morph_probability: float = 0.33  # share of frames that blend two scenes
    zoom_max: float = 1.4
    no_flip_markers: list[str] = field(default_factory=lambda: ["saloon", "noflip"])  # filename substrings; lettering: never mirror
    # Keep a person readable against detailed scenes once dithered to 1 bit.
    # How the person is separated from a detailed scene without a drawn border. A solid white
    # ring (even 1 px) printed as an "angel glow", so the default is a soft local dodge/burn:
    # stipple (a fading scatter of dots) | separate (soft dodge/burn) | none | ring (the old glow)
    edge_mode: str = "stipple"
    outline_px: int = 5  # width in px of the edge band (separate/stipple/ring); 0 turns it off
    edge_strength: float = 0.3  # 0..1
    person_pop: float = 0.15  # 0..1: a little extra saturation and tone contrast on the person
    background_soften: float = 0.33  # 0..1: blur, flatten and brighten the scene
    background_max_width: int = 720  # loaded backgrounds are downscaled to this (final frames are 360 px)
    # Where the person stands in each frame: center, left, right, alternate (left/right by
    # frame, first side chosen by the seed) or random (center/left/right per frame).
    anchor_mode: str = "alternate"
    anchor_offset: float = 0.18  # how far from center left/right sit, as a fraction of width
    group_min_area_ratio: float = 0.12  # people smaller than this share of the biggest are ignored
    vintage: bool = True  # sepia + vignette + film grain; the 1-bit dither adds its own grain
    normalize_person: bool = True
    person_height_ratio: float = 0.35  # tuned by eye: 30-40% suits an adult against these scenes
    person_scale_min: float = 0.1  # a floor above ~0.3 silently overrode small ratios for people filling the camera frame
    person_scale_max: float = 1.8
    backgrounds: list[str] = field(default_factory=list)  # strip mode cycles through these


@dataclass
class StripConfig:
    """Old-school photo-booth reel: N frames stacked on one 384 px wide strip."""

    shots: int = 4
    frame_width: int = 360
    frame_height: int = 270  # 4:3 frames
    margin: int = 12
    header: str = "SKELLY'S PIRATE BOOTH"
    footer: str = ""  # empty -> today's date


@dataclass
class PrinterConfig:
    enabled: bool = False
    backend: str = "phomemo_m02"
    bluetooth_address: str | None = None
    dry_run_path: str | None = None
    # The M02 has a small buffer and no flow control; see PhomemoM02Printer.
    # Lower these if prints still stop part way (60 / 255 = one block at a time).
    lines_per_second: float = 100.0
    max_backlog_lines: int = 400
    tail_pad_s: float = 1.5


@dataclass
class PhotoBoothConfig:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    compose: ComposeConfig = field(default_factory=ComposeConfig)
    printer: PrinterConfig = field(default_factory=PrinterConfig)
    strip: StripConfig = field(default_factory=StripConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "PhotoBoothConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls(
            capture=CaptureConfig(**data.get("capture", {})),
            compose=ComposeConfig(**data.get("compose", {})),
            printer=PrinterConfig(**data.get("printer", {})),
            strip=StripConfig(**data.get("strip", {})),
        )

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
class InteractionConfig:
    """The booth's spoken, animated visitor sequence (see photobooth/interaction/)."""

    enabled: bool = False
    voice_dir: str = "assets/voice"  # rendered by scripts/make_voice_lines.py
    play_audio: bool = True
    # The MAX98357A card only takes stereo; plughw converts the mono clips for us.
    audio_device: str = "plughw:CARD=sndmax98357adac,DEV=0"
    volume: float = 1.0  # 0..1; the MAX98357A is a small mono amp, speech below ~30% is hard to understand
    # Moving the real eyes and mouth is opt-in: off means a simulated skeleton, audio only.
    use_gpio: bool = False
    move_mouth: bool = True
    flash_eyes: bool = True
    auto_print: bool = False  # studio-first: review the reel before spending paper

    welcome: bool = True  # the long instructions line; turn off for a faster line of visitors
    invite_max: int = 4  # invitation lines before giving up on nobody coming
    invite_gap_s: float = 4.0  # quiet time after each invitation, still watching for visitors
    presence_poll_s: float = 1.0  # how often to look for a person while waiting
    presence_confirmations: int = 2  # consecutive looks with a person before greeting them
    absence_confirmations: int = 4  # consecutive empty looks before deciding the visitor walked off
    hold_seconds: float = 3.0  # at least this long of "hold still" talk before the countdown
    count_gap_s: float = 0.5  # pause between spoken countdown words
    cooldown_s: float = 15.0  # no new sequence from a trigger for this long afterwards

    # Mouth motor limits: the jaw is a small DC motor, so never chatter it.
    mouth_on_threshold: float = 0.30  # normalized loudness that opens the mouth
    mouth_off_threshold: float = 0.14  # ... and that lets it close (hysteresis)
    mouth_min_open_s: float = 0.175  # measured on the skelly: a 0.35 s open+close cycle keeps up
    mouth_min_closed_s: float = 0.175
    mouth_max_open_s: float = 0.45  # never hold the motor on longer than this
    # The jaw drive seems to need recovery time (a capacitor/transistor stage): it opens fully when rested and
    # goes sluggish, then stalls, after many cycles. The mouth is a charge budget: each open second drains it,
    # each closed second refills `mouth_recharge_per_s` of a second, and an open is skipped if it could not be
    # paid for. Steady-state this caps the jaw at recharge/(1+recharge) of the time open (0.6 -> 37%).
    mouth_budget_s: float = 2.0  # seconds of open time available when fully rested
    mouth_recharge_per_s: float = 0.6
    # Eyes glow while speaking and go dark briefly at irregular intervals (the LEDs/relay are too slow for fast flicker).
    eye_blink_s: float = 0.30  # how long the eyes stay dark for a blink
    eye_blink_min_gap_s: float = 0.8  # lit time between blinks is random in this range
    eye_blink_max_gap_s: float = 1.5

    # Optional physical trigger: a button to ground (active low) or a PIR sensor (active high).
    trigger_line: int | None = None  # gpiochip line number; None disables the trigger
    trigger_chip: str = "/dev/gpiochip3"
    trigger_active_low: bool = True
    # "button": someone pressed it, so skip the invitations. "motion": a PIR saw movement, so invite them over.
    trigger_kind: str = "button"
    trigger_debounce_s: float = 0.05


@dataclass
class PhotoBoothConfig:
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    compose: ComposeConfig = field(default_factory=ComposeConfig)
    printer: PrinterConfig = field(default_factory=PrinterConfig)
    strip: StripConfig = field(default_factory=StripConfig)
    interaction: InteractionConfig = field(default_factory=InteractionConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "PhotoBoothConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        return cls(
            capture=CaptureConfig(**data.get("capture", {})),
            compose=ComposeConfig(**data.get("compose", {})),
            printer=PrinterConfig(**data.get("printer", {})),
            strip=StripConfig(**data.get("strip", {})),
            interaction=InteractionConfig(**data.get("interaction", {})),
        )

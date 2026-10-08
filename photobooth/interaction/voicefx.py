"""Turns a clean text-to-speech clip into a spooky pirate ghoul.

numpy only, so it runs anywhere and is easy to test. Used when the voice
lines are rendered (scripts/make_voice_lines.py), never at show time: the
skelly just plays the finished WAV files.
"""

from __future__ import annotations

import numpy as np

# Each preset is a chain applied in a fixed order. `semitones` lowers the pitch
# by resampling, which also slows the speech: a drawl is exactly what a ghoul
# wants, so there is deliberately no tempo correction.
PRESETS: dict[str, dict] = {
    "dry": {},
    "ghoul": {"semitones": -4, "saturate": 1.4, "echo": (0.17, 0.22), "reverb": (1.0, 0.25), "tremolo": (5.0, 0.10), "lowpass": 7500},
    "ghost": {"semitones": -2, "echo": (0.24, 0.35), "reverb": (1.6, 0.45), "tremolo": (6.5, 0.18), "lowpass": 6500},
    # ghoul with the echo left to linger: more repeats, a longer cave.
    "ghoul_echo": {"semitones": -4, "saturate": 1.4, "echo": (0.26, 0.40, 4), "reverb": (1.3, 0.30), "tremolo": (5.0, 0.10), "lowpass": 7500},
    "ghoul_haunt": {"semitones": -4, "saturate": 1.4, "echo": (0.34, 0.50, 6), "reverb": (1.7, 0.36), "tremolo": (5.0, 0.10), "lowpass": 7000},
    # Between ghoul and crypt: deeper and grittier than ghoul, still easy to understand.
    "tomb": {"semitones": -5, "saturate": 1.6, "echo": (0.15, 0.20), "reverb": (1.1, 0.27), "tremolo": (4.0, 0.10), "lowpass": 6800},
    "crypt": {"semitones": -7, "saturate": 2.0, "echo": (0.14, 0.20), "reverb": (1.3, 0.30), "tremolo": (3.5, 0.12), "lowpass": 5500},
}


def pitch_down(x: np.ndarray, semitones: float) -> np.ndarray:
    """Lower the pitch by `semitones` (negative) by resampling; slows the clip too."""
    if semitones == 0:
        return x
    factor = 2.0 ** (semitones / 12.0)  # < 1 for lower pitch
    positions = np.arange(0, len(x) - 1, factor)
    return np.interp(positions, np.arange(len(x)), x).astype(np.float32)


def tremolo(x: np.ndarray, sr: int, rate_hz: float, depth: float) -> np.ndarray:
    t = np.arange(len(x)) / sr
    return (x * (1.0 - depth / 2.0 + (depth / 2.0) * np.sin(2 * np.pi * rate_hz * t))).astype(np.float32)


def saturate(x: np.ndarray, drive: float) -> np.ndarray:
    return (np.tanh(drive * x) / np.tanh(drive)).astype(np.float32)


def echo(x: np.ndarray, sr: int, delay_s: float, gain: float, taps: int = 2) -> np.ndarray:
    """`taps` repeats, each `gain` times quieter than the last; more taps linger longer."""
    delay = int(delay_s * sr)
    out = np.concatenate([x, np.zeros(delay * taps, np.float32)])
    for k in range(1, taps + 1):
        out[k * delay : k * delay + len(x)] += (gain ** k) * x
    return out


def lowpass(x: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1.0 / sr)
    spectrum *= 1.0 / (1.0 + (freqs / cutoff_hz) ** 4)  # gentle 4th-order roll-off
    return np.fft.irfft(spectrum, len(x)).astype(np.float32)


def reverb(x: np.ndarray, sr: int, decay_s: float, wet: float, seed: int = 7) -> np.ndarray:
    """A synthetic cave: exponentially decaying noise as the impulse response."""
    rng = np.random.default_rng(seed)
    n = int(decay_s * sr)
    impulse = rng.standard_normal(n) * np.exp(-6.0 * np.arange(n) / n)
    impulse /= np.abs(impulse).sum() / 8.0  # keep the wet level predictable
    size = len(x) + n
    fft_size = 1 << int(np.ceil(np.log2(size)))
    wet_signal = np.fft.irfft(np.fft.rfft(x, fft_size) * np.fft.rfft(impulse, fft_size), fft_size)[:size]
    dry = np.concatenate([x, np.zeros(n, np.float32)])
    return ((1.0 - wet) * dry + wet * wet_signal).astype(np.float32)


def tighten_pauses(x: np.ndarray, sr: int, max_pause_s: float = 0.28, threshold_db: float = -42.0) -> np.ndarray:
    """Shortens every silent gap longer than `max_pause_s` to that length.

    Text-to-speech leaves a long beat at each full stop, and slowing the voice
    down for the ghoul effect stretches it further; a skeleton that hangs on
    "booth." sounds broken. Gaps are cut in the middle so the fades on both
    sides stay intact.
    """
    quiet = np.abs(x) < 10 ** (threshold_db / 20.0) * max(float(np.abs(x).max()), 1e-9)
    hop = max(1, int(0.01 * sr))
    keep = np.ones(len(x), bool)
    i, n = 0, len(x)
    while i < n:
        if not quiet[i]:
            i += 1
            continue
        j = i
        while j < n and quiet[j]:
            j += 1
        excess = (j - i) - int(max_pause_s * sr)
        if excess > hop and i > 0 and j < n:  # interior gaps only; edges are trimmed elsewhere
            cut_start = i + (j - i - excess) // 2
            keep[cut_start : cut_start + excess] = False
        i = j
    return x[keep]


def trim_silence(x: np.ndarray, sr: int, threshold_db: float = -45.0, keep_s: float = 0.04) -> np.ndarray:
    """Trim leading/trailing silence so spoken timing (and the capture at 'got it') is tight."""
    loud = np.nonzero(np.abs(x) > 10 ** (threshold_db / 20.0) * max(np.abs(x).max(), 1e-9))[0]
    if loud.size == 0:
        return x
    keep = int(keep_s * sr)
    return x[max(0, loud[0] - keep) : min(len(x), loud[-1] + keep)]


def normalize_peak(x: np.ndarray, peak_db: float = -3.0) -> np.ndarray:
    peak = float(np.abs(x).max())
    return x if peak == 0 else (x * (10 ** (peak_db / 20.0) / peak)).astype(np.float32)


def apply_preset(x: np.ndarray, sr: int, name: str) -> np.ndarray:
    """Float32 mono in [-1, 1] -> processed float32 mono, trimmed and peak-normalized."""
    if name not in PRESETS:
        raise ValueError(f"Unknown voice preset {name!r}; choose from {', '.join(PRESETS)}")
    cfg = PRESETS[name]
    x = tighten_pauses(trim_silence(np.asarray(x, np.float32), sr), sr)
    if "semitones" in cfg:
        x = pitch_down(x, cfg["semitones"])
    if "saturate" in cfg:
        x = saturate(x, cfg["saturate"])
    if "echo" in cfg:
        x = echo(x, sr, *cfg["echo"])
    if "reverb" in cfg:
        x = reverb(x, sr, *cfg["reverb"])
    if "tremolo" in cfg:
        x = tremolo(x, sr, *cfg["tremolo"])
    if "lowpass" in cfg:
        x = lowpass(x, sr, cfg["lowpass"])
    return normalize_peak(trim_silence(x, sr, threshold_db=-50.0, keep_s=0.08))

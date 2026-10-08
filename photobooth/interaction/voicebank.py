from __future__ import annotations

import json
import os
import random
import wave
from dataclasses import dataclass

import numpy as np

ENVELOPE_HOP_S = 0.02


@dataclass(frozen=True)
class Line:
    id: str
    category: str
    text: str
    path: str
    seconds: float
    envelope: np.ndarray  # loudness per ENVELOPE_HOP_S, normalized to roughly 0..1 (drives the mouth)

    @property
    def hop_s(self) -> float:
        return ENVELOPE_HOP_S


def loudness_envelope(samples: np.ndarray, rate: int, hop_s: float = ENVELOPE_HOP_S) -> np.ndarray:
    """RMS per hop, lightly smoothed, scaled so a normal loud syllable is ~1."""
    hop = max(1, int(hop_s * rate))
    frames = len(samples) // hop
    if frames == 0:
        return np.zeros(0, np.float32)
    rms = np.sqrt((samples[: frames * hop].reshape(frames, hop) ** 2).mean(axis=1))
    rms = np.convolve(rms, np.array([0.25, 0.5, 0.25]), mode="same")
    scale = float(np.percentile(rms, 95))
    return np.clip(rms / scale, 0.0, 1.0).astype(np.float32) if scale > 1e-6 else np.zeros_like(rms, np.float32)


def _read_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as wav:
        rate, channels, width = wav.getframerate(), wav.getnchannels(), wav.getsampwidth()
        raw = wav.readframes(wav.getnframes())
    if width != 2:
        raise ValueError(f"{path}: expected 16-bit PCM")
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return (samples.reshape(-1, channels).mean(axis=1) if channels > 1 else samples), rate


class VoiceBank:
    """The rendered voice lines, grouped by category (invite, welcome, filler,
    three, two, one, got_it, printing, done, nobody, come_back).
    """

    def __init__(self, lines: list[Line] | None = None):
        self._lines = list(lines or [])
        self._by_id = {line.id: line for line in self._lines}

    @classmethod
    def load(cls, directory: str) -> "VoiceBank":
        manifest = os.path.join(directory, "manifest.json")
        if not os.path.exists(manifest):
            return cls()
        with open(manifest, encoding="utf-8") as f:
            entries = json.load(f)["lines"]
        lines = []
        for entry in entries:
            path = os.path.join(directory, entry["file"])
            samples, rate = _read_wav(path)
            lines.append(Line(entry["id"], entry["category"], entry["text"], path, len(samples) / rate,
                              loudness_envelope(samples, rate)))
        return cls(lines)

    def __len__(self) -> int:
        return len(self._lines)

    def __iter__(self):
        return iter(self._lines)

    def get(self, line_id: str) -> Line:
        return self._by_id[line_id]

    def has(self, category: str) -> bool:
        return any(line.category == category for line in self._lines)

    def category(self, category: str) -> list[Line]:
        return [line for line in self._lines if line.category == category]

    def pick(self, category: str, rng: random.Random | None = None, avoid: str | None = None) -> Line:
        """A random line from the category, not the one just played when there is a choice."""
        options = self.category(category)
        if not options:
            raise KeyError(f"No voice lines in category {category!r}")
        if avoid and len(options) > 1:
            options = [line for line in options if line.id != avoid] or options
        return (rng or random).choice(options)

    def catalog(self) -> list[dict]:
        return [{"id": l.id, "category": l.category, "text": l.text, "seconds": round(l.seconds, 2)} for l in self._lines]

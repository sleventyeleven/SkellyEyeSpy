"""Renders the booth's voice lines offline: Piper text-to-speech, then the
ghoul effect chain, written as small mono WAV files plus a manifest.

Piper is only needed here, never at show time (the skelly just plays the
WAVs). It is a separate optional tool, installed on its own:

    python -m venv voicegen && voicegen/Scripts/pip install piper-tts pyyaml numpy
    voicegen/Scripts/python scripts/make_voice_lines.py            # render assets/voice/
    voicegen/Scripts/python scripts/make_voice_lines.py --audition  # A/B voices and presets

Voice models are downloaded to scratch/piper_models on first use. Check each
voice's license before sharing the rendered audio: en_US-ryan-high is
CC BY-NC-SA 4.0 (non-commercial, share-alike); en_GB-alan-medium points at the
Mimic3 dataset's own license.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import urllib.request
import wave

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from photobooth.interaction.voicefx import PRESETS, apply_preset  # noqa: E402

HF = "https://huggingface.co/rhasspy/piper-voices/resolve/main"


def voice_url(voice: str) -> str:
    lang, name, quality = voice.split("-", 2)  # en_GB-alan-medium
    return f"{HF}/{lang.split('_')[0]}/{lang}/{name}/{quality}/{voice}"


def ensure_model(voice: str, models_dir: str) -> str:
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, voice + ".onnx")
    for suffix in (".onnx", ".onnx.json"):
        target = os.path.join(models_dir, voice + suffix)
        if not os.path.exists(target):
            print(f"Downloading {voice}{suffix} ...")
            urllib.request.urlretrieve(voice_url(voice) + suffix, target)
    return path


def synthesize(model_path: str, text: str) -> tuple[np.ndarray, int]:
    from piper import PiperVoice

    voice = PiperVoice.load(model_path)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        voice.synthesize_wav(text, wav)
    buffer.seek(0)
    with wave.open(buffer, "rb") as wav:
        rate = wav.getframerate()
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    return samples, rate


def write_wav(path: str, samples: np.ndarray, rate: int) -> float:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm.tobytes())
    return len(pcm) / rate


def render(script_path: str, out_dir: str, models_dir: str, voice: str | None, preset: str | None, only: set[str]) -> None:
    with open(script_path, encoding="utf-8") as f:
        script = yaml.safe_load(f)
    voice = voice or script["voice"]
    preset = preset or script["preset"]
    model = ensure_model(voice, models_dir)
    os.makedirs(out_dir, exist_ok=True)

    manifest = {"version": 1, "voice": voice, "preset": preset, "lines": []}
    total = 0.0
    for category, entries in script["lines"].items():
        for entry in entries:
            if only and entry["id"] not in only:
                continue
            samples, rate = synthesize(model, entry["text"])
            seconds = write_wav(os.path.join(out_dir, entry["id"] + ".wav"), apply_preset(samples, rate, preset), rate)
            total += seconds
            manifest["lines"].append({"id": entry["id"], "category": category, "text": entry["text"],
                                      "file": entry["id"] + ".wav", "seconds": round(seconds, 3)})
            print(f"  {entry['id']:12s} {seconds:5.2f}s  {entry['text'][:60]}")
    if not only:
        with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    print(f"Rendered {len(manifest['lines'])} lines, {total:.1f}s of audio, voice={voice}, preset={preset}")


def audition(models_dir: str, out_dir: str, voices: list[str], presets: list[str], text: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for voice in voices:
        model = ensure_model(voice, models_dir)
        samples, rate = synthesize(model, text)
        for preset in presets:
            name = f"{voice}_{preset}.wav"
            seconds = write_wav(os.path.join(out_dir, name), apply_preset(samples, rate, preset), rate)
            print(f"  {name:36s} {seconds:5.2f}s")


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--script", default=os.path.join(root, "assets", "voice", "script.yaml"))
    parser.add_argument("--out", default=os.path.join(root, "assets", "voice"))
    parser.add_argument("--models", default=os.path.join(root, "scratch", "piper_models"))
    parser.add_argument("--voice", help="override the script's voice")
    parser.add_argument("--preset", choices=sorted(PRESETS), help="override the script's preset")
    parser.add_argument("--only", help="comma separated line ids (does not rewrite the manifest)")
    parser.add_argument("--audition", action="store_true", help="render the welcome line in several voices and presets")
    parser.add_argument("--audition-voices", default="en_GB-alan-medium,en_US-ryan-high")
    parser.add_argument("--audition-out", default=os.path.join(root, "scratch", "voice_samples"))
    args = parser.parse_args()

    if args.audition:
        with open(args.script, encoding="utf-8") as f:
            welcome = yaml.safe_load(f)["lines"]["welcome"][0]["text"]
        audition(args.models, args.audition_out, args.audition_voices.split(","), [p for p in PRESETS if p != "dry"], welcome)
    else:
        render(args.script, args.out, args.models, args.voice, args.preset, set(filter(None, (args.only or "").split(","))))

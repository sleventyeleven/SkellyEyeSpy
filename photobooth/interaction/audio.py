from __future__ import annotations

import os
import subprocess
import tempfile
import time
import wave
from typing import Callable, Protocol

import numpy as np


class PlayHandle(Protocol):
    def wait(self, timeout: float | None = None) -> bool: ...
    def stop(self) -> None: ...


class AudioPlayer(Protocol):
    def play(self, path: str, seconds: float, volume: float = 1.0) -> PlayHandle: ...


class SimHandle:
    """A pretend playback that 'finishes' after the clip's length on an injected clock."""

    def __init__(self, seconds: float, clock: Callable[[], float], sleep: Callable[[float], None]):
        self._end = clock() + seconds
        self._clock, self._sleep = clock, sleep
        self.stopped = False

    def wait(self, timeout: float | None = None) -> bool:
        remaining = self._end - self._clock()
        if remaining > 0 and not self.stopped:
            self._sleep(remaining if timeout is None else min(timeout, remaining))
        return self.stopped or self._clock() >= self._end

    def stop(self) -> None:
        self.stopped = True


class SimPlayer:
    """No sound: records what would have played. For tests and for developing without the speaker."""

    def __init__(self, clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep):
        self._clock, self._sleep = clock, sleep
        self.plays: list[tuple[str, float, float]] = []  # (file name, start time, volume)
        self.handles: list[SimHandle] = []

    def play(self, path: str, seconds: float, volume: float = 1.0) -> SimHandle:
        self.plays.append((os.path.basename(path), self._clock(), volume))
        handle = SimHandle(seconds, self._clock, self._sleep)
        self.handles.append(handle)
        return handle


class AplayHandle:
    def __init__(self, process: subprocess.Popen, temp_path: str):
        self._process, self._temp_path = process, temp_path

    def wait(self, timeout: float | None = None) -> bool:
        try:
            self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
        self._cleanup()
        return True

    def stop(self) -> None:
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._cleanup()

    def _cleanup(self) -> None:
        if self._temp_path and os.path.exists(self._temp_path):
            try:
                os.remove(self._temp_path)
            except OSError:
                pass
            self._temp_path = ""


class AplayPlayer:
    """Plays through ALSA's `aplay`. The volume is applied to the samples (the
    MAX98357A has no software mixer) and mono is duplicated to the stereo the
    amp's I2S card requires. Playback is asynchronous: play() returns at once.
    """

    def __init__(
        self,
        device: str = "plughw:CARD=sndmax98357adac,DEV=0",
        aplay: str = "aplay",
        popen: Callable[..., subprocess.Popen] = subprocess.Popen,
    ):
        self._device, self._aplay, self._popen = device, aplay, popen

    def play(self, path: str, seconds: float, volume: float = 1.0) -> AplayHandle:
        with wave.open(path, "rb") as wav:
            rate, channels = wav.getframerate(), wav.getnchannels()
            pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16)
        if channels == 1:
            pcm = np.repeat(pcm, 2)  # same signal on both channels; the amp sums them to mono
        scaled = np.clip(pcm.astype(np.float32) * float(np.clip(volume, 0.0, 1.0)), -32768, 32767).astype(np.int16)

        handle, temp_path = tempfile.mkstemp(suffix=".wav", prefix="booth_voice_")
        os.close(handle)
        with wave.open(temp_path, "wb") as out:
            out.setnchannels(2)
            out.setsampwidth(2)
            out.setframerate(rate)
            out.writeframes(scaled.tobytes())
        process = self._popen([self._aplay, "-q", "-D", self._device, temp_path],
                              stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return AplayHandle(process, temp_path)

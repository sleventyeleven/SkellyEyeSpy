from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..config import InteractionConfig
from .voicebank import Line

COUNT_WORDS = ("three", "two", "one")
_SLICE_S = 0.05  # how often waits wake to check for stop/interrupt


@dataclass(frozen=True)
class Event:
    t: float  # seconds after the audio starts
    kind: str  # mouth_open | mouth_close | eyes_on | eyes_off


def plan_mouth(envelope: np.ndarray, hop_s: float, cfg: InteractionConfig) -> list[Event]:
    """Opens the jaw on loud syllables and closes it in the gaps, never faster
    or longer than the motor limits: a minimum open time, a minimum closed
    time, and a hard cap on how long it may stay open. Always ends closed.
    """
    events: list[Event] = []
    is_open = False
    last_change = -1e9
    for i, loudness in enumerate(envelope):
        t = i * hop_s
        if not is_open:
            if loudness >= cfg.mouth_on_threshold and t - last_change >= cfg.mouth_min_closed_s:
                events.append(Event(t, "mouth_open"))
                is_open, last_change = True, t
        else:
            held = t - last_change
            if (loudness <= cfg.mouth_off_threshold and held >= cfg.mouth_min_open_s) or held >= cfg.mouth_max_open_s:
                events.append(Event(t, "mouth_close"))
                is_open, last_change = False, t
    if is_open:
        events.append(Event(max(len(envelope) * hop_s, last_change + cfg.mouth_min_open_s), "mouth_close"))
    return events


def plan_eyes(category: str, seconds: float, cfg: InteractionConfig | None = None, rng: random.Random | None = None) -> list[Event]:
    """Eyes glow while the skeleton speaks and go dark for a moment every
    ~0.8-1.5 s, like a slow, irregular blink. Real eyes are too slow for fast
    flicker, so every dark or lit stretch is at least `eye_blink_s` long.
    Countdown words blink once; "got it" flashes twice.
    """
    cfg = cfg or InteractionConfig()
    rng = rng or random
    blink = cfg.eye_blink_s
    if category in COUNT_WORDS:
        raw = [(0.0, "eyes_on"), (min(blink, seconds), "eyes_off")]
    elif category == "got_it":
        step = max(blink, 0.25)
        raw = [(0.0, "eyes_on"), (step, "eyes_off"), (2 * step, "eyes_on"), (3 * step, "eyes_off")]
        raw = [(t, k) for t, k in raw if t < seconds - 0.05]  # two slow flashes, like a camera, if the line is long enough
    else:
        raw = [(0.0, "eyes_on")]
        t = rng.uniform(cfg.eye_blink_min_gap_s, cfg.eye_blink_max_gap_s)
        while t + blink < seconds - cfg.eye_blink_min_gap_s / 2:  # no blink right at the end of the line
            raw += [(t, "eyes_off"), (t + blink, "eyes_on")]
            t += blink + rng.uniform(cfg.eye_blink_min_gap_s, cfg.eye_blink_max_gap_s)
        raw.append((seconds, "eyes_off"))
    return [Event(min(t, seconds), kind) for t, kind in raw]


def choreograph(line: Line, cfg: InteractionConfig, move_mouth: bool, flash_eyes: bool, rng: random.Random | None = None) -> list[Event]:
    events: list[Event] = []
    if move_mouth:
        events += plan_mouth(line.envelope, line.hop_s, cfg)
    if flash_eyes:
        events += plan_eyes(line.category, line.seconds, cfg, rng)
    order = {"mouth_close": 0, "eyes_off": 0, "mouth_open": 1, "eyes_on": 1}  # closing first at equal times
    return sorted(events, key=lambda e: (e.t, order[e.kind]))


class MouthGovernor:
    """Charge budget for the jaw: opening spends it, being closed refills it.

    Decides at the moment of each planned open whether the jaw can afford it;
    a refused open also suppresses its matching close, so the mouth just
    stays shut for that syllable. Persists across lines, so many quick
    interactions in a row cannot run the drive flat.
    """

    def __init__(self, cfg: InteractionConfig, clock: Callable[[], float] = time.monotonic):
        self._cfg, self._clock = cfg, clock
        self.level = cfg.mouth_budget_s
        self._last = clock()
        self._open = False
        self.skipped = 0

    def _advance(self) -> None:
        now = self._clock()
        dt = max(0.0, now - self._last)
        self._last = now
        cfg = self._cfg
        self.level = max(0.0, self.level - dt) if self._open else min(cfg.mouth_budget_s, self.level + dt * cfg.mouth_recharge_per_s)

    def try_open(self) -> bool:
        self._advance()
        if self.level < self._cfg.mouth_max_open_s:  # must be able to afford the longest open we would allow
            self.skipped += 1
            return False
        self._open = True
        return True

    def closed(self) -> None:
        self._advance()
        self._open = False


class Performer:
    """Plays one voice line while moving the skeleton in time with it.

    Whatever happens (finished, interrupted, stopped, an error), the mouth is
    closed and the eyes are off again afterwards.
    """

    def __init__(
        self,
        player,
        skeleton,
        cfg: InteractionConfig,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        stop_event: threading.Event | None = None,
        rng: random.Random | None = None,
    ):
        self._rng = rng or random.Random()
        self.governor = MouthGovernor(cfg, clock)
        self._mouth_allowed = False
        self._player, self._skeleton, self._cfg = player, skeleton, cfg
        self._clock, self._sleep = clock, sleep
        self.stop_event = stop_event or threading.Event()

    def speak(
        self,
        line: Line,
        volume: float | None = None,
        move_mouth: bool | None = None,
        flash_eyes: bool | None = None,
        interrupt: Callable[[], bool] | None = None,
    ) -> bool:
        """True if the line played to the end; False if it was cut short."""
        cfg = self._cfg
        use_mouth = cfg.move_mouth if move_mouth is None else move_mouth
        use_eyes = cfg.flash_eyes if flash_eyes is None else flash_eyes
        events = choreograph(line, cfg, use_mouth, use_eyes, self._rng)
        handle = self._player.play(line.path, line.seconds, cfg.volume if volume is None else volume)
        start = self._clock()
        completed = True
        try:
            for event in events:
                if not self._wait_until(start + event.t, interrupt):
                    completed = False
                    break
                self._apply(event)
            if completed:
                completed = self._wait_until(start + line.seconds, interrupt)
        finally:
            if completed:
                handle.wait(2.0)
            else:
                handle.stop()
            self._rest(use_mouth, use_eyes)
        return completed

    def wiggle(self, part: str, period_s: float, cycles: int) -> int:
        """Moves just the mouth or the eyes at a steady rhythm, to find out how
        fast the real hardware can go. `period_s` is one open+close (or on+off)
        cycle; bounded so a typo cannot hammer the motor. Returns the cycles
        done (fewer if stopped), and always rests the part afterwards.
        """
        if part not in ("mouth", "eyes"):
            raise ValueError("part must be 'mouth' or 'eyes'")
        period_s = max(0.1, float(period_s))
        cycles = max(1, min(int(cycles), 20))
        on, off = ((self._skeleton.mouth_open, self._skeleton.mouth_close) if part == "mouth"
                   else (self._skeleton.eyes_on, self._skeleton.eyes_off))
        done = 0
        try:
            for _ in range(cycles):
                on()
                if not self.sleep_watching(period_s / 2):
                    break
                off()
                if not self.sleep_watching(period_s / 2):
                    break
                done += 1
        finally:
            self._rest(part == "mouth", part == "eyes")
        return done

    def sleep_watching(self, seconds: float, interrupt: Callable[[], bool] | None = None) -> bool:
        """Silent wait that still honors stop/interrupt. True if the full time elapsed."""
        return self._wait_until(self._clock() + seconds, interrupt)

    def _wait_until(self, deadline: float, interrupt: Callable[[], bool] | None) -> bool:
        while True:
            if self.stop_event.is_set() or (interrupt is not None and interrupt()):
                return False
            remaining = deadline - self._clock()
            if remaining <= 0:
                return True
            self._sleep(min(remaining, _SLICE_S))

    def _apply(self, event: Event) -> None:
        if event.kind == "mouth_open":
            self._mouth_allowed = self.governor.try_open()
            if not self._mouth_allowed:
                return
        elif event.kind == "mouth_close":
            if not self._mouth_allowed:
                return
            self.governor.closed()
        {
            "mouth_open": self._skeleton.mouth_open,
            "mouth_close": self._skeleton.mouth_close,
            "eyes_on": self._skeleton.eyes_on,
            "eyes_off": self._skeleton.eyes_off,
        }[event.kind]()

    def _rest(self, mouth: bool, eyes: bool) -> None:
        """Put back only what this line was allowed to move."""
        try:
            if mouth:
                self._skeleton.mouth_close()
                self.governor.closed()
            if eyes:
                self._skeleton.eyes_off()
        except Exception:  # noqa: BLE001 - never let cleanup mask the real problem
            pass

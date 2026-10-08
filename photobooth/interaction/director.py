from __future__ import annotations

import random
import threading
import time
from typing import Callable

from ..config import InteractionConfig
from .animation import Performer
from .voicebank import Line, VoiceBank

# The sequence's phases, as reported in status() and shown on the studio page.
IDLE, INVITING, GREETING, HOLDING, COUNTING, CAPTURING, RENDERING, PRINTING, FINISHING, COOLDOWN = (
    "idle", "inviting", "greeting", "holding", "counting", "capturing", "rendering", "printing", "finishing", "cooldown")


class _Presence:
    """Looks for people at most every `presence_poll_s` and keeps the streaks."""

    def __init__(self, session, cfg: InteractionConfig, clock: Callable[[], float]):
        self._session, self._cfg, self._clock = session, cfg, clock
        self._last = -1e9
        self.hits = 0
        self.misses = 0
        self.people = 0

    def poll(self) -> None:
        now = self._clock()
        if now - self._last < self._cfg.presence_poll_s:
            return
        self._last = now
        self.people = self._session.people()
        if self.people > 0:
            self.hits, self.misses = self.hits + 1, 0
        else:
            self.hits, self.misses = 0, self.misses + 1

    def reset(self) -> None:
        self.hits = self.misses = 0
        self._last = -1e9

    def arrived(self) -> bool:
        self.poll()
        return self.hits >= self._cfg.presence_confirmations

    def left(self) -> bool:
        self.poll()
        return self.misses >= self._cfg.absence_confirmations


class Director:
    """Runs one visitor sequence: invite them over, greet them when someone
    steps into frame, talk (instead of a visual timer) while they hold still,
    count three-two-one, and take the photo as the skeleton says "got it".

    The camera is held open for the whole sequence so presence checks are cheap
    and the capture lands exactly on the word. Triggers (button, motion sensor,
    the studio) all call start(); only one sequence runs at a time and a
    cooldown follows each one. Stop is honored within ~50 ms and always leaves
    the eyes off and the mouth closed.
    """

    def __init__(
        self,
        service,
        bank: VoiceBank,
        performer: Performer,
        cfg: InteractionConfig,
        stop_event: threading.Event | None = None,
        clock: Callable[[], float] = time.monotonic,
        rng: random.Random | None = None,
    ):
        self._service, self._bank, self._performer, self._cfg = service, bank, performer, cfg
        self._stop = stop_event or performer.stop_event
        self._clock = clock
        self._rng = rng or random.Random()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._busy = False
        self._cooldown_until = 0.0
        self._status: dict = {"state": IDLE, "outcome": None, "line": None, "people": 0, "error": None, "runs": 0}
        self._last_line: dict[str, str] = {}

    # ---- public control ----

    def available(self) -> bool:
        """True if a sequence could start right now."""
        return not self._busy and self._clock() >= self._cooldown_until and len(self._bank) > 0

    def status(self) -> dict:
        with self._lock:
            out = dict(self._status)
        out["running"] = self._busy
        remaining = max(0.0, self._cooldown_until - self._clock())
        out["cooldown_s"] = round(remaining, 1)
        if out["state"] == COOLDOWN and remaining <= 0:
            out["state"] = IDLE
        out["draft_stamp"] = getattr(self._service, "draft_stamp", 0)
        out["lines"] = len(self._bank)
        return out

    def start(self, skip_invites: bool = False, reason: str = "manual", ignore_cooldown: bool = False) -> bool:
        """Begins a sequence in the background. False if one is running, the
        cooldown is still on, or there are no voice lines.
        """
        with self._lock:
            if self._busy:
                return False
            if not ignore_cooldown and self._clock() < self._cooldown_until:
                return False
            if len(self._bank) == 0:
                self._status.update(state=IDLE, outcome="error", error="No voice lines found; run scripts/make_voice_lines.py")
                return False
            self._busy = True
            self._stop.clear()
        self._thread = threading.Thread(target=self._guarded_run, args=(skip_invites, reason), daemon=True, name="booth-sequence")
        self._thread.start()
        return True

    def say(self, line_id: str, volume: float | None = None, move_mouth: bool | None = None, flash_eyes: bool | None = None) -> bool:
        """Plays a single line (studio testing of the voice and the jaw/eyes)."""
        line = self._bank.get(line_id)
        with self._lock:
            if self._busy:
                return False
            self._busy = True
            self._stop.clear()

        def run() -> None:
            try:
                self._set(state=IDLE, outcome=None, error=None, line=line.id)
                self._performer.speak(line, volume=volume, move_mouth=move_mouth, flash_eyes=flash_eyes)
                self._set(outcome="said")
            except Exception as exc:  # noqa: BLE001 - reported through status
                self._set(outcome="error", error=str(exc))
            finally:
                self._busy = False

        self._thread = threading.Thread(target=run, daemon=True, name="booth-say")
        self._thread.start()
        return True

    def wiggle(self, part: str, period_s: float, cycles: int) -> bool:
        """Hardware check: move the mouth or eyes at a steady rate (studio only)."""
        if part not in ("mouth", "eyes"):
            raise ValueError("part must be 'mouth' or 'eyes'")
        with self._lock:
            if self._busy:
                return False
            self._busy = True
            self._stop.clear()

        def run() -> None:
            try:
                self._set(state=IDLE, outcome=None, error=None, line=f"wiggle {part} {period_s:.2f}s")
                done = self._performer.wiggle(part, period_s, cycles)
                self._set(outcome=f"wiggled {part} {done}x at {period_s:.2f}s")
            except Exception as exc:  # noqa: BLE001
                self._set(outcome="error", error=str(exc))
            finally:
                self._busy = False

        self._thread = threading.Thread(target=run, daemon=True, name="booth-wiggle")
        self._thread.start()
        return True

    def stop(self, wait_s: float = 3.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(wait_s)

    def join(self, timeout: float | None = None) -> None:
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    # ---- the sequence ----

    def _guarded_run(self, skip_invites: bool, reason: str) -> None:
        try:
            self.run_sequence(skip_invites=skip_invites, reason=reason)
        finally:
            self._busy = False

    def run_sequence(self, skip_invites: bool = False, reason: str = "manual") -> str:
        """Runs the whole sequence in the calling thread. Returns the outcome:
        captured | nobody | left | stopped | error.
        """
        cfg = self._cfg
        self._set(state=INVITING if not skip_invites else GREETING, outcome=None, error=None, people=0, line=None, reason=reason)
        outcome = "error"
        session = None
        try:
            session = self._service.open_visitor_session()
            presence = _Presence(session, cfg, self._clock)

            if skip_invites:
                presence.reset()
            elif not self._invite(presence):
                outcome = "stopped" if self._stop.is_set() else "nobody"
                if outcome == "nobody" and self._bank.has("nobody"):
                    self._say("nobody")
                return self._finish(outcome)

            if self._stop.is_set():
                return self._finish("stopped")

            if not self._brief(presence):
                outcome = "stopped" if self._stop.is_set() else "left"
                if outcome == "left" and self._bank.has("come_back"):
                    self._say("come_back")
                return self._finish(outcome)

            result = self._count_and_capture(session)
            if result is None:
                return self._finish("stopped")
            if result.get("people", 0) == 0:
                self._set(people=0)
                if self._bank.has("come_back"):
                    self._say("come_back")
                return self._finish("left")
            self._set(people=result["people"])
        except Exception as exc:  # noqa: BLE001 - reported through status; the skeleton is rested below
            self._set(error=str(exc))
            return self._finish("error")
        finally:
            if session is not None:
                session.close()

        try:
            self._deliver()
        except Exception as exc:  # noqa: BLE001
            self._set(error=str(exc))
            return self._finish("error")
        return self._finish("stopped" if self._stop.is_set() else "captured")

    def _invite(self, presence: _Presence) -> bool:
        """Invites until someone steps into frame. False if nobody did (or stop)."""
        cfg = self._cfg
        for _ in range(max(1, cfg.invite_max)):
            if self._stop.is_set():
                return False
            if presence.arrived():
                return True
            if self._bank.has("invite"):
                self._say("invite", interrupt=presence.arrived)
                if presence.hits >= cfg.presence_confirmations:
                    return True
            if not self._performer.sleep_watching(cfg.invite_gap_s, interrupt=presence.arrived):
                if self._stop.is_set():
                    return False
                return True
        return presence.arrived()

    def _brief(self, presence: _Presence) -> bool:
        """Welcome and the 'hold still' talk. False if the visitor left or stop."""
        cfg = self._cfg
        self._set(state=GREETING)
        presence.reset()
        if cfg.welcome and self._bank.has("welcome"):
            if not self._say("welcome", interrupt=presence.left):
                return False
        self._set(state=HOLDING)
        if self._bank.has("filler"):
            if not self._say("filler", interrupt=presence.left):
                return False
        if cfg.hold_seconds > 0 and not self._performer.sleep_watching(cfg.hold_seconds, interrupt=presence.left):
            return False
        return not presence.left()

    def _count_and_capture(self, session) -> dict | None:
        """Three, two, one, then 'got it' with the photo taken as it starts."""
        cfg = self._cfg
        self._set(state=COUNTING)
        for word in ("three", "two", "one"):
            if not self._say(word):
                return None
            if not self._performer.sleep_watching(cfg.count_gap_s):
                return None
        self._set(state=CAPTURING)
        box: dict = {}

        def shoot() -> None:
            try:
                box["result"] = session.capture()
            except Exception as exc:  # noqa: BLE001 - re-raised on the sequence thread
                box["error"] = exc

        shooter = threading.Thread(target=shoot, daemon=True, name="booth-capture")
        shooter.start()
        self._say("got_it")
        shooter.join()
        if "error" in box:
            raise box["error"]
        return box.get("result")

    def _deliver(self) -> None:
        """Render the reel (and print it if configured), talking while it works."""
        cfg = self._cfg
        self._set(state=RENDERING)
        self._service.draft_render()
        if cfg.auto_print:
            self._set(state=PRINTING)
            box: dict = {}

            def do_print() -> None:
                try:
                    box["result"] = self._service.draft_print()
                except Exception as exc:  # noqa: BLE001
                    box["error"] = exc

            printer = threading.Thread(target=do_print, daemon=True, name="booth-print")
            printer.start()
            if self._bank.has("printing"):
                self._say("printing")
            printer.join()
            if "error" in box:
                raise box["error"]
            result = box.get("result")
            if result is not None and getattr(result, "error", None):
                raise RuntimeError(result.error)
        self._set(state=FINISHING)
        if self._bank.has("done"):
            self._say("done")

    # ---- helpers ----

    def _say(self, category: str, interrupt: Callable[[], bool] | None = None) -> bool:
        line = self._bank.pick(category, self._rng, avoid=self._last_line.get(category))
        self._last_line[category] = line.id
        self._set(line=line.id)
        return self._performer.speak(line, interrupt=interrupt)

    def _finish(self, outcome: str) -> str:
        with self._lock:
            self._status.update(state=COOLDOWN if outcome in ("captured", "nobody", "left") else IDLE, outcome=outcome,
                                runs=self._status["runs"] + 1)
        if outcome in ("captured", "nobody", "left"):
            self._cooldown_until = self._clock() + self._cfg.cooldown_s
        return outcome

    def _set(self, **changes) -> None:
        with self._lock:
            self._status.update(changes)

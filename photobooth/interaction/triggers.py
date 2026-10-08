from __future__ import annotations

import threading
import time
from typing import Callable


class TriggerWatcher:
    """Polls a digital input and calls `on_trigger` once per press (or per
    motion event) after it has stayed active for the debounce time.

    Polling rather than kernel edge events: a button press every few seconds
    is nowhere near fast enough to need them, and it works the same with a
    simulated input. The input must go inactive again before it can fire a
    second time, so a held button or a PIR that stays high fires once.
    """

    def __init__(
        self,
        read: Callable[[], bool],
        on_trigger: Callable[[], object],
        active_low: bool = True,
        debounce_s: float = 0.05,
        poll_s: float = 0.02,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._read, self._on_trigger = read, on_trigger
        self._active_low, self._debounce_s, self._poll_s = active_low, debounce_s, poll_s
        self._clock, self._sleep = clock, sleep
        self._active_since: float | None = None
        self._armed = True
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.fired = 0

    def poll_once(self) -> bool:
        """One look at the input. True if it fired the trigger on this look."""
        level = bool(self._read())
        active = (not level) if self._active_low else level
        if not active:
            self._active_since, self._armed = None, True
            return False
        now = self._clock()
        if self._active_since is None:
            self._active_since = now
        if self._armed and now - self._active_since >= self._debounce_s:
            self._armed = False
            self.fired += 1
            self._on_trigger()
            return True
        return False

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="booth-trigger")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(1.0)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.poll_once()
            except Exception:  # noqa: BLE001 - a flaky read must not kill the watcher
                pass
            self._sleep(self._poll_s)

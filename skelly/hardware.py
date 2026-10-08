from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Protocol


@dataclass(frozen=True)
class PinMap:
    """Radxa Zero 3W header pins as wired on the skelly (gpiochip3 line
    numbers confirmed with `gpiofind` in 2025; see docs/hardware-research.md).
    """

    chip: str = "/dev/gpiochip3"
    mouth: int = 20  # PIN_7
    eyes: int = 1  # PIN_11
    body: int = 8  # PIN_15
    spare: int = 7  # PIN_36


class GpioBackend(Protocol):
    def write(self, line: int, value: bool) -> None: ...
    def close(self) -> None: ...


class SimGpioBackend:
    """No hardware: records levels so the dev machine and tests can run."""

    def __init__(self) -> None:
        self.levels: dict[int, bool] = {}

    def write(self, line: int, value: bool) -> None:
        self.levels[line] = value

    def close(self) -> None:
        pass


class PeripheryGpioBackend:
    """python-periphery GPIO (already installed on the board). Chosen over
    python-gpiod because a venv's `pip install gpiod` is the v2 API, which
    has no `Chip.get_line` -- code written for v1 then fails silently.

    Lines are opened lazily with the first value to write ("high"/"low"
    direction), so opening a line never drives a level nobody asked for,
    and are then held open for the process lifetime rather than released
    after every write.
    """

    def __init__(self, chip: str, gpio_factory: Callable | None = None) -> None:
        if gpio_factory is None:
            from periphery import GPIO  # imported lazily: absent on dev machines

            gpio_factory = GPIO
        self._chip = chip
        self._factory = gpio_factory
        self._lines: dict[int, object] = {}

    def write(self, line: int, value: bool) -> None:
        gpio = self._lines.get(line)
        if gpio is None:
            self._lines[line] = self._factory(self._chip, line, "high" if value else "low")
        else:
            gpio.write(value)

    def close(self) -> None:
        for gpio in self._lines.values():
            gpio.close()
        self._lines.clear()


def build_backend(pins: PinMap, simulate: bool | None = None) -> GpioBackend:
    """simulate=None auto-detects: real GPIO only if the chip device exists
    and python-periphery imports; otherwise falls back to simulation.
    """
    if simulate is None:
        simulate = not os.path.exists(pins.chip)
    if not simulate:
        try:
            return PeripheryGpioBackend(pins.chip)
        except ImportError:
            pass
    return SimGpioBackend()


class Skeleton:
    """Eyes, mouth and body motor. Construction touches no hardware."""

    def __init__(
        self,
        backend: GpioBackend | None = None,
        pins: PinMap | None = None,
        mouth_open_level: bool = True,
        mouth_rest_open: bool = True,
    ) -> None:
        self.pins = pins or PinMap()
        self.backend = backend or build_backend(self.pins)
        self._mouth_open_level = mouth_open_level
        # The 2025 code ends every run with the mouth pin HIGH ("to prevent
        # limp jaw", per its comment). Kept configurable until we confirm on
        # the real skeleton what that level does physically.
        self._mouth_rest_open = mouth_rest_open
        self._state = {"eyes": False, "mouth_open": False, "body": False}

    def eyes_on(self) -> None:
        self._set("eyes", self.pins.eyes, True)

    def eyes_off(self) -> None:
        self._set("eyes", self.pins.eyes, False)

    def mouth_open(self) -> None:
        self.backend.write(self.pins.mouth, self._mouth_open_level)
        self._state["mouth_open"] = True

    def mouth_close(self) -> None:
        self.backend.write(self.pins.mouth, not self._mouth_open_level)
        self._state["mouth_open"] = False

    def body_start(self) -> None:
        self._set("body", self.pins.body, True)

    def body_stop(self) -> None:
        self._set("body", self.pins.body, False)

    def state(self) -> dict[str, bool]:
        return dict(self._state)

    def cleanup(self) -> None:
        self.eyes_off()
        self.body_stop()
        if self._mouth_rest_open:
            self.mouth_open()
        else:
            self.mouth_close()
        self.backend.close()

    def _set(self, key: str, line: int, value: bool) -> None:
        self.backend.write(line, value)
        self._state[key] = value

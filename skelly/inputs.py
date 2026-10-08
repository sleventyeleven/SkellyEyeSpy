from __future__ import annotations

from typing import Callable, Protocol


class DigitalInput(Protocol):
    def read(self) -> bool: ...
    def close(self) -> None: ...


class SimInput:
    """No hardware: the level is whatever a test or the studio last set."""

    def __init__(self, level: bool = False) -> None:
        self.level = level

    def read(self) -> bool:
        return self.level

    def close(self) -> None:
        pass


class PeripheryInput:
    """One GPIO line as an input (python-periphery, like the outputs).

    A button wired to ground needs the pull-up; the board's kernel may not
    support software bias, in which case an external pull-up resistor is needed
    and the line is opened without one.
    """

    def __init__(self, chip: str, line: int, pull_up: bool = True, gpio_factory: Callable | None = None) -> None:
        if gpio_factory is None:
            from periphery import GPIO  # imported lazily: absent on dev machines

            gpio_factory = GPIO
        try:
            self._gpio = gpio_factory(chip, line, "in", bias="pull_up" if pull_up else "default")
        except (TypeError, ValueError, OSError):
            self._gpio = gpio_factory(chip, line, "in")

    def read(self) -> bool:
        return bool(self._gpio.read())

    def close(self) -> None:
        self._gpio.close()

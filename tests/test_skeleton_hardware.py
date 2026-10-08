import pytest

from skelly.hardware import PeripheryGpioBackend, PinMap, SimGpioBackend, Skeleton, build_backend


class FakeGpio:
    opened: list = []

    def __init__(self, chip, line, direction):
        self.chip, self.line, self.direction = chip, line, direction
        self.writes: list = []
        self.closed = False
        FakeGpio.opened.append(self)

    def write(self, value):
        self.writes.append(value)

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def _reset_fake():
    FakeGpio.opened = []


def test_construction_touches_no_hardware():
    backend = SimGpioBackend()
    Skeleton(backend=backend)
    assert backend.levels == {}


def test_eyes_body_levels_and_state():
    backend, pins = SimGpioBackend(), PinMap()
    skel = Skeleton(backend=backend, pins=pins)
    skel.eyes_on()
    skel.body_start()
    assert backend.levels[pins.eyes] is True and backend.levels[pins.body] is True
    assert skel.state() == {"eyes": True, "mouth_open": False, "body": True}
    skel.eyes_off()
    assert backend.levels[pins.eyes] is False


def test_mouth_open_level_is_configurable():
    backend, pins = SimGpioBackend(), PinMap()
    skel = Skeleton(backend=backend, pins=pins, mouth_open_level=False)
    skel.mouth_open()
    assert backend.levels[pins.mouth] is False
    skel.mouth_close()
    assert backend.levels[pins.mouth] is True


def test_cleanup_turns_off_eyes_body_and_rests_mouth_open_by_default():
    backend, pins = SimGpioBackend(), PinMap()
    skel = Skeleton(backend=backend, pins=pins)
    skel.eyes_on()
    skel.body_start()
    skel.cleanup()
    assert backend.levels[pins.eyes] is False
    assert backend.levels[pins.body] is False
    assert backend.levels[pins.mouth] is True


def test_periphery_opens_line_with_first_value_then_reuses_it():
    backend = PeripheryGpioBackend("/dev/gpiochip3", gpio_factory=FakeGpio)
    backend.write(1, True)
    backend.write(1, False)
    assert len(FakeGpio.opened) == 1
    gpio = FakeGpio.opened[0]
    assert (gpio.chip, gpio.line, gpio.direction) == ("/dev/gpiochip3", 1, "high")
    assert gpio.writes == [False]


def test_periphery_first_low_write_opens_low():
    backend = PeripheryGpioBackend("/dev/gpiochip3", gpio_factory=FakeGpio)
    backend.write(8, False)
    assert FakeGpio.opened[0].direction == "low"


def test_periphery_close_closes_all_lines():
    backend = PeripheryGpioBackend("/dev/gpiochip3", gpio_factory=FakeGpio)
    backend.write(1, True)
    backend.write(20, True)
    backend.close()
    assert all(g.closed for g in FakeGpio.opened)


def test_build_backend_falls_back_to_sim_without_chip_device():
    backend = build_backend(PinMap(chip="/dev/definitely-not-a-chip"))
    assert isinstance(backend, SimGpioBackend)

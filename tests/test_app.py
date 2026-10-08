import pytest

import app as app_module
from skelly.hardware import PinMap, SimGpioBackend, Skeleton


@pytest.fixture
def client(monkeypatch):
    backend = SimGpioBackend()
    monkeypatch.setattr(app_module, "skeleton", Skeleton(backend=backend))
    monkeypatch.setattr(app_module.time, "sleep", lambda _s: None)
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client(), backend


def test_eyes_on_off_drive_the_eye_pin(client):
    c, backend = client
    pins = PinMap()
    assert c.post("/api/eyes/on").get_json()["status"] == "success"
    assert backend.levels[pins.eyes] is True
    c.post("/api/eyes/off")
    assert backend.levels[pins.eyes] is False


def test_mouth_pulses_open_then_closed(client):
    c, backend = client
    assert c.post("/api/mouth/open").get_json()["status"] == "success"
    assert backend.levels[PinMap().mouth] is False  # ends closed, like the 2025 app


def test_body_move_ends_stopped(client):
    c, backend = client
    assert c.post("/api/body/move").get_json()["status"] == "success"
    assert backend.levels[PinMap().body] is False


def test_status_reports_real_state_and_backend(client):
    c, _ = client
    c.post("/api/eyes/on")
    data = c.get("/api/status").get_json()["data"]
    assert data["gpio_pins"]["eye_leds"] is True
    assert data["gpio_backend"] == "SimGpioBackend"


def test_gpio_failure_returns_500_not_crash(monkeypatch):
    class Broken:
        def eyes_on(self):
            raise OSError("gpio busy")

    monkeypatch.setattr(app_module, "skeleton", Broken())
    response = app_module.app.test_client().post("/api/eyes/on")
    assert response.status_code == 500
    assert "gpio busy" in response.get_json()["message"]


def test_booth_registration_failure_does_not_break_app(monkeypatch, tmp_path):
    monkeypatch.setenv("PHOTOBOOTH_CONFIG", str(tmp_path / "missing.yaml"))
    from flask import Flask

    assert app_module.register_photobooth(Flask("x")) is False

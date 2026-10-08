import cv2
import numpy as np
import pytest
from flask import Flask
from PIL import Image

from photobooth.compose.strip import crop_around_person, render_strip, strip_height
from photobooth.config import PhotoBoothConfig, StripConfig
from photobooth.printer import PRINT_WIDTH_PX
from photobooth.printer import phomemo as phomemo_module
from photobooth.printer.phomemo import PhomemoM02Printer
from photobooth.service import _WARMUP_FRAMES, PhotoBoothService
from photobooth.web import create_blueprint

from test_booth_service import FakeCamera, FakePrinter, FakeSegmenter


def test_crop_has_requested_aspect_and_follows_the_person():
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    mask = np.zeros((720, 1280), dtype=np.uint8)
    mask[:, 1000:1100] = 255  # person standing far right
    crop, mask_crop = crop_around_person(frame, mask, 4, 3)
    assert crop.shape[:2] == (720, 960) and mask_crop.shape == crop.shape[:2]
    assert (mask_crop > 0).any()  # person is inside the crop, not cut off


def test_crop_without_person_is_centered():
    frame = np.arange(720 * 1280 * 3, dtype=np.uint32).reshape(720, 1280, 3).astype(np.uint8)
    crop, _ = crop_around_person(frame, np.zeros((720, 1280), np.uint8), 4, 3)
    assert np.array_equal(crop, frame[:, 160:1120])


def test_strip_is_printer_width_and_height_matches_formula():
    cfg = StripConfig(shots=4)
    shots = [np.full((cfg.frame_height, cfg.frame_width, 3), 80 * i, np.uint8) for i in range(4)]
    strip = render_strip(shots, cfg, "Oct 07, 2026")
    assert strip.size == (PRINT_WIDTH_PX, strip_height(cfg, 4))


def test_strip_needs_at_least_one_shot():
    with pytest.raises(ValueError):
        render_strip([], StripConfig(), "x")


@pytest.fixture
def strip_config(tmp_path):
    paths = []
    for i, shade in enumerate((40, 200)):
        path = tmp_path / f"bg{i}.png"
        cv2.imwrite(str(path), np.full((100, 100, 3), shade, dtype=np.uint8))
        paths.append(str(path))
    cfg = PhotoBoothConfig()
    cfg.compose.backgrounds = paths
    cfg.compose.sepia_strength = cfg.compose.vignette_strength = cfg.compose.grain_amount = 0
    # these tests are about layout/printing, so keep the background order fixed and unvaried
    cfg.compose.randomize_backgrounds = cfg.compose.vary_backgrounds = False
    cfg.compose.morph_probability = 0.0
    cfg.printer.enabled = True
    cfg.printer.bluetooth_address = "AA:BB:CC:DD:EE:FF"
    return cfg


def make_service(config, tmp_path):
    return PhotoBoothService(
        config,
        output_dir=str(tmp_path / "out"),
        camera_factory=FakeCamera,
        segmenter_factory=FakeSegmenter,
        printer_factory=FakePrinter,
    )


def test_each_shot_uses_the_next_background_and_cycles(strip_config, tmp_path):
    service = make_service(strip_config, tmp_path)
    service.strip_start(3)
    names = [service.strip_shot()["background"] for _ in range(3)]
    assert names == ["bg0", "bg1", "bg0"]
    right_edge = [cv2.imread(service.strip_shot_path(i))[130, -5, 0] for i in range(3)]
    assert right_edge[0] < 100 < right_edge[1] and right_edge[2] < 100  # FakeSegmenter: right half is background


def test_finish_builds_one_strip_and_prints_it(strip_config, tmp_path):
    service = make_service(strip_config, tmp_path)
    service.strip_start(4)
    for _ in range(4):
        service.strip_shot()
    result = service.strip_finish()
    strip = Image.open(result.image_path)
    assert strip.size == (PRINT_WIDTH_PX, strip_height(strip_config.strip, 4))
    assert result.printed and FakePrinter.printed[-1] == strip.size
    assert service.last is result


def test_cannot_exceed_shot_count_or_shoot_before_start(strip_config, tmp_path):
    service = make_service(strip_config, tmp_path)
    with pytest.raises(RuntimeError):
        service.strip_shot()
    service.strip_start(1)
    service.strip_shot()
    with pytest.raises(RuntimeError):
        service.strip_shot()


def test_finish_without_shots_raises(strip_config, tmp_path):
    service = make_service(strip_config, tmp_path)
    service.strip_start(2)
    with pytest.raises(RuntimeError):
        service.strip_finish()


def test_strip_endpoints_full_flow(strip_config, tmp_path):
    app = Flask(__name__)
    app.register_blueprint(create_blueprint(make_service(strip_config, tmp_path)))
    client = app.test_client()

    assert client.post("/api/photobooth/strip/shot").status_code == 500  # not started
    assert client.post("/api/photobooth/strip/start?shots=2").get_json()["shots"] == 2
    first = client.post("/api/photobooth/strip/shot").get_json()
    assert first["index"] == 0 and first["total"] == 2
    client.post("/api/photobooth/strip/shot")
    assert client.get("/api/photobooth/strip/shot/1").status_code == 200
    assert client.get("/api/photobooth/strip/shot/5").status_code == 404
    done = client.post("/api/photobooth/strip/finish?print=0").get_json()
    assert done["status"] == "success" and done["printed"] is False
    assert client.get("/api/photobooth/last").status_code == 200


def test_one_shot_reel_captures_once_and_uses_a_different_background_per_frame(strip_config, tmp_path):
    FakeCamera.frames_read = 0
    service = make_service(strip_config, tmp_path)
    result = service.strip_one_shot(shots=4)

    assert FakeCamera.frames_read == _WARMUP_FRAMES  # one capture, not four
    strip = Image.open(result.image_path)
    assert strip.size == (PRINT_WIDTH_PX, strip_height(strip_config.strip, 4))
    # backgrounds alternate dark/light/dark/light; the top-right corner is always background
    corners = [cv2.imread(str(tmp_path / "out" / _only_strip_dir(tmp_path) / f"shot_{i}.png"))[3, -3, 0] for i in range(4)]
    assert corners[0] < 100 < corners[1] and corners[2] < 100 < corners[3]
    assert result.printed and FakePrinter.printed[-1] == strip.size
    assert service.last is result


def test_one_shot_defaults_to_configured_frame_count(strip_config, tmp_path):
    strip_config.strip.shots = 3
    result = make_service(strip_config, tmp_path).strip_one_shot()
    assert Image.open(result.image_path).height == strip_height(strip_config.strip, 3)


def test_one_shot_endpoint(strip_config, tmp_path):
    app = Flask(__name__)
    app.register_blueprint(create_blueprint(make_service(strip_config, tmp_path)))
    client = app.test_client()
    body = client.post("/api/photobooth/strip/oneshot?shots=2&print=0").get_json()
    assert body["status"] == "success" and body["printed"] is False
    assert client.get("/api/photobooth/last").status_code == 200
    assert b"oneshot" in client.get("/photobooth/view").data


def test_one_shot_reports_capture_failure(strip_config, tmp_path):
    from test_booth_service import NoFrameCamera

    service = PhotoBoothService(
        strip_config, output_dir=str(tmp_path / "out"), camera_factory=NoFrameCamera,
        segmenter_factory=FakeSegmenter, printer_factory=FakePrinter,
    )
    app = Flask(__name__)
    app.register_blueprint(create_blueprint(service))
    assert app.test_client().post("/api/photobooth/strip/oneshot").status_code == 500


def _only_strip_dir(tmp_path):
    import os

    (name,) = [d for d in os.listdir(tmp_path / "out") if d.startswith("strip_")]
    return name

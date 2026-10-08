import cv2
import numpy as np
import pytest
from flask import Flask
from PIL import Image

from photobooth.config import PhotoBoothConfig
from photobooth.printer import PRINT_WIDTH_PX, preview_image
from photobooth.service import _WARMUP_FRAMES, PhotoBoothService
from photobooth.web import create_blueprint


class FakeCamera:
    frames_read = 0

    def __init__(self, _config, frame=None):
        self._frame = np.full((720, 1280, 3), 120, dtype=np.uint8) if frame is None else frame

    def read(self):
        FakeCamera.frames_read += 1
        return self._frame

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class NoFrameCamera(FakeCamera):
    def read(self):
        return None


class FakeSegmenter:
    closed = False

    def person_mask(self, frame):
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        mask[:, : frame.shape[1] // 2] = 255
        return mask

    def close(self):
        FakeSegmenter.closed = True


class FakePrinter:
    printed = []
    fail = False

    def __init__(self, bluetooth_address=None, dry_run_path=None, **_printer_settings):
        pass

    def __enter__(self):
        if FakePrinter.fail:
            raise ConnectionRefusedError("printer off")
        return self

    def __exit__(self, *_):
        pass

    def print_image(self, image):
        FakePrinter.printed.append(image.size)


@pytest.fixture(autouse=True)
def _reset():
    FakeCamera.frames_read = 0
    FakeSegmenter.closed = False
    FakePrinter.printed = []
    FakePrinter.fail = False


@pytest.fixture
def config(tmp_path):
    bg = tmp_path / "bg.png"
    cv2.imwrite(str(bg), np.full((100, 100, 3), 50, dtype=np.uint8))
    cfg = PhotoBoothConfig()
    cfg.compose.background = str(bg)
    cfg.printer.enabled = True
    cfg.printer.bluetooth_address = "AA:BB:CC:DD:EE:FF"
    return cfg


def make_service(config, tmp_path, **overrides):
    kwargs = dict(
        output_dir=str(tmp_path / "out"),
        camera_factory=FakeCamera,
        segmenter_factory=FakeSegmenter,
        printer_factory=FakePrinter,
    )
    kwargs.update(overrides)
    return PhotoBoothService(config, **kwargs)


def test_snap_saves_image_and_print_preview_and_prints(config, tmp_path):
    result = make_service(config, tmp_path).snap()
    assert result.printed and result.error is None
    saved = Image.open(result.image_path)
    assert saved.width == config.compose.working_width  # downscaled from 1280
    preview = Image.open(result.preview_path)
    assert preview.width == PRINT_WIDTH_PX
    assert FakePrinter.printed  # printer was actually handed an image
    debug = Image.open(result.debug_path)
    assert debug.width == 3 * saved.width  # raw | mask | result
    assert FakeCamera.frames_read == _WARMUP_FRAMES


def test_print_failure_is_reported_not_raised(config, tmp_path):
    FakePrinter.fail = True
    result = make_service(config, tmp_path).snap()
    assert result.printed is False
    assert "ConnectionRefusedError" in result.error
    assert Image.open(result.image_path)  # photo still saved


def test_print_it_false_skips_printer(config, tmp_path):
    result = make_service(config, tmp_path).snap(print_it=False)
    assert not result.printed and FakePrinter.printed == []


def test_no_frame_raises(config, tmp_path):
    service = make_service(config, tmp_path, camera_factory=NoFrameCamera)
    with pytest.raises(RuntimeError):
        service.snap()


def test_close_closes_segmenter(config, tmp_path):
    service = make_service(config, tmp_path)
    service.snap()
    service.close()
    assert FakeSegmenter.closed


def test_preview_image_is_black_and_white_at_print_width():
    preview = preview_image(Image.new("RGB", (800, 400), (255, 255, 255)))
    assert preview.size[0] == PRINT_WIDTH_PX
    assert set(np.unique(np.array(preview))) <= {0, 255}


def test_endpoint_snap_then_fetch_last(config, tmp_path):
    app = Flask(__name__)
    app.register_blueprint(create_blueprint(make_service(config, tmp_path)))
    client = app.test_client()

    assert client.get("/api/photobooth/last").status_code == 404

    body = client.post("/api/photobooth/snap").get_json()
    assert body["status"] == "success" and body["printed"] is True
    assert client.get("/api/photobooth/last").status_code == 200
    assert client.get("/api/photobooth/last_print").status_code == 200
    assert client.get("/api/photobooth/last_debug").status_code == 200


def test_endpoint_print_zero_skips_printing(config, tmp_path):
    app = Flask(__name__)
    app.register_blueprint(create_blueprint(make_service(config, tmp_path)))
    body = app.test_client().post("/api/photobooth/snap?print=0").get_json()
    assert body["printed"] is False and FakePrinter.printed == []


def test_relative_output_dir_still_serves_files(config, tmp_path, monkeypatch):
    # send_file resolves relative paths against the Flask app folder, not the CWD
    monkeypatch.chdir(tmp_path)
    app = Flask(__name__, root_path=str(tmp_path / "elsewhere"))
    app.register_blueprint(create_blueprint(make_service(config, tmp_path, output_dir="rel_out")))
    client = app.test_client()
    client.post("/api/photobooth/snap")
    assert client.get("/api/photobooth/last").status_code == 200


def test_endpoint_reports_capture_failure_as_500(config, tmp_path):
    app = Flask(__name__)
    app.register_blueprint(create_blueprint(make_service(config, tmp_path, camera_factory=NoFrameCamera)))
    response = app.test_client().post("/api/photobooth/snap")
    assert response.status_code == 500
    assert response.get_json()["status"] == "error"


def test_aim_jpeg_is_a_decodable_frame_with_tinted_person(config, tmp_path):
    service = make_service(config, tmp_path)
    data = service.aim_jpeg()
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert frame.shape[1] == config.compose.working_width
    left, right = frame[300, 50], frame[300, -50]  # FakeSegmenter marks the left half as person
    assert left[1] > right[1]  # green channel boosted on the person side


def test_aim_endpoint_and_view_page(config, tmp_path):
    app = Flask(__name__)
    app.register_blueprint(create_blueprint(make_service(config, tmp_path)))
    client = app.test_client()
    response = client.get("/api/photobooth/aim")
    assert response.status_code == 200 and response.mimetype == "image/jpeg"
    page = client.get("/photobooth/view").data
    assert b"/api/photobooth/aim" in page and b"/api/photobooth/snap" in page and b"countdown" in page.lower()


# ---------- visitor session (camera held open for the spoken sequence) ----------

class ReleasableCamera(FakeCamera):
    opened = 0
    released = 0

    def __init__(self, config):
        super().__init__(config)
        ReleasableCamera.opened += 1

    def release(self):
        ReleasableCamera.released += 1


def test_visitor_session_holds_the_camera_and_captures_a_draft(config, tmp_path):
    ReleasableCamera.opened = ReleasableCamera.released = 0
    service = make_service(config, tmp_path, camera_factory=ReleasableCamera)
    stamp = service.draft_stamp
    session = service.open_visitor_session()
    assert ReleasableCamera.opened == 1  # opened once for the whole sequence
    assert session.people() == 1
    assert session.people() == 1 and ReleasableCamera.opened == 1
    with pytest.raises(RuntimeError, match="in use"):
        service.draft_capture()  # the aim view / snapshots must wait
    with pytest.raises(RuntimeError, match="already"):
        service.open_visitor_session()
    info = session.capture()
    assert info["people"] == 1 and service.draft_stamp == stamp + 1
    session.close()
    session.close()  # harmless twice
    assert ReleasableCamera.released == 1
    assert service.draft_render()["shots"] >= 1  # the captured frame is the current draft
    assert service.draft_capture()["people"] == 1  # camera is free again


def test_visitor_session_releases_the_busy_flag_if_the_camera_fails(config, tmp_path):
    def broken(_config):
        raise RuntimeError("no camera")

    service = make_service(config, tmp_path, camera_factory=broken)
    with pytest.raises(RuntimeError, match="no camera"):
        service.open_visitor_session()
    assert service._visitor_active is False  # a failed open must not lock the booth out

import socket

import cv2
import numpy as np
from flask import Flask
from PIL import Image

from _tolerance import close
from photobooth.compose.framing import place_person
from photobooth.config import PhotoBoothConfig
from photobooth.pipeline import compose_normalized
from photobooth.printer import PRINT_WIDTH_PX, image_to_escpos
from photobooth.printer.phomemo import PhomemoM02Printer
from photobooth.printer.raster import image_to_escpos_job
from photobooth.service import PhotoBoothService
from photobooth.web import create_blueprint

from test_booth_service import FakeCamera, FakePrinter, FakeSegmenter


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class RecordingSock:
    def __init__(self, clock):
        self._clock = clock
        self.sends = []  # (time, bytes)

    def sendall(self, data):
        self.sends.append((self._clock(), bytes(data)))

    def close(self):
        pass


def black_image(rows):
    return Image.new("L", (PRINT_WIDTH_PX, rows), color=0)


def make_printer(clock, **kwargs):
    printer = PhomemoM02Printer(bluetooth_address="AA:BB:CC:DD:EE:FF", clock=clock, sleep=clock.sleep, **kwargs)
    printer._sock = RecordingSock(clock)
    return printer


def block_sends(printer):
    """(time, rows) for the first chunk of every GS v 0 block."""
    out = []
    for t, data in printer._sock.sends:
        if data.startswith(b"\x1dv0"):
            out.append((t, int.from_bytes(data[6:8], "little")))
    return out


def test_job_pieces_reassemble_to_the_same_bytes():
    image = black_image(600)
    job = image_to_escpos_job(image)
    assert [b.rows for b in job.blocks] == [255, 255, 90] and job.total_rows == 600
    assert job.to_bytes() == image_to_escpos(image)


def test_everything_is_sent_in_order():
    clock = FakeClock()
    printer = make_printer(clock)
    image = black_image(1000)
    printer.print_image(image)
    assert b"".join(data for _t, data in printer._sock.sends) == image_to_escpos(image)


def test_unprinted_backlog_never_exceeds_the_limit():
    clock = FakeClock()
    printer = make_printer(clock, lines_per_second=100.0, max_backlog_lines=400)
    printer.print_image(black_image(1224))  # a 4-frame strip

    sent_rows = 0
    sends = block_sends(printer)
    assert len(sends) == 5
    for t, rows in sends:
        sent_rows += rows
        printed_estimate = t * 100.0
        assert sent_rows - printed_estimate <= 400 + 1e-6
    assert sends[1][0] > 0  # the second block had to wait; it was not a single burst


def test_first_block_goes_out_immediately():
    clock = FakeClock()
    printer = make_printer(clock)
    printer.print_image(black_image(200))
    assert block_sends(printer)[0][0] == 0


def test_connection_is_held_until_the_estimated_end_of_the_print():
    clock = FakeClock()
    printer = make_printer(clock, lines_per_second=100.0, tail_pad_s=1.5)
    printer.print_image(black_image(1000))
    assert clock.now >= 1000 / 100.0 + 1.5  # closing earlier can cut the print off


def test_slower_rate_means_longer_gaps():
    fast, slow = FakeClock(), FakeClock()
    p_fast, p_slow = make_printer(fast, lines_per_second=200.0), make_printer(slow, lines_per_second=50.0)
    p_fast.print_image(black_image(1000))
    p_slow.print_image(black_image(1000))
    assert slow.now > fast.now


def test_printer_replies_are_recorded_while_waiting():
    clock = FakeClock()

    class TalkingSock(RecordingSock):
        replied = False

        def settimeout(self, _t):
            pass

        def recv(self, _n):
            if not TalkingSock.replied:
                TalkingSock.replied = True
                return b"\x1a\x0f"
            clock.sleep(0.1)
            raise socket.timeout()

    printer = PhomemoM02Printer(bluetooth_address="AA:BB:CC:DD:EE:FF", clock=clock, sleep=clock.sleep)
    printer._sock = TalkingSock(clock)
    printer.print_image(black_image(300))
    assert printer.responses == [b"\x1a\x0f"]


def test_dry_run_still_writes_the_full_payload(tmp_path):
    path = tmp_path / "p.bin"
    PhomemoM02Printer(dry_run_path=str(path)).print_image(black_image(50))
    assert path.read_bytes() == image_to_escpos(black_image(50))


# ---- old-timey overlay switch ---------------------------------------------------------


def _scene():
    frame = np.full((360, 640, 3), 90, dtype=np.uint8)
    mask = np.zeros((360, 640), np.uint8)
    frame[60:360, 250:390] = (0, 0, 255)
    mask[60:360, 250:390] = 255
    background = np.full((100, 100, 3), 200, dtype=np.uint8)
    return frame, mask, background


def test_vintage_off_returns_the_plain_composite():
    frame, mask, background = _scene()
    cfg = PhotoBoothConfig().compose
    plain = compose_normalized(frame, mask, background, (360, 270), cfg, vintage=False)
    placed = place_person(frame, mask, background, (360, 270), cfg.person_height_ratio, (cfg.person_scale_min, cfg.person_scale_max), outline_px=cfg.outline_px, edge_mode=cfg.edge_mode, edge_strength=cfg.edge_strength, person_pop=cfg.person_pop, soften=cfg.background_soften)
    assert close(plain, placed)
    assert not np.array_equal(compose_normalized(frame, mask, background, (360, 270), cfg, vintage=True), placed)


def test_config_default_applies_when_no_override_is_given():
    frame, mask, background = _scene()
    cfg = PhotoBoothConfig().compose
    cfg.vintage = False
    placed = place_person(frame, mask, background, (360, 270), cfg.person_height_ratio, (cfg.person_scale_min, cfg.person_scale_max), outline_px=cfg.outline_px, edge_mode=cfg.edge_mode, edge_strength=cfg.edge_strength, person_pop=cfg.person_pop, soften=cfg.background_soften)
    assert close(compose_normalized(frame, mask, background, (360, 270), cfg), placed)
    assert not np.array_equal(compose_normalized(frame, mask, background, (360, 270), cfg, vintage=True), placed)


def test_vintage_query_flag_reaches_the_pipeline(tmp_path):
    bg = tmp_path / "bg.png"
    cv2.imwrite(str(bg), np.full((100, 100, 3), 120, dtype=np.uint8))
    cfg = PhotoBoothConfig()
    cfg.compose.backgrounds = [str(bg)]
    cfg.compose.grain_amount = 0.3
    cfg.compose.randomize_backgrounds = cfg.compose.vary_backgrounds = False
    cfg.compose.morph_probability = 0.0
    cfg.printer.enabled = False
    service = PhotoBoothService(
        cfg, output_dir=str(tmp_path / "out"), camera_factory=FakeCamera,
        segmenter_factory=FakeSegmenter, printer_factory=FakePrinter,
    )
    app = Flask(__name__)
    app.register_blueprint(create_blueprint(service))
    client = app.test_client()

    def shot_after(url):
        body = client.post(url).get_json()
        assert body["status"] == "success"
        return cv2.imread(service.last.image_path)

    plain_a = shot_after("/api/photobooth/strip/oneshot?shots=1&vintage=0")
    plain_b = shot_after("/api/photobooth/strip/oneshot?shots=1&vintage=0")
    grainy_a = shot_after("/api/photobooth/strip/oneshot?shots=1&vintage=1")
    grainy_b = shot_after("/api/photobooth/strip/oneshot?shots=1&vintage=1")
    assert close(plain_a, plain_b)  # no overlay: deterministic up to numeric noise
    assert not np.array_equal(grainy_a, grainy_b)  # overlay: random grain differs run to run
    assert not np.array_equal(plain_a, grainy_a)
    assert b"vintage" in client.get("/photobooth/view").data

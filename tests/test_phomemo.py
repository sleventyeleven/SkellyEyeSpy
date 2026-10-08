import os

import pytest
from PIL import Image

from photobooth.printer.phomemo import PhomemoM02Printer
from photobooth.printer.raster import PRINT_WIDTH_PX


def test_requires_address_or_dry_run_path():
    with pytest.raises(ValueError):
        PhomemoM02Printer()


def test_dry_run_writes_expected_escpos_bytes(tmp_path):
    out_path = str(tmp_path / "print.bin")
    printer = PhomemoM02Printer(dry_run_path=out_path)
    printer.connect()  # no-op in dry-run mode, must not raise

    image = Image.new("L", (PRINT_WIDTH_PX, 10), color=0)
    printer.print_image(image)
    printer.close()

    assert os.path.exists(out_path)
    with open(out_path, "rb") as f:
        payload = f.read()
    assert payload.startswith(b"\x1b@")


def test_print_without_connect_raises_for_real_backend():
    printer = PhomemoM02Printer(bluetooth_address="00:11:22:33:44:55")
    image = Image.new("L", (PRINT_WIDTH_PX, 10), color=0)
    with pytest.raises(RuntimeError):
        printer.print_image(image)


def test_context_manager_dry_run(tmp_path):
    out_path = str(tmp_path / "print.bin")
    image = Image.new("L", (PRINT_WIDTH_PX, 10), color=0)
    with PhomemoM02Printer(dry_run_path=out_path) as printer:
        printer.print_image(image)
    assert os.path.exists(out_path)

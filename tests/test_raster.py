from PIL import Image

from photobooth.printer.raster import BYTES_PER_LINE, PRINT_WIDTH_PX, image_to_escpos


def _find_raster_blocks(payload: bytes):
    """Parses GS v 0 blocks back out of an encoded payload for assertions."""
    blocks = []
    i = 0
    while i < len(payload):
        if payload[i : i + 3] == b"\x1dv0":
            mode = payload[i + 3]
            bytes_per_line = int.from_bytes(payload[i + 4 : i + 6], "little")
            num_lines = int.from_bytes(payload[i + 6 : i + 8], "little")
            data_start = i + 8
            data_end = data_start + bytes_per_line * num_lines
            blocks.append((mode, bytes_per_line, num_lines, payload[data_start:data_end]))
            i = data_end
        else:
            i += 1
    return blocks


def test_starts_with_init_and_justify():
    image = Image.new("L", (PRINT_WIDTH_PX, 10), color=255)
    payload = image_to_escpos(image)
    assert payload.startswith(b"\x1b@\x1ba\x01")


def test_ends_with_feed():
    image = Image.new("L", (PRINT_WIDTH_PX, 10), color=255)
    payload = image_to_escpos(image, feed_lines=5)
    assert payload.endswith(b"\x1bd\x05")


def test_all_white_image_encodes_to_zero_bits():
    image = Image.new("L", (PRINT_WIDTH_PX, 4), color=255)
    payload = image_to_escpos(image)
    blocks = _find_raster_blocks(payload)
    assert len(blocks) == 1
    _, bytes_per_line, num_lines, data = blocks[0]
    assert bytes_per_line == BYTES_PER_LINE
    assert num_lines == 4
    assert data == b"\x00" * (BYTES_PER_LINE * 4)


def test_all_black_image_encodes_to_set_bits():
    image = Image.new("L", (PRINT_WIDTH_PX, 4), color=0)
    payload = image_to_escpos(image)
    blocks = _find_raster_blocks(payload)
    _, _, _, data = blocks[0]
    assert data == b"\xff" * (BYTES_PER_LINE * 4)


def test_wide_image_downscaled_to_print_width():
    image = Image.new("L", (PRINT_WIDTH_PX * 2, 100), color=255)
    payload = image_to_escpos(image)
    blocks = _find_raster_blocks(payload)
    assert blocks[0][1] == BYTES_PER_LINE  # bytes/line always matches fixed print width


def test_tall_image_splits_across_multiple_blocks():
    image = Image.new("L", (PRINT_WIDTH_PX, 600), color=0)
    payload = image_to_escpos(image)
    blocks = _find_raster_blocks(payload)
    assert len(blocks) == 3  # 255 + 255 + 90
    assert [b[2] for b in blocks] == [255, 255, 90]
    assert sum(b[2] for b in blocks) == 600

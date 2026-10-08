from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

ESC = b"\x1b"
GS = b"\x1d"

# Fixed by the M02's 203 DPI head / ~48mm print width -- not configurable.
PRINT_WIDTH_PX = 384
BYTES_PER_LINE = PRINT_WIDTH_PX // 8
_MAX_LINES_PER_BLOCK = 255


@dataclass(frozen=True)
class RasterBlock:
    rows: int
    data: bytes  # includes the GS v 0 header


@dataclass(frozen=True)
class EscposJob:
    """An encoded print job kept in pieces so a sender can pace the blocks
    instead of pushing the whole image at a printer with a small buffer.
    """

    header: bytes
    blocks: list[RasterBlock]
    footer: bytes

    @property
    def total_rows(self) -> int:
        return sum(block.rows for block in self.blocks)

    def to_bytes(self) -> bytes:
        return self.header + b"".join(block.data for block in self.blocks) + self.footer


def _dot_matrix(image: Image.Image) -> np.ndarray:
    """Resizes to PRINT_WIDTH_PX wide (preserving aspect ratio) and
    dithers to 1-bit. Returns a bool array, True where a dot should be
    printed (i.e. a dark/ink pixel) -- explicit about polarity rather
    than trusting PIL's own packed-bit convention, since ESC/POS's "1 =
    print" is the opposite of PIL's "1 = white" internal packing.
    """
    grayscale = image.convert("L")
    width, height = grayscale.size
    if width != PRINT_WIDTH_PX:  # no resampling when already at print width
        new_height = max(1, round(height * (PRINT_WIDTH_PX / width)))
        grayscale = grayscale.resize((PRINT_WIDTH_PX, new_height))

    dithered = grayscale.convert("1")  # Floyd-Steinberg by default
    pixels = np.array(dithered)  # 0 = black, 255 = white
    return pixels == 0


def image_to_escpos_job(image: Image.Image, justification: int = 1, feed_lines: int = 3) -> EscposJob:
    """Encodes a PIL image for the Phomemo M02: init + justify (header), one
    GS v 0 raster block per <=255 lines, then a feed (footer). See
    docs/hardware-research.md for where this protocol came from
    (vivier/phomemo-tools, reverse-engineered from the Android app).
    """
    dots = _dot_matrix(image)
    packed = np.packbits(dots, axis=1)  # MSB-first, matches ESC/POS's own bit order
    assert packed.shape[1] == BYTES_PER_LINE

    header = ESC + b"@" + ESC + b"a" + justification.to_bytes(1, "little")

    blocks: list[RasterBlock] = []
    total_lines = packed.shape[0]
    line = 0
    while line < total_lines:
        block_lines = min(_MAX_LINES_PER_BLOCK, total_lines - line)
        data = (
            GS
            + b"v0"
            + (0).to_bytes(1, "little")  # mode: normal
            + BYTES_PER_LINE.to_bytes(2, "little")
            + block_lines.to_bytes(2, "little")
            + packed[line : line + block_lines].tobytes()
        )
        blocks.append(RasterBlock(block_lines, data))
        line += block_lines

    footer = ESC + b"d" + feed_lines.to_bytes(1, "little")
    return EscposJob(header, blocks, footer)


def image_to_escpos(image: Image.Image, justification: int = 1, feed_lines: int = 3) -> bytes:
    return image_to_escpos_job(image, justification, feed_lines).to_bytes()


def preview_image(image: Image.Image) -> Image.Image:
    """The exact 1-bit, 384 px wide result the printer will produce (black
    ink on white), so the look can be tuned without spending paper.
    """
    dots = _dot_matrix(image)
    return Image.fromarray(np.where(dots, 0, 255).astype(np.uint8), mode="L")

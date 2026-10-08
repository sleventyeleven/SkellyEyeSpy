from __future__ import annotations

import socket
import time
from typing import Callable

from PIL import Image

from .raster import EscposJob, image_to_escpos_job

_RFCOMM_CHANNEL = 1
_CHUNK_BYTES = 2048
_CHUNK_DELAY_S = 0.005
_POLL_S = 0.1


class PhomemoM02Printer:
    """Talks to a Phomemo M02 over classic Bluetooth RFCOMM (not BLE --
    see docs/hardware-research.md for why). `dry_run_path` writes the raw
    ESC/POS bytes to a file instead of opening a socket.

    The M02 has a small receive buffer and no flow control: a whole photo
    strip sent at Bluetooth speed (~60 KB in well under a second) overflows
    it, and the printer silently drops the tail (observed: a strip stopped
    at the start of its third frame). So the job is sent block by block,
    holding back whenever more than `max_backlog_lines` rows would be
    sitting unprinted, judged against an estimated print position
    (`lines_per_second`; the M02 is rated 10-15 mm/s = ~80-120 lines/s at
    203 dpi). The connection is then held open until the estimated end of
    the print, because closing early can cut the print off.
    """

    def __init__(
        self,
        bluetooth_address: str | None = None,
        dry_run_path: str | None = None,
        lines_per_second: float = 100.0,
        max_backlog_lines: int = 400,
        tail_pad_s: float = 1.5,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if not bluetooth_address and not dry_run_path:
            raise ValueError("Need either bluetooth_address or dry_run_path")
        self._address = bluetooth_address
        self._dry_run_path = dry_run_path
        self._lines_per_second = lines_per_second
        self._max_backlog_lines = max_backlog_lines
        self._tail_pad_s = tail_pad_s
        self._clock = clock
        self._sleep = sleep
        self._sock: socket.socket | None = None
        self.responses: list[bytes] = []  # anything the printer sent back (diagnostics)

    def connect(self) -> None:
        if self._dry_run_path:
            return
        self._sock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
        self._sock.connect((self._address, _RFCOMM_CHANNEL))

    def print_image(self, image: Image.Image) -> None:
        job = image_to_escpos_job(image)
        if self._dry_run_path:
            with open(self._dry_run_path, "wb") as f:
                f.write(job.to_bytes())
            return
        if self._sock is None:
            raise RuntimeError("Not connected -- call connect() first")
        self._send_paced(job)

    def _send_paced(self, job: EscposJob) -> None:
        self._send(job.header)
        started = self._clock()
        sent_rows = 0
        for block in job.blocks:
            # allow at most max_backlog_lines unprinted rows once this block is queued
            rows_printed_needed = sent_rows + block.rows - self._max_backlog_lines
            self._wait_until(started + rows_printed_needed / self._lines_per_second)
            self._send(block.data)
            sent_rows += block.rows
        self._send(job.footer)
        self._wait_until(started + sent_rows / self._lines_per_second + self._tail_pad_s)

    def _send(self, data: bytes) -> None:
        for offset in range(0, len(data), _CHUNK_BYTES):
            self._sock.sendall(data[offset : offset + _CHUNK_BYTES])
            if len(data) > _CHUNK_BYTES:
                self._sleep(_CHUNK_DELAY_S)

    def _wait_until(self, deadline: float) -> None:
        """Waits for `deadline`, recording any bytes the printer sends back
        while we wait (so its own status messages can be studied later).
        """
        while True:
            remaining = deadline - self._clock()
            if remaining <= 0:
                return
            step = min(_POLL_S, remaining)
            if hasattr(self._sock, "settimeout") and hasattr(self._sock, "recv"):
                try:
                    self._sock.settimeout(step)
                    data = self._sock.recv(64)
                    if data:
                        self.responses.append(data)
                    elif not data:  # printer closed the connection
                        self._sleep(step)
                except (socket.timeout, TimeoutError):
                    pass
                except OSError:
                    self._sleep(step)
                finally:
                    self._sock.settimeout(None)
            else:
                self._sleep(step)

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def __enter__(self) -> "PhomemoM02Printer":
        self.connect()
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

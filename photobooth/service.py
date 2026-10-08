from __future__ import annotations

import glob
import os
import secrets
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Callable

import cv2
import numpy as np
from PIL import Image

from .capture import Camera
from .compose.backgrounds import anchor_for, make_background
from .compose.strip import render_strip
from .compose.framing import EDGE_MODES, count_people, place_person
from .config import ComposeConfig, PhotoBoothConfig
from .pipeline import compose_normalized, compose_photo, load_background
from .printer import PhomemoM02Printer, preview_image
from .segmentation import SelfieSegmenter

_WARMUP_FRAMES = 15  # ~0.5s at 30fps: lets auto-exposure settle


# query-string name -> ComposeConfig field, for the studio's per-render overrides
_DRAFT_PARAMS = {
    "size": "person_height_ratio",
    "outline": "outline_px",
    "edge": "edge_mode",
    "strength": "edge_strength",
    "pop": "person_pop",
    "soften": "background_soften",
    "morph": "morph_probability",
    "anchor": "anchor_mode",
    "offset": "anchor_offset",
}
_ANCHOR_MODES = ("center", "left", "right", "alternate", "random")


@dataclass
class Draft:
    """One capture that can be re-rendered as often as wanted (different
    seed, size, placement...) before anything is sent to the printer.
    """

    frame: np.ndarray
    mask: np.ndarray
    dir: str
    people: int = 0
    ignored: int = 0
    seed: int | None = None
    shots: int = 0
    overrides: dict[int, int] = field(default_factory=dict)  # frame index -> reroll seed
    params: dict = field(default_factory=dict)  # last slider values, kept between renders
    strip: Image.Image | None = None
    labels: list[str] = field(default_factory=list)
    # backgrounds already built for this capture, so moving a size/outline/soften/placement
    # slider does not rebuild them (they only depend on the seed and the variation settings)
    bg_cache: dict = field(default_factory=dict)


@dataclass
class SnapResult:
    image_path: str
    preview_path: str
    debug_path: str
    printed: bool
    error: str | None
    seconds: float
    seed: int | None = None  # reproduces the backgrounds: pass ?seed=N
    backgrounds: list[str] = field(default_factory=list)


class PhotoBoothService:
    """One snapshot: capture -> render -> save (+ 1-bit preview) -> print.

    Serialized by a lock because the camera has a single owner. Printing
    failures are reported in the result, never raised: a dead or absent
    printer must not wedge the show.
    """

    def __init__(
        self,
        config: PhotoBoothConfig,
        output_dir: str = "scratch/booth",
        camera_factory: Callable = Camera,
        segmenter_factory: Callable = SelfieSegmenter,
        printer_factory: Callable = PhomemoM02Printer,
    ) -> None:
        self._config = config
        self._output_dir = os.path.abspath(output_dir)
        self._camera_factory = camera_factory
        self._segmenter_factory = segmenter_factory
        self._printer_factory = printer_factory
        self._segmenter = None
        self._bg_cache: dict[str, np.ndarray] = {}
        self._pool_cache: list[str] | None = None
        self._preview_seed: int | None = None
        self._strip_seed: int | None = None
        self._strip_labels: list[str] = []
        self._strip_shots: list[np.ndarray] = []
        self._strip_target = 0
        self._strip_dir: str | None = None
        self._draft: Draft | None = None
        self._lock = threading.Lock()
        self.last: SnapResult | None = None

    def snap(self, print_it: bool = True, vintage: bool | None = None, seed: int | None = None) -> SnapResult:
        with self._lock:
            started = time.monotonic()
            frame = self._capture()
            if self._segmenter is None:
                self._segmenter = self._segmenter_factory()

            mask = self._segmenter.person_mask(frame)
            used_seed = self._take_seed(seed)
            background, label = self._make_bg(used_seed, 0)
            result_bgr = self._compose(
                frame, mask, background, (frame.shape[1], frame.shape[0]), vintage, anchor_x=self._anchor(used_seed, 0)
            )

            os.makedirs(self._output_dir, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_path = os.path.join(self._output_dir, f"booth_{stamp}.png")
            preview_path = os.path.join(self._output_dir, f"booth_{stamp}_print.png")
            debug_path = os.path.join(self._output_dir, f"booth_{stamp}_debug.jpg")
            cv2.imwrite(image_path, result_bgr)
            # raw | person mask | result: shows at a glance whether a bad photo is
            # a framing problem or a segmentation problem
            cv2.imwrite(debug_path, np.hstack([frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), result_bgr]))
            rgb = Image.fromarray(result_bgr[:, :, ::-1])
            preview_image(rgb).save(preview_path)

            printed, error = (False, None)
            if print_it and self._config.printer.enabled:
                printed, error = self._print(rgb)

            self.last = SnapResult(
                image_path, preview_path, debug_path, printed, error, time.monotonic() - started, used_seed, [label]
            )
            return self.last

    def _capture(self, warmup: int = _WARMUP_FRAMES) -> np.ndarray:
        with self._camera_factory(self._config.capture) as camera:
            frame = None
            for _ in range(warmup):  # let auto-exposure settle
                frame = camera.read()
        if frame is None:
            raise RuntimeError("Failed to read a frame from the camera")
        width = self._config.compose.working_width
        if frame.shape[1] > width:
            height = round(frame.shape[0] * width / frame.shape[1])
            frame = cv2.resize(frame, (width, height))
        return frame

    def _compose(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        background: np.ndarray,
        size: tuple[int, int],
        vintage: bool | None = None,
        cfg: ComposeConfig | None = None,
        anchor_x: float = 0.5,
    ) -> np.ndarray:
        compose_cfg = cfg or self._config.compose
        if compose_cfg.normalize_person:
            return compose_normalized(frame, mask, background, size, compose_cfg, vintage, anchor_x)
        return compose_photo(frame, mask, background, compose_cfg, vintage)

    def _anchor(self, seed: int, index: int, cfg: ComposeConfig | None = None) -> float:
        cfg = cfg or self._config.compose
        return anchor_for(seed, index, cfg.anchor_mode, cfg.anchor_offset)

    def aim_jpeg(self, composite: bool = False) -> bytes:
        """One quick frame with the detected person tinted green, for aiming
        the camera. Shares the lock with snap(): one owner of the camera.
        """
        with self._lock:
            frame = self._capture(warmup=3)
            if self._segmenter is None:
                self._segmenter = self._segmenter_factory()
            mask = self._segmenter.person_mask(frame) > 0
            tinted = frame.copy()
            tinted[mask] = (0.55 * tinted[mask] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
            coverage = 100.0 * mask.mean()
            kept, ignored = count_people(mask, self._config.compose.group_min_area_ratio)
            label = f"people {kept}" + (f" (+{ignored} ignored)" if ignored else "") + f"  {coverage:.0f}%"
            cv2.putText(tinted, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            if composite:
                # what the next strip shot will look like (no grain), beside the live view
                cfg = self._config.strip
                in_progress = self._strip_target > 0 and self._strip_seed is not None
                preview_seed = self._strip_seed if in_progress else self._peek_seed()
                index = len(self._strip_shots) if in_progress else 0
                background, _ = self._make_bg(preview_seed, index)
                cc = self._config.compose
                placed = place_person(
                    frame, mask, background, (cfg.frame_width, cfg.frame_height),
                    height_ratio=cc.person_height_ratio,
                    scale_limits=(cc.person_scale_min, cc.person_scale_max),
                    outline_px=cc.outline_px,
                    edge_mode=cc.edge_mode,
                    edge_strength=cc.edge_strength,
                    person_pop=cc.person_pop,
                    soften=cc.background_soften,
                    anchor_x=self._anchor(preview_seed, index),
                    group_min_area_ratio=cc.group_min_area_ratio,
                )
                scale = tinted.shape[0] / placed.shape[0]
                placed = cv2.resize(placed, (round(placed.shape[1] * scale), tinted.shape[0]))
                tinted = np.hstack([tinted, placed])
            ok, buf = cv2.imencode(".jpg", tinted)
            if not ok:
                raise RuntimeError("JPEG encode failed")
            return buf.tobytes()

    def _print(self, image: Image.Image) -> tuple[bool, str | None]:
        try:
            printer_cfg = self._config.printer
            printer = self._printer_factory(
                bluetooth_address=printer_cfg.bluetooth_address,
                dry_run_path=printer_cfg.dry_run_path,
                lines_per_second=printer_cfg.lines_per_second,
                max_backlog_lines=printer_cfg.max_backlog_lines,
                tail_pad_s=printer_cfg.tail_pad_s,
            )
            with printer:
                printer.print_image(image)
            return True, None
        except Exception as exc:  # noqa: BLE001 - show must go on
            return False, f"{type(exc).__name__}: {exc}"

    # ---- photo strip: N shots on different backgrounds, printed as one reel ----

    # ---- backgrounds: a seed fully determines a reel's backgrounds ----

    @property
    def strip_seed(self) -> int | None:
        return self._strip_seed

    def _pool(self) -> list[str]:
        if self._pool_cache is None:
            entries = self._config.compose.backgrounds or [self._config.compose.background]
            pool: list[str] = []
            for entry in entries:
                matches = sorted(glob.glob(entry))
                pool.extend(matches if matches else [entry])
            self._pool_cache = list(dict.fromkeys(pool))
        return self._pool_cache

    def _load_cached(self, path: str) -> np.ndarray:
        if path not in self._bg_cache:
            image = load_background(path)
            limit = self._config.compose.background_max_width
            if limit and image.shape[1] > limit:  # final frames are 360 px; keep ~2x for zoom headroom
                image = cv2.resize(image, (limit, round(image.shape[0] * limit / image.shape[1])), interpolation=cv2.INTER_AREA)
            self._bg_cache[path] = image
        return self._bg_cache[path]

    def _make_bg(self, seed: int, index: int) -> tuple[np.ndarray, str]:
        return make_background(seed, index, self._pool(), self._load_cached, self._config.compose)

    @staticmethod
    def _new_seed() -> int:
        return secrets.randbelow(2**31)

    def _peek_seed(self) -> int:
        """The seed the next reel will use unless one is given, so the aim
        preview shows the real first background of the reel to come."""
        if self._preview_seed is None:
            self._preview_seed = self._new_seed()
        return self._preview_seed

    def _take_seed(self, explicit: int | None) -> int:
        if explicit is not None:
            return explicit
        seed = self._peek_seed()
        self._preview_seed = None
        return seed

    def _new_strip_dir(self, seed: int) -> str:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self._output_dir, f"strip_{stamp}_s{seed}")
        os.makedirs(path, exist_ok=True)
        return path

    def strip_start(self, shots: int | None = None, seed: int | None = None) -> int:
        with self._lock:
            self._strip_target = max(1, shots or self._config.strip.shots)
            self._strip_shots = []
            self._strip_seed = self._take_seed(seed)
            self._strip_labels = []
            self._strip_dir = self._new_strip_dir(self._strip_seed)
            return self._strip_target

    def strip_shot(self, vintage: bool | None = None) -> dict:
        with self._lock:
            if not self._strip_target or self._strip_dir is None:
                raise RuntimeError("No strip in progress -- call strip_start first")
            index = len(self._strip_shots)
            if index >= self._strip_target:
                raise RuntimeError("Strip already has all its shots")

            frame = self._capture(warmup=8)
            if self._segmenter is None:
                self._segmenter = self._segmenter_factory()
            mask = self._segmenter.person_mask(frame)

            cfg = self._config.strip
            size = (cfg.frame_width, cfg.frame_height)
            background, label = self._make_bg(self._strip_seed, index)
            self._strip_labels.append(label)
            # composed directly at the final print size, so grain/vintage act on real print pixels
            shot = self._compose(frame, mask, background, size, vintage, anchor_x=self._anchor(self._strip_seed, index))

            self._strip_shots.append(shot)
            cv2.imwrite(os.path.join(self._strip_dir, f"shot_{index}.png"), shot)
            return {
                "index": index,
                "total": self._strip_target,
                "background": label,
                "seed": self._strip_seed,
                "person_percent": round(float((mask > 0).mean() * 100), 1),
            }

    def strip_shot_path(self, index: int) -> str | None:
        if self._strip_dir is None:
            return None
        path = os.path.join(self._strip_dir, f"shot_{index}.png")
        return path if os.path.exists(path) else None

    def strip_finish(self, print_it: bool = True) -> SnapResult:
        with self._lock:
            return self._finish_strip_locked(print_it, time.monotonic())

    def strip_one_shot(
        self, shots: int | None = None, print_it: bool = True, vintage: bool | None = None, seed: int | None = None
    ) -> SnapResult:
        """One capture, the whole reel: the same photo placed on each of the
        next `shots` backgrounds. Simpler for a visitor than posing N times.
        """
        with self._lock:
            started = time.monotonic()
            cfg = self._config.strip
            count = max(1, shots or cfg.shots)

            frame = self._capture()
            if self._segmenter is None:
                self._segmenter = self._segmenter_factory()
            mask = self._segmenter.person_mask(frame)

            self._strip_seed = self._take_seed(seed)
            self._strip_dir = self._new_strip_dir(self._strip_seed)
            size = (cfg.frame_width, cfg.frame_height)
            backgrounds = [self._make_bg(self._strip_seed, i) for i in range(count)]
            self._strip_labels = [label for _image, label in backgrounds]
            # each frame gets its own composite (and its own grain), only the capture is shared
            self._strip_shots = [
                self._compose(frame, mask, image, size, vintage, anchor_x=self._anchor(self._strip_seed, i))
                for i, (image, _label) in enumerate(backgrounds)
            ]
            for i, shot in enumerate(self._strip_shots):
                cv2.imwrite(os.path.join(self._strip_dir, f"shot_{i}.png"), shot)
            return self._finish_strip_locked(print_it, started)

    def _finish_strip_locked(self, print_it: bool, started: float) -> SnapResult:
        if not self._strip_shots or self._strip_dir is None:
            raise RuntimeError("No shots taken")
        cfg = self._config.strip
        footer = cfg.footer or datetime.now().strftime("%b %d, %Y")
        strip = render_strip(self._strip_shots, cfg, footer)

        image_path = os.path.join(self._strip_dir, "strip.png")
        preview_path = os.path.join(self._strip_dir, "strip_print.png")
        strip.save(image_path)
        preview_image(strip).save(preview_path)

        printed, error = (False, None)
        if print_it and self._config.printer.enabled:
            printed, error = self._print(strip)

        self.last = SnapResult(
            image_path, preview_path, image_path, printed, error, time.monotonic() - started,
            self._strip_seed, list(self._strip_labels),
        )
        self._strip_shots = []
        self._strip_target = 0
        return self.last

    # ---- studio: preview and iterate before anything is printed ----

    def draft_capture(self) -> dict:
        """Takes the photo once (and segments it). Rendering is separate so
        backgrounds, size and placement can be tuned on this same capture.
        """
        with self._lock:
            frame = self._capture()
            if self._segmenter is None:
                self._segmenter = self._segmenter_factory()
            mask = self._segmenter.person_mask(frame)

            folder = os.path.join(self._output_dir, "draft_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
            os.makedirs(folder, exist_ok=True)
            cv2.imwrite(os.path.join(folder, "capture.jpg"), frame)
            cv2.imwrite(os.path.join(folder, "mask.png"), mask)
            people, ignored = count_people(mask, self._config.compose.group_min_area_ratio)
            self._draft = Draft(frame, mask, folder, people, ignored)
            return {"people": people, "ignored": ignored, "person_percent": round(float((mask > 0).mean() * 100), 1)}

    def draft_render(
        self,
        shots: int | None = None,
        seed: int | None = None,
        vintage: bool | None = None,
        reroll: int | None = None,
        reroll_all: bool = False,
        **params,
    ) -> dict:
        """Renders the reel for the current capture. Slider values stick
        between calls, so changing one setting re-renders with the others as
        they were; `seed` pins a reel, `reroll_all` draws a fresh one and
        `reroll=k` replaces only frame k's background. Never prints.
        """
        with self._lock:
            draft = self._draft
            if draft is None:
                raise RuntimeError("No capture yet -- take a photo first")
            started = time.monotonic()

            for key, value in params.items():
                if value is None:
                    continue
                if key not in _DRAFT_PARAMS:
                    raise ValueError(f"Unknown setting {key!r}")
                if key == "anchor" and value not in _ANCHOR_MODES:
                    raise ValueError(f"anchor must be one of {', '.join(_ANCHOR_MODES)}")
                if key == "edge" and value not in EDGE_MODES:
                    raise ValueError(f"edge must be one of {', '.join(EDGE_MODES)}")
                draft.params[key] = value
            compose_cfg = replace(self._config.compose, **{_DRAFT_PARAMS[k]: v for k, v in draft.params.items()})

            count = max(1, shots or draft.shots or self._config.strip.shots)
            draft.shots = count
            if seed is not None:
                draft.seed, draft.overrides = seed, {}
            elif reroll_all or draft.seed is None:
                draft.seed, draft.overrides = self._take_seed(None), {}
            if reroll is not None and 0 <= reroll < count:
                draft.overrides[reroll] = self._new_seed()
            draft.overrides = {i: s for i, s in draft.overrides.items() if i < count}

            cfg = self._config.strip
            size = (cfg.frame_width, cfg.frame_height)
            shots_bgr, labels = [], []
            for i in range(count):
                bg_seed, bg_index = (draft.overrides[i], 0) if i in draft.overrides else (draft.seed, i)
                key = (bg_seed, bg_index, compose_cfg.randomize_backgrounds, compose_cfg.vary_backgrounds,
                       compose_cfg.morph_probability, compose_cfg.zoom_max)
                if key not in draft.bg_cache:
                    if len(draft.bg_cache) > 48:
                        draft.bg_cache.clear()
                    draft.bg_cache[key] = make_background(bg_seed, bg_index, self._pool(), self._load_cached, compose_cfg)
                background, label = draft.bg_cache[key]
                anchor = anchor_for(bg_seed, bg_index, compose_cfg.anchor_mode, compose_cfg.anchor_offset)
                shots_bgr.append(self._compose(draft.frame, draft.mask, background, size, vintage, compose_cfg, anchor))
                labels.append(label)

            strip = render_strip(shots_bgr, cfg, cfg.footer or datetime.now().strftime("%b %d, %Y"))
            image_path = os.path.join(draft.dir, "strip.png")
            preview_path = os.path.join(draft.dir, "strip_print.png")
            strip.save(image_path)
            preview_image(strip).save(preview_path)
            for i, shot in enumerate(shots_bgr):
                cv2.imwrite(os.path.join(draft.dir, f"shot_{i}.png"), shot)
            for stale in range(count, 12):
                leftover = os.path.join(draft.dir, f"shot_{stale}.png")
                if os.path.exists(leftover):
                    os.remove(leftover)

            draft.strip, draft.labels = strip, labels
            seconds = time.monotonic() - started
            self.last = SnapResult(image_path, preview_path, image_path, False, None, seconds, draft.seed, labels)
            return {
                "seed": draft.seed,
                "overrides": {str(i): s for i, s in draft.overrides.items()},
                "backgrounds": labels,
                "shots": count,
                "params": dict(draft.params),
                "people": draft.people,
                "ignored": draft.ignored,
                "seconds": round(seconds, 2),
            }

    def draft_print(self) -> SnapResult:
        """Prints exactly the strip last rendered, so what was approved on
        screen is what comes out. Does not touch the camera.
        """
        with self._lock:
            draft = self._draft
            if draft is None or draft.strip is None or self.last is None:
                raise RuntimeError("Nothing to print yet -- take a photo and render a reel first")
            started = time.monotonic()
            if self._config.printer.enabled:
                printed, error = self._print(draft.strip)
            else:
                printed, error = False, "printing is disabled in the config"
            self.last = replace(self.last, printed=printed, error=error, seconds=time.monotonic() - started)
            return self.last

    def draft_path(self, name: str) -> str | None:
        """Files of the current draft: strip, print (1-bit preview), capture,
        mask, or shot_N."""
        if self._draft is None:
            return None
        names = {"strip": "strip.png", "print": "strip_print.png", "capture": "capture.jpg", "mask": "mask.png"}
        filename = names.get(name, f"{name}.png" if name.startswith("shot_") else None)
        if filename is None:
            return None
        path = os.path.join(self._draft.dir, filename)
        return path if os.path.exists(path) else None

    def close(self) -> None:
        if self._segmenter is not None:
            self._segmenter.close()
            self._segmenter = None

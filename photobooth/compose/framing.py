from __future__ import annotations

import cv2
import numpy as np

_FEATHER_PX = 5
_WIDTH_FILL = 0.92  # a wide subject (group) may fill at most this much of the frame width
_PAD_FRAC = 0.04
_TARGET_LUMINANCE = 118.0  # mean level a dark scene is lifted toward before dithering
_MIN_GAMMA = 0.45
EDGE_MODES = ("none", "separate", "stipple", "ring")
_BAYER = (np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float32) + 0.5) / 16.0
_LUMA = np.array([0.114, 0.587, 0.299], np.float32)  # BGR weights
_DODGE_BIAS = 12.0  # the scene must be this much lighter than the person before it is lightened


def clean_mask(mask: np.ndarray, min_area_frac: float = 0.004) -> np.ndarray:
    """Drops stray specks (clutter the model briefly mistakes for a person)
    and closes small gaps. Keeps every sizeable blob so groups survive.
    """
    closed = cv2.morphologyEx((mask > 0).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    min_area = min_area_frac * mask.shape[0] * mask.shape[1]
    keep = np.zeros(mask.shape, dtype=np.uint8)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == label] = 255
    return keep


def _blobs(mask: np.ndarray):
    count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    return count, labels, stats


def drop_small_people(mask: np.ndarray, min_ratio: float) -> np.ndarray:
    """Keeps every blob at least `min_ratio` of the largest blob's area. A
    child standing beside a parent stays; someone small and far away in the
    background (who would otherwise shrink the whole group to fit them in)
    is dropped.
    """
    if min_ratio <= 0:
        return mask
    count, labels, stats = _blobs(mask)
    if count <= 2:  # background plus at most one blob
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep_labels = [i + 1 for i, area in enumerate(areas) if area >= min_ratio * areas.max()]
    return np.where(np.isin(labels, keep_labels), 255, 0).astype(np.uint8)


def count_people(mask: np.ndarray, min_ratio: float = 0.0) -> tuple[int, int]:
    """(people kept, people ignored) after cleanup. Touching people merge
    into one blob, so this undercounts a tight huddle; it is a guide, not a
    head count.
    """
    cleaned = clean_mask(mask)
    kept = drop_small_people(cleaned, min_ratio)
    total = _blobs(cleaned)[0] - 1
    return _blobs(kept)[0] - 1, total - (_blobs(kept)[0] - 1)


def person_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """(x0, y0, x1, y1), x1/y1 exclusive, or None when nobody is present."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def cover_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize to fill width x height, cropping the overflow (no stretching)."""
    src_h, src_w = image.shape[:2]
    scale = max(width / src_w, height / src_h)
    resized = cv2.resize(image, (max(width, round(src_w * scale)), max(height, round(src_h * scale))), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    y0 = (resized.shape[0] - height) // 2
    x0 = (resized.shape[1] - width) // 2
    return resized[y0 : y0 + height, x0 : x0 + width]


def soften_background(background_bgr: np.ndarray, amount: float) -> np.ndarray:
    """Calms a busy scene so the person reads against it, which matters a
    lot after the 1-bit dither: blends toward a blurred copy, pulls contrast
    toward the mean, and brightens scenes whose average is dark (they
    otherwise print as near-solid black) with a gamma lift scaled by
    `amount`. 0 leaves the image untouched.
    """
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount == 0.0:
        return background_bgr
    blurred = cv2.GaussianBlur(background_bgr, (0, 0), 0.6 + 2.4 * amount)
    mix = 0.8 * amount
    softened = background_bgr.astype(np.float32) * (1 - mix) + blurred.astype(np.float32) * mix
    mean = float(softened.mean())
    softened = np.clip(np.rint(mean + (softened - mean) * (1 - 0.35 * amount)), 0, 255).astype(np.uint8)

    luminance = float(cv2.cvtColor(softened, cv2.COLOR_BGR2GRAY).mean())
    if luminance < _TARGET_LUMINANCE:
        full_gamma = np.log(_TARGET_LUMINANCE / 255.0) / np.log(max(luminance, 1.0) / 255.0)
        gamma = max(_MIN_GAMMA, 1.0 - (1.0 - min(1.0, full_gamma)) * min(1.0, amount * 1.4))
        lut = (255.0 * (np.arange(256) / 255.0) ** gamma).astype(np.uint8)
        softened = cv2.LUT(softened, lut)
    return softened


def pop_person(patch_bgr: np.ndarray, alpha: np.ndarray, amount: float) -> np.ndarray:
    """Separates the person from the scene by tone instead of by a drawn
    border: a little more saturation, and more contrast about their own mean
    brightness. `amount` 0 leaves them untouched.
    """
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount == 0.0:
        return patch_bgr
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    inside = alpha > 0.5
    mean_v = float(hsv[..., 2][inside].mean()) if inside.any() else 128.0
    hsv[..., 1] = np.clip(hsv[..., 1] * (1 + 0.6 * amount), 0, 255)
    hsv[..., 2] = np.clip(mean_v + (hsv[..., 2] - mean_v) * (1 + 0.35 * amount), 0, 255)
    return cv2.cvtColor(np.rint(hsv).astype(np.uint8), cv2.COLOR_HSV2BGR)


def place_person(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    background_bgr: np.ndarray,
    out_size: tuple[int, int],
    height_ratio: float = 0.72,
    scale_limits: tuple[float, float] = (0.1, 1.8),
    outline_px: int = 0,
    outline_color: tuple[int, int, int] = (255, 255, 255),
    soften: float = 0.0,
    anchor_x: float = 0.5,
    group_min_area_ratio: float = 0.0,
    edge_mode: str = "ring",
    edge_strength: float = 0.5,
    person_pop: float = 0.0,
) -> np.ndarray:
    """Cuts the person(s) out and places them on the background at a
    normalized size: their visible height becomes `height_ratio` of the
    output height regardless of how tall they are or how close they stand,
    so the background always has room to show. Centered horizontally and
    anchored to the bottom (to the very bottom edge when the camera cut
    them off there, slightly above it when their feet are in view).

    `outline_px` is the width of an edge band around the person, drawn
    according to `edge_mode`: `ring` (a solid die-cut halo, which prints as
    an angel glow and is kept only for comparison), `stipple` (a fading
    scatter of dots, dense at the person and sparser outward), `separate`
    (no line at all: a soft dodge-and-burn of the background that makes it
    darker where the person is lighter and the reverse), or `none`. The last
    two pick their direction from the person's and the scene's brightness.
    `edge_strength` scales them, `person_pop` adds tone contrast to the
    person, and `soften` calms the background; all exist to keep a person
    readable against a detailed scene once it is dithered to one bit. `anchor_x` is where the person (or group) is centered across
    the frame, 0.5 being the middle; it is clamped so the group stays
    inside the frame. Several people are placed together as one group sized
    by the tallest, and `group_min_area_ratio` drops tiny distant bystanders.
    Returns the background alone when nobody is detected.
    """
    out_w, out_h = out_size
    background = soften_background(cover_resize(background_bgr, out_w, out_h), soften)
    frame_h, frame_w = frame_bgr.shape[:2]

    cleaned = drop_small_people(clean_mask(mask), group_min_area_ratio)
    bbox = person_bbox(cleaned)
    if bbox is None:
        return background
    x0, y0, x1, y1 = bbox
    person_w, person_h = x1 - x0, y1 - y0

    scale = float(np.clip(height_ratio * out_h / person_h, *scale_limits))
    scale = min(scale, _WIDTH_FILL * out_w / person_w)

    pad = round(_PAD_FRAC * max(person_w, person_h))
    cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
    cx1, cy1 = min(frame_w, x1 + pad), min(frame_h, y1 + pad)
    patch = frame_bgr[cy0:cy1, cx0:cx1]
    patch_mask = cleaned[cy0:cy1, cx0:cx1]

    new_size = (max(1, round(patch.shape[1] * scale)), max(1, round(patch.shape[0] * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    patch = cv2.resize(patch, new_size, interpolation=interp)
    patch_mask = cv2.resize(patch_mask, new_size, interpolation=cv2.INTER_LINEAR)
    kernel = _FEATHER_PX | 1
    alpha = cv2.GaussianBlur(patch_mask, (kernel, kernel), 0).astype(np.float32) / 255.0
    patch = pop_person(patch, alpha, person_pop)

    touches_bottom = y1 >= frame_h - 2
    bottom_margin = 0 if touches_bottom else round(0.03 * out_h)
    person_center_x = ((x0 + x1) / 2 - cx0) * scale
    person_bottom_y = (y1 - cy0) * scale
    half_group = person_w * scale / 2
    edge = 0.02 * out_w
    if 2 * half_group + 2 * edge <= out_w:
        target_x = float(np.clip(anchor_x * out_w, half_group + edge, out_w - half_group - edge))
    else:
        target_x = out_w / 2  # wider than the frame allows: just center it
    dest_x = round(target_x - person_center_x)
    dest_y = round(out_h - bottom_margin - person_bottom_y)

    if edge_mode not in EDGE_MODES:
        raise ValueError(f"edge_mode must be one of {', '.join(EDGE_MODES)}")
    k = int(outline_px)
    edge_on = k > 0 and edge_mode != "none"
    halo = falloff = None
    if edge_on:
        margin = k + 2  # room for the band to extend past the cutout's own bounds
        alpha = cv2.copyMakeBorder(alpha, margin, margin, margin, margin, cv2.BORDER_CONSTANT, value=0)
        patch = cv2.copyMakeBorder(patch, margin, margin, margin, margin, cv2.BORDER_REPLICATE)
        outside = (alpha < 0.5).astype(np.uint8)
        distance = cv2.distanceTransform(outside, cv2.DIST_L2, 3)  # px from the person, 0 inside
        falloff = np.clip(1.0 - distance / k, 0.0, 1.0) * outside  # 1 at the edge, 0 at k px away
        if edge_mode == "ring":
            ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
            halo = np.clip(cv2.GaussianBlur(cv2.dilate(alpha, ring), (3, 3), 0), 0.0, 1.0)
        dest_x -= margin
        dest_y -= margin
        new_size = (patch.shape[1], patch.shape[0])

    # clip the paste region to the output frame
    ox0, oy0 = max(0, dest_x), max(0, dest_y)
    ox1, oy1 = min(out_w, dest_x + new_size[0]), min(out_h, dest_y + new_size[1])
    if ox1 <= ox0 or oy1 <= oy0:
        return background
    sx0, sy0 = ox0 - dest_x, oy0 - dest_y
    h, w = oy1 - oy0, ox1 - ox0
    region_alpha = alpha[sy0 : sy0 + h, sx0 : sx0 + w, None]
    region_patch = patch[sy0 : sy0 + h, sx0 : sx0 + w].astype(np.float32)
    region_bg = background[oy0:oy1, ox0:ox1].astype(np.float32)

    if edge_on and edge_mode == "ring":
        region_halo = halo[sy0 : sy0 + h, sx0 : sx0 + w, None]
        region_bg = region_bg * (1 - region_halo) + np.array(outline_color, dtype=np.float32) * region_halo
    elif edge_on:
        region_w = falloff[sy0 : sy0 + h, sx0 : sx0 + w]
        strength = float(np.clip(edge_strength, 0.0, 1.0))
        # Which way to push the scene is decided locally, against the nearest part of the
        # person (a pale face beside pale sand needs the sand darkened even if the shirt is
        # dark). When tones are alike it burns, like a soft contact shadow.
        person_lum_map = region_patch @ _LUMA
        sigma = max(2.0, k * 0.6)
        near = cv2.GaussianBlur(person_lum_map * region_alpha[..., 0], (0, 0), sigma)
        near_weight = cv2.GaussianBlur(region_alpha[..., 0], (0, 0), sigma)
        person_near = near / np.maximum(near_weight, 1e-3)
        diff = region_bg @ _LUMA - person_near - _DODGE_BIAS
        direction = np.tanh(diff / 25.0)  # -1 burn .. +1 dodge, 0 on the fence
        if edge_mode == "separate":
            region_bg = np.clip(region_bg + (direction * strength * 80.0 * region_w ** 1.5)[..., None], 0, 255)
        else:  # stipple: ordered-dither dots, denser near the person, in the separating tone
            ys, xs = np.mgrid[oy0:oy1, ox0:ox1]
            dots = ((strength * region_w) > _BAYER[ys % 4, xs % 4]) & (region_w > 0)
            dot_value = np.where(direction > 0, 255.0, 0.0)[..., None]
            region_bg = np.where(dots[..., None], dot_value, region_bg)

    result = background.copy()
    result[oy0:oy1, ox0:ox1] = np.rint(region_patch * region_alpha + region_bg * (1 - region_alpha)).astype(np.uint8)
    return result

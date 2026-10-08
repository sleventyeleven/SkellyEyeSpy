import numpy as np

from photobooth.compose.framing import clean_mask, cover_resize, person_bbox, place_person

OUT = (360, 270)
RED = (0, 0, 255)  # BGR: the "person" colour in these synthetic frames
BG = (200, 200, 200)


def scene(person_box, frame_size=(640, 360)):
    """Frame with a solid red rectangle for the person, plus its exact mask."""
    w, h = frame_size
    frame = np.full((h, w, 3), 30, dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    x0, y0, x1, y1 = person_box
    frame[y0:y1, x0:x1] = RED
    mask[y0:y1, x0:x1] = 255
    return frame, mask


def person_pixels(result):
    return (result[:, :, 2] > 200) & (result[:, :, 0] < 60)


def person_rows(result):
    return np.nonzero(person_pixels(result).any(axis=1))[0]


def person_cols(result):
    return np.nonzero(person_pixels(result).any(axis=0))[0]


def run(person_box, **kwargs):
    frame, mask = scene(person_box)
    background = np.full((100, 100, 3), BG, dtype=np.uint8)
    return place_person(frame, mask, background, OUT, **kwargs)


def height_of(result):
    rows = person_rows(result)
    return rows.max() - rows.min() + 1


def test_tall_person_filling_the_frame_is_scaled_down():
    result = run((250, 0, 390, 360))  # fills the whole camera height
    assert abs(height_of(result) - 0.72 * OUT[1]) <= 6
    assert (result[5, 5] == BG).all()  # background shows around them


def test_short_person_is_scaled_up_to_the_same_visible_height():
    result = run((300, 200, 340, 360))  # 160 px tall, 4:1 ratio 0.44 of frame
    assert abs(height_of(result) - 0.72 * OUT[1]) <= 6


def test_different_heights_end_up_the_same_size():
    tall = height_of(run((250, 0, 390, 360)))
    small = height_of(run((300, 160, 340, 360)))
    assert abs(tall - small) <= 8


def test_scale_up_is_capped_so_a_tiny_or_far_person_is_not_blown_up():
    result = run((300, 300, 330, 360))  # 60 px tall: wants 3.2x, capped at 1.8x
    assert abs(height_of(result) - 60 * 1.8) <= 6


def test_wide_group_is_limited_by_width_not_height():
    result = run((20, 100, 620, 360))  # very wide blob
    cols = person_cols(result)
    assert (cols.max() - cols.min() + 1) <= 0.92 * OUT[0] + 4


def test_person_is_centered_horizontally():
    result = run((500, 0, 600, 360))  # standing at the far right of the camera frame
    cols = person_cols(result)
    assert abs((cols.min() + cols.max()) / 2 - OUT[0] / 2) <= 4


def test_person_cut_off_at_camera_bottom_sits_flush_with_output_bottom():
    result = run((250, 100, 390, 360))
    assert person_rows(result).max() == OUT[1] - 1


def test_full_body_in_view_floats_slightly_above_the_bottom():
    result = run((250, 20, 390, 300))  # feet visible, not touching camera bottom
    assert person_rows(result).max() < OUT[1] - 1


def test_no_person_returns_plain_background():
    frame = np.full((360, 640, 3), 30, dtype=np.uint8)
    background = np.full((100, 100, 3), BG, dtype=np.uint8)
    result = place_person(frame, np.zeros((360, 640), np.uint8), background, OUT)
    assert result.shape == (OUT[1], OUT[0], 3) and (result == BG).all()


def test_stray_specks_are_ignored_but_a_real_person_is_kept():
    frame, mask = scene((250, 0, 390, 360))
    mask[10:16, 600:606] = 255  # tiny clutter blob far from the person
    cleaned = clean_mask(mask)
    assert cleaned[12, 602] == 0 and cleaned[200, 300] == 255
    assert person_bbox(cleaned) == (250, 0, 390, 360)


def test_only_specks_means_nobody():
    mask = np.zeros((360, 640), np.uint8)
    mask[10:16, 600:606] = 255
    assert person_bbox(clean_mask(mask)) is None


def test_cover_resize_fills_without_stretching():
    wide = np.zeros((100, 400, 3), np.uint8)
    wide[:, :200] = 255  # left half white, right half black
    out = cover_resize(wide, 100, 100)
    assert out.shape == (100, 100, 3)
    assert 0 < out.mean() < 255  # center crop straddles both halves, i.e. cropped not squashed


def test_small_size_ratio_is_honoured_even_for_a_person_filling_the_camera():
    # A minimum-scale floor of 0.35 used to turn a requested 35% into ~47% for anyone whose
    # visible height was most of the camera frame. 35% must now mean 35%.
    result = run((250, 0, 390, 360), height_ratio=0.35)
    assert abs(height_of(result) - 0.35 * OUT[1]) <= 5


def test_configured_defaults_match_the_tuned_values():
    from photobooth.config import ComposeConfig

    cfg = ComposeConfig()
    assert cfg.person_height_ratio == 0.35 and cfg.edge_mode == "stipple"
    assert cfg.outline_px == 5 and cfg.edge_strength == 0.3 and cfg.person_pop == 0.15
    assert cfg.background_soften == 0.33 and cfg.morph_probability == 0.33
    assert cfg.anchor_mode == "alternate" and cfg.person_scale_min <= 0.15

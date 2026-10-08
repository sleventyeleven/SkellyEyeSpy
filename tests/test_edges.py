import numpy as np
import pytest

from photobooth.compose.framing import EDGE_MODES, place_person, pop_person

from _tolerance import close
from test_backgrounds import make_service, varied_config  # noqa: F401  (fixture)

OUT = (360, 270)
ROW = 200  # a row through the torso, well inside the frame


def scene(person_value, bg_value):
    """Flat person rectangle (bottom-cut, like a bust shot) on a flat background."""
    frame = np.full((360, 640, 3), 30, np.uint8)
    mask = np.zeros((360, 640), np.uint8)
    frame[40:360, 250:390] = person_value
    mask[40:360, 250:390] = 255
    return frame, mask, np.full((100, 100, 3), bg_value, np.uint8)


def render(person_value, bg_value, **kw):
    frame, mask, background = scene(person_value, bg_value)
    kw.setdefault("height_ratio", 0.6)
    return place_person(frame, mask, background, OUT, **kw).astype(int)


def person_edge(person_value, bg_value):
    """Left-most column of the person, found from an edge-free render."""
    base = render(person_value, bg_value, edge_mode="none", outline_px=0)
    cols = np.nonzero(np.abs(base[ROW, :, 0] - person_value) <= 2)[0]
    return int(cols.min())


# ---- switches ----------------------------------------------------------------------------------------------


def test_none_and_zero_width_and_zero_strength_all_leave_the_scene_alone():
    base = render(140, 130, edge_mode="none", outline_px=0)
    same = lambda other: close(other, base, tol=2, frac=0.999)  # noqa: E731  (tolerant: see _tolerance.py)
    assert same(render(140, 130, edge_mode="separate", outline_px=0))
    assert same(render(140, 130, edge_mode="none", outline_px=14))
    assert same(render(140, 130, edge_mode="separate", outline_px=14, edge_strength=0.0))


def test_unknown_edge_mode_is_rejected():
    with pytest.raises(ValueError):
        render(140, 130, edge_mode="sparkle", outline_px=5)
    assert set(EDGE_MODES) == {"none", "separate", "stipple", "ring"}


# ---- separate: soft, no line, direction chosen against the person ---------------------------------------------


def test_separate_burns_when_the_tones_are_alike_like_a_contact_shadow():
    left = person_edge(140, 130)
    result = render(140, 130, edge_mode="separate", outline_px=14, edge_strength=0.5)
    assert result[ROW, left - 3, 0] < 130 - 8  # darkened next to the person
    assert abs(result[ROW, 5, 0] - 130) <= 1  # untouched far away


def test_separate_lightens_a_dark_person_against_a_much_lighter_scene():
    left = person_edge(40, 200)
    result = render(40, 200, edge_mode="separate", outline_px=14, edge_strength=0.5)
    assert result[ROW, left - 6, 0] > 200 + 8  # pushed even lighter (sampled past the person's own 3 px feather)


def test_separate_has_no_hard_edge():
    left = person_edge(140, 130)
    row = render(140, 130, edge_mode="separate", outline_px=14, edge_strength=0.5)[ROW, :, 0]
    outside = np.abs(np.diff(row[: left - 2]))
    assert outside.max() <= 8  # a gentle gradient, nothing like a ring


def test_separate_fades_out_with_distance():
    left = person_edge(140, 130)
    row = render(140, 130, edge_mode="separate", outline_px=14, edge_strength=0.5)[ROW, :, 0]
    close, far = 130 - row[left - 3], 130 - row[left - 12]
    assert close > far >= 0


# ---- stipple: alternating pixels, not a line ---------------------------------------------------------------------


def _touched(result, bg_value):
    return np.abs(result[:, :, 0] - bg_value) > 1


def test_stipple_is_a_scatter_of_dots_denser_near_the_person():
    left = person_edge(140, 130)
    result = render(140, 130, edge_mode="stipple", outline_px=12, edge_strength=0.6)
    touched = _touched(result, 130)
    rows = slice(150, 250)
    near = touched[rows, left - 4 : left - 1].mean()
    farther = touched[rows, left - 11 : left - 8].mean()
    assert 0.15 < near < 0.85  # alternating pixels: neither empty nor a solid line
    assert near > farther


def test_stipple_dots_take_the_separating_tone():
    left = person_edge(140, 130)
    alike = render(140, 130, edge_mode="stipple", outline_px=12, edge_strength=0.8)
    dots = alike[150:250, left - 4 : left - 1, 0]
    assert dots.min() == 0 and dots.max() <= 130  # similar tones -> dark dots (burn), never white

    left2 = person_edge(40, 220)
    light_scene = render(40, 220, edge_mode="stipple", outline_px=12, edge_strength=0.8)
    assert light_scene[150:250, left2 - 9 : left2 - 5, 0].max() == 255  # light scene -> white dots


def test_old_ring_is_still_available_and_is_a_solid_white_halo():
    left = person_edge(140, 130)
    result = render(140, 130, edge_mode="ring", outline_px=3)
    assert result[ROW, left - 2].min() >= 200  # solid white right at the edge


# ---- pop ----------------------------------------------------------------------------------------------------------


def test_pop_zero_is_a_no_op_and_pop_raises_saturation_and_contrast():
    rng = np.random.default_rng(1)
    patch = np.clip(rng.normal([90, 110, 150], [30, 30, 30], (60, 60, 3)), 0, 255).astype(np.uint8)
    alpha = np.ones((60, 60), np.float32)
    assert np.array_equal(pop_person(patch, alpha, 0.0), patch)
    popped = pop_person(patch, alpha, 1.0)
    luma = np.array([0.114, 0.587, 0.299])
    assert (popped.astype(float) @ luma).std() > (patch.astype(float) @ luma).std()
    chroma = lambda a: (a.max(axis=2).astype(int) - a.min(axis=2)).mean()  # noqa: E731
    assert chroma(popped) > chroma(patch)


# ---- studio plumbing ------------------------------------------------------------------------------------------------


def test_studio_accepts_edge_settings_and_rejects_bad_ones(varied_config, tmp_path):  # noqa: F811
    service = make_service(varied_config, tmp_path)
    service.draft_capture()
    params = service.draft_render(shots=2, edge="stipple", strength=0.7, pop=0.4, outline=9)["params"]
    assert params == {"edge": "stipple", "strength": 0.7, "pop": 0.4, "outline": 9}
    with pytest.raises(ValueError):
        service.draft_render(edge="sparkle")


def test_edge_setting_changes_the_render_and_page_exposes_it(varied_config, tmp_path):  # noqa: F811
    import cv2
    from flask import Flask

    from photobooth.web import create_blueprint

    service = make_service(varied_config, tmp_path)
    service.draft_capture()
    service.draft_render(shots=1, seed=1, edge="none", vintage=False, morph=0.0)
    plain = cv2.imread(service.draft_path("shot_0"))
    service.draft_render(edge="separate", outline=14, strength=1.0)
    separated = cv2.imread(service.draft_path("shot_0"))
    assert not np.array_equal(plain, separated)

    app = Flask(__name__)
    app.register_blueprint(create_blueprint(service))
    page = app.test_client().get("/photobooth/view").data.decode()
    assert 'id="edge"' in page and 'id="estrength"' in page and 'id="pop"' in page and "ring (old glow)" in page

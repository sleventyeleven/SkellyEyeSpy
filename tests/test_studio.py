import cv2
import numpy as np
import pytest
from flask import Flask

from photobooth.compose.backgrounds import anchor_for
from photobooth.compose.framing import count_people, drop_small_people, place_person, soften_background
from photobooth.web import create_blueprint

from _tolerance import close
from test_backgrounds import make_service, varied_config  # noqa: F401  (fixture)
from test_booth_service import FakeCamera, FakePrinter

OUT = (360, 270)
RED = (0, 0, 255)
BG = (200, 200, 200)


def scene(boxes, frame_size=(640, 360)):
    w, h = frame_size
    frame = np.full((h, w, 3), 30, dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    for x0, y0, x1, y1 in boxes:
        frame[y0:y1, x0:x1] = RED
        mask[y0:y1, x0:x1] = 255
    return frame, mask


def red_pixels(result):
    return (result[:, :, 2] > 200) & (result[:, :, 0] < 60)


def run(boxes, **kw):
    frame, mask = scene(boxes)
    return place_person(frame, mask, np.full((100, 100, 3), BG, np.uint8), OUT, **kw)


def center_x(result):
    cols = np.nonzero(red_pixels(result).any(axis=0))[0]
    return (cols.min() + cols.max()) / 2


# ---- anchor ---------------------------------------------------------------------------------


def test_default_anchor_is_centered():
    assert abs(center_x(run([(250, 0, 390, 360)])) - 180) <= 3


@pytest.mark.parametrize("anchor", [0.3, 0.7])
def test_anchor_places_the_person_there(anchor):
    result = run([(250, 0, 390, 360)], anchor_x=anchor, height_ratio=0.5)
    assert abs(center_x(result) - anchor * OUT[0]) <= 4


def test_anchor_is_clamped_so_the_person_stays_inside_the_frame():
    result = run([(250, 0, 390, 360)], anchor_x=0.98, height_ratio=0.72)
    cols = np.nonzero(red_pixels(result).any(axis=0))[0]
    assert cols.max() < OUT[0] - 1 and cols.min() > OUT[0] // 2  # pushed right, not cut off


def test_anchor_for_modes():
    assert anchor_for(1, 0, "center") == 0.5
    assert anchor_for(1, 3, "left", 0.2) == pytest.approx(0.3)
    assert anchor_for(1, 3, "right", 0.2) == pytest.approx(0.7)
    alt = [anchor_for(5, i, "alternate", 0.2) for i in range(4)]
    assert alt[0] == alt[2] and alt[1] == alt[3] and alt[0] != alt[1]
    rand = {round(anchor_for(s, i, "random", 0.2), 3) for s in range(10) for i in range(4)}
    assert rand <= {0.3, 0.5, 0.7} and len(rand) >= 2
    assert anchor_for(9, 2, "random") == anchor_for(9, 2, "random")  # reproducible from the seed


# ---- groups ----------------------------------------------------------------------------------


def test_count_people_and_ignored_bystanders():
    _frame, mask = scene([(150, 20, 280, 360), (360, 150, 470, 360), (600, 300, 628, 340)])
    assert count_people(mask, 0.12) == (2, 1)  # adult + child kept, tiny distant blob ignored
    assert count_people(mask, 0.0) == (3, 0)


def test_small_distant_bystander_does_not_shrink_the_group():
    with_bystander = run([(150, 20, 280, 360), (360, 150, 470, 360), (600, 300, 628, 340)], group_min_area_ratio=0.12)
    without = run([(150, 20, 280, 360), (360, 150, 470, 360)], group_min_area_ratio=0.12)
    assert close(with_bystander, without, tol=2, frac=0.999)


def test_a_child_next_to_an_adult_is_kept_and_the_tallest_sets_the_scale():
    result = run([(150, 20, 280, 360), (360, 170, 470, 360)], height_ratio=0.5, group_min_area_ratio=0.12)
    rows = np.nonzero(red_pixels(result).any(axis=1))[0]
    assert abs((rows.max() - rows.min() + 1) - 0.5 * OUT[1]) <= 6  # group height = adult's
    cols = np.nonzero(red_pixels(result).any(axis=0))[0]
    assert np.diff(cols).max() > 1  # two separate people, not merged into one slab


def test_wide_group_still_fits_the_frame():
    result = run([(30, 100, 300, 360), (340, 100, 610, 360)], height_ratio=0.9, anchor_x=0.9)
    cols = np.nonzero(red_pixels(result).any(axis=0))[0]
    assert cols.min() >= 0 and cols.max() <= OUT[0] - 1


def test_drop_small_people_leaves_a_single_blob_alone():
    _frame, mask = scene([(250, 0, 390, 360)])
    assert np.array_equal(drop_small_people(mask, 0.5), mask)


# ---- outline + soften ----------------------------------------------------------------------------


def test_outline_adds_a_halo_that_plain_does_not_have():
    plain = run([(250, 60, 390, 360)], outline_px=0)
    outlined = run([(250, 60, 390, 360)], outline_px=4, outline_color=(255, 255, 255))
    cols = np.nonzero(red_pixels(plain).any(axis=0))[0]
    row = int(np.nonzero(red_pixels(plain).any(axis=1))[0].mean())
    just_outside = cols.min() - 3  # inside the 4 px halo, outside the feathered edge
    assert abs(int(plain[row, just_outside, 0]) - BG[0]) <= 3
    assert outlined[row, just_outside].min() > BG[0] + 20  # whiter than the 200-grey background


def test_soften_zero_is_a_no_op_and_soften_calms_a_dark_noisy_scene():
    rng = np.random.default_rng(3)
    dark = (rng.integers(0, 90, (120, 160, 3))).astype(np.uint8)
    assert np.array_equal(soften_background(dark, 0.0), dark)
    soft = soften_background(dark, 0.8)
    assert soft.mean() > dark.mean() + 15  # dark scenes are lifted so they do not print as black
    assert soft.std() < dark.std()  # and flattened


# ---- studio (draft) service ----------------------------------------------------------------------------


def test_render_before_capture_is_an_error(varied_config, tmp_path):  # noqa: F811
    service = make_service(varied_config, tmp_path)
    with pytest.raises(RuntimeError):
        service.draft_render()
    with pytest.raises(RuntimeError):
        service.draft_print()


def test_capture_once_then_render_many_times_without_touching_the_camera(varied_config, tmp_path):  # noqa: F811
    service = make_service(varied_config, tmp_path)
    FakeCamera.frames_read = 0
    info = service.draft_capture()
    after_capture = FakeCamera.frames_read
    assert info["people"] == 1
    for _ in range(3):
        service.draft_render(shots=3)
    service.draft_render(reroll_all=True, size=0.5)
    assert FakeCamera.frames_read == after_capture


def test_same_seed_same_reel_and_reroll_all_changes_it(varied_config, tmp_path):  # noqa: F811
    service = make_service(varied_config, tmp_path)
    service.draft_capture()
    a = service.draft_render(shots=4, seed=77, vintage=False)
    b = service.draft_render(shots=4, vintage=False)  # no seed given: the draft keeps its own
    assert a["seed"] == b["seed"] == 77 and a["backgrounds"] == b["backgrounds"]
    c = service.draft_render(reroll_all=True, vintage=False)
    assert c["seed"] != 77 and c["backgrounds"] != a["backgrounds"]


def test_reroll_replaces_only_that_frame(varied_config, tmp_path):  # noqa: F811
    service = make_service(varied_config, tmp_path)
    service.draft_capture()
    before = service.draft_render(shots=4, seed=5, vintage=False)
    after = service.draft_render(reroll=2, vintage=False)
    assert after["seed"] == 5 and list(after["overrides"]) == ["2"]
    assert after["backgrounds"][2] != before["backgrounds"][2]
    for i in (0, 1, 3):
        assert after["backgrounds"][i] == before["backgrounds"][i]
    # a later render keeps the rerolled frame; a new seed clears the overrides
    assert service.draft_render(vintage=False)["backgrounds"][2] == after["backgrounds"][2]
    assert service.draft_render(seed=6, vintage=False)["overrides"] == {}


def test_slider_values_stick_between_renders(varied_config, tmp_path):  # noqa: F811
    service = make_service(varied_config, tmp_path)
    service.draft_capture()
    service.draft_render(shots=2, size=0.5, outline=0, anchor="left")
    params = service.draft_render(soften=0.9)["params"]
    assert params == {"size": 0.5, "outline": 0, "anchor": "left", "soften": 0.9}


def test_size_setting_really_changes_the_render(varied_config, tmp_path):  # noqa: F811
    service = make_service(varied_config, tmp_path)
    service.draft_capture()
    service.draft_render(shots=1, seed=1, size=0.8, vintage=False, morph=0.0)
    big = cv2.imread(service.draft_path("shot_0"))
    service.draft_render(size=0.4)
    small = cv2.imread(service.draft_path("shot_0"))
    assert not np.array_equal(big, small)


def test_bad_settings_are_rejected(varied_config, tmp_path):  # noqa: F811
    service = make_service(varied_config, tmp_path)
    service.draft_capture()
    with pytest.raises(ValueError):
        service.draft_render(anchor="diagonal")
    with pytest.raises(ValueError):
        service.draft_render(bogus=1)


def test_print_sends_the_rendered_strip_and_does_not_recapture(varied_config, tmp_path):  # noqa: F811
    varied_config.printer.enabled = True
    varied_config.printer.bluetooth_address = "AA:BB:CC:DD:EE:FF"
    service = make_service(varied_config, tmp_path)
    service.draft_capture()
    service.draft_render(shots=3, seed=2)
    FakePrinter.printed = []
    reads = FakeCamera.frames_read
    result = service.draft_print()
    assert result.printed and FakePrinter.printed[-1] == service._draft.strip.size
    assert FakeCamera.frames_read == reads
    assert result.seed == 2


def test_print_reports_when_printing_is_disabled(varied_config, tmp_path):  # noqa: F811
    varied_config.printer.enabled = False
    service = make_service(varied_config, tmp_path)
    service.draft_capture()
    service.draft_render()
    result = service.draft_print()
    assert result.printed is False and "disabled" in result.error


def test_draft_files_are_served_by_name(varied_config, tmp_path):  # noqa: F811
    service = make_service(varied_config, tmp_path)
    assert service.draft_path("strip") is None
    service.draft_capture()
    service.draft_render(shots=2)
    for name in ("strip", "print", "capture", "mask", "shot_0", "shot_1"):
        assert service.draft_path(name), name
    assert service.draft_path("shot_5") is None and service.draft_path("../etc/passwd") is None


def test_fewer_frames_removes_stale_shot_files(varied_config, tmp_path):  # noqa: F811
    service = make_service(varied_config, tmp_path)
    service.draft_capture()
    service.draft_render(shots=4)
    service.draft_render(shots=2)
    assert service.draft_path("shot_3") is None


def test_background_images_are_downscaled_once(varied_config, tmp_path):  # noqa: F811
    big = tmp_path / "big_43.png"
    cv2.imwrite(str(big), np.full((1500, 2000, 3), 90, np.uint8))
    varied_config.compose.backgrounds = [str(big)]
    service = make_service(varied_config, tmp_path)
    loaded = service._load_cached(str(big))
    assert loaded.shape[1] == varied_config.compose.background_max_width
    assert service._load_cached(str(big)) is loaded


# ---- endpoints ---------------------------------------------------------------------------------------------


def test_studio_endpoints_end_to_end(varied_config, tmp_path):  # noqa: F811
    varied_config.printer.enabled = True
    varied_config.printer.bluetooth_address = "AA:BB:CC:DD:EE:FF"
    app = Flask(__name__)
    app.register_blueprint(create_blueprint(make_service(varied_config, tmp_path)))
    client = app.test_client()

    assert client.post("/api/photobooth/draft/render").status_code == 500  # nothing captured
    assert client.get("/api/photobooth/draft/image/strip").status_code == 404

    assert client.post("/api/photobooth/draft/capture").get_json()["people"] == 1
    body = client.post("/api/photobooth/draft/render?shots=3&seed=9&size=0.5&anchor=right&outline=2&soften=0.3&morph=0.1&vintage=0").get_json()
    assert body["status"] == "success" and body["seed"] == 9 and body["shots"] == 3
    assert body["params"] == {"size": 0.5, "anchor": "right", "outline": 2, "soften": 0.3, "morph": 0.1}
    assert client.get("/api/photobooth/draft/image/strip").status_code == 200
    assert client.get("/api/photobooth/draft/image/print").status_code == 200
    assert client.get("/api/photobooth/draft/image/shot_2").status_code == 200

    assert client.post("/api/photobooth/draft/render?reroll=1").get_json()["overrides"].keys() == {"1"}
    assert client.post("/api/photobooth/draft/render?anchor=diagonal").status_code == 400
    printed = client.post("/api/photobooth/draft/print").get_json()
    assert printed["status"] == "success" and printed["printed"] is True


def test_studio_is_the_default_page_mode(varied_config, tmp_path):  # noqa: F811
    app = Flask(__name__)
    app.register_blueprint(create_blueprint(make_service(varied_config, tmp_path)))
    page = app.test_client().get("/photobooth/view").data.decode()
    assert 'value="studio" checked' in page and "Print this reel" in page and "draft/render" in page
    assert 'id="size"' in page and 'id="anchor"' in page


def test_aim_view_reports_people_and_still_renders_the_composite(varied_config, tmp_path):  # noqa: F811
    service = make_service(varied_config, tmp_path)
    data = service.aim_jpeg(composite=True)
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    assert frame.shape[1] > varied_config.compose.working_width  # camera view plus the composite beside it


def test_moving_a_framing_slider_does_not_rebuild_the_backgrounds(varied_config, tmp_path, monkeypatch):  # noqa: F811
    import photobooth.service as service_module

    calls = []
    real = service_module.make_background
    monkeypatch.setattr(service_module, "make_background", lambda *a, **k: calls.append(a[:2]) or real(*a, **k))

    service = make_service(varied_config, tmp_path)
    service.draft_capture()
    service.draft_render(shots=4, seed=3)
    built = len(calls)
    assert built == 4
    service.draft_render(size=0.5)
    service.draft_render(outline=0, soften=0.9, anchor="left")
    assert len(calls) == built  # framing changes reuse the cached backgrounds
    service.draft_render(morph=0.9)  # a variation setting does change them
    assert len(calls) == built + 4
    service.draft_render(reroll=1)  # and a reroll rebuilds just that frame
    assert len(calls) == built + 5


# ---- per-frame studio changes: stand position and a retaken photo -------------------------------------------


def _shots(service, n):
    return [cv2.imread(service.draft_path(f"shot_{i}")) for i in range(n)]


def test_pinning_one_frames_position_changes_only_that_frame(varied_config, tmp_path):  # noqa: F811
    service = make_service(varied_config, tmp_path)
    service.draft_capture()
    before = service.draft_render(shots=3, seed=5, anchor="center")
    shots = _shots(service, 3)
    after = service.draft_render(frame=1, position="left")
    assert after["frame_anchor"] == {"1": "left"}
    moved = _shots(service, 3)
    assert close(shots[0], moved[0]) and close(shots[2], moved[2]) and not close(shots[1], moved[1])
    assert before["backgrounds"] == after["backgrounds"]  # same backgrounds, only the person moved
    assert service.draft_render(frame=1, position="auto")["frame_anchor"] == {}
    assert close(shots[1], _shots(service, 3)[1])  # back to automatic restores the original look


def test_position_must_be_valid_and_name_a_frame(varied_config, tmp_path):  # noqa: F811
    service = make_service(varied_config, tmp_path)
    service.draft_capture()
    service.draft_render(shots=2)
    with pytest.raises(ValueError, match="position"):
        service.draft_render(frame=0, position="diagonal")
    with pytest.raises(ValueError, match="frame"):
        service.draft_render(position="left")


def test_retaking_one_frame_uses_a_new_photo_for_it_alone(varied_config, tmp_path):  # noqa: F811
    service = make_service(varied_config, tmp_path)
    service.draft_capture()
    service.draft_render(shots=3, seed=8)
    original = _shots(service, 3)

    green = np.zeros((720, 1280, 3), np.uint8)
    green[:] = (0, 220, 0)
    service._camera_factory = lambda cfg: FakeCamera(cfg, frame=green)
    info = service.draft_retake(1)
    assert info["frame"] == 1 and info["people"] == 1
    result = service.draft_render()
    assert result["retaken"] == [1]
    now = _shots(service, 3)
    assert close(original[0], now[0]) and close(original[2], now[2]) and not close(original[1], now[1])

    assert service.draft_render(restore=1)["retaken"] == []
    assert close(original[1], _shots(service, 3)[1])


def test_retake_needs_a_rendered_reel_and_a_valid_frame(varied_config, tmp_path):  # noqa: F811
    service = make_service(varied_config, tmp_path)
    with pytest.raises(RuntimeError, match="Render a reel"):
        service.draft_retake(0)
    service.draft_capture()
    service.draft_render(shots=2)
    with pytest.raises(ValueError, match="frame"):
        service.draft_retake(5)


def test_a_new_capture_clears_per_frame_changes(varied_config, tmp_path):  # noqa: F811
    service = make_service(varied_config, tmp_path)
    service.draft_capture()
    service.draft_render(shots=2, seed=3)
    service.draft_render(frame=0, position="right")
    service.draft_capture()
    assert service.draft_render(shots=2)["frame_anchor"] == {}


def test_per_frame_endpoints(varied_config, tmp_path):  # noqa: F811
    app = Flask(__name__)
    app.register_blueprint(create_blueprint(make_service(varied_config, tmp_path)))
    client = app.test_client()
    client.post("/api/photobooth/draft/capture")
    client.post("/api/photobooth/draft/render?shots=2&seed=1")
    assert client.post("/api/photobooth/draft/render?frame=0&position=right").get_json()["frame_anchor"] == {"0": "right"}
    assert client.post("/api/photobooth/draft/render?frame=0&position=up").status_code == 400
    assert client.post("/api/photobooth/draft/retake").status_code == 400
    assert client.post("/api/photobooth/draft/retake?index=9").status_code == 400
    assert client.post("/api/photobooth/draft/retake?index=1").get_json()["status"] == "success"
    assert client.post("/api/photobooth/draft/render").get_json()["retaken"] == [1]


def test_page_has_the_boxed_sections_in_order_and_smooth_scaled_down_preview(varied_config, tmp_path):  # noqa: F811
    app = Flask(__name__)
    app.register_blueprint(create_blueprint(make_service(varied_config, tmp_path)))
    page = app.test_client().get("/photobooth/view").data.decode()
    assert page.index('id="voicePanel"') < page.index('id="photoCard"') < page.index('id="studio"')  # voice first, then photo, then render
    for placeholder in ('id="renderTools"', 'id="previews"', "slot", "retakeFrame", "&position="):
        assert placeholder in page
    assert "imageRendering" in page and "nearest neighbour" in page  # scaled-down 1-bit previews are smoothed, not decimated

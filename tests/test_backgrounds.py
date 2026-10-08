import cv2
import numpy as np
import pytest
from flask import Flask

from _tolerance import close
from photobooth.compose.backgrounds import _flip_allowed, make_background, morph, path_sequence, vary
from photobooth.config import ComposeConfig, PhotoBoothConfig
from photobooth.service import PhotoBoothService
from photobooth.web import create_blueprint

from test_booth_service import FakeCamera, FakePrinter, FakeSegmenter

POOL = ["a.png", "b.png", "c.png", "d.png", "e.png", "f.png"]


def test_sequence_is_deterministic_for_a_seed_and_differs_across_seeds():
    assert path_sequence(7, POOL, 4) == path_sequence(7, POOL, 4)
    assert len({tuple(path_sequence(seed, POOL, 4)) for seed in range(20)}) > 10


def test_sequence_has_no_repeats_within_a_pool_pass():
    for seed in range(30):
        assert len(set(path_sequence(seed, POOL, 6))) == 6


def test_sequence_never_repeats_a_file_back_to_back_even_when_cycling():
    for seed in range(30):
        seq = path_sequence(seed, POOL[:3], 12)
        assert all(a != b for a, b in zip(seq, seq[1:]))


def test_a_longer_sequence_starts_with_the_shorter_one():
    for seed in range(10):
        assert path_sequence(seed, POOL, 3) == path_sequence(seed, POOL, 6)[:3]


def test_not_randomized_keeps_the_pool_order():
    assert path_sequence(1, POOL, 8, randomize=False) == POOL + POOL[:2]


def test_empty_pool_is_an_error():
    with pytest.raises(ValueError):
        path_sequence(1, [], 2)


def _loader(images):
    return lambda path: images[path]


def _images():
    rng = np.random.default_rng(0)
    return {p: rng.integers(0, 255, (120, 160, 3), dtype=np.uint8) for p in POOL}


def test_same_seed_and_index_give_identical_background():
    images, cfg = _images(), ComposeConfig()
    a, label_a = make_background(5, 2, POOL, _loader(images), cfg)
    b, label_b = make_background(5, 2, POOL, _loader(images), cfg)
    assert close(a, b) and label_a == label_b


def test_different_seeds_give_different_backgrounds():
    images, cfg = _images(), ComposeConfig()
    results = [make_background(seed, 0, POOL, _loader(images), cfg)[0].tobytes() for seed in range(8)]
    assert len(set(results)) > 4


def test_frame_i_does_not_depend_on_how_long_the_reel_is():
    # so the aim preview of frame 0 is the real frame 0 of the reel that follows
    images, cfg = _images(), ComposeConfig()
    short = path_sequence(3, POOL, 2)
    long_ = path_sequence(3, POOL, 5)
    a, _ = make_background(3, 1, POOL, _loader(images), cfg, sequence=short)
    b, _ = make_background(3, 1, POOL, _loader(images), cfg, sequence=long_)
    assert close(a, b)


def test_variation_keeps_aspect_and_changes_the_image():
    image = _images()["a.png"]
    out, label = vary(image, np.random.default_rng(1), flip_ok=True, zoom_max=1.4)
    assert abs(out.shape[1] / out.shape[0] - image.shape[1] / image.shape[0]) < 0.02
    assert label.startswith("zoom")
    assert out.shape != image.shape or not np.array_equal(out, image)


def test_everything_off_returns_the_plain_image():
    images = _images()
    cfg = ComposeConfig(randomize_backgrounds=False, vary_backgrounds=False, morph_probability=0.0)
    out, label = make_background(9, 0, POOL, _loader(images), cfg)
    assert np.array_equal(out, images["a.png"]) and label == "a"


def test_morph_produces_a_blend_of_both_images():
    black = np.zeros((90, 120, 3), np.uint8)
    white = np.full((90, 120, 3), 255, np.uint8)
    out = morph(black, white, np.random.default_rng(2))
    assert out.min() < 40 and out.max() > 215  # both themes visible somewhere
    assert ((out > 40) & (out < 215)).any()  # and a soft transition between them


def test_morph_probability_one_always_morphs_and_labels_it():
    images = _images()
    cfg = ComposeConfig(morph_probability=1.0, vary_backgrounds=False)
    for seed in range(5):
        assert "morphed with" in make_background(seed, 0, POOL, _loader(images), cfg)[1]


def test_lettering_backgrounds_are_never_mirrored():
    assert not _flip_allowed(["assets/saloon_43.png"], ["saloon", "noflip"])
    assert not _flip_allowed(["x_noflip.png", "kraken.png"], ["saloon", "noflip"])
    assert _flip_allowed(["kraken.png"], ["saloon", "noflip"])

    asym = np.zeros((60, 80, 3), np.uint8)
    asym[:, :40] = 255  # white left half
    for seed in range(40):
        out, _ = vary(asym, np.random.default_rng(seed), flip_ok=False, zoom_max=1.0)
        assert int(out[30, 5, 1]) > int(out[30, -5, 1]) - 60  # never swapped sides (tone jitter only)


# ---- service level: seeds, glob pools, preview == reel ---------------------------------


@pytest.fixture
def varied_config(tmp_path):
    for i, shade in enumerate((30, 90, 150, 210)):
        cv2.imwrite(str(tmp_path / f"theme{i}_43.png"), np.full((90, 120, 3), shade, dtype=np.uint8))
    cfg = PhotoBoothConfig()
    cfg.compose.backgrounds = [str(tmp_path / "theme*_43.png")]  # a glob, like dropping art in a folder
    cfg.compose.sepia_strength = cfg.compose.vignette_strength = cfg.compose.grain_amount = 0
    cfg.printer.enabled = False
    return cfg


def make_service(cfg, tmp_path):
    return PhotoBoothService(
        cfg, output_dir=str(tmp_path / "out"), camera_factory=FakeCamera,
        segmenter_factory=FakeSegmenter, printer_factory=FakePrinter,
    )


def test_glob_pool_picks_up_every_matching_file(varied_config, tmp_path):
    assert len(make_service(varied_config, tmp_path)._pool()) == 4


def test_same_seed_reproduces_the_reel_and_a_new_seed_changes_it(varied_config, tmp_path):
    service = make_service(varied_config, tmp_path)
    a = service.strip_one_shot(shots=4, print_it=False, vintage=False, seed=123)
    pixels_a = cv2.imread(a.image_path)
    b = service.strip_one_shot(shots=4, print_it=False, vintage=False, seed=123)
    assert a.seed == b.seed == 123 and a.backgrounds == b.backgrounds
    assert close(pixels_a, cv2.imread(b.image_path))
    c = service.strip_one_shot(shots=4, print_it=False, vintage=False, seed=124)
    assert c.backgrounds != a.backgrounds and not np.array_equal(pixels_a, cv2.imread(c.image_path))


def test_each_unseeded_reel_is_unique(varied_config, tmp_path):
    service = make_service(varied_config, tmp_path)
    seeds = {service.strip_one_shot(shots=4, print_it=False).seed for _ in range(6)}
    assert len(seeds) == 6


def test_aim_preview_seed_is_the_seed_the_next_reel_uses_then_rotates(varied_config, tmp_path):
    service = make_service(varied_config, tmp_path)
    service.aim_jpeg(composite=True)
    previewed = service._preview_seed
    assert previewed is not None
    result = service.strip_one_shot(shots=2, print_it=False)
    assert result.seed == previewed and service._preview_seed is None
    service.aim_jpeg(composite=True)
    assert service._preview_seed != previewed


def test_multi_shot_strip_uses_one_seed_and_distinct_backgrounds(varied_config, tmp_path):
    service = make_service(varied_config, tmp_path)
    service.strip_start(3, seed=9)
    labels = [service.strip_shot(vintage=False)["background"] for _ in range(3)]
    assert service.strip_seed == 9
    assert len({lab.split(",")[0] for lab in labels}) == 3  # three different themes
    assert service.strip_finish(print_it=False).seed == 9


def test_endpoints_accept_and_return_the_seed(varied_config, tmp_path):
    app = Flask(__name__)
    app.register_blueprint(create_blueprint(make_service(varied_config, tmp_path)))
    client = app.test_client()

    body = client.post("/api/photobooth/strip/oneshot?shots=3&print=0&vintage=0&seed=42").get_json()
    assert body["seed"] == 42 and len(body["backgrounds"]) == 3
    assert client.post("/api/photobooth/strip/start?shots=2&seed=7").get_json()["seed"] == 7
    assert client.post("/api/photobooth/strip/shot").get_json()["seed"] == 7
    assert client.post("/api/photobooth/snap?print=0&seed=5").get_json()["seed"] == 5
    assert b"seed" in client.get("/photobooth/view").data

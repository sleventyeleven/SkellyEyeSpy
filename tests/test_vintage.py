import numpy as np

from photobooth.compose.vintage import apply_vintage


def test_output_shape_and_dtype_preserved():
    image = np.random.randint(0, 255, (50, 60, 3), dtype=np.uint8)
    result = apply_vintage(image)
    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_zero_strength_grain_is_deterministic():
    image = np.full((10, 10, 3), 128, dtype=np.uint8)
    rng = np.random.default_rng(0)
    result = apply_vintage(image, sepia_strength=0, vignette_strength=0, grain_amount=0, rng=rng)
    # no sepia, no vignette, no grain -- should be unchanged
    assert np.array_equal(result, image)


def test_vignette_darkens_corners_relative_to_center():
    image = np.full((100, 100, 3), 200, dtype=np.uint8)
    result = apply_vintage(image, sepia_strength=0, vignette_strength=0.5, grain_amount=0)
    center = result[50, 50].astype(int).sum()
    corner = result[0, 0].astype(int).sum()
    assert corner < center

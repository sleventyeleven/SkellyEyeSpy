import numpy as np

from photobooth.compose.compositor import composite_on_background


def test_output_shape_matches_frame():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    mask = np.zeros((100, 200), dtype=np.uint8)
    background = np.full((50, 80, 3), 128, dtype=np.uint8)  # deliberately different size

    result = composite_on_background(frame, mask, background)
    assert result.shape == frame.shape


def test_fully_masked_out_pixel_is_background():
    frame = np.full((20, 20, 3), 255, dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=np.uint8)  # no person anywhere
    background = np.full((20, 20, 3), 10, dtype=np.uint8)

    result = composite_on_background(frame, mask, background)
    # feathering only affects edges; center of an all-zero mask stays background
    assert abs(int(result[10, 10, 0]) - 10) < 5


def test_fully_masked_in_pixel_is_frame():
    frame = np.full((20, 20, 3), 255, dtype=np.uint8)
    mask = np.full((20, 20), 255, dtype=np.uint8)  # person everywhere
    background = np.full((20, 20, 3), 10, dtype=np.uint8)

    result = composite_on_background(frame, mask, background)
    assert abs(int(result[10, 10, 0]) - 255) < 5

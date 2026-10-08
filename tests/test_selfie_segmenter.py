import numpy as np

from photobooth.segmentation.selfie import SelfieSegmenter


class _FakeMask:
    def __init__(self, array):
        self._array = array

    def numpy_view(self):
        return self._array


class _FakeResult:
    def __init__(self, probability):
        self.confidence_masks = [_FakeMask(probability[:, :, None])]  # model returns HxWx1


class _FakeSegmenter:
    def __init__(self, probability):
        self._probability = probability

    def segment(self, _image):
        return _FakeResult(self._probability)


def _segmenter_returning(probability):
    seg = SelfieSegmenter.__new__(SelfieSegmenter)  # skip loading the real model
    seg._segmenter = _FakeSegmenter(probability)
    return seg


def test_mask_is_255_where_person_probability_exceeds_threshold():
    probability = np.zeros((4, 6), dtype=np.float32)
    probability[1:3, 2:4] = 0.9
    probability[0, 0] = 0.4  # below default threshold
    frame = np.zeros((4, 6, 3), dtype=np.uint8)

    mask = _segmenter_returning(probability).person_mask(frame)

    assert mask.shape == (4, 6) and mask.dtype == np.uint8
    assert set(np.unique(mask)) == {0, 255}
    assert mask[1, 2] == 255 and mask[0, 0] == 0
    assert (mask == 255).sum() == 4


def test_threshold_is_respected():
    probability = np.full((2, 2), 0.3, dtype=np.float32)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    seg = _segmenter_returning(probability)
    assert (seg.person_mask(frame, threshold=0.5) == 255).sum() == 0
    assert (seg.person_mask(frame, threshold=0.2) == 255).sum() == 4

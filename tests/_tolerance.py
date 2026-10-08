import numpy as np


def close(a, b, tol=16, frac=0.995):
    """True when at least `frac` of pixels agree to within `tol` levels.

    Seeds reproduce the *look* of a reel, not bit-exact pixels: on the Windows dev PC used
    for this project, identical numpy float expressions were observed returning different
    results from run to run (pure numpy, no OpenCV, reproducible with tests/../hw_stress
    style loops; the Radxa board's numpy was rock solid). The differences are tiny, so the
    tests compare with a tolerance instead of exact equality.
    """
    if a.shape != b.shape:
        return False
    diff = np.abs(a.astype(np.int32) - b.astype(np.int32))
    return float((diff <= tol).mean()) >= frac

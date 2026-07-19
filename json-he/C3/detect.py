"""Blue-LED detector — brightest-blue-area weighted centroid.

Contract:  detect(bgr) -> (u, v, confidence) | None.
Single job: find the brightest blue region and return its intensity-weighted
centroid. u is weighted for sub-pixel accuracy (u matters more than v here).
Stateless, per-frame; temporal smoothing (if wanted) lives downstream.

Imaging assumption (Ae/Awb off, ~800us, near-black background): the LED is the
only blue feature and is far brighter than its glossy-screen reflection, so the
score-weighted centroid sits on the direct LED, not the dim reflection tail.
All spatial params are fractions of frame width -> resolution-relative.
"""
import cv2
import numpy as np

_SCORE_MAX = 255.0 * 255.0   # max of (B-R)*V; normalizes the peak to [0,1]
_PRESENT_FLOOR = 0.02        # min normalized peak for "LED present" (pen down)
_CORE_REL = 0.5              # brightest-blue area = pixels >= this fraction of peak
_BLUR_FRAC = 0.005           # blur sigma as a fraction of width
_CONF_REF = 0.40             # peak/MAX that maps to confidence 1.0


def _score(bgr):
    """blueness (B-R) * brightness (V), blurred. Global peak = the LED."""
    bgr = bgr.astype(np.float32)
    b, r = bgr[..., 0], bgr[..., 2]
    v = bgr.max(2)
    sigma = max(1.0, _BLUR_FRAC * bgr.shape[1])
    return cv2.GaussianBlur(np.clip(b - r, 0, None) * v, (0, 0), sigma)


def detect(bgr, top_mask_frac=0.5):
    """Return (u, v, confidence) of the LED, or None if none present.

    top_mask_frac: rows above this fraction of height are ignored (off-screen
      clutter). Default 0.5; calibration can tighten it.
    """
    h = bgr.shape[0]
    s = _score(bgr)
    if top_mask_frac > 0:
        s[: int(top_mask_frac * h)] = 0

    peak = float(s.max())
    if peak / _SCORE_MAX < _PRESENT_FLOOR:
        return None  # pen up / no LED

    # Brightest blue area = the peak-containing blob of the >=CORE_REL*peak mask.
    mask = (s >= _CORE_REL * peak).astype(np.uint8)
    n, lab, _, _ = cv2.connectedComponentsWithStats(mask, 8)
    _, _, _, (px, py) = cv2.minMaxLoc(s)
    blob = (lab == lab[py, px])

    ys, xs = np.where(blob)
    w = s[ys, xs]  # weight by blue score -> sub-pixel, favors the bright core
    u = float((xs * w).sum() / w.sum())
    v = float((ys * w).sum() / w.sum())

    conf = float(min(1.0, (peak / _SCORE_MAX) / _CONF_REF))
    return u, v, conf

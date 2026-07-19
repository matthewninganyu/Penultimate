"""Shared blue-LED detection front-end for C3 candidates A and B.

Contract: BGR frame -> (u, v, confidence) of the pen-tip CONTACT, or None.
A uses u only; B uses (u, v). Detection is identical for both.

Physical target (from the sample frames): the lit tip on the glass is a
WHITE-ISH saturated CORE (all channels high, low saturation) wrapped in a
LIGHT-BLUE glow. The pen barrel above and the reflection streak below are blue
but dimmer; white glare (tape, screen, overhead) is a white core with NO strong
blue glow. Two-stage detector:
  1. SEED = global peak of blueness*brightness = (B-R)*V, blurred. This reliably
     lands in the LED bloom (brightest blue-white), NOT on tape/screen/barrel.
     Doubles as the pen-up/down gate via a normalized present-floor.
  2. LOCALIZE = the WHITE CORE nearest the seed; the physical tip touches the
     glass at the BOTTOM of that core, so CONTACT = its bottom edge (max-v),
     not the glow centroid (which sits too high, up the bloom).

All spatial params are fractions of frame dimensions -> resolution-relative
(1852x1422 fixtures, 1280x720 runtime, full-res) with no per-mode tuning.
"""
import cv2
import numpy as np

# --- appearance thresholds (intensity-based -> resolution-independent) ---
_CORE_V_MIN = 230      # white core: near-saturated brightness (max channel)
_CORE_S_MAX = 0.35     # white core: low saturation (near-neutral)
_SCORE_MAX = 255.0 * 255.0   # max of (B-R)*V; normalizes the seed peak to [0,1]
_PRESENT_FLOOR = 0.45  # min NORMALIZED seed peak for "LED present" (pen down)

# --- spatial params (fractions of frame dims -> resolution-relative) ---
_BLUR_FRAC = 0.005          # seed blur sigma (of width)
_WIN_FRAC = 0.06            # half-window (of width) around the seed to find the core
_MIN_CORE_AREA_FRAC = 2e-5  # drop specks
_BOTTOM_BAND_FRAC = 0.15    # bottom slice of the core used for the contact u


def _seed_score(bgr):
    """blueness (B-R) * brightness (V), blurred. Global peak = LED bloom."""
    bgr = bgr.astype(np.float32)
    b, r = bgr[..., 0], bgr[..., 2]
    v = bgr.max(2)
    sigma = max(1.0, _BLUR_FRAC * bgr.shape[1])
    return cv2.GaussianBlur(np.clip(b - r, 0, None) * v, (0, 0), sigma)


def _white_core(bgr):
    bgr = bgr.astype(np.float32)
    v = bgr.max(2)
    s = (v - bgr.min(2)) / np.maximum(v, 1.0)
    return ((v >= _CORE_V_MIN) & (s <= _CORE_S_MAX)).astype(np.uint8)


def detect(bgr, top_mask_frac=0.5):
    """Return (u, v, confidence) of the contact, or None if no LED.

    top_mask_frac: rows above this fraction of height are ignored (off-screen
      clutter: face, keyboard, room). Default 0.5; calibration can tighten it.
    """
    h, w = bgr.shape[:2]
    score = _seed_score(bgr)
    if top_mask_frac > 0:
        score[: int(top_mask_frac * h)] = 0

    peak = float(score.max())
    conf = min(1.0, (peak / _SCORE_MAX) / 0.85)
    if peak / _SCORE_MAX < _PRESENT_FLOOR:
        return None  # pen up / no LED (also rejects white glare: no blue -> low peak)

    _, _, _, (sx, sy) = cv2.minMaxLoc(score)

    # White core nearest the seed, within a local window.
    win = max(5, int(_WIN_FRAC * w))
    y0, y1 = max(0, sy - win), min(h, sy + win)
    x0, x1 = max(0, sx - win), min(w, sx + win)
    core = _white_core(bgr)[y0:y1, x0:x1]

    n, lab, stats, cent = cv2.connectedComponentsWithStats(core, 8)
    min_area = _MIN_CORE_AREA_FRAC * h * w
    seed_local = (sx - x0, sy - y0)
    chosen, best_d = None, 1e18
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        cx, cy = cent[i]
        d = (cx - seed_local[0]) ** 2 + (cy - seed_local[1]) ** 2
        if d < best_d:
            chosen, best_d = i, d

    if chosen is None:
        return float(sx), float(sy), conf * 0.5  # fallback: seed, low conf

    ys, xs = np.where(lab == chosen)
    ys, xs = ys + y0, xs + x0
    v_b = int(ys.max())
    band_h = max(1, int(_BOTTOM_BAND_FRAC * (ys.max() - ys.min() + 1)))
    u = float(xs[ys >= v_b - band_h].mean())
    return u, float(v_b), conf

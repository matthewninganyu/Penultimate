"""Candidate A position solver: two-bearing triangulation.

Pure geometry, no I/O. One horizontal pixel `u` per camera -> azimuth bearing
(1-D calibrated fit) -> cast a ray from the known top-corner -> intersect the
two rays on the 2D screen plane -> normalized ScreenPoint.

Coordinate frame (screen plane, centimetres):
  origin = top-left corner, +x to the right, +y downward into the screen.
  corner0 = (0, 0)  top-left camera
  corner1 = (W, 0)  top-right camera
A bearing is the azimuth of the corner->point direction: dir = (cos b, sin b),
so b = atan2(y - cy, x - cx) for a point (x, y).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Below this |sin(gamma)| the two bearings are effectively collinear; we floor
# the denominator instead of dividing by ~0 and report 1/|sin| as conditioning.
_SIN_FLOOR = 1e-6


@dataclass
class Detection:
    """One camera's blob for one frame. A uses `u` only (v picks the blob upstream)."""
    cam_id: int
    u: float
    v: float
    timestamp_us: int
    confidence: float


@dataclass
class ScreenPoint:
    x_norm: float          # [0, 1] left->right
    y_norm: float          # [0, 1] top->bottom
    pen_down: bool
    timestamp_us: int


def pixel_to_bearing(bearing_map, u: float) -> float:
    """Map a horizontal pixel `u` to an azimuth bearing (radians).

    `bearing_map` is a calibrated 1-D fit callable (np.poly1d / np.polynomial).
    It absorbs the wide-lens horizontal barrel warp + toe-in directly from taps.
    """
    return float(bearing_map(u))


def intersect(bearing0, corner0, bearing1, corner1):
    """Intersect two bearing rays on the screen plane.

    Returns (x, y, conditioning) where conditioning = 1/|sin gamma|, gamma =
    angle between the two bearings. Large conditioning => near-collinear (top
    strip) => the (x, y) is poorly determined. We floor the denominator rather
    than divide by ~0, so the point is always finite.
    """
    c0 = np.asarray(corner0, dtype=float)
    c1 = np.asarray(corner1, dtype=float)
    d0 = np.array([math.cos(bearing0), math.sin(bearing0)])
    d1 = np.array([math.cos(bearing1), math.sin(bearing1)])

    denom = d0[0] * d1[1] - d0[1] * d1[0]        # = sin(bearing1 - bearing0)
    sin_gamma = abs(denom)
    conditioning = 1.0 / max(sin_gamma, _SIN_FLOOR)
    if sin_gamma < _SIN_FLOOR:
        denom = math.copysign(_SIN_FLOOR, denom) if denom else _SIN_FLOOR

    r = c1 - c0
    t0 = (r[0] * d1[1] - r[1] * d1[0]) / denom
    p = c0 + t0 * d0
    return float(p[0]), float(p[1]), conditioning


def solve(det0, det1, screen_config):
    """Two detections -> ScreenPoint, or None on structural dropout.

    A needs BOTH cameras: one bearing cannot fix a 2D point, so a missing
    detection returns None (documented weakness vs B). A large inter-camera
    Δt also returns None (stale pairing would triangulate a smeared point).
    """
    # Structural dropout: A cannot triangulate from one bearing.
    if det0 is None or det1 is None:
        return None

    # Δt guard: skip pairs that are too far apart in time.
    dt_threshold_us = screen_config.get("dt_threshold_us")
    if dt_threshold_us is not None:
        if abs(det0.timestamp_us - det1.timestamp_us) > dt_threshold_us:
            return None

    b0 = pixel_to_bearing(screen_config["bearing_map0"], det0.u)
    b1 = pixel_to_bearing(screen_config["bearing_map1"], det1.u)

    x, y, _cond = intersect(b0, screen_config["corner0"],
                            b1, screen_config["corner1"])

    # ponytail: normalize only, no clip — valid taps land inside [0,1]; clamping
    # here would hide a real out-of-bounds solve. Clip at the consumer if needed.
    return ScreenPoint(
        x_norm=x / screen_config["width_cm"],
        y_norm=y / screen_config["height_cm"],
        pen_down=True,                              # both blobs present == contact
        timestamp_us=(det0.timestamp_us + det1.timestamp_us) // 2,
    )

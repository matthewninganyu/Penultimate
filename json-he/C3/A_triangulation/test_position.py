"""Synthetic, no-rig test for candidate A's two-bearing solver.

Forward-model a two-camera geometry, fit `pixel_to_bearing` from a training
subset of taps, then evaluate `solve()` on a held-out subset. Injects the two
error sources A must live with:
  (a) +/-0.3 px random noise on u (centroid jitter), and
  (b) a small v->u coupling that models the out-of-plane down-tilt — a
      *systematic* bias that grows toward the frame edges (A's known weakness).

Asserts interior / edge / top-strip budgets and, crucially, that the
near-collinear top strip degrades but stays BOUNDED. Also prints A's
tilt-bias ceiling: the max systematic (x,y) error from the v->u coupling,
so we know A's accuracy limit before any hardware exists.

Run:  python test_position.py      (assert-based, exits non-zero on failure)
Also importable by pytest (test_* functions).
"""

import math

import numpy as np

from position import Detection, intersect, pixel_to_bearing, solve

# ----------------------------------------------------------------------------
# Geometry (design: 30.4 x 19.7 cm active area, cameras at the two top corners,
# max baseline = screen width, toed-in). 720p sensor => width 1280 px.
# ----------------------------------------------------------------------------
W, H = 30.4, 19.7                      # screen active area, cm
CORNER0 = (0.0, 0.0)                   # top-left camera
CORNER1 = (W, 0.0)                     # top-right camera
SENSOR_W = 1280                        # 720p horizontal pixels
U_CENTER = SENSOR_W / 2.0

# Wide-lens forward model per camera: azimuth -> pixel u.
#   phi   = bearing - optical_axis           (off-axis angle)
#   u     = U_CENTER + F*(phi + K*phi^3)      (smooth monotonic barrel warp)
#           + TILT_GAIN * v_norm * phi        (out-of-plane v->u coupling bias)
# The barrel term is what the 1-D fit is meant to absorb; the coupling term is
# a function of v (range), which a u-only fit structurally cannot invert.
F = 560.0
K = 0.12                               # barrel nonlinearity (edges curve)
TILT_GAIN = 2.5                        # px of v->u coupling at full off-axis+depth
NOISE_PX = 0.3                         # +/- centroid jitter
POLY_DEG = 6                           # 1-D u->bearing fit degree

# Optical axes toed-in toward screen centre.
AXIS0 = math.atan2(H / 2, W / 2 - CORNER0[0])
AXIS1 = math.atan2(H / 2, W / 2 - CORNER1[0])


def _true_bearings(x, y):
    b0 = math.atan2(y - CORNER0[1], x - CORNER0[0])
    b1 = math.atan2(y - CORNER1[1], x - CORNER1[0])
    return b0, b1


def _ranges(x, y):
    r0 = math.hypot(x - CORNER0[0], y - CORNER0[1])
    r1 = math.hypot(x - CORNER1[0], y - CORNER1[1])
    return r0, r1


# Range span used to normalize the depth (v proxy) into [0, 1].
_R_MIN, _R_MAX = 0.5, math.hypot(W, H)


def forward_u(x, y, with_tilt=True):
    """True screen point -> (u0, u1). v proxy = normalized range (depth)."""
    b0, b1 = _true_bearings(x, y)
    r0, r1 = _ranges(x, y)
    out = []
    for b, axis, r in ((b0, AXIS0, r0), (b1, AXIS1, r1)):
        phi = b - axis
        u = U_CENTER + F * (phi + K * phi ** 3)
        if with_tilt:
            v_norm = (r - _R_MIN) / (_R_MAX - _R_MIN)     # deeper => larger bias
            u += TILT_GAIN * v_norm * phi                 # grows toward edges
        out.append(u)
    return out[0], out[1]


def _grid(nx=15, ny=11):
    """Screen taps, out to the edges but clear of the singular top corners."""
    xs = np.linspace(1.0, W - 1.0, nx)
    ys = np.linspace(1.0, H, ny)
    return [(float(x), float(y)) for y in ys for x in xs]


def _fit_bearing_maps(train_pts):
    """Calibrate 1-D u->bearing polys from taps (barrel + bias baked in).

    ponytail: fit on noiseless taps — a dense real tap grid averages the +/-0.3px
    jitter out; the systematic tilt bias is what survives into the residual.
    """
    u0, u1, b0, b1 = [], [], [], []
    for x, y in train_pts:
        cu0, cu1 = forward_u(x, y, with_tilt=True)
        tb0, tb1 = _true_bearings(x, y)
        u0.append(cu0); b0.append(tb0)
        u1.append(cu1); b1.append(tb1)
    map0 = np.poly1d(np.polyfit(u0, b0, POLY_DEG))
    map1 = np.poly1d(np.polyfit(u1, b1, POLY_DEG))
    return map0, map1


def _make_config(map0, map1, dt_threshold_us=8000):
    return {
        "width_cm": W, "height_cm": H,
        "corner0": CORNER0, "corner1": CORNER1,
        "bearing_map0": map0, "bearing_map1": map1,
        "dt_threshold_us": dt_threshold_us,
    }


def _region(x, y):
    if y < 3.0:
        return "top"                       # near-collinear strip (GDOP weak)
    if x < 3.0 or x > W - 3.0:
        return "edge"                      # worst barrel + tilt
    return "interior"


def _solve_point(x, y, cfg, rng, with_noise=True):
    u0, u1 = forward_u(x, y, with_tilt=True)
    if with_noise:
        u0 += rng.uniform(-NOISE_PX, NOISE_PX)
        u1 += rng.uniform(-NOISE_PX, NOISE_PX)
    d0 = Detection(0, u0, 0.0, 1000, 1.0)
    d1 = Detection(1, u1, 0.0, 1000, 1.0)
    sp = solve(d0, d1, cfg)
    assert sp is not None
    err_mm = math.hypot(sp.x_norm * W - x, sp.y_norm * H - y) * 10.0
    return err_mm


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------
def _evaluate():
    """Fit on a train split, evaluate solve() on a held-out split. Returns
    (per-region max errors with noise, tilt-bias ceiling in mm, worst top cond)."""
    all_pts = _grid()
    train = all_pts[0::2]
    test = all_pts[1::2]
    map0, map1 = _fit_bearing_maps(train)
    cfg = _make_config(map0, map1)

    rng = np.random.default_rng(1234)     # seeded for reproducibility

    max_err = {"interior": 0.0, "edge": 0.0, "top": 0.0}
    for x, y in test:
        e = _solve_point(x, y, cfg, rng, with_noise=True)
        reg = _region(x, y)
        max_err[reg] = max(max_err[reg], e)

    # Tilt-bias ceiling: same held-out taps, NO noise -> pure systematic error
    # attributable to the v->u coupling (plus tiny fit residual). This is A's
    # accuracy limit before hardware.
    ceiling_mm = 0.0
    ceiling_at = None
    for x, y in test:
        e = _solve_point(x, y, cfg, rng, with_noise=False)
        if e > ceiling_mm:
            ceiling_mm, ceiling_at = e, (x, y)

    # Worst conditioning in the top strip (sanity: near-collinear => large).
    worst_top_cond = 0.0
    for x, y in test:
        if _region(x, y) != "top":
            continue
        b0, b1 = _true_bearings(x, y)
        _, _, cond = intersect(b0, CORNER0, b1, CORNER1)
        worst_top_cond = max(worst_top_cond, cond)

    return max_err, ceiling_mm, ceiling_at, worst_top_cond


def test_accuracy_budget():
    max_err, ceiling_mm, ceiling_at, worst_top_cond = _evaluate()

    # Budgets (mm). Interior tightest; edge looser (barrel+tilt); top loosest
    # but must stay BOUNDED — the near-collinear strip degrades, never blows up.
    assert max_err["interior"] < 1.5, max_err
    assert max_err["edge"] < 4.0, max_err
    assert max_err["top"] < 8.0, max_err
    assert math.isfinite(max_err["top"])

    # Conditioning must be finite and worse in the top strip than the interior.
    assert math.isfinite(worst_top_cond)
    assert worst_top_cond > 1.0


def test_tilt_bias_ceiling_bounded():
    _, ceiling_mm, _, _ = _evaluate()
    # The systematic v->u ceiling is A's hard floor on accuracy. It must exist
    # (coupling is present) yet stay within the edge budget, else fall to B.
    assert 0.0 < ceiling_mm < 5.0, ceiling_mm


def test_single_camera_dropout():
    map0, map1 = _fit_bearing_maps(_grid())
    cfg = _make_config(map0, map1)
    d = Detection(0, 640.0, 0.0, 1000, 1.0)
    assert solve(d, None, cfg) is None
    assert solve(None, d, cfg) is None
    assert solve(None, None, cfg) is None


def test_dt_guard():
    map0, map1 = _fit_bearing_maps(_grid())
    cfg = _make_config(map0, map1, dt_threshold_us=8000)
    d0 = Detection(0, 640.0, 0.0, 0, 1.0)
    d1_ok = Detection(1, 700.0, 0.0, 5000, 1.0)      # within 8ms
    d1_bad = Detection(1, 700.0, 0.0, 20000, 1.0)    # 20ms apart -> skip
    assert solve(d0, d1_ok, cfg) is not None
    assert solve(d0, d1_bad, cfg) is None


def test_intersect_recovers_point():
    # Sanity: exact true bearings must intersect back at the point.
    x, y = 15.0, 10.0
    b0, b1 = _true_bearings(x, y)
    px, py, cond = intersect(b0, CORNER0, b1, CORNER1)
    assert math.isclose(px, x, abs_tol=1e-9)
    assert math.isclose(py, y, abs_tol=1e-9)
    assert math.isfinite(cond)


def main():
    max_err, ceiling_mm, ceiling_at, worst_top_cond = _evaluate()
    print("=" * 62)
    print("Candidate A — synthetic two-bearing solver (no rig)")
    print("=" * 62)
    print(f"  screen                : {W} x {H} cm, baseline {W} cm, 720p")
    print(f"  taps (train/test)     : {len(_grid()[0::2])}/{len(_grid()[1::2])}")
    print(f"  injected u-noise      : +/-{NOISE_PX} px   fit degree {POLY_DEG}")
    print("-" * 62)
    print("  max position error (with noise), by region:")
    print(f"    interior            : {max_err['interior']:.3f} mm")
    print(f"    edge                : {max_err['edge']:.3f} mm")
    print(f"    top strip (GDOP)    : {max_err['top']:.3f} mm  (bounded)")
    print(f"    worst top cond 1/sin: {worst_top_cond:.2f}")
    print("-" * 62)
    print(f"  >> A TILT-BIAS CEILING: {ceiling_mm:.3f} mm  (systematic v->u,")
    print(f"     worst at (x,y)={ceiling_at[0]:.1f},{ceiling_at[1]:.1f} cm — no-noise limit)")
    print("=" * 62)

    # Run assertions.
    test_accuracy_budget()
    test_tilt_bias_ceiling_bounded()
    test_single_camera_dropout()
    test_dt_guard()
    test_intersect_recovers_point()
    print("ALL ASSERTIONS PASSED")


if __name__ == "__main__":
    main()

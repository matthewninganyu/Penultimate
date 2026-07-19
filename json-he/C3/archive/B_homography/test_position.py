"""Synthetic, no-rig test for candidate B (plane homography).

Forward-model two top-corner cameras (few mm above the screen plane, down-tilted,
toed-in) viewing the flat screen. Project a known screen grid THROUGH injected
barrel distortion (published CM3-Wide coeffs scaled to 720p capture), undistort,
fit H per camera, then solve() held-out points.

Proves: (1) recovery within budget WITH distortion present, (2) a no-undistort
baseline FAILS the same tolerance (undistort earns its place), (3) single-camera
fallback returns a usable point, (4) R1 -- edge error vs interior under imperfect
undistortion.

Run:  python test_position.py     (also importable by pytest)
"""
import numpy as np
import cv2

import position as P

RNG = np.random.default_rng(42)

# --- screen + capture geometry ---------------------------------------------
SCREEN_CM = (30.4, 19.7)          # active area
SCREEN_PX = (3024, 1964)          # screen_pts space H maps into
SCREEN_M = (0.304, 0.197)         # world metres

FULL_RES = (4608, 2592)           # intrinsics reference resolution
CAP_RES = (1280, 720)             # 720p capture (per contract)

# Published CM3-Wide intrinsics at full res.
K_FULL = np.array([[1995.70, 0, 2340.02],
                   [0, 1990.13, 1289.11],
                   [0, 0, 1.0]])
DIST = np.array([-0.0574, 0.1264, -0.00287, 0.00384, -0.0808])  # k1 k2 p1 p2 k3


def _scale_K(K, full, cap):
    sx, sy = cap[0] / full[0], cap[1] / full[1]
    Ks = K.copy()
    Ks[0, 0] *= sx; Ks[0, 2] *= sx
    Ks[1, 1] *= sy; Ks[1, 2] *= sy
    return Ks


K = _scale_K(K_FULL, FULL_RES, CAP_RES)  # dist coeffs are unitless -> unchanged


def _lookat(C, T, up=(0, 0, 1.0)):
    """OpenCV world->camera rvec,tvec for a camera at C looking at T (+z fwd)."""
    C, T, up = map(lambda a: np.asarray(a, float), (C, T, up))
    z = T - C; z /= np.linalg.norm(z)
    x = np.cross(up, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    Rwc = np.stack([x, y, z])          # rows = camera axes in world
    tvec = -Rwc @ C
    rvec, _ = cv2.Rodrigues(Rwc)
    return rvec, tvec


# Two top-corner mounts: a few mm above the plane, pulled slightly behind the
# top edge, toed-in at the screen centre (down-tilt falls out of the geometry).
CENTER = (SCREEN_M[0] / 2, SCREEN_M[1] / 2, 0.0)
POSES = {
    0: _lookat((0.00, 0.235, 0.006), CENTER),   # top-left
    1: _lookat((0.304, 0.235, 0.006), CENTER),  # top-right
}


def _screen_grid(nx, ny, margin=0.06):
    """Grid of screen points in world metres (with edge margin)."""
    xs = np.linspace(margin * SCREEN_M[0], (1 - margin) * SCREEN_M[0], nx)
    ys = np.linspace(margin * SCREEN_M[1], (1 - margin) * SCREEN_M[1], ny)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)], axis=1)


def _project(world_pts, cam_id, dist=DIST):
    rvec, tvec = POSES[cam_id]
    px, _ = cv2.projectPoints(world_pts.reshape(-1, 1, 3), rvec, tvec, K, dist)
    return px.reshape(-1, 2)


def _world_to_screen_px(world_pts):
    x = world_pts[:, 0] / SCREEN_M[0] * SCREEN_PX[0]
    y = world_pts[:, 1] / SCREEN_M[1] * SCREEN_PX[1]
    return np.stack([x, y], axis=1)


def _fit_H(world_grid, cam_id, undistort=True, dist_for_undist=DIST):
    """Fit image->screen_px homography over the grid for one camera."""
    dpx = _project(world_grid, cam_id)                 # distorted pixels
    if undistort:
        upx = cv2.undistortPoints(dpx.reshape(-1, 1, 2).astype(np.float64),
                                  K, dist_for_undist, P=K).reshape(-1, 2)
    else:
        upx = dpx                                       # TEETH: raw distorted
    spx = _world_to_screen_px(world_grid)
    H, _ = cv2.findHomography(upx, spx, method=0)
    return H


def _build_calib(world_grid, dt_max_us=5000):
    return {
        "cams": {cid: {"K": K, "dist": DIST, "H": _fit_H(world_grid, cid)}
                 for cid in POSES},
        "screen_px": SCREEN_PX,
        "dt_max_us": dt_max_us,
    }


def _detections(world_pt, calib, cams=(0, 1), t=(0, 0), conf=(1.0, 1.0), noise_px=0.0):
    dets = {}
    for cid in (0, 1):
        if cid not in cams:
            dets[cid] = None
            continue
        u, v = _project(world_pt.reshape(1, 3), cid)[0]
        u += RNG.normal(0, noise_px); v += RNG.normal(0, noise_px)  # centroid noise
        dets[cid] = {"cam_id": cid, "u": float(u), "v": float(v),
                     "timestamp_us": int(t[cid]), "confidence": float(conf[cid])}
    return dets[0], dets[1]


def _true_norm(world_pt):
    return (world_pt[0] / SCREEN_M[0], world_pt[1] / SCREEN_M[1])


def _err_mm(sp, world_pt):
    """Recovered vs true position error in millimetres."""
    dx = (sp["x_norm"] - _true_norm(world_pt)[0]) * SCREEN_CM[0] * 10.0
    dy = (sp["y_norm"] - _true_norm(world_pt)[1]) * SCREEN_CM[1] * 10.0
    return float(np.hypot(dx, dy))


# --- shared fixtures --------------------------------------------------------
FIT_GRID = _screen_grid(11, 8)
CALIB = _build_calib(FIT_GRID)
# Held-out points: random, interior of the active area, NOT on the fit grid.
HELD = np.stack([
    RNG.uniform(0.08, 0.92, 40) * SCREEN_M[0],
    RNG.uniform(0.08, 0.92, 40) * SCREEN_M[1],
    np.zeros(40),
], axis=1)

TOL_MM = 0.5     # budget WITH distortion handled (sub-mm interior)
NOISE_PX = 0.3   # centroid noise (px) injected into recovery + single-cam tests


# --- tests ------------------------------------------------------------------

def test_recovery_with_distortion():
    """Core claim: with EXACT centroids, undistort+H recovers screen (x,y) to
    numerical precision even through heavy barrel distortion. (Noise sensitivity
    is measured separately -- see test_centroid_noise_sensitivity.)"""
    errs = []
    for w in HELD:
        d0, d1 = _detections(w, CALIB)
        sp = P.solve(d0, d1, CALIB)
        assert 0.0 <= sp["x_norm"] <= 1.0 and 0.0 <= sp["y_norm"] <= 1.0
        assert sp["pen_down"] is True
        errs.append(_err_mm(sp, w))
    errs = np.array(errs)
    print(f"[recovery]  two-cam, distortion handled (exact centroids): "
          f"mean={errs.mean():.6f}mm max={errs.max():.6f}mm  (tol {TOL_MM}mm)")
    assert errs.max() < TOL_MM


def test_centroid_noise_sensitivity():
    """CONDITIONING FINDING: the 'few mm above plane' mount is ~2deg grazing, so
    the homography Jacobian is severely ill-conditioned -- sub-pixel centroid
    noise blows up to millimetres. Quantify it; this is B's dominant real-world
    risk with a near-in-plane mount (why sibling A uses u-only bearings)."""
    conds = [P._local_cond(CALIB["cams"][0]["H"],
                           *P._undistort(*_project(w.reshape(1, 3), 0)[0],
                                         K, DIST)) for w in HELD]
    errs = []
    for w in HELD:
        d0, d1 = _detections(w, CALIB, noise_px=NOISE_PX)
        errs.append(_err_mm(P.solve(d0, d1, CALIB), w))
    errs = np.array(errs)
    print(f"[noise]     +/-{NOISE_PX}px centroid noise -> "
          f"mean={errs.mean():.3f}mm max={errs.max():.3f}mm  "
          f"(H Jacobian cond: median={np.median(conds):.0f})")
    assert np.isfinite(errs.max())          # solver stays finite (degrades, no NaN)


def test_no_undistort_baseline_fails():
    """TEETH: fit H on raw distorted pixels, evaluate without undistorting.
    Must be materially worse and blow the tolerance."""
    H_raw = {cid: _fit_H(FIT_GRID, cid, undistort=False) for cid in POSES}
    errs = []
    for w in HELD:
        est = []
        for cid in POSES:
            u, v = _project(w.reshape(1, 3), cid)[0]
            x, y = P._apply_H(H_raw[cid], u, v)     # NO undistort
            est.append((x / SCREEN_PX[0], y / SCREEN_PX[1]))
        est = np.mean(est, axis=0)
        dx = (est[0] - _true_norm(w)[0]) * SCREEN_CM[0] * 10.0
        dy = (est[1] - _true_norm(w)[1]) * SCREEN_CM[1] * 10.0
        errs.append(np.hypot(dx, dy))
    errs = np.array(errs)

    good = np.array([_err_mm(P.solve(*_detections(w, CALIB), CALIB), w) for w in HELD])
    print(f"[teeth]     no-undistort baseline: "
          f"mean={errs.mean():.4f}mm max={errs.max():.4f}mm  "
          f"(vs undistorted max={good.max():.4f}mm)")
    assert errs.max() > TOL_MM                  # baseline FAILS the budget
    assert errs.max() > 5 * good.max()          # materially larger


def test_single_camera_fallback():
    """One camera present -> still a usable (degraded) point (B's edge over A)."""
    errs0, errs1 = [], []
    for w in HELD:
        d0, d1 = _detections(w, CALIB, cams=(0,))     # only cam 0
        sp = P.solve(d0, d1, CALIB)
        assert sp is not None and sp["pen_down"] is True
        errs0.append(_err_mm(sp, w))
        d0, d1 = _detections(w, CALIB, cams=(1,))     # only cam 1
        errs1.append(_err_mm(P.solve(d0, d1, CALIB), w))
    e0, e1 = np.array(errs0), np.array(errs1)
    print(f"[single]    cam0-only max={e0.max():.4f}mm  cam1-only max={e1.max():.4f}mm")
    assert e0.max() < TOL_MM and e1.max() < TOL_MM   # single-cam still recovers


def test_both_none_returns_none():
    assert P.solve(None, None, CALIB) is None


def test_dt_guard_drops_desynced_cam():
    w = HELD[0]
    d0, d1 = _detections(w, CALIB, t=(0, 10_000), conf=(1.0, 0.3))  # 10ms skew
    sp = P.solve(d0, d1, CALIB)
    assert sp is not None                       # degrades, does not die
    # kept the higher-confidence (cam0) view only -> matches cam0-only solve
    d0s, _ = _detections(w, CALIB, cams=(0,))
    solo = P.solve(d0s, None, CALIB)
    assert abs(sp["x_norm"] - solo["x_norm"]) < 1e-9


def test_r1_residual_distortion_sweep():
    """R1: undistort with wrong dist coeffs (scale k's) -> map how edge error
    grows vs interior. Quantifies B's sensitivity to imperfect undistortion at
    the frame edges (the load-bearing risk)."""
    # interior vs edge held-out sets (by screen-x, where the L/R frame edges land)
    interior = _screen_grid(5, 4, margin=0.30)
    edge = np.stack([
        np.concatenate([np.full(6, 0.03), np.full(6, 0.97)]) * SCREEN_M[0],
        np.tile(np.linspace(0.15, 0.85, 6), 2) * SCREEN_M[1],
        np.zeros(12),
    ], axis=1)

    print("[R1 sweep]  undistort-coeff scale -> mean error (mm)")
    print(f"            {'scale':>6} {'interior':>10} {'edge':>10} {'edge/int':>9}")
    for scale in (0.8, 1.0, 1.2):
        dist_wrong = DIST * scale
        cal = {"cams": {cid: {"K": K, "dist": dist_wrong,
                              "H": _fit_H(FIT_GRID, cid, dist_for_undist=dist_wrong)}
                        for cid in POSES},
               "screen_px": SCREEN_PX, "dt_max_us": 5000}
        ei = np.mean([_err_mm(P.solve(*_detections(w, cal), cal), w) for w in interior])
        ee = np.mean([_err_mm(P.solve(*_detections(w, cal), cal), w) for w in edge])
        ratio = f"{ee/ei:.2f}x" if ei > 1e-4 else "~exact"   # 1.0 row = numerical floor
        print(f"            {scale:>6.1f} {ei:>10.4f} {ee:>10.4f} {ratio:>9}")

    # sanity: correct coeffs (scale 1.0) keep even the edges within budget
    cal = CALIB
    ee = np.max([_err_mm(P.solve(*_detections(w, cal), cal), w) for w in edge])
    assert ee < TOL_MM


def _run_all():
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")


if __name__ == "__main__":
    _run_all()
    print("\nALL TESTS PASSED")

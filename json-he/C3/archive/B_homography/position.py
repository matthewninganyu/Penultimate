"""Candidate B position solver: 4-point plane homography (image -> screen).

Runtime per camera = undistort(u,v) then one 3x3 homography multiply. A plane
homography recovers full (x,y) from a SINGLE view, so B degrades to one camera
instead of dropping out (its edge over A's two-bearing intersection).

Data contract (from ../PROJECT.md):
  Detection  = {cam_id, u, v, timestamp_us, confidence}   # B uses full u,v
  ScreenPoint= {x_norm in [0,1], y_norm in [0,1], pen_down: bool, timestamp_us}

calib layout (produced by calibrate.py; synthetic in test_position.py):
  calib = {
    "cams": {cam_id: {"K": 3x3, "dist": (k1,k2,p1,p2,k3), "H": 3x3 image->screen_px}},
    "screen_px": (W, H),      # screen_pts space H maps into; used to normalize
    "dt_max_us": int,         # inter-camera sync guard
  }
"""
import numpy as np
import cv2


# --- primitives -------------------------------------------------------------

def _undistort(u, v, K, dist):
    """Barrel-correct one pixel back to ideal pinhole pixel space (P=K).

    H is fit on undistorted pixels, so runtime MUST undistort first -- a
    homography is linear projective and cannot model lens distortion.
    """
    pts = np.array([[[float(u), float(v)]]], dtype=np.float64)
    und = cv2.undistortPoints(pts, np.asarray(K, float), np.asarray(dist, float),
                              P=np.asarray(K, float))
    return float(und[0, 0, 0]), float(und[0, 0, 1])


def _apply_H(H, uu, vv):
    w = np.asarray(H, float) @ np.array([uu, vv, 1.0])
    return w[0] / w[2], w[1] / w[2]


def apply_homography(H, u, v, K, dist):
    """Undistort pixel (K,dist) then map image->screen via H. Returns (x,y).

    # ponytail: signature carries K,dist beyond the contract's (H,u,v) -- the
    # undistort step is mandatory and needs the intrinsics; no globals.
    """
    return _apply_H(H, *_undistort(u, v, K, dist))


def _local_cond(H, uu, vv):
    """Condition number of the homography Jacobian at an (undistorted) point.

    Foreshortened / grazing views compress one axis -> anisotropic Jacobian ->
    large cond. Used to downweight the more grazing camera in combine().
    """
    H = np.asarray(H, float)
    D = H[2, 0] * uu + H[2, 1] * vv + H[2, 2]
    Nx = H[0, 0] * uu + H[0, 1] * vv + H[0, 2]
    Ny = H[1, 0] * uu + H[1, 1] * vv + H[1, 2]
    J = np.array([
        [(H[0, 0] * D - Nx * H[2, 0]) / D**2, (H[0, 1] * D - Nx * H[2, 1]) / D**2],
        [(H[1, 0] * D - Ny * H[2, 0]) / D**2, (H[1, 1] * D - Ny * H[2, 1]) / D**2],
    ])
    return float(np.linalg.cond(J))


def combine(est0, conf0, est1, conf1):
    """Confidence/conditioning-weighted average of per-camera (x,y) estimates.

    Weights already fold in local conditioning (see solve). None est = absent.
    """
    es, ws = [], []
    for e, w in ((est0, conf0), (est1, conf1)):
        if e is None:
            continue
        es.append(np.asarray(e, float))
        ws.append(max(float(w), 0.0))
    if not es:
        return None
    ws = np.array(ws)
    if ws.sum() <= 0:
        ws = np.ones_like(ws)
    return tuple((np.stack(es) * ws[:, None]).sum(0) / ws.sum())


# --- solver -----------------------------------------------------------------

def solve(det0, det1, calib):
    """Undistort+H each present camera, combine, normalize. Returns ScreenPoint|None.

    Single camera present -> still returns a (degraded) point (B's advantage).
    Delta-t guard: if both present but |t0-t1| too large, drop the lower-confidence
    camera rather than combine desynced views.
    """
    cams = calib["cams"]
    W, Hh = calib["screen_px"]
    dt_max = calib.get("dt_max_us", float("inf"))

    present = [d for d in (det0, det1) if d is not None]
    if not present:
        return None

    # Delta-t guard: desynced pair -> keep only the more confident camera.
    if len(present) == 2 and abs(present[0]["timestamp_us"] - present[1]["timestamp_us"]) > dt_max:
        present = [max(present, key=lambda d: d["confidence"])]

    ests, weights, ts = [], [], []
    for d in present:
        c = cams[d["cam_id"]]
        uu, vv = _undistort(d["u"], d["v"], c["K"], c["dist"])
        ests.append(_apply_H(c["H"], uu, vv))
        # downweight grazing/foreshortened view: conf / condition-number
        weights.append(d["confidence"] / _local_cond(c["H"], uu, vv))
        ts.append(d["timestamp_us"])

    if len(ests) == 1:
        xy = ests[0]
    else:
        xy = combine(ests[0], weights[0], ests[1], weights[1])

    return {
        "x_norm": float(np.clip(xy[0] / W, 0.0, 1.0)),
        "y_norm": float(np.clip(xy[1] / Hh, 0.0, 1.0)),
        "pen_down": True,  # any detection present == LED lit == tip contact
        "timestamp_us": int(np.mean(ts)),
    }

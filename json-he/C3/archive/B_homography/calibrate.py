"""Phase-2 calibration FIT for candidate B (plane homography).

Produces the artifacts position.py's solve() consumes:
  intrinsics_cam{N}.npz  keys: K (3x3), dist (5 pinhole | 4 fisheye), model (str)
  homography_cam{N}.npy  H, image->screen_px (no runtime inversion)
  screen_config.yaml     screen_px space + dt guard

Pipeline per camera (see ../PROJECT.md):
  1. Intrinsics: checkerboard imgs -> BOTH cv2.calibrateCamera (pinhole) and
     cv2.fisheye.calibrate; keep the lower RMS-reprojection model (report both).
  2. Homography: taps_cam{N}.csv -> undistort (u,v) with the chosen intrinsics ->
     cv2.findHomography(undistorted_px -> screen_px).

# ponytail: model selector = RMS reprojection (both calibrators return it); the
# L/R-edge straight-line residual in PROJECT is the on-rig refinement, not needed
# to pick a model on clean data. Upgrade path: swap the metric if edges misbehave.
"""
import argparse
import csv
import glob
import os

import numpy as np
import cv2

# Canonical screen_pts space (matches ../PROJECT.md + position.py normalization).
SCREEN_PX = (3024, 1964)          # 30.4 x 19.7 cm active area
DT_MAX_US = 5000                  # inter-camera sync guard
CHECKER = (9, 6)                  # inner corners (cols, rows) -- real board must match


# --- intrinsics -------------------------------------------------------------

def calibrate_intrinsics(img_dir, checker=CHECKER):
    """Pinhole + fisheye on checkerboard imgs; return dict for the lower-RMS model.

    Returns None if the dir is missing / has too few detectable boards.
    """
    paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    if not paths:
        return None

    objp = np.zeros((checker[0] * checker[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checker[0], 0:checker[1]].T.reshape(-1, 2)
    objpoints, imgpoints, size = [], [], None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        size = (img.shape[1], img.shape[0])
        ok, corners = cv2.findChessboardCorners(img, checker, None)
        if not ok:
            continue
        corners = cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), crit)
        objpoints.append(objp)
        imgpoints.append(corners)
    if len(objpoints) < 3:
        return None  # ponytail: <3 views can't constrain a lens model; bail like "no imgs"

    rms_p, K_p, d_p, *_ = cv2.calibrateCamera(objpoints, imgpoints, size, None, None)

    rms_f, K_f, d_f = np.inf, None, None
    try:  # fisheye is finicky; a throw == "this model doesn't fit" -> pinhole wins
        objf = [o.reshape(-1, 1, 3).astype(np.float64) for o in objpoints]
        imgf = [c.reshape(-1, 1, 2).astype(np.float64) for c in imgpoints]
        K_f, d_f = np.zeros((3, 3)), np.zeros((4, 1))
        rms_f, K_f, d_f, *_ = cv2.fisheye.calibrate(
            objf, imgf, size, K_f, d_f,
            flags=cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW)
    except cv2.error:
        pass

    report = {"rms_pinhole": float(rms_p), "rms_fisheye": float(rms_f),
              "n_views": len(objpoints)}
    if rms_p <= rms_f:
        return {"K": K_p, "dist": d_p.ravel(), "model": "pinhole", **report}
    return {"K": K_f, "dist": d_f.ravel(), "model": "fisheye", **report}


def undistort_px(uv, K, dist, model):
    """Barrel-correct pixels back to ideal pinhole pixel space (P=K)."""
    pts = np.asarray(uv, np.float64).reshape(-1, 1, 2)
    if model == "fisheye":
        und = cv2.fisheye.undistortPoints(pts, np.asarray(K, float),
                                          np.asarray(dist, float).reshape(-1, 1),
                                          P=np.asarray(K, float))
    else:
        und = cv2.undistortPoints(pts, np.asarray(K, float),
                                  np.asarray(dist, float), P=np.asarray(K, float))
    return und.reshape(-1, 2)


# --- homography -------------------------------------------------------------

def _read_taps(csv_path):
    """Rows of taps_cam{N}.csv -> (Nx2 undistort-input px, Nx2 screen_px)."""
    uv, screen = [], []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            if not r.get("u") or not r.get("v"):
                continue
            uv.append([float(r["u"]), float(r["v"])])
            screen.append([float(r["x_norm"]) * SCREEN_PX[0],
                           float(r["y_norm"]) * SCREEN_PX[1]])
    return np.array(uv, float), np.array(screen, float)


def fit_homography(csv_path, K, dist, model):
    """Undistort tap pixels, fit image->screen_px H over the whole grid + RANSAC."""
    uv, screen = _read_taps(csv_path)
    upx = undistort_px(uv, K, dist, model)
    # ponytail: RANSAC threshold ~1% of screen width (~30px) rejects misclicks but
    # keeps sub-px centroid noise. Tighten if the grid gets denser/cleaner.
    H, _ = cv2.findHomography(upx, screen, cv2.RANSAC, 0.01 * SCREEN_PX[0])
    return H


# --- artifacts --------------------------------------------------------------

def _write_yaml(path, d):
    try:
        import yaml
        with open(path, "w") as f:
            yaml.safe_dump(d, f)
        return
    except ImportError:
        pass
    with open(path, "w") as f:  # ponytail: flat scalars/lists only -> tiny hand-roll
        for k, v in d.items():
            if isinstance(v, (list, tuple)):
                f.write(f"{k}: [{', '.join(map(str, v))}]\n")
            else:
                f.write(f"{k}: {v}\n")


def load_calib(out_dir, cams=(0, 1), dt_max_us=DT_MAX_US):
    """Load written artifacts into the calib dict position.py's solve() expects."""
    cam_d = {}
    for c in cams:
        npz = os.path.join(out_dir, f"intrinsics_cam{c}.npz")
        npy = os.path.join(out_dir, f"homography_cam{c}.npy")
        if not (os.path.exists(npz) and os.path.exists(npy)):
            continue
        d = np.load(npz)
        cam_d[c] = {"K": d["K"], "dist": d["dist"], "H": np.load(npy),
                    "model": str(d["model"])}
    return {"cams": cam_d, "screen_px": SCREEN_PX, "dt_max_us": dt_max_us}


# --- orchestration ----------------------------------------------------------

def run(taps_dir, calib_imgs_dir, out_dir, cams=(0, 1), checker=CHECKER):
    """Calibrate each camera end-to-end; write artifacts. Returns a report dict."""
    os.makedirs(out_dir, exist_ok=True)
    report = {}
    for c in cams:
        intr = calibrate_intrinsics(os.path.join(calib_imgs_dir, f"cam{c}"), checker)
        if intr is None:
            print(f"[cam{c}] WARN no/too-few checkerboard images -> skipping "
                  f"intrinsics + homography")
            report[c] = None
            continue
        print(f"[cam{c}] intrinsics: pinhole RMS={intr['rms_pinhole']:.4f}px  "
              f"fisheye RMS={intr['rms_fisheye']:.4f}px  -> chose {intr['model']} "
              f"({intr['n_views']} views)")
        np.savez(os.path.join(out_dir, f"intrinsics_cam{c}.npz"),
                 K=intr["K"], dist=intr["dist"], model=intr["model"])

        taps = os.path.join(taps_dir, f"taps_cam{c}.csv")
        if not os.path.exists(taps):
            print(f"[cam{c}] WARN {taps} missing -> intrinsics only")
            report[c] = intr
            continue
        H = fit_homography(taps, intr["K"], intr["dist"], intr["model"])
        np.save(os.path.join(out_dir, f"homography_cam{c}.npy"), H)
        print(f"[cam{c}] homography written (image->screen_px)")
        report[c] = intr

    _write_yaml(os.path.join(out_dir, "screen_config.yaml"),
                {"screen_px": list(SCREEN_PX), "dt_max_us": DT_MAX_US,
                 "screen_cm": [30.4, 19.7]})
    return report


def main():
    ap = argparse.ArgumentParser(description="Candidate B calibration fit")
    ap.add_argument("--taps-dir", default="..", help="dir holding taps_cam{N}.csv (default C3/)")
    ap.add_argument("--calib-imgs", default="../calib_imgs", help="dir holding cam{N}/*.png")
    ap.add_argument("--out-dir", default=".", help="where to write artifacts")
    a = ap.parse_args()
    run(a.taps_dir, a.calib_imgs, a.out_dir)


if __name__ == "__main__":
    main()

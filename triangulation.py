from __future__ import annotations

import cv2
import numpy as np

from models import LedCandidate, ScreenCalibration


def triangulate_candidates(
    candidate_0: LedCandidate,
    candidate_1: LedCandidate,
    calibration: ScreenCalibration,
) -> np.ndarray | None:
    point_0 = undistort_point(candidate_0.image_point(), calibration.K0, calibration.D0)
    point_1 = undistort_point(candidate_1.image_point(), calibration.K1, calibration.D1)
    point_4d = cv2.triangulatePoints(calibration.P0, calibration.P1, point_0, point_1)
    scale = float(point_4d[3, 0])
    if abs(scale) < 1e-9:
        return None
    point_3d = (point_4d[:3, 0] / scale).astype(np.float64)
    return point_3d


def undistort_point(point: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
    reshaped = np.asarray(point, dtype=np.float64).reshape(1, 1, 2)
    undistorted = cv2.undistortPoints(reshaped, K, D, P=K)
    return undistorted.reshape(2, 1)


def project_point(point_3d: np.ndarray, K: np.ndarray, D: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(R)
    projected, _ = cv2.projectPoints(
        np.asarray(point_3d, dtype=np.float64).reshape(1, 3),
        rvec,
        t.reshape(3, 1),
        K,
        D,
    )
    return projected.reshape(2)


def reprojection_error(
    point_3d: np.ndarray,
    candidate_0: LedCandidate,
    candidate_1: LedCandidate,
    calibration: ScreenCalibration,
) -> float:
    projected_0 = project_point(point_3d, calibration.K0, calibration.D0, calibration.R0, calibration.t0)
    projected_1 = project_point(point_3d, calibration.K1, calibration.D1, calibration.R1, calibration.t1)
    error_0 = float(np.linalg.norm(projected_0 - candidate_0.image_point()))
    error_1 = float(np.linalg.norm(projected_1 - candidate_1.image_point()))
    return (error_0 + error_1) / 2.0


def point_in_front_of_camera(point_3d: np.ndarray, R: np.ndarray, t: np.ndarray) -> bool:
    camera_space = R @ point_3d.reshape(3, 1) + t.reshape(3, 1)
    return float(camera_space[2, 0]) > 0.0


def point_in_front_of_both_cameras(point_3d: np.ndarray, calibration: ScreenCalibration) -> bool:
    return point_in_front_of_camera(point_3d, calibration.R0, calibration.t0) and point_in_front_of_camera(
        point_3d,
        calibration.R1,
        calibration.t1,
    )


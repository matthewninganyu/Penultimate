from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config import DEFAULT_HEIGHT, DEFAULT_WIDTH


DEFAULT_HOMOGRAPHY_CALIBRATION = Path("calibration") / "screen_homography.npz"


@dataclass(frozen=True)
class HomographyCalibration:
    H0: np.ndarray
    H1: np.ndarray
    screen_width_px: int
    screen_height_px: int
    camera_0_points: np.ndarray
    camera_1_points: np.ndarray
    screen_points: np.ndarray
    image_width: int
    image_height: int
    timestamp: float


def build_homography(camera_points: np.ndarray, screen_points: np.ndarray) -> np.ndarray:
    camera_points = np.asarray(camera_points, dtype=np.float32)
    screen_points = np.asarray(screen_points, dtype=np.float32)
    if camera_points.shape != (4, 2):
        raise ValueError(f"camera_points must have shape (4, 2), got {camera_points.shape}.")
    if screen_points.shape != (4, 2):
        raise ValueError(f"screen_points must have shape (4, 2), got {screen_points.shape}.")
    return cv2.getPerspectiveTransform(camera_points, screen_points)


def screen_corner_points(screen_width_px: int, screen_height_px: int) -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0],
            [float(screen_width_px - 1), 0.0],
            [float(screen_width_px - 1), float(screen_height_px - 1)],
            [0.0, float(screen_height_px - 1)],
        ],
        dtype=np.float32,
    )


def _inverse_bilinear(
    p: np.ndarray,
    q0: np.ndarray,
    q1: np.ndarray,
    q2: np.ndarray,
    q3: np.ndarray,
) -> tuple[float, float] | None:
    """
    Solve (u, v) s.t. p = (1-u)(1-v)*q0 + u(1-v)*q1 + uv*q2 + (1-u)v*q3.
    Corners: q0=TL, q1=TR, q2=BR, q3=BL.
    Returns (u, v) where (0,0)=TL, (1,0)=TR, (1,1)=BR, (0,1)=BL.
    """
    e = q1 - q0
    f = q3 - q0
    g = q0 - q1 + q2 - q3
    h = p - q0

    ka = g[0] * f[1] - g[1] * f[0]
    kb = e[0] * f[1] - e[1] * f[0] + h[0] * g[1] - h[1] * g[0]
    kc = h[0] * e[1] - h[1] * e[0]

    if abs(ka) < 1e-10:
        if abs(kb) < 1e-10:
            return None
        v = -kc / kb
    else:
        disc = kb * kb - 4.0 * ka * kc
        sq = math.sqrt(max(disc, 0.0))
        v1 = (-kb + sq) / (2.0 * ka)
        v2 = (-kb - sq) / (2.0 * ka)
        v = v1 if abs(v1 - 0.5) < abs(v2 - 0.5) else v2

    dx = e[0] + v * g[0]
    dy = e[1] + v * g[1]
    if abs(dx) >= abs(dy):
        if abs(dx) < 1e-10:
            return None
        u = (h[0] - v * f[0]) / dx
    else:
        if abs(dy) < 1e-10:
            return None
        u = (h[1] - v * f[1]) / dy

    return float(u), float(v)


def map_camera_point(
    point: np.ndarray,
    camera_corners: np.ndarray,
    screen_width_px: int,
    screen_height_px: int,
) -> np.ndarray:
    """Map a camera pixel to screen pixel coords via inverse bilinear interpolation."""
    corners = camera_corners.astype(np.float64)
    p = np.asarray(point, dtype=np.float64).reshape(2)
    result = _inverse_bilinear(p, corners[0], corners[1], corners[2], corners[3])
    if result is None:
        return np.array([0.5 * (screen_width_px - 1), 0.5 * (screen_height_px - 1)], dtype=np.float64)
    u, v = result
    return np.array([u * (screen_width_px - 1), v * (screen_height_px - 1)], dtype=np.float64)


def combine_screen_points(
    screen_0: np.ndarray | None,
    screen_1: np.ndarray | None,
    confidence_0: float = 1.0,
    confidence_1: float = 1.0,
) -> np.ndarray | None:
    if screen_0 is None and screen_1 is None:
        return None
    if screen_0 is None:
        return np.asarray(screen_1, dtype=np.float64)
    if screen_1 is None:
        return np.asarray(screen_0, dtype=np.float64)

    total = confidence_0 + confidence_1
    if total <= 0:
        return (np.asarray(screen_0, dtype=np.float64) + np.asarray(screen_1, dtype=np.float64)) / 2.0
    return (
        (np.asarray(screen_0, dtype=np.float64) * confidence_0)
        + (np.asarray(screen_1, dtype=np.float64) * confidence_1)
    ) / total


def save_homography_calibration(
    path: Path,
    H0: np.ndarray,
    H1: np.ndarray,
    screen_width_px: int,
    screen_height_px: int,
    camera_0_points: np.ndarray,
    camera_1_points: np.ndarray,
    screen_points: np.ndarray,
    image_width: int,
    image_height: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        H0=np.asarray(H0, dtype=np.float64),
        H1=np.asarray(H1, dtype=np.float64),
        screen_width_px=screen_width_px,
        screen_height_px=screen_height_px,
        camera_0_points=np.asarray(camera_0_points, dtype=np.float64),
        camera_1_points=np.asarray(camera_1_points, dtype=np.float64),
        screen_points=np.asarray(screen_points, dtype=np.float64),
        image_width=image_width,
        image_height=image_height,
        timestamp=time.time(),
    )


def load_homography_calibration(path: Path = DEFAULT_HOMOGRAPHY_CALIBRATION) -> HomographyCalibration:
    if not path.exists():
        raise FileNotFoundError(f"Missing homography calibration: {path}")
    data = np.load(path, allow_pickle=True)
    required = (
        "H0",
        "H1",
        "screen_width_px",
        "screen_height_px",
        "camera_0_points",
        "camera_1_points",
        "screen_points",
        "image_width",
        "image_height",
        "timestamp",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Homography calibration {path} is missing keys: {', '.join(missing)}")
    return HomographyCalibration(
        H0=np.asarray(data["H0"], dtype=np.float64),
        H1=np.asarray(data["H1"], dtype=np.float64),
        screen_width_px=int(data["screen_width_px"]),
        screen_height_px=int(data["screen_height_px"]),
        camera_0_points=np.asarray(data["camera_0_points"], dtype=np.float64),
        camera_1_points=np.asarray(data["camera_1_points"], dtype=np.float64),
        screen_points=np.asarray(data["screen_points"], dtype=np.float64),
        image_width=int(data["image_width"]),
        image_height=int(data["image_height"]),
        timestamp=float(data["timestamp"]),
    )


def map_raw_coordinates(
    camera_0_point: np.ndarray | None,
    camera_1_point: np.ndarray | None,
    calibration: HomographyCalibration,
    confidence_0: float = 1.0,
    confidence_1: float = 1.0,
) -> dict[str, Any]:
    screen_0 = None
    screen_1 = None
    if camera_0_point is not None:
        screen_0 = map_camera_point(camera_0_point, calibration.camera_0_points, calibration.screen_width_px, calibration.screen_height_px)
    if camera_1_point is not None:
        screen_1 = map_camera_point(camera_1_point, calibration.camera_1_points, calibration.screen_width_px, calibration.screen_height_px)

    combined = combine_screen_points(screen_0, screen_1, confidence_0, confidence_1)
    if combined is None:
        return {
            "valid": False,
            "screen_0": None,
            "screen_1": None,
            "pixel_x": None,
            "pixel_y": None,
            "normalized_x": None,
            "normalized_y": None,
        }

    pixel_x = int(round(float(combined[0])))
    pixel_y = int(round(float(combined[1])))
    return {
        "valid": True,
        "screen_0": None if screen_0 is None else {"x": float(screen_0[0]), "y": float(screen_0[1])},
        "screen_1": None if screen_1 is None else {"x": float(screen_1[0]), "y": float(screen_1[1])},
        "pixel_x": pixel_x,
        "pixel_y": pixel_y,
        "normalized_x": float(combined[0]) / float(calibration.screen_width_px - 1),
        "normalized_y": float(combined[1]) / float(calibration.screen_height_px - 1),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map raw camera LED coordinates through saved screen homographies.")
    parser.add_argument("--camera-0-x", type=float, default=None)
    parser.add_argument("--camera-0-y", type=float, default=None)
    parser.add_argument("--camera-1-x", type=float, default=None)
    parser.add_argument("--camera-1-y", type=float, default=None)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_HOMOGRAPHY_CALIBRATION)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def optional_point(x: float | None, y: float | None, label: str) -> np.ndarray | None:
    if x is None and y is None:
        return None
    if x is None or y is None:
        raise ValueError(f"{label} requires both x and y.")
    return np.array([x, y], dtype=np.float64)


def main() -> None:
    args = parse_args()
    calibration = load_homography_calibration(args.calibration)
    camera_0_point = optional_point(args.camera_0_x, args.camera_0_y, "camera 0")
    camera_1_point = optional_point(args.camera_1_x, args.camera_1_y, "camera 1")
    result = map_raw_coordinates(camera_0_point, camera_1_point, calibration)
    if args.json:
        print(json.dumps(result, separators=(",", ":")))
        return
    if not result["valid"]:
        print("No valid screen point.")
        return
    print(f"Pixel: x={result['pixel_x']} y={result['pixel_y']}")
    print(f"Normalized: x={result['normalized_x']:.4f} y={result['normalized_y']:.4f}")
    print(f"Camera 0 estimate: {result['screen_0']}")
    print(f"Camera 1 estimate: {result['screen_1']}")


if __name__ == "__main__":
    main()

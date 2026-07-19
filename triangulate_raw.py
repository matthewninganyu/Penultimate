from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from config import (
    DEFAULT_MAX_REPROJECTION_ERROR,
    DEFAULT_SCREEN_CALIBRATION,
    DEFAULT_SCREEN_MARGIN_MM,
)
from screen_mapping import load_screen_calibration, point_to_screen_position
from triangulation import point_in_front_of_both_cameras, project_point, undistort_point


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map one pair of raw camera LED coordinates into screen coordinates "
            "using calibration/screen_calibration.npz."
        )
    )
    parser.add_argument("--camera-0-x", type=float, required=True)
    parser.add_argument("--camera-0-y", type=float, required=True)
    parser.add_argument("--camera-1-x", type=float, required=True)
    parser.add_argument("--camera-1-y", type=float, required=True)
    parser.add_argument(
        "--screen-calibration",
        type=Path,
        default=DEFAULT_SCREEN_CALIBRATION,
    )
    parser.add_argument(
        "--max-reprojection-error",
        type=float,
        default=DEFAULT_MAX_REPROJECTION_ERROR,
    )
    parser.add_argument(
        "--screen-margin-mm",
        type=float,
        default=DEFAULT_SCREEN_MARGIN_MM,
    )
    parser.add_argument("--json", action="store_true", help="Print compact JSON.")
    return parser.parse_args()


def triangulate_raw_points(
    point_0: np.ndarray,
    point_1: np.ndarray,
    calibration_path: Path,
    max_reprojection_error: float,
    screen_margin_mm: float,
) -> dict[str, object]:
    calibration = load_screen_calibration(calibration_path)
    undistorted_0 = undistort_point(point_0, calibration.K0, calibration.D0)
    undistorted_1 = undistort_point(point_1, calibration.K1, calibration.D1)
    point_4d = cv2.triangulatePoints(
        calibration.P0,
        calibration.P1,
        undistorted_0,
        undistorted_1,
    )
    scale = float(point_4d[3, 0])
    if abs(scale) < 1e-9:
        return invalid_result("homogeneous scale is too close to zero")

    point_3d = (point_4d[:3, 0] / scale).astype(np.float64)
    if not point_in_front_of_both_cameras(point_3d, calibration):
        return invalid_result("triangulated point is behind at least one camera", point_3d)

    projected_0 = project_point(point_3d, calibration.K0, calibration.D0, calibration.R0, calibration.t0)
    projected_1 = project_point(point_3d, calibration.K1, calibration.D1, calibration.R1, calibration.t1)
    error_0 = float(np.linalg.norm(projected_0 - point_0))
    error_1 = float(np.linalg.norm(projected_1 - point_1))
    reprojection_error = (error_0 + error_1) / 2.0
    if reprojection_error > max_reprojection_error:
        return invalid_result(
            f"reprojection error {reprojection_error:.3f}px exceeds {max_reprojection_error:.3f}px",
            point_3d,
            reprojection_error,
        )

    position = point_to_screen_position(point_3d, calibration, screen_margin_mm)
    if position is None:
        return invalid_result("triangulated point is far outside the screen workspace", point_3d, reprojection_error)

    return {
        "valid": True,
        "camera_0": {"x": float(point_0[0]), "y": float(point_0[1])},
        "camera_1": {"x": float(point_1[0]), "y": float(point_1[1])},
        "x_mm": position.x_mm,
        "y_mm": position.y_mm,
        "distance_mm": position.distance_mm,
        "normalized_x": position.normalized_x,
        "normalized_y": position.normalized_y,
        "pixel_x": position.pixel_x,
        "pixel_y": position.pixel_y,
        "reprojection_error": reprojection_error,
    }


def invalid_result(
    reason: str,
    point_3d: np.ndarray | None = None,
    reprojection_error: float | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {"valid": False, "reason": reason}
    if point_3d is not None:
        result["x_mm"] = float(point_3d[0])
        result["y_mm"] = float(point_3d[1])
        result["distance_mm"] = float(point_3d[2])
    if reprojection_error is not None:
        result["reprojection_error"] = reprojection_error
    return result


def print_human(result: dict[str, object]) -> None:
    if not result["valid"]:
        print(f"Invalid triangulation: {result['reason']}")
        if "x_mm" in result:
            print(
                "Raw 3D estimate: "
                f"x={result['x_mm']:.2f}mm y={result['y_mm']:.2f}mm "
                f"z={result['distance_mm']:.2f}mm"
            )
        return

    print(
        "Screen position: "
        f"x={result['x_mm']:.2f}mm y={result['y_mm']:.2f}mm "
        f"z={result['distance_mm']:.2f}mm"
    )
    print(
        "Normalized: "
        f"x={result['normalized_x']:.4f} y={result['normalized_y']:.4f}"
    )
    print(f"Pixels: x={result['pixel_x']} y={result['pixel_y']}")
    print(f"Reprojection error: {result['reprojection_error']:.3f}px")


def main() -> None:
    args = parse_args()
    point_0 = np.array([args.camera_0_x, args.camera_0_y], dtype=np.float64)
    point_1 = np.array([args.camera_1_x, args.camera_1_y], dtype=np.float64)
    result = triangulate_raw_points(
        point_0,
        point_1,
        args.screen_calibration,
        args.max_reprojection_error,
        args.screen_margin_mm,
    )
    if args.json:
        print(json.dumps(result, separators=(",", ":")))
    else:
        print_human(result)


if __name__ == "__main__":
    main()

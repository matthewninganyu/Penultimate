from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from camera_manager import Picamera2, ensure_picamera2_available
from config import CAMERA_FORMAT, DEFAULT_HEIGHT, DEFAULT_WIDTH


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate one Raspberry Pi Camera Module.")
    parser.add_argument("--camera", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--board", choices=("auto", "charuco", "checkerboard"), default="auto")
    parser.add_argument("--checkerboard-cols", type=int, default=9, help="Inner checkerboard corners across.")
    parser.add_argument("--checkerboard-rows", type=int, default=6, help="Inner checkerboard corners down.")
    parser.add_argument("--charuco-squares-x", type=int, default=7)
    parser.add_argument("--charuco-squares-y", type=int, default=5)
    parser.add_argument("--charuco-marker-size-mm", type=float, default=18.0)
    parser.add_argument("--square-size-mm", type=float, default=24.0)
    return parser.parse_args()


def create_object_points(cols: int, rows: int, square_size_mm: float) -> np.ndarray:
    points = np.zeros((rows * cols, 3), np.float32)
    points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    points *= square_size_mm
    return points


def detect_checkerboard(gray: np.ndarray, cols: int, rows: int) -> tuple[bool, np.ndarray | None]:
    found, corners = cv2.findChessboardCorners(gray, (cols, rows))
    if not found or corners is None:
        return False, None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, refined


def aruco_available() -> bool:
    return hasattr(cv2, "aruco") and hasattr(cv2.aruco, "calibrateCameraCharuco")


def create_charuco_board(args: argparse.Namespace):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    if hasattr(cv2.aruco, "CharucoBoard"):
        return cv2.aruco.CharucoBoard(
            (args.charuco_squares_x, args.charuco_squares_y),
            args.square_size_mm,
            args.charuco_marker_size_mm,
            dictionary,
        )
    return cv2.aruco.CharucoBoard_create(
        args.charuco_squares_x,
        args.charuco_squares_y,
        args.square_size_mm,
        args.charuco_marker_size_mm,
        dictionary,
    )


def detect_charuco(
    gray: np.ndarray,
    board,
) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
    if ids is None or len(ids) == 0:
        return False, None, None
    _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
    if charuco_corners is None or charuco_ids is None or len(charuco_ids) < 6:
        return False, None, None
    return True, charuco_corners, charuco_ids


def calibrate_camera(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    image_size: tuple[int, int],
) -> tuple[float, np.ndarray, np.ndarray, list[float]]:
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        None,
        None,
    )
    per_image_errors: list[float] = []
    for obj, img, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
        error = cv2.norm(img, projected, cv2.NORM_L2) / len(projected)
        per_image_errors.append(float(error))
    return float(rms), K, D, per_image_errors


def calibrate_charuco_camera(
    charuco_corners: list[np.ndarray],
    charuco_ids: list[np.ndarray],
    board,
    image_size: tuple[int, int],
) -> tuple[float, np.ndarray, np.ndarray, list[float]]:
    rms, K, D, _, _ = cv2.aruco.calibrateCameraCharuco(
        charuco_corners,
        charuco_ids,
        board,
        image_size,
        None,
        None,
    )
    return float(rms), K, D, []


def save_intrinsics(
    path: Path,
    camera_index: int,
    width: int,
    height: int,
    K: np.ndarray,
    D: np.ndarray,
    rms_error: float,
    per_image_errors: list[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        K=K,
        D=D,
        camera_index=camera_index,
        image_width=width,
        image_height=height,
        rms_error=rms_error,
        per_image_errors=np.array(per_image_errors, dtype=np.float64),
        timestamp=time.time(),
    )


def sample_count(use_charuco: bool, image_points: list[np.ndarray], charuco_corners: list[np.ndarray]) -> int:
    return len(charuco_corners) if use_charuco else len(image_points)


def configure_camera(camera_index: int, width: int, height: int):
    ensure_picamera2_available()
    camera = Picamera2(camera_index)
    config = camera.create_video_configuration(main={"size": (width, height), "format": CAMERA_FORMAT})
    camera.configure(config)
    return camera


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    use_charuco = args.board in ("auto", "charuco") and aruco_available()
    if args.board == "charuco" and not use_charuco:
        raise RuntimeError("This OpenCV build does not expose cv2.aruco ChArUco calibration APIs.")
    if args.board == "auto" and not use_charuco:
        print("cv2.aruco ChArUco APIs unavailable; falling back to checkerboard calibration.")

    charuco_board = create_charuco_board(args) if use_charuco else None
    object_template = create_object_points(args.checkerboard_cols, args.checkerboard_rows, args.square_size_mm)
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    charuco_corners: list[np.ndarray] = []
    charuco_ids: list[np.ndarray] = []
    camera = None

    try:
        camera = configure_camera(args.camera, args.width, args.height)
        camera.start()
        print("Move the board through different positions, rotations, distances, and angles.")
        print("SPACE captures a valid sample, Q quits without saving.")
        while sample_count(use_charuco, image_points, charuco_corners) < args.samples:
            frame = camera.capture_array()[:, :, :3]
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if use_charuco:
                found, corners, ids = detect_charuco(gray, charuco_board)
            else:
                found, corners = detect_checkerboard(gray, args.checkerboard_cols, args.checkerboard_rows)
                ids = None
            preview = frame.copy()
            if found and corners is not None:
                if use_charuco:
                    cv2.aruco.drawDetectedCornersCharuco(preview, corners, ids)
                else:
                    cv2.drawChessboardCorners(preview, (args.checkerboard_cols, args.checkerboard_rows), corners, found)
            cv2.putText(
                preview,
                f"Samples: {sample_count(use_charuco, image_points, charuco_corners)}/{args.samples} "
                f"{'DETECTED' if found else 'NOT DETECTED'}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0) if found else (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Penultimate Intrinsic Calibration", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                print("Calibration cancelled.")
                return
            if key == ord(" ") and found and corners is not None:
                if use_charuco:
                    charuco_corners.append(corners.copy())
                    charuco_ids.append(ids.copy())
                else:
                    object_points.append(object_template.copy())
                    image_points.append(corners.copy())
        print(f"Captured sample {sample_count(use_charuco, image_points, charuco_corners)}/{args.samples}")

        if use_charuco:
            rms, K, D, per_image_errors = calibrate_charuco_camera(charuco_corners, charuco_ids, charuco_board, (args.width, args.height))
        else:
            rms, K, D, per_image_errors = calibrate_camera(object_points, image_points, (args.width, args.height))
        save_intrinsics(args.output, args.camera, args.width, args.height, K, D, rms, per_image_errors)
        print(f"Saved {args.output}")
        if per_image_errors:
            print(f"RMS error: {rms:.4f} px; mean per-image error: {np.mean(per_image_errors):.4f} px")
        else:
            print(f"RMS error: {rms:.4f} px")
        if rms > 1.5:
            print("Warning: calibration RMS is high. Recapture with sharper, more varied samples.")
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        if camera is not None:
            for method in ("stop", "close"):
                try:
                    getattr(camera, method)()
                except Exception as error:
                    LOGGER.warning("Failed to %s camera: %s", method, error)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

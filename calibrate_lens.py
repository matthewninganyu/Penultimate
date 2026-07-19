from __future__ import annotations

import argparse
import glob as glob_module
import time
from pathlib import Path

import cv2
import numpy as np

from camera import (
    capture_bgr_frame,
    configure_camera,
    ensure_frame_size,
    safe_camera_call,
    set_full_fov_crop,
)


DEFAULT_CAMERA_INDEX = 0
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 960
DEFAULT_BOARD_COLS = 9
DEFAULT_BOARD_ROWS = 6
DEFAULT_SQUARE_SIZE_MM = 25.0
DEFAULT_MIN_SAMPLES = 15

# Sub-pixel refinement stop criteria
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

WINDOW_CAPTURE = "Lens Calibration - Capture"
WINDOW_PREVIEW = "Lens Calibration - Undistortion Preview"

WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate a Raspberry Pi CSI camera lens using a printed checkerboard "
            "to compute the camera matrix and distortion coefficients. Saves a .npz "
            "file that other scripts can load to undistort frames."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Live capture from camera 0:\n"
            "  python calibrate_lens.py --camera 0\n\n"
            "  # Calibrate camera 1 at higher resolution:\n"
            "  python calibrate_lens.py --camera 1 --width 1920 --height 1080\n\n"
            "  # Calibrate from a folder of saved images:\n"
            "  python calibrate_lens.py --images calibration_frames/ --output cal.npz\n\n"
            "Board size note:\n"
            "  --board-cols / --board-rows are the number of *inner* corners, which is\n"
            "  one less than the number of squares in each direction. A common A4\n"
            "  printout with 10x7 squares has 9x6 inner corners (the default)."
        ),
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=DEFAULT_CAMERA_INDEX,
        help="Picamera2 camera index to calibrate.",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument(
        "--board-cols",
        type=int,
        default=DEFAULT_BOARD_COLS,
        metavar="N",
        help="Inner corner count along the horizontal axis (squares - 1).",
    )
    parser.add_argument(
        "--board-rows",
        type=int,
        default=DEFAULT_BOARD_ROWS,
        metavar="N",
        help="Inner corner count along the vertical axis (squares - 1).",
    )
    parser.add_argument(
        "--square-size",
        type=float,
        default=DEFAULT_SQUARE_SIZE_MM,
        metavar="MM",
        help="Physical side length of one checkerboard square in millimetres.",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=DEFAULT_MIN_SAMPLES,
        help="Minimum accepted board captures required before calibrating.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output .npz file. Defaults to calibration_{camera}.npz in the "
            "current directory."
        ),
    )
    parser.add_argument(
        "--images",
        type=str,
        default=None,
        metavar="DIR_OR_GLOB",
        help=(
            "Directory or glob pattern of pre-existing images to calibrate from "
            "instead of live capture (e.g. 'frames/*.jpg' or 'frames/')."
        ),
    )
    parser.add_argument(
        "--save-frames",
        type=Path,
        default=None,
        metavar="DIR",
        help="Save each accepted calibration frame to this directory.",
    )
    parser.add_argument(
        "--fov-mode",
        choices=("full", "current"),
        default="full",
        help="Camera FOV mode passed to Picamera2 (ignored with --images).",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help=(
            "Skip the live undistortion preview shown after successful calibration. "
            "Automatically set when using --images."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Core calibration helpers
# ---------------------------------------------------------------------------


def make_object_points(
    board_cols: int,
    board_rows: int,
    square_size_mm: float,
) -> np.ndarray:
    """Return the 3-D world coordinates for one checkerboard pose (Z = 0)."""
    objp = np.zeros((board_rows * board_cols, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2)
    objp *= square_size_mm
    return objp


def find_checkerboard_corners(
    gray: np.ndarray,
    board_size: tuple[int, int],
) -> np.ndarray | None:
    """Detect and refine checkerboard inner corners. Returns None if not found."""
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )
    try:
        found, corners = cv2.findChessboardCorners(gray, board_size, None, flags)
    except Exception:
        return None
    if not found or corners is None or corners.size == 0:
        return None
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), SUBPIX_CRITERIA)


def run_calibration(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    image_size: tuple[int, int],
) -> tuple[float, np.ndarray, np.ndarray]:
    camera_matrix = np.zeros((3, 3), dtype=np.float64)
    dist_coeffs = np.zeros((5, 1), dtype=np.float64)
    rms, camera_matrix, dist_coeffs, _rvecs, _tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        camera_matrix,
        dist_coeffs,
    )
    return float(rms), camera_matrix, dist_coeffs.ravel()


def report_and_save(
    output_path: Path,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    image_size: tuple[int, int],
    rms: float,
    sample_count: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(output_path),
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=np.array(image_size),
        rms_error=np.float64(rms),
    )
    if rms < 0.5:
        quality = "excellent"
    elif rms < 1.0:
        quality = "good"
    else:
        quality = "poor - recapture with more varied board poses"
    print(f"\nCalibration complete ({sample_count} frames).")
    print(f"  RMS reprojection error : {rms:.4f} px  [{quality}]")
    print(f"  Image size             : {image_size[0]}x{image_size[1]}")
    print(f"  Camera matrix          :\n{camera_matrix}")
    print(f"  Distortion coefficients: {dist_coeffs}")
    print(f"  Saved to               : {output_path}")


# ---------------------------------------------------------------------------
# --images path: calibrate from a directory or glob of saved frames
# ---------------------------------------------------------------------------


def collect_image_paths(images_arg: str) -> list[Path]:
    target = Path(images_arg)
    if target.is_dir():
        paths: list[Path] = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif"):
            paths.extend(sorted(target.glob(ext)))
            paths.extend(sorted(target.glob(ext.upper())))
        return sorted(set(paths))

    matched = [Path(p) for p in sorted(glob_module.glob(images_arg))]
    if not matched:
        raise FileNotFoundError(f"No images matched: {images_arg!r}")
    return matched


def calibrate_from_images(args: argparse.Namespace) -> int:
    board_size = (args.board_cols, args.board_rows)
    objp = make_object_points(args.board_cols, args.board_rows, args.square_size)

    image_paths = collect_image_paths(args.images)
    print(f"Found {len(image_paths)} image(s) - processing...\n")

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"  [skip] cannot read: {image_path}")
            continue

        h, w = frame.shape[:2]
        if image_size is None:
            image_size = (w, h)
        elif image_size != (w, h):
            print(
                f"  [skip] size mismatch {w}x{h} vs {image_size[0]}x{image_size[1]}: "
                f"{image_path.name}"
            )
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners = find_checkerboard_corners(gray, board_size)

        if corners is None:
            print(f"  [skip] no board: {image_path.name}")
            cv2.imshow(WINDOW_CAPTURE, frame)
            cv2.waitKey(200)
            continue

        object_points.append(objp)
        image_points.append(corners)
        print(f"  [ok]   {image_path.name}  ({len(object_points)} accepted)")

        preview = frame.copy()
        cv2.drawChessboardCorners(preview, board_size, corners, True)
        cv2.imshow(WINDOW_CAPTURE, preview)
        cv2.waitKey(150)

    cv2.destroyAllWindows()
    accepted = len(object_points)

    if accepted < args.min_samples:
        print(
            f"\nError: {accepted} accepted image(s); need at least {args.min_samples}. "
            "Try adding more images with different board poses."
        )
        return 1

    print(f"\nRunning calibration on {accepted} frame(s)...")
    assert image_size is not None
    rms, camera_matrix, dist_coeffs = run_calibration(
        object_points, image_points, image_size
    )
    output_path = args.output or Path(f"calibration_{args.camera}.npz")
    report_and_save(output_path, camera_matrix, dist_coeffs, image_size, rms, accepted)
    return 0


# ---------------------------------------------------------------------------
# Live capture path
# ---------------------------------------------------------------------------


def _draw_hud(
    frame: np.ndarray,
    camera_index: int,
    accepted: int,
    min_samples: int,
    board_detected: bool,
    board_size: tuple[int, int],
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    cv2.putText(
        out,
        f"Lens Calibration - Camera {camera_index}  {w}x{h}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        WHITE,
        2,
        cv2.LINE_AA,
    )

    ready = accepted >= min_samples
    count_color = GREEN if ready else YELLOW
    count_label = (
        f"Captures: {accepted}/{min_samples}  READY - press C to calibrate"
        if ready
        else f"Captures: {accepted}/{min_samples}"
    )
    cv2.putText(
        out,
        count_label,
        (10, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        count_color,
        2,
        cv2.LINE_AA,
    )

    if board_detected:
        board_label = (
            f"Board detected ({board_size[0]}x{board_size[1]}) - SPACE to capture"
        )
        board_color = GREEN
    else:
        board_label = "Searching for checkerboard..."
        board_color = RED
    cv2.putText(
        out,
        board_label,
        (10, 94),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        board_color,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        out,
        "SPACE: capture   C: calibrate   Q: quit",
        (10, h - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        WHITE,
        1,
        cv2.LINE_AA,
    )

    return out


def _undistortion_preview(
    camera: object,
    camera_index: int,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    image_size: tuple[int, int],
) -> None:
    """Show a live side-by-side original / undistorted preview."""
    w, h = image_size
    new_cam_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), alpha=1, newImgSize=(w, h)
    )
    x, y, rw, rh = roi
    print("Undistortion preview open. Press Q or ESC to close.")

    while True:
        try:
            frame = capture_bgr_frame(camera, f"Camera {camera_index}", "rgb")
        except RuntimeError as e:
            print(f"Preview capture error: {e}")
            break
        frame = ensure_frame_size(frame, w, h)
        undistorted = cv2.undistort(
            frame, camera_matrix, dist_coeffs, None, new_cam_matrix
        )

        # Crop to valid ROI then scale back to original size so the side-by-side
        # panels stay the same pixel dimensions.
        if rw > 0 and rh > 0:
            cropped = cv2.resize(undistorted[y : y + rh, x : x + rw], (w, h))
        else:
            cropped = undistorted

        cv2.putText(
            frame,
            "Original",
            (10, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            RED,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            cropped,
            "Undistorted",
            (10, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            GREEN,
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(WINDOW_PREVIEW, np.hstack((frame, cropped)))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break

    cv2.destroyWindow(WINDOW_PREVIEW)


def calibrate_from_camera(args: argparse.Namespace) -> int:
    board_size = (args.board_cols, args.board_rows)
    objp = make_object_points(args.board_cols, args.board_rows, args.square_size)
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    capture_count = 0

    if args.save_frames:
        args.save_frames.mkdir(parents=True, exist_ok=True)
        existing = sorted(args.save_frames.glob("frame_*.png"))
        if existing:
            print(f"Loading {len(existing)} existing frame(s) from {args.save_frames}...")
            for path in existing:
                img = cv2.imread(str(path))
                if img is None:
                    print(f"  [skip] cannot read: {path.name}")
                    continue
                h, w = img.shape[:2]
                if image_size is None:
                    image_size = (w, h)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                corners = find_checkerboard_corners(gray, board_size)
                if corners is None:
                    print(f"  [skip] no board found: {path.name}")
                    continue
                object_points.append(objp)
                image_points.append(corners)
                capture_count += 1
                print(f"  [ok]   {path.name}")
            print(f"Loaded {len(object_points)} usable frame(s).")

    print(
        f"Camera {args.camera}  {args.width}x{args.height}\n"
        f"Board  {args.board_cols}x{args.board_rows} inner corners  "
        f"{args.square_size:.1f} mm squares\n"
        f"Minimum {args.min_samples} captures needed.\n"
        "\nTilt and move the checkerboard to different positions and angles.\n"
        "SPACE: capture frame   C: run calibration (once ready)   Q: quit\n"
    )

    camera = None
    calibration_result: tuple[float, np.ndarray, np.ndarray] | None = None

    try:
        camera, full_fov_crop = configure_camera(
            args.camera, args.width, args.height, args.fov_mode
        )
        camera.start()
        set_full_fov_crop(camera, args.camera, full_fov_crop)
        time.sleep(1.0)

        while True:
            try:
                frame = capture_bgr_frame(camera, f"Camera {args.camera}", "rgb")
            except RuntimeError as error:
                print(f"Capture warning: {error}")
                time.sleep(0.05)
                continue

            frame = ensure_frame_size(frame, args.width, args.height)
            if image_size is None:
                image_size = (frame.shape[1], frame.shape[0])

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners = find_checkerboard_corners(gray, board_size)

            hud = _draw_hud(
                frame,
                args.camera,
                len(object_points),
                args.min_samples,
                corners is not None,
                board_size,
            )
            if corners is not None:
                cv2.drawChessboardCorners(hud, board_size, corners, True)

            cv2.imshow(WINDOW_CAPTURE, hud)
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q"), 27):
                break

            if key == ord(" "):
                if corners is None:
                    print("No board visible - adjust position and try again.")
                else:
                    capture_count += 1
                    object_points.append(objp)
                    image_points.append(corners)
                    print(
                        f"  Captured {capture_count} "
                        f"({len(object_points)}/{args.min_samples} minimum)"
                    )
                    if args.save_frames:
                        cv2.imwrite(
                            str(args.save_frames / f"frame_{capture_count:03d}.png"),
                            frame,
                        )

            if key in (ord("c"), ord("C")) and len(object_points) >= args.min_samples:
                cv2.destroyWindow(WINDOW_CAPTURE)
                print(f"\nRunning calibration on {len(object_points)} frames...")
                assert image_size is not None
                rms, camera_matrix, dist_coeffs = run_calibration(
                    object_points, image_points, image_size
                )
                calibration_result = (rms, camera_matrix, dist_coeffs)

                output_path = args.output or Path(f"calibration_{args.camera}.npz")
                report_and_save(
                    output_path,
                    camera_matrix,
                    dist_coeffs,
                    image_size,
                    rms,
                    len(object_points),
                )

                if not args.no_preview:
                    assert image_size is not None
                    _undistortion_preview(
                        camera, args.camera, camera_matrix, dist_coeffs, image_size
                    )
                break

    finally:
        safe_camera_call(camera, "stop", f"camera {args.camera}")
        safe_camera_call(camera, "close", f"camera {args.camera}")
        cv2.destroyAllWindows()

    if calibration_result is None:
        n = len(object_points)
        if n < args.min_samples:
            print(
                f"\nCalibration not run: only {n} frame(s) captured "
                f"(need {args.min_samples}). Run again and press C when ready."
            )
        return 1

    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    try:
        if args.images is not None:
            exit_code = calibrate_from_images(args)
        else:
            exit_code = calibrate_from_camera(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        exit_code = 1
    except Exception as error:
        print(f"Error: {error}")
        raise SystemExit(1) from error
    finally:
        cv2.destroyAllWindows()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
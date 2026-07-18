from __future__ import annotations

import argparse
import logging
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from camera_manager import DualCameraManager
from config import (
    DEFAULT_CAMERA_0,
    DEFAULT_CAMERA_0_INTRINSICS,
    DEFAULT_CAMERA_1,
    DEFAULT_CAMERA_1_INTRINSICS,
    DEFAULT_HEIGHT,
    DEFAULT_MAX_FRAME_SKEW_MS,
    DEFAULT_MIN_AREA,
    DEFAULT_SCREEN_CALIBRATION,
    DEFAULT_WIDTH,
)
from led_detection import LedDetector, parse_hsv_triplet
from models import LedCandidate
from screen_mapping import load_intrinsics


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Guided screen-pose calibration for Penultimate.")
    parser.add_argument("--camera-0", type=int, default=DEFAULT_CAMERA_0)
    parser.add_argument("--camera-1", type=int, default=DEFAULT_CAMERA_1)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--camera-0-intrinsics", type=Path, default=DEFAULT_CAMERA_0_INTRINSICS)
    parser.add_argument("--camera-1-intrinsics", type=Path, default=DEFAULT_CAMERA_1_INTRINSICS)
    parser.add_argument("--output", type=Path, default=DEFAULT_SCREEN_CALIBRATION)
    parser.add_argument("--screen-width-mm", type=float, required=True)
    parser.add_argument("--screen-height-mm", type=float, required=True)
    parser.add_argument("--screen-width-px", type=int, required=True)
    parser.add_argument("--screen-height-px", type=int, required=True)
    parser.add_argument("--nine-point", action="store_true")
    parser.add_argument("--samples-per-point", type=int, default=25)
    parser.add_argument("--stable-window", type=int, default=12)
    parser.add_argument("--stable-std-px", type=float, default=1.8)
    parser.add_argument("--max-frame-skew-ms", type=float, default=DEFAULT_MAX_FRAME_SKEW_MS)
    parser.add_argument("--min-area", type=float, default=DEFAULT_MIN_AREA)
    parser.add_argument("--lower-h", type=int, default=10)
    parser.add_argument("--lower-s", type=int, default=150)
    parser.add_argument("--lower-v", type=int, default=220)
    parser.add_argument("--upper-h", type=int, default=40)
    parser.add_argument("--upper-s", type=int, default=255)
    parser.add_argument("--upper-v", type=int, default=255)
    parser.add_argument("--show-mask", action="store_true")
    return parser.parse_args()


def calibration_points(width_mm: float, height_mm: float, nine_point: bool) -> list[tuple[str, np.ndarray]]:
    base = [
        ("TOP LEFT", np.array([0.0, 0.0, 0.0], dtype=np.float64)),
        ("TOP RIGHT", np.array([width_mm, 0.0, 0.0], dtype=np.float64)),
        ("BOTTOM RIGHT", np.array([width_mm, height_mm, 0.0], dtype=np.float64)),
        ("BOTTOM LEFT", np.array([0.0, height_mm, 0.0], dtype=np.float64)),
    ]
    if not nine_point:
        return base
    return [
        ("TOP LEFT", np.array([0.0, 0.0, 0.0])),
        ("TOP CENTRE", np.array([width_mm / 2.0, 0.0, 0.0])),
        ("TOP RIGHT", np.array([width_mm, 0.0, 0.0])),
        ("MIDDLE RIGHT", np.array([width_mm, height_mm / 2.0, 0.0])),
        ("BOTTOM RIGHT", np.array([width_mm, height_mm, 0.0])),
        ("BOTTOM CENTRE", np.array([width_mm / 2.0, height_mm, 0.0])),
        ("BOTTOM LEFT", np.array([0.0, height_mm, 0.0])),
        ("MIDDLE LEFT", np.array([0.0, height_mm / 2.0, 0.0])),
        ("CENTRE", np.array([width_mm / 2.0, height_mm / 2.0, 0.0])),
    ]


def most_likely_led(candidates: list[LedCandidate]) -> LedCandidate | None:
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: (candidate.peak_brightness, -candidate.area))


def is_stable(points_0: deque[np.ndarray], points_1: deque[np.ndarray], std_px: float) -> bool:
    if len(points_0) < points_0.maxlen or len(points_1) < points_1.maxlen:
        return False
    arr0 = np.vstack(points_0)
    arr1 = np.vstack(points_1)
    return bool(np.max(np.std(arr0, axis=0)) <= std_px and np.max(np.std(arr1, axis=0)) <= std_px)


def estimate_pose(
    object_points: np.ndarray,
    image_points: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    flags = cv2.SOLVEPNP_IPPE if len(object_points) >= 4 else cv2.SOLVEPNP_ITERATIVE
    solutions = cv2.solvePnPGeneric(object_points, image_points, K, D, flags=flags)
    retval, rvecs, tvecs = solutions[:3]
    if not retval:
        raise RuntimeError("solvePnPGeneric failed.")
    best: tuple[np.ndarray, np.ndarray, float] | None = None
    for rvec, tvec in zip(rvecs, tvecs):
        projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, D)
        error = float(cv2.norm(image_points, projected.reshape(-1, 2), cv2.NORM_L2) / len(object_points))
        R, _ = cv2.Rodrigues(rvec)
        camera_space = (R @ object_points.T) + tvec.reshape(3, 1)
        if np.any(camera_space[2, :] <= 0):
            continue
        if best is None or error < best[2]:
            best = (rvec.reshape(3, 1), tvec.reshape(3, 1), error)
    if best is None:
        raise RuntimeError("No physically valid pose solution found.")
    rvec, tvec, _ = best
    if hasattr(cv2, "solvePnPRefineLM"):
        rvec, tvec = cv2.solvePnPRefineLM(object_points, image_points, K, D, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, K, D)
    error = float(cv2.norm(image_points, projected.reshape(-1, 2), cv2.NORM_L2) / len(object_points))
    return rvec, tvec, R, error


def save_screen_calibration(
    args: argparse.Namespace,
    K0: np.ndarray,
    D0: np.ndarray,
    rvec0: np.ndarray,
    tvec0: np.ndarray,
    R0: np.ndarray,
    error0: float,
    K1: np.ndarray,
    D1: np.ndarray,
    rvec1: np.ndarray,
    tvec1: np.ndarray,
    R1: np.ndarray,
    error1: float,
    object_points: np.ndarray,
    image_points_0: np.ndarray,
    image_points_1: np.ndarray,
    raw_samples_0: np.ndarray,
    raw_samples_1: np.ndarray,
) -> None:
    P0 = K0 @ np.hstack((R0, tvec0.reshape(3, 1)))
    P1 = K1 @ np.hstack((R1, tvec1.reshape(3, 1)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        K0=K0,
        D0=D0,
        rvec0=rvec0,
        t0=tvec0,
        R0=R0,
        P0=P0,
        K1=K1,
        D1=D1,
        rvec1=rvec1,
        t1=tvec1,
        R1=R1,
        P1=P1,
        screen_width_mm=args.screen_width_mm,
        screen_height_mm=args.screen_height_mm,
        screen_width_px=args.screen_width_px,
        screen_height_px=args.screen_height_px,
        image_width=args.width,
        image_height=args.height,
        reprojection_error_0=error0,
        reprojection_error_1=error1,
        object_points=object_points,
        image_points_0=image_points_0,
        image_points_1=image_points_1,
        raw_samples_0=raw_samples_0,
        raw_samples_1=raw_samples_1,
        timestamp=time.time(),
    )


def draw_instruction(frame: np.ndarray, step: int, total: int, label: str, status: str) -> np.ndarray:
    out = frame.copy()
    cv2.putText(out, f"STEP {step} OF {total}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(out, f"TOUCH THE {label} CORNER" if "CENTRE" not in label else f"TOUCH {label}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
    cv2.putText(out, "HOLD THE LED TIP STILL", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)
    cv2.putText(out, status, (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    intr0 = load_intrinsics(args.camera_0_intrinsics)
    intr1 = load_intrinsics(args.camera_1_intrinsics)
    if (intr0.width, intr0.height) != (args.width, args.height) or (intr1.width, intr1.height) != (args.width, args.height):
        raise RuntimeError("Intrinsic calibration resolution does not match requested screen calibration resolution.")

    points = calibration_points(args.screen_width_mm, args.screen_height_mm, args.nine_point)
    detector = LedDetector(
        parse_hsv_triplet(args.lower_h, args.lower_s, args.lower_v),
        parse_hsv_triplet(args.upper_h, args.upper_s, args.upper_v),
        args.min_area,
    )
    manager = DualCameraManager(args.camera_0, args.camera_1, args.width, args.height)
    object_points: list[np.ndarray] = []
    median_0: list[np.ndarray] = []
    median_1: list[np.ndarray] = []
    raw_0: list[np.ndarray] = []
    raw_1: list[np.ndarray] = []

    try:
        manager.start()
        point_index = 0
        while point_index < len(points):
            label, object_point = points[point_index]
            window_0: deque[np.ndarray] = deque(maxlen=args.stable_window)
            window_1: deque[np.ndarray] = deque(maxlen=args.stable_window)
            samples_0: list[np.ndarray] = []
            samples_1: list[np.ndarray] = []
            while len(samples_0) < args.samples_per_point:
                frame0, frame1, skew_ms = manager.capture_pair()
                candidates0, mask0 = detector.detect(frame0.frame)
                candidates1, mask1 = detector.detect(frame1.frame)
                led0 = most_likely_led(candidates0)
                led1 = most_likely_led(candidates1)
                detected = led0 is not None and led1 is not None and skew_ms <= args.max_frame_skew_ms
                if led0 is not None:
                    window_0.append(led0.image_point())
                if led1 is not None:
                    window_1.append(led1.image_point())
                stable = detected and is_stable(window_0, window_1, args.stable_std_px)
                if stable:
                    samples_0.append(led0.image_point())
                    samples_1.append(led1.image_point())
                status = (
                    f"{'DETECTED' if detected else 'NOT DETECTED'} | "
                    f"{'STABLE' if stable else 'MOVING'} | "
                    f"samples={len(samples_0)}/{args.samples_per_point} skew={skew_ms:.1f}ms"
                )
                preview = np.hstack((frame0.frame, frame1.frame))
                cv2.imshow("Penultimate Screen Calibration", draw_instruction(preview, point_index + 1, len(points), label, status))
                if args.show_mask:
                    cv2.imshow("Penultimate Calibration Masks", np.hstack((mask0, mask1)))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    print("Calibration cancelled.")
                    return
                if key in (ord("r"), ord("R")):
                    object_points.clear()
                    median_0.clear()
                    median_1.clear()
                    raw_0.clear()
                    raw_1.clear()
                    point_index = 0
                    break
                if key == 8 and point_index > 0:
                    object_points.pop()
                    median_0.pop()
                    median_1.pop()
                    raw_0.pop()
                    raw_1.pop()
                    point_index -= 1
                    break
                if key == ord(" ") and detected:
                    samples_0.append(led0.image_point())
                    samples_1.append(led1.image_point())
            else:
                object_points.append(object_point)
                median_0.append(np.median(np.vstack(samples_0), axis=0))
                median_1.append(np.median(np.vstack(samples_1), axis=0))
                raw_0.append(np.vstack(samples_0))
                raw_1.append(np.vstack(samples_1))
                print(f"Captured {label}: cam0={median_0[-1]} cam1={median_1[-1]}")
                point_index += 1

        obj = np.vstack(object_points).astype(np.float64)
        img0 = np.vstack(median_0).astype(np.float64)
        img1 = np.vstack(median_1).astype(np.float64)
        rvec0, tvec0, R0, err0 = estimate_pose(obj, img0, intr0.K, intr0.D)
        rvec1, tvec1, R1, err1 = estimate_pose(obj, img1, intr1.K, intr1.D)
        save_screen_calibration(args, intr0.K, intr0.D, rvec0, tvec0, R0, err0, intr1.K, intr1.D, rvec1, tvec1, R1, err1, obj, img0, img1, np.array(raw_0, dtype=object), np.array(raw_1, dtype=object))
        print(f"Saved {args.output}")
        print(f"Camera 0 reprojection error: {err0:.3f}px")
        print(f"Camera 1 reprojection error: {err1:.3f}px")
        if err0 > 4.0 or err1 > 4.0:
            print("Warning: high reprojection error. Re-run calibration with steadier corner touches.")
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        manager.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


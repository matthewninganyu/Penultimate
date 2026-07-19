from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from camera import (
    COLOR_ORDERS,
    FOV_MODES,
    capture_bgr_frame,
    configure_camera,
    ensure_frame_size,
    safe_camera_call,
    set_full_fov_crop,
)
from config import DEFAULT_HEIGHT, DEFAULT_MIN_AREA, DEFAULT_WIDTH
from blue_led_detection import LedDetector, parse_hsv_threshold
from models import LedCandidate
from screen_mapper import (
    DEFAULT_HOMOGRAPHY_CALIBRATION,
    build_homography,
    save_homography_calibration,
    screen_corner_points,
)


DEFAULT_CAMERA_0 = 0
DEFAULT_CAMERA_1 = 1
DEFAULT_SAMPLES_PER_CORNER = 20
DEFAULT_STABLE_WINDOW = 10
DEFAULT_STABLE_STD_PX = 2.0

WINDOW_NAME = "Penultimate Homography Calibration"
MASK_WINDOW_NAME = "Penultimate Homography Masks"

CORNER_LABELS = ("TOP LEFT", "TOP RIGHT", "BOTTOM RIGHT", "BOTTOM LEFT")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate per-camera screen homographies from four LED corner touches."
    )
    parser.add_argument("--camera-0", type=int, default=DEFAULT_CAMERA_0)
    parser.add_argument("--camera-1", type=int, default=DEFAULT_CAMERA_1)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fov-mode", choices=FOV_MODES, default="full")
    parser.add_argument("--color-order", choices=COLOR_ORDERS, default="rgb")
    parser.add_argument("--screen-width-px", type=int, required=True)
    parser.add_argument("--screen-height-px", type=int, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_HOMOGRAPHY_CALIBRATION)
    parser.add_argument("--min-area", type=float, default=DEFAULT_MIN_AREA)
    parser.add_argument("--hsv-lower", default="100,150,180", help="Lower HSV blue halo threshold as H,S,V.")
    parser.add_argument("--hsv-upper", default="130,255,255", help="Upper HSV blue halo threshold as H,S,V.")
    parser.add_argument("--samples-per-corner", type=int, default=DEFAULT_SAMPLES_PER_CORNER)
    parser.add_argument("--stable-window", type=int, default=DEFAULT_STABLE_WINDOW)
    parser.add_argument("--stable-std-px", type=float, default=DEFAULT_STABLE_STD_PX)
    parser.add_argument("--show-mask", action="store_true")
    return parser.parse_args()


def best_candidate(candidates: list[LedCandidate]) -> LedCandidate | None:
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: (candidate.peak_brightness, candidate.area))


def stable(points: deque[np.ndarray], std_px: float) -> bool:
    if len(points) < points.maxlen:
        return False
    values = np.vstack(points)
    return bool(np.max(np.std(values, axis=0)) <= std_px)


def draw_candidate(frame: np.ndarray, candidate: LedCandidate | None, label: str) -> np.ndarray:
    output = frame.copy()
    cv2.putText(output, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    if candidate is not None:
        center = (int(round(candidate.x)), int(round(candidate.y)))
        cv2.circle(output, center, 18, (0, 255, 0), 2)
        cv2.putText(
            output,
            f"x={candidate.x:.1f} y={candidate.y:.1f}",
            (center[0] + 10, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return output


def draw_instructions(
    combined: np.ndarray,
    corner_index: int,
    total_corners: int,
    status: str,
) -> np.ndarray:
    output = combined.copy()
    label = CORNER_LABELS[corner_index]
    lines = [
        f"STEP {corner_index + 1} OF {total_corners}",
        f"TOUCH {label}",
        "HOLD LED TIP STILL",
        status,
        "SPACE capture stable point | BACKSPACE redo previous | R restart | Q quit",
    ]
    for index, line in enumerate(lines):
        color = (0, 255, 255) if index in (1, 2) else (255, 255, 255)
        cv2.putText(
            output,
            line,
            (20, 36 + index * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75 if index < 3 else 0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return output


def open_dual_cameras(args: argparse.Namespace) -> tuple[Any, Any, tuple[int, int, int, int] | None, tuple[int, int, int, int] | None]:
    camera_0, crop_0 = configure_camera(args.camera_0, args.width, args.height, args.fov_mode)
    camera_1, crop_1 = configure_camera(args.camera_1, args.width, args.height, args.fov_mode)
    camera_0.start()
    camera_1.start()
    set_full_fov_crop(camera_0, args.camera_0, crop_0)
    set_full_fov_crop(camera_1, args.camera_1, crop_1)
    return camera_0, camera_1, crop_0, crop_1


def main() -> None:
    args = parse_args()
    lower = parse_hsv_threshold(args.hsv_lower, "--hsv-lower")
    upper = parse_hsv_threshold(args.hsv_upper, "--hsv-upper")
    detector = LedDetector(lower, upper, args.min_area)
    camera_0 = None
    camera_1 = None
    captured_0: list[np.ndarray] = []
    captured_1: list[np.ndarray] = []

    try:
        camera_0, camera_1, _, _ = open_dual_cameras(args)
        print("Touch the LED tip to each prompted screen corner.")
        print("Calibration order: TOP LEFT, TOP RIGHT, BOTTOM RIGHT, BOTTOM LEFT.")

        corner_index = 0
        while corner_index < len(CORNER_LABELS):
            recent_0: deque[np.ndarray] = deque(maxlen=args.stable_window)
            recent_1: deque[np.ndarray] = deque(maxlen=args.stable_window)
            samples_0: list[np.ndarray] = []
            samples_1: list[np.ndarray] = []

            while True:
                frame_0 = capture_bgr_frame(camera_0, "Camera 0", args.color_order)
                frame_1 = capture_bgr_frame(camera_1, "Camera 1", args.color_order)
                frame_0 = ensure_frame_size(frame_0, args.width, args.height)
                frame_1 = ensure_frame_size(frame_1, args.width, args.height)

                candidates_0, mask_0 = detector.detect(frame_0)
                candidates_1, mask_1 = detector.detect(frame_1)
                led_0 = best_candidate(candidates_0)
                led_1 = best_candidate(candidates_1)

                if led_0 is not None:
                    recent_0.append(led_0.image_point())
                if led_1 is not None:
                    recent_1.append(led_1.image_point())

                detected = led_0 is not None and led_1 is not None
                is_stable = detected and stable(recent_0, args.stable_std_px) and stable(recent_1, args.stable_std_px)
                if is_stable:
                    samples_0.append(led_0.image_point())
                    samples_1.append(led_1.image_point())
                    if len(samples_0) > args.samples_per_corner:
                        samples_0 = samples_0[-args.samples_per_corner :]
                        samples_1 = samples_1[-args.samples_per_corner :]

                status = (
                    f"{'DETECTED' if detected else 'NOT DETECTED'} | "
                    f"{'STABLE' if is_stable else 'MOVING'} | "
                    f"samples={len(samples_0)}/{args.samples_per_corner}"
                )
                preview_0 = draw_candidate(frame_0, led_0, "CAMERA 0")
                preview_1 = draw_candidate(frame_1, led_1, "CAMERA 1")
                combined = np.hstack((preview_0, preview_1))
                cv2.imshow(
                    WINDOW_NAME,
                    draw_instructions(combined, corner_index, len(CORNER_LABELS), status),
                )
                if args.show_mask:
                    cv2.imshow(MASK_WINDOW_NAME, np.hstack((mask_0, mask_1)))

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    print("Calibration cancelled.")
                    return
                if key in (ord("r"), ord("R")):
                    captured_0.clear()
                    captured_1.clear()
                    corner_index = 0
                    print("Restarted homography calibration.")
                    break
                if key == 8 and corner_index > 0:
                    captured_0.pop()
                    captured_1.pop()
                    corner_index -= 1
                    print(f"Redoing {CORNER_LABELS[corner_index]}.")
                    break
                if key == ord(" "):
                    if len(samples_0) < args.samples_per_corner or len(samples_1) < args.samples_per_corner:
                        print("Not enough stable samples yet; keep holding still.")
                        continue
                    point_0 = np.median(np.vstack(samples_0), axis=0)
                    point_1 = np.median(np.vstack(samples_1), axis=0)
                    captured_0.append(point_0)
                    captured_1.append(point_1)
                    print(f"Captured {CORNER_LABELS[corner_index]}: cam0={point_0}, cam1={point_1}")
                    corner_index += 1
                    break

        camera_0_points = np.vstack(captured_0).astype(np.float32)
        camera_1_points = np.vstack(captured_1).astype(np.float32)
        screen_points = screen_corner_points(args.screen_width_px, args.screen_height_px)
        H0 = build_homography(camera_0_points, screen_points)
        H1 = build_homography(camera_1_points, screen_points)
        save_homography_calibration(
            args.output,
            H0,
            H1,
            args.screen_width_px,
            args.screen_height_px,
            camera_0_points,
            camera_1_points,
            screen_points,
            args.width,
            args.height,
        )
        print(f"Saved homography calibration: {args.output}")
        print(f"Camera 0 points:\n{camera_0_points}")
        print(f"Camera 1 points:\n{camera_1_points}")
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        safe_camera_call(camera_0, "stop", "camera 0")
        safe_camera_call(camera_1, "stop", "camera 1")
        safe_camera_call(camera_0, "close", "camera 0")
        safe_camera_call(camera_1, "close", "camera 1")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

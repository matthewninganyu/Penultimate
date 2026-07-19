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
from screen_mapper import DEFAULT_HOMOGRAPHY_CALIBRATION, build_homography, save_homography_calibration


DEFAULT_CAMERA_0 = 0
DEFAULT_CAMERA_1 = 1
DEFAULT_SAMPLES_PER_TARGET = 20
DEFAULT_STABLE_WINDOW = 10
DEFAULT_STABLE_STD_PX = 2.0

WINDOW_NAME = "Penultimate Homography Calibration"
MASK_WINDOW_NAME = "Penultimate Homography Masks"

TARGET_LABELS_4 = ("TOP LEFT", "TOP RIGHT", "BOTTOM RIGHT", "BOTTOM LEFT")
TARGET_LABELS_6 = ("TOP LEFT", "TOP MIDDLE", "TOP RIGHT", "BOTTOM LEFT", "BOTTOM MIDDLE", "BOTTOM RIGHT")
TARGET_LABELS_9 = (
    "TOP LEFT",
    "TOP MIDDLE",
    "TOP RIGHT",
    "MIDDLE LEFT",
    "CENTER",
    "MIDDLE RIGHT",
    "BOTTOM LEFT",
    "BOTTOM MIDDLE",
    "BOTTOM RIGHT",
)
TARGET_LABEL_SETS = {
    "4": TARGET_LABELS_4,
    "6": TARGET_LABELS_6,
    "9": TARGET_LABELS_9,
}
DEFAULT_SKIP_CAMERA_0 = ("TOP LEFT",)
DEFAULT_SKIP_CAMERA_1 = ("TOP RIGHT",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate per-camera screen homographies from LED screen target touches."
    )
    parser.add_argument("--camera-0", type=int, default=DEFAULT_CAMERA_0)
    parser.add_argument("--camera-1", type=int, default=DEFAULT_CAMERA_1)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fov-mode", choices=FOV_MODES, default="full")
    parser.add_argument("--color-order", choices=COLOR_ORDERS, default="rgb")
    parser.add_argument("--screen-width-px", type=int, required=True)
    parser.add_argument("--screen-height-px", type=int, required=True)
    parser.add_argument(
        "--target-layout",
        choices=tuple(TARGET_LABEL_SETS.keys()),
        default="9",
        help="Number/layout of screen calibration targets. 9 is the robust default.",
    )
    parser.add_argument(
        "--skip-camera-0-targets",
        default=",".join(DEFAULT_SKIP_CAMERA_0),
        help="Comma-separated target labels to ignore for camera 0 / left camera.",
    )
    parser.add_argument(
        "--skip-camera-1-targets",
        default=",".join(DEFAULT_SKIP_CAMERA_1),
        help="Comma-separated target labels to ignore for camera 1 / right camera.",
    )
    parser.add_argument(
        "--target-image",
        type=Path,
        default=None,
        help="Optional PNG path for a screen-sized calibration target image.",
    )
    parser.add_argument(
        "--write-target-only",
        action="store_true",
        help="Write --target-image and exit without opening cameras.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_HOMOGRAPHY_CALIBRATION)
    parser.add_argument("--min-area", type=float, default=DEFAULT_MIN_AREA)
    parser.add_argument("--hsv-lower", default="100,150,180", help="Lower HSV blue halo threshold as H,S,V.")
    parser.add_argument("--hsv-upper", default="130,255,255", help="Upper HSV blue halo threshold as H,S,V.")
    parser.add_argument("--samples-per-target", type=int, default=DEFAULT_SAMPLES_PER_TARGET)
    parser.add_argument(
        "--samples-per-corner",
        type=int,
        default=None,
        help="Backward-compatible alias for --samples-per-target.",
    )
    parser.add_argument("--stable-window", type=int, default=DEFAULT_STABLE_WINDOW)
    parser.add_argument("--stable-std-px", type=float, default=DEFAULT_STABLE_STD_PX)
    parser.add_argument(
        "--max-candidate-area",
        type=float,
        default=None,
        help="Optional per-camera rejection threshold for oversized bloomed LED blobs.",
    )
    parser.add_argument("--show-mask", action="store_true")
    args = parser.parse_args()
    if args.samples_per_corner is not None:
        args.samples_per_target = args.samples_per_corner
    if args.samples_per_target <= 0:
        parser.error("--samples-per-target must be positive.")
    if args.write_target_only and args.target_image is None:
        parser.error("--write-target-only requires --target-image.")
    labels = TARGET_LABEL_SETS[args.target_layout]
    try:
        args.skip_camera_0_targets = parse_target_list(args.skip_camera_0_targets, labels)
        args.skip_camera_1_targets = parse_target_list(args.skip_camera_1_targets, labels)
    except ValueError as error:
        parser.error(str(error))
    if len(labels) - len(args.skip_camera_0_targets) < 4:
        parser.error("Camera 0 must have at least 4 non-skipped targets.")
    if len(labels) - len(args.skip_camera_1_targets) < 4:
        parser.error("Camera 1 must have at least 4 non-skipped targets.")
    return args


def parse_target_list(value: str, valid_labels: tuple[str, ...]) -> set[str]:
    if not value.strip():
        return set()
    labels = {part.strip().upper().replace("_", " ") for part in value.split(",") if part.strip()}
    invalid = sorted(labels - set(valid_labels))
    if invalid:
        raise ValueError(f"unknown targets: {', '.join(invalid)}")
    return labels


def best_candidate(candidates: list[LedCandidate]) -> LedCandidate | None:
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: (candidate.peak_brightness, candidate.area))


def usable_candidate(candidate: LedCandidate | None, max_area: float | None) -> LedCandidate | None:
    if candidate is None:
        return None
    if max_area is not None and candidate.area > max_area:
        return None
    return candidate


def stable(points: deque[np.ndarray], std_px: float) -> bool:
    if len(points) < points.maxlen:
        return False
    values = np.vstack(points)
    return bool(np.max(np.std(values, axis=0)) <= std_px)


def target_screen_points(
    labels: tuple[str, ...],
    screen_width_px: int,
    screen_height_px: int,
) -> np.ndarray:
    x0 = 0.0
    x1 = float(screen_width_px - 1)
    xm = x1 / 2.0
    y0 = 0.0
    y1 = float(screen_height_px - 1)
    ym = y1 / 2.0
    points = {
        "TOP LEFT": (x0, y0),
        "TOP MIDDLE": (xm, y0),
        "TOP RIGHT": (x1, y0),
        "MIDDLE LEFT": (x0, ym),
        "CENTER": (xm, ym),
        "MIDDLE RIGHT": (x1, ym),
        "BOTTOM LEFT": (x0, y1),
        "BOTTOM MIDDLE": (xm, y1),
        "BOTTOM RIGHT": (x1, y1),
    }
    return np.array([points[label] for label in labels], dtype=np.float32)


def write_target_image(path: Path, width: int, height: int, labels: tuple[str, ...], points: np.ndarray) -> None:
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    for index, (label, point) in enumerate(zip(labels, points), start=1):
        center = (int(round(float(point[0]))), int(round(float(point[1]))))
        cv2.circle(image, center, 32, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.drawMarker(
            image,
            center,
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=54,
            thickness=3,
        )
        text = f"{index}. {label}"
        text_x = min(max(center[0] + 24, 12), width - 260)
        text_y = min(max(center[1] - 24, 32), height - 16)
        cv2.putText(image, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def draw_candidate(frame: np.ndarray, candidate: LedCandidate | None, label: str, skipped: bool) -> np.ndarray:
    output = frame.copy()
    status = "SKIPPED" if skipped else label
    color = (128, 128, 128) if skipped else (255, 255, 255)
    cv2.putText(output, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    if candidate is not None and not skipped:
        center = (int(round(candidate.x)), int(round(candidate.y)))
        cv2.circle(output, center, 18, (0, 255, 0), 2)
        cv2.putText(
            output,
            f"x={candidate.x:.1f} y={candidate.y:.1f} area={candidate.area:.0f}",
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
    target_label: str,
    target_index: int,
    total_targets: int,
    screen_point: np.ndarray,
    status: str,
) -> np.ndarray:
    output = combined.copy()
    lines = [
        f"STEP {target_index + 1} OF {total_targets}",
        f"TOUCH {target_label}",
        f"SCREEN POINT ({screen_point[0]:.0f}, {screen_point[1]:.0f})",
        status,
        "SPACE capture when ready | BACKSPACE redo previous | R restart | Q quit",
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


def append_if_enabled(
    enabled: bool,
    samples: list[np.ndarray],
    candidate: LedCandidate | None,
    max_samples: int,
) -> None:
    if enabled and candidate is not None:
        samples.append(candidate.image_point())
        if len(samples) > max_samples:
            del samples[:-max_samples]


def enough_samples(enabled: bool, samples: list[np.ndarray], required: int) -> bool:
    return not enabled or len(samples) >= required


def fit_and_save(
    args: argparse.Namespace,
    labels: tuple[str, ...],
    screen_points: np.ndarray,
    captured_0: list[np.ndarray | None],
    captured_1: list[np.ndarray | None],
) -> None:
    cam0_pairs = [(point, screen) for point, screen in zip(captured_0, screen_points) if point is not None]
    cam1_pairs = [(point, screen) for point, screen in zip(captured_1, screen_points) if point is not None]
    if len(cam0_pairs) < 4:
        raise RuntimeError(f"Camera 0 only has {len(cam0_pairs)} valid targets; need at least 4.")
    if len(cam1_pairs) < 4:
        raise RuntimeError(f"Camera 1 only has {len(cam1_pairs)} valid targets; need at least 4.")

    camera_0_points = np.vstack([pair[0] for pair in cam0_pairs]).astype(np.float32)
    camera_1_points = np.vstack([pair[0] for pair in cam1_pairs]).astype(np.float32)
    screen_points_0 = np.vstack([pair[1] for pair in cam0_pairs]).astype(np.float32)
    screen_points_1 = np.vstack([pair[1] for pair in cam1_pairs]).astype(np.float32)
    H0 = build_homography(camera_0_points, screen_points_0)
    H1 = build_homography(camera_1_points, screen_points_1)
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
    used_0 = [label for label, point in zip(labels, captured_0) if point is not None]
    used_1 = [label for label, point in zip(labels, captured_1) if point is not None]
    print(f"Saved homography calibration: {args.output}")
    print(f"Camera 0 used {len(used_0)} targets: {', '.join(used_0)}")
    print(f"Camera 1 used {len(used_1)} targets: {', '.join(used_1)}")


def main() -> None:
    args = parse_args()
    labels = TARGET_LABEL_SETS[args.target_layout]
    screen_points = target_screen_points(labels, args.screen_width_px, args.screen_height_px)
    lower = parse_hsv_threshold(args.hsv_lower, "--hsv-lower")
    upper = parse_hsv_threshold(args.hsv_upper, "--hsv-upper")
    detector = LedDetector(lower, upper, args.min_area)
    camera_0 = None
    camera_1 = None
    captured_0: list[np.ndarray | None] = []
    captured_1: list[np.ndarray | None] = []

    try:
        if args.target_image is not None:
            write_target_image(args.target_image, args.screen_width_px, args.screen_height_px, labels, screen_points)
            print(f"Wrote calibration target image: {args.target_image}")
            if args.write_target_only:
                return

        camera_0, camera_1, _, _ = open_dual_cameras(args)
        print("Touch the LED tip to each prompted screen target.")
        print(f"Calibration order: {', '.join(labels)}")
        print(f"Camera 0 / left skipped targets: {', '.join(sorted(args.skip_camera_0_targets)) or 'none'}")
        print(f"Camera 1 / right skipped targets: {', '.join(sorted(args.skip_camera_1_targets)) or 'none'}")

        target_index = 0
        while target_index < len(labels):
            label = labels[target_index]
            screen_point = screen_points[target_index]
            use_camera_0 = label not in args.skip_camera_0_targets
            use_camera_1 = label not in args.skip_camera_1_targets
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
                led_0 = usable_candidate(best_candidate(candidates_0), args.max_candidate_area)
                led_1 = usable_candidate(best_candidate(candidates_1), args.max_candidate_area)

                if use_camera_0 and led_0 is not None:
                    recent_0.append(led_0.image_point())
                if use_camera_1 and led_1 is not None:
                    recent_1.append(led_1.image_point())

                stable_0 = not use_camera_0 or stable(recent_0, args.stable_std_px)
                stable_1 = not use_camera_1 or stable(recent_1, args.stable_std_px)
                if stable_0 and stable_1:
                    append_if_enabled(use_camera_0, samples_0, led_0, args.samples_per_target)
                    append_if_enabled(use_camera_1, samples_1, led_1, args.samples_per_target)

                ready_0 = enough_samples(use_camera_0, samples_0, args.samples_per_target)
                ready_1 = enough_samples(use_camera_1, samples_1, args.samples_per_target)
                status = (
                    f"cam0={'SKIP' if not use_camera_0 else f'{len(samples_0)}/{args.samples_per_target}'} "
                    f"cam1={'SKIP' if not use_camera_1 else f'{len(samples_1)}/{args.samples_per_target}'} "
                    f"{'READY' if ready_0 and ready_1 else 'HOLD STILL'}"
                )
                preview_0 = draw_candidate(frame_0, led_0, "CAMERA 0", not use_camera_0)
                preview_1 = draw_candidate(frame_1, led_1, "CAMERA 1", not use_camera_1)
                combined = np.hstack((preview_0, preview_1))
                cv2.imshow(
                    WINDOW_NAME,
                    draw_instructions(combined, label, target_index, len(labels), screen_point, status),
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
                    target_index = 0
                    print("Restarted homography calibration.")
                    break
                if key == 8 and target_index > 0:
                    captured_0.pop()
                    captured_1.pop()
                    target_index -= 1
                    print(f"Redoing {labels[target_index]}.")
                    break
                if key == ord(" "):
                    if not ready_0 or not ready_1:
                        print("Not enough stable samples yet; keep holding still.")
                        continue
                    point_0 = None if not use_camera_0 else np.median(np.vstack(samples_0), axis=0)
                    point_1 = None if not use_camera_1 else np.median(np.vstack(samples_1), axis=0)
                    captured_0.append(point_0)
                    captured_1.append(point_1)
                    print(f"Captured {label}: cam0={point_0 if point_0 is not None else 'skipped'} cam1={point_1 if point_1 is not None else 'skipped'}")
                    target_index += 1
                    break

        fit_and_save(args, labels, screen_points, captured_0, captured_1)
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

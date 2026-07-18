from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from camera import (
    COLOR_ORDERS,
    FOV_MODES,
    CAMERA_FORMAT,
    capture_bgr_frame,
    configure_camera,
    ensure_frame_size,
    print_available_cameras,
    safe_camera_call,
    set_full_fov_crop,
)
from led_detection import (
    COORDINATE_PRINT_DELTA_PIXELS,
    DEFAULT_HSV_LOWER,
    DEFAULT_HSV_UPPER,
    MORPH_KERNEL_SIZE,
    SELECTION_STRATEGIES,
    LedCandidate,
    annotate_frame,
    coordinate_changed,
    create_led_mask,
    find_led_candidates,
    format_candidate,
    parse_hsv_threshold,
    select_physical_led,
)


DEFAULT_CAMERA_LEFT = 0
DEFAULT_CAMERA_RIGHT = 1
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_MIN_AREA = 30.0
DEFAULT_LEFT_STRATEGY = "rightmost"
DEFAULT_RIGHT_STRATEGY = "leftmost"
DEFAULT_PRINT_INTERVAL_SECONDS = 1.0

FRAME_WINDOW_NAME = "Dual Camera LED Tracking"
MASK_WINDOW_NAME = "Dual Camera LED Masks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realtime dual Raspberry Pi CSI camera LED detector."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=DEFAULT_CAMERA_LEFT,
        help=(
            "Backward-compatible alias for --camera-left. "
            "Ignored when --camera-left is supplied."
        ),
    )
    parser.add_argument("--camera-left", type=int, default=None)
    parser.add_argument("--camera-right", type=int, default=DEFAULT_CAMERA_RIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument(
        "--fov-mode",
        choices=FOV_MODES,
        default="full",
        help=(
            "Use 'full' to request the widest available sensor crop and "
            "downscale it to the preview size, or 'current' to keep "
            "Picamera2's automatic mode choice."
        ),
    )
    parser.add_argument("--min-area", type=float, default=DEFAULT_MIN_AREA)
    parser.add_argument(
        "--color-order",
        choices=COLOR_ORDERS,
        default="rgb",
        help=(
            "Channel order returned by Picamera2 capture_array. Use 'bgr' if "
            "the main preview shows blue skin or swapped red/blue colors."
        ),
    )
    parser.add_argument(
        "--hsv-lower",
        default=",".join(str(v) for v in DEFAULT_HSV_LOWER.tolist()),
        help="Lower HSV LED threshold as H,S,V.",
    )
    parser.add_argument(
        "--hsv-upper",
        default=",".join(str(v) for v in DEFAULT_HSV_UPPER.tolist()),
        help="Upper HSV LED threshold as H,S,V.",
    )
    parser.add_argument(
        "--left-strategy",
        choices=SELECTION_STRATEGIES,
        default=DEFAULT_LEFT_STRATEGY,
        help=(
            "Candidate selection strategy for camera 0 / left preview. "
            "Defaults to rightmost for the current dual-camera glare layout."
        ),
    )
    parser.add_argument(
        "--right-strategy",
        choices=SELECTION_STRATEGIES,
        default=DEFAULT_RIGHT_STRATEGY,
        help=(
            "Candidate selection strategy for camera 1 / right preview. "
            "Defaults to leftmost for the current dual-camera glare layout."
        ),
    )
    parser.add_argument(
        "--show-mask",
        action="store_true",
        help="Show both binary LED masks in a second OpenCV window.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run detection without opening OpenCV display windows.",
    )
    parser.add_argument(
        "--print-interval",
        type=float,
        default=DEFAULT_PRINT_INTERVAL_SECONDS,
        help="Maximum seconds between headless coordinate updates.",
    )
    args = parser.parse_args()
    if args.camera_left is None:
        args.camera_left = args.camera
    args.hsv_lower = parse_hsv_threshold(args.hsv_lower, "--hsv-lower")
    args.hsv_upper = parse_hsv_threshold(args.hsv_upper, "--hsv-upper")
    return args


def should_print_coordinates(
    previous_left: LedCandidate | None,
    previous_right: LedCandidate | None,
    current_left: LedCandidate | None,
    current_right: LedCandidate | None,
    last_print_time: float,
    print_interval: float,
) -> bool:
    if coordinate_changed(previous_left, current_left, COORDINATE_PRINT_DELTA_PIXELS):
        return True
    if coordinate_changed(previous_right, current_right, COORDINATE_PRINT_DELTA_PIXELS):
        return True
    return time.perf_counter() - last_print_time >= print_interval


def run_detection(args: argparse.Namespace) -> None:
    camera_info = print_available_cameras()
    if len(camera_info) < 2:
        raise RuntimeError(
            f"Fewer than two cameras detected: found {len(camera_info)}."
        )
    if args.camera_left == args.camera_right:
        raise RuntimeError("Camera indexes must be different for dual-camera preview.")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)
    camera_left = None
    camera_right = None
    last_time = time.perf_counter()
    last_print_time = 0.0
    fps = 0.0
    last_printed_left: LedCandidate | None = None
    last_printed_right: LedCandidate | None = None

    print(
        "Starting dual realtime LED detector: "
        f"camera_left={args.camera_left}, camera_right={args.camera_right}, "
        f"resolution={args.width}x{args.height}, format={CAMERA_FORMAT}, "
        f"fov_mode={args.fov_mode}, "
        f"min_area={args.min_area}, left_strategy={args.left_strategy}, "
        f"right_strategy={args.right_strategy}, show_mask={args.show_mask}, "
        f"headless={args.headless}, color_order={args.color_order}, "
        f"hsv_lower={args.hsv_lower.tolist()}, hsv_upper={args.hsv_upper.tolist()}"
    )

    try:
        camera_left, full_fov_crop_left = configure_camera(
            args.camera_left,
            args.width,
            args.height,
            args.fov_mode,
        )
        camera_right, full_fov_crop_right = configure_camera(
            args.camera_right,
            args.width,
            args.height,
            args.fov_mode,
        )

        camera_left.start()
        camera_right.start()
        set_full_fov_crop(camera_left, args.camera_left, full_fov_crop_left)
        set_full_fov_crop(camera_right, args.camera_right, full_fov_crop_right)
        time.sleep(1.0)
        print(
            "Cameras started. Sequential capture is used for this preview, "
            "so the two frames are not yet hardware synchronized."
        )
        print("Press Q to quit the preview, or Ctrl+C to stop.")

        while True:
            try:
                frame_left = capture_bgr_frame(
                    camera_left,
                    "Camera 0",
                    args.color_order,
                )
                frame_right = capture_bgr_frame(
                    camera_right,
                    "Camera 1",
                    args.color_order,
                )
            except RuntimeError as error:
                print(f"Capture warning: {error}")
                time.sleep(0.05)
                continue

            if frame_left.shape[:2] != (args.height, args.width):
                print(
                    "Camera 0 returned a frame with a different size; "
                    f"resizing from {frame_left.shape[1]}x{frame_left.shape[0]} "
                    f"to {args.width}x{args.height}."
                )
                frame_left = ensure_frame_size(frame_left, args.width, args.height)
            if frame_right.shape[:2] != (args.height, args.width):
                print(
                    "Camera 1 returned a frame with a different size; "
                    f"resizing from {frame_right.shape[1]}x{frame_right.shape[0]} "
                    f"to {args.width}x{args.height}."
                )
                frame_right = ensure_frame_size(frame_right, args.width, args.height)

            mask_left = create_led_mask(
                frame_left, kernel, args.hsv_lower, args.hsv_upper
            )
            mask_right = create_led_mask(
                frame_right, kernel, args.hsv_lower, args.hsv_upper
            )
            candidates_left = find_led_candidates(mask_left, args.min_area)
            candidates_right = find_led_candidates(mask_right, args.min_area)
            selected_left = select_physical_led(candidates_left, args.left_strategy)
            selected_right = select_physical_led(candidates_right, args.right_strategy)

            now = time.perf_counter()
            elapsed = now - last_time
            last_time = now
            if elapsed > 0:
                instant_fps = 1.0 / elapsed
                fps = instant_fps if fps == 0 else (0.9 * fps) + (0.1 * instant_fps)

            if should_print_coordinates(
                last_printed_left,
                last_printed_right,
                selected_left,
                selected_right,
                last_print_time,
                args.print_interval,
            ):
                print(
                    f"Camera 0: {format_candidate(selected_left)} | "
                    f"Camera 1: {format_candidate(selected_right)}"
                )
                last_print_time = now
                last_printed_left = selected_left
                last_printed_right = selected_right

            if args.headless:
                continue

            annotated_left = annotate_frame(
                frame_left, candidates_left, selected_left, "CAMERA 0", fps
            )
            annotated_right = annotate_frame(
                frame_right, candidates_right, selected_right, "CAMERA 1", fps
            )
            combined_frame = np.hstack((annotated_left, annotated_right))
            cv2.imshow(FRAME_WINDOW_NAME, combined_frame)

            if args.show_mask:
                combined_mask = np.hstack(
                    (
                        ensure_frame_size(mask_left, args.width, args.height),
                        ensure_frame_size(mask_right, args.width, args.height),
                    )
                )
                cv2.imshow(MASK_WINDOW_NAME, combined_mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                break
    finally:
        safe_camera_call(camera_left, "stop", "camera 0")
        safe_camera_call(camera_right, "stop", "camera 1")
        safe_camera_call(camera_left, "close", "camera 0")
        safe_camera_call(camera_right, "close", "camera 1")
        cv2.destroyAllWindows()
        print("Dual realtime LED detector shut down cleanly.")


def main() -> None:
    args = parse_args()
    try:
        run_detection(args)
    except KeyboardInterrupt:
        print("Interrupted by Ctrl+C.")
    except Exception as error:
        print(f"Error: {error}")
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
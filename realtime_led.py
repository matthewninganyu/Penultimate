from __future__ import annotations

import argparse
import time

import cv2

from camera import (
    capture_bgr_frame,
    configure_camera,
    ensure_frame_size,
    safe_camera_call,
    set_full_fov_crop,
)
from blue_led_detection import (
    COORDINATE_PRINT_DELTA_PIXELS,
    DEFAULT_HSV_LOWER,
    DEFAULT_HSV_UPPER,
    MORPH_KERNEL_SIZE,
    LedCandidate,
    annotate_frame,
    coordinate_changed,
    create_led_mask,
    find_led_candidates,
    select_physical_led,
)


DEFAULT_CAMERA_INDEX = 0
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_MIN_AREA = 30.0

FRAME_WINDOW_NAME = "Realtime LED Detection"
MASK_WINDOW_NAME = "LED Mask"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Realtime Raspberry Pi CSI camera LED detector."
    )
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--min-area", type=float, default=DEFAULT_MIN_AREA)
    parser.add_argument(
        "--show-mask",
        action="store_true",
        help="Show the binary LED mask in a second OpenCV window.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run detection without opening OpenCV display windows.",
    )
    return parser.parse_args()


def run_detection(args: argparse.Namespace) -> None:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)
    camera = None
    last_time = time.perf_counter()
    fps = 0.0
    last_printed_candidate: LedCandidate | None = None

    print(
        "Starting realtime LED detector: "
        f"camera={args.camera}, resolution={args.width}x{args.height}, "
        f"min_area={args.min_area}, show_mask={args.show_mask}, "
        f"headless={args.headless}"
    )

    try:
        camera, full_fov_crop = configure_camera(args.camera, args.width, args.height)
        camera.start()
        set_full_fov_crop(camera, args.camera, full_fov_crop)
        time.sleep(1.0)
        print("Camera started. Press Q to quit the preview, or Ctrl+C to stop.")

        while True:
            try:
                frame = capture_bgr_frame(camera, "Camera", "rgb")
            except RuntimeError as error:
                print(f"Capture warning: {error}")
                time.sleep(0.05)
                continue

            if frame.shape[:2] != (args.height, args.width):
                frame = ensure_frame_size(frame, args.width, args.height)

            mask = create_led_mask(frame, kernel, DEFAULT_HSV_LOWER, DEFAULT_HSV_UPPER)
            candidates = find_led_candidates(mask, args.min_area)
            selected = select_physical_led(candidates, "rightmost")

            now = time.perf_counter()
            elapsed = now - last_time
            last_time = now
            if elapsed > 0:
                instant_fps = 1.0 / elapsed
                fps = instant_fps if fps == 0 else (0.9 * fps) + (0.1 * instant_fps)

            if coordinate_changed(
                last_printed_candidate, selected, COORDINATE_PRINT_DELTA_PIXELS
            ):
                if selected is None:
                    print("Selected LED: none")
                else:
                    print(f"Selected LED: x={selected.x}, y={selected.y}")
                last_printed_candidate = selected

            if args.headless:
                continue

            annotated = annotate_frame(frame, candidates, selected, "CAMERA", fps)
            cv2.imshow(FRAME_WINDOW_NAME, annotated)
            if args.show_mask:
                cv2.imshow(MASK_WINDOW_NAME, mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                break
    finally:
        safe_camera_call(camera, "stop", "camera")
        safe_camera_call(camera, "close", "camera")
        cv2.destroyAllWindows()
        print("Realtime LED detector shut down cleanly.")


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

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

try:
    from picamera2 import Picamera2
except ImportError:  # pragma: no cover - exercised on non-Raspberry Pi systems.
    Picamera2 = None  # type: ignore[assignment]


DEFAULT_CAMERA_INDEX = 0
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_MIN_AREA = 30.0

LOWER_LED = np.array([10, 150, 220], dtype=np.uint8)
UPPER_LED = np.array([40, 255, 255], dtype=np.uint8)

MORPH_KERNEL_SIZE = (5, 5)
MORPH_OPEN_ITERATIONS = 1
MORPH_CLOSE_ITERATIONS = 2

SELECTED_RADIUS = 24
CENTROID_RADIUS = 4
COORDINATE_PRINT_DELTA_PIXELS = 8

FRAME_WINDOW_NAME = "Realtime LED Detection"
MASK_WINDOW_NAME = "LED Mask"

YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)


@dataclass(frozen=True)
class LedCandidate:
    x: int
    y: int
    area: float
    circularity: float
    contour: np.ndarray


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


def configure_camera(camera_index: int, width: int, height: int) -> Any:
    if Picamera2 is None:
        raise RuntimeError(
            "Picamera2 is not installed. On Raspberry Pi OS, run: "
            "sudo apt install -y python3-picamera2 python3-opencv python3-venv"
        )

    camera = Picamera2(camera_index)
    video_config = camera.create_video_configuration(
        main={"size": (width, height), "format": "BGR888"}
    )
    camera.configure(video_config)
    return camera


def create_led_mask(frame: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_LED, UPPER_LED)
    opened = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=MORPH_OPEN_ITERATIONS,
    )
    return cv2.morphologyEx(
        opened,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=MORPH_CLOSE_ITERATIONS,
    )


def calculate_circularity(contour: np.ndarray, area: float) -> float:
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0.0
    return float((4.0 * math.pi * area) / (perimeter * perimeter))


def find_led_candidates(mask: np.ndarray, min_area: float) -> list[LedCandidate]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[LedCandidate] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue

        x = int(round(moments["m10"] / moments["m00"]))
        y = int(round(moments["m01"] / moments["m00"]))
        candidates.append(
            LedCandidate(
                x=x,
                y=y,
                area=area,
                circularity=calculate_circularity(contour, area),
                contour=contour,
            )
        )

    return candidates


def select_physical_led(candidates: list[LedCandidate]) -> LedCandidate | None:
    """Select the likely physical LED using a temporary orientation heuristic.

    For this prototype, the selected candidate is the rightmost valid LED blob.
    This depends on the current camera orientation and reflection layout. It
    should eventually be replaced by stereo-camera geometry and the calibrated
    laptop-screen plane.
    """
    if not candidates:
        return None
    return max(candidates, key=lambda candidate: candidate.x)


def annotate_frame(
    frame: np.ndarray,
    candidates: list[LedCandidate],
    selected: LedCandidate | None,
    fps: float,
) -> np.ndarray:
    annotated = frame.copy()

    cv2.putText(
        annotated,
        f"FPS: {fps:.1f}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        WHITE,
        2,
        cv2.LINE_AA,
    )

    if not candidates:
        cv2.putText(
            annotated,
            "NO LED DETECTED",
            (12, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            RED,
            2,
            cv2.LINE_AA,
        )
        return annotated

    for candidate in candidates:
        center = (candidate.x, candidate.y)
        cv2.drawContours(annotated, [candidate.contour], -1, YELLOW, 2)
        cv2.circle(annotated, center, CENTROID_RADIUS, YELLOW, -1)

    if selected is not None:
        selected_center = (selected.x, selected.y)
        cv2.circle(annotated, selected_center, SELECTED_RADIUS, GREEN, 3)
        cv2.putText(
            annotated,
            "SELECTED LED",
            (selected.x + 12, selected.y - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            GREEN,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"x={selected.x} y={selected.y}",
            (selected.x + 12, selected.y + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            GREEN,
            2,
            cv2.LINE_AA,
        )

    return annotated


def coordinate_changed(
    previous: LedCandidate | None,
    current: LedCandidate | None,
    min_delta: int,
) -> bool:
    if current is None:
        return previous is not None
    if previous is None:
        return True

    return (
        abs(current.x - previous.x) >= min_delta
        or abs(current.y - previous.y) >= min_delta
    )


def run_detection(args: argparse.Namespace) -> None:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)
    camera = configure_camera(args.camera, args.width, args.height)
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
        camera.start()
        time.sleep(1.0)
        print("Camera started. Press Q to quit the preview, or Ctrl+C to stop.")

        while True:
            frame = camera.capture_array()
            mask = create_led_mask(frame, kernel)
            candidates = find_led_candidates(mask, args.min_area)
            selected = select_physical_led(candidates)

            now = time.perf_counter()
            elapsed = now - last_time
            last_time = now
            if elapsed > 0:
                instant_fps = 1.0 / elapsed
                fps = instant_fps if fps == 0 else (0.9 * fps) + (0.1 * instant_fps)

            if coordinate_changed(
                last_printed_candidate,
                selected,
                COORDINATE_PRINT_DELTA_PIXELS,
            ):
                if selected is None:
                    print("Selected LED: none")
                else:
                    print(f"Selected LED: x={selected.x}, y={selected.y}")
                last_printed_candidate = selected

            if args.headless:
                continue

            annotated = annotate_frame(frame, candidates, selected, fps)
            cv2.imshow(FRAME_WINDOW_NAME, annotated)
            if args.show_mask:
                cv2.imshow(MASK_WINDOW_NAME, mask)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                break
    finally:
        camera.stop()
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

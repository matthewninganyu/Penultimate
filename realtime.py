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


DEFAULT_CAMERA_LEFT = 0
DEFAULT_CAMERA_RIGHT = 1
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_MIN_AREA = 30.0
DEFAULT_LEFT_STRATEGY = "rightmost"
DEFAULT_RIGHT_STRATEGY = "leftmost"
DEFAULT_PRINT_INTERVAL_SECONDS = 1.0

CAMERA_FORMAT = "RGB888"
SELECTION_STRATEGIES = ("rightmost", "leftmost", "largest")
COLOR_ORDERS = ("rgb", "bgr")
FOV_MODES = ("full", "current")

LOWER_LED = np.array([10, 150, 220], dtype=np.uint8)
UPPER_LED = np.array([40, 255, 255], dtype=np.uint8)

MORPH_KERNEL_SIZE = (5, 5)
MORPH_OPEN_ITERATIONS = 1
MORPH_CLOSE_ITERATIONS = 2

SELECTED_RADIUS = 24
CENTROID_RADIUS = 4
COORDINATE_PRINT_DELTA_PIXELS = 8

FRAME_WINDOW_NAME = "Dual Camera LED Tracking"
MASK_WINDOW_NAME = "Dual Camera LED Masks"

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
        default="10,150,220",
        help="Lower HSV LED threshold as H,S,V.",
    )
    parser.add_argument(
        "--hsv-upper",
        default="40,255,255",
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


def parse_hsv_threshold(value: str, argument_name: str) -> np.ndarray:
    try:
        parts = [int(part.strip()) for part in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{argument_name} must contain three integers like 10,150,220."
        ) from error

    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            f"{argument_name} must contain exactly three values: H,S,V."
        )

    hue, saturation, value_channel = parts
    if not 0 <= hue <= 179:
        raise argparse.ArgumentTypeError(f"{argument_name} hue must be 0..179.")
    if not 0 <= saturation <= 255 or not 0 <= value_channel <= 255:
        raise argparse.ArgumentTypeError(
            f"{argument_name} saturation/value must be 0..255."
        )

    return np.array(parts, dtype=np.uint8)


def ensure_picamera2_available() -> None:
    if Picamera2 is None:
        raise RuntimeError(
            "Picamera2 is not installed. On Raspberry Pi OS, run: "
            "sudo apt install -y python3-picamera2 python3-opencv python3-venv"
        )


def print_available_cameras() -> list[dict[str, Any]]:
    ensure_picamera2_available()
    camera_info = Picamera2.global_camera_info()
    print("Available Picamera2 cameras:")
    if not camera_info:
        print("  none")
    for index, info in enumerate(camera_info):
        print(f"  [{index}] {info}")
    return camera_info


def crop_limits_to_tuple(crop_limits: Any) -> tuple[int, int, int, int] | None:
    if crop_limits is None:
        return None

    if all(hasattr(crop_limits, attr) for attr in ("x", "y", "width", "height")):
        return (
            int(crop_limits.x),
            int(crop_limits.y),
            int(crop_limits.width),
            int(crop_limits.height),
        )

    try:
        x_offset, y_offset, width, height = crop_limits
    except (TypeError, ValueError):
        return None

    return (int(x_offset), int(y_offset), int(width), int(height))


def crop_area(crop_limits: tuple[int, int, int, int]) -> int:
    return crop_limits[2] * crop_limits[3]


def find_largest_fov_sensor_mode(
    camera: Any,
) -> tuple[int, dict[str, Any], tuple[int, int, int, int]] | None:
    sensor_modes = getattr(camera, "sensor_modes", None)
    if not sensor_modes:
        return None

    largest_mode: tuple[int, dict[str, Any], tuple[int, int, int, int]] | None = None
    for mode_index, sensor_mode in enumerate(sensor_modes):
        crop_limits = crop_limits_to_tuple(sensor_mode.get("crop_limits"))
        if crop_limits is None:
            continue
        if largest_mode is None or crop_area(crop_limits) > crop_area(largest_mode[2]):
            largest_mode = (mode_index, sensor_mode, crop_limits)

    return largest_mode


def format_sensor_mode(sensor_mode: dict[str, Any]) -> str:
    details = []
    for key in ("size", "format", "bit_depth", "fps", "crop_limits"):
        if key in sensor_mode:
            details.append(f"{key}={sensor_mode[key]}")
    return ", ".join(details) if details else str(sensor_mode)


def configure_camera(
    camera_index: int,
    width: int,
    height: int,
    fov_mode: str,
) -> tuple[Any, tuple[int, int, int, int] | None]:
    ensure_picamera2_available()

    try:
        camera = Picamera2(camera_index)
    except Exception as error:
        raise RuntimeError(f"Camera {camera_index} cannot be opened: {error}") from error

    full_fov_crop = None
    selected_mode = None
    if fov_mode == "full":
        selected_mode = find_largest_fov_sensor_mode(camera)
        if selected_mode is None:
            print(
                f"Camera {camera_index}: no sensor mode with crop_limits was found; "
                "falling back to Picamera2's automatic mode choice."
            )
        else:
            mode_index, sensor_mode, full_fov_crop = selected_mode
            print(
                f"Camera {camera_index}: selected full-FOV sensor mode "
                f"[{mode_index}] ({format_sensor_mode(sensor_mode)}), "
                f"requested preview={width}x{height}."
            )

    main = {"size": (width, height), "format": CAMERA_FORMAT}
    if selected_mode is not None:
        try:
            video_config = camera.create_video_configuration(
                main=main,
                raw=selected_mode[1],
            )
            camera.configure(video_config)
        except Exception as selected_error:
            print(
                f"Camera {camera_index}: could not configure the selected "
                f"full-FOV raw sensor mode ({selected_error}); using "
                "Picamera2's automatic mode choice."
            )
            full_fov_crop = None
            try:
                video_config = camera.create_video_configuration(main=main)
                camera.configure(video_config)
            except Exception as fallback_error:
                raise RuntimeError(
                    f"Camera {camera_index} cannot be configured: {fallback_error}"
                ) from fallback_error
    else:
        try:
            video_config = camera.create_video_configuration(main=main)
            camera.configure(video_config)
        except Exception as error:
            raise RuntimeError(
                f"Camera {camera_index} cannot be configured: {error}"
            ) from error

    return camera, full_fov_crop


def set_full_fov_crop(
    camera: Any,
    camera_index: int,
    full_fov_crop: tuple[int, int, int, int] | None,
) -> None:
    if full_fov_crop is None:
        return

    try:
        camera.set_controls({"ScalerCrop": full_fov_crop})
    except Exception as error:
        print(
            f"Warning: Camera {camera_index} could not set full-FOV "
            f"ScalerCrop={full_fov_crop}: {error}"
        )
        return

    active_crop = None
    try:
        metadata = camera.capture_metadata()
        active_crop = metadata.get("ScalerCrop")
    except Exception as error:
        print(
            f"Warning: Camera {camera_index} could not read active ScalerCrop: {error}"
        )

    if active_crop is None:
        print(f"Camera {camera_index}: requested ScalerCrop={full_fov_crop}.")
    else:
        print(
            f"Camera {camera_index}: requested ScalerCrop={full_fov_crop}, "
            f"active ScalerCrop={active_crop}."
        )


def capture_bgr_frame(
    camera: Any,
    camera_label: str,
    color_order: str,
) -> np.ndarray:
    try:
        frame = camera.capture_array()
    except Exception as error:
        raise RuntimeError(f"{camera_label} failed while capturing: {error}") from error

    if frame is None:
        raise RuntimeError(f"{camera_label} returned an empty frame.")

    if frame.ndim != 3 or frame.shape[2] < 3:
        raise RuntimeError(
            f"{camera_label} returned an unsupported frame shape: {frame.shape}"
        )

    frame = frame[:, :, :3]
    if color_order == "rgb":
        # Picamera2 is configured for RGB888, while the existing OpenCV
        # detection and annotation pipeline expects BGR channel order.
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if color_order == "bgr":
        # Some camera/libcamera combinations can already produce BGR-like
        # arrays. Use this when the preview has swapped red/blue colors.
        return frame
    raise ValueError(f"Unsupported color order: {color_order}")


def ensure_frame_size(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def create_led_mask(
    frame: np.ndarray,
    kernel: np.ndarray,
    hsv_lower: np.ndarray,
    hsv_upper: np.ndarray,
) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
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


def select_physical_led(
    candidates: list[LedCandidate],
    strategy: str,
) -> LedCandidate | None:
    """Select the likely physical LED using a camera-specific heuristic.

    In the current dual-camera layout, one camera should use the rightmost
    candidate and the other should use the leftmost candidate. The opposite
    blob is treated as screen glare until calibrated stereo geometry replaces
    this orientation rule.
    """
    if not candidates:
        return None

    if strategy == "rightmost":
        return max(candidates, key=lambda candidate: candidate.x)
    if strategy == "leftmost":
        return min(candidates, key=lambda candidate: candidate.x)
    if strategy == "largest":
        return max(candidates, key=lambda candidate: candidate.area)

    raise ValueError(f"Unsupported LED selection strategy: {strategy}")


def annotate_frame(
    frame: np.ndarray,
    candidates: list[LedCandidate],
    selected: LedCandidate | None,
    camera_label: str,
    fps: float,
) -> np.ndarray:
    annotated = frame.copy()

    cv2.putText(
        annotated,
        camera_label,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        WHITE,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        f"FPS: {fps:.1f}",
        (12, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        WHITE,
        2,
        cv2.LINE_AA,
    )

    if not candidates:
        cv2.putText(
            annotated,
            "NO LED DETECTED",
            (12, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
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


def should_print_coordinates(
    previous_left: LedCandidate | None,
    previous_right: LedCandidate | None,
    current_left: LedCandidate | None,
    current_right: LedCandidate | None,
    last_print_time: float,
    print_interval: float,
) -> bool:
    if coordinate_changed(
        previous_left,
        current_left,
        COORDINATE_PRINT_DELTA_PIXELS,
    ):
        return True
    if coordinate_changed(
        previous_right,
        current_right,
        COORDINATE_PRINT_DELTA_PIXELS,
    ):
        return True
    return time.perf_counter() - last_print_time >= print_interval


def format_candidate(candidate: LedCandidate | None) -> str:
    if candidate is None:
        return "none"
    return f"x={candidate.x}, y={candidate.y}"


def safe_camera_call(camera: Any | None, method_name: str, label: str) -> None:
    if camera is None:
        return
    try:
        getattr(camera, method_name)()
    except Exception as error:
        print(f"Warning: failed to {method_name} {label}: {error}")


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
                frame_left,
                kernel,
                args.hsv_lower,
                args.hsv_upper,
            )
            mask_right = create_led_mask(
                frame_right,
                kernel,
                args.hsv_lower,
                args.hsv_upper,
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
                frame_left,
                candidates_left,
                selected_left,
                "CAMERA 0",
                fps,
            )
            annotated_right = annotate_frame(
                frame_right,
                candidates_right,
                selected_right,
                "CAMERA 1",
                fps,
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

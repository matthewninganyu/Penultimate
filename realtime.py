from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path

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
from blue_led_detection import (
    COORDINATE_PRINT_DELTA_PIXELS,
    DEFAULT_HSV_LOWER,
    DEFAULT_HSV_UPPER,
    MORPH_KERNEL_SIZE,
    SELECTION_STRATEGIES,
    WHITE,
    LedCandidate,
    annotate_frame,
    candidates_from_mask_and_frame,
    coordinate_changed,
    create_led_mask,
    format_candidate,
    parse_hsv_threshold,
    select_physical_led,
)
from screen_mapper import (
    DEFAULT_HOMOGRAPHY_CALIBRATION,
    HomographyCalibration,
    load_homography_calibration,
    map_raw_coordinates,
)

DEFAULT_CAMERA_LEFT = 0
DEFAULT_CAMERA_RIGHT = 1
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_MIN_AREA = 30.0
DEFAULT_MIN_BRIGHTNESS = 160.0
BRIGHTNESS_STEP = 5.0
DEFAULT_LEFT_STRATEGY = "rightmost"
DEFAULT_RIGHT_STRATEGY = "leftmost"
DEFAULT_PRINT_INTERVAL_SECONDS = 1.0
RAW_OUTPUT_FORMATS = ("human", "json", "csv")
DEFAULT_BROADCAST_IP = "255.255.255.255"
DEFAULT_LAPTOP_PORT = 5005

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
        "--min-brightness",
        type=float,
        default=DEFAULT_MIN_BRIGHTNESS,
        help=(
            "Minimum peak brightness (0-255) a candidate must have to be "
            "considered. Adjust live with [ and ]."
        ),
    )
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
        default="100,150,180",
        help="Lower HSV blue halo threshold as H,S,V.",
    )
    parser.add_argument(
        "--hsv-upper",
        default="130,255,255",
        help="Upper HSV blue halo threshold as H,S,V.",
    )
    parser.add_argument(
        "--roi-start",
        type=float,
        default=0.45,
        help="Top of the detector ROI as a frame-height fraction.",
    )
    parser.add_argument(
        "--roi-end",
        type=float,
        default=0.85,
        help="Bottom of the detector ROI as a frame-height fraction.",
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
    parser.add_argument(
        "--raw-output",
        choices=RAW_OUTPUT_FORMATS,
        default="human",
        help=(
            "Format for selected raw camera LED coordinates printed to stdout. "
            "Use json or csv when feeding coordinates into another script."
        ),
    )
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="Skip screen homography mapping and output only raw camera coordinates.",
    )
    parser.add_argument(
        "--homography",
        type=Path,
        default=DEFAULT_HOMOGRAPHY_CALIBRATION,
        help="Saved calibration from calibrate_homography.py.",
    )
    parser.add_argument(
        "--send-udp",
        action="store_true",
        help="Send selected coordinates to the laptop over UDP.",
    )
    parser.add_argument(
        "--laptop-ip",
        default=DEFAULT_BROADCAST_IP,
        help=(
            "UDP target for --send-udp. Defaults to broadcast so the laptop "
            "can receive over USB/Ethernet without entering its IP."
        ),
    )
    parser.add_argument(
        "--laptop-port",
        type=int,
        default=DEFAULT_LAPTOP_PORT,
        help="Laptop UDP port for --send-udp.",
    )
    args = parser.parse_args()
    if args.camera_left is None:
        args.camera_left = args.camera
    args.hsv_lower = parse_hsv_threshold(args.hsv_lower, "--hsv-lower")
    args.hsv_upper = parse_hsv_threshold(args.hsv_upper, "--hsv-upper")
    if not 0.0 <= args.roi_start < args.roi_end <= 1.0:
        parser.error("--roi-start and --roi-end must satisfy 0 <= start < end <= 1.")
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


def candidate_payload(candidate: LedCandidate | None) -> dict[str, float] | None:
    if candidate is None:
        return None
    return {
        "x": float(candidate.x),
        "y": float(candidate.y),
        "area": float(candidate.area),
        "peak_brightness": float(candidate.peak_brightness),
    }


def raw_coordinate_packet(
    sequence: int,
    selected_left: LedCandidate | None,
    selected_right: LedCandidate | None,
) -> dict[str, object]:
    return {
        "type": "raw_coordinates",
        "sequence": sequence,
        "timestamp": time.time(),
        "valid": selected_left is not None and selected_right is not None,
        "camera_0": candidate_payload(selected_left),
        "camera_1": candidate_payload(selected_right),
    }


def point_from_candidate(candidate: LedCandidate | None) -> np.ndarray | None:
    if candidate is None:
        return None
    return candidate.image_point()


def coordinate_packet(
    sequence: int,
    selected_left: LedCandidate | None,
    selected_right: LedCandidate | None,
    homography: HomographyCalibration | None,
) -> dict[str, object]:
    raw_packet = raw_coordinate_packet(sequence, selected_left, selected_right)
    if homography is None:
        return raw_packet

    mapped = map_raw_coordinates(
        point_from_candidate(selected_left),
        point_from_candidate(selected_right),
        homography,
    )
    return {
        "type": "screen_coordinates",
        "sequence": sequence,
        "timestamp": raw_packet["timestamp"],
        "valid": bool(mapped["valid"]),
        "camera_0": raw_packet["camera_0"],
        "camera_1": raw_packet["camera_1"],
        "pixel_x": mapped["pixel_x"],
        "pixel_y": mapped["pixel_y"],
        "normalized_x": mapped["normalized_x"],
        "normalized_y": mapped["normalized_y"],
        "screen_0": mapped["screen_0"],
        "screen_1": mapped["screen_1"],
    }


def send_raw_udp_packet(
    udp_socket: socket.socket,
    address: tuple[str, int],
    packet: dict[str, object],
) -> None:
    payload = json.dumps(packet, separators=(",", ":"), allow_nan=False).encode("utf-8")
    try:
        udp_socket.sendto(payload, address)
    except OSError as error:
        print(f"UDP send warning: {error}")


def print_coordinate_packet(
    packet: dict[str, object],
    raw_output: str,
    csv_header_printed: bool,
) -> bool:
    if raw_output == "json":
        print(json.dumps(packet, separators=(",", ":")))
        return csv_header_printed

    if raw_output == "csv":
        if not csv_header_printed:
            print(
                "timestamp,camera_0_x,camera_0_y,camera_1_x,camera_1_y,"
                "valid,pixel_x,pixel_y,normalized_x,normalized_y"
            )
            csv_header_printed = True
        camera_0 = packet["camera_0"]
        camera_1 = packet["camera_1"]
        left_x = "" if camera_0 is None else f"{camera_0['x']:.3f}"
        left_y = "" if camera_0 is None else f"{camera_0['y']:.3f}"
        right_x = "" if camera_1 is None else f"{camera_1['x']:.3f}"
        right_y = "" if camera_1 is None else f"{camera_1['y']:.3f}"
        pixel_x = "" if packet.get("pixel_x") is None else str(packet["pixel_x"])
        pixel_y = "" if packet.get("pixel_y") is None else str(packet["pixel_y"])
        normalized_x = "" if packet.get("normalized_x") is None else f"{packet['normalized_x']:.6f}"
        normalized_y = "" if packet.get("normalized_y") is None else f"{packet['normalized_y']:.6f}"
        print(
            f"{packet['timestamp']:.6f},{left_x},{left_y},{right_x},{right_y},"
            f"{packet['valid']},{pixel_x},{pixel_y},{normalized_x},{normalized_y}"
        )
        return csv_header_printed

    camera_0 = packet["camera_0"]
    camera_1 = packet["camera_1"]
    raw_text = (
        f"Camera 0: {format_packet_camera_point(camera_0)} | "
        f"Camera 1: {format_packet_camera_point(camera_1)}"
    )
    if packet["type"] == "screen_coordinates":
        if packet["valid"]:
            print(
                f"Screen: pixel=({packet['pixel_x']}, {packet['pixel_y']}) "
                f"normalized=({packet['normalized_x']:.4f}, {packet['normalized_y']:.4f}) | "
                f"{raw_text}"
            )
        else:
            print(f"Screen: invalid | {raw_text}")
    else:
        print(raw_text)
    return csv_header_printed


def format_packet_camera_point(camera_point: object) -> str:
    if camera_point is None:
        return "none"
    point = camera_point
    return f"x={point['x']:.1f}, y={point['y']:.1f}"


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
    udp_socket = None
    udp_address = None
    last_time = time.perf_counter()
    last_print_time = 0.0
    sequence = 0
    fps = 0.0
    last_printed_left: LedCandidate | None = None
    last_printed_right: LedCandidate | None = None
    csv_header_printed = False
    min_brightness = args.min_brightness
    homography = None

    print(
        "Starting dual realtime LED detector: "
        f"camera_left={args.camera_left}, camera_right={args.camera_right}, "
        f"resolution={args.width}x{args.height}, format={CAMERA_FORMAT}, "
        f"fov_mode={args.fov_mode}, "
        f"min_area={args.min_area}, left_strategy={args.left_strategy}, "
        f"right_strategy={args.right_strategy}, show_mask={args.show_mask}, "
        f"headless={args.headless}, color_order={args.color_order}, "
        f"hsv_lower={args.hsv_lower.tolist()}, hsv_upper={args.hsv_upper.tolist()}, "
        f"roi_start={args.roi_start}, roi_end={args.roi_end}, "
        f"min_brightness={args.min_brightness}, "
        f"send_udp={args.send_udp}, laptop_ip={args.laptop_ip}, "
        f"laptop_port={args.laptop_port}"
    )

    try:
        if args.send_udp:
            udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            udp_address = (args.laptop_ip, args.laptop_port)
            print(
                f"Sending coordinate UDP packets to {udp_address[0]}:{udp_address[1]}."
            )

        if not args.raw_only:
            homography = load_homography_calibration(args.homography)
            if homography.image_width != args.width or homography.image_height != args.height:
                raise RuntimeError(
                    "Homography calibration resolution mismatch: "
                    f"calibration={homography.image_width}x{homography.image_height}, "
                    f"runtime={args.width}x{args.height}. Re-run calibrate_homography.py "
                    "or use matching --width/--height."
                )
            print(f"Loaded screen homography calibration: {args.homography}")

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
                args.roi_start,
                args.roi_end,
            )
            mask_right = create_led_mask(
                frame_right,
                kernel,
                args.hsv_lower,
                args.hsv_upper,
                args.roi_start,
                args.roi_end,
            )
            candidates_left = [
                c
                for c in candidates_from_mask_and_frame(
                    frame_left, mask_left, args.min_area
                )
                if c.peak_brightness >= min_brightness
            ]
            candidates_right = [
                c
                for c in candidates_from_mask_and_frame(
                    frame_right, mask_right, args.min_area
                )
                if c.peak_brightness >= min_brightness
            ]
            selected_left = select_physical_led(candidates_left, args.left_strategy)
            selected_right = select_physical_led(candidates_right, args.right_strategy)
            packet = coordinate_packet(sequence, selected_left, selected_right, homography)
            if udp_socket is not None and udp_address is not None:
                send_raw_udp_packet(udp_socket, udp_address, packet)
            sequence += 1

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
                csv_header_printed = print_coordinate_packet(
                    packet,
                    args.raw_output,
                    csv_header_printed,
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
            cv2.putText(
                combined_frame,
                f"Brightness >= {min_brightness:.0f}  [ / ] to adjust",
                (12, combined_frame.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                WHITE,
                1,
                cv2.LINE_AA,
            )
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
            if key == ord("["):
                min_brightness = max(0.0, min_brightness - BRIGHTNESS_STEP)
                print(f"Min brightness: {min_brightness:.0f}")
            if key == ord("]"):
                min_brightness = min(255.0, min_brightness + BRIGHTNESS_STEP)
                print(f"Min brightness: {min_brightness:.0f}")

    finally:
        safe_camera_call(camera_left, "stop", "camera 0")
        safe_camera_call(camera_right, "stop", "camera 1")
        safe_camera_call(camera_left, "close", "camera 0")
        safe_camera_call(camera_right, "close", "camera 1")
        if udp_socket is not None:
            udp_socket.close()
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

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np

from camera_manager import DualCameraManager, OfflineFrameSource
from config import (
    DEFAULT_CAMERA_0,
    DEFAULT_CAMERA_1,
    DEFAULT_HEIGHT,
    DEFAULT_MAX_FRAME_SKEW_MS,
    DEFAULT_MAX_JUMP_MM,
    DEFAULT_MAX_REPROJECTION_ERROR,
    DEFAULT_MIN_AREA,
    DEFAULT_SCREEN_CALIBRATION,
    DEFAULT_SCREEN_MARGIN_MM,
    DEFAULT_SMOOTHING_ALPHA,
    DEFAULT_TRACKING_CONFIDENCE_THRESHOLD,
    DEFAULT_WIDTH,
)
from contact_detection import ContactDetector
from led_detection import LedDetector, parse_hsv_triplet
from models import CameraFrame, ContactEvidence, LedCandidate, PenState, ScreenCalibration, ScreenPosition, StereoMatch
from network_sender import UdpPenSender
from screen_mapping import load_screen_calibration, point_to_screen_position, validate_runtime_resolution
from stereo_matching import choose_best_stereo_match
from tracking_filter import ExponentialPenFilter


LOGGER = logging.getLogger(__name__)
FRAME_WINDOW_NAME = "Penultimate Tracking"
MASK_WINDOW_NAME = "Penultimate LED Masks"
SNAPSHOT_DIR = Path("output")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Penultimate calibrated dual-camera pen tracker.")
    parser.add_argument("--camera-0", type=int, default=DEFAULT_CAMERA_0)
    parser.add_argument("--camera-1", type=int, default=DEFAULT_CAMERA_1)
    parser.add_argument("--camera-left", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--camera-right", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fov-mode", choices=("full", "current"), default="full")
    parser.add_argument("--color-order", choices=("rgb", "bgr"), default="rgb")
    parser.add_argument("--min-area", type=float, default=DEFAULT_MIN_AREA)
    parser.add_argument("--lower-h", type=int, default=10)
    parser.add_argument("--lower-s", type=int, default=150)
    parser.add_argument("--lower-v", type=int, default=220)
    parser.add_argument("--upper-h", type=int, default=40)
    parser.add_argument("--upper-s", type=int, default=255)
    parser.add_argument("--upper-v", type=int, default=255)
    parser.add_argument("--screen-calibration", type=Path, default=DEFAULT_SCREEN_CALIBRATION)
    parser.add_argument("--show-mask", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--send-udp", action="store_true")
    parser.add_argument("--laptop-ip", default="127.0.0.1")
    parser.add_argument("--laptop-port", type=int, default=5005)
    parser.add_argument("--tracking-confidence-threshold", type=float, default=DEFAULT_TRACKING_CONFIDENCE_THRESHOLD)
    parser.add_argument("--max-reprojection-error", type=float, default=DEFAULT_MAX_REPROJECTION_ERROR)
    parser.add_argument("--max-frame-skew-ms", type=float, default=DEFAULT_MAX_FRAME_SKEW_MS)
    parser.add_argument("--smoothing-alpha", type=float, default=DEFAULT_SMOOTHING_ALPHA)
    parser.add_argument("--max-jump-mm", type=float, default=DEFAULT_MAX_JUMP_MM)
    parser.add_argument("--screen-margin-mm", type=float, default=DEFAULT_SCREEN_MARGIN_MM)
    parser.add_argument("--preview-only", action="store_true", help="Run LED candidate preview without calibration.")
    parser.add_argument("--offline-left", type=Path, default=None)
    parser.add_argument("--offline-right", type=Path, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.camera_left is not None:
        args.camera_0 = args.camera_left
    if args.camera_right is not None:
        args.camera_1 = args.camera_right
    if (args.offline_left is None) != (args.offline_right is None):
        parser.error("--offline-left and --offline-right must be supplied together.")
    return args


def make_invalid_state(sequence: int, frame_skew_ms: float | None) -> PenState:
    return PenState(
        sequence=sequence,
        timestamp=time.time(),
        valid=False,
        normalized_x=None,
        normalized_y=None,
        pixel_x=None,
        pixel_y=None,
        x_mm=None,
        y_mm=None,
        distance_mm=None,
        touching=False,
        contact_confidence=0.0,
        tracking_confidence=0.0,
        frame_skew_ms=frame_skew_ms,
    )


def make_valid_state(
    sequence: int,
    position: ScreenPosition,
    contact: ContactEvidence,
    tracking_confidence: float,
    frame_skew_ms: float,
) -> PenState:
    return PenState(
        sequence=sequence,
        timestamp=time.time(),
        valid=True,
        normalized_x=position.normalized_x,
        normalized_y=position.normalized_y,
        pixel_x=position.pixel_x,
        pixel_y=position.pixel_y,
        x_mm=position.x_mm,
        y_mm=position.y_mm,
        distance_mm=position.distance_mm,
        touching=contact.touching,
        contact_confidence=contact.confidence,
        tracking_confidence=tracking_confidence,
        frame_skew_ms=frame_skew_ms,
    )


def load_runtime_calibration(args: argparse.Namespace) -> ScreenCalibration | None:
    if args.preview_only:
        return None
    calibration = load_screen_calibration(args.screen_calibration)
    validate_runtime_resolution(calibration, args.width, args.height)
    return calibration


def create_frame_source(args: argparse.Namespace) -> DualCameraManager | OfflineFrameSource:
    if args.offline_left is not None and args.offline_right is not None:
        return OfflineFrameSource(str(args.offline_left), str(args.offline_right))
    return DualCameraManager(
        args.camera_0,
        args.camera_1,
        args.width,
        args.height,
        fov_mode=args.fov_mode,
        color_order=args.color_order,
    )


def annotate_camera(
    frame: np.ndarray,
    camera_label: str,
    candidates: list[LedCandidate],
    selected: LedCandidate | None,
    reflection: LedCandidate | None,
    detailed: bool,
) -> np.ndarray:
    out = frame.copy()
    cv2.putText(out, camera_label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    for candidate in candidates:
        center = (int(round(candidate.x)), int(round(candidate.y)))
        cv2.drawContours(out, [candidate.contour], -1, (0, 255, 255), 1)
        cv2.circle(out, center, 4, (0, 255, 255), -1)
        if detailed:
            cv2.putText(
                out,
                f"a={candidate.area:.0f} b={candidate.peak_brightness:.0f}",
                (center[0] + 6, center[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
    if reflection is not None:
        center = (int(round(reflection.x)), int(round(reflection.y)))
        cv2.circle(out, center, 14, (255, 0, 255), 2)
        cv2.putText(out, "REFLECTION", (center[0] + 8, center[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)
    if selected is not None:
        center = (int(round(selected.x)), int(round(selected.y)))
        cv2.circle(out, center, 22, (0, 255, 0), 2)
        cv2.putText(out, f"LED x={selected.x:.1f} y={selected.y:.1f}", (center[0] + 10, center[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return out


def draw_system_overlay(
    combined: np.ndarray,
    state: PenState,
    fps: float,
    processing_ms: dict[str, float],
    match: StereoMatch | None,
) -> np.ndarray:
    lines = [
        f"FPS {fps:.1f} skew={state.frame_skew_ms if state.frame_skew_ms is not None else -1:.1f}ms",
        f"valid={state.valid} track={state.tracking_confidence:.2f} contact={state.contact_confidence:.2f} {'TOUCH' if state.touching else 'HOVER'}",
    ]
    if state.valid:
        lines.extend(
            [
                f"mm=({state.x_mm:.1f}, {state.y_mm:.1f}, {state.distance_mm:.1f})",
                f"norm=({state.normalized_x:.3f}, {state.normalized_y:.3f}) px=({state.pixel_x}, {state.pixel_y})",
            ]
        )
    if match is not None:
        lines.append(f"reproj={match.reprojection_error:.2f}px geom={match.geometry_score:.2f}")
    lines.append(
        " ".join(f"{key}={value:.1f}ms" for key, value in processing_ms.items())
    )
    for index, line in enumerate(lines):
        cv2.putText(combined, line, (12, 32 + index * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    return combined


def save_snapshot(frame0: np.ndarray, frame1: np.ndarray, mask0: np.ndarray, mask1: np.ndarray, sequence: int) -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(SNAPSHOT_DIR / f"snapshot_{sequence:06d}_camera_0.png"), frame0)
    cv2.imwrite(str(SNAPSHOT_DIR / f"snapshot_{sequence:06d}_camera_1.png"), frame1)
    cv2.imwrite(str(SNAPSHOT_DIR / f"snapshot_{sequence:06d}_mask_0.png"), mask0)
    cv2.imwrite(str(SNAPSHOT_DIR / f"snapshot_{sequence:06d}_mask_1.png"), mask1)


def process_pair(
    sequence: int,
    frame0: CameraFrame,
    frame1: CameraFrame,
    frame_skew_ms: float,
    detector: LedDetector,
    calibration: ScreenCalibration | None,
    contact_detector: ContactDetector,
    tracking_filter: ExponentialPenFilter,
    args: argparse.Namespace,
) -> tuple[PenState, list[LedCandidate], list[LedCandidate], np.ndarray, np.ndarray, StereoMatch | None, dict[str, float]]:
    timings: dict[str, float] = {}
    start = time.perf_counter()
    det_start = time.perf_counter()
    candidates0, mask0 = detector.detect(frame0.frame)
    candidates1, mask1 = detector.detect(frame1.frame)
    timings["detect"] = (time.perf_counter() - det_start) * 1000.0

    if calibration is None:
        state = make_invalid_state(sequence, frame_skew_ms)
        timings["total"] = (time.perf_counter() - start) * 1000.0
        return state, candidates0, candidates1, mask0, mask1, None, timings

    if frame_skew_ms > args.max_frame_skew_ms:
        state = make_invalid_state(sequence, frame_skew_ms)
        timings["total"] = (time.perf_counter() - start) * 1000.0
        return state, candidates0, candidates1, mask0, mask1, None, timings

    match_start = time.perf_counter()
    previous_point = tracking_filter.point
    match = choose_best_stereo_match(
        candidates0,
        candidates1,
        calibration,
        previous_point=previous_point,
        max_reprojection_error=args.max_reprojection_error,
        screen_margin_mm=args.screen_margin_mm,
        max_jump_mm=args.max_jump_mm,
    )
    timings["match"] = (time.perf_counter() - match_start) * 1000.0

    if match is None or match.reprojection_error > args.max_reprojection_error:
        tracking_filter.update(None)
        state = make_invalid_state(sequence, frame_skew_ms)
        timings["total"] = (time.perf_counter() - start) * 1000.0
        return state, candidates0, candidates1, mask0, mask1, match, timings

    filter_start = time.perf_counter()
    smoothed, filter_confidence = tracking_filter.update(match.point_3d)
    timings["filter"] = (time.perf_counter() - filter_start) * 1000.0
    if smoothed is None:
        state = make_invalid_state(sequence, frame_skew_ms)
        timings["total"] = (time.perf_counter() - start) * 1000.0
        return state, candidates0, candidates1, mask0, mask1, match, timings

    position = point_to_screen_position(smoothed, calibration, args.screen_margin_mm)
    if position is None:
        state = make_invalid_state(sequence, frame_skew_ms)
        timings["total"] = (time.perf_counter() - start) * 1000.0
        return state, candidates0, candidates1, mask0, mask1, match, timings

    contact_start = time.perf_counter()
    contact = contact_detector.estimate(candidates0, candidates1, match)
    timings["contact"] = (time.perf_counter() - contact_start) * 1000.0
    tracking_confidence = min(match.confidence, filter_confidence)
    if tracking_confidence < args.tracking_confidence_threshold:
        state = make_invalid_state(sequence, frame_skew_ms)
    else:
        state = make_valid_state(sequence, position, contact, tracking_confidence, frame_skew_ms)

    timings["total"] = (time.perf_counter() - start) * 1000.0
    return state, candidates0, candidates1, mask0, mask1, match, timings


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    lower = parse_hsv_triplet(args.lower_h, args.lower_s, args.lower_v)
    upper = parse_hsv_triplet(args.upper_h, args.upper_s, args.upper_v)
    detector = LedDetector(lower, upper, args.min_area)
    calibration = load_runtime_calibration(args)
    source = create_frame_source(args)
    contact_detector = ContactDetector()
    tracking_filter = ExponentialPenFilter(args.smoothing_alpha, args.max_jump_mm)
    sender = UdpPenSender(args.laptop_ip, args.laptop_port) if args.send_udp else None
    detailed = bool(args.debug)
    sequence = 0
    fps = 0.0
    last_frame_time = time.perf_counter()

    try:
        if isinstance(source, DualCameraManager):
            source.start()
        LOGGER.info("Penultimate runtime started. Press Q to quit, S snapshot, D details, C recalibration reminder.")
        while True:
            capture_start = time.perf_counter()
            frame0, frame1, skew_ms = source.capture_pair()
            capture_ms = (time.perf_counter() - capture_start) * 1000.0
            state, candidates0, candidates1, mask0, mask1, match, timings = process_pair(
                sequence,
                frame0,
                frame1,
                skew_ms,
                detector,
                calibration,
                contact_detector,
                tracking_filter,
                args,
            )
            timings = {"capture": capture_ms, **timings}
            if sender is not None:
                sender.send(state)
            if args.debug or sequence % 15 == 0:
                LOGGER.info("%s", state.to_packet())

            now = time.perf_counter()
            elapsed = now - last_frame_time
            last_frame_time = now
            if elapsed > 0:
                instant_fps = 1.0 / elapsed
                fps = instant_fps if fps == 0.0 else 0.9 * fps + 0.1 * instant_fps

            if not args.headless:
                selected0 = match.camera_0_candidate if match else None
                selected1 = match.camera_1_candidate if match else None
                reflection0 = match.reflection_candidate_0 if match else None
                reflection1 = match.reflection_candidate_1 if match else None
                annotated0 = annotate_camera(frame0.frame, "CAMERA 0", candidates0, selected0, reflection0, detailed)
                annotated1 = annotate_camera(frame1.frame, "CAMERA 1", candidates1, selected1, reflection1, detailed)
                combined = np.hstack((annotated0, annotated1))
                cv2.imshow(FRAME_WINDOW_NAME, draw_system_overlay(combined, state, fps, timings, match))
                if args.show_mask:
                    cv2.imshow(MASK_WINDOW_NAME, np.hstack((mask0, mask1)))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    break
                if key in (ord("d"), ord("D")):
                    detailed = not detailed
                if key in (ord("s"), ord("S")):
                    save_snapshot(frame0.frame, frame1.frame, mask0, mask1, sequence)
                    LOGGER.info("Saved snapshot %s", sequence)
                if key in (ord("c"), ord("C")):
                    LOGGER.info("Run calibrate_screen.py after moving either camera.")

            sequence += 1
            if args.offline_left is not None:
                break
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by Ctrl+C.")
    finally:
        source.close()
        if sender is not None:
            sender.close()
        cv2.destroyAllWindows()
        LOGGER.info("Penultimate runtime shut down cleanly.")


def main() -> None:
    try:
        run(parse_args())
    except Exception as error:
        LOGGER.error("%s", error)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()

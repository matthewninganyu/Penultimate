from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import cv2
import numpy as np


MORPH_KERNEL_SIZE = (5, 5)
MORPH_OPEN_ITERATIONS = 1
MORPH_CLOSE_ITERATIONS = 2

DEFAULT_HSV_LOWER = np.array([10, 150, 220], dtype=np.uint8)
DEFAULT_HSV_UPPER = np.array([40, 255, 255], dtype=np.uint8)

SELECTED_RADIUS = 24
CENTROID_RADIUS = 4
COORDINATE_PRINT_DELTA_PIXELS = 8

YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)

SELECTION_STRATEGIES = ("rightmost", "leftmost", "largest")


@dataclass(frozen=True)
class LedCandidate:
    x: int
    y: int
    area: float
    circularity: float
    contour: np.ndarray


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


def calculate_circularity(contour: np.ndarray, area: float) -> float:
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0.0
    return float((4.0 * math.pi * area) / (perimeter * perimeter))


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
        return max(candidates, key=lambda c: c.x)
    if strategy == "leftmost":
        return min(candidates, key=lambda c: c.x)
    if strategy == "largest":
        return max(candidates, key=lambda c: c.area)

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


def format_candidate(candidate: LedCandidate | None) -> str:
    if candidate is None:
        return "none"
    return f"x={candidate.x}, y={candidate.y}"

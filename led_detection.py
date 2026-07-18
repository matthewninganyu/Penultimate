from __future__ import annotations

import argparse
import math

import cv2
import numpy as np

from config import (
    DEFAULT_LOWER_LED,
    DEFAULT_MIN_AREA,
    DEFAULT_UPPER_LED,
    MORPH_CLOSE_ITERATIONS,
    MORPH_KERNEL_SIZE,
    MORPH_OPEN_ITERATIONS,
)
from models import LedCandidate


DEFAULT_HSV_LOWER = DEFAULT_LOWER_LED
DEFAULT_HSV_UPPER = DEFAULT_UPPER_LED

SELECTED_RADIUS = 24
CENTROID_RADIUS = 4
COORDINATE_PRINT_DELTA_PIXELS = 8

YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)

SELECTION_STRATEGIES = ("rightmost", "leftmost", "largest")


class LedDetector:
    def __init__(
        self,
        lower_hsv: np.ndarray | None = None,
        upper_hsv: np.ndarray | None = None,
        min_area: float = DEFAULT_MIN_AREA,
    ) -> None:
        self.lower_hsv = DEFAULT_LOWER_LED.copy() if lower_hsv is None else lower_hsv
        self.upper_hsv = DEFAULT_UPPER_LED.copy() if upper_hsv is None else upper_hsv
        self.min_area = min_area
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)

    def detect(self, bgr_frame: np.ndarray) -> tuple[list[LedCandidate], np.ndarray]:
        mask = create_led_mask(bgr_frame, self.kernel, self.lower_hsv, self.upper_hsv)
        return candidates_from_mask_and_frame(bgr_frame, mask, self.min_area), mask


def parse_hsv_triplet(h: int, s: int, v: int) -> np.ndarray:
    if not 0 <= h <= 179:
        raise ValueError("HSV hue must be 0..179.")
    if not 0 <= s <= 255 or not 0 <= v <= 255:
        raise ValueError("HSV saturation/value must be 0..255.")
    return np.array([h, s, v], dtype=np.uint8)


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

    try:
        return parse_hsv_triplet(parts[0], parts[1], parts[2])
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{argument_name} {error}") from error


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


def candidates_from_mask_and_frame(
    bgr_frame: np.ndarray,
    mask: np.ndarray,
    min_area: float,
) -> list[LedCandidate]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    candidates: list[LedCandidate] = []

    for contour in contours:
        candidate = candidate_from_contour(contour, gray, mask, min_area)
        if candidate is not None:
            candidates.append(candidate)

    return sorted(candidates, key=lambda candidate: candidate.peak_brightness, reverse=True)


def find_led_candidates(mask: np.ndarray, min_area: float) -> list[LedCandidate]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    synthetic_brightness = np.where(mask > 0, 255, 0).astype(np.uint8)
    candidates: list[LedCandidate] = []

    for contour in contours:
        candidate = candidate_from_contour(contour, synthetic_brightness, mask, min_area)
        if candidate is not None:
            candidates.append(candidate)

    return candidates


def candidate_from_contour(
    contour: np.ndarray,
    brightness_image: np.ndarray,
    mask: np.ndarray,
    min_area: float,
) -> LedCandidate | None:
    area = float(cv2.contourArea(contour))
    if area < min_area:
        return None

    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return None

    contour_x = float(moments["m10"] / moments["m00"])
    contour_y = float(moments["m01"] / moments["m00"])
    _, _, width, height = cv2.boundingRect(contour)
    (_, _), radius = cv2.minEnclosingCircle(contour)

    roi_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(roi_mask, [contour], -1, 255, thickness=-1)
    pixel_values = brightness_image[roi_mask > 0].astype(np.float64)
    ys, xs = np.nonzero(roi_mask)
    if pixel_values.size == 0:
        return None

    weight_sum = float(pixel_values.sum())
    if weight_sum > 0:
        weighted_x = float((xs * pixel_values).sum() / weight_sum)
        weighted_y = float((ys * pixel_values).sum() / weight_sum)
    else:
        weighted_x = contour_x
        weighted_y = contour_y

    peak_index = int(np.argmax(pixel_values))
    return LedCandidate(
        x=weighted_x,
        y=weighted_y,
        contour_x=contour_x,
        contour_y=contour_y,
        peak_x=float(xs[peak_index]),
        peak_y=float(ys[peak_index]),
        area=area,
        radius=float(radius),
        width=float(width),
        height=float(height),
        circularity=calculate_circularity(contour, area),
        mean_brightness=float(pixel_values.mean()),
        peak_brightness=float(pixel_values.max()),
        contour=contour,
    )


def select_physical_led(
    candidates: list[LedCandidate],
    strategy: str,
) -> LedCandidate | None:
    """Temporary camera-specific fallback heuristic for preview-only scripts."""
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
        center = (int(round(candidate.x)), int(round(candidate.y)))
        cv2.drawContours(annotated, [candidate.contour], -1, YELLOW, 2)
        cv2.circle(annotated, center, CENTROID_RADIUS, YELLOW, -1)

    if selected is not None:
        selected_center = (int(round(selected.x)), int(round(selected.y)))
        cv2.circle(annotated, selected_center, SELECTED_RADIUS, GREEN, 3)
        cv2.putText(
            annotated,
            "SELECTED LED",
            (selected_center[0] + 12, selected_center[1] - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            GREEN,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"x={selected.x:.1f} y={selected.y:.1f}",
            (selected_center[0] + 12, selected_center[1] + 10),
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
    return f"x={candidate.x:.1f}, y={candidate.y:.1f}"

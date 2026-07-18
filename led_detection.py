from __future__ import annotations

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
        hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        opened = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            self.kernel,
            iterations=MORPH_OPEN_ITERATIONS,
        )
        cleaned = cv2.morphologyEx(
            opened,
            cv2.MORPH_CLOSE,
            self.kernel,
            iterations=MORPH_CLOSE_ITERATIONS,
        )
        return self._candidates_from_mask(bgr_frame, cleaned), cleaned

    def _candidates_from_mask(
        self,
        bgr_frame: np.ndarray,
        mask: np.ndarray,
    ) -> list[LedCandidate]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        candidates: list[LedCandidate] = []

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < self.min_area:
                continue

            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue

            contour_x = float(moments["m10"] / moments["m00"])
            contour_y = float(moments["m01"] / moments["m00"])
            x, y, width, height = cv2.boundingRect(contour)
            (_, _), radius = cv2.minEnclosingCircle(contour)

            roi_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(roi_mask, [contour], -1, 255, thickness=-1)
            pixel_values = gray[roi_mask > 0].astype(np.float64)
            ys, xs = np.nonzero(roi_mask)
            if pixel_values.size == 0:
                continue

            weight_sum = float(pixel_values.sum())
            if weight_sum > 0:
                weighted_x = float((xs * pixel_values).sum() / weight_sum)
                weighted_y = float((ys * pixel_values).sum() / weight_sum)
            else:
                weighted_x = contour_x
                weighted_y = contour_y

            peak_index = int(np.argmax(pixel_values))
            peak_x = float(xs[peak_index])
            peak_y = float(ys[peak_index])
            perimeter = cv2.arcLength(contour, True)
            circularity = 0.0
            if perimeter > 0:
                circularity = float((4.0 * math.pi * area) / (perimeter * perimeter))

            # The intensity-weighted centroid is the primary geometric point; it
            # is usually better than the full contour centre for merged glare.
            candidates.append(
                LedCandidate(
                    x=weighted_x,
                    y=weighted_y,
                    contour_x=contour_x,
                    contour_y=contour_y,
                    peak_x=peak_x,
                    peak_y=peak_y,
                    area=area,
                    radius=float(radius),
                    width=float(width),
                    height=float(height),
                    circularity=circularity,
                    mean_brightness=float(pixel_values.mean()),
                    peak_brightness=float(pixel_values.max()),
                    contour=contour,
                )
            )

        return sorted(candidates, key=lambda candidate: candidate.peak_brightness, reverse=True)


def parse_hsv_triplet(h: int, s: int, v: int) -> np.ndarray:
    if not 0 <= h <= 179:
        raise ValueError("HSV hue must be 0..179.")
    if not 0 <= s <= 255 or not 0 <= v <= 255:
        raise ValueError("HSV saturation/value must be 0..255.")
    return np.array([h, s, v], dtype=np.uint8)


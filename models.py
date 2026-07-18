from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass
class LedCandidate:
    x: float
    y: float
    contour_x: float
    contour_y: float
    peak_x: float
    peak_y: float
    area: float
    radius: float
    width: float
    height: float
    circularity: float
    mean_brightness: float
    peak_brightness: float
    contour: np.ndarray

    def image_point(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float64)


@dataclass
class CameraFrame:
    camera_index: int
    frame: np.ndarray
    sensor_timestamp_ns: int


@dataclass
class CameraObservation:
    camera_index: int
    timestamp_ns: int
    candidates: list[LedCandidate]
    mask: np.ndarray | None = None


@dataclass
class StereoMatch:
    camera_0_candidate: LedCandidate
    camera_1_candidate: LedCandidate
    point_3d: np.ndarray
    reprojection_error: float
    geometry_score: float
    temporal_score: float
    confidence: float
    reflection_candidate_0: LedCandidate | None = None
    reflection_candidate_1: LedCandidate | None = None


@dataclass
class ScreenPosition:
    x_mm: float
    y_mm: float
    distance_mm: float
    normalized_x: float
    normalized_y: float
    pixel_x: int
    pixel_y: int


@dataclass
class ContactEvidence:
    touching: bool
    confidence: float
    camera_0_score: float
    camera_1_score: float
    merged_blob_detected: bool
    normalized_separation: float | None


@dataclass
class PenState:
    sequence: int
    timestamp: float
    valid: bool
    normalized_x: float | None
    normalized_y: float | None
    pixel_x: int | None
    pixel_y: int | None
    x_mm: float | None
    y_mm: float | None
    distance_mm: float | None
    touching: bool
    contact_confidence: float
    tracking_confidence: float
    frame_skew_ms: float | None

    def to_packet(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Intrinsics:
    camera_index: int
    width: int
    height: int
    K: np.ndarray
    D: np.ndarray
    rms_error: float
    timestamp: float


@dataclass
class ScreenCalibration:
    K0: np.ndarray
    D0: np.ndarray
    R0: np.ndarray
    t0: np.ndarray
    P0: np.ndarray
    K1: np.ndarray
    D1: np.ndarray
    R1: np.ndarray
    t1: np.ndarray
    P1: np.ndarray
    screen_width_mm: float
    screen_height_mm: float
    screen_width_px: int
    screen_height_px: int
    image_width: int
    image_height: int
    reprojection_error_0: float
    reprojection_error_1: float
    timestamp: float


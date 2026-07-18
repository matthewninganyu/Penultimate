from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_CAMERA_0 = 0
DEFAULT_CAMERA_1 = 1
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
CAMERA_FORMAT = "RGB888"

DEFAULT_CALIBRATION_DIR = Path("calibration")
DEFAULT_CAMERA_0_INTRINSICS = DEFAULT_CALIBRATION_DIR / "camera_0_intrinsics.npz"
DEFAULT_CAMERA_1_INTRINSICS = DEFAULT_CALIBRATION_DIR / "camera_1_intrinsics.npz"
DEFAULT_SCREEN_CALIBRATION = DEFAULT_CALIBRATION_DIR / "screen_calibration.npz"

DEFAULT_LOWER_LED = np.array([10, 150, 220], dtype=np.uint8)
DEFAULT_UPPER_LED = np.array([40, 255, 255], dtype=np.uint8)
DEFAULT_MIN_AREA = 30.0
MORPH_KERNEL_SIZE = (5, 5)
MORPH_OPEN_ITERATIONS = 1
MORPH_CLOSE_ITERATIONS = 2

DEFAULT_MAX_FRAME_SKEW_MS = 20.0
DEFAULT_MAX_REPROJECTION_ERROR = 8.0
DEFAULT_TRACKING_CONFIDENCE_THRESHOLD = 0.25
DEFAULT_SMOOTHING_ALPHA = 0.45
DEFAULT_MAX_JUMP_MM = 80.0
DEFAULT_SCREEN_MARGIN_MM = 15.0

TOUCH_START_SCORE = 0.68
TOUCH_END_SCORE = 0.42
TOUCH_CONFIRM_FRAMES = 2
HOVER_CONFIRM_FRAMES = 2


@dataclass(frozen=True)
class LedDetectionConfig:
    lower_hsv: np.ndarray
    upper_hsv: np.ndarray
    min_area: float = DEFAULT_MIN_AREA


@dataclass(frozen=True)
class RuntimeConfig:
    camera_0: int = DEFAULT_CAMERA_0
    camera_1: int = DEFAULT_CAMERA_1
    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    min_area: float = DEFAULT_MIN_AREA
    max_frame_skew_ms: float = DEFAULT_MAX_FRAME_SKEW_MS
    max_reprojection_error: float = DEFAULT_MAX_REPROJECTION_ERROR
    tracking_confidence_threshold: float = DEFAULT_TRACKING_CONFIDENCE_THRESHOLD
    smoothing_alpha: float = DEFAULT_SMOOTHING_ALPHA
    max_jump_mm: float = DEFAULT_MAX_JUMP_MM
    screen_margin_mm: float = DEFAULT_SCREEN_MARGIN_MM


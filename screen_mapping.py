from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from config import DEFAULT_SCREEN_MARGIN_MM
from models import Intrinsics, ScreenCalibration, ScreenPosition


def load_intrinsics(path: Path) -> Intrinsics:
    if not path.exists():
        raise FileNotFoundError(f"Missing camera intrinsic calibration: {path}")
    data = np.load(path, allow_pickle=True)
    required = ("K", "D", "camera_index", "image_width", "image_height", "rms_error", "timestamp")
    _require_keys(data, required, path)
    return Intrinsics(
        camera_index=int(data["camera_index"]),
        width=int(data["image_width"]),
        height=int(data["image_height"]),
        K=np.asarray(data["K"], dtype=np.float64),
        D=np.asarray(data["D"], dtype=np.float64),
        rms_error=float(data["rms_error"]),
        timestamp=float(data["timestamp"]),
    )


def load_screen_calibration(path: Path) -> ScreenCalibration:
    if not path.exists():
        raise FileNotFoundError(f"Missing screen calibration: {path}")
    data = np.load(path, allow_pickle=True)
    required = (
        "K0",
        "D0",
        "R0",
        "t0",
        "P0",
        "K1",
        "D1",
        "R1",
        "t1",
        "P1",
        "screen_width_mm",
        "screen_height_mm",
        "screen_width_px",
        "screen_height_px",
        "image_width",
        "image_height",
        "reprojection_error_0",
        "reprojection_error_1",
        "timestamp",
    )
    _require_keys(data, required, path)
    return ScreenCalibration(
        K0=np.asarray(data["K0"], dtype=np.float64),
        D0=np.asarray(data["D0"], dtype=np.float64),
        R0=np.asarray(data["R0"], dtype=np.float64),
        t0=np.asarray(data["t0"], dtype=np.float64).reshape(3, 1),
        P0=np.asarray(data["P0"], dtype=np.float64),
        K1=np.asarray(data["K1"], dtype=np.float64),
        D1=np.asarray(data["D1"], dtype=np.float64),
        R1=np.asarray(data["R1"], dtype=np.float64),
        t1=np.asarray(data["t1"], dtype=np.float64).reshape(3, 1),
        P1=np.asarray(data["P1"], dtype=np.float64),
        screen_width_mm=float(data["screen_width_mm"]),
        screen_height_mm=float(data["screen_height_mm"]),
        screen_width_px=int(data["screen_width_px"]),
        screen_height_px=int(data["screen_height_px"]),
        image_width=int(data["image_width"]),
        image_height=int(data["image_height"]),
        reprojection_error_0=float(data["reprojection_error_0"]),
        reprojection_error_1=float(data["reprojection_error_1"]),
        timestamp=float(data["timestamp"]),
    )


def validate_runtime_resolution(
    calibration: ScreenCalibration,
    width: int,
    height: int,
) -> None:
    if calibration.image_width != width or calibration.image_height != height:
        raise ValueError(
            "Calibration resolution mismatch: "
            f"calibration={calibration.image_width}x{calibration.image_height}, "
            f"runtime={width}x{height}. Recalibrate or use matching --width/--height."
        )


def point_to_screen_position(
    point_3d: np.ndarray,
    calibration: ScreenCalibration,
    margin_mm: float = DEFAULT_SCREEN_MARGIN_MM,
) -> ScreenPosition | None:
    x_mm = float(point_3d[0])
    y_mm = float(point_3d[1])
    z_mm = float(point_3d[2])

    if x_mm < -margin_mm or x_mm > calibration.screen_width_mm + margin_mm:
        return None
    if y_mm < -margin_mm or y_mm > calibration.screen_height_mm + margin_mm:
        return None

    normalized_x = x_mm / calibration.screen_width_mm
    normalized_y = y_mm / calibration.screen_height_mm
    if normalized_x < 0.0:
        normalized_x = 0.0
    elif normalized_x > 1.0:
        normalized_x = 1.0
    if normalized_y < 0.0:
        normalized_y = 0.0
    elif normalized_y > 1.0:
        normalized_y = 1.0

    pixel_x = int(round(normalized_x * (calibration.screen_width_px - 1)))
    pixel_y = int(round(normalized_y * (calibration.screen_height_px - 1)))
    return ScreenPosition(
        x_mm=x_mm,
        y_mm=y_mm,
        distance_mm=z_mm,
        normalized_x=normalized_x,
        normalized_y=normalized_y,
        pixel_x=pixel_x,
        pixel_y=pixel_y,
    )


def _require_keys(data: Any, keys: tuple[str, ...], path: Path) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise ValueError(f"Calibration file {path} is missing keys: {', '.join(missing)}")


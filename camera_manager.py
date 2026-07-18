from __future__ import annotations

import logging
import time
from typing import Any

import cv2
import numpy as np

from config import CAMERA_FORMAT
from models import CameraFrame

try:
    from picamera2 import Picamera2
except ImportError:  # pragma: no cover - unavailable on most dev machines.
    Picamera2 = None  # type: ignore[assignment]


LOGGER = logging.getLogger(__name__)


def ensure_picamera2_available() -> None:
    if Picamera2 is None:
        raise RuntimeError(
            "Picamera2 is not installed. On Raspberry Pi OS, install "
            "python3-picamera2 and create the venv with --system-site-packages."
        )


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


def select_largest_fov_sensor_mode(
    camera: Any,
) -> tuple[int, dict[str, Any], tuple[int, int, int, int]] | None:
    modes = getattr(camera, "sensor_modes", None)
    if not modes:
        return None
    selected: tuple[int, dict[str, Any], tuple[int, int, int, int]] | None = None
    for index, mode in enumerate(modes):
        crop = crop_limits_to_tuple(mode.get("crop_limits"))
        if crop is None:
            continue
        if selected is None or crop[2] * crop[3] > selected[2][2] * selected[2][3]:
            selected = (index, mode, crop)
    return selected


class DualCameraManager:
    def __init__(
        self,
        camera_0_index: int,
        camera_1_index: int,
        width: int,
        height: int,
        fov_mode: str = "full",
        color_order: str = "rgb",
    ) -> None:
        self.camera_0_index = camera_0_index
        self.camera_1_index = camera_1_index
        self.width = width
        self.height = height
        self.fov_mode = fov_mode
        self.color_order = color_order
        self.camera_0: Any | None = None
        self.camera_1: Any | None = None
        self._crop_0: tuple[int, int, int, int] | None = None
        self._crop_1: tuple[int, int, int, int] | None = None

    @staticmethod
    def available_cameras() -> list[dict[str, Any]]:
        ensure_picamera2_available()
        return Picamera2.global_camera_info()

    def start(self) -> None:
        ensure_picamera2_available()
        info = self.available_cameras()
        if len(info) < 2:
            raise RuntimeError(f"Fewer than two cameras detected: found {len(info)}.")
        if self.camera_0_index == self.camera_1_index:
            raise RuntimeError("Camera indexes must be different.")

        self.camera_0, self._crop_0 = self._open_camera(self.camera_0_index)
        self.camera_1, self._crop_1 = self._open_camera(self.camera_1_index)
        self.camera_0.start()
        self.camera_1.start()
        self._apply_crop(self.camera_0, self.camera_0_index, self._crop_0)
        self._apply_crop(self.camera_1, self.camera_1_index, self._crop_1)
        time.sleep(1.0)

    def capture_pair(self) -> tuple[CameraFrame, CameraFrame, float]:
        if self.camera_0 is None or self.camera_1 is None:
            raise RuntimeError("Cameras are not started.")
        frame_0, ts_0 = self._capture_one(self.camera_0, self.camera_0_index)
        frame_1, ts_1 = self._capture_one(self.camera_1, self.camera_1_index)
        skew_ms = abs(ts_0 - ts_1) / 1_000_000.0
        return (
            CameraFrame(self.camera_0_index, frame_0, ts_0),
            CameraFrame(self.camera_1_index, frame_1, ts_1),
            skew_ms,
        )

    def close(self) -> None:
        for camera, label in (
            (self.camera_0, f"camera {self.camera_0_index}"),
            (self.camera_1, f"camera {self.camera_1_index}"),
        ):
            if camera is None:
                continue
            for method_name in ("stop", "close"):
                try:
                    getattr(camera, method_name)()
                except Exception as error:  # pragma: no cover - hardware cleanup.
                    LOGGER.warning("Failed to %s %s: %s", method_name, label, error)

    def _open_camera(self, index: int) -> tuple[Any, tuple[int, int, int, int] | None]:
        camera = Picamera2(index)
        selected = select_largest_fov_sensor_mode(camera) if self.fov_mode == "full" else None
        crop = selected[2] if selected else None
        main = {"size": (self.width, self.height), "format": CAMERA_FORMAT}
        if selected is not None:
            try:
                config = camera.create_video_configuration(main=main, raw=selected[1])
                camera.configure(config)
                LOGGER.info("Camera %s selected full-FOV mode %s", index, selected[1])
                return camera, crop
            except Exception as error:
                LOGGER.warning("Camera %s full-FOV mode failed: %s", index, error)
        config = camera.create_video_configuration(main=main)
        camera.configure(config)
        return camera, None

    @staticmethod
    def _apply_crop(
        camera: Any,
        index: int,
        crop: tuple[int, int, int, int] | None,
    ) -> None:
        if crop is None:
            return
        try:
            camera.set_controls({"ScalerCrop": crop})
            metadata = camera.capture_metadata()
            LOGGER.info("Camera %s requested ScalerCrop=%s active=%s", index, crop, metadata.get("ScalerCrop"))
        except Exception as error:
            LOGGER.warning("Camera %s could not set ScalerCrop=%s: %s", index, crop, error)

    def _capture_one(self, camera: Any, camera_index: int) -> tuple[np.ndarray, int]:
        request = camera.capture_request()
        try:
            frame = request.make_array("main")
            metadata = request.get_metadata()
        finally:
            request.release()
        timestamp = int(metadata.get("SensorTimestamp", time.time_ns()))
        frame = frame[:, :, :3]
        if self.color_order == "rgb":
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif self.color_order != "bgr":
            raise ValueError(f"Unsupported color order: {self.color_order}")
        return frame, timestamp


class OfflineFrameSource:
    def __init__(self, left_path: str, right_path: str) -> None:
        self.left_path = left_path
        self.right_path = right_path

    def capture_pair(self) -> tuple[CameraFrame, CameraFrame, float]:
        left = cv2.imread(self.left_path, cv2.IMREAD_COLOR)
        right = cv2.imread(self.right_path, cv2.IMREAD_COLOR)
        if left is None:
            raise RuntimeError(f"Could not read offline left image: {self.left_path}")
        if right is None:
            raise RuntimeError(f"Could not read offline right image: {self.right_path}")
        timestamp = time.time_ns()
        return CameraFrame(0, left, timestamp), CameraFrame(1, right, timestamp), 0.0

    def close(self) -> None:
        return


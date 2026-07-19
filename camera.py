from __future__ import annotations

from typing import Any

import cv2
import numpy as np

try:
    from picamera2 import Picamera2
except ImportError:  # pragma: no cover - exercised on non-Raspberry Pi systems.
    Picamera2 = None  # type: ignore[assignment]


CAMERA_FORMAT = "RGB888"
COLOR_ORDERS = ("rgb", "bgr")
FOV_MODES = ("full", "current")

# Fixed exposure/AWB so the blue LED is a stable, unclipped blue blob instead of
# a blown-out white core, and glare/reflections fall below threshold. Runtime
# controls only -- reapplied every process start via apply_led_controls().
# Sweep EXPOSURE_TIME_US (100..8000) on hardware to find where the LED core
# turns blue and ambient goes near-black. Capped under ~10000us at 100 FPS.
EXPOSURE_TIME_US = 1000
ANALOGUE_GAIN = 1.0
COLOUR_GAINS = (2.0, 2.0)  # (red, blue) manual WB gains; tune so blue reads blue
LENS_POSITION = 4.5  # dioptres (1/m); set for the fixed working distance


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
    fov_mode: str = "full",
) -> tuple[Any, tuple[int, int, int, int] | None]:
    ensure_picamera2_available()

    try:
        camera = Picamera2(camera_index)
    except Exception as error:
        raise RuntimeError(
            f"Camera {camera_index} cannot be opened: {error}"
        ) from error

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


def apply_led_controls(camera: Any, camera_index: int) -> None:
    """Lock exposure/AWB/focus for blue-LED detection. Call after camera.start()."""
    from libcamera import controls  # Pi-only; imported lazily so Mac dev works.

    settings = {
        "AeEnable": False,
        "AwbEnable": False,
        "AnalogueGain": ANALOGUE_GAIN,
        "ColourGains": COLOUR_GAINS,
        "ExposureTime": EXPOSURE_TIME_US,
        "AfMode": controls.AfModeEnum.Manual,
        "LensPosition": LENS_POSITION,
        "HdrMode": controls.HdrModeEnum.Off,
    }
    try:
        camera.set_controls(settings)
    except Exception as error:  # pragma: no cover - hardware dependent.
        print(f"Warning: Camera {camera_index} could not set LED controls: {error}")


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
        return frame
    raise ValueError(f"Unsupported color order: {color_order}")


def ensure_frame_size(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    if frame.shape[1] == width and frame.shape[0] == height:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def safe_camera_call(camera: Any | None, method_name: str, label: str) -> None:
    if camera is None:
        return
    try:
        getattr(camera, method_name)()
    except Exception as error:
        print(f"Warning: failed to {method_name} {label}: {error}")
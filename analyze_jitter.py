from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SERIES_CHOICES = ("auto", "camera0", "camera1", "screen-px", "normalized", "screen-mm")

BACKGROUND = (250, 250, 250)
GRID = (220, 220, 220)
AXIS = (70, 70, 70)
TEXT = (30, 30, 30)
BLUE = (200, 90, 30)
RED = (40, 40, 210)
GREEN = (50, 150, 60)
PURPLE = (150, 70, 150)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze coordinate jitter from realtime.py or pen_receiver.py logs."
    )
    parser.add_argument("input", type=Path, help="CSV log or newline-delimited JSON packet log.")
    parser.add_argument(
        "--series",
        choices=SERIES_CHOICES,
        default="auto",
        help="Coordinate series to analyze. auto prefers screen pixels, then camera 0.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG plot. Defaults to output/<input-stem>_jitter.png.",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        default=None,
        help="Optional JSON file for the computed metrics.",
    )
    parser.add_argument(
        "--min-valid-samples",
        type=int,
        default=10,
        help="Fail if fewer valid coordinate samples are available.",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if not text:
        return []
    json_rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        json_rows.append(flatten_packet(json.loads(stripped)))
    if json_rows:
        return json_rows
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


def flatten_packet(packet: dict[str, Any]) -> dict[str, Any]:
    camera_0 = packet.get("camera_0") or {}
    camera_1 = packet.get("camera_1") or {}
    return {
        "sequence": packet.get("sequence"),
        "timestamp": packet.get("timestamp"),
        "type": packet.get("type"),
        "valid": packet.get("valid"),
        "camera_0_x": camera_0.get("x", ""),
        "camera_0_y": camera_0.get("y", ""),
        "camera_0_area": camera_0.get("area", ""),
        "camera_0_peak_brightness": camera_0.get("peak_brightness", ""),
        "camera_1_x": camera_1.get("x", ""),
        "camera_1_y": camera_1.get("y", ""),
        "camera_1_area": camera_1.get("area", ""),
        "camera_1_peak_brightness": camera_1.get("peak_brightness", ""),
        "pixel_x": packet.get("pixel_x", ""),
        "pixel_y": packet.get("pixel_y", ""),
        "normalized_x": packet.get("normalized_x", ""),
        "normalized_y": packet.get("normalized_y", ""),
        "x_mm": packet.get("x_mm", ""),
        "y_mm": packet.get("y_mm", ""),
    }


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        if isinstance(value, str) and value.lower() in ("none", "null", "nan"):
            return None
        number = float(value)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def resolve_series(rows: list[dict[str, Any]], requested: str) -> tuple[str, str, str, str]:
    candidates = {
        "screen-px": ("pixel_x", "pixel_y", "screen pixels"),
        "normalized": ("normalized_x", "normalized_y", "normalized screen"),
        "screen-mm": ("x_mm", "y_mm", "screen millimeters"),
        "camera0": ("camera_0_x", "camera_0_y", "camera 0 pixels"),
        "camera1": ("camera_1_x", "camera_1_y", "camera 1 pixels"),
    }
    if requested != "auto":
        x_key, y_key, label = candidates[requested]
        return requested, x_key, y_key, label
    for name in ("screen-px", "screen-mm", "normalized", "camera0", "camera1"):
        x_key, y_key, label = candidates[name]
        if any(as_float(row.get(x_key)) is not None and as_float(row.get(y_key)) is not None for row in rows):
            return name, x_key, y_key, label
    raise ValueError("No usable coordinate columns found.")


def extract_samples(
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
) -> tuple[np.ndarray, int]:
    samples: list[tuple[float, float, float]] = []
    invalid_count = 0
    fallback_time = 0.0
    for row in rows:
        valid = as_bool(row.get("valid", True))
        x = as_float(row.get(x_key))
        y = as_float(row.get(y_key))
        timestamp = as_float(row.get("timestamp"))
        if timestamp is None:
            timestamp = fallback_time
            fallback_time += 1.0
        if not valid or x is None or y is None:
            invalid_count += 1
            continue
        samples.append((timestamp, x, y))
    if not samples:
        return np.empty((0, 3), dtype=np.float64), invalid_count
    data = np.array(samples, dtype=np.float64)
    order = np.argsort(data[:, 0])
    return data[order], invalid_count


def percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values, q))


def compute_metrics(data: np.ndarray, series_name: str, label: str, total_rows: int, invalid_rows: int) -> dict[str, Any]:
    t = data[:, 0]
    x = data[:, 1]
    y = data[:, 2]
    center_x = float(np.median(x))
    center_y = float(np.median(y))
    dx = x - center_x
    dy = y - center_y
    radial = np.sqrt((dx * dx) + (dy * dy))
    step = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2) if len(data) > 1 else np.array([], dtype=np.float64)
    duration = float(max(0.0, t[-1] - t[0])) if len(data) > 1 else 0.0
    sample_rate = float((len(data) - 1) / duration) if duration > 0 else 0.0
    return {
        "series": series_name,
        "label": label,
        "total_rows": total_rows,
        "valid_samples": int(len(data)),
        "invalid_or_missing_rows": int(invalid_rows),
        "duration_seconds": duration,
        "sample_rate_hz": sample_rate,
        "center_x": center_x,
        "center_y": center_y,
        "std_x": float(np.std(x)),
        "std_y": float(np.std(y)),
        "rms_jitter": float(np.sqrt(np.mean(radial * radial))),
        "p95_jitter": percentile(radial, 95),
        "max_jitter": float(np.max(radial)),
        "peak_to_peak_x": float(np.max(x) - np.min(x)),
        "peak_to_peak_y": float(np.max(y) - np.min(y)),
        "mean_frame_step": float(np.mean(step)) if step.size else 0.0,
        "p95_frame_step": percentile(step, 95),
        "max_frame_step": float(np.max(step)) if step.size else 0.0,
    }


def scale_points(
    xs: np.ndarray,
    ys: np.ndarray,
    rect: tuple[int, int, int, int],
    pad_fraction: float = 0.08,
) -> list[tuple[int, int]]:
    left, top, width, height = rect
    min_x, max_x = float(np.min(xs)), float(np.max(xs))
    min_y, max_y = float(np.min(ys)), float(np.max(ys))
    range_x = max(max_x - min_x, 1e-9)
    range_y = max(max_y - min_y, 1e-9)
    min_x -= range_x * pad_fraction
    max_x += range_x * pad_fraction
    min_y -= range_y * pad_fraction
    max_y += range_y * pad_fraction
    range_x = max(max_x - min_x, 1e-9)
    range_y = max(max_y - min_y, 1e-9)
    return [
        (
            int(round(left + ((x - min_x) / range_x) * width)),
            int(round(top + height - ((y - min_y) / range_y) * height)),
        )
        for x, y in zip(xs, ys)
    ]


def draw_panel_title(image: np.ndarray, title: str, rect: tuple[int, int, int, int]) -> None:
    x, y, _, _ = rect
    cv2.putText(image, title, (x, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.62, TEXT, 2, cv2.LINE_AA)


def draw_axes(image: np.ndarray, rect: tuple[int, int, int, int]) -> None:
    x, y, w, h = rect
    cv2.rectangle(image, (x, y), (x + w, y + h), AXIS, 1)
    for i in range(1, 5):
        gx = x + int(round(w * i / 5))
        gy = y + int(round(h * i / 5))
        cv2.line(image, (gx, y), (gx, y + h), GRID, 1)
        cv2.line(image, (x, gy), (x + w, gy), GRID, 1)


def draw_polyline(image: np.ndarray, points: list[tuple[int, int]], color: tuple[int, int, int], thickness: int = 2) -> None:
    if len(points) < 2:
        return
    cv2.polylines(image, [np.array(points, dtype=np.int32)], False, color, thickness, cv2.LINE_AA)


def draw_plot(data: np.ndarray, metrics: dict[str, Any], output_path: Path) -> None:
    image = np.full((900, 1400, 3), BACKGROUND, dtype=np.uint8)
    cv2.putText(
        image,
        f"Jitter: {metrics['label']} | samples={metrics['valid_samples']} | rms={metrics['rms_jitter']:.3f} | p95={metrics['p95_jitter']:.3f}",
        (36, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        TEXT,
        2,
        cv2.LINE_AA,
    )

    t = data[:, 0] - data[0, 0]
    x = data[:, 1]
    y = data[:, 2]
    radial = np.sqrt((x - metrics["center_x"]) ** 2 + (y - metrics["center_y"]) ** 2)

    scatter_rect = (60, 100, 520, 330)
    time_rect = (660, 100, 660, 330)
    radial_rect = (60, 530, 1260, 270)

    for title, rect in (
        ("XY scatter around stationary point", scatter_rect),
        ("X/Y over time", time_rect),
        ("Radial jitter from median point over time", radial_rect),
    ):
        draw_panel_title(image, title, rect)
        draw_axes(image, rect)

    scatter_points = scale_points(x, y, scatter_rect)
    for point in scatter_points:
        cv2.circle(image, point, 2, BLUE, -1, cv2.LINE_AA)
    center_point = scale_points(np.array([metrics["center_x"]]), np.array([metrics["center_y"]]), scatter_rect)[0]
    cv2.drawMarker(image, center_point, RED, markerType=cv2.MARKER_CROSS, markerSize=22, thickness=2)

    draw_polyline(image, scale_points(t, x, time_rect), BLUE)
    draw_polyline(image, scale_points(t, y, time_rect), GREEN)
    cv2.putText(image, "x", (time_rect[0] + time_rect[2] - 54, time_rect[1] + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, BLUE, 2, cv2.LINE_AA)
    cv2.putText(image, "y", (time_rect[0] + time_rect[2] - 26, time_rect[1] + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, GREEN, 2, cv2.LINE_AA)

    draw_polyline(image, scale_points(t, radial, radial_rect), PURPLE)
    summary_lines = [
        f"center=({metrics['center_x']:.3f}, {metrics['center_y']:.3f})",
        f"std x/y=({metrics['std_x']:.3f}, {metrics['std_y']:.3f})",
        f"peak-to-peak x/y=({metrics['peak_to_peak_x']:.3f}, {metrics['peak_to_peak_y']:.3f})",
        f"frame step mean/p95/max={metrics['mean_frame_step']:.3f}/{metrics['p95_frame_step']:.3f}/{metrics['max_frame_step']:.3f}",
        f"duration={metrics['duration_seconds']:.2f}s rate={metrics['sample_rate_hz']:.1f}Hz invalid={metrics['invalid_or_missing_rows']}",
    ]
    for index, line in enumerate(summary_lines):
        cv2.putText(image, line, (60, 835 + index * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, TEXT, 1, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def print_metrics(metrics: dict[str, Any], output_path: Path) -> None:
    print(f"Series: {metrics['series']} ({metrics['label']})")
    print(f"Samples: {metrics['valid_samples']} valid / {metrics['total_rows']} rows")
    print(f"Duration: {metrics['duration_seconds']:.2f}s at {metrics['sample_rate_hz']:.1f}Hz")
    print(f"Center: x={metrics['center_x']:.3f}, y={metrics['center_y']:.3f}")
    print(f"Std dev: x={metrics['std_x']:.3f}, y={metrics['std_y']:.3f}")
    print(f"RMS jitter: {metrics['rms_jitter']:.3f}")
    print(f"95th percentile jitter: {metrics['p95_jitter']:.3f}")
    print(f"Max jitter: {metrics['max_jitter']:.3f}")
    print(f"Frame-to-frame step mean/p95/max: {metrics['mean_frame_step']:.3f}/{metrics['p95_frame_step']:.3f}/{metrics['max_frame_step']:.3f}")
    print(f"Wrote plot: {output_path}")


def main() -> None:
    args = parse_args()
    rows = read_rows(args.input)
    if not rows:
        raise SystemExit(f"No log rows found in {args.input}")

    series_name, x_key, y_key, label = resolve_series(rows, args.series)
    data, invalid_rows = extract_samples(rows, x_key, y_key)
    if len(data) < args.min_valid_samples:
        raise SystemExit(
            f"Only {len(data)} valid samples found for {series_name}; "
            f"need at least {args.min_valid_samples}."
        )

    metrics = compute_metrics(data, series_name, label, len(rows), invalid_rows)
    output_path = args.output or Path("output") / f"{args.input.stem}_{series_name}_jitter.png"
    draw_plot(data, metrics, output_path)
    if args.metrics_json is not None:
        args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print_metrics(metrics, output_path)


if __name__ == "__main__":
    main()

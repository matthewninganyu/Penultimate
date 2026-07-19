from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np

from config import DEFAULT_MIN_AREA, MORPH_KERNEL_SIZE
from models import LedCandidate


DEFAULT_HSV_LOWER = np.array([90, 100, 120], dtype=np.uint8)
DEFAULT_HSV_UPPER = np.array([140, 255, 255], dtype=np.uint8)
DEFAULT_ROI_START = 0.45
DEFAULT_ROI_END = 0.85

BLUE_NEARBY_KERNEL_SIZE = (21, 21)
CORE_CLOSE_KERNEL_SIZE = (5, 5)
MIN_CORE_VALUE = 245
MIN_CORE_BLUE = 235
MIN_CORE_GREEN = 220
MIN_CORE_RED = 180

SELECTED_RADIUS = 24
CENTROID_RADIUS = 4
COORDINATE_PRINT_DELTA_PIXELS = 8

CYAN = (255, 255, 0)
YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)

SELECTION_STRATEGIES = ("rightmost", "leftmost", "largest")
MORPH_OPEN_ITERATIONS = 0
MORPH_CLOSE_ITERATIONS = 1


class LedDetector:
    def __init__(
        self,
        lower_hsv: np.ndarray | None = None,
        upper_hsv: np.ndarray | None = None,
        min_area: float = DEFAULT_MIN_AREA,
        roi_start: float = DEFAULT_ROI_START,
        roi_end: float = DEFAULT_ROI_END,
    ) -> None:
        self.lower_hsv = DEFAULT_HSV_LOWER.copy() if lower_hsv is None else lower_hsv
        self.upper_hsv = DEFAULT_HSV_UPPER.copy() if upper_hsv is None else upper_hsv
        self.min_area = min_area
        self.roi_start = roi_start
        self.roi_end = roi_end
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)

    def detect(self, bgr_frame: np.ndarray) -> tuple[list[LedCandidate], np.ndarray]:
        mask = create_led_mask(
            bgr_frame,
            self.kernel,
            self.lower_hsv,
            self.upper_hsv,
            self.roi_start,
            self.roi_end,
        )
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
            f"{argument_name} must contain three integers like 100,150,180."
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


def _roi_bounds(height: int, roi_start: float, roi_end: float) -> tuple[int, int]:
    start = min(max(float(roi_start), 0.0), 1.0)
    end = min(max(float(roi_end), 0.0), 1.0)
    if end <= start:
        raise ValueError("--roi-end must be greater than --roi-start.")
    return int(round(height * start)), int(round(height * end))


def create_led_mask(
    frame: np.ndarray,
    kernel: np.ndarray | None = None,
    hsv_lower: np.ndarray | None = None,
    hsv_upper: np.ndarray | None = None,
    roi_start: float = DEFAULT_ROI_START,
    roi_end: float = DEFAULT_ROI_END,
) -> np.ndarray:
    """Return a full-frame mask of bright LED cores backed by nearby blue halo."""
    del kernel
    lower = DEFAULT_HSV_LOWER if hsv_lower is None else hsv_lower
    upper = DEFAULT_HSV_UPPER if hsv_upper is None else hsv_upper

    height = frame.shape[0]
    y0, y1 = _roi_bounds(height, roi_start, roi_end)
    roi = frame[y0:y1]
    if roi.size == 0:
        return np.zeros(frame.shape[:2], dtype=np.uint8)

    blurred = cv2.GaussianBlur(roi, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    _, _, value = cv2.split(hsv)
    blue, green, red = cv2.split(blurred)

    blue_halo = cv2.inRange(hsv, lower, upper)
    blue_nearby = cv2.dilate(
        blue_halo,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, BLUE_NEARBY_KERNEL_SIZE),
    )
    bright_core = (
        (value >= MIN_CORE_VALUE)
        & (blue >= MIN_CORE_BLUE)
        & (green >= MIN_CORE_GREEN)
        & (red >= MIN_CORE_RED)
    ).astype(np.uint8) * 255

    core_near_blue = cv2.bitwise_and(bright_core, blue_nearby)
    core_near_blue = cv2.morphologyEx(
        core_near_blue,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, CORE_CLOSE_KERNEL_SIZE),
        iterations=MORPH_CLOSE_ITERATIONS,
    )

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask[y0:y1] = core_near_blue
    return mask


def candidates_from_mask_and_frame(
    bgr_frame: np.ndarray,
    mask: np.ndarray,
    min_area: float,
) -> list[LedCandidate]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    candidates: list[LedCandidate] = []

    for label in range(1, count):
        candidate = _candidate_from_component(label, labels, stats, gray, mask, min_area)
        if candidate is not None:
            candidates.append(candidate)

    return sorted(candidates, key=lambda candidate: candidate.peak_brightness, reverse=True)


def find_led_candidates(mask: np.ndarray, min_area: float) -> list[LedCandidate]:
    synthetic = np.where(mask > 0, 255, 0).astype(np.uint8)
    return candidates_from_mask_and_frame(cv2.cvtColor(synthetic, cv2.COLOR_GRAY2BGR), mask, min_area)


def _candidate_from_component(
    label: int,
    labels: np.ndarray,
    stats: np.ndarray,
    brightness_image: np.ndarray,
    mask: np.ndarray,
    min_area: float,
) -> LedCandidate | None:
    component_area = float(stats[label, cv2.CC_STAT_AREA])
    if component_area < min_area:
        return None

    x0 = int(stats[label, cv2.CC_STAT_LEFT])
    y0 = int(stats[label, cv2.CC_STAT_TOP])
    width = int(stats[label, cv2.CC_STAT_WIDTH])
    height = int(stats[label, cv2.CC_STAT_HEIGHT])
    component = labels[y0 : y0 + height, x0 : x0 + width] == label
    if not np.any(component):
        return None

    roi_brightness = brightness_image[y0 : y0 + height, x0 : x0 + width].astype(np.float64)
    ys, xs = np.nonzero(component)
    pixel_values = roi_brightness[component]
    if pixel_values.size == 0:
        return None

    weights = (pixel_values / 255.0) ** 4
    weight_sum = float(weights.sum())
    if weight_sum > 0:
        weighted_x = float(x0 + (xs * weights).sum() / weight_sum)
        weighted_y = float(y0 + (ys * weights).sum() / weight_sum)
    else:
        weighted_x = float(x0 + xs.mean())
        weighted_y = float(y0 + ys.mean())

    contour_mask = np.zeros(mask.shape, dtype=np.uint8)
    contour_mask[y0 : y0 + height, x0 : x0 + width][component] = 255
    contours, _ = cv2.findContours(contour_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    contour_area = float(cv2.contourArea(contour))
    moments = cv2.moments(contour)
    if moments["m00"] != 0:
        contour_x = float(moments["m10"] / moments["m00"])
        contour_y = float(moments["m01"] / moments["m00"])
    else:
        contour_x = weighted_x
        contour_y = weighted_y

    (_, _), radius = cv2.minEnclosingCircle(contour)
    peak_index = int(np.argmax(pixel_values))
    return LedCandidate(
        x=weighted_x,
        y=weighted_y,
        contour_x=contour_x,
        contour_y=contour_y,
        peak_x=float(x0 + xs[peak_index]),
        peak_y=float(y0 + ys[peak_index]),
        area=component_area,
        radius=float(radius),
        width=float(width),
        height=float(height),
        circularity=calculate_circularity(contour, contour_area),
        mean_brightness=float(pixel_values.mean()),
        peak_brightness=float(pixel_values.max()),
        contour=contour,
    )


def select_physical_led(
    candidates: list[LedCandidate],
    strategy: str,
) -> LedCandidate | None:
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

    cv2.putText(annotated, camera_label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2, cv2.LINE_AA)
    cv2.putText(annotated, f"FPS: {fps:.1f}", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 2, cv2.LINE_AA)

    if not candidates:
        cv2.putText(annotated, "NO LED DETECTED", (12, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.75, RED, 2, cv2.LINE_AA)
        return annotated

    for candidate in candidates:
        center = (int(round(candidate.x)), int(round(candidate.y)))
        cv2.drawContours(annotated, [candidate.contour], -1, YELLOW, 2)
        cv2.circle(annotated, center, CENTROID_RADIUS, CYAN, -1)

    if selected is not None:
        selected_center = (int(round(selected.x)), int(round(selected.y)))
        cv2.circle(annotated, selected_center, SELECTED_RADIUS, GREEN, 3)
        cv2.drawMarker(
            annotated,
            selected_center,
            RED,
            markerType=cv2.MARKER_CROSS,
            markerSize=34,
            thickness=3,
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


def annotate(
    frame: np.ndarray,
    selected: LedCandidate | None,
    candidates: list[LedCandidate],
) -> np.ndarray:
    marked = frame.copy()

    for index, candidate in enumerate(candidates, start=1):
        x, y, width, height = cv2.boundingRect(candidate.contour)
        center = (int(round(candidate.x)), int(round(candidate.y)))
        cv2.rectangle(marked, (x, y), (x + width, y + height), CYAN, 2)
        cv2.circle(marked, center, 14, CYAN, 2)
        cv2.putText(
            marked,
            f"#{index}",
            (center[0] + 16, center[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            YELLOW,
            2,
            cv2.LINE_AA,
        )

    if selected is not None:
        center = (int(round(selected.x)), int(round(selected.y)))
        cv2.circle(marked, center, 28, GREEN, 4)
        cv2.drawMarker(marked, center, RED, markerType=cv2.MARKER_CROSS, markerSize=38, thickness=4)
        cv2.putText(
            marked,
            f"SELECTED ({selected.x:.0f}, {selected.y:.0f})",
            (center[0] + 20, center[1] + 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            GREEN,
            2,
            cv2.LINE_AA,
        )

    return marked


def coordinate_changed(
    previous: LedCandidate | None,
    current: LedCandidate | None,
    min_delta: int,
) -> bool:
    if current is None:
        return previous is not None
    if previous is None:
        return True
    return abs(current.x - previous.x) >= min_delta or abs(current.y - previous.y) >= min_delta


def format_candidate(candidate: LedCandidate | None) -> str:
    if candidate is None:
        return "none"
    return f"x={candidate.x:.1f}, y={candidate.y:.1f}"


def format_candidate_verbose(candidate: LedCandidate | None) -> str:
    if candidate is None:
        return "none"
    return (
        f"x={candidate.x:.1f}, y={candidate.y:.1f}, area={candidate.area:.0f}, "
        f"peak={candidate.peak_brightness:.0f}, mean={candidate.mean_brightness:.1f}"
    )


def detect_blue_led(frame: np.ndarray) -> tuple[LedCandidate | None, list[LedCandidate], np.ndarray]:
    detector = LedDetector()
    candidates, mask = detector.detect(frame)
    return select_physical_led(candidates, "largest"), candidates, mask


def run_image(image_path: Path, output_dir: Path) -> None:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    selected, candidates, mask = detect_blue_led(frame)
    marked = annotate(frame, selected, candidates)

    output_dir.mkdir(parents=True, exist_ok=True)
    marked_path = output_dir / f"{image_path.stem}_blue_led_candidates.png"
    mask_path = output_dir / f"{image_path.stem}_blue_led_mask.png"
    cv2.imwrite(str(marked_path), marked)
    cv2.imwrite(str(mask_path), mask)

    print(f"Candidates: {len(candidates)}")
    print(f"Selected: {format_candidate_verbose(selected)}")
    print(f"Wrote: {marked_path}")
    print(f"Wrote: {mask_path}")


def run_camera(camera_index: int) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")

    while True:
        success, frame = cap.read()
        if not success:
            break

        selected, candidates, mask = detect_blue_led(frame)
        cv2.imshow("Camera", annotate(frame, selected, candidates))
        cv2.imshow("LED Mask", mask)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect and mark blue LED candidates.")
    parser.add_argument("--image", type=Path, help="Still image to test, for example photos/4136.jpeg.")
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    if args.image is not None:
        run_image(args.image, args.output_dir)
    else:
        run_camera(args.camera)


if __name__ == "__main__":
    main()

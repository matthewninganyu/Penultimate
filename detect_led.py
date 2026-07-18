from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


OUTPUT_DIR = Path("output")
MASK_OUTPUT_NAME = "led_mask.png"
DETECTION_OUTPUT_NAME = "led_detection.png"

HSV_LOWER_YELLOW_ORANGE = np.array([5, 200, 235], dtype=np.uint8)
HSV_UPPER_YELLOW_ORANGE = np.array([35, 255, 255], dtype=np.uint8)

MORPH_KERNEL_SIZE = (5, 5)
MORPH_OPEN_ITERATIONS = 1
MORPH_CLOSE_ITERATIONS = 2

MIN_CONTOUR_AREA = 100.0
STRONG_CANDIDATE_MIN_AREA = 1_000.0
STRONG_CANDIDATE_MIN_CIRCULARITY = 0.10

ANNOTATION_RADIUS = 18
SELECTED_RADIUS = 28


@dataclass(frozen=True)
class LedCandidate:
    index: int
    centroid_x: float
    centroid_y: float
    area: float
    circularity: float
    bounding_box: tuple[int, int, int, int]
    contour: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect yellow-orange LED candidates in a still image."
    )
    parser.add_argument(
        "--image",
        required=True,
        type=Path,
        help="Path to the input image, for example IMG_3298.jpeg.",
    )
    return parser.parse_args()


def resolve_image_path(image_path: Path) -> Path:
    if image_path.is_file():
        return image_path

    if not image_path.is_absolute():
        photos_path = Path("photos") / image_path
        if photos_path.is_file():
            print(
                f"Input image '{image_path}' was not found in the current directory; "
                f"using '{photos_path}' instead."
            )
            return photos_path

    raise FileNotFoundError(f"Input image not found: {image_path}")


def load_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"OpenCV could not load image: {image_path}")
    return image


def create_led_mask(image_bgr: np.ndarray) -> np.ndarray:
    image_hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    raw_mask = cv2.inRange(
        image_hsv,
        HSV_LOWER_YELLOW_ORANGE,
        HSV_UPPER_YELLOW_ORANGE,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, MORPH_KERNEL_SIZE)
    opened = cv2.morphologyEx(
        raw_mask,
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


def contour_circularity(contour: np.ndarray, area: float) -> float:
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0.0
    return float((4.0 * np.pi * area) / (perimeter * perimeter))


def find_led_candidates(mask: np.ndarray) -> list[LedCandidate]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[LedCandidate] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < MIN_CONTOUR_AREA:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue

        centroid_x = float(moments["m10"] / moments["m00"])
        centroid_y = float(moments["m01"] / moments["m00"])
        bounding_box = tuple(int(value) for value in cv2.boundingRect(contour))

        candidates.append(
            LedCandidate(
                index=0,
                centroid_x=centroid_x,
                centroid_y=centroid_y,
                area=area,
                circularity=contour_circularity(contour, area),
                bounding_box=bounding_box,
                contour=contour,
            )
        )

    candidates.sort(key=lambda candidate: candidate.centroid_x)
    return [
        LedCandidate(
            index=index,
            centroid_x=candidate.centroid_x,
            centroid_y=candidate.centroid_y,
            area=candidate.area,
            circularity=candidate.circularity,
            bounding_box=candidate.bounding_box,
            contour=candidate.contour,
        )
        for index, candidate in enumerate(candidates, start=1)
    ]


def select_physical_led_candidate(
    candidates: list[LedCandidate],
) -> LedCandidate | None:
    """Temporary heuristic for IMG_3298.jpeg.

    The prototype image includes the physical pen-tip LED and a reflected LED on
    the laptop screen. For this specific image, the physical LED is expected to
    be the rightmost strong yellow-orange blob. Replace this with geometry,
    calibration, or temporal tracking once later tasks add more system context.
    """
    strong_candidates = [
        candidate
        for candidate in candidates
        if candidate.area >= STRONG_CANDIDATE_MIN_AREA
        and candidate.circularity >= STRONG_CANDIDATE_MIN_CIRCULARITY
    ]
    if not strong_candidates:
        return None
    return max(strong_candidates, key=lambda candidate: candidate.centroid_x)


def print_candidates(candidates: list[LedCandidate]) -> None:
    if not candidates:
        print("No LED candidates found.")
        return

    print("LED candidates:")
    for candidate in candidates:
        x, y, width, height = candidate.bounding_box
        print(
            f"  #{candidate.index}: "
            f"x={candidate.centroid_x:.1f}, "
            f"y={candidate.centroid_y:.1f}, "
            f"area={candidate.area:.1f}, "
            f"circularity={candidate.circularity:.3f}, "
            f"bbox=({x}, {y}, {width}, {height})"
        )


def annotate_detection(
    image_bgr: np.ndarray,
    candidates: list[LedCandidate],
    selected: LedCandidate | None,
) -> np.ndarray:
    annotated = image_bgr.copy()

    for candidate in candidates:
        center = (round(candidate.centroid_x), round(candidate.centroid_y))
        cv2.drawContours(annotated, [candidate.contour], -1, (255, 0, 0), 2)
        cv2.circle(annotated, center, ANNOTATION_RADIUS, (255, 0, 0), 2)
        cv2.putText(
            annotated,
            f"#{candidate.index}",
            (center[0] + 12, center[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    if selected is not None:
        selected_center = (round(selected.centroid_x), round(selected.centroid_y))
        cv2.circle(annotated, selected_center, SELECTED_RADIUS, (0, 0, 255), 4)
        cv2.drawMarker(
            annotated,
            selected_center,
            (0, 0, 255),
            cv2.MARKER_CROSS,
            48,
            4,
        )
        cv2.putText(
            annotated,
            "selected physical LED",
            (selected_center[0] + 20, selected_center[1] + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )

    return annotated


def save_outputs(mask: np.ndarray, annotated: np.ndarray) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mask_path = OUTPUT_DIR / MASK_OUTPUT_NAME
    detection_path = OUTPUT_DIR / DETECTION_OUTPUT_NAME

    if not cv2.imwrite(str(mask_path), mask):
        raise OSError(f"Failed to save mask image: {mask_path}")
    if not cv2.imwrite(str(detection_path), annotated):
        raise OSError(f"Failed to save annotated image: {detection_path}")

    print(f"Saved mask: {mask_path}")
    print(f"Saved detection image: {detection_path}")


def main() -> int:
    args = parse_args()

    try:
        image_path = resolve_image_path(args.image)
        image_bgr = load_image(image_path)
        mask = create_led_mask(image_bgr)
        candidates = find_led_candidates(mask)
        selected = select_physical_led_candidate(candidates)

        print_candidates(candidates)
        if selected is None:
            print("Selected LED: none")
        else:
            print(
                "Selected LED coordinates: "
                f"x={selected.centroid_x:.1f}, y={selected.centroid_y:.1f}"
            )

        annotated = annotate_detection(image_bgr, candidates, selected)
        save_outputs(mask, annotated)
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"Error: {error}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from detect_led import MASK_OUTPUT_NAME, create_led_mask


OUTPUT_DIR = Path("output")
MARKED_OUTPUT_NAME = "led_marked.png"
MIN_AREA = 100.0

CONTOUR_COLOR_YELLOW = (0, 255, 255)
SELECTED_COLOR_GREEN = (0, 255, 0)
ERROR_COLOR_RED = (0, 0, 255)

CENTROID_DOT_RADIUS = 5
SELECTED_CIRCLE_RADIUS = 36
WINDOW_NAME = "LED Marked"


@dataclass(frozen=True)
class LedCandidate:
    x: int
    y: int
    area: float
    contour: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect and mark LED candidates on the original image."
    )
    parser.add_argument(
        "--image",
        required=True,
        type=Path,
        help="Path to the original image, for example IMG_3298.jpeg.",
    )
    parser.add_argument(
        "--mask",
        required=False,
        type=Path,
        help=(
            "Optional path to an existing binary LED mask, for example "
            "led_mask.png. If omitted, the mask is generated from --image."
        ),
    )
    return parser.parse_args()


def resolve_existing_path(path: Path, fallback_dir: Path) -> Path:
    if path.is_file():
        return path

    if not path.is_absolute():
        fallback_path = fallback_dir / path
        if fallback_path.is_file():
            print(f"Using '{fallback_path}' for requested path '{path}'.")
            return fallback_path

    raise FileNotFoundError(f"File not found: {path}")


def load_image(original_path: Path) -> np.ndarray:
    resolved_original_path = resolve_existing_path(original_path, Path("photos"))
    original = cv2.imread(str(resolved_original_path), cv2.IMREAD_COLOR)
    if original is None:
        raise ValueError(f"OpenCV could not load original image: {resolved_original_path}")
    return original


def load_images(original_path: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray]:
    original = load_image(original_path)
    resolved_mask_path = resolve_existing_path(mask_path, OUTPUT_DIR)
    mask = cv2.imread(str(resolved_mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"OpenCV could not load mask image: {resolved_mask_path}")

    validate_mask_dimensions(original, mask)
    return original, mask


def validate_mask_dimensions(original: np.ndarray, mask: np.ndarray) -> None:
    original_height, original_width = original.shape[:2]
    mask_height, mask_width = mask.shape[:2]
    if (original_width, original_height) != (mask_width, mask_height):
        raise ValueError(
            "Original image and mask dimensions do not match: "
            f"original={original_width}x{original_height}, "
            f"mask={mask_width}x{mask_height}"
        )


def save_mask(mask: np.ndarray) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mask_path = OUTPUT_DIR / MASK_OUTPUT_NAME
    if not cv2.imwrite(str(mask_path), mask):
        raise OSError(f"Failed to save mask image: {mask_path}")
    return mask_path


def find_led_candidates(mask: np.ndarray, min_area: float) -> list[LedCandidate]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[LedCandidate] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue

        x = int(round(moments["m10"] / moments["m00"]))
        y = int(round(moments["m01"] / moments["m00"]))
        candidates.append(LedCandidate(x=x, y=y, area=area, contour=contour))

    return sorted(candidates, key=lambda candidate: candidate.x)


def select_physical_led(candidates: list[LedCandidate]) -> LedCandidate | None:
    """Temporary heuristic for the current prototype images.

    The mask can include both the real LED near the pen tip and its reflection
    on the laptop screen. This prototype first ranks candidates by relative
    contour area and circularity to keep the two strongest LED-like blobs, then
    chooses the rightmost of those two. Replace this with calibrated geometry or
    tracking later.
    """
    if not candidates:
        return None

    max_area = max(candidate.area for candidate in candidates)
    if max_area <= 0:
        return None

    ranked_candidates = sorted(
        candidates,
        key=lambda candidate: (candidate.area / max_area)
        * candidate_circularity(candidate),
        reverse=True,
    )
    strongest_candidates = ranked_candidates[:2]
    return max(strongest_candidates, key=lambda candidate: candidate.x)


def candidate_circularity(candidate: LedCandidate) -> float:
    perimeter = cv2.arcLength(candidate.contour, True)
    if perimeter == 0:
        return 0.0
    return float((4.0 * np.pi * candidate.area) / (perimeter * perimeter))


def annotate_candidates(
    image: np.ndarray,
    candidates: list[LedCandidate],
    selected: LedCandidate | None,
) -> np.ndarray:
    marked = image.copy()

    if not candidates:
        cv2.putText(
            marked,
            "NO LED DETECTED",
            (50, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            ERROR_COLOR_RED,
            5,
            cv2.LINE_AA,
        )
        return marked

    for candidate in candidates:
        center = (candidate.x, candidate.y)
        cv2.drawContours(marked, [candidate.contour], -1, CONTOUR_COLOR_YELLOW, 3)
        cv2.circle(marked, center, CENTROID_DOT_RADIUS, CONTOUR_COLOR_YELLOW, -1)
        cv2.putText(
            marked,
            f"({candidate.x}, {candidate.y})",
            (candidate.x + 12, candidate.y - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            CONTOUR_COLOR_YELLOW,
            2,
            cv2.LINE_AA,
        )

    if selected is not None:
        selected_center = (selected.x, selected.y)
        cv2.circle(
            marked,
            selected_center,
            SELECTED_CIRCLE_RADIUS,
            SELECTED_COLOR_GREEN,
            5,
        )
        cv2.putText(
            marked,
            "SELECTED LED",
            (selected.x + 24, selected.y + 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            SELECTED_COLOR_GREEN,
            3,
            cv2.LINE_AA,
        )

    return marked


def save_marked_image(marked: np.ndarray) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / MARKED_OUTPUT_NAME
    if not cv2.imwrite(str(output_path), marked):
        raise OSError(f"Failed to save marked image: {output_path}")
    return output_path


def display_marked_image(marked: np.ndarray) -> None:
    cv2.imshow(WINDOW_NAME, marked)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> int:
    args = parse_args()

    try:
        if args.mask is None:
            original = load_image(args.image)
            mask = create_led_mask(original)
            mask_path = save_mask(mask)
            print(f"Generated mask: {mask_path}")
        else:
            original, mask = load_images(args.image, args.mask)

        candidates = find_led_candidates(mask, MIN_AREA)
        selected = select_physical_led(candidates)

        if not candidates:
            print("No LED candidates found in the mask.")
        elif selected is not None:
            print(f"Selected LED coordinates: x={selected.x}, y={selected.y}")

        marked = annotate_candidates(original, candidates, selected)
        output_path = save_marked_image(marked)
        print(f"Saved marked image: {output_path}")
        display_marked_image(marked)
    except (FileNotFoundError, ValueError, OSError) as error:
        print(f"Error: {error}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import math

import cv2
import numpy as np

from config import HOVER_CONFIRM_FRAMES, TOUCH_CONFIRM_FRAMES, TOUCH_END_SCORE, TOUCH_START_SCORE
from models import ContactEvidence, LedCandidate, StereoMatch


class ContactDetector:
    def __init__(
        self,
        touch_start_score: float = TOUCH_START_SCORE,
        touch_end_score: float = TOUCH_END_SCORE,
        touch_confirm_frames: int = TOUCH_CONFIRM_FRAMES,
        hover_confirm_frames: int = HOVER_CONFIRM_FRAMES,
    ) -> None:
        self.touch_start_score = touch_start_score
        self.touch_end_score = touch_end_score
        self.touch_confirm_frames = touch_confirm_frames
        self.hover_confirm_frames = hover_confirm_frames
        self.touching = False
        self._touch_count = 0
        self._hover_count = 0

    def estimate(
        self,
        candidates_0: list[LedCandidate],
        candidates_1: list[LedCandidate],
        match: StereoMatch | None,
    ) -> ContactEvidence:
        score_0, sep_0, merged_0 = camera_contact_score(candidates_0, match.camera_0_candidate if match else None)
        score_1, sep_1, merged_1 = camera_contact_score(candidates_1, match.camera_1_candidate if match else None)
        distance_score = 0.0
        if match is not None:
            distance_score = max(0.0, min(1.0, 1.0 - abs(float(match.point_3d[2])) / 18.0))

        combined = (0.35 * score_0) + (0.35 * score_1) + (0.30 * distance_score)
        if self.touching:
            if combined < self.touch_end_score:
                self._hover_count += 1
            else:
                self._hover_count = 0
            if self._hover_count >= self.hover_confirm_frames:
                self.touching = False
                self._touch_count = 0
        else:
            if combined >= self.touch_start_score:
                self._touch_count += 1
            else:
                self._touch_count = 0
            if self._touch_count >= self.touch_confirm_frames:
                self.touching = True
                self._hover_count = 0

        separations = [value for value in (sep_0, sep_1) if value is not None]
        normalized_separation = float(np.mean(separations)) if separations else None
        return ContactEvidence(
            touching=self.touching,
            confidence=max(0.0, min(1.0, combined)),
            camera_0_score=score_0,
            camera_1_score=score_1,
            merged_blob_detected=merged_0 or merged_1,
            normalized_separation=normalized_separation,
        )


def camera_contact_score(
    candidates: list[LedCandidate],
    physical: LedCandidate | None,
) -> tuple[float, float | None, bool]:
    if not candidates:
        return 0.0, None, False
    if physical is None:
        physical = candidates[0]

    merged_score = merged_blob_score(physical)
    if len(candidates) >= 2:
        nearest = min(
            (candidate for candidate in candidates if candidate is not physical),
            key=lambda candidate: float(np.linalg.norm(candidate.image_point() - physical.image_point())),
        )
        separation_px = float(np.linalg.norm(nearest.image_point() - physical.image_point()))
        radius_sum = max(1.0, physical.radius + nearest.radius)
        normalized_separation = separation_px / radius_sum
        separation_score = max(0.0, min(1.0, 1.0 - normalized_separation / 2.0))
        score = max(separation_score, merged_score * 0.8)
        return score, normalized_separation, merged_score > 0.65

    return merged_score, None, merged_score > 0.65


def merged_blob_score(candidate: LedCandidate) -> float:
    aspect = max(candidate.width, candidate.height) / max(1.0, min(candidate.width, candidate.height))
    size_score = max(0.0, min(1.0, (candidate.area - 60.0) / 300.0))
    aspect_score = max(0.0, min(1.0, 1.4 - abs(aspect - 1.0)))
    brightness_score = max(0.0, min(1.0, candidate.peak_brightness / 255.0))
    return (0.45 * size_score) + (0.25 * aspect_score) + (0.30 * brightness_score)


def contours_overlap(candidate_a: LedCandidate, candidate_b: LedCandidate) -> bool:
    rect_a = cv2.boundingRect(candidate_a.contour)
    rect_b = cv2.boundingRect(candidate_b.contour)
    ax, ay, aw, ah = rect_a
    bx, by, bw, bh = rect_b
    return ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by


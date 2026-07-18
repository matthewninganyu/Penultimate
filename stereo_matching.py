from __future__ import annotations

import math

import numpy as np

from config import DEFAULT_MAX_REPROJECTION_ERROR, DEFAULT_SCREEN_MARGIN_MM
from models import LedCandidate, ScreenCalibration, StereoMatch
from triangulation import point_in_front_of_both_cameras, reprojection_error, triangulate_candidates


def choose_best_stereo_match(
    candidates_0: list[LedCandidate],
    candidates_1: list[LedCandidate],
    calibration: ScreenCalibration,
    previous_point: np.ndarray | None = None,
    max_reprojection_error: float = DEFAULT_MAX_REPROJECTION_ERROR,
    screen_margin_mm: float = DEFAULT_SCREEN_MARGIN_MM,
    max_jump_mm: float = 80.0,
) -> StereoMatch | None:
    best: StereoMatch | None = None
    best_score = math.inf

    for candidate_0 in candidates_0:
        for candidate_1 in candidates_1:
            point_3d = triangulate_candidates(candidate_0, candidate_1, calibration)
            if point_3d is None:
                continue
            if not point_in_front_of_both_cameras(point_3d, calibration):
                continue

            reproj = reprojection_error(point_3d, candidate_0, candidate_1, calibration)
            if reproj > max_reprojection_error * 4.0:
                continue

            workspace_penalty = workspace_distance_penalty(point_3d, calibration, screen_margin_mm)
            behind_screen_penalty = max(0.0, -float(point_3d[2])) / 10.0
            jump_penalty = 0.0
            if previous_point is not None:
                jump_mm = float(np.linalg.norm(point_3d - previous_point))
                if jump_mm > max_jump_mm:
                    jump_penalty = (jump_mm - max_jump_mm) / max_jump_mm

            geometry_score = (reproj / max_reprojection_error) + workspace_penalty + behind_screen_penalty
            temporal_score = jump_penalty
            total_score = geometry_score + temporal_score
            confidence = max(0.0, min(1.0, 1.0 / (1.0 + total_score)))

            if total_score < best_score:
                best_score = total_score
                best = StereoMatch(
                    camera_0_candidate=candidate_0,
                    camera_1_candidate=candidate_1,
                    point_3d=point_3d,
                    reprojection_error=reproj,
                    geometry_score=geometry_score,
                    temporal_score=temporal_score,
                    confidence=confidence,
                    reflection_candidate_0=nearest_other_candidate(candidate_0, candidates_0),
                    reflection_candidate_1=nearest_other_candidate(candidate_1, candidates_1),
                )

    return best


def workspace_distance_penalty(
    point_3d: np.ndarray,
    calibration: ScreenCalibration,
    margin_mm: float,
) -> float:
    x_mm, y_mm, _ = [float(v) for v in point_3d]
    penalty = 0.0
    if x_mm < -margin_mm:
        penalty += abs(x_mm + margin_mm) / max(1.0, margin_mm)
    elif x_mm > calibration.screen_width_mm + margin_mm:
        penalty += abs(x_mm - calibration.screen_width_mm - margin_mm) / max(1.0, margin_mm)
    if y_mm < -margin_mm:
        penalty += abs(y_mm + margin_mm) / max(1.0, margin_mm)
    elif y_mm > calibration.screen_height_mm + margin_mm:
        penalty += abs(y_mm - calibration.screen_height_mm - margin_mm) / max(1.0, margin_mm)
    return penalty


def nearest_other_candidate(
    selected: LedCandidate,
    candidates: list[LedCandidate],
) -> LedCandidate | None:
    others = [candidate for candidate in candidates if candidate is not selected]
    if not others:
        return None
    selected_point = selected.image_point()
    return min(others, key=lambda candidate: float(np.linalg.norm(candidate.image_point() - selected_point)))


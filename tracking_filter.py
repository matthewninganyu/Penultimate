from __future__ import annotations

import time

import numpy as np


class ExponentialPenFilter:
    def __init__(
        self,
        alpha: float = 0.45,
        max_jump_mm: float = 80.0,
        tracking_loss_reset_seconds: float = 0.75,
    ) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1].")
        self.alpha = alpha
        self.max_jump_mm = max_jump_mm
        self.tracking_loss_reset_seconds = tracking_loss_reset_seconds
        self._point: np.ndarray | None = None
        self._last_valid_time: float | None = None

    @property
    def point(self) -> np.ndarray | None:
        return None if self._point is None else self._point.copy()

    def update(self, measurement: np.ndarray | None, timestamp: float | None = None) -> tuple[np.ndarray | None, float]:
        now = time.time() if timestamp is None else timestamp
        if measurement is None:
            if self._last_valid_time is not None and now - self._last_valid_time > self.tracking_loss_reset_seconds:
                self.reset()
            return self.point, 0.0

        measurement = np.asarray(measurement, dtype=np.float64)
        if self._point is None:
            self._point = measurement.copy()
            self._last_valid_time = now
            return self.point, 1.0

        jump = float(np.linalg.norm(measurement - self._point))
        if jump > self.max_jump_mm:
            confidence = max(0.0, self.max_jump_mm / jump)
            if confidence < 0.35:
                return self.point, confidence
        else:
            confidence = 1.0

        self._point = self.alpha * measurement + (1.0 - self.alpha) * self._point
        self._last_valid_time = now
        return self.point, confidence

    def reset(self) -> None:
        self._point = None
        self._last_valid_time = None


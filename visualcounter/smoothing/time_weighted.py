from __future__ import annotations

from visualcounter.config import SmoothingSettings
from visualcounter.smoothing.base import Smoother
from visualcounter.smoothing.registry import SmootherFactory


class TimeWeightedAverageSmoother(Smoother):
    def __init__(self, window_seconds: float) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")
        self._window_seconds = window_seconds

    @property
    def retention_seconds(self) -> float:
        return self._window_seconds

    def smooth(self, samples: list[tuple[float, float]], now: float) -> float:
        if not samples:
            return 0.0
        if len(samples) == 1:
            return samples[0][1]

        window_start = now - self._window_seconds
        total = 0.0
        weight = 0.0

        t0, v0 = samples[0]
        if t0 > window_start:
            dt = t0 - window_start
            total += v0 * dt
            weight += dt

        for i in range(1, len(samples)):
            t_prev, v_prev = samples[i - 1]
            t_curr, v_curr = samples[i]
            seg_start = max(t_prev, window_start)
            seg_end = min(t_curr, now)
            if seg_end <= seg_start:
                continue

            if t_curr == t_prev:
                v_start = v_prev
                v_end = v_curr
            else:
                start_ratio = (seg_start - t_prev) / (t_curr - t_prev)
                end_ratio = (seg_end - t_prev) / (t_curr - t_prev)
                v_start = v_prev + (v_curr - v_prev) * start_ratio
                v_end = v_prev + (v_curr - v_prev) * end_ratio

            dt = seg_end - seg_start
            total += (v_start + v_end) * 0.5 * dt
            weight += dt

        if weight <= 0:
            return samples[-1][1]
        return total / weight


class TimeWeightedAverageFactory(SmootherFactory):
    def create(self, settings: SmoothingSettings) -> Smoother:
        window = float(settings.params.get("window_seconds", 10.0))
        return TimeWeightedAverageSmoother(window)

from __future__ import annotations

from visualcounter.config import SmoothingSettings
from visualcounter.smoothing.base import Smoother
from visualcounter.smoothing.registry import SmootherFactory


class NoSmoothing(Smoother):
    def smooth(self, samples: list[tuple[float, float]], now: float) -> float:
        del now
        if not samples:
            return 0.0
        return samples[-1][1]

    @property
    def retention_seconds(self) -> float:
        return 1.0


class NoSmoothingFactory(SmootherFactory):
    def create(self, settings: SmoothingSettings) -> Smoother:
        del settings
        return NoSmoothing()

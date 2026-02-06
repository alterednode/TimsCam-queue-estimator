from __future__ import annotations

from visualcounter.config import DetectorSettings
from visualcounter.detectors.base import Detector, DetectorFactory


class DetectorRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, DetectorFactory] = {}

    def register(self, detector_type: str, factory: DetectorFactory) -> None:
        self._factories[detector_type] = factory

    def create(self, settings: DetectorSettings) -> Detector:
        factory = self._factories.get(settings.type)
        if factory is None:
            known = ", ".join(sorted(self._factories)) or "<none>"
            raise ValueError(f"Unknown detector type '{settings.type}'. Registered types: {known}")
        return factory.create(settings)

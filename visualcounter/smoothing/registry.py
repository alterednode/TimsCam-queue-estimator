from __future__ import annotations

from visualcounter.config import SmoothingSettings
from visualcounter.smoothing.base import Smoother


class SmootherFactory:
    def create(self, settings: SmoothingSettings) -> Smoother:
        raise NotImplementedError


class SmootherRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, SmootherFactory] = {}

    def register(self, smoother_type: str, factory: SmootherFactory) -> None:
        self._factories[smoother_type] = factory

    def create(self, settings: SmoothingSettings | None) -> Smoother | None:
        if settings is None:
            return None
        factory = self._factories.get(settings.type)
        if factory is None:
            known = ", ".join(sorted(self._factories)) or "<none>"
            raise ValueError(f"Unknown smoothing type '{settings.type}'. Registered types: {known}")
        return factory.create(settings)

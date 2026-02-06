from __future__ import annotations

from abc import ABC, abstractmethod


class Smoother(ABC):
    @abstractmethod
    def smooth(self, samples: list[tuple[float, float]], now: float) -> float:
        raise NotImplementedError

    @property
    @abstractmethod
    def retention_seconds(self) -> float:
        raise NotImplementedError

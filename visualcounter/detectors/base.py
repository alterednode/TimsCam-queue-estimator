from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from visualcounter.config import DetectorSettings
from visualcounter.models import Detection


class Detector(ABC):
    @abstractmethod
    def infer(self, frame: np.ndarray) -> list[Detection]:
        raise NotImplementedError


class DetectorFactory(ABC):
    @abstractmethod
    def create(self, settings: DetectorSettings) -> Detector:
        raise NotImplementedError

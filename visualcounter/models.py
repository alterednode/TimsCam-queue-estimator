from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    x: int
    y: int
    width: int
    height: int
    confidence: float

    @property
    def centroid(self) -> tuple[int, int]:
        return (int(self.x + self.width / 2), int(self.y + self.height / 2))


@dataclass(frozen=True)
class Snapshot:
    sequence: int
    timestamp: float
    frame_shape: tuple[int, int]
    detections: list[Detection]

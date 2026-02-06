from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from visualcounter.config import Point
from visualcounter.models import Detection


def parse_roi_string(raw: str) -> list[Point]:
    points: list[Point] = []
    for token in raw.split(";"):
        token = token.strip()
        if not token:
            continue
        parts = token.split(",")
        if len(parts) != 2:
            raise ValueError(f"Invalid ROI point '{token}', expected x,y")
        x, y = parts
        points.append((int(x), int(y)))

    if len(points) < 3:
        raise ValueError("ROI must contain at least 3 points")
    return points


def count_in_roi(detections: Iterable[Detection], roi_points: list[Point]) -> int:
    polygon = np.array(roi_points, dtype=np.int32)
    count = 0
    for det in detections:
        cx, cy = det.centroid
        if cv2.pointPolygonTest(polygon, (cx, cy), False) >= 0:
            count += 1
    return count


def roi_to_key(roi_name: str | None, roi_points: list[Point]) -> str:
    if roi_name:
        return f"name:{roi_name}"
    joined = ";".join(f"{x},{y}" for x, y in roi_points)
    return f"points:{joined}"

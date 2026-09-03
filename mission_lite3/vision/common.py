from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Generic, Iterable, Optional, TypeVar


T = TypeVar("T")


@dataclass
class BBox:
    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2.0, self.y + self.h / 2.0


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: Optional[BBox] = None


class StableVote(Generic[T]):
    def __init__(self, window: int, votes: int):
        self.window = window
        self.votes = votes
        self.values: Deque[Optional[T]] = deque(maxlen=window)

    def add(self, value: Optional[T]) -> Optional[T]:
        self.values.append(value)
        if value is None:
            return None
        candidate = self.current()
        return candidate if candidate == value else None

    def clear(self) -> None:
        self.values.clear()

    def current(self) -> Optional[T]:
        counts = self.counts()
        for item in reversed(self.values):
            if item is not None and counts.get(item, 0) >= self.votes:
                return item
        return None

    def counts(self) -> Dict[T, int]:
        counts: Dict[T, int] = {}
        for value in self.values:
            if value is None:
                continue
            counts[value] = counts.get(value, 0) + 1
        return counts

    def count_for(self, value: Optional[T]) -> int:
        if value is None:
            return 0
        return self.counts().get(value, 0)


def largest_detection(detections: Iterable[Detection]) -> Optional[Detection]:
    selected = None
    selected_area = -1
    for det in detections:
        area = det.bbox.area if det.bbox else 0
        if area > selected_area:
            selected = det
            selected_area = area
    return selected

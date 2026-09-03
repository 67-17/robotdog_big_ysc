from __future__ import annotations

from typing import Iterable, List

from .common import BBox, Detection


class HSVDetector:
    def __init__(self, label: str, lower: Iterable[int], upper: Iterable[int], min_area: int):
        self.label = label
        self.lower = list(lower)
        self.upper = list(upper)
        self.min_area = int(min_area)

    def detect(self, image) -> List[Detection]:
        import cv2 as cv
        import numpy as np

        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, np.array(self.lower, dtype=np.uint8), np.array(self.upper, dtype=np.uint8))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        detections: List[Detection] = []
        img_area = max(1, image.shape[0] * image.shape[1])
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area < self.min_area:
                continue
            x, y, w, h = cv.boundingRect(cnt)
            detections.append(Detection(self.label, min(1.0, float(area) / img_area * 20.0), BBox(x, y, w, h)))
        return detections


class MultiHSVDetector:
    def __init__(self, detectors: Iterable[HSVDetector]):
        self.detectors = list(detectors)

    def detect(self, image) -> List[Detection]:
        detections: List[Detection] = []
        for detector in self.detectors:
            detections.extend(detector.detect(image))
        return detections


def build_competition_color_detectors(config: dict) -> dict[str, object]:
    v = config["vision"]
    cone = v["cone_hsv"]
    red1 = v["red_hsv_1"]
    red2 = v["red_hsv_2"]
    green = v["green_hsv"]
    return {
        "cone": HSVDetector("cone", cone["lower"], cone["upper"], cone["min_area"]),
        "red_bar": MultiHSVDetector(
            [
                HSVDetector("red_bar", red1["lower"], red1["upper"], red1["min_area"]),
                HSVDetector("red_bar", red2["lower"], red2["upper"], red2["min_area"]),
            ]
        ),
        "green_bar": HSVDetector("green_bar", green["lower"], green["upper"], green["min_area"]),
    }

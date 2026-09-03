from __future__ import annotations

from typing import Dict, Optional

from .common import BBox, Detection


class LetterRecognizer:
    def __init__(self, min_confidence: float = 0.48):
        self.min_confidence = min_confidence
        self.templates = None

    def recognize(self, image, roi: Optional[BBox] = None) -> Optional[Detection]:
        import cv2 as cv

        if self.templates is None:
            self.templates = self._build_templates()
        crop = image
        offset_x = 0
        offset_y = 0
        if roi is not None:
            crop = image[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w]
            offset_x, offset_y = roi.x, roi.y
        if crop.size == 0:
            return None
        target, bbox = self._normalize(crop)
        if target is None:
            return None
        scores = {letter: float(cv.matchTemplate(target, tmpl, cv.TM_CCOEFF_NORMED).max()) for letter, tmpl in self.templates.items()}
        label, confidence = max(scores.items(), key=lambda kv: kv[1])
        if confidence < self.min_confidence:
            return None
        if bbox is not None:
            bbox = BBox(bbox.x + offset_x, bbox.y + offset_y, bbox.w, bbox.h)
        return Detection(label, confidence, bbox)

    def _build_templates(self) -> Dict[str, object]:
        import cv2 as cv
        import numpy as np

        templates: Dict[str, object] = {}
        for letter in "ABCD":
            img = np.zeros((160, 160), dtype=np.uint8)
            scale = 4.0
            thickness = 8
            (tw, th), _ = cv.getTextSize(letter, cv.FONT_HERSHEY_SIMPLEX, scale, thickness)
            cv.putText(img, letter, ((160 - tw) // 2, (160 + th) // 2), cv.FONT_HERSHEY_SIMPLEX, scale, 255, thickness)
            templates[letter] = img
        return templates

    def _normalize(self, image):
        import cv2 as cv
        import numpy as np

        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) if image.ndim == 3 else image
        gray = cv.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        if cv.countNonZero(binary) < 50:
            _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None
        cnt = max(contours, key=cv.contourArea)
        x, y, w, h = cv.boundingRect(cnt)
        if w * h < 80:
            return None, None
        crop = binary[y : y + h, x : x + w]
        canvas = np.zeros((160, 160), dtype=np.uint8)
        scale = min(130 / max(w, 1), 130 / max(h, 1))
        resized = cv.resize(crop, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv.INTER_AREA)
        yy = (160 - resized.shape[0]) // 2
        xx = (160 - resized.shape[1]) // 2
        canvas[yy : yy + resized.shape[0], xx : xx + resized.shape[1]] = resized
        return canvas, BBox(x, y, w, h)

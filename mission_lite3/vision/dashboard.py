from __future__ import annotations

import math
from typing import Optional

from .common import BBox, Detection


class DashboardRecognizer:
    def __init__(self, min_radius: int = 45):
        self.min_radius = min_radius

    def recognize(self, image, roi: Optional[BBox] = None) -> Optional[Detection]:
        crop = image
        ox = 0
        oy = 0
        if roi is not None:
            crop = image[roi.y : roi.y + roi.h, roi.x : roi.x + roi.w]
            ox, oy = roi.x, roi.y
        if crop.size == 0:
            return None
        work = self._white_balance(crop)
        circle = self._find_circle(work)
        if circle is None:
            h, w = crop.shape[:2]
            circle = (w // 2, h // 2, min(w, h) // 2)
        cx, cy, radius = circle
        if radius < self.min_radius:
            return None
        angle = self._pointer_angle(work, cx, cy, radius)
        if angle is None:
            return None
        level, confidence = self._classify_angle(angle)
        bbox = BBox(int(cx - radius + ox), int(cy - radius + oy), int(radius * 2), int(radius * 2))
        return Detection(level, confidence, bbox)

    def _white_balance(self, image):
        import numpy as np

        if image.ndim != 3 or image.shape[2] != 3:
            return image
        pixels = image.reshape(-1, 3).astype(np.float32)
        means = pixels.mean(axis=0)
        if float(means.min()) <= 1.0:
            return image
        target = float(means.mean())
        gains = target / means
        balanced = np.clip(image.astype(np.float32) * gains, 0, 255)
        return balanced.astype(np.uint8)

    def _find_circle(self, image):
        import cv2 as cv
        import numpy as np

        h, w = image.shape[:2]
        max_dim = max(h, w)
        scale = min(1.0, 640.0 / max(1, max_dim))
        if scale < 1.0:
            small = cv.resize(image, (int(w * scale), int(h * scale)), interpolation=cv.INTER_AREA)
        else:
            small = image

        gray = cv.cvtColor(small, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (7, 7), 1.5)
        gray = cv.equalizeHist(gray)
        min_radius = max(12, int(self.min_radius * scale))
        max_radius = max(min_radius + 1, int(min(small.shape[:2]) * 0.52))
        min_dist = max(40, int(80 * scale))
        circles = cv.HoughCircles(
            gray,
            cv.HOUGH_GRADIENT,
            1.2,
            min_dist,
            param1=120,
            param2=28,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        if circles is None:
            return None
        candidates = []
        for x, y, radius in circles[0]:
            cx = float(x) / scale
            cy = float(y) / scale
            r = float(radius) / scale
            if r < self.min_radius:
                continue
            ring_score = self._ring_color_score(image, cx, cy, r)
            candidates.append((cx, cy, r, ring_score * 1000.0 + r))
        if not candidates:
            return None
        cx, cy, radius, _ = max(candidates, key=lambda c: c[3])
        return int(round(cx)), int(round(cy)), int(round(radius))

    def _ring_color_score(self, image, cx: float, cy: float, radius: float) -> float:
        import cv2 as cv
        import numpy as np

        h, w = image.shape[:2]
        yy, xx = np.ogrid[:h, :w]
        distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        annulus = (distance >= radius * 0.70) & (distance <= radius * 1.06)
        total = int(annulus.sum())
        if total == 0 or image.ndim != 3:
            return 0.0
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        saturated = (hsv[:, :, 1] > 50) & (hsv[:, :, 2] > 45)
        dark = hsv[:, :, 2] < 55
        return float(((saturated | dark) & annulus).sum()) / float(total)

    def _pointer_angle(self, image, cx: int, cy: int, radius: int) -> Optional[float]:
        import cv2 as cv
        import numpy as np

        x0 = max(0, int(cx - radius * 0.90))
        x1 = min(image.shape[1], int(cx + radius * 0.90))
        y0 = max(0, int(cy - radius * 0.90))
        y1 = min(image.shape[0], int(cy + radius * 0.90))
        crop = image[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
        gray = cv.GaussianBlur(gray, (3, 3), 0)
        dark_cutoff = max(45, int(np.percentile(gray, 35)))
        _, mask = cv.threshold(gray, dark_cutoff, 255, cv.THRESH_BINARY_INV)
        adaptive = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 21, 8)
        mask = cv.bitwise_or(mask, adaptive)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        center = np.array([cx - x0, cy - y0], dtype=np.float32)
        yy, xx = np.ogrid[: mask.shape[0], : mask.shape[1]]
        distance = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
        dial_mask = (distance >= radius * 0.08) & (distance <= radius * 0.88)
        mask = cv.bitwise_and(mask, mask, mask=dial_mask.astype(np.uint8) * 255)

        line_angle = self._pointer_angle_from_lines(mask, center, radius)
        if line_angle is not None:
            return line_angle
        return self._pointer_angle_from_scan(mask, center, radius)

    def _pointer_angle_from_lines(self, mask, center, radius: int) -> Optional[float]:
        import cv2 as cv
        import numpy as np

        threshold = max(18, int(radius * 0.18))
        min_line_length = max(18, int(radius * 0.34))
        max_line_gap = max(4, int(radius * 0.08))
        lines = cv.HoughLinesP(mask, 1, np.pi / 180, threshold, minLineLength=min_line_length, maxLineGap=max_line_gap)
        if lines is None:
            return None
        best_point = None
        best_score = -1.0
        for line in lines.reshape(-1, 4):
            p1 = np.array([line[0], line[1]], dtype=np.float32)
            p2 = np.array([line[2], line[3]], dtype=np.float32)
            d1 = float(np.linalg.norm(p1 - center))
            d2 = float(np.linalg.norm(p2 - center))
            near = min(d1, d2)
            far = max(d1, d2)
            if near > radius * 0.40 or far < radius * 0.34:
                continue
            length = float(np.linalg.norm(p2 - p1))
            farthest = p1 if d1 >= d2 else p2
            score = length + far - near * 1.5
            if score > best_score:
                best_score = score
                best_point = farthest
        if best_point is None:
            return None
        return self._angle_from_point(best_point, center)

    def _pointer_angle_from_scan(self, mask, center, radius: int) -> Optional[float]:
        import numpy as np

        angles = np.arange(0.0, 360.0, 2.0, dtype=np.float32)
        radii = np.linspace(radius * 0.16, radius * 0.86, 80, dtype=np.float32)
        weights = np.linspace(0.5, 1.0, len(radii), dtype=np.float32)
        best_angle = None
        best_score = -1.0
        h, w = mask.shape[:2]
        for angle in angles:
            radians = math.radians(float(angle))
            xs = np.rint(center[0] + np.cos(radians) * radii).astype(np.int32)
            ys = np.rint(center[1] - np.sin(radians) * radii).astype(np.int32)
            valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
            if not bool(valid.any()):
                continue
            samples = (mask[ys[valid], xs[valid]] > 0).astype(np.float32)
            score = float((samples * weights[valid]).sum())
            if score > best_score:
                best_score = score
                best_angle = float(angle)
        if best_angle is None or best_score < max(8.0, len(radii) * 0.16):
            return None
        return best_angle

    def _angle_from_point(self, point, center) -> float:
        dx = float(point[0] - center[0])
        dy = float(center[1] - point[1])
        return math.degrees(math.atan2(dy, dx)) % 360

    def _classify_angle(self, angle_deg: float) -> tuple[str, float]:
        if 145 <= angle_deg <= 245:
            return "偏低", 0.8
        if 45 <= angle_deg < 145:
            return "正常", 0.8
        if angle_deg >= 300 or angle_deg <= 45:
            return "偏高", 0.8
        return "正常", 0.55

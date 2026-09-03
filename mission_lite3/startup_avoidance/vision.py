from .model import BBox, Detection


class ConeDetector(object):
    def __init__(self, config):
        self.config = config

    def detect(self, image):
        import cv2
        import numpy as np

        hsv_config = self.config["hsv"]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array(hsv_config["lower"], dtype=np.uint8)
        upper = np.array(hsv_config["upper"], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        size = int(hsv_config["kernel_size"])
        kernel = np.ones((size, size), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contour_result = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = contour_result[-2]
        detections = []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < float(hsv_config["min_area"]):
                continue
            x, y, w, h = cv2.boundingRect(contour)
            detections.append(Detection(BBox(x, y, w, h), area))
        return sorted(
            detections, key=lambda item: item.contour_area, reverse=True
        )


def classify_zone(bbox, config):
    right = bbox.x + bbox.w
    center_x = bbox.x + bbox.w / 2.0
    zones = config["zones"]
    if right <= zones["safe_left_right_edge_max"]:
        return "safe_left"
    if bbox.x >= zones["safe_right_left_edge_min"]:
        return "safe_right"
    if zones["front_center_min"] <= center_x <= zones["front_center_max"]:
        return "front"
    return "side"

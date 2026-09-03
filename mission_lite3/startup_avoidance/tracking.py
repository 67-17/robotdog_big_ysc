import math

from .model import TrackUpdate, TrackView
from .vision import classify_zone


def bbox_iou(first, second):
    left = max(first.x, second.x)
    top = max(first.y, second.y)
    right = min(first.x + first.w, second.x + second.w)
    bottom = min(first.y + first.h, second.y + second.h)
    intersection = max(0, right - left) * max(0, bottom - top)
    union = first.w * first.h + second.w * second.h - intersection
    if union <= 0:
        return 0.0
    return float(intersection) / float(union)


def center_distance(first, second):
    first_x = first.x + first.w / 2.0
    first_y = first.y + first.h / 2.0
    second_x = second.x + second.w / 2.0
    second_y = second.y + second.h / 2.0
    return math.hypot(second_x - first_x, second_y - first_y)


def area_ratio(old_detection, new_detection):
    if old_detection.contour_area <= 0:
        return float("inf")
    return float(new_detection.contour_area) / float(
        old_detection.contour_area
    )


def association_cost(old_detection, new_detection, config):
    tracking = config.get("tracking", config)
    distance_cost = center_distance(
        old_detection.bbox, new_detection.bbox
    ) / float(tracking["max_center_distance_px"])
    ratio = area_ratio(old_detection, new_detection)
    if ratio <= 0:
        return float("inf")
    return distance_cost + abs(math.log(ratio))


class _Track(object):
    def __init__(self, track_id, detection, zone):
        self.track_id = track_id
        self.detection = detection
        self.zone = zone
        self.missing_frames = 0


class TargetTracker(object):
    def __init__(self, config):
        self.config = config
        self.tracking = config["tracking"]
        self._tracks = {}
        self._next_track_id = 1
        self._active_id = None

    @property
    def active_id(self):
        return self._active_id

    def set_active(self, track_id):
        if track_id not in self._tracks:
            raise ValueError("unknown track id: %s" % track_id)
        self._active_id = track_id

    def clear_active(self):
        self._active_id = None

    def update(self, detections):
        detections = list(detections)
        available = set(range(len(detections)))
        cleared_ids = []
        ambiguous = False

        track_ids = sorted(self._tracks)
        if self._active_id in self._tracks:
            track_ids.remove(self._active_id)
            track_ids.insert(0, self._active_id)

        for track_id in track_ids:
            track = self._tracks.get(track_id)
            if track is None:
                continue

            candidates = self._candidates(track, detections, available)
            is_active = track_id == self._active_id
            if is_active and self._is_ambiguous(candidates):
                ambiguous = True
                continue
            match_index = candidates[0][1] if candidates else None

            if match_index is not None:
                track.detection = detections[match_index]
                track.missing_frames = 0
                available.remove(match_index)
            elif is_active:
                if track.zone in ("safe_left", "safe_right"):
                    track.missing_frames += 1
                    if track.missing_frames >= self.tracking[
                        "max_missing_frames"
                    ]:
                        cleared_ids.append(track_id)
                        del self._tracks[track_id]
                else:
                    ambiguous = True
            else:
                track.missing_frames += 1
                if track.missing_frames >= self.tracking[
                    "max_missing_frames"
                ]:
                    del self._tracks[track_id]

        for detection_index in sorted(available):
            self._create_track(detections[detection_index])

        tracks = [
            self._track_view(self._tracks[track_id])
            for track_id in sorted(self._tracks)
        ]
        return TrackUpdate(tracks, sorted(cleared_ids), ambiguous)

    def _candidates(self, track, detections, available):
        candidates = []
        for detection_index in sorted(available):
            detection = detections[detection_index]
            if not self._eligible(track.detection, detection):
                continue
            cost = association_cost(track.detection, detection, self.tracking)
            candidates.append((cost, detection_index))
        return sorted(candidates)

    def _eligible(self, old_detection, new_detection):
        if bbox_iou(old_detection.bbox, new_detection.bbox) >= self.tracking[
            "min_iou"
        ]:
            return True
        ratio = area_ratio(old_detection, new_detection)
        return (
            center_distance(old_detection.bbox, new_detection.bbox)
            <= self.tracking["max_center_distance_px"]
            and self.tracking["min_area_ratio"]
            <= ratio
            <= self.tracking["max_area_ratio"]
        )

    def _is_ambiguous(self, candidates):
        if len(candidates) < 2:
            return False
        best_cost = candidates[0][0]
        second_cost = candidates[1][0]
        return second_cost <= best_cost * (
            1.0 + self.tracking["ambiguous_cost_fraction"]
        )

    def _create_track(self, detection):
        track_id = self._next_track_id
        self._next_track_id += 1
        self._tracks[track_id] = _Track(
            track_id,
            detection,
            classify_zone(detection.bbox, self.config),
        )

    def _track_view(self, track):
        track.zone = classify_zone(track.detection.bbox, self.config)
        return TrackView(
            track.track_id,
            track.detection,
            track.zone,
            track.missing_frames,
        )

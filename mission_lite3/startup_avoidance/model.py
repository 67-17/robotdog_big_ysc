from collections import namedtuple


BBox = namedtuple("BBox", "x y w h")
Detection = namedtuple("Detection", "bbox contour_area")
TrackView = namedtuple("TrackView", "track_id detection zone missing_frames")
TrackUpdate = namedtuple("TrackUpdate", "tracks cleared_ids ambiguous")
SensorFrame = namedtuple(
    "SensorFrame",
    "now tracks cleared_ids ambiguous ultrasound_m odom_x odom_y yaw image_age_s ultrasound_age_s odom_age_s",
)
Decision = namedtuple("Decision", "state vx vy wz reason finished fault")


def bbox_right(bbox):
    return bbox.x + bbox.w


def bbox_center_x(bbox):
    return bbox.x + bbox.w / 2.0


def bbox_area(bbox):
    return bbox.w * bbox.h

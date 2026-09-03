from __future__ import annotations

import json
import itertools
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Mapping, Optional

import cv2
import numpy as np

from .wide_camera import BoxParallelResult, cardboard_mask, detect_box_parallel


LETTERS = ("A", "B", "C", "D")

DEFAULT_BOX_CENTER_CONFIG = {
    "enabled": False,
    "frames_per_measurement": 7,
    "min_valid_frames": 4,
    "max_center_range_fraction": 0.03,
    "tolerance_fraction": 0.05,
    "max_corrections": 3,
    "strafe_speed_mps": 0.08,
    "max_single_strafe_m": 0.25,
    "max_total_strafe_m": 0.75,
    "adjacent_box_spacing_m": 0.30,
    "pickup_m_per_pixel": 0.001,
    "positive_error_strafe_sign": -1,
    "settle_seconds": 0.5,
    "placement_roi": [0.0, 0.28, 1.0, 1.0],
    "placement_min_span_fraction": 0.30,
    "placement_min_center_gap_fraction": 0.05,
    "placement_tracking_enabled": True,
    "placement_tracking_min_separators": 2,
    "placement_tracking_max_scale_change_fraction": 0.20,
    "placement_tracking_max_residual_fraction": 0.025,
    "placement_tracking_min_motion_gain": 0.20,
    "placement_tracking_max_motion_gain": 1.50,
    "strafe_pose_hold_enabled": True,
    "forward_hold_kp_s": 1.0,
    "max_vx_correction_mps": 0.04,
    "forward_deadband_m": 0.003,
    "max_forward_drift_m": 0.15,
    "yaw_hold_kp_s": 1.2,
    "max_wz_correction_rad_s": 0.12,
    "yaw_deadband_deg": 0.30,
    "max_yaw_drift_deg": 5.0,
    "placement_letter_min_confidence": 0.50,
    "placement_label_min_area_fraction": 0.00035,
    "placement_label_max_area_fraction": 0.12,
    "placement_white_max_saturation": 90,
    "placement_white_min_value": 165,
    "placement_glyph_fallback_enabled": True,
    "placement_glyph_roi": [0.0, 0.48, 1.0, 0.95],
    "placement_glyph_min_width_fraction": 0.012,
    "placement_glyph_max_width_fraction": 0.080,
    "placement_glyph_min_height_fraction": 0.030,
    "placement_glyph_max_height_fraction": 0.150,
    "placement_glyph_min_aspect": 0.30,
    "placement_glyph_max_aspect": 1.40,
    "placement_glyph_expand_x": 3.0,
    "placement_glyph_expand_y": 2.5,
    "placement_label_row_fallback_enabled": True,
    "placement_label_row_gray_min": 180,
    "placement_label_row_roi": [0.15, 0.58, 0.85, 0.80],
    "placement_label_row_min_area_fraction": 0.0015,
    "placement_label_row_max_area_fraction": 0.015,
    "placement_label_row_min_fill": 0.72,
    "placement_label_row_min_aspect": 0.75,
    "placement_label_row_max_aspect": 1.80,
    "placement_label_row_min_gap_fraction": 0.055,
    "placement_label_row_max_gap_fraction": 0.16,
    "placement_label_row_max_y_range_fraction": 0.045,
    "placement_label_row_min_size_ratio": 0.70,
    "placement_label_row_anchor_confidence": 0.60,
    "placement_label_row_anchorless_order_enabled": False,
    "placement_label_row_anchorless_min_confidence": 0.75,
    "recognition_run_log_dir": "box_recognition_runs",
    "alignment_run_log_dir": "box_center_alignment_runs",
    "fallback_enabled": True,
    "fallback_offsets_m": {
        "A": 0.15,
        "B": -0.15,
        "C": -0.45,
        "D": -0.75,
    },
}


@dataclass(frozen=True)
class PlacementCandidate:
    center: tuple[float, float]
    label_bbox: Optional[tuple[int, int, int, int]]
    box_bbox: tuple[int, int, int, int]
    recognized_letter: str
    confidence: float


@dataclass(frozen=True)
class PlacementLetterCandidate:
    center: tuple[float, float]
    label_bbox: tuple[int, int, int, int]
    recognized_letter: str
    confidence: float


@dataclass(frozen=True)
class PlacementLetterFrameResult:
    ok: bool
    reason: str
    frame_width: int
    frame_height: int
    candidates: tuple[PlacementLetterCandidate, ...] = ()


@dataclass(frozen=True)
class BoxCenterFrameResult:
    ok: bool
    reason: str
    mode: str
    frame_width: int
    frame_height: int
    centers: dict[str, tuple[float, float]] = field(default_factory=dict)
    target_label: Optional[str] = None
    target_center: Optional[tuple[float, float]] = None
    spacing_px: tuple[float, ...] = ()
    confidence: float = 0.0
    candidates: tuple[PlacementCandidate, ...] = ()
    box_x_range: Optional[tuple[int, int]] = None
    detector_reason: str = ""


@dataclass(frozen=True)
class BoxCenterMeasurement:
    ok: bool
    reason: str
    mode: str
    centers: dict[str, tuple[float, float]]
    target_label: Optional[str]
    target_center: Optional[tuple[float, float]]
    frame_width: int
    frame_height: int
    stable_frames: int
    requested_frames: int
    confidence: float
    spacing_px: tuple[float, ...]
    center_errors_px: dict[str, float]
    target_error_px: Optional[float]
    target_error_fraction: Optional[float]
    failure_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class BoxCenterAlignmentResult:
    ok: bool
    reason: str
    mode: str
    target_label: Optional[str]
    correction_count: int
    motion_command_count: int
    measurement_count: int
    initial_error_px: Optional[float]
    final_error_px: Optional[float]
    visual_strafe_m: float
    net_strafe_m: float
    rollback_attempted: bool
    rollback_ok: bool
    run_dir: Optional[str]


def _config_with_defaults(config: Mapping[str, object]) -> dict[str, object]:
    merged = dict(DEFAULT_BOX_CENTER_CONFIG)
    merged.update(config)
    return merged


def _normalized_roi(
    value: object,
) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return 0.0, 0.28, 1.0, 1.0
    x0, y0, x1, y1 = (float(item) for item in value)
    return (
        max(0.0, min(1.0, x0)),
        max(0.0, min(1.0, y0)),
        max(0.0, min(1.0, x1)),
        max(0.0, min(1.0, y1)),
    )


def map_placement_candidates(
    candidates: list[PlacementCandidate],
    *,
    frame_width: int,
    frame_height: int,
    target_letter: Optional[str] = None,
    placement_roi: object = (0.0, 0.28, 1.0, 1.0),
    min_center_gap_fraction: float = 0.05,
) -> BoxCenterFrameResult:
    """Filter the background row, then map physical left-to-right order to A-D."""
    if target_letter is not None and target_letter not in LETTERS:
        return BoxCenterFrameResult(
            False,
            "invalid_target_letter",
            "placement",
            frame_width,
            frame_height,
            target_label=target_letter,
        )
    x0, y0, x1, y1 = _normalized_roi(placement_roi)
    filtered = [
        candidate
        for candidate in candidates
        if candidate.recognized_letter in LETTERS
        and candidate.box_bbox is not None
        and x0 * frame_width <= candidate.center[0] <= x1 * frame_width
        and y0 * frame_height
        <= (
            candidate.label_bbox[1] + candidate.label_bbox[3] / 2.0
            if candidate.label_bbox is not None
            else candidate.center[1]
        )
        <= y1 * frame_height
    ]
    filtered.sort(key=lambda candidate: candidate.center[0])
    if len(filtered) < 4:
        return BoxCenterFrameResult(
            False,
            "missing_boxes",
            "placement",
            frame_width,
            frame_height,
            target_label=target_letter,
            candidates=tuple(filtered),
        )
    if len(filtered) > 4:
        return BoxCenterFrameResult(
            False,
            "duplicate_or_extra_boxes",
            "placement",
            frame_width,
            frame_height,
            target_label=target_letter,
            candidates=tuple(filtered),
        )
    spacing = tuple(
        filtered[index + 1].center[0] - filtered[index].center[0]
        for index in range(3)
    )
    minimum_gap = max(1.0, frame_width * float(min_center_gap_fraction))
    if any(gap < minimum_gap for gap in spacing):
        return BoxCenterFrameResult(
            False,
            "duplicate_boxes",
            "placement",
            frame_width,
            frame_height,
            target_label=target_letter,
            candidates=tuple(filtered),
        )
    centers = {
        letter: tuple(float(value) for value in candidate.center)
        for letter, candidate in zip(LETTERS, filtered)
    }
    target_center = centers.get(target_letter) if target_letter else None
    return BoxCenterFrameResult(
        True,
        "",
        "placement",
        frame_width,
        frame_height,
        centers=centers,
        target_label=target_letter,
        target_center=target_center,
        spacing_px=spacing,
        confidence=float(np.mean([candidate.confidence for candidate in filtered])),
        candidates=tuple(filtered),
    )


def _infer_box_region(
    frame: np.ndarray,
    label_bbox: tuple[int, int, int, int],
) -> tuple[Optional[tuple[int, int, int, int]], float]:
    height, width = frame.shape[:2]
    x, y, label_width, label_height = label_bbox
    label_cx = x + label_width / 2.0
    label_cy = y + label_height / 2.0
    pad_x = max(label_width, int(round(0.035 * width)))
    top = max(0, y - label_height)
    bottom = min(height, y + max(label_height * 5, int(round(0.22 * height))))
    left = max(0, x - pad_x)
    right = min(width, x + label_width + pad_x)
    if bottom <= top or right <= left:
        return None, 0.0

    local = frame[top:bottom, left:right]
    gray = cv2.cvtColor(local, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        np.ones((7, 7), dtype=np.uint8),
        iterations=2,
    )
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    options: list[tuple[float, tuple[int, int, int, int]]] = []
    for contour in contours:
        bx, by, bw, bh = cv2.boundingRect(contour)
        global_box = (left + bx, top + by, bw, bh)
        gx, gy, gw, gh = global_box
        contains_label = (
            gx <= label_cx <= gx + gw
            and gy <= label_cy <= gy + gh
        )
        if not contains_label:
            continue
        if gw < label_width * 1.15 or gh < label_height * 1.7:
            continue
        if gh > height * 0.65 or gw > width * 0.38:
            continue
        area_score = min(1.0, (gw * gh) / max(1.0, width * height * 0.035))
        center_score = max(0.0, 1.0 - abs((gx + gw / 2.0) - label_cx) / max(gw, 1))
        options.append((0.55 * area_score + 0.45 * center_score, global_box))
    if options:
        best_score, best_box = max(options, key=lambda item: item[0])
        return best_box, best_score

    # Cardboard texture can lack a closed outer contour.  Require measurable
    # warm-colour or edge support below the tag before using a conservative
    # local box extent centered on the tag.
    below_y0 = min(height, y + label_height)
    below_y1 = min(height, below_y0 + max(label_height * 2, int(0.10 * height)))
    below_x0 = max(0, x - label_width // 2)
    below_x1 = min(width, x + label_width + label_width // 2)
    support = frame[below_y0:below_y1, below_x0:below_x1]
    if support.size == 0:
        return None, 0.0
    warm_ratio = float(np.count_nonzero(cardboard_mask(support))) / support.shape[0] / support.shape[1]
    support_gray = cv2.cvtColor(support, cv2.COLOR_BGR2GRAY)
    edge_ratio = float(np.count_nonzero(cv2.Canny(support_gray, 40, 120))) / support_gray.size
    support_score = min(1.0, max(warm_ratio / 0.20, edge_ratio / 0.06))
    if support_score < 0.35:
        return None, support_score
    inferred_width = min(int(round(width * 0.24)), max(label_width * 2, int(width * 0.10)))
    inferred_height = min(int(round(height * 0.34)), max(label_height * 4, int(height * 0.16)))
    inferred_x = max(0, min(width - inferred_width, int(round(label_cx - inferred_width / 2))))
    inferred_y = max(0, min(height - inferred_height, y - label_height // 2))
    return (inferred_x, inferred_y, inferred_width, inferred_height), support_score


def _placement_white_mask(
    frame: np.ndarray,
    config: Mapping[str, object],
) -> np.ndarray:
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(
        hsv,
        np.asarray(
            (0, 0, int(config["placement_white_min_value"])),
            dtype=np.uint8,
        ),
        np.asarray(
            (179, int(config["placement_white_max_saturation"]), 255),
            dtype=np.uint8,
        ),
    )
    kernel_size = max(3, int(round(min(width, height) * 0.005)) | 1)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel, iterations=1)


def _nested_letter_candidate(
    candidate: PlacementLetterCandidate,
    existing: PlacementLetterCandidate,
    *,
    frame_width: int,
    frame_height: int,
) -> bool:
    x1, y1, width1, height1 = candidate.label_bbox
    x2, y2, width2, height2 = existing.label_bbox
    overlap_width = max(0, min(x1 + width1, x2 + width2) - max(x1, x2))
    overlap_height = max(0, min(y1 + height1, y2 + height2) - max(y1, y2))
    overlap = overlap_width * overlap_height
    smaller_area = min(width1 * height1, width2 * height2)
    if smaller_area <= 0 or overlap / smaller_area < 0.80:
        return False
    return (
        abs(candidate.center[0] - existing.center[0])
        <= max(4.0, frame_width * 0.01)
        and abs(candidate.center[1] - existing.center[1])
        <= max(4.0, frame_height * 0.02)
    )


def _glyph_letter_candidates(
    frame: np.ndarray,
    contours: list[np.ndarray],
    _hierarchy: Optional[np.ndarray],
    config: Mapping[str, object],
) -> list[PlacementLetterCandidate]:
    """Recover letters from nested glyph contours when the white card merges."""
    if not bool(config["placement_glyph_fallback_enabled"]):
        return []
    from PIL import Image

    from .inspection_runtime.letter_recognition import recognize_letter_roi

    height, width = frame.shape[:2]
    roi_x0, roi_y0, roi_x1, roi_y1 = _normalized_roi(
        config["placement_glyph_roi"]
    )
    minimum_confidence = float(config["placement_letter_min_confidence"])
    minimum_width = width * float(config["placement_glyph_min_width_fraction"])
    maximum_width = width * float(config["placement_glyph_max_width_fraction"])
    minimum_height = height * float(config["placement_glyph_min_height_fraction"])
    maximum_height = height * float(config["placement_glyph_max_height_fraction"])
    minimum_aspect = float(config["placement_glyph_min_aspect"])
    maximum_aspect = float(config["placement_glyph_max_aspect"])
    expand_x = float(config["placement_glyph_expand_x"])
    expand_y = float(config["placement_glyph_expand_y"])
    candidates: list[PlacementLetterCandidate] = []

    for contour in contours:
        x, y, glyph_width, glyph_height = cv2.boundingRect(contour)
        if not minimum_width <= glyph_width <= maximum_width:
            continue
        if not minimum_height <= glyph_height <= maximum_height:
            continue
        aspect = glyph_width / max(1.0, glyph_height)
        if not minimum_aspect <= aspect <= maximum_aspect:
            continue
        glyph_center_x = x + glyph_width / 2.0
        glyph_center_y = y + glyph_height / 2.0
        if not (
            roi_x0 * width <= glyph_center_x <= roi_x1 * width
            and roi_y0 * height <= glyph_center_y <= roi_y1 * height
        ):
            continue

        crop_width = max(glyph_width + 2, int(round(glyph_width * expand_x)))
        crop_height = max(glyph_height + 2, int(round(glyph_height * expand_y)))
        crop_x0 = max(0, int(round(glyph_center_x - crop_width / 2.0)))
        crop_y0 = max(0, int(round(glyph_center_y - crop_height / 2.0)))
        crop_x1 = min(width, crop_x0 + crop_width)
        crop_y1 = min(height, crop_y0 + crop_height)
        crop = frame[crop_y0:crop_y1, crop_x0:crop_x1]
        if crop.size == 0:
            continue
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        letter_result = recognize_letter_roi(
            Image.fromarray(rgb),
            min_confidence=minimum_confidence,
        )
        recognized = letter_result.get("label")
        confidence = float(letter_result.get("confidence", 0.0))
        center_px = letter_result.get("center_px")
        if (
            recognized not in LETTERS
            or confidence < minimum_confidence
            or not isinstance(center_px, (list, tuple))
            or len(center_px) != 2
        ):
            continue
        letter_center = (
            crop_x0 + float(center_px[0]),
            crop_y0 + float(center_px[1]),
        )
        if not all(math.isfinite(value) for value in letter_center):
            continue
        candidates.append(
            PlacementLetterCandidate(
                center=letter_center,
                label_bbox=(
                    crop_x0,
                    crop_y0,
                    crop_x1 - crop_x0,
                    crop_y1 - crop_y0,
                ),
                recognized_letter=str(recognized),
                confidence=confidence,
            )
        )
    return candidates


def detect_labeled_placement_candidates(
    frame: np.ndarray,
    config: Mapping[str, object],
) -> PlacementLetterFrameResult:
    """Return only directly OCR-recognized A-D placement tags."""
    if (
        not isinstance(frame, np.ndarray)
        or frame.ndim != 3
        or frame.shape[2] != 3
        or frame.shape[0] == 0
        or frame.shape[1] == 0
    ):
        return PlacementLetterFrameResult(False, "invalid_frame", 0, 0)

    cfg = _config_with_defaults(config)
    height, width = frame.shape[:2]
    white = _placement_white_mask(frame, cfg)
    contours, hierarchy = cv2.findContours(
        white,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    from PIL import Image

    from .inspection_runtime.letter_recognition import recognize_letter_roi

    candidates: list[PlacementLetterCandidate] = []
    image_area = float(width * height)
    roi_x0, roi_y0, roi_x1, roi_y1 = _normalized_roi(cfg["placement_roi"])
    minimum_confidence = float(cfg["placement_letter_min_confidence"])
    minimum_area = float(cfg["placement_label_min_area_fraction"])
    maximum_area = float(cfg["placement_label_max_area_fraction"])
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        area_fraction = box_width * box_height / image_area
        aspect = box_width / max(1.0, box_height)
        if not minimum_area <= area_fraction <= maximum_area:
            continue
        if not 0.55 <= aspect <= 3.5:
            continue
        fill = float(
            np.count_nonzero(white[y : y + box_height, x : x + box_width])
        ) / max(1, box_width * box_height)
        if fill < 0.52:
            continue
        label_center_x = x + box_width / 2.0
        label_center_y = y + box_height / 2.0
        if not (
            roi_x0 * width <= label_center_x <= roi_x1 * width
            and roi_y0 * height <= label_center_y <= roi_y1 * height
        ):
            continue
        crop = frame[y : y + box_height, x : x + box_width]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        letter_result = recognize_letter_roi(
            Image.fromarray(rgb),
            min_confidence=minimum_confidence,
        )
        recognized = letter_result.get("label")
        confidence = float(letter_result.get("confidence", 0.0))
        center_px = letter_result.get("center_px")
        if (
            recognized not in LETTERS
            or confidence < minimum_confidence
            or not isinstance(center_px, (list, tuple))
            or len(center_px) != 2
        ):
            continue
        glyph_center = (x + float(center_px[0]), y + float(center_px[1]))
        if not all(math.isfinite(value) for value in glyph_center):
            continue
        candidates.append(
            PlacementLetterCandidate(
                center=glyph_center,
                label_bbox=(x, y, box_width, box_height),
                recognized_letter=str(recognized),
                confidence=confidence,
            )
        )

    candidates.extend(
        _glyph_letter_candidates(
            frame,
            contours,
            hierarchy,
            cfg,
        )
    )

    # RETR_LIST may return nested white contours for one physical label.
    deduplicated: list[PlacementLetterCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.confidence, reverse=True):
        if any(
            _nested_letter_candidate(
                candidate,
                existing,
                frame_width=width,
                frame_height=height,
            )
            for existing in deduplicated
        ):
            continue
        deduplicated.append(candidate)
    deduplicated.sort(key=lambda item: item.center[0])
    if not deduplicated:
        return PlacementLetterFrameResult(
            False,
            "no_recognized_letter",
            width,
            height,
        )
    return PlacementLetterFrameResult(
        True,
        "",
        width,
        height,
        candidates=tuple(deduplicated),
    )


def _placement_label_row(
    frame: np.ndarray,
    config: Mapping[str, object],
    direct: PlacementLetterFrameResult,
) -> tuple[PlacementLetterCandidate, ...]:
    """Recover the full A-D row only when direct OCR anchors its identity."""
    cfg = _config_with_defaults(config)
    if not bool(cfg["placement_label_row_fallback_enabled"]):
        return ()
    height, width = frame.shape[:2]
    image_area = float(width * height)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(
        gray,
        int(cfg["placement_label_row_gray_min"]),
        255,
    )
    roi_x0, roi_y0, roi_x1, roi_y1 = _normalized_roi(
        cfg["placement_label_row_roi"]
    )
    mask[: int(round(roi_y0 * height)), :] = 0
    mask[int(round(roi_y1 * height)) :, :] = 0
    mask[:, : int(round(roi_x0 * width))] = 0
    mask[:, int(round(roi_x1 * width)) :] = 0
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    rectangles: list[tuple[float, float, int, int, tuple[int, int, int, int]]] = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        area_fraction = box_width * box_height / image_area
        aspect = box_width / max(1.0, box_height)
        fill = cv2.contourArea(contour) / max(1.0, box_width * box_height)
        if not (
            float(cfg["placement_label_row_min_area_fraction"])
            <= area_fraction
            <= float(cfg["placement_label_row_max_area_fraction"])
        ):
            continue
        if not (
            float(cfg["placement_label_row_min_aspect"])
            <= aspect
            <= float(cfg["placement_label_row_max_aspect"])
        ):
            continue
        if fill < float(cfg["placement_label_row_min_fill"]):
            continue
        rectangles.append(
            (
                x + box_width / 2.0,
                y + box_height / 2.0,
                box_width,
                box_height,
                (x, y, box_width, box_height),
            )
        )

    best = None
    for raw_row in itertools.combinations(rectangles, len(LETTERS)):
        row = tuple(sorted(raw_row, key=lambda item: item[0]))
        x_values = np.asarray([item[0] for item in row], dtype=np.float64)
        y_values = np.asarray([item[1] for item in row], dtype=np.float64)
        widths = np.asarray([item[2] for item in row], dtype=np.float64)
        heights = np.asarray([item[3] for item in row], dtype=np.float64)
        gaps = np.diff(x_values)
        if (
            float(np.min(gaps))
            < width * float(cfg["placement_label_row_min_gap_fraction"])
            or float(np.max(gaps))
            > width * float(cfg["placement_label_row_max_gap_fraction"])
        ):
            continue
        if float(np.ptp(y_values)) > (
            height * float(cfg["placement_label_row_max_y_range_fraction"])
        ):
            continue
        size_ratio = min(
            float(np.min(widths) / np.max(widths)),
            float(np.min(heights) / np.max(heights)),
        )
        if size_ratio < float(cfg["placement_label_row_min_size_ratio"]):
            continue
        gap_score = 1.0 - min(
            1.0,
            float(np.std(gaps)) / max(float(np.mean(gaps)), 1.0),
        )
        y_score = 1.0 - min(
            1.0,
            float(np.std(y_values))
            / max(
                height * float(cfg["placement_label_row_max_y_range_fraction"]),
                1.0,
            ),
        )
        score = 0.50 * gap_score + 0.25 * y_score + 0.25 * size_ratio
        if best is None or score > best[0]:
            best = (score, row)
    if best is None:
        return ()

    geometry_score, row = best
    anchor_minimum = float(cfg["placement_label_row_anchor_confidence"])
    matched_anchors: list[PlacementLetterCandidate] = []
    for direct_candidate in direct.candidates:
        nearest_index = min(
            range(len(row)),
            key=lambda index: abs(row[index][0] - direct_candidate.center[0]),
        )
        nearest = row[nearest_index]
        proximity_limit = max(8.0, 0.60 * nearest[2])
        if abs(nearest[0] - direct_candidate.center[0]) > proximity_limit:
            continue
        if direct_candidate.recognized_letter != LETTERS[nearest_index]:
            return ()
        if direct_candidate.confidence >= anchor_minimum:
            matched_anchors.append(direct_candidate)
    if matched_anchors:
        confidence = min(
            0.90,
            0.45 * max(item.confidence for item in matched_anchors)
            + 0.55 * geometry_score,
        )
    else:
        if not bool(cfg["placement_label_row_anchorless_order_enabled"]):
            return ()
        minimum = float(
            cfg["placement_label_row_anchorless_min_confidence"]
        )
        if geometry_score < minimum:
            return ()
        confidence = min(0.82, geometry_score)
    return tuple(
        PlacementLetterCandidate(
            center=(item[0], item[1]),
            label_bbox=item[4],
            recognized_letter=letter,
            confidence=confidence,
        )
        for letter, item in zip(LETTERS, row)
    )


def detect_placement_letter_candidates(
    frame: np.ndarray,
    config: Mapping[str, object],
) -> PlacementLetterFrameResult:
    direct = detect_labeled_placement_candidates(frame, config)
    if direct.reason == "invalid_frame":
        return direct
    recovered = _placement_label_row(frame, config, direct)
    if not recovered:
        return direct
    return PlacementLetterFrameResult(
        True,
        "",
        direct.frame_width,
        direct.frame_height,
        candidates=recovered,
    )


def annotate_placement_letters(
    frame: np.ndarray,
    result: PlacementLetterFrameResult,
    target_letter: Optional[str],
    action: str,
) -> np.ndarray:
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    image_center = (width // 2, height // 2)
    cv2.drawMarker(
        annotated,
        image_center,
        (255, 255, 0),
        cv2.MARKER_CROSS,
        24,
        2,
    )
    for candidate in result.candidates:
        x, y, box_width, box_height = candidate.label_bbox
        is_target = candidate.recognized_letter == target_letter
        color = (0, 220, 255) if is_target else (0, 210, 0)
        cv2.rectangle(
            annotated,
            (x, y),
            (x + box_width, y + box_height),
            color,
            2,
        )
        glyph_center = (
            int(round(candidate.center[0])),
            int(round(candidate.center[1])),
        )
        cv2.drawMarker(
            annotated,
            glyph_center,
            color,
            cv2.MARKER_TILTED_CROSS,
            18,
            2,
        )
        cv2.putText(
            annotated,
            f"{candidate.recognized_letter} {candidate.confidence:.2f}",
            (x, max(18, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        annotated,
        f"target={target_letter or '-'} action={action}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


def _detect_labeled_placement_box_centers(
    frame: np.ndarray,
    config: Mapping[str, object],
    *,
    target_letter: Optional[str] = None,
) -> BoxCenterFrameResult:
    letter_result = detect_labeled_placement_candidates(frame, config)
    if letter_result.reason == "invalid_frame":
        return BoxCenterFrameResult(False, "invalid_frame", "placement", 0, 0)

    cfg = _config_with_defaults(config)
    white = _placement_white_mask(frame, cfg)
    candidates: list[PlacementCandidate] = []
    for letter_candidate in letter_result.candidates:
        box_bbox, support_confidence = _infer_box_region(
            frame,
            letter_candidate.label_bbox,
        )
        if box_bbox is None:
            continue
        x, y, label_width, label_height = letter_candidate.label_bbox
        fill = float(
            np.count_nonzero(white[y : y + label_height, x : x + label_width])
        ) / max(1, label_width * label_height)
        bx, by, box_width, box_height = box_bbox
        confidence = min(
            1.0,
            0.45 * letter_candidate.confidence
            + 0.25 * min(1.0, fill)
            + 0.30 * support_confidence,
        )
        candidates.append(
            PlacementCandidate(
                center=(bx + box_width / 2.0, by + box_height / 2.0),
                label_bbox=letter_candidate.label_bbox,
                box_bbox=box_bbox,
                recognized_letter=letter_candidate.recognized_letter,
                confidence=confidence,
            )
        )
    return map_placement_candidates(
        candidates,
        frame_width=letter_result.frame_width,
        frame_height=letter_result.frame_height,
        target_letter=target_letter,
        placement_roi=cfg["placement_roi"],
        min_center_gap_fraction=float(cfg["placement_min_center_gap_fraction"]),
    )


def _longest_contiguous_run(
    points: list[tuple[int, int]],
    *,
    max_gap: int,
) -> list[tuple[int, int]]:
    if not points:
        return []
    ordered = sorted(points, key=lambda point: point[0])
    runs: list[list[tuple[int, int]]] = [[ordered[0]]]
    for point in ordered[1:]:
        if point[0] - runs[-1][-1][0] <= max_gap:
            runs[-1].append(point)
        else:
            runs.append([point])
    return max(runs, key=lambda run: (run[-1][0] - run[0][0], len(run)))


def _cardboard_top_span(
    frame: np.ndarray,
    *,
    min_span_fraction: float,
    max_span_fraction: float = 0.90,
) -> tuple[Optional[tuple[int, int]], Optional[int], float, str]:
    """Find a lower-scene cardboard face without requiring a wide seam."""
    height, width = frame.shape[:2]
    mask = cardboard_mask(frame)
    search_y0 = int(round(0.45 * height))
    search_y1 = int(round(0.72 * height))
    occupancy_depth = max(40, int(round(0.19 * height)))
    points: list[tuple[int, int]] = []
    for x in range(int(round(0.10 * width)), int(round(0.90 * width))):
        candidates = np.flatnonzero(mask[search_y0:search_y1, x])
        if candidates.size == 0:
            continue
        y = search_y0 + int(candidates[0])
        bottom = min(height, y + occupancy_depth)
        occupancy = float(np.count_nonzero(mask[y:bottom, x])) / max(1, bottom - y)
        if occupancy >= 0.65:
            points.append((x, y))
    run = _longest_contiguous_run(
        points,
        # The dark gap between adjacent cartons interrupts the colour mask.
        # Bridge a narrow separator while keeping genuinely separate objects
        # in different runs.
        max_gap=max(10, int(round(0.015 * width))),
    )
    if len(run) < max(60, int(round(0.06 * width))):
        return None, None, 0.0, "cardboard_top_not_found"
    median_y = float(np.median([point[1] for point in run]))
    run = [point for point in run if abs(point[1] - median_y) < 45.0]
    if len(run) < max(60, int(round(0.06 * width))):
        return None, None, 0.0, "cardboard_top_inconsistent"
    x0 = min(point[0] for point in run)
    x1 = max(point[0] for point in run)
    span_fraction = (x1 - x0) / max(1.0, width)
    if span_fraction < min_span_fraction:
        return None, None, 0.0, "cardboard_span_too_small"
    if span_fraction > max_span_fraction:
        return None, None, 0.0, "cardboard_span_too_large"
    confidence = min(1.0, span_fraction / max(min_span_fraction * 1.5, 1e-6))
    return (x0, x1), int(round(median_y)), confidence, ""


def _cluster_vertical_lines(
    lines: np.ndarray,
    *,
    x_min: float,
    x_max: float,
    frame_height: int,
    cluster_gap_px: float,
) -> list[tuple[float, float]]:
    candidates: list[tuple[float, float]] = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = math.hypot(dx, dy)
        if abs(dy) < 0.10 * frame_height:
            continue
        if abs(dx) > 0.14 * abs(dy):
            continue
        center_x = 0.5 * float(x1 + x2)
        if x_min <= center_x <= x_max:
            candidates.append((center_x, length))
    candidates.sort(key=lambda item: item[0])
    clusters: list[list[tuple[float, float]]] = []
    for candidate in candidates:
        if clusters and candidate[0] - clusters[-1][-1][0] <= cluster_gap_px:
            clusters[-1].append(candidate)
        else:
            clusters.append([candidate])
    output: list[tuple[float, float]] = []
    for cluster in clusters:
        total_length = sum(item[1] for item in cluster)
        center_x = sum(item[0] * item[1] for item in cluster) / max(total_length, 1e-6)
        output.append((center_x, total_length))
    return output


def _placement_separator_geometry(
    frame: np.ndarray,
    *,
    min_span_fraction: float,
) -> tuple[
    Optional[tuple[int, int]],
    Optional[int],
    Optional[int],
    float,
    list[tuple[float, float]],
    str,
]:
    height, width = frame.shape[:2]
    x_range, top_y, span_confidence, reason = _cardboard_top_span(
        frame,
        min_span_fraction=min_span_fraction,
    )
    if x_range is None or top_y is None:
        return None, None, None, span_confidence, [], reason

    x0, x1 = x_range
    span = float(x1 - x0)
    face_bottom = min(height - 1, top_y + int(round(0.29 * height)))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 35, 110)
    roi = np.zeros_like(edges)
    x_pad = int(round(0.03 * width))
    cv2.rectangle(
        roi,
        (max(0, x0 - x_pad), max(0, top_y - int(round(0.02 * height)))),
        (min(width - 1, x1 + x_pad), face_bottom),
        255,
        -1,
    )
    lines = cv2.HoughLinesP(
        cv2.bitwise_and(edges, roi),
        1,
        np.pi / 360.0,
        threshold=max(24, int(round(0.045 * height))),
        minLineLength=max(50, int(round(0.10 * height))),
        maxLineGap=max(12, int(round(0.035 * height))),
    )
    if lines is None:
        return (
            x_range,
            top_y,
            face_bottom,
            span_confidence,
            [],
            "placement_separators_not_found",
        )
    clusters = _cluster_vertical_lines(
        lines,
        x_min=x0 + 0.06 * span,
        x_max=x1 - 0.06 * span,
        frame_height=height,
        cluster_gap_px=max(8.0, 0.011 * width),
    )
    return x_range, top_y, face_bottom, span_confidence, clusters, ""


def _detect_unlabelled_placement_box_centers(
    frame: np.ndarray,
    config: Mapping[str, object],
    *,
    target_letter: Optional[str] = None,
) -> BoxCenterFrameResult:
    height, width = frame.shape[:2]
    cfg = _config_with_defaults(config)
    (
        x_range,
        top_y,
        face_bottom,
        span_confidence,
        clusters,
        reason,
    ) = _placement_separator_geometry(
        frame,
        min_span_fraction=float(cfg["placement_min_span_fraction"]),
    )
    if reason or x_range is None or top_y is None or face_bottom is None:
        return BoxCenterFrameResult(
            False,
            reason,
            "placement",
            width,
            height,
            target_label=target_letter,
            box_x_range=x_range,
        )
    x0, x1 = x_range
    span = float(x1 - x0)
    best: Optional[tuple[float, list[float], np.ndarray, float]] = None
    for combination in itertools.combinations(clusters, 3):
        boundaries = [float(x0), *(item[0] for item in combination), float(x1)]
        widths = np.diff(boundaries) / max(span, 1.0)
        if float(np.min(widths)) < 0.12 or float(np.max(widths)) > 0.38:
            continue
        line_strength = sum(
            min(1.0, item[1] / max(1.0, face_bottom - top_y))
            for item in combination
        ) / 3.0
        geometry_score = max(0.0, 1.0 - float(np.std(widths)) / 0.12)
        score = 0.65 * line_strength + 0.35 * geometry_score
        if best is None or score > best[0]:
            best = (score, boundaries, widths, line_strength)
    if best is None:
        return BoxCenterFrameResult(
            False,
            "placement_separator_geometry_invalid",
            "placement",
            width,
            height,
            target_label=target_letter,
            box_x_range=x_range,
        )
    score, boundaries, _widths, _line_strength = best
    center_y = 0.5 * float(top_y + face_bottom)
    candidates: list[PlacementCandidate] = []
    for letter, left, right in zip(LETTERS, boundaries, boundaries[1:]):
        box_left = int(round(left))
        box_right = int(round(right))
        candidates.append(
            PlacementCandidate(
                center=(0.5 * (left + right), center_y),
                label_bbox=None,
                box_bbox=(box_left, top_y, max(1, box_right - box_left), face_bottom - top_y),
                recognized_letter=letter,
                confidence=min(1.0, 0.35 * span_confidence + 0.65 * score),
            )
        )
    result = map_placement_candidates(
        candidates,
        frame_width=width,
        frame_height=height,
        target_letter=target_letter,
        placement_roi=cfg["placement_roi"],
        min_center_gap_fraction=float(
            cfg["placement_min_center_gap_fraction"]
        ),
    )
    return BoxCenterFrameResult(
        result.ok,
        result.reason,
        result.mode,
        result.frame_width,
        result.frame_height,
        centers=result.centers,
        target_label=result.target_label,
        target_center=result.target_center,
        spacing_px=result.spacing_px,
        confidence=result.confidence,
        candidates=result.candidates,
        box_x_range=x_range,
        detector_reason="cardboard_vertical_separators",
    )


def _detect_tracked_placement_box_centers(
    frame: np.ndarray,
    config: Mapping[str, object],
    *,
    target_letter: Optional[str],
    reference_centers: Mapping[str, tuple[float, float]],
    expected_target_x: float,
) -> BoxCenterFrameResult:
    """Keep A-D identity when an outer box or cardboard edge is cropped."""
    height, width = frame.shape[:2]
    cfg = _config_with_defaults(config)
    if target_letter not in LETTERS:
        return BoxCenterFrameResult(
            False,
            "invalid_target_letter",
            "placement",
            width,
            height,
            target_label=target_letter,
        )
    try:
        reference_x = np.asarray(
            [float(reference_centers[letter][0]) for letter in LETTERS],
            dtype=np.float64,
        )
        reference_y = np.asarray(
            [float(reference_centers[letter][1]) for letter in LETTERS],
            dtype=np.float64,
        )
        expected_target_x = float(expected_target_x)
    except (KeyError, TypeError, ValueError, IndexError):
        return BoxCenterFrameResult(
            False,
            "placement_tracking_reference_invalid",
            "placement",
            width,
            height,
            target_label=target_letter,
        )
    if (
        not np.isfinite(reference_x).all()
        or not np.isfinite(reference_y).all()
        or not math.isfinite(expected_target_x)
        or np.any(np.diff(reference_x) <= 0.0)
    ):
        return BoxCenterFrameResult(
            False,
            "placement_tracking_reference_invalid",
            "placement",
            width,
            height,
            target_label=target_letter,
        )

    (
        x_range,
        top_y,
        face_bottom,
        span_confidence,
        clusters,
        geometry_reason,
    ) = _placement_separator_geometry(
        frame,
        min_span_fraction=float(cfg["placement_min_span_fraction"]),
    )
    if geometry_reason or x_range is None or top_y is None or face_bottom is None:
        return BoxCenterFrameResult(
            False,
            geometry_reason,
            "placement",
            width,
            height,
            target_label=target_letter,
            box_x_range=x_range,
        )

    face_height = max(1.0, float(face_bottom - top_y))
    usable_clusters = [
        cluster for cluster in clusters if cluster[1] >= 0.70 * face_height
    ]
    minimum_separators = max(
        2,
        min(3, int(cfg["placement_tracking_min_separators"])),
    )
    if len(usable_clusters) < minimum_separators:
        return BoxCenterFrameResult(
            False,
            "placement_tracking_separators_insufficient",
            "placement",
            width,
            height,
            target_label=target_letter,
            box_x_range=x_range,
        )

    reference_boundaries = 0.5 * (reference_x[:-1] + reference_x[1:])
    target_index = LETTERS.index(target_letter)
    reference_target_x = float(reference_x[target_index])
    expected_shift = expected_target_x - reference_target_x
    if abs(expected_shift) < 1.0:
        return BoxCenterFrameResult(
            False,
            "placement_tracking_expected_motion_invalid",
            "placement",
            width,
            height,
            target_label=target_letter,
            box_x_range=x_range,
        )

    max_scale_change = float(cfg["placement_tracking_max_scale_change_fraction"])
    max_residual = max(
        2.0,
        width * float(cfg["placement_tracking_max_residual_fraction"]),
    )
    min_gain = float(cfg["placement_tracking_min_motion_gain"])
    max_gain = float(cfg["placement_tracking_max_motion_gain"])
    best: Optional[tuple[float, float, float, tuple[tuple[float, float], ...], float]] = None
    # If all three separators are visible, all three must agree with the
    # reference. Dropping a real separator could shift identity by one box.
    match_count = min(3, len(usable_clusters))
    for count in (match_count,):
        for observed in itertools.combinations(usable_clusters, count):
            observed_x = np.asarray([item[0] for item in observed], dtype=np.float64)
            for reference_indices in itertools.combinations(range(3), count):
                source_x = reference_boundaries[list(reference_indices)]
                scale, offset = np.polyfit(source_x, observed_x, 1)
                if not math.isfinite(float(scale)) or not math.isfinite(float(offset)):
                    continue
                if abs(float(scale) - 1.0) > max_scale_change:
                    continue
                residual = float(np.max(np.abs(scale * source_x + offset - observed_x)))
                if residual > max_residual:
                    continue
                tracked_target_x = float(scale * reference_target_x + offset)
                motion_gain = (tracked_target_x - reference_target_x) / expected_shift
                if not min_gain <= motion_gain <= max_gain:
                    continue
                if not 0.0 <= tracked_target_x < width:
                    continue
                line_quality = float(
                    np.mean([min(1.0, item[1] / face_height) for item in observed])
                )
                score = (
                    abs(motion_gain - 1.0)
                    + 2.0 * abs(float(scale) - 1.0)
                    + residual / max_residual
                    + 0.15 * (3 - count)
                    + 0.10 * (1.0 - line_quality)
                )
                candidate = (
                    score,
                    float(scale),
                    float(offset),
                    tuple(observed),
                    line_quality,
                )
                if best is None or candidate[0] < best[0]:
                    best = candidate
    if best is None:
        return BoxCenterFrameResult(
            False,
            "placement_tracking_geometry_invalid",
            "placement",
            width,
            height,
            target_label=target_letter,
            box_x_range=x_range,
        )

    score, scale, offset, _observed, line_quality = best
    tracked_x = scale * reference_x + offset
    center_y = 0.5 * float(top_y + face_bottom)
    centers = {
        letter: (float(x), center_y)
        for letter, x in zip(LETTERS, tracked_x)
    }
    spacing = tuple(float(value) for value in np.diff(tracked_x))
    boundaries = [
        float(tracked_x[0] - 0.5 * spacing[0]),
        *(0.5 * (tracked_x[:-1] + tracked_x[1:])),
        float(tracked_x[-1] + 0.5 * spacing[-1]),
    ]
    confidence = min(
        0.90,
        max(
            0.0,
            0.35 * span_confidence
            + 0.35 * line_quality
            + 0.30 * max(0.0, 1.0 - score / 2.0),
        ),
    )
    candidates: list[PlacementCandidate] = []
    for letter, left, right in zip(LETTERS, boundaries, boundaries[1:]):
        clipped_left = max(0, min(width - 1, int(round(left))))
        clipped_right = max(clipped_left + 1, min(width, int(round(right))))
        candidates.append(
            PlacementCandidate(
                center=centers[letter],
                label_bbox=None,
                box_bbox=(
                    clipped_left,
                    top_y,
                    clipped_right - clipped_left,
                    face_bottom - top_y,
                ),
                recognized_letter=letter,
                confidence=confidence,
            )
        )
    return BoxCenterFrameResult(
        True,
        "",
        "placement",
        width,
        height,
        centers=centers,
        target_label=target_letter,
        target_center=centers[target_letter],
        spacing_px=spacing,
        confidence=confidence,
        candidates=tuple(candidates),
        box_x_range=x_range,
        detector_reason="cardboard_separator_tracking",
    )


def detect_placement_box_centers(
    frame: np.ndarray,
    config: Mapping[str, object],
    *,
    target_letter: Optional[str] = None,
    reference_centers: Optional[Mapping[str, tuple[float, float]]] = None,
    expected_target_x: Optional[float] = None,
) -> BoxCenterFrameResult:
    if frame is None or frame.ndim != 3:
        return BoxCenterFrameResult(False, "invalid_frame", "placement", 0, 0)
    unlabelled = _detect_unlabelled_placement_box_centers(
        frame,
        config,
        target_letter=target_letter,
    )
    if unlabelled.ok:
        return unlabelled
    labelled = _detect_labeled_placement_box_centers(
        frame,
        config,
        target_letter=target_letter,
    )
    if labelled.ok:
        return labelled
    recovered_letters = detect_placement_letter_candidates(frame, config)
    if len(recovered_letters.candidates) == len(LETTERS):
        centers = {
            candidate.recognized_letter: candidate.center
            for candidate in recovered_letters.candidates
        }
        spacing = tuple(
            centers[LETTERS[index + 1]][0] - centers[LETTERS[index]][0]
            for index in range(len(LETTERS) - 1)
        )
        candidates = tuple(
            PlacementCandidate(
                center=candidate.center,
                label_bbox=candidate.label_bbox,
                box_bbox=candidate.label_bbox,
                recognized_letter=candidate.recognized_letter,
                confidence=candidate.confidence,
            )
            for candidate in recovered_letters.candidates
        )
        return BoxCenterFrameResult(
            True,
            "",
            "placement",
            recovered_letters.frame_width,
            recovered_letters.frame_height,
            centers=centers,
            target_label=target_letter,
            target_center=(
                centers.get(target_letter)
                if target_letter in LETTERS
                else None
            ),
            spacing_px=spacing,
            confidence=min(item.confidence for item in candidates),
            candidates=candidates,
            box_x_range=(
                int(round(candidates[0].center[0] - spacing[0] / 2.0)),
                int(round(candidates[-1].center[0] + spacing[-1] / 2.0)),
            ),
            detector_reason="ocr_anchored_label_row",
        )
    cfg = _config_with_defaults(config)
    if (
        bool(cfg["placement_tracking_enabled"])
        and reference_centers is not None
        and expected_target_x is not None
    ):
        return _detect_tracked_placement_box_centers(
            frame,
            cfg,
            target_letter=target_letter,
            reference_centers=reference_centers,
            expected_target_x=expected_target_x,
        )
    return unlabelled


def detect_pickup_box_center(
    frame: np.ndarray,
    config: Mapping[str, object],
    *,
    target_letter: Optional[str] = None,
    detector: Callable[[np.ndarray], BoxParallelResult] = detect_box_parallel,
) -> BoxCenterFrameResult:
    del config, target_letter
    if frame is None or frame.ndim != 3:
        return BoxCenterFrameResult(False, "invalid_frame", "pickup", 0, 0)
    height, width = frame.shape[:2]
    result = detector(frame)
    fallback_used = False
    top_y: Optional[int] = None
    if result.box_x_range is None:
        fallback_range, top_y, fallback_confidence, fallback_reason = _cardboard_top_span(
            frame,
            min_span_fraction=0.12,
            max_span_fraction=0.55,
        )
        if fallback_range is None:
            return BoxCenterFrameResult(
                False,
                "box_x_range_unavailable",
                "pickup",
                width,
                height,
                detector_reason=f"{result.reason};{fallback_reason}",
            )
        x0, x1 = fallback_range
        fallback_used = True
    else:
        x0, x1 = (int(value) for value in result.box_x_range)
        fallback_confidence = 0.0
    if x0 < 0 or x1 <= x0 or x1 >= width:
        return BoxCenterFrameResult(
            False,
            "invalid_box_x_range",
            "pickup",
            width,
            height,
            box_x_range=(x0, x1),
            detector_reason=result.reason,
        )
    center_y = height / 2.0
    if result.top_line is not None:
        center_y = float(result.top_line[1] + result.top_line[3]) / 2.0
    elif top_y is not None:
        center_y = min(height - 1.0, top_y + 0.14 * height)
    center = ((x0 + x1) / 2.0, center_y)
    span_confidence = min(1.0, (x1 - x0) / max(1.0, width * 0.40))
    confidence = (
        min(1.0, 0.5 * fallback_confidence + 0.5 * span_confidence)
        if fallback_used
        else float(result.confidence) if result.ok else 0.6 * span_confidence
    )
    return BoxCenterFrameResult(
        True,
        "",
        "pickup",
        width,
        height,
        centers={"pickup": center},
        target_label="pickup",
        target_center=center,
        confidence=confidence,
        box_x_range=(x0, x1),
        detector_reason=(
            f"{result.reason};narrow_cardboard_center_fallback"
            if fallback_used
            else result.reason
        ),
    )


def summarize_center_frames(
    results: list[BoxCenterFrameResult],
    *,
    mode: str,
    requested_frames: int,
    min_valid_frames: int,
    max_center_range_fraction: float,
    target_letter: Optional[str] = None,
) -> BoxCenterMeasurement:
    valid = [result for result in results if result.ok and result.mode == mode]
    failure_reasons = tuple(result.reason for result in results if not result.ok)
    if len(valid) < min_valid_frames:
        width = next((result.frame_width for result in results if result.frame_width), 0)
        height = next((result.frame_height for result in results if result.frame_height), 0)
        return BoxCenterMeasurement(
            False,
            "insufficient_valid_frames",
            mode,
            {},
            target_letter if mode == "placement" else "pickup",
            None,
            width,
            height,
            len(valid),
            requested_frames,
            0.0,
            (),
            {},
            None,
            None,
            failure_reasons,
        )
    widths = [result.frame_width for result in valid]
    heights = [result.frame_height for result in valid]
    width = int(round(float(np.median(widths))))
    height = int(round(float(np.median(heights))))
    keys = LETTERS if mode == "placement" else ("pickup",)
    centers: dict[str, tuple[float, float]] = {}
    unstable: list[str] = []
    for key in keys:
        samples = [result.centers[key] for result in valid if key in result.centers]
        if len(samples) < min_valid_frames:
            unstable.append(key)
            continue
        x_values = [sample[0] for sample in samples]
        y_values = [sample[1] for sample in samples]
        if max(x_values) - min(x_values) > width * max_center_range_fraction:
            unstable.append(key)
        centers[key] = (
            float(np.median(x_values)),
            float(np.median(y_values)),
        )
    if unstable:
        return BoxCenterMeasurement(
            False,
            "center_measurement_unstable",
            mode,
            centers,
            target_letter if mode == "placement" else "pickup",
            None,
            width,
            height,
            len(valid),
            requested_frames,
            float(np.median([result.confidence for result in valid])),
            (),
            {key: center[0] - width / 2.0 for key, center in centers.items()},
            None,
            None,
            tuple(unstable),
        )
    spacing = (
        tuple(centers[LETTERS[index + 1]][0] - centers[LETTERS[index]][0] for index in range(3))
        if mode == "placement"
        else ()
    )
    resolved_target = target_letter if mode == "placement" else "pickup"
    target_center = centers.get(resolved_target) if resolved_target else None
    center_errors = {key: center[0] - width / 2.0 for key, center in centers.items()}
    target_error = center_errors.get(resolved_target) if resolved_target else None
    return BoxCenterMeasurement(
        True,
        "stable",
        mode,
        centers,
        resolved_target,
        target_center,
        width,
        height,
        len(valid),
        requested_frames,
        float(np.median([result.confidence for result in valid])),
        spacing,
        center_errors,
        target_error,
        None if target_error is None or width <= 0 else target_error / width,
        failure_reasons,
    )


def annotate_box_centers(
    frame: np.ndarray,
    result: BoxCenterFrameResult,
) -> np.ndarray:
    output = frame.copy()
    height, width = output.shape[:2]
    cv2.line(output, (width // 2, 0), (width // 2, height - 1), (255, 0, 255), 2)
    cv2.putText(
        output,
        "IMAGE CENTER",
        (min(width - 190, width // 2 + 8), height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 0, 255),
        2,
        cv2.LINE_AA,
    )
    for candidate in result.candidates:
        bx, by, bw, bh = candidate.box_bbox
        cv2.rectangle(output, (bx, by), (bx + bw, by + bh), (0, 200, 255), 2)
        if candidate.label_bbox is not None:
            lx, ly, lw, lh = candidate.label_bbox
            cv2.rectangle(output, (lx, ly), (lx + lw, ly + lh), (255, 255, 255), 2)
    for label, center in result.centers.items():
        x, y = (int(round(value)) for value in center)
        colour = (0, 0, 255) if label == result.target_label else (0, 255, 0)
        cv2.drawMarker(output, (x, y), colour, cv2.MARKER_CROSS, 24, 3)
        center_label = "BOX CENTER" if label == "pickup" else f"{label} CENTER"
        cv2.putText(
            output,
            center_label,
            (min(width - 170, x + 8), max(24, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            colour,
            2,
            cv2.LINE_AA,
        )
        cv2.line(output, (width // 2, y), (x, y), colour, 2)
        cv2.putText(
            output,
            f"dx={x - width // 2:+d}px",
            (max(4, min(width - 120, x - 44)), max(24, y + 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            colour,
            2,
            cv2.LINE_AA,
        )
    if result.box_x_range is not None:
        x0, x1 = result.box_x_range
        if result.candidates:
            range_y0 = min(candidate.box_bbox[1] for candidate in result.candidates)
            range_y1 = max(
                candidate.box_bbox[1] + candidate.box_bbox[3]
                for candidate in result.candidates
            )
        elif result.target_center is not None:
            range_y0 = max(0, int(round(result.target_center[1] - 0.15 * height)))
            range_y1 = min(height - 1, int(round(result.target_center[1] + 0.15 * height)))
        else:
            range_y0, range_y1 = 0, height - 1
        cv2.line(output, (x0, range_y0), (x0, range_y1), (0, 255, 255), 2)
        cv2.line(output, (x1, range_y0), (x1, range_y1), (0, 255, 255), 2)
    text = (
        f"{result.mode}: ok confidence={result.confidence:.2f}"
        if result.ok
        else f"{result.mode}: {result.reason}"
    )
    cv2.putText(
        output,
        text,
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 180, 0) if result.ok else (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return output


class BoxCenterMeasurer:
    def __init__(
        self,
        *,
        camera,
        undistorter,
        config: Mapping[str, object],
        placement_detector: Callable[..., BoxCenterFrameResult] = detect_placement_box_centers,
        pickup_detector: Callable[..., BoxCenterFrameResult] = detect_pickup_box_center,
    ) -> None:
        self.camera = camera
        self.undistorter = undistorter
        self.config = _config_with_defaults(config)
        self.placement_detector = placement_detector
        self.pickup_detector = pickup_detector

    def measure(
        self,
        mode: str,
        target_letter: Optional[str] = None,
        *,
        placement_reference_centers: Optional[
            Mapping[str, tuple[float, float]]
        ] = None,
        placement_expected_target_x: Optional[float] = None,
        frame_callback: Optional[
            Callable[[int, Optional[np.ndarray], Optional[np.ndarray], BoxCenterFrameResult], None]
        ] = None,
    ) -> BoxCenterMeasurement:
        requested = max(1, int(self.config["frames_per_measurement"]))
        results: list[BoxCenterFrameResult] = []
        for index in range(1, requested + 1):
            raw = self.camera.read()
            if raw is None:
                result = BoxCenterFrameResult(False, "camera_read_failed", mode, 0, 0)
                undistorted = None
            else:
                try:
                    undistorted = self.undistorter.apply(raw)
                    if mode == "placement":
                        detector_kwargs: dict[str, object] = {
                            "target_letter": target_letter,
                        }
                        if (
                            placement_reference_centers is not None
                            and placement_expected_target_x is not None
                        ):
                            detector_kwargs.update(
                                {
                                    "reference_centers": placement_reference_centers,
                                    "expected_target_x": placement_expected_target_x,
                                }
                            )
                        result = self.placement_detector(
                            undistorted,
                            self.config,
                            **detector_kwargs,
                        )
                    elif mode == "pickup":
                        result = self.pickup_detector(
                            undistorted,
                            self.config,
                            target_letter=None,
                        )
                    else:
                        result = BoxCenterFrameResult(False, "invalid_mode", mode, 0, 0)
                except Exception as exc:
                    result = BoxCenterFrameResult(
                        False,
                        f"detector_exception:{type(exc).__name__}",
                        mode,
                        int(raw.shape[1]),
                        int(raw.shape[0]),
                    )
                    undistorted = raw
            results.append(result)
            if frame_callback is not None:
                frame_callback(index, raw, undistorted, result)
        return summarize_center_frames(
            results,
            mode=mode,
            requested_frames=requested,
            min_valid_frames=max(1, int(self.config["min_valid_frames"])),
            max_center_range_fraction=max(0.0, float(self.config["max_center_range_fraction"])),
            target_letter=target_letter,
        )


def _metres_per_pixel_for_measurement(
    measurement: BoxCenterMeasurement,
    config: Mapping[str, object],
) -> float:
    if not measurement.ok or measurement.target_error_px is None:
        raise ValueError("stable target measurement is required")
    cfg = _config_with_defaults(config)
    if measurement.mode == "placement":
        positive_spacings = [value for value in measurement.spacing_px if value > 0.0]
        if len(positive_spacings) != 3:
            raise ValueError("four-box pixel spacing is unavailable")
        metres_per_pixel = float(cfg["adjacent_box_spacing_m"]) / float(np.median(positive_spacings))
    elif measurement.mode == "pickup":
        metres_per_pixel = float(cfg["pickup_m_per_pixel"])
    else:
        raise ValueError(f"unknown box center mode: {measurement.mode!r}")
    if not math.isfinite(metres_per_pixel) or metres_per_pixel <= 0.0:
        raise ValueError("invalid metres-per-pixel scale")
    return metres_per_pixel


def strafe_correction_for_measurement(
    measurement: BoxCenterMeasurement,
    config: Mapping[str, object],
) -> float:
    cfg = _config_with_defaults(config)
    sign = int(cfg["positive_error_strafe_sign"])
    if sign not in {-1, 1}:
        raise ValueError("positive_error_strafe_sign must be -1 or 1")
    metres_per_pixel = _metres_per_pixel_for_measurement(measurement, cfg)
    return sign * measurement.target_error_px * metres_per_pixel


def strafe_distance_for_box_center(
    motion,
    distance_m: float,
    config: Mapping[str, object],
) -> None:
    cfg = _config_with_defaults(config)
    pose_held_strafe = getattr(type(motion), "strafe_distance_pose_hold", None)
    if bool(cfg["strafe_pose_hold_enabled"]) and callable(pose_held_strafe):
        bound_pose_held_strafe = getattr(motion, "strafe_distance_pose_hold")
        bound_pose_held_strafe(
            distance_m,
            speed_mps=float(cfg["strafe_speed_mps"]),
            forward_hold_kp_s=float(cfg["forward_hold_kp_s"]),
            max_vx_correction_mps=float(cfg["max_vx_correction_mps"]),
            forward_deadband_m=float(cfg["forward_deadband_m"]),
            max_forward_drift_m=float(cfg["max_forward_drift_m"]),
            yaw_hold_kp_s=float(cfg["yaw_hold_kp_s"]),
            max_wz_correction_rad_s=float(cfg["max_wz_correction_rad_s"]),
            yaw_deadband_deg=float(cfg["yaw_deadband_deg"]),
            max_yaw_drift_deg=float(cfg["max_yaw_drift_deg"]),
        )
        return
    motion.strafe_distance(
        distance_m,
        speed_mps=float(cfg["strafe_speed_mps"]),
    )


class BoxCenterAligner:
    def __init__(
        self,
        *,
        camera,
        undistorter,
        motion,
        config: Mapping[str, object],
        measurement_provider: Optional[
            Callable[[str, Optional[str]], BoxCenterMeasurement]
        ] = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.camera = camera
        self.undistorter = undistorter
        self.motion = motion
        self.config = _config_with_defaults(config)
        self.measurement_provider = measurement_provider
        self.sleep = sleep
        self.measurer = BoxCenterMeasurer(
            camera=camera,
            undistorter=undistorter,
            config=self.config,
        )

    def _safe_stop(self) -> None:
        try:
            self.motion.stop()
        except Exception:
            pass

    def _release_camera(self) -> None:
        release = getattr(self.camera, "release", None)
        if release is not None:
            try:
                release()
            except Exception:
                pass

    def _create_run_dir(self) -> Optional[Path]:
        try:
            root = Path(str(self.config["alignment_run_log_dir"]))
            run_dir = root / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
        except Exception:
            return None

    @staticmethod
    def _write_json(run_dir: Optional[Path], name: str, payload: object) -> None:
        if run_dir is None:
            return
        try:
            (run_dir / name).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _measure(
        self,
        mode: str,
        target_letter: Optional[str],
        measurement_index: int,
        run_dir: Optional[Path],
        *,
        placement_reference_centers: Optional[
            Mapping[str, tuple[float, float]]
        ] = None,
        placement_expected_target_x: Optional[float] = None,
    ) -> BoxCenterMeasurement:
        if self.measurement_provider is not None:
            measurement = self.measurement_provider(mode, target_letter)
        else:
            def save_frame(
                frame_index: int,
                raw: Optional[np.ndarray],
                undistorted: Optional[np.ndarray],
                result: BoxCenterFrameResult,
            ) -> None:
                if run_dir is None:
                    return
                try:
                    if raw is not None:
                        cv2.imwrite(
                            str(run_dir / f"measure_{measurement_index:02d}_{frame_index:03d}_raw.jpg"),
                            raw,
                        )
                    if undistorted is not None:
                        cv2.imwrite(
                            str(run_dir / f"measure_{measurement_index:02d}_{frame_index:03d}_annotated.jpg"),
                            annotate_box_centers(undistorted, result),
                        )
                except Exception:
                    pass

            measurement = self.measurer.measure(
                mode,
                target_letter,
                placement_reference_centers=placement_reference_centers,
                placement_expected_target_x=placement_expected_target_x,
                frame_callback=save_frame,
            )
        self._write_json(
            run_dir,
            f"measurement_{measurement_index:02d}.json",
            asdict(measurement),
        )
        return measurement

    def run(
        self,
        mode: str,
        target_letter: Optional[str] = None,
        *,
        tolerance_fraction: Optional[float] = None,
    ) -> BoxCenterAlignmentResult:
        run_dir = self._create_run_dir()
        correction_count = 0
        motion_command_count = 0
        measurement_count = 0
        initial_error: Optional[float] = None
        final_error: Optional[float] = None
        signed_visual_strafe = 0.0
        cumulative_abs_strafe = 0.0
        placement_reference_centers: Optional[dict[str, tuple[float, float]]] = None
        placement_expected_target_x: Optional[float] = None
        active_tolerance_fraction = (
            float(self.config["tolerance_fraction"])
            if tolerance_fraction is None
            else float(tolerance_fraction)
        )

        def finish_failure(reason: str) -> BoxCenterAlignmentResult:
            nonlocal motion_command_count
            rollback_attempted = abs(signed_visual_strafe) > 1e-9
            rollback_ok = True
            net_strafe = signed_visual_strafe
            if rollback_attempted:
                try:
                    strafe_distance_for_box_center(
                        self.motion,
                        -signed_visual_strafe,
                        self.config,
                    )
                    motion_command_count += 1
                    net_strafe = 0.0
                except Exception:
                    rollback_ok = False
                finally:
                    self._safe_stop()
            result = BoxCenterAlignmentResult(
                False,
                reason if rollback_ok else f"{reason};rollback_failed",
                mode,
                target_letter if mode == "placement" else "pickup",
                correction_count,
                motion_command_count,
                measurement_count,
                initial_error,
                final_error,
                signed_visual_strafe,
                net_strafe,
                rollback_attempted,
                rollback_ok,
                None if run_dir is None else str(run_dir),
            )
            self._write_json(run_dir, "result.json", asdict(result))
            return result

        def finish_success(reason: str) -> BoxCenterAlignmentResult:
            result = BoxCenterAlignmentResult(
                True,
                reason,
                mode,
                target_letter if mode == "placement" else "pickup",
                correction_count,
                motion_command_count,
                measurement_count,
                initial_error,
                final_error,
                signed_visual_strafe,
                signed_visual_strafe,
                False,
                True,
                None if run_dir is None else str(run_dir),
            )
            self._write_json(run_dir, "result.json", asdict(result))
            return result

        try:
            if not bool(self.config["enabled"]):
                return finish_success("disabled")
            if (
                not math.isfinite(active_tolerance_fraction)
                or active_tolerance_fraction < 0.0
                or active_tolerance_fraction > 1.0
            ):
                return finish_failure("invalid_tolerance_fraction")
            if mode not in {"placement", "pickup"}:
                return finish_failure("invalid_mode")
            if mode == "placement" and target_letter not in LETTERS:
                return finish_failure("invalid_target_letter")
            max_corrections = max(0, int(self.config["max_corrections"]))
            for measurement_index in range(1, max_corrections + 2):
                measurement_count += 1
                measurement = self._measure(
                    mode,
                    target_letter,
                    measurement_index,
                    run_dir,
                    placement_reference_centers=placement_reference_centers,
                    placement_expected_target_x=placement_expected_target_x,
                )
                final_error = measurement.target_error_px
                if initial_error is None:
                    initial_error = final_error
                if not measurement.ok:
                    return finish_failure(measurement.reason)
                if final_error is None:
                    return finish_failure("target_center_unavailable")
                tolerance_px = measurement.frame_width * active_tolerance_fraction
                if abs(final_error) <= tolerance_px:
                    return finish_success("aligned")
                if correction_count >= max_corrections:
                    return finish_failure("max_corrections")
                try:
                    correction = strafe_correction_for_measurement(measurement, self.config)
                except ValueError as exc:
                    return finish_failure(str(exc))
                remaining = float(self.config["max_total_strafe_m"]) - cumulative_abs_strafe
                if remaining <= 1e-9:
                    return finish_failure("max_total_strafe")
                limit = min(float(self.config["max_single_strafe_m"]), remaining)
                correction = math.copysign(min(abs(correction), limit), correction)
                if abs(correction) <= 1e-9:
                    return finish_failure("zero_correction")
                if mode == "placement" and measurement.target_center is not None:
                    metres_per_pixel = _metres_per_pixel_for_measurement(
                        measurement,
                        self.config,
                    )
                    direction_sign = int(self.config["positive_error_strafe_sign"])
                    expected_pixel_shift = -correction / (
                        direction_sign * metres_per_pixel
                    )
                    placement_reference_centers = dict(measurement.centers)
                    placement_expected_target_x = (
                        float(measurement.target_center[0]) + expected_pixel_shift
                    )
                correction_count += 1
                self._write_json(
                    run_dir,
                    f"correction_{correction_count:02d}.json",
                    {
                        "target_error_px": final_error,
                        "strafe_m": correction,
                        "cumulative_abs_strafe_m": cumulative_abs_strafe + abs(correction),
                    },
                )
                strafe_distance_for_box_center(
                    self.motion,
                    correction,
                    self.config,
                )
                motion_command_count += 1
                signed_visual_strafe += correction
                cumulative_abs_strafe += abs(correction)
                self._safe_stop()
                self._release_camera()
                self.sleep(max(0.0, float(self.config["settle_seconds"])))
            return finish_failure("internal_loop_error")
        finally:
            self._safe_stop()
            self._release_camera()

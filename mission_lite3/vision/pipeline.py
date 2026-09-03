from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from statistics import median
from typing import Any, Dict, List, Optional

from ..inspection_tags import InspectionTagDetector, InspectionTagObservation
from .common import Detection, StableVote, largest_detection
from .color import build_competition_color_detectors
from .dashboard import DashboardRecognizer
from .letters import LetterRecognizer


@dataclass(frozen=True)
class InspectionRecord:
    letter: str
    level: str
    state: str
    confidence: float
    frame_id: int
    timestamp: str = ""
    source_camera: str = ""
    stability_votes: Dict[str, int] = field(default_factory=dict)
    evidence_image: Optional[str] = None


def runtime_result_to_record_fields(
    result: dict[str, Any],
    min_letter_confidence: float = 0.70,
) -> Optional[tuple[str, str, str, float]]:
    if not result or not result.get("ok", False):
        return None

    meter = result.get("meter_detection") or {}
    runtime_state = meter.get("state")
    description = str(meter.get("description") or "")
    if runtime_state == "normal":
        level = "正常"
        state = "正常"
    elif runtime_state == "abnormal" and description in {"偏低", "偏高"}:
        level = description
        state = "异常"
    else:
        return None

    letter_detection = result.get("letter_detection") or {}
    letter = str(letter_detection.get("label") or "").upper()
    confidence = float(letter_detection.get("confidence", 0.0) or 0.0)
    if confidence < min_letter_confidence:
        return None

    if letter not in {"A", "B", "C", "D"}:
        return None
    return letter, level, state, confidence


class VisionPipeline:
    def __init__(self, config: dict):
        self.config = config
        vision = config["vision"]
        self.inspection_backend = str(vision.get("inspection_backend", "runtime_meter_anchor"))
        self.runtime_min_letter_confidence = float(
            vision.get("runtime_min_letter_confidence", vision.get("letter_min_confidence", 0.70))
        )
        self.runtime_fast_accept_confidence = float(
            vision.get("runtime_fast_accept_confidence", 0.84)
        )
        self.runtime_fast_accept_margin = float(
            vision.get("runtime_fast_accept_margin", 0.20)
        )
        self.runtime_best_candidate_confidence = float(
            vision.get("runtime_best_candidate_confidence", 0.82)
        )
        self.runtime_fast_min_pointer_hit_ratio = float(
            vision.get("runtime_fast_min_pointer_hit_ratio", 0.60)
        )
        self.runtime_fast_min_pointer_run_ratio = float(
            vision.get("runtime_fast_min_pointer_run_ratio", 0.45)
        )
        self.runtime_frame_pipeline = None
        self.letter_recognizer = LetterRecognizer(float(vision["letter_min_confidence"]))
        self.dashboard_recognizer = DashboardRecognizer(int(vision["dashboard_min_radius"]))
        self.color_detectors = build_competition_color_detectors(config)
        self.runtime_window_size = int(vision["stable_window"])
        self.runtime_required_votes = int(vision["stable_votes"])
        self.letter_vote = StableVote[str](self.runtime_window_size, self.runtime_required_votes)
        self.dashboard_vote = StableVote[str](self.runtime_window_size, self.runtime_required_votes)
        self.runtime_vote = StableVote[tuple](
            self.runtime_window_size,
            self.runtime_required_votes,
        )
        self._runtime_angle_history = deque(maxlen=self.runtime_window_size)
        self._runtime_fallback_history = deque(maxlen=self.runtime_window_size)
        self._runtime_results_by_frame: Dict[int, dict[str, Any]] = {}
        self.frame_id = 0
        self._best_inspection_record: Optional[InspectionRecord] = None
        self._best_inspection_frame = None
        self.inspection_tag_detector = InspectionTagDetector(config)
        self._latest_inspection_tags: tuple[InspectionTagObservation, ...] = ()
        self._tag_detection_error_logged = False
        self._best_inspection_score = float("-inf")
        self._load_runtime_backend()

    def inspect_frame(self, frame, source_camera: str = "") -> Optional[InspectionRecord]:
        if self.inspection_backend == "runtime_meter_anchor" and self.runtime_frame_pipeline is not None:
            return self._inspect_runtime_frame(frame, source_camera=source_camera)
        return self._inspect_legacy_frame(frame, source_camera=source_camera)

    def reset_inspection_votes(self) -> None:
        self.letter_vote.clear()
        self.dashboard_vote.clear()
        self.runtime_vote.clear()
        self._runtime_angle_history.clear()
        self._runtime_fallback_history.clear()
        self._runtime_results_by_frame.clear()
        self._best_inspection_record = None
        self._best_inspection_frame = None
        self._latest_inspection_tags = ()
        self._best_inspection_score = float("-inf")

    def best_inspection_candidate(self):
        """Return the best valid real-camera result and its matching frame for this stop."""
        fallback = self._runtime_angle_median_fallback()
        if fallback is not None:
            return fallback
        if self._best_inspection_record is None:
            return None
        return self._best_inspection_record, self._best_inspection_frame

    def inspection_diagnostics(self, frame_id: int) -> Optional[dict[str, Any]]:
        result = self._runtime_results_by_frame.get(int(frame_id))
        if result is None:
            return None
        meter = result.get("meter_detection") or {}
        center = meter.get("center_px")
        tip = meter.get("pointer_tip_px")
        angle_deg = None
        if (
            isinstance(center, (list, tuple))
            and isinstance(tip, (list, tuple))
            and len(center) == 2
            and len(tip) == 2
        ):
            import math

            angle_deg = math.degrees(
                math.atan2(float(tip[1]) - float(center[1]), float(tip[0]) - float(center[0]))
            )
        return {
            "frame_id": int(frame_id),
            "geometry_source": result.get("geometry_source"),
            "letter_detection": result.get("letter_detection"),
            "meter_detection": {
                "state": meter.get("state"),
                "description": meter.get("description"),
                "center_px": center,
                "radius_px": meter.get("radius_px"),
                "pointer_tip_px": tip,
                "pointer_angle_deg": angle_deg,
                "pointer_support": meter.get("pointer_support"),
                "pointer_method": meter.get("pointer_method"),
                "pointer_line": meter.get("pointer_line"),
                "status_evidence": meter.get("status_evidence"),
            },
        }

    @staticmethod
    def _runtime_relative_deg(result: dict[str, Any]) -> Optional[float]:
        meter = result.get("meter_detection") or {}
        evidence = meter.get("status_evidence") or {}
        value = evidence.get("pointer_relative_to_red_deg")
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if -180.0 <= value <= 180.0 else None

    @staticmethod
    def _level_from_relative_deg(relative_deg: float) -> Optional[tuple[str, str]]:
        absolute_deg = abs(relative_deg)
        if abs(absolute_deg - 40.0) < 8.0 or 180.0 - absolute_deg < 8.0:
            return None
        if absolute_deg < 40.0:
            return "\u504f\u9ad8", "\u5f02\u5e38"
        if relative_deg < 0.0:
            return "\u6b63\u5e38", "\u6b63\u5e38"
        return "\u504f\u4f4e", "\u5f02\u5e38"

    def _runtime_angle_median_fallback(self):
        if not self._runtime_fallback_history:
            return None
        letter_counts = Counter(
            entry[0][0] for entry in self._runtime_fallback_history
        )
        letter, _count = letter_counts.most_common(1)[0]
        entries = [
            entry
            for entry in self._runtime_fallback_history
            if entry[0][0] == letter
        ]
        angles = [
            angle
            for _fields, _result, _frame, _frame_id, angle in entries
            if angle is not None
        ]
        if angles:
            median_angle = float(median(angles))
            classified = self._level_from_relative_deg(median_angle)
        else:
            median_angle = None
            classified = None
        selected = max(entries, key=lambda entry: entry[0][3])
        if median_angle is not None:
            entries_with_angles = [entry for entry in entries if entry[4] is not None]
            selected = min(
                entries_with_angles,
                key=lambda entry: abs(float(entry[4]) - median_angle),
            )
        fields, _result, frame, frame_id, _angle = selected
        if classified is None:
            level, state = fields[1], fields[2]
        else:
            level, state = classified
        confidence = float(median([entry[0][3] for entry in entries]))
        record = InspectionRecord(
            letter,
            level,
            state,
            confidence,
            frame_id,
            datetime.now(timezone.utc).isoformat(),
            "front",
            {
                "runtime_angle_median": len(angles),
                "runtime_letter_samples": len(entries),
            },
        )
        return record, frame

    def detect_inspection_tags(self, frame) -> list[InspectionTagObservation]:
        try:
            observations = self.inspection_tag_detector.detect(frame)
        except Exception as exc:
            if not self._tag_detection_error_logged:
                print(f'[inspect-tag] detector error; continue without tags: {exc}')
                self._tag_detection_error_logged = True
            observations = []
        self._latest_inspection_tags = tuple(observations)
        return list(observations)

    def latest_inspection_tags(self) -> tuple[InspectionTagObservation, ...]:
        return self._latest_inspection_tags

    def _inspect_legacy_frame(self, frame, source_camera: str = "") -> Optional[InspectionRecord]:
        self.frame_id += 1
        letter = self.letter_recognizer.recognize(frame)
        dashboard = self.dashboard_recognizer.recognize(frame)
        stable_letter = self.letter_vote.add(letter.label if letter else None)
        stable_level = self.dashboard_vote.add(dashboard.label if dashboard else None)
        if stable_letter is None or stable_level is None:
            return None
        state = "正常" if stable_level == "正常" else "异常"
        confidence = min(letter.confidence if letter else 0.5, dashboard.confidence if dashboard else 0.5)
        stability_votes = {
            "letter": self.letter_vote.count_for(stable_letter),
            "dashboard": self.dashboard_vote.count_for(stable_level),
        }
        timestamp = datetime.now(timezone.utc).isoformat()
        return InspectionRecord(stable_letter, stable_level, state, confidence, self.frame_id, timestamp, source_camera, stability_votes)

    def _inspect_runtime_frame(self, frame, source_camera: str = "") -> Optional[InspectionRecord]:
        self.frame_id += 1
        result = self._analyze_runtime_frame(frame)
        self._runtime_results_by_frame[self.frame_id] = result
        while len(self._runtime_results_by_frame) > self.runtime_window_size * 2:
            self._runtime_results_by_frame.pop(min(self._runtime_results_by_frame))
        fields = runtime_result_to_record_fields(
            result,
            min_letter_confidence=self.runtime_min_letter_confidence,
        )
        reliable = fields is not None and self._is_reliable_runtime_result(result)
        signature = fields[:3] if reliable else None
        stable_signature = self.runtime_vote.add(signature)
        if fields is not None and reliable:
            angle = self._runtime_relative_deg(result)
            frame_copy = frame.copy() if hasattr(frame, "copy") else frame
            self._runtime_angle_history.append((signature, angle))
            self._runtime_fallback_history.append(
                (fields, result, frame_copy, self.frame_id, angle)
            )
            candidate = self._runtime_record(
                fields,
                source_camera,
                {"runtime_best": self.runtime_vote.count_for(signature)},
            )
            if self._is_best_runtime_candidate(result, fields):
                score = self._runtime_candidate_score(result, fields)
                if score > self._best_inspection_score:
                    self._best_inspection_score = score
                    self._best_inspection_record = candidate
                    self._best_inspection_frame = (
                        frame_copy
                    )
        else:
            self._runtime_angle_history.append((None, None))
        if stable_signature is None:
            return None
        letter, level, state = stable_signature
        if fields is None or fields[:3] != stable_signature:
            return None
        matching_angles = [
            angle
            for signature_value, angle in self._runtime_angle_history
            if signature_value == stable_signature and angle is not None
        ]
        median_angle = (
            float(median(matching_angles)) if matching_angles else None
        )
        classified = (
            None
            if median_angle is None
            else self._level_from_relative_deg(median_angle)
        )
        if classified is not None:
            level, state = classified
        consensus_fields = (letter, level, state, fields[3])
        stability_votes = {
            "runtime": self.runtime_vote.count_for(stable_signature),
            "runtime_angle_samples": len(matching_angles),
        }
        return self._runtime_record(
            consensus_fields,
            source_camera,
            stability_votes,
        )

    def _runtime_record(
        self,
        fields: tuple[str, str, str, float],
        source_camera: str,
        stability_votes: Dict[str, int],
    ) -> InspectionRecord:
        letter, level, state, confidence = fields
        timestamp = datetime.now(timezone.utc).isoformat()
        return InspectionRecord(
            letter,
            level,
            state,
            confidence,
            self.frame_id,
            timestamp,
            source_camera,
            stability_votes,
        )

    @staticmethod
    def _runtime_pointer_support(result: dict[str, Any]) -> tuple[float, float]:
        meter = result.get("meter_detection") or {}
        support = meter.get("pointer_support") or {}
        return (
            float(support.get("hit_ratio", 0.0) or 0.0),
            float(support.get("longest_run_ratio", 0.0) or 0.0),
        )

    def _is_fast_runtime_result(
        self,
        result: dict[str, Any],
        fields: tuple[str, str, str, float],
    ) -> bool:
        letter = result.get("letter_detection") or {}
        margin = float(letter.get("margin", 0.0) or 0.0)
        clear_letter = (
            fields[3] >= self.runtime_fast_accept_confidence
            or (
                fields[3] >= self.runtime_min_letter_confidence
                and margin >= self.runtime_fast_accept_margin
            )
        )
        return self._is_reliable_runtime_result(result) and clear_letter

    def _is_reliable_runtime_result(self, result: dict[str, Any]) -> bool:
        hit_ratio, run_ratio = self._runtime_pointer_support(result)
        meter = result.get("meter_detection") or {}
        status_evidence = meter.get("status_evidence") or {}
        return (
            result.get("geometry_source") == "letter_anchor"
            and hit_ratio >= self.runtime_fast_min_pointer_hit_ratio
            and run_ratio >= self.runtime_fast_min_pointer_run_ratio
            and status_evidence.get("status_agreement", True) is not False
            and status_evidence.get("status_supported", True) is not False
        )

    def _is_best_runtime_candidate(
        self,
        result: dict[str, Any],
        fields: tuple[str, str, str, float],
    ) -> bool:
        return (
            self._is_reliable_runtime_result(result)
            and fields[3] >= self.runtime_best_candidate_confidence
        )

    def _runtime_candidate_score(
        self,
        result: dict[str, Any],
        fields: tuple[str, str, str, float],
    ) -> float:
        hit_ratio, run_ratio = self._runtime_pointer_support(result)
        return fields[3] + hit_ratio * 0.35 + run_ratio * 0.20

    def _analyze_runtime_frame(self, frame) -> dict[str, Any]:
        import cv2 as cv
        from PIL import Image

        observations = self.detect_inspection_tags(frame)
        recognition_frame = frame
        letter_anchor_min_height_px = None
        if observations:
            letter_anchor_min_height_px = (
                self.inspection_tag_detector.recommended_letter_min_height_px(
                    observations
                )
            )
            if self.inspection_tag_detector.mask_for_recognition:
                recognition_frame = self.inspection_tag_detector.mask(
                    frame,
                    observations,
                )
        frame_rgb = cv.cvtColor(recognition_frame, cv.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        result = self.runtime_frame_pipeline.analyze_inspection_frame(
            image,
            letter_anchor_min_height_px=letter_anchor_min_height_px,
        )
        result['inspection_tags'] = [
            {
                'tag_id': item.tag_id,
                'center_x_px': item.center_x_px,
                'center_y_px': item.center_y_px,
                'edge_px': item.edge_px,
                'distance_m': item.distance_m,
            }
            for item in observations
        ]
        result['letter_anchor_min_height_px'] = letter_anchor_min_height_px
        return result

    def _load_runtime_backend(self) -> None:
        if self.inspection_backend != "runtime_meter_anchor":
            return
        try:
            from mission_lite3.inspection_runtime import frame_pipeline

            frame_pipeline.warm_up_recognizers()
            self.runtime_frame_pipeline = frame_pipeline
        except Exception as exc:
            if bool(self.config.get("vision", {}).get("allow_legacy_fallback", False)):
                print(f"[vision] runtime_meter_anchor unavailable, explicit legacy fallback enabled: {exc}")
                self.inspection_backend = "legacy"
                self.runtime_frame_pipeline = None
                return
            raise RuntimeError(f"runtime_meter_anchor backend is unavailable: {exc}") from exc

    def detect_cones(self, frame) -> List[Detection]:
        return self.color_detectors["cone"].detect(frame)

    def detect_red_bars(self, frame) -> List[Detection]:
        return self.color_detectors["red_bar"].detect(frame)

    def detect_green_bars(self, frame) -> List[Detection]:
        return self.color_detectors["green_bar"].detect(frame)

    def best_red_bar(self, frame) -> Optional[Detection]:
        return largest_detection(self.detect_red_bars(frame))

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from .vision import InspectionRecord


AREA_ORDER = ("A", "B", "C", "D")
SCHEMA_VERSION = 2
DEFAULT_ROUND_RESULT_PATH = Path("round_result.json")
DEFAULT_LATEST_STOP_RESULT_PATH = Path("latest_stop_result.json")


@dataclass(frozen=True)
class RoundGate:
    allowed: bool
    abnormal_areas: list[str]
    unknown_areas: list[str]
    block_reason: Optional[str]
    data: dict[str, Any]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _area_key(value: Any) -> Optional[str]:
    if value is None:
        return None
    area = str(value).strip().upper()
    return area if area in AREA_ORDER else None


def _area_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    areas: list[str] = []
    for item in value:
        area = _area_key(item)
        if area is not None and area not in areas:
            areas.append(area)
    return areas


def record_to_dict(record: InspectionRecord | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(record, InspectionRecord):
        return {
            "letter": record.letter,
            "area": record.letter,
            "level": record.level,
            "state": record.state,
            "confidence": record.confidence,
            "frame_id": record.frame_id,
            "timestamp": record.timestamp or utc_timestamp(),
            "source_camera": record.source_camera,
            "stability_votes": dict(record.stability_votes),
            "evidence_image": record.evidence_image,
        }
    letter = _area_key(record.get("letter") or record.get("area")) or ""
    return {
        "letter": letter,
        "area": letter,
        "level": str(record.get("level", "")),
        "state": str(record.get("state", "")),
        "confidence": float(record.get("confidence", 0.0) or 0.0),
        "frame_id": int(record.get("frame_id", -1) or -1),
        "timestamp": str(record.get("timestamp") or utc_timestamp()),
        "source_camera": str(record.get("source_camera") or ""),
        "stability_votes": dict(record.get("stability_votes") or {}),
        "evidence_image": record.get("evidence_image"),
    }


def record_from_dict(data: Mapping[str, Any]) -> Optional[InspectionRecord]:
    area = _area_key(data.get("letter") or data.get("area"))
    if area is None:
        return None
    return InspectionRecord(
        area,
        str(data.get("level", "")),
        str(data.get("state", "")),
        float(data.get("confidence", 0.0) or 0.0),
        int(data.get("frame_id", -1) or -1),
        str(data.get("timestamp") or ""),
        str(data.get("source_camera") or ""),
        dict(data.get("stability_votes") or {}),
        data.get("evidence_image"),
    )


def empty_round_result(
    source_camera: str = "",
    block_reason: str = "unknown_area",
    run_id: str = "",
) -> dict[str, Any]:
    unknown_areas = list(AREA_ORDER)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": str(run_id or ""),
        "timestamp": utc_timestamp(),
        "source_camera": source_camera,
        "records": {},
        "abnormal_areas": [],
        "unknown_areas": unknown_areas,
        "count_check": {
            "normal": 0,
            "abnormal": 0,
            "unknown": len(unknown_areas),
            "expected_normal": 2,
            "expected_abnormal": 2,
            "passed": False,
        },
        "ready": False,
        "block_reason": block_reason,
        "stability_votes": {},
        "evidence_image": {},
        "evidence_images": {},
    }


def build_round_result(
    records: Mapping[str, InspectionRecord | Mapping[str, Any]],
    source_camera: str = "",
    block_reason: Optional[str] = None,
    run_id: str = "",
) -> dict[str, Any]:
    normalized: dict[str, dict[str, Any]] = {}
    for key, value in records.items():
        record = record_to_dict(value)
        area = _area_key(record.get("letter") or key)
        if area is None:
            continue
        record["letter"] = area
        record["area"] = area
        normalized[area] = record

    normal_areas: list[str] = []
    abnormal_areas: list[str] = []
    unknown_areas: list[str] = []
    for area in AREA_ORDER:
        record = normalized.get(area)
        if record is None:
            unknown_areas.append(area)
            continue
        state = record.get("state")
        if state == "正常":
            normal_areas.append(area)
        elif state == "异常":
            abnormal_areas.append(area)
        else:
            unknown_areas.append(area)

    count_passed = len(normal_areas) == 2 and len(abnormal_areas) == 2 and not unknown_areas
    ready = count_passed and not unknown_areas and not block_reason
    if ready:
        resolved_block_reason = None
    elif block_reason:
        resolved_block_reason = block_reason
    elif unknown_areas:
        resolved_block_reason = "unknown_area"
    else:
        resolved_block_reason = "count_check_fail"

    stability_votes = {area: normalized[area].get("stability_votes", {}) for area in sorted(normalized)}
    evidence_images = {area: normalized[area].get("evidence_image") for area in sorted(normalized)}
    source_camera = source_camera or _latest_source_camera(normalized)
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": str(run_id or ""),
        "timestamp": utc_timestamp(),
        "source_camera": source_camera,
        "records": {area: normalized[area] for area in AREA_ORDER if area in normalized},
        "abnormal_areas": abnormal_areas,
        "unknown_areas": unknown_areas,
        "count_check": {
            "normal": len(normal_areas),
            "abnormal": len(abnormal_areas),
            "unknown": len(unknown_areas),
            "expected_normal": 2,
            "expected_abnormal": 2,
            "passed": count_passed,
        },
        "ready": ready,
        "block_reason": resolved_block_reason,
        "stability_votes": stability_votes,
        "evidence_image": evidence_images,
        "evidence_images": evidence_images,
    }


def _latest_source_camera(records: Mapping[str, Mapping[str, Any]]) -> str:
    latest = ""
    latest_timestamp = ""
    for record in records.values():
        timestamp = str(record.get("timestamp") or "")
        if timestamp >= latest_timestamp:
            latest_timestamp = timestamp
            latest = str(record.get("source_camera") or "")
    return latest


def load_round_result(path: Path | str = DEFAULT_ROUND_RESULT_PATH, source_camera: str = "") -> dict[str, Any]:
    result_path = Path(path)
    if not result_path.exists():
        return empty_round_result(source_camera=source_camera, block_reason="round_result_missing")
    try:
        text = result_path.read_text(encoding="utf-8")
    except OSError:
        return empty_round_result(source_camera=source_camera, block_reason="round_result_unreadable")
    if not text.strip():
        return empty_round_result(source_camera=source_camera, block_reason="round_result_empty")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return empty_round_result(source_camera=source_camera, block_reason="round_result_invalid")
    if not isinstance(data, dict):
        return empty_round_result(source_camera=source_camera, block_reason="round_result_invalid")
    return _normalize_loaded_round_result(data, source_camera=source_camera)


def _normalize_loaded_round_result(data: Mapping[str, Any], source_camera: str = "") -> dict[str, Any]:
    records_data = data.get("records")
    records: dict[str, Mapping[str, Any]] = {}
    if isinstance(records_data, Mapping):
        for key, value in records_data.items():
            if isinstance(value, Mapping):
                area = _area_key(value.get("letter") or value.get("area") or key)
                if area is not None:
                    records[area] = value
    else:
        for area in AREA_ORDER:
            value = data.get(area)
            if isinstance(value, Mapping):
                records[area] = value
    if not records and any(key in data for key in ("abnormal_areas", "unknown_areas", "ready", "count_check")):
        return _normalize_area_only_round_result(data, source_camera=source_camera)
    normalized = build_round_result(
        records,
        source_camera=source_camera or str(data.get("source_camera") or ""),
        run_id=str(data.get("run_id") or ""),
    )
    if data.get("ready") is False and data.get("block_reason") and not normalized["ready"]:
        normalized["block_reason"] = str(data["block_reason"])
    return normalized


def _normalize_area_only_round_result(data: Mapping[str, Any], source_camera: str = "") -> dict[str, Any]:
    abnormal_areas = _area_list(data.get("abnormal_areas"))
    unknown_areas = _area_list(data.get("unknown_areas"))
    normal_count = max(0, len(AREA_ORDER) - len(abnormal_areas) - len(unknown_areas))
    ready = bool(data.get("ready")) and not unknown_areas
    count_check = data.get("count_check")
    if isinstance(count_check, Mapping):
        count_passed = bool(count_check.get("passed", ready))
        normal_count = int(count_check.get("normal", normal_count) or 0)
        abnormal_count = int(count_check.get("abnormal", len(abnormal_areas)) or 0)
        unknown_count = int(count_check.get("unknown", len(unknown_areas)) or 0)
    else:
        count_passed = ready
        abnormal_count = len(abnormal_areas)
        unknown_count = len(unknown_areas)
    ready = ready and count_passed
    if ready:
        block_reason = None
    else:
        block_reason = str(data.get("block_reason") or ("unknown_area" if unknown_areas else "count_check_fail"))
    return {
        "schema_version": int(data.get("schema_version") or 1),
        "run_id": str(data.get("run_id") or ""),
        "timestamp": str(data.get("timestamp") or utc_timestamp()),
        "source_camera": source_camera or str(data.get("source_camera") or ""),
        "records": {},
        "abnormal_areas": abnormal_areas,
        "unknown_areas": unknown_areas,
        "count_check": {
            "normal": normal_count,
            "abnormal": abnormal_count,
            "unknown": unknown_count,
            "expected_normal": 2,
            "expected_abnormal": 2,
            "passed": count_passed,
        },
        "ready": ready,
        "block_reason": block_reason,
        "stability_votes": dict(data.get("stability_votes") or {}),
        "evidence_image": data.get("evidence_image") or {},
        "evidence_images": data.get("evidence_images") or data.get("evidence_image") or {},
    }


def write_json_atomic(path: Path | str, data: Mapping[str, Any]) -> None:
    result_path = Path(path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = result_path.with_name(f".{result_path.name}.tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_path, result_path)


def write_empty_round_result(
    path: Path | str = DEFAULT_ROUND_RESULT_PATH,
    source_camera: str = "",
    block_reason: str = "unknown_area",
    run_id: str = "",
) -> dict[str, Any]:
    data = empty_round_result(source_camera=source_camera, block_reason=block_reason, run_id=run_id)
    write_json_atomic(path, data)
    return data


def write_latest_stop_result(
    path: Path | str,
    record: InspectionRecord | Mapping[str, Any],
) -> dict[str, Any]:
    data = record_to_dict(record)
    write_json_atomic(path, data)
    return data


def merge_record_into_round(
    path: Path | str,
    record: InspectionRecord | Mapping[str, Any],
    source_camera: str = "",
    run_id: str = "",
) -> dict[str, Any]:
    current = load_round_result(path, source_camera=source_camera)
    current_run_id = str(current.get("run_id") or "")
    if run_id and current_run_id != run_id:
        records: dict[str, Any] = {}
    else:
        records = dict(current.get("records") or {})
    record_data = record_to_dict(record)
    area = _area_key(record_data.get("letter"))
    if area is not None:
        records[area] = record_data
    updated = build_round_result(
        records,
        source_camera=source_camera or record_data.get("source_camera", ""),
        run_id=run_id or current_run_id,
    )
    write_json_atomic(path, updated)
    return updated


def load_round_areas(path: Path | str = DEFAULT_ROUND_RESULT_PATH) -> tuple[list[str], list[str], dict[str, Any]]:
    data = load_round_result(path)
    return list(data["abnormal_areas"]), list(data["unknown_areas"]), data


def evaluate_round_gate(path: Path | str = DEFAULT_ROUND_RESULT_PATH, *, expected_run_id: str = "") -> RoundGate:
    data = load_round_result(path)
    unknown_areas = list(data.get("unknown_areas") or [])
    abnormal_areas = list(data.get("abnormal_areas") or [])
    allowed = bool(data.get("ready")) and not unknown_areas
    if expected_run_id and str(data.get("run_id") or "") != expected_run_id:
        allowed = False
        data["block_reason"] = "round_result_run_id_mismatch"
    block_reason = None if allowed else str(data.get("block_reason") or "round_not_ready")
    return RoundGate(allowed, abnormal_areas, unknown_areas, block_reason, data)


def records_from_round_result(data: Mapping[str, Any]) -> dict[str, InspectionRecord]:
    records: dict[str, InspectionRecord] = {}
    records_data = data.get("records") or {}
    if not isinstance(records_data, Mapping):
        return records
    for key, value in records_data.items():
        if not isinstance(value, Mapping):
            continue
        record = record_from_dict({"letter": key, **value})
        if record is not None:
            records[record.letter] = record
    return records

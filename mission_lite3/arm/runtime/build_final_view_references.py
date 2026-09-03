import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE_DIR = MODULE_DIR / "可抓取参考图"


def _long_side(feature: Mapping[str, Any]) -> float:
    width, height = feature["size_px"]
    return max(abs(float(width)), abs(float(height)))


def _short_side(feature: Mapping[str, Any]) -> float:
    width, height = feature["size_px"]
    return min(abs(float(width)), abs(float(height)))


def _positive_float(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric <= 0.0:
        return None
    return numeric


def _far_edge_over_depth(feature: Mapping[str, Any]) -> Optional[float]:
    explicit = _positive_float(feature.get("far_edge_over_depth"))
    if explicit is not None:
        return explicit
    far_edge = _positive_float(feature.get("far_edge_px"))
    depth_edge = _positive_float(feature.get("visible_depth_edge_px"))
    if far_edge is not None and depth_edge is not None:
        return far_edge / depth_edge
    return None


def _visible_depth(feature: Mapping[str, Any]) -> float:
    far_edge = _positive_float(feature.get("far_edge_px"))
    depth_edge = _positive_float(feature.get("visible_depth_edge_px"))
    if far_edge is not None and depth_edge is not None:
        return depth_edge / far_edge
    long_side = _long_side(feature)
    return 0.0 if long_side <= 0.0 else _short_side(feature) / long_side


def _long_over_short(feature: Mapping[str, Any]) -> float:
    far_edge_over_depth = _far_edge_over_depth(feature)
    if far_edge_over_depth is not None:
        return far_edge_over_depth
    short_side = _short_side(feature)
    return 0.0 if short_side <= 0.0 else _long_side(feature) / short_side


def component_touches_horizontal_frame_edge(
    feature: Mapping[str, Any],
    frame_width: int,
    *,
    margin_px: float = 3.0,
) -> bool:
    bbox = feature.get("bbox_px")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False
    x, _y, width, _height = (float(value) for value in bbox)
    return x <= margin_px or x + width >= float(frame_width) - margin_px


def build_reference_document(image_features: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    accepted: List[Dict[str, Any]] = []
    reviewed_images: List[Dict[str, Any]] = []
    for item in image_features:
        reviewed_images.append(dict(item))
        if bool(item.get("needs_review", False)):
            continue
        selected = item.get("selected")
        if not isinstance(selected, Mapping):
            continue
        feature = dict(selected)
        feature["image"] = str(item.get("image", ""))
        feature["visible_depth"] = _visible_depth(feature)
        feature["long_over_short"] = _long_over_short(feature)
        accepted.append(feature)
    if not accepted:
        raise ValueError("no accepted final grasp reference features")
    return {
        "schema_version": 1,
        "kind": "final_grasp_reference_feature_set",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": "arm/可抓取参考图",
        "features": accepted,
        "bounds": {
            "max_visible_depth": max(_visible_depth(feature) for feature in accepted),
            "max_long_over_short": max(_long_over_short(feature) for feature in accepted),
            "min_long_side_px": min(_long_side(feature) for feature in accepted),
            "min_area_px": min(float(feature.get("area_px", 0.0)) for feature in accepted),
        },
        "reviewed_images": reviewed_images,
    }


def write_reference_document(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_final_grasp_matcher():
    if str(MODULE_DIR) not in sys.path:
        sys.path.insert(0, str(MODULE_DIR))
    import final_grasp_matcher

    return final_grasp_matcher


def _load_reference_hint(
    final_grasp_matcher,
    detector_config: Mapping[str, Any],
    match_config: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    try:
        return final_grasp_matcher._load_reference_feature(
            detector_config,
            match_config,
            MODULE_DIR,
        )
    except Exception:
        return None


def _extract_image_features(
    image_dir: Path,
    detector_config: Mapping[str, Any],
    match_config: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    import cv2

    final_grasp_matcher = _load_final_grasp_matcher()
    reference_hint = _load_reference_hint(
        final_grasp_matcher,
        detector_config,
        match_config,
    )
    features: List[Dict[str, Any]] = []
    for image_path in sorted(image_dir.glob("*.jpg")):
        frame = cv2.imread(str(image_path))
        if frame is None:
            features.append(
                {
                    "image": image_path.name,
                    "selected": None,
                    "candidates": [],
                    "needs_review": True,
                    "reason": "image could not be read",
                }
            )
            continue
        selected = final_grasp_matcher.extract_red_component_feature(
            frame,
            detector_config,
            match_config,
            reference_hint,
        )
        touches_edge = bool(
            selected
            and component_touches_horizontal_frame_edge(
                selected,
                int(frame.shape[1]),
            )
        )
        features.append(
            {
                "image": image_path.name,
                "selected": selected,
                "candidates": [selected] if selected else [],
                "needs_review": selected is None or touches_edge,
                "reason": (
                    "selected component touches horizontal frame edge"
                    if touches_edge
                    else ""
                ),
            }
        )
    return features


def build_from_files(
    image_dir: Path,
    config_path: Path,
    reference_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    detector_config = json.loads(config_path.read_text(encoding="utf-8"))
    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    match_config = dict(reference.get("final_view_match", {}))
    image_features = _extract_image_features(image_dir, detector_config, match_config)
    document = build_reference_document(image_features)
    write_reference_document(output_path, document)
    return document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build multi-reference final grasp feature file"
    )
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--config", default=str(MODULE_DIR / "strip_detector_grasp_config.json"))
    parser.add_argument("--reference", default=str(MODULE_DIR / "grasp_reference_square_face.json"))
    parser.add_argument("--output", default=str(MODULE_DIR / "grasp_final_view_references.json"))
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    document = build_from_files(
        Path(args.image_dir),
        Path(args.config),
        Path(args.reference),
        Path(args.output),
    )
    print(
        json.dumps(
            {
                "ok": True,
                "features": len(document["features"]),
                "reviewed_images": len(document["reviewed_images"]),
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

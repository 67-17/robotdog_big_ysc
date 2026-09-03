from __future__ import annotations

import argparse
from pathlib import Path

from ..config_loader import load_config
from ..route_validation import route_boundary_errors, simulate_route_sequence


DEFAULT_SEQUENCE = (
    "pass_obstacle",
    "inspect_stop_1_arrive",
    "inspect_stop_3_arrive",
    "inspect_stop_2_arrive",
    "inspect_stop_4_arrive",
    "inspect_stop_4_depart",
    "pickup_from_upper_inspection",
    "place_from_pickup",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Print the approximate field-frame pose after each scripted route")
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument("routes", nargs="*", default=list(DEFAULT_SEQUENCE))
    args = parser.parse_args()
    config = load_config(args.config_dir)
    poses = simulate_route_sequence(config, args.routes)
    for name, pose in poses.items():
        print(f"{name}: x={pose.x:.3f}m y={pose.y:.3f}m yaw={pose.yaw:.4f}rad")
    errors = route_boundary_errors(config, poses)
    for error in errors:
        print(f"ERROR: {error}")
    return 2 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

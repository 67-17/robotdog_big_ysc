from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class RoutePose:
    x: float
    y: float
    yaw: float


def normalize_yaw(yaw: float) -> float:
    return (float(yaw) + math.pi) % (2.0 * math.pi) - math.pi


def simulate_route_actions(start: RoutePose, actions: Iterable[Mapping[str, Any]]) -> RoutePose:
    """Integrate configured robot-frame actions into the field frame."""
    x, y, yaw = float(start.x), float(start.y), float(start.yaw)
    for action in actions:
        kind = str(action.get("action") or "")
        if kind == "turn":
            yaw = normalize_yaw(yaw + float(action.get("yaw_rad", 0.0)))
            continue
        if kind in {
            "wait",
            "",
            "placement_row_yaw_align",
            "placement_lane_strafe",
            "placement_letter_approach",
            "pickup_lane_restore",
        }:
            continue
        if kind == "obstacle_forward":
            forward = max(0, int(action.get("steps", 1))) * float(action.get("clear_step_m", 0.25))
            strafe = 0.0
        elif kind == "forward":
            forward = float(action.get("distance_m", 0.0))
            strafe = 0.0
        elif kind == "backward":
            forward = -abs(float(action.get("distance_m", 0.0)))
            strafe = 0.0
        elif kind == "strafe":
            forward = 0.0
            strafe = float(action.get("distance_m", 0.0))
        else:
            raise ValueError(f"unknown route action: {kind!r}")
        x += forward * math.cos(yaw) - strafe * math.sin(yaw)
        y += forward * math.sin(yaw) + strafe * math.cos(yaw)
    return RoutePose(x, y, yaw)


def simulate_route_sequence(config: Mapping[str, Any], route_names: Iterable[str]) -> dict[str, RoutePose]:
    start_cfg = config["waypoints"]["start"]
    pose = RoutePose(float(start_cfg["x"]), float(start_cfg["y"]), float(start_cfg["yaw"]))
    route = config["scripted_route"]
    results: dict[str, RoutePose] = {}
    for name in route_names:
        actions = route.get(name)
        if not isinstance(actions, list):
            raise ValueError(f"route is missing or not a list: {name}")
        pose = simulate_route_actions(pose, actions)
        results[name] = pose
    return results


def route_boundary_errors(config: Mapping[str, Any], poses: Mapping[str, RoutePose]) -> list[str]:
    field = config["field"]
    width = float(field["width_m"])
    length = float(field["length_m"])
    # Configured cardinal yaws use four decimal places, so tolerate the
    # millimeter-scale drift introduced by integrating those approximations.
    epsilon = 1e-3
    errors: list[str] = []
    for name, pose in poses.items():
        if (
            pose.x < -width - epsilon
            or pose.x > epsilon
            or pose.y < -epsilon
            or pose.y > length + epsilon
        ):
            errors.append(
                f"{name} leaves field bounds: x={pose.x:.3f}, y={pose.y:.3f}, "
                f"expected x=[{-width:.3f},0.000], y=[0.000,{length:.3f}]"
            )
    return errors

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_DISTANCE_MM = 20.0
MAX_SEGMENT_DISTANCE_MM = 20.0
MAX_TOTAL_DISTANCE_MM = 50.0
DEFAULT_PROBE_DELTA_DEG = 2.0
MIN_PROBE_DELTA_DEG = 2.0
MAX_PROBE_DELTA_DEG = 3.0
DEFAULT_MAX_PROBE_ENDPOINT_MM = 30.0
DEFAULT_MAX_PROBE_Z_MM = 20.0
DEFAULT_MAX_JOINT_DELTA_DEG = 6.0
DEFAULT_SPEED = 3.0
DEFAULT_ACCELERATION = 3.0
MAX_SPEED = 5.0
MAX_ACCELERATION = 5.0
DEFAULT_GRIPPER_OPEN_MAX_DEG = -30.0
DEFAULT_NEGATIVE_E_COMPENSATION_DEG = 5.0
DEFAULT_JOINT_LIMITS = {
    "s": (-85.0, 85.0),
    "e": (-85.0, 85.0),
}
CONTROLLER_JOINT_LIMITS = {
    "s": (-90.0, 90.0),
    "e": (-90.0, 90.0),
}
DEFAULT_MOTION_TIMEOUT_SECONDS = 12.0
MIN_MOTION_TIMEOUT_SECONDS = 1.0
MAX_MOTION_TIMEOUT_SECONDS = 60.0
STABLE_WINDOW_SAMPLES = 4
STABLE_JOINT_RANGE_DEG = 0.05
STABLE_ENDPOINT_RANGE_MM = 0.4
MOVEMENT_JOINT_THRESHOLD_DEG = 0.15
MOVEMENT_ENDPOINT_THRESHOLD_MM = 0.8
DEFAULT_OUTPUT_ROOT = MODULE_DIR / "run-log"


class SafetyStop(RuntimeError):
    """Raised when another arm movement would violate a configured gate."""


def _finite_float(value: Any, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def _endpoint_xyz(status: Mapping[str, Any]) -> tuple[float, float, float]:
    endpoint = status.get("endpoint_xyz_mm")
    if endpoint is not None:
        if len(endpoint) != 3:
            raise ValueError("endpoint_xyz_mm must contain x, y, z")
        return (
            _finite_float(endpoint[0], "x"),
            _finite_float(endpoint[1], "y"),
            _finite_float(endpoint[2], "z"),
        )
    return (
        _finite_float(status["x"], "x"),
        _finite_float(status["y"], "y"),
        _finite_float(status["z"], "z"),
    )


def endpoint_rz(status: Mapping[str, Any]) -> tuple[float, float]:
    x, y, z = _endpoint_xyz(status)
    return math.hypot(x, y), z


def endpoint_displacement(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, float]:
    before_x, before_y, before_z = _endpoint_xyz(before)
    after_x, after_y, after_z = _endpoint_xyz(after)
    before_r = math.hypot(before_x, before_y)
    after_r = math.hypot(after_x, after_y)
    delta_x = after_x - before_x
    delta_y = after_y - before_y
    delta_z = after_z - before_z
    return {
        "delta_x_mm": delta_x,
        "delta_y_mm": delta_y,
        "delta_z_mm": delta_z,
        "delta_r_mm": after_r - before_r,
        "endpoint_step_mm": math.sqrt(
            delta_x * delta_x + delta_y * delta_y + delta_z * delta_z
        ),
    }


def build_local_jacobian(
    samples: Mapping[str, Mapping[str, Any]],
    *,
    min_actual_joint_delta_deg: float = 0.3,
) -> tuple[tuple[float, float], tuple[float, float]]:
    columns: dict[str, tuple[float, float]] = {}
    for joint in ("s", "e"):
        sample = samples[joint]
        actual_delta = _finite_float(
            sample["actual_joint_delta_deg"],
            f"{joint} actual_joint_delta_deg",
        )
        if abs(actual_delta) < float(min_actual_joint_delta_deg):
            raise SafetyStop(
                f"{joint} probe actual movement is below "
                f"{float(min_actual_joint_delta_deg):g} degrees"
            )
        columns[joint] = (
            _finite_float(sample["delta_r_mm"], f"{joint} delta_r_mm")
            / actual_delta,
            _finite_float(sample["delta_z_mm"], f"{joint} delta_z_mm")
            / actual_delta,
        )
    return (
        (columns["s"][0], columns["e"][0]),
        (columns["s"][1], columns["e"][1]),
    )


def solve_joint_delta(
    matrix: tuple[tuple[float, float], tuple[float, float]],
    target_rz_mm: tuple[float, float],
    *,
    min_normalized_determinant: float = 0.08,
) -> dict[str, float]:
    a = _finite_float(matrix[0][0], "dr/ds")
    b = _finite_float(matrix[0][1], "dr/de")
    c = _finite_float(matrix[1][0], "dz/ds")
    d = _finite_float(matrix[1][1], "dz/de")
    target_r = _finite_float(target_rz_mm[0], "target delta_r")
    target_z = _finite_float(target_rz_mm[1], "target delta_z")
    determinant = a * d - b * c
    first_norm = math.hypot(a, c)
    second_norm = math.hypot(b, d)
    scale = first_norm * second_norm
    normalized = abs(determinant) / scale if scale > 0.0 else 0.0
    if normalized < float(min_normalized_determinant):
        raise SafetyStop(
            "local s/e Jacobian is singular or its columns are near parallel"
        )
    return {
        "s": (d * target_r - b * target_z) / determinant,
        "e": (-c * target_r + a * target_z) / determinant,
    }


def validate_distance_mm(
    distance_mm: float,
    *,
    max_distance_mm: float = MAX_SEGMENT_DISTANCE_MM,
) -> float:
    distance = _finite_float(distance_mm, "distance_mm")
    if distance <= 0.0:
        raise ValueError("distance_mm must be positive")
    if distance > float(max_distance_mm):
        raise ValueError(
            f"one guarded segment cannot exceed {float(max_distance_mm):g} mm"
        )
    return distance


def plan_segment_distances(
    total_distance_mm: float,
    *,
    max_segment_distance_mm: float = MAX_SEGMENT_DISTANCE_MM,
    max_total_distance_mm: float = MAX_TOTAL_DISTANCE_MM,
) -> list[float]:
    total = _finite_float(total_distance_mm, "total_distance_mm")
    if total <= 0.0:
        raise ValueError("total distance must be positive")
    if total > float(max_total_distance_mm):
        raise ValueError(
            f"total distance cannot exceed "
            f"{float(max_total_distance_mm):g} mm"
        )
    segment_limit = _finite_float(
        max_segment_distance_mm,
        "max_segment_distance_mm",
    )
    if segment_limit <= 0.0:
        raise ValueError("segment distance must be positive")
    segments: list[float] = []
    remaining = total
    while remaining > 1e-9:
        segment = min(segment_limit, remaining)
        segments.append(float(segment))
        remaining -= segment
    return segments


def validate_runtime_parameters(
    *,
    probe_delta_deg: float,
    spd: float,
    acc: float,
    timeout_seconds: float,
) -> dict[str, float]:
    probe = _finite_float(probe_delta_deg, "probe_delta_deg")
    speed = _finite_float(spd, "spd")
    acceleration = _finite_float(acc, "acc")
    timeout = _finite_float(timeout_seconds, "timeout_seconds")
    if probe < MIN_PROBE_DELTA_DEG or probe > MAX_PROBE_DELTA_DEG:
        raise ValueError(
            f"probe angle must be between {MIN_PROBE_DELTA_DEG:g} "
            f"and {MAX_PROBE_DELTA_DEG:g} degrees"
        )
    if speed <= 0.0 or speed > MAX_SPEED:
        raise ValueError(
            f"speed must be positive and at most {MAX_SPEED:g}"
        )
    if acceleration <= 0.0 or acceleration > MAX_ACCELERATION:
        raise ValueError(
            "acceleration must be positive and at most "
            f"{MAX_ACCELERATION:g}"
        )
    if (
        timeout < MIN_MOTION_TIMEOUT_SECONDS
        or timeout > MAX_MOTION_TIMEOUT_SECONDS
    ):
        raise ValueError(
            f"timeout must be between {MIN_MOTION_TIMEOUT_SECONDS:g} "
            f"and {MAX_MOTION_TIMEOUT_SECONDS:g} seconds"
        )
    return {
        "probe_delta_deg": probe,
        "spd": speed,
        "acc": acceleration,
        "timeout_seconds": timeout,
    }


def validate_joint_solution(
    solution: Mapping[str, float],
    *,
    max_joint_delta_deg: float = DEFAULT_MAX_JOINT_DELTA_DEG,
) -> dict[str, float]:
    validated = {
        joint: _finite_float(solution[joint], f"{joint} joint delta")
        for joint in ("s", "e")
    }
    largest = max(abs(value) for value in validated.values())
    if largest > float(max_joint_delta_deg):
        raise SafetyStop(
            f"solved joint change {largest:.2f} exceeds "
            f"{float(max_joint_delta_deg):g} degrees"
        )
    return validated


def validate_joint_targets(
    targets: Mapping[str, float],
    *,
    joint_limits: Mapping[str, tuple[float, float]] = DEFAULT_JOINT_LIMITS,
) -> dict[str, float]:
    validated: dict[str, float] = {}
    for joint in ("s", "e"):
        value = _finite_float(targets[joint], f"{joint} target")
        lower, upper = joint_limits[joint]
        if value < float(lower) or value > float(upper):
            raise SafetyStop(
                f"{joint} target {value:.2f} is outside "
                f"[{float(lower):g}, {float(upper):g}] degrees"
            )
        validated[joint] = value
    return validated


def validate_gripper_open(
    pose_deg: Mapping[str, float],
    *,
    open_max_deg: float = DEFAULT_GRIPPER_OPEN_MAX_DEG,
) -> float:
    gripper = _finite_float(pose_deg["h"], "gripper angle")
    if gripper > float(open_max_deg):
        raise SafetyStop(
            f"gripper must be open at or below {float(open_max_deg):g} degrees"
        )
    return gripper


def build_joint_command(
    current_pose_deg: Mapping[str, float],
    targets: Mapping[str, float],
    *,
    spd: float,
    acc: float,
) -> dict[str, float | int]:
    unexpected = set(targets) - {"s", "e"}
    if unexpected:
        raise ValueError(
            "local Cartesian jog may only change s/e: "
            + ", ".join(sorted(unexpected))
        )
    command: dict[str, float | int] = {"T": 122}
    for joint in ("b", "s", "e", "w"):
        command[joint] = _finite_float(current_pose_deg[joint], joint)
    for joint, value in targets.items():
        command[joint] = _finite_float(value, f"{joint} target")
    command["h"] = _finite_float(current_pose_deg["h"], "h")
    command["spd"] = _finite_float(spd, "spd")
    command["acc"] = _finite_float(acc, "acc")
    return command


def command_targets_for_expected(
    current_pose_deg: Mapping[str, float],
    expected_targets: Mapping[str, float],
    *,
    negative_e_compensation_deg: float = DEFAULT_NEGATIVE_E_COMPENSATION_DEG,
    command_joint_limits: Mapping[
        str, tuple[float, float]
    ] = CONTROLLER_JOINT_LIMITS,
) -> tuple[dict[str, float], bool]:
    command_targets = {
        joint: _finite_float(value, f"{joint} expected target")
        for joint, value in expected_targets.items()
    }
    compensated = False
    if (
        "e" in command_targets
        and command_targets["e"]
        < _finite_float(current_pose_deg["e"], "current e") - 0.1
    ):
        command_targets["e"] -= abs(
            _finite_float(
                negative_e_compensation_deg,
                "negative_e_compensation_deg",
            )
        )
        compensated = True
    for joint, value in command_targets.items():
        if joint not in command_joint_limits:
            continue
        lower, upper = command_joint_limits[joint]
        if value < float(lower) or value > float(upper):
            raise SafetyStop(
                f"{joint} compensated command {value:.2f} exceeds "
                f"controller limit [{float(lower):g}, {float(upper):g}]"
            )
    return command_targets, compensated


def validate_final_displacement(
    *,
    requested_distance_mm: float,
    delta_r_mm: float,
    delta_z_mm: float,
    endpoint_step_mm: float,
) -> dict[str, Any]:
    requested = validate_distance_mm(requested_distance_mm)
    delta_r = _finite_float(delta_r_mm, "delta_r_mm")
    delta_z = _finite_float(delta_z_mm, "delta_z_mm")
    endpoint_step = _finite_float(endpoint_step_mm, "endpoint_step_mm")
    minimum_radial = max(2.0, requested - 15.0)
    maximum_radial = requested + 20.0
    maximum_height = 12.0
    maximum_endpoint_step = requested + 25.0
    if delta_r <= 0.0:
        raise SafetyStop("endpoint moved in the wrong radial direction")
    if delta_r < minimum_radial or delta_r > maximum_radial:
        raise SafetyStop(
            f"radial movement {delta_r:.2f} mm is outside "
            f"[{minimum_radial:.2f}, {maximum_radial:.2f}] mm"
        )
    if abs(delta_z) > maximum_height:
        raise SafetyStop(
            f"height change {delta_z:.2f} mm exceeds "
            f"{maximum_height:.2f} mm"
        )
    if endpoint_step > maximum_endpoint_step:
        raise SafetyStop(
            f"endpoint step {endpoint_step:.2f} mm exceeds "
            f"{maximum_endpoint_step:.2f} mm"
        )
    return {
        "accepted": True,
        "minimum_radial_mm": minimum_radial,
        "maximum_radial_mm": maximum_radial,
        "maximum_height_mm": maximum_height,
        "maximum_endpoint_step_mm": maximum_endpoint_step,
    }


def validate_total_displacement(
    *,
    requested_distance_mm: float,
    delta_r_mm: float,
    delta_z_mm: float,
    endpoint_step_mm: float,
) -> dict[str, Any]:
    requested = _finite_float(
        requested_distance_mm,
        "requested total distance",
    )
    if requested <= 0.0 or requested > MAX_TOTAL_DISTANCE_MM:
        raise ValueError(
            f"requested total distance must be positive and at most "
            f"{MAX_TOTAL_DISTANCE_MM:g} mm"
        )
    delta_r = _finite_float(delta_r_mm, "total delta_r_mm")
    delta_z = _finite_float(delta_z_mm, "total delta_z_mm")
    endpoint_step = _finite_float(
        endpoint_step_mm,
        "total endpoint_step_mm",
    )
    minimum_radial = max(2.0, requested - 20.0)
    maximum_radial = requested + 20.0
    maximum_height = 20.0
    maximum_endpoint_step = requested + 30.0
    if delta_r < minimum_radial or delta_r > maximum_radial:
        raise SafetyStop(
            f"total radial movement {delta_r:.2f} mm is outside "
            f"[{minimum_radial:.2f}, {maximum_radial:.2f}] mm"
        )
    if abs(delta_z) > maximum_height:
        raise SafetyStop(
            f"total height change {delta_z:.2f} mm exceeds "
            f"{maximum_height:.2f} mm"
        )
    if endpoint_step > maximum_endpoint_step:
        raise SafetyStop(
            f"total endpoint step {endpoint_step:.2f} mm exceeds "
            f"{maximum_endpoint_step:.2f} mm"
        )
    return {
        "accepted": True,
        "minimum_radial_mm": minimum_radial,
        "maximum_radial_mm": maximum_radial,
        "maximum_height_mm": maximum_height,
        "maximum_endpoint_step_mm": maximum_endpoint_step,
    }


def _feedback_is_stable(status: Mapping[str, Any]) -> bool:
    if "feedback_stable" in status:
        return bool(status["feedback_stable"])
    return int(status.get("move", 1)) == 0


def validate_final_status(
    baseline: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    max_preserved_joint_drift_deg: float = 0.75,
    max_gripper_drift_deg: float = 1.5,
) -> dict[str, Any]:
    if not _feedback_is_stable(current):
        raise SafetyStop("arm feedback is not stable after movement")
    baseline_pose = _pose(baseline)
    current_pose = _pose(current)
    validate_gripper_open(current_pose)
    for joint in ("b", "w"):
        drift = abs(
            float(current_pose[joint]) - float(baseline_pose[joint])
        )
        if drift > float(max_preserved_joint_drift_deg):
            raise SafetyStop(
                f"preserved joint {joint} drift {drift:.2f} exceeds "
                f"{float(max_preserved_joint_drift_deg):g} degrees"
            )
    gripper_drift = abs(
        float(current_pose["h"]) - float(baseline_pose["h"])
    )
    if gripper_drift > float(max_gripper_drift_deg):
        raise SafetyStop(
            f"gripper drift {gripper_drift:.2f} exceeds "
            f"{float(max_gripper_drift_deg):g} degrees"
        )
    return {
        "feedback_stable": True,
        "move": int(current.get("move", 0)),
        "max_preserved_joint_drift_deg": float(
            max_preserved_joint_drift_deg
        ),
        "max_gripper_drift_deg": float(max_gripper_drift_deg),
    }


def validate_restored_status(
    baseline: Mapping[str, Any],
    restored: Mapping[str, Any],
    *,
    max_restore_joint_error_deg: float = 0.75,
    max_restore_endpoint_mm: float = 2.0,
    max_restore_z_mm: float = 1.5,
) -> dict[str, Any]:
    invariant = validate_final_status(baseline, restored)
    baseline_pose = _pose(baseline)
    restored_pose = _pose(restored)
    largest_joint_error = max(
        abs(float(restored_pose[joint]) - float(baseline_pose[joint]))
        for joint in ("s", "e")
    )
    if largest_joint_error > float(max_restore_joint_error_deg):
        raise SafetyStop(
            f"baseline restore joint error {largest_joint_error:.2f} exceeds "
            f"{float(max_restore_joint_error_deg):g} degrees"
        )
    displacement = endpoint_displacement(baseline, restored)
    if displacement["endpoint_step_mm"] > float(max_restore_endpoint_mm):
        raise SafetyStop(
            f"baseline restore endpoint error "
            f"{displacement['endpoint_step_mm']:.2f} exceeds "
            f"{float(max_restore_endpoint_mm):g} mm"
        )
    if abs(displacement["delta_z_mm"]) > float(max_restore_z_mm):
        raise SafetyStop(
            f"baseline restore z error {displacement['delta_z_mm']:.2f} "
            f"exceeds {float(max_restore_z_mm):g} mm"
        )
    return {
        **invariant,
        "largest_joint_error_deg": largest_joint_error,
        **displacement,
    }


def choose_probe_delta(
    current_deg: float,
    *,
    magnitude_deg: float,
    limits: tuple[float, float],
) -> float:
    current = _finite_float(current_deg, "current joint angle")
    magnitude = abs(_finite_float(magnitude_deg, "probe magnitude"))
    lower, upper = float(limits[0]), float(limits[1])
    if current + magnitude <= upper:
        return magnitude
    if current - magnitude >= lower:
        return -magnitude
    raise SafetyStop("no safe direction is available for the joint probe")


def _pose(status: Mapping[str, Any]) -> Mapping[str, float]:
    pose = status.get("pose_deg")
    if not isinstance(pose, Mapping):
        raise ValueError("status is missing pose_deg")
    return pose


def _plain_status(status: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "pose_deg": {
            key: float(value)
            for key, value in _pose(status).items()
        },
        "endpoint_xyz_mm": list(_endpoint_xyz(status)),
        "move": int(status.get("move", 0)),
        "feedback_stable": _feedback_is_stable(status),
        "raw": dict(status.get("raw", {})),
    }


class LocalCartesianJogRunner:
    def __init__(
        self,
        motion: Any,
        *,
        probe_delta_deg: float = DEFAULT_PROBE_DELTA_DEG,
        max_probe_endpoint_mm: float = DEFAULT_MAX_PROBE_ENDPOINT_MM,
        max_probe_z_mm: float = DEFAULT_MAX_PROBE_Z_MM,
        max_joint_delta_deg: float = DEFAULT_MAX_JOINT_DELTA_DEG,
        joint_limits: Mapping[str, tuple[float, float]] = DEFAULT_JOINT_LIMITS,
        gripper_open_max_deg: float = DEFAULT_GRIPPER_OPEN_MAX_DEG,
        spd: float = DEFAULT_SPEED,
        acc: float = DEFAULT_ACCELERATION,
    ):
        runtime = validate_runtime_parameters(
            probe_delta_deg=probe_delta_deg,
            spd=spd,
            acc=acc,
            timeout_seconds=DEFAULT_MOTION_TIMEOUT_SECONDS,
        )
        self.motion = motion
        self.probe_delta_deg = runtime["probe_delta_deg"]
        self.max_probe_endpoint_mm = float(max_probe_endpoint_mm)
        self.max_probe_z_mm = float(max_probe_z_mm)
        self.max_joint_delta_deg = float(max_joint_delta_deg)
        self.joint_limits = {
            joint: (float(bounds[0]), float(bounds[1]))
            for joint, bounds in joint_limits.items()
        }
        self.gripper_open_max_deg = float(gripper_open_max_deg)
        self.spd = runtime["spd"]
        self.acc = runtime["acc"]

    def _restore_baseline(
        self,
        baseline: Mapping[str, Any],
    ) -> tuple[bool, Optional[dict[str, Any]], str]:
        try:
            baseline_pose = _pose(baseline)
            restored = self.motion.move_joint_targets(
                {
                    "s": float(baseline_pose["s"]),
                    "e": float(baseline_pose["e"]),
                },
                spd=self.spd,
                acc=self.acc,
            )
            validate_restored_status(baseline, restored)
            return True, _plain_status(restored), ""
        except Exception as exc:
            return False, None, str(exc)

    def _probe_joint(
        self,
        joint: str,
        baseline: Mapping[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        baseline_pose = _pose(baseline)
        probe_delta = choose_probe_delta(
            float(baseline_pose[joint]),
            magnitude_deg=self.probe_delta_deg,
            limits=self.joint_limits[joint],
        )
        targets = {
            "s": float(baseline_pose["s"]),
            "e": float(baseline_pose["e"]),
        }
        targets[joint] += probe_delta
        after: Optional[Mapping[str, Any]] = None
        restored = False
        try:
            after = self.motion.move_joint_targets(
                targets,
                spd=self.spd,
                acc=self.acc,
            )
            displacement = endpoint_displacement(baseline, after)
            actual_delta = (
                float(_pose(after)[joint]) - float(baseline_pose[joint])
            )
            if abs(actual_delta) < 0.3:
                raise SafetyStop(
                    f"{joint} probe actual movement is below 0.3 degrees"
                )
            if displacement["endpoint_step_mm"] > self.max_probe_endpoint_mm:
                raise SafetyStop(
                    f"{joint} probe endpoint step exceeds "
                    f"{self.max_probe_endpoint_mm:g} mm"
                )
            if abs(displacement["delta_z_mm"]) > self.max_probe_z_mm:
                raise SafetyStop(
                    f"{joint} probe z change exceeds "
                    f"{self.max_probe_z_mm:g} mm"
                )
            sample = {
                "joint": joint,
                "command_delta_deg": probe_delta,
                "actual_joint_delta_deg": actual_delta,
                **displacement,
                "status_after": _plain_status(after),
            }
            return sample, restored
        finally:
            restored, _, restore_error = self._restore_baseline(baseline)
            if not restored:
                raise SafetyStop(
                    "failed to restore baseline after "
                    f"{joint} probe: {restore_error}"
                )

    def measure_jacobian(
        self,
        baseline: Mapping[str, Any],
    ) -> tuple[dict[str, dict[str, Any]], tuple[tuple[float, float], tuple[float, float]]]:
        samples: dict[str, dict[str, Any]] = {}
        for joint in ("s", "e"):
            sample, _ = self._probe_joint(joint, baseline)
            samples[joint] = sample
        matrix = build_local_jacobian(samples)
        return samples, matrix

    def run(
        self,
        *,
        distance_mm: float,
        execute: bool = False,
        execute_probes: bool = False,
    ) -> dict[str, Any]:
        requested = validate_distance_mm(distance_mm)
        if not execute and not execute_probes:
            return {
                "ok": True,
                "stage": "DRY_RUN",
                "requested_distance_mm": requested,
                "execute": False,
                "execute_probes": False,
                "probe_delta_deg": self.probe_delta_deg,
                "samples": {},
                "baseline_restored": True,
            }

        stage = "READ_BASELINE"
        baseline: Optional[dict[str, Any]] = None
        samples: dict[str, dict[str, Any]] = {}
        matrix: Optional[tuple[tuple[float, float], tuple[float, float]]] = None
        solution: Optional[dict[str, float]] = None
        actual: Optional[dict[str, float]] = None
        baseline_restored = False
        restore_error = ""
        try:
            baseline = _plain_status(self.motion.read_stable_status())
            validate_gripper_open(
                _pose(baseline),
                open_max_deg=self.gripper_open_max_deg,
            )
            stage = "PROBE"
            samples, matrix = self.measure_jacobian(baseline)
            solution = solve_joint_delta(matrix, (requested, 0.0))
            solution = validate_joint_solution(
                solution,
                max_joint_delta_deg=self.max_joint_delta_deg,
            )
            baseline_pose = _pose(baseline)
            targets = validate_joint_targets(
                {
                    "s": float(baseline_pose["s"]) + solution["s"],
                    "e": float(baseline_pose["e"]) + solution["e"],
                },
                joint_limits=self.joint_limits,
            )
            baseline_restored = True
            if not execute:
                return {
                    "ok": True,
                    "stage": "PROBES_COMPLETE",
                    "requested_distance_mm": requested,
                    "execute": False,
                    "execute_probes": True,
                    "baseline": baseline,
                    "samples": samples,
                    "jacobian_rz_mm_per_deg": matrix,
                    "solution_delta_deg": solution,
                    "target_joints_deg": targets,
                    "baseline_restored": True,
                }

            stage = "FORWARD_MOVE"
            final_status = self.motion.move_joint_targets(
                targets,
                spd=self.spd,
                acc=self.acc,
            )
            actual = endpoint_displacement(baseline, final_status)
            stage = "FINAL_VALIDATION"
            status_validation = validate_final_status(
                baseline,
                final_status,
            )
            validation = validate_final_displacement(
                requested_distance_mm=requested,
                delta_r_mm=actual["delta_r_mm"],
                delta_z_mm=actual["delta_z_mm"],
                endpoint_step_mm=actual["endpoint_step_mm"],
            )
            baseline_restored = False
            return {
                "ok": True,
                "stage": "COMPLETE",
                "requested_distance_mm": requested,
                "execute": True,
                "execute_probes": True,
                "baseline": baseline,
                "samples": samples,
                "jacobian_rz_mm_per_deg": matrix,
                "solution_delta_deg": solution,
                "target_joints_deg": targets,
                "final_status": _plain_status(final_status),
                "actual": actual,
                "validation": {
                    **validation,
                    "status": status_validation,
                },
                "baseline_restored": False,
            }
        except Exception as exc:
            if baseline is not None:
                baseline_restored, _, restore_error = self._restore_baseline(
                    baseline
                )
            return {
                "ok": False,
                "stage": stage,
                "reason": str(exc),
                "requested_distance_mm": requested,
                "execute": bool(execute),
                "execute_probes": bool(execute or execute_probes),
                "baseline": baseline,
                "samples": samples,
                "jacobian_rz_mm_per_deg": matrix,
                "solution_delta_deg": solution,
                "actual": actual,
                "baseline_restored": baseline_restored,
                "restore_error": restore_error,
            }
        except KeyboardInterrupt:
            if baseline is not None:
                self._restore_baseline(baseline)
            raise


class LocalCartesianSequenceRunner:
    def __init__(
        self,
        motion: Any,
        segment_runner: LocalCartesianJogRunner,
    ):
        self.motion = motion
        self.segment_runner = segment_runner

    def run(
        self,
        *,
        total_distance_mm: float,
        execute: bool = False,
    ) -> dict[str, Any]:
        planned_segments = plan_segment_distances(total_distance_mm)
        if not execute:
            return {
                "ok": True,
                "stage": "SEQUENCE_DRY_RUN",
                "requested_total_distance_mm": float(total_distance_mm),
                "planned_segments_mm": planned_segments,
                "segments": [],
                "baseline_restored": True,
            }

        initial = _plain_status(self.motion.read_stable_status())
        validate_gripper_open(_pose(initial))
        segment_results: list[dict[str, Any]] = []
        cumulative_requested = 0.0
        for index, distance in enumerate(planned_segments, start=1):
            result = self.segment_runner.run(
                distance_mm=distance,
                execute=True,
            )
            segment_results.append(result)
            if not result.get("ok"):
                return {
                    "ok": False,
                    "stage": f"SEGMENT_{index}",
                    "reason": str(result.get("reason", "segment failed")),
                    "requested_total_distance_mm": float(
                        total_distance_mm
                    ),
                    "planned_segments_mm": planned_segments,
                    "segments": segment_results,
                    "initial_status": initial,
                    "baseline_restored": bool(
                        result.get("baseline_restored", False)
                    ),
                }
            cumulative_requested += float(distance)
            cumulative_actual = endpoint_displacement(
                initial,
                result["final_status"],
            )
            try:
                validate_total_displacement(
                    requested_distance_mm=cumulative_requested,
                    delta_r_mm=cumulative_actual["delta_r_mm"],
                    delta_z_mm=cumulative_actual["delta_z_mm"],
                    endpoint_step_mm=cumulative_actual["endpoint_step_mm"],
                )
            except (SafetyStop, ValueError) as exc:
                restored, restored_status, restore_error = (
                    self.segment_runner._restore_baseline(result["baseline"])
                )
                return {
                    "ok": False,
                    "stage": f"SEQUENCE_VALIDATION_{index}",
                    "reason": str(exc),
                    "requested_total_distance_mm": float(
                        total_distance_mm
                    ),
                    "requested_cumulative_distance_mm": cumulative_requested,
                    "planned_segments_mm": planned_segments,
                    "segments": segment_results,
                    "initial_status": initial,
                    "final_status": result["final_status"],
                    "actual": cumulative_actual,
                    "baseline_restored": restored,
                    "restored_status": restored_status,
                    "restore_error": restore_error,
                }

        final_status = segment_results[-1]["final_status"]
        actual = endpoint_displacement(initial, final_status)
        try:
            status_validation = validate_final_status(
                initial,
                final_status,
            )
            total_validation = validate_total_displacement(
                requested_distance_mm=float(total_distance_mm),
                delta_r_mm=actual["delta_r_mm"],
                delta_z_mm=actual["delta_z_mm"],
                endpoint_step_mm=actual["endpoint_step_mm"],
            )
        except (SafetyStop, ValueError) as exc:
            return {
                "ok": False,
                "stage": "SEQUENCE_VALIDATION",
                "reason": str(exc),
                "requested_total_distance_mm": float(
                    total_distance_mm
                ),
                "planned_segments_mm": planned_segments,
                "segments": segment_results,
                "initial_status": initial,
                "final_status": final_status,
                "actual": actual,
                "baseline_restored": False,
            }
        return {
            "ok": True,
            "stage": "SEQUENCE_COMPLETE",
            "requested_total_distance_mm": float(total_distance_mm),
            "planned_segments_mm": planned_segments,
            "segments": segment_results,
            "initial_status": initial,
            "final_status": final_status,
            "actual": actual,
            "validation": {
                **total_validation,
                "status": status_validation,
            },
            "baseline_restored": False,
        }


class SerialMotionAdapter:
    def __init__(
        self,
        serial_motion: Any,
        *,
        spd: float,
        acc: float,
        timeout_seconds: float = DEFAULT_MOTION_TIMEOUT_SECONDS,
        poll_seconds: float = 0.2,
        negative_e_compensation_deg: float = DEFAULT_NEGATIVE_E_COMPENSATION_DEG,
    ):
        self.serial_motion = serial_motion
        self.spd = float(spd)
        self.acc = float(acc)
        self.timeout_seconds = float(timeout_seconds)
        self.poll_seconds = float(poll_seconds)
        self.negative_e_compensation_deg = float(
            negative_e_compensation_deg
        )

    def _normalize(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        for key in ("x", "y", "z", "move"):
            if key not in raw:
                raise SafetyStop(f"arm status is missing {key}")
        pose = {
            joint: float(value)
            for joint, value in self.serial_motion.arm.status_to_command_degrees(
                raw
            ).items()
        }
        pose["h"] = float(
            self.serial_motion.arm.status_to_gripper_degrees(raw)
        )
        return {
            "pose_deg": pose,
            "endpoint_xyz_mm": [
                _finite_float(raw["x"], "x"),
                _finite_float(raw["y"], "y"),
                _finite_float(raw["z"], "z"),
            ],
            "move": int(raw["move"]),
            "feedback_stable": int(raw["move"]) == 0,
            "raw": dict(raw),
        }

    def read_stable_status(self) -> dict[str, Any]:
        initial = self._normalize(self.serial_motion._query_status())
        return self._wait_for_stable_feedback(
            initial,
            require_movement=False,
        )

    @staticmethod
    def _sample_change(
        before: Mapping[str, Any],
        after: Mapping[str, Any],
    ) -> tuple[float, float]:
        before_pose = _pose(before)
        after_pose = _pose(after)
        joint_change = max(
            abs(float(after_pose[joint]) - float(before_pose[joint]))
            for joint in ("s", "e")
        )
        endpoint_change = endpoint_displacement(before, after)[
            "endpoint_step_mm"
        ]
        return joint_change, endpoint_change

    @staticmethod
    def _has_monotonic_drift(values: list[float]) -> bool:
        deltas = [
            float(current) - float(previous)
            for previous, current in zip(values, values[1:])
        ]
        net_change = float(values[-1]) - float(values[0])
        return (
            net_change > 0.0 and all(delta >= 0.0 for delta in deltas)
        ) or (
            net_change < 0.0 and all(delta <= 0.0 for delta in deltas)
        )

    @classmethod
    def _stable_window_is_valid(
        cls,
        samples: list[Mapping[str, Any]],
    ) -> bool:
        joint_values = {
            joint: [float(_pose(sample)[joint]) for sample in samples]
            for joint in ("s", "e")
        }
        largest_joint_range = max(
            max(values) - min(values)
            for values in joint_values.values()
        )
        endpoints = [_endpoint_xyz(sample) for sample in samples]
        largest_endpoint_range = max(
            math.dist(first, second)
            for index, first in enumerate(endpoints)
            for second in endpoints[index + 1 :]
        )
        scalar_series = [
            *joint_values.values(),
            *[
                [endpoint[axis] for endpoint in endpoints]
                for axis in range(3)
            ],
        ]
        return (
            largest_joint_range <= STABLE_JOINT_RANGE_DEG
            and largest_endpoint_range <= STABLE_ENDPOINT_RANGE_MM
            and not any(
                cls._has_monotonic_drift(values)
                for values in scalar_series
            )
        )

    def _wait_for_stable_feedback(
        self,
        baseline: Mapping[str, Any],
        *,
        require_movement: bool,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout_seconds
        stable_window: list[dict[str, Any]] = (
            [] if require_movement else [_plain_status(baseline)]
        )
        movement_observed = not require_movement
        while time.monotonic() < deadline:
            time.sleep(self.poll_seconds)
            current = self._normalize(
                self.serial_motion._query_fast_status()
            )
            if not movement_observed:
                joint_from_start, endpoint_from_start = self._sample_change(
                    baseline,
                    current,
                )
                if (
                    joint_from_start >= MOVEMENT_JOINT_THRESHOLD_DEG
                    or endpoint_from_start
                    >= MOVEMENT_ENDPOINT_THRESHOLD_MM
                ):
                    movement_observed = True
                    stable_window = [current]
                continue
            stable_window.append(current)
            if len(stable_window) > STABLE_WINDOW_SAMPLES:
                stable_window.pop(0)
            if (
                len(stable_window) == STABLE_WINDOW_SAMPLES
                and self._stable_window_is_valid(stable_window)
            ):
                current["feedback_stable"] = True
                return current
        if require_movement:
            raise SafetyStop(
                "joint command did not produce stable feedback before timeout"
            )
        raise SafetyStop(
            "arm feedback did not become stable before timeout"
        )

    def _wait_after_command(
        self,
        baseline: Mapping[str, Any],
    ) -> dict[str, Any]:
        return self._wait_for_stable_feedback(
            baseline,
            require_movement=True,
        )

    def move_joint_targets(
        self,
        targets: Mapping[str, float],
        *,
        spd: float,
        acc: float,
    ) -> dict[str, Any]:
        if set(targets) - {"s", "e"}:
            raise ValueError("real local Cartesian movement may only target s/e")
        baseline = self.read_stable_status()
        baseline_pose = _pose(baseline)
        if all(
            abs(float(targets[joint]) - float(baseline_pose[joint])) <= 0.1
            for joint in targets
        ):
            return baseline
        command_targets, _ = command_targets_for_expected(
            baseline_pose,
            targets,
            negative_e_compensation_deg=self.negative_e_compensation_deg,
        )
        command = build_joint_command(
            baseline_pose,
            command_targets,
            spd=spd,
            acc=acc,
        )
        self.serial_motion._send(command)
        self.serial_motion._remember_command_pose(command)
        return self._wait_after_command(baseline)

    def command_log(self) -> list[dict[str, Any]]:
        return [
            dict(event["command"])
            for event in getattr(self.serial_motion, "event_log", [])
            if isinstance(event, Mapping)
            and event.get("type") == "command"
            and isinstance(event.get("command"), Mapping)
        ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate a local s/e Jacobian and move the open gripper "
            "approximately forward without T=104"
        )
    )
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument(
        "--distance-mm",
        type=float,
        default=DEFAULT_DISTANCE_MM,
    )
    parser.add_argument(
        "--total-distance-mm",
        type=float,
        default=None,
        help=(
            "execute a guarded multi-segment move up to 50 mm; "
            "50 uses 20+20+10 mm"
        ),
    )
    parser.add_argument(
        "--probe-delta-deg",
        type=float,
        default=DEFAULT_PROBE_DELTA_DEG,
    )
    parser.add_argument("--spd", type=float, default=DEFAULT_SPEED)
    parser.add_argument("--acc", type=float, default=DEFAULT_ACCELERATION)
    parser.add_argument(
        "--motion-timeout",
        type=float,
        default=DEFAULT_MOTION_TIMEOUT_SECONDS,
    )
    parser.add_argument("--output-dir", default="")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--execute-probes",
        action="store_true",
        help="execute s/e probes and restore, but do not move forward",
    )
    modes.add_argument(
        "--execute",
        action="store_true",
        help="execute probes and one guarded forward segment",
    )
    return parser


def default_output_dir() -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return DEFAULT_OUTPUT_ROOT / f"local-cartesian-jog-{stamp}"


def render_summary(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Local Cartesian Jog Summary",
        "",
        f"- ok: {payload.get('ok', False)}",
        f"- stage: {payload.get('stage', '')}",
        f"- requested_distance_mm: {payload.get('requested_distance_mm', '')}",
        f"- reason: {payload.get('reason', '')}",
        f"- baseline_restored: {payload.get('baseline_restored', False)}",
    ]
    actual = payload.get("actual")
    if isinstance(actual, Mapping):
        lines.extend(
            [
                f"- actual_delta_r_mm: {actual.get('delta_r_mm', '')}",
                f"- actual_delta_z_mm: {actual.get('delta_z_mm', '')}",
                f"- actual_endpoint_step_mm: {actual.get('endpoint_step_mm', '')}",
            ]
        )
    solution = payload.get("solution_delta_deg")
    if isinstance(solution, Mapping):
        lines.extend(
            [
                f"- solution_delta_s_deg: {solution.get('s', '')}",
                f"- solution_delta_e_deg: {solution.get('e', '')}",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def write_outputs(output_dir: Path, payload: Mapping[str, Any]) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(
        render_summary(payload),
        encoding="utf-8",
    )


def _load_local_module(name: str) -> Any:
    module_path = MODULE_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load local module: {name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _motion_command_log(motion: Optional[Any]) -> list[dict[str, Any]]:
    if motion is None:
        return []
    command_log = getattr(motion, "command_log", None)
    if callable(command_log):
        return [dict(command) for command in command_log()]
    actions = getattr(motion, "actions", None)
    if isinstance(actions, list):
        return [
            dict(action[1])
            for action in actions
            if isinstance(action, tuple)
            and len(action) >= 2
            and action[0] == "move"
            and isinstance(action[1], Mapping)
        ]
    return []


def run_cli(
    args: argparse.Namespace,
    *,
    motion: Optional[Any] = None,
) -> tuple[int, dict[str, Any], Path]:
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else default_output_dir()
    )
    try:
        if args.total_distance_mm is None:
            validate_distance_mm(args.distance_mm)
        else:
            plan_segment_distances(args.total_distance_mm)
            if args.execute_probes:
                raise ValueError(
                    "--execute-probes cannot be combined with "
                    "--total-distance-mm"
                )
        validate_runtime_parameters(
            probe_delta_deg=args.probe_delta_deg,
            spd=args.spd,
            acc=args.acc,
            timeout_seconds=args.motion_timeout,
        )
        if motion is None and (args.execute or args.execute_probes):
            arm_task = _load_local_module("arm_task")
            serial_motion = arm_task.ArmTestSerialMotion(
                port=args.port,
                baudrate=getattr(arm_task, "DEFAULT_BAUD", 115200),
                timeout=getattr(arm_task, "DEFAULT_TIMEOUT", 2.0),
            )
            motion = SerialMotionAdapter(
                serial_motion,
                spd=args.spd,
                acc=args.acc,
                timeout_seconds=args.motion_timeout,
            )
        runner = LocalCartesianJogRunner(
            motion,
            probe_delta_deg=args.probe_delta_deg,
            spd=args.spd,
            acc=args.acc,
        )
        if args.total_distance_mm is None:
            payload = runner.run(
                distance_mm=args.distance_mm,
                execute=args.execute,
                execute_probes=args.execute_probes,
            )
        else:
            sequence = LocalCartesianSequenceRunner(motion, runner)
            payload = sequence.run(
                total_distance_mm=args.total_distance_mm,
                execute=args.execute,
            )
    except Exception as exc:
        payload = {
            "ok": False,
            "stage": "SETUP",
            "reason": str(exc),
            "requested_distance_mm": args.distance_mm,
            "requested_total_distance_mm": args.total_distance_mm,
            "execute": bool(args.execute),
            "execute_probes": bool(args.execute_probes),
            "baseline_restored": False,
            "samples": {},
        }
    payload["commands_sent"] = _motion_command_log(motion)
    payload["port"] = args.port
    payload["output_dir"] = str(output_dir)
    write_outputs(output_dir, payload)
    return (0 if payload.get("ok") else 1), payload, output_dir


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    exit_code, payload, _ = run_cli(args)
    print(json.dumps(payload, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Protocol


class PickupTransferMotion(Protocol):
    limits: object

    def move(self, vx: float, vy: float, wz: float) -> None: ...

    def stop(self) -> None: ...

    def strafe_distance_pose_hold(
        self,
        distance_m: float,
        speed_mps: Optional[float] = None,
        **kwargs: float,
    ) -> None: ...


class PickupTransferStateReader(Protocol):
    state: object

    def safety_error(
        self,
        *,
        require_ultrasound: bool = False,
        require_fresh: Optional[bool] = None,
    ) -> Optional[str]: ...


@dataclass(frozen=True)
class PickupRetreatConfig:
    retreat_target_front_m: float = 0.80
    retreat_stop_threshold_m: float = 0.77
    retreat_max_front_m: float = 0.90
    retreat_speed_mps: float = 0.06
    retreat_timeout_s: float = 12.0
    retreat_max_odom_m: float = 0.55
    retreat_stuck_front_fallback_enabled: bool = True
    retreat_stuck_front_value_m: float = 0.28
    retreat_stuck_front_tolerance_m: float = 0.01
    retreat_stuck_front_min_samples: int = 5
    retreat_odom_fallback_target_m: float = 0.44
    retreat_lateral_hold_kp_s: float = 1.0
    retreat_max_vy_correction_mps: float = 0.04
    retreat_lateral_deadband_m: float = 0.003
    retreat_max_lateral_drift_m: float = 0.10
    yaw_hold_kp_s: float = 1.2
    max_wz_correction_rad_s: float = 0.12
    yaw_deadband_deg: float = 0.30
    max_yaw_drift_deg: float = 5.0
    lane_strafe_speed_mps: float = 0.08
    lane_forward_hold_kp_s: float = 1.0
    lane_max_vx_correction_mps: float = 0.04
    lane_forward_deadband_m: float = 0.003
    lane_max_forward_drift_m: float = 0.15
    max_recorded_lane_strafe_m: float = 1.05

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "PickupRetreatConfig":
        defaults = cls()
        kwargs = {}
        for name in cls.__dataclass_fields__:
            default = getattr(defaults, name)
            value = values.get(name, default)
            if isinstance(default, bool):
                kwargs[name] = bool(value)
            elif isinstance(default, int):
                kwargs[name] = int(value)
            else:
                kwargs[name] = float(value)
        return cls(**kwargs)


@dataclass(frozen=True)
class PickupRetreatResult:
    ok: bool
    reason: str
    start_front_m: Optional[float]
    final_front_m: Optional[float]
    odom_retreat_m: float
    elapsed_s: float
    motion_command_count: int


@dataclass(frozen=True)
class LaneMovementResult:
    ok: bool
    reason: str
    requested_distance_m: float
    measured_distance_m: float
    forward_drift_m: float
    yaw_drift_deg: float
    motion_command_count: int


def normalize_yaw(yaw: float) -> float:
    return (float(yaw) + math.pi) % (2.0 * math.pi) - math.pi


def body_frame_delta(
    reference_pose: tuple[float, float, float],
    current_pose: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Return forward, leftward and yaw deltas in the reference body frame."""
    reference_x, reference_y, reference_yaw = reference_pose
    current_x, current_y, current_yaw = current_pose
    delta_x = current_x - reference_x
    delta_y = current_y - reference_y
    forward = math.cos(reference_yaw) * delta_x + math.sin(reference_yaw) * delta_y
    lateral = -math.sin(reference_yaw) * delta_x + math.cos(reference_yaw) * delta_y
    return forward, lateral, normalize_yaw(current_yaw - reference_yaw)


class PickupTransferController:
    def __init__(
        self,
        motion: PickupTransferMotion,
        state_reader: PickupTransferStateReader,
        config: Mapping[str, object],
        *,
        dry_run: bool = False,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.motion = motion
        self.state_reader = state_reader
        self.config = PickupRetreatConfig.from_mapping(config)
        self.dry_run = bool(dry_run)
        self.clock = clock
        self.sleep = sleep

    def retreat_to_front_distance(
        self,
        *,
        initial_odom_retreat_m: float = 0.0,
    ) -> PickupRetreatResult:
        cfg = self.config
        started_at = self.clock()
        start_front: Optional[float] = None
        final_front: Optional[float] = None
        odom_retreat = float(initial_odom_retreat_m)
        command_count = 0
        stuck_front_samples = 0
        stuck_front_reference_m: Optional[float] = None
        reference_pose: Optional[tuple[float, float, float]] = None

        def result(ok: bool, reason: str) -> PickupRetreatResult:
            return PickupRetreatResult(
                ok=ok,
                reason=reason,
                start_front_m=start_front,
                final_front_m=final_front,
                odom_retreat_m=odom_retreat,
                elapsed_s=max(0.0, self.clock() - started_at),
                motion_command_count=command_count,
            )

        if (
            not math.isfinite(odom_retreat)
            or odom_retreat < 0.0
            or odom_retreat > cfg.retreat_max_odom_m
        ):
            return result(False, "invalid_initial_retreat_odometry")
        prior_retreat_m = odom_retreat

        if self.dry_run:
            return result(True, "dry_run")

        period = 1.0 / max(1.0, float(getattr(self.motion.limits, "command_hz", 20.0)))
        try:
            while True:
                error = self.state_reader.safety_error(
                    require_ultrasound=True,
                    require_fresh=True,
                )
                if error:
                    return result(False, f"state_rejected:{error}")
                state = self.state_reader.state
                try:
                    distance_m = float(getattr(state, "front_ultrasound_m"))
                    current_pose = (
                        float(getattr(state, "x")),
                        float(getattr(state, "y")),
                        float(getattr(state, "yaw")),
                    )
                except (TypeError, ValueError, AttributeError):
                    return result(False, "invalid_front_or_odometry_sample")
                if not math.isfinite(distance_m):
                    return result(False, "invalid_front_ultrasound_sample")
                if not all(math.isfinite(value) for value in current_pose):
                    return result(False, "invalid_odometry_sample")
                if reference_pose is None:
                    reference_pose = current_pose
                    start_front = distance_m
                final_front = distance_m
                if stuck_front_reference_m is None:
                    stuck_front_reference_m = distance_m
                    stuck_front_samples = 1
                elif (
                    abs(distance_m - stuck_front_reference_m)
                    <= cfg.retreat_stuck_front_tolerance_m
                ):
                    stuck_front_samples += 1
                else:
                    stuck_front_reference_m = distance_m
                    stuck_front_samples = 1
                forward, lateral, yaw_delta = body_frame_delta(reference_pose, current_pose)
                odom_retreat = prior_retreat_m + max(0.0, -forward)

                if distance_m >= cfg.retreat_stop_threshold_m:
                    return result(
                        True,
                        (
                            "target_reached_after_overshoot"
                            if distance_m > cfg.retreat_max_front_m
                            else "target_reached"
                        ),
                    )
                if odom_retreat >= cfg.retreat_max_odom_m:
                    return result(False, "retreat_odometry_limit")
                if (
                    cfg.retreat_stuck_front_fallback_enabled
                    and stuck_front_samples >= cfg.retreat_stuck_front_min_samples
                    and odom_retreat >= cfg.retreat_odom_fallback_target_m
                ):
                    return result(True, "target_reached_odom_fallback")
                if abs(lateral) > cfg.retreat_max_lateral_drift_m:
                    return result(False, "retreat_lateral_drift_limit")
                if abs(math.degrees(yaw_delta)) > cfg.max_yaw_drift_deg:
                    return result(False, "retreat_yaw_drift_limit")
                if self.clock() - started_at >= cfg.retreat_timeout_s:
                    return result(False, "retreat_timeout")

                vy = (
                    0.0
                    if abs(lateral) <= cfg.retreat_lateral_deadband_m
                    else -cfg.retreat_lateral_hold_kp_s * lateral
                )
                wz = (
                    0.0
                    if abs(math.degrees(yaw_delta)) <= cfg.yaw_deadband_deg
                    else -cfg.yaw_hold_kp_s * yaw_delta
                )
                vy = max(
                    -cfg.retreat_max_vy_correction_mps,
                    min(cfg.retreat_max_vy_correction_mps, vy),
                )
                wz = max(
                    -cfg.max_wz_correction_rad_s,
                    min(cfg.max_wz_correction_rad_s, wz),
                )
                self.motion.move(-abs(cfg.retreat_speed_mps), vy, wz)
                command_count += 1
                self.sleep(period)
        except Exception as exc:
            return result(False, f"retreat_exception:{type(exc).__name__}:{exc}")
        finally:
            self.motion.stop()

    def move_lane(self, distance_m: float) -> LaneMovementResult:
        cfg = self.config
        requested = float(distance_m)
        if not math.isfinite(requested):
            return LaneMovementResult(False, "invalid_lane_distance", requested, 0.0, 0.0, 0.0, 0)
        if abs(requested) > cfg.max_recorded_lane_strafe_m + 1e-9:
            return LaneMovementResult(False, "lane_distance_exceeds_limit", requested, 0.0, 0.0, 0.0, 0)
        if abs(requested) <= 1e-9:
            self.motion.stop()
            return LaneMovementResult(True, "center_lane", requested, 0.0, 0.0, 0.0, 0)

        reference_pose: Optional[tuple[float, float, float]] = None
        if not self.dry_run:
            error = self.state_reader.safety_error(require_fresh=True)
            if error:
                return LaneMovementResult(False, f"state_rejected:{error}", requested, 0.0, 0.0, 0.0, 0)
            reference_pose = self._state_pose()
            if reference_pose is None:
                return LaneMovementResult(False, "invalid_odometry_sample", requested, 0.0, 0.0, 0.0, 0)

        try:
            self.motion.strafe_distance_pose_hold(
                requested,
                speed_mps=cfg.lane_strafe_speed_mps,
                forward_hold_kp_s=cfg.lane_forward_hold_kp_s,
                max_vx_correction_mps=cfg.lane_max_vx_correction_mps,
                forward_deadband_m=cfg.lane_forward_deadband_m,
                max_forward_drift_m=cfg.lane_max_forward_drift_m,
                yaw_hold_kp_s=cfg.yaw_hold_kp_s,
                max_wz_correction_rad_s=cfg.max_wz_correction_rad_s,
                yaw_deadband_deg=cfg.yaw_deadband_deg,
                max_yaw_drift_deg=cfg.max_yaw_drift_deg,
            )
        except Exception as exc:
            return LaneMovementResult(
                False,
                f"lane_motion_failed:{type(exc).__name__}:{exc}",
                requested,
                0.0,
                0.0,
                0.0,
                1,
            )

        if self.dry_run:
            return LaneMovementResult(True, "dry_run", requested, requested, 0.0, 0.0, 1)
        error = self.state_reader.safety_error(require_fresh=True)
        if error:
            return LaneMovementResult(False, f"state_rejected:{error}", requested, 0.0, 0.0, 0.0, 1)
        current_pose = self._state_pose()
        if reference_pose is None or current_pose is None:
            return LaneMovementResult(False, "invalid_odometry_sample", requested, 0.0, 0.0, 0.0, 1)
        forward, lateral, yaw_delta = body_frame_delta(reference_pose, current_pose)
        if abs(lateral) > cfg.max_recorded_lane_strafe_m + 1e-9:
            return LaneMovementResult(
                False,
                "measured_lane_distance_exceeds_limit",
                requested,
                lateral,
                forward,
                math.degrees(yaw_delta),
                1,
            )
        return LaneMovementResult(
            True,
            "completed",
            requested,
            lateral,
            forward,
            math.degrees(yaw_delta),
            1,
        )

    def _state_pose(self) -> Optional[tuple[float, float, float]]:
        state = self.state_reader.state
        try:
            pose = (
                float(getattr(state, "x")),
                float(getattr(state, "y")),
                float(getattr(state, "yaw")),
            )
        except (TypeError, ValueError, AttributeError):
            return None
        return pose if all(math.isfinite(value) for value in pose) else None

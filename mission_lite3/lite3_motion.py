from __future__ import annotations

import math
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple


HEARTBEAT = 0x21040001
STAND_OR_LIE = 0x21010202
SOFT_ESTOP = 0x21020C0E
MOVE_MODE = 0x21010D06
AUTONOMOUS_MODE = 0x21010C03
MANUAL_MODE = 0x21010C02
LOW_SPEED_GAIT = 0x21010300
AXIS_FORWARD = 0x21010130
AXIS_STRAFE = 0x21010131
AXIS_YAW = 0x21010135
VEL_FORWARD = 0x0140
VEL_YAW = 0x0141
VEL_STRAFE = 0x0145
VOICE_COMMAND = 0x21010C0A


def pack_simple_command(code: int, value: int = 0, msg_type: int = 0) -> bytes:
    """Pack the Lite3 simple UDP command used by the official examples."""
    return struct.pack("<3i", int(code), int(value), int(msg_type))


def pack_velocity_command(code: int, value: float) -> bytes:
    payload = struct.pack("<d", float(value))
    return struct.pack("<3I", int(code), len(payload), 1) + payload


@dataclass
class MotionLimits:
    max_vx: float
    max_vy: float
    max_wz: float
    command_hz: float


class MotionBackend:
    name = "base"

    def start(self) -> None:
        pass

    def close(self) -> None:
        pass

    def send_simple(self, code: int, value: int = 0, msg_type: int = 0) -> None:
        raise NotImplementedError

    def send_velocity(self, vx: float, vy: float, wz: float) -> None:
        raise NotImplementedError


class DryRunBackend(MotionBackend):
    name = "dry-run"

    def __init__(self) -> None:
        self._last_velocity: Optional[Tuple[float, float, float]] = None

    def send_simple(self, code: int, value: int = 0, msg_type: int = 0) -> None:
        print(f"[dry-run] simple code=0x{code:08X} value={value} type={msg_type}")

    def send_velocity(self, vx: float, vy: float, wz: float) -> None:
        velocity = (round(vx, 3), round(vy, 3), round(wz, 3))
        if velocity == self._last_velocity:
            return
        self._last_velocity = velocity
        print(f"[dry-run] velocity vx={vx:.3f} vy={vy:.3f} wz={wz:.3f}")


class UdpLite3Backend(MotionBackend):
    name = "udp"

    def __init__(self, host: str, port: int, heartbeat_hz: float, axis_fallback: bool = False, axis_scale: int = 30000):
        self.address = (host, port)
        self.heartbeat_hz = heartbeat_hz
        self.axis_fallback = axis_fallback
        self.axis_scale = axis_scale
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.sock.close()

    def _heartbeat_loop(self) -> None:
        period = 1.0 / max(self.heartbeat_hz, 2.0)
        while not self._stop.is_set():
            self.send_simple(HEARTBEAT)
            time.sleep(period)

    def send_simple(self, code: int, value: int = 0, msg_type: int = 0) -> None:
        self.sock.sendto(pack_simple_command(code, value, msg_type), self.address)

    def _send_complex_double(self, code: int, value: float) -> None:
        self.sock.sendto(pack_velocity_command(code, value), self.address)

    def send_velocity(self, vx: float, vy: float, wz: float) -> None:
        if self.axis_fallback:
            self.send_simple(AXIS_FORWARD, int(vx * self.axis_scale), 0)
            self.send_simple(AXIS_STRAFE, int(-vy * self.axis_scale), 0)
            self.send_simple(AXIS_YAW, int(-wz * self.axis_scale), 0)
            return
        self._send_complex_double(VEL_FORWARD, vx)
        self._send_complex_double(VEL_STRAFE, vy)
        self._send_complex_double(VEL_YAW, wz)


class Ros2CmdVelBackend(MotionBackend):
    name = "ros2"

    def __init__(self, topic: str):
        self.topic = topic
        self.rclpy = None
        self.node = None
        self.publisher = None
        self.twist_cls = None
        self._owns_rclpy = False

    @classmethod
    def available(cls) -> bool:
        try:
            import rclpy  # noqa: F401
            from geometry_msgs.msg import Twist  # noqa: F401

            return True
        except Exception:
            return False

    def start(self) -> None:
        import rclpy
        from geometry_msgs.msg import Twist

        self.rclpy = rclpy
        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init(args=None)
        self.node = rclpy.create_node("mission_lite3_motion")
        self.publisher = self.node.create_publisher(Twist, self.topic, 10)
        self.twist_cls = Twist

    def close(self) -> None:
        if self.node is not None:
            self.node.destroy_node()
        if self.rclpy is not None and self._owns_rclpy and self.rclpy.ok():
            self.rclpy.shutdown()

    def send_simple(self, code: int, value: int = 0, msg_type: int = 0) -> None:
        print(f"[ros2] simple command not supported here: code=0x{code:08X} value={value}")

    def send_velocity(self, vx: float, vy: float, wz: float) -> None:
        if self.publisher is None or self.twist_cls is None:
            raise RuntimeError("ROS2 backend is not started")
        msg = self.twist_cls()
        msg.linear.x = float(vx)
        msg.linear.y = float(vy)
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = float(wz)
        self.publisher.publish(msg)


class Ros2WithUdpBackend(MotionBackend):
    name = "ros2+udp"

    def __init__(self, ros2_backend: Ros2CmdVelBackend, udp_backend: UdpLite3Backend):
        self.ros2_backend = ros2_backend
        self.udp_backend = udp_backend

    def start(self) -> None:
        self.udp_backend.start()
        self.ros2_backend.start()

    def close(self) -> None:
        self.ros2_backend.close()
        self.udp_backend.close()

    def send_simple(self, code: int, value: int = 0, msg_type: int = 0) -> None:
        self.udp_backend.send_simple(code, value, msg_type)

    def send_velocity(self, vx: float, vy: float, wz: float) -> None:
        self.ros2_backend.send_velocity(vx, vy, wz)


class Lite3MotionController:
    def __init__(self, config: dict, dry_run: bool = False, udp_fallback: bool = False, axis_fallback: bool = False):
        self.config = config
        self.dry_run = dry_run
        motion_cfg = config["motion"]
        self.limits = MotionLimits(
            max_vx=float(motion_cfg["max_vx"]),
            max_vy=float(motion_cfg["max_vy"]),
            max_wz=float(motion_cfg["max_wz"]),
            command_hz=float(motion_cfg["command_hz"]),
        )
        navigation_cfg = config.get("navigation", {})
        self.distance_tolerance_m = float(navigation_cfg.get("distance_tolerance_m", 0.03))
        self.yaw_tolerance_rad = float(navigation_cfg.get("yaw_tolerance_rad", 0.04))
        self.action_timeout_scale = float(navigation_cfg.get("action_timeout_scale", 2.0))
        self.minimum_action_timeout_s = float(navigation_cfg.get("minimum_action_timeout_s", 2.0))
        self.translation_path_hold_enabled = bool(
            navigation_cfg.get("translation_path_hold_enabled", True)
        )
        self.translation_cross_track_kp_s = float(
            navigation_cfg.get("translation_cross_track_kp_s", 1.0)
        )
        self.translation_max_cross_track_correction_mps = float(
            navigation_cfg.get("translation_max_cross_track_correction_mps", 0.04)
        )
        self.translation_cross_track_deadband_m = float(
            navigation_cfg.get("translation_cross_track_deadband_m", 0.003)
        )
        self.translation_max_cross_track_drift_m = float(
            navigation_cfg.get("translation_max_cross_track_drift_m", 0.15)
        )
        self.translation_yaw_hold_kp_s = float(
            navigation_cfg.get("translation_yaw_hold_kp_s", 1.2)
        )
        self.translation_max_wz_correction_rad_s = float(
            navigation_cfg.get("translation_max_wz_correction_rad_s", 0.12)
        )
        self.translation_yaw_deadband_deg = float(
            navigation_cfg.get("translation_yaw_deadband_deg", 0.30)
        )
        self.translation_max_yaw_drift_deg = float(
            navigation_cfg.get("translation_max_yaw_drift_deg", 5.0)
        )
        self._guard: Optional[Callable[[float, float, float], None]] = None
        self._pose_provider: Optional[Callable[[], Tuple[float, float, float]]] = None
        self._feedback_required = False
        network = config["network"]
        if dry_run:
            self.backend: MotionBackend = DryRunBackend()
        elif not udp_fallback and config["ros2"].get("enabled", True) and Ros2CmdVelBackend.available():
            ros2_backend = Ros2CmdVelBackend(config["ros2"]["cmd_vel_topic"])
            udp_backend = UdpLite3Backend(
                network["motion_ip"],
                int(network["motion_port"]),
                float(motion_cfg["heartbeat_hz"]),
                axis_fallback=False,
                axis_scale=int(motion_cfg["axis_full_scale"]),
            )
            self.backend = Ros2WithUdpBackend(ros2_backend, udp_backend)
        else:
            self.backend = UdpLite3Backend(
                network["motion_ip"],
                int(network["motion_port"]),
                float(motion_cfg["heartbeat_hz"]),
                axis_fallback=axis_fallback,
                axis_scale=int(motion_cfg["axis_full_scale"]),
            )

    @property
    def backend_name(self) -> str:
        return self.backend.name

    def start(self) -> None:
        self.backend.start()

    def configure_safety(
        self,
        guard: Callable[[float, float, float], None],
        pose_provider: Callable[[], Tuple[float, float, float]],
        *,
        feedback_required: bool,
    ) -> None:
        self._guard = guard
        self._pose_provider = pose_provider
        self._feedback_required = bool(feedback_required)

    def close(self) -> None:
        try:
            self.stop()
        except Exception as exc:
            print(f"[motion] failed to send final stop: {exc}")
        self.backend.close()

    def stand_up(self) -> None:
        self.backend.send_simple(STAND_OR_LIE)
        self.prepare_walk()

    def prepare_walk(self) -> None:
        self.backend.send_simple(MOVE_MODE)
        self.backend.send_simple(LOW_SPEED_GAIT)

    def set_autonomous(self) -> None:
        self.backend.send_simple(AUTONOMOUS_MODE)

    def set_manual(self) -> None:
        self.backend.send_simple(MANUAL_MODE)

    def soft_estop(self) -> None:
        self.backend.send_simple(SOFT_ESTOP)

    def stop(self) -> None:
        self.backend.send_velocity(0.0, 0.0, 0.0)

    def voice_turn_left_90(self) -> None:
        self.backend.send_simple(VOICE_COMMAND, 13, 0)

    def move(self, vx: float, vy: float, wz: float) -> None:
        vx = max(-self.limits.max_vx, min(self.limits.max_vx, vx))
        vy = max(-self.limits.max_vy, min(self.limits.max_vy, vy))
        wz = max(-self.limits.max_wz, min(self.limits.max_wz, wz))
        self.backend.send_velocity(vx, vy, wz)

    def hold_velocity(self, vx: float, vy: float, wz: float, duration: float) -> None:
        if self.dry_run:
            self.move(vx, vy, wz)
            self.stop()
            return
        period = 1.0 / self.limits.command_hz
        deadline = time.monotonic() + max(0.0, duration)
        primary_error: Optional[BaseException] = None
        try:
            while time.monotonic() < deadline:
                self._run_guard(vx, vy, wz)
                self.move(vx, vy, wz)
                time.sleep(min(period, max(0.0, deadline - time.monotonic())))
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            self._final_stop(primary_error)

    def hold_velocity_feedback(
        self,
        velocity_provider: Callable[[], Tuple[float, float, float]],
        duration: float,
    ) -> None:
        """Hold a velocity while refreshing all three axes every control tick."""
        if self.dry_run:
            vx, vy, wz = velocity_provider()
            self.move(vx, vy, wz)
            self.stop()
            return
        period = 1.0 / self.limits.command_hz
        deadline = time.monotonic() + max(0.0, duration)
        primary_error: Optional[BaseException] = None
        try:
            while time.monotonic() < deadline:
                velocity = tuple(float(value) for value in velocity_provider())
                if len(velocity) != 3 or not all(math.isfinite(value) for value in velocity):
                    raise RuntimeError(
                        "feedback velocity provider must return three finite values"
                    )
                vx, vy, wz = velocity
                self._run_guard(vx, vy, wz)
                self.move(vx, vy, wz)
                time.sleep(min(period, max(0.0, deadline - time.monotonic())))
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            self._final_stop(primary_error)

    def go_distance(self, distance_m: float, speed_mps: Optional[float] = None) -> None:
        speed = abs(speed_mps or self.config["motion"]["cruise_vx"])
        if speed <= 0:
            return
        self._move_distance(math.copysign(speed, distance_m), 0.0, abs(distance_m), speed)

    def strafe_distance(self, distance_m: float, speed_mps: Optional[float] = None) -> None:
        speed = abs(speed_mps or self.config["motion"]["strafe_vy"])
        if speed <= 0:
            return
        self._move_distance(0.0, math.copysign(speed, distance_m), abs(distance_m), speed)

    def strafe_distance_pose_hold(
        self,
        distance_m: float,
        speed_mps: Optional[float] = None,
        *,
        completion_tolerance_m: Optional[float] = None,
        forward_hold_kp_s: float = 1.0,
        max_vx_correction_mps: float = 0.04,
        forward_deadband_m: float = 0.003,
        forward_velocity_provider: Optional[Callable[[], float]] = None,
        max_forward_drift_m: float = 0.15,
        yaw_hold_kp_s: float = 1.2,
        max_wz_correction_rad_s: float = 0.12,
        yaw_deadband_deg: float = 0.30,
        max_yaw_drift_deg: float = 5.0,
    ) -> None:
        speed = abs(speed_mps or self.config["motion"]["strafe_vy"])
        target_distance = abs(float(distance_m))
        completion_tolerance = (
            self.distance_tolerance_m
            if completion_tolerance_m is None
            else float(completion_tolerance_m)
        )
        if not math.isfinite(completion_tolerance) or completion_tolerance < 0.0:
            raise ValueError("strafe completion tolerance must be non-negative")
        if speed <= 0.0:
            return
        if self.dry_run or self._pose_provider is None:
            if self._feedback_required and not self.dry_run:
                raise RuntimeError("pose-held strafe requires a pose provider")
            self.strafe_distance(distance_m, speed_mps=speed)
            return
        if target_distance <= completion_tolerance:
            self.stop()
            return

        assert self._pose_provider is not None
        direction = math.copysign(1.0, distance_m)
        reference_x, reference_y, reference_yaw = self._pose_provider()
        period = 1.0 / self.limits.command_hz
        deadline = self._action_deadline(target_distance / speed)
        primary_error: Optional[BaseException] = None
        try:
            while True:
                current_x, current_y, current_yaw = self._pose_provider()
                delta_x = current_x - reference_x
                delta_y = current_y - reference_y
                forward_drift = (
                    math.cos(reference_yaw) * delta_x
                    + math.sin(reference_yaw) * delta_y
                )
                lateral_progress = direction * (
                    -math.sin(reference_yaw) * delta_x
                    + math.cos(reference_yaw) * delta_y
                )
                yaw_error = (
                    current_yaw - reference_yaw + math.pi
                ) % (2.0 * math.pi) - math.pi
                if lateral_progress >= max(
                    0.0,
                    target_distance - completion_tolerance,
                ):
                    return
                if abs(forward_drift) > max(0.0, float(max_forward_drift_m)):
                    raise RuntimeError(
                        "pose-held strafe exceeded forward drift limit: "
                        f"{forward_drift:.3f}m"
                    )
                if abs(math.degrees(yaw_error)) > max(
                    0.0,
                    float(max_yaw_drift_deg),
                ):
                    raise RuntimeError(
                        "pose-held strafe exceeded yaw drift limit: "
                        f"{math.degrees(yaw_error):.3f}deg"
                    )
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "pose-held strafe timed out: "
                        f"target={target_distance:.3f}m "
                        f"lateral={lateral_progress:.3f}m"
                    )

                if forward_velocity_provider is None:
                    vx = (
                        0.0
                        if abs(forward_drift)
                        <= max(0.0, float(forward_deadband_m))
                        else -float(forward_hold_kp_s) * forward_drift
                    )
                else:
                    vx = float(forward_velocity_provider())
                    if not math.isfinite(vx):
                        raise RuntimeError(
                            "forward velocity provider returned a non-finite value"
                        )
                wz = (
                    0.0
                    if abs(math.degrees(yaw_error))
                    <= max(0.0, float(yaw_deadband_deg))
                    else -float(yaw_hold_kp_s) * yaw_error
                )
                vx_limit = max(0.0, float(max_vx_correction_mps))
                wz_limit = max(0.0, float(max_wz_correction_rad_s))
                vx = max(-vx_limit, min(vx_limit, vx))
                wz = max(-wz_limit, min(wz_limit, wz))
                vy = direction * speed
                self._run_guard(vx, vy, wz)
                self.move(vx, vy, wz)
                time.sleep(period)
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            self._final_stop(primary_error)

    def turn_by(self, yaw_rad: float, wz: Optional[float] = None) -> None:
        speed = abs(wz or self.config["motion"]["turn_wz"])
        if speed <= 0:
            return
        if self.dry_run or self._pose_provider is None:
            if self._feedback_required and not self.dry_run:
                raise RuntimeError("closed-loop turn requires a pose provider")
            self.hold_velocity(0.0, 0.0, math.copysign(speed, yaw_rad), abs(yaw_rad) / speed)
            return
        self._turn_feedback(yaw_rad, speed)

    def _run_guard(self, vx: float, vy: float, wz: float) -> None:
        if self._guard is not None:
            self._guard(vx, vy, wz)

    def _final_stop(self, primary_error: Optional[BaseException]) -> None:
        try:
            self.stop()
        except Exception as stop_error:
            if primary_error is None:
                raise
            print(f"[motion] stop failed while handling {type(primary_error).__name__}: {stop_error}")

    def _action_deadline(self, expected_seconds: float) -> float:
        timeout = max(self.minimum_action_timeout_s, expected_seconds * self.action_timeout_scale)
        return time.monotonic() + timeout

    def _move_distance(self, vx: float, vy: float, distance_m: float, speed: float) -> None:
        if distance_m <= self.distance_tolerance_m:
            self.stop()
            return
        if self.dry_run or self._pose_provider is None:
            if self._feedback_required and not self.dry_run:
                raise RuntimeError("closed-loop translation requires a pose provider")
            self.hold_velocity(vx, vy, 0.0, distance_m / speed)
            return
        period = 1.0 / self.limits.command_hz
        self._run_guard(vx, vy, 0.0)
        start_x, start_y, reference_yaw = self._pose_provider()
        local_forward = vx / speed
        local_lateral = vy / speed
        cos_yaw = math.cos(reference_yaw)
        sin_yaw = math.sin(reference_yaw)
        target_world_x = cos_yaw * local_forward - sin_yaw * local_lateral
        target_world_y = sin_yaw * local_forward + cos_yaw * local_lateral
        cross_world_x = -target_world_y
        cross_world_y = target_world_x
        cross_local_forward = -local_lateral
        cross_local_lateral = local_forward
        deadline = self._action_deadline(distance_m / speed)
        primary_error: Optional[BaseException] = None
        try:
            while True:
                x, y, yaw = self._pose_provider()
                delta_x = x - start_x
                delta_y = y - start_y
                progress = delta_x * target_world_x + delta_y * target_world_y
                cross_track_error = (
                    delta_x * cross_world_x + delta_y * cross_world_y
                )
                yaw_error = (yaw - reference_yaw + math.pi) % (2.0 * math.pi) - math.pi
                if progress >= max(0.0, distance_m - self.distance_tolerance_m):
                    return
                if self.translation_path_hold_enabled:
                    if (
                        abs(cross_track_error)
                        > self.translation_max_cross_track_drift_m
                    ):
                        raise RuntimeError(
                            "translation exceeded cross-track drift limit: "
                            f"{cross_track_error:.3f}m"
                        )
                    if (
                        abs(math.degrees(yaw_error))
                        > self.translation_max_yaw_drift_deg
                    ):
                        raise RuntimeError(
                            "translation exceeded yaw drift limit: "
                            f"{math.degrees(yaw_error):.3f}deg"
                        )
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "translation timed out: "
                        f"target={distance_m:.3f}m progress={progress:.3f}m "
                        f"cross_track={cross_track_error:.3f}m"
                    )

                cross_correction = 0.0
                wz = 0.0
                if self.translation_path_hold_enabled:
                    if (
                        abs(cross_track_error)
                        > self.translation_cross_track_deadband_m
                    ):
                        cross_correction = (
                            -self.translation_cross_track_kp_s
                            * cross_track_error
                        )
                    correction_limit = (
                        self.translation_max_cross_track_correction_mps
                    )
                    cross_correction = max(
                        -correction_limit,
                        min(correction_limit, cross_correction),
                    )
                    if (
                        abs(math.degrees(yaw_error))
                        > self.translation_yaw_deadband_deg
                    ):
                        wz = -self.translation_yaw_hold_kp_s * yaw_error
                    wz_limit = self.translation_max_wz_correction_rad_s
                    wz = max(-wz_limit, min(wz_limit, wz))

                command_vx = vx + cross_correction * cross_local_forward
                command_vy = vy + cross_correction * cross_local_lateral
                self._run_guard(command_vx, command_vy, wz)
                self.move(command_vx, command_vy, wz)
                time.sleep(period)
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            self._final_stop(primary_error)

    def _turn_feedback(self, yaw_rad: float, speed: float) -> None:
        if abs(yaw_rad) <= self.yaw_tolerance_rad:
            self.stop()
            return
        assert self._pose_provider is not None
        wz = math.copysign(speed, yaw_rad)
        period = 1.0 / self.limits.command_hz
        self._run_guard(0.0, 0.0, wz)
        _, _, previous_yaw = self._pose_provider()
        accumulated = 0.0
        deadline = self._action_deadline(abs(yaw_rad) / speed)
        direction = math.copysign(1.0, yaw_rad)
        primary_error: Optional[BaseException] = None
        try:
            while True:
                self._run_guard(0.0, 0.0, wz)
                _, _, current_yaw = self._pose_provider()
                delta = (current_yaw - previous_yaw + math.pi) % (2 * math.pi) - math.pi
                accumulated += delta
                previous_yaw = current_yaw
                if accumulated * direction >= max(0.0, abs(yaw_rad) - self.yaw_tolerance_rad):
                    return
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"turn timed out: target={yaw_rad:.3f}rad turned={accumulated:.3f}rad"
                    )
                self.move(0.0, 0.0, wz)
                time.sleep(period)
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            self._final_stop(primary_error)

    def turn_to(self, target_yaw: float, current_yaw: Optional[float]) -> None:
        if current_yaw is None:
            self.turn_by(target_yaw)
            return
        diff = (target_yaw - current_yaw + math.pi) % (2 * math.pi) - math.pi
        self.turn_by(diff)

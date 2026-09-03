from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional


POLL_CALLBACK_BUDGET = 64


@dataclass
class RobotState:
    x: float = 0.0
    y: float = 0.0
    # yaw is the odometry-frame heading used with x/y for motion feedback.
    yaw: float = 0.0
    odom_yaw: float = 0.0
    imu_yaw: float = 0.0
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    front_ultrasound_m: Optional[float] = None
    rear_ultrasound_m: Optional[float] = None
    battery_fraction: Optional[float] = None
    healthy: bool = True
    odom_updated_at: Optional[float] = None
    imu_updated_at: Optional[float] = None
    ultrasound_updated_at: Optional[float] = None
    rear_ultrasound_updated_at: Optional[float] = None


class StateReader:
    def __init__(self, config: dict, dry_run: bool = False):
        self.config = config
        self.dry_run = dry_run
        self.state = RobotState()
        self.rclpy = None
        self.node = None
        self._started = False
        self._owns_rclpy = False
        self._front_ultrasound_history = deque(maxlen=512)

    @classmethod
    def ros2_available(cls) -> bool:
        try:
            import rclpy  # noqa: F401
            from nav_msgs.msg import Odometry  # noqa: F401
            from sensor_msgs.msg import Imu  # noqa: F401
            from std_msgs.msg import Float64  # noqa: F401

            return True
        except Exception:
            return False

    def start(self) -> None:
        if self.dry_run:
            now = time.monotonic()
            self.state.odom_updated_at = now
            self.state.imu_updated_at = now
            self.state.ultrasound_updated_at = now
            self._started = True
            return
        if not self.config["ros2"].get("enabled", True):
            raise RuntimeError("ROS2 state reader is disabled in real mode")
        if not self.ros2_available():
            raise RuntimeError("ROS2 state dependencies are unavailable in real mode")
        import rclpy
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import Imu
        from std_msgs.msg import Float64

        self.rclpy = rclpy
        self._owns_rclpy = not rclpy.ok()
        if self._owns_rclpy:
            rclpy.init(args=None)
        self.node = rclpy.create_node("mission_lite3_state_reader")
        self.node.create_subscription(Odometry, self.config["ros2"]["odom_topic"], self._on_odom, 10)
        self.node.create_subscription(Imu, self.config["ros2"]["imu_topic"], self._on_imu, 10)
        self.node.create_subscription(Float64, self.config["ros2"]["ultrasound_topic"], self._on_ultrasound, 10)
        rear_topic = self.config["ros2"].get("rear_ultrasound_topic")
        if rear_topic:
            self.node.create_subscription(
                Float64,
                str(rear_topic),
                self._on_rear_ultrasound,
                10,
            )
        self._started = True

    def close(self) -> None:
        if self.node is not None:
            self.node.destroy_node()
        if self.rclpy is not None and self._owns_rclpy and self.rclpy.ok():
            self.rclpy.shutdown()

    def poll(self) -> RobotState:
        if self.rclpy is not None and self.node is not None:
            # Camera inference can occupy the main thread for longer than the
            # state freshness window.  Drain the bounded subscription queues
            # before evaluating safety so a queued recent sample is not
            # rejected merely because an older callback was dispatched first.
            for _ in range(POLL_CALLBACK_BUDGET):
                self.rclpy.spin_once(self.node, timeout_sec=0.0)
        return self.state

    def _on_odom(self, msg) -> None:
        self.state.x = float(msg.pose.pose.position.x)
        self.state.y = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        odom_yaw = _yaw_from_quaternion(q.x, q.y, q.z, q.w)
        self.state.yaw = odom_yaw
        self.state.odom_yaw = odom_yaw
        self.state.odom_updated_at = time.monotonic()

    def _on_imu(self, msg) -> None:
        q = msg.orientation
        roll, pitch, yaw = _rpy_from_quaternion(q.x, q.y, q.z, q.w)
        self.state.roll_deg = math.degrees(roll)
        self.state.pitch_deg = math.degrees(pitch)
        self.state.imu_yaw = yaw
        self.state.imu_updated_at = time.monotonic()

    def _on_ultrasound(self, msg) -> None:
        value = float(msg.data)
        now = time.monotonic()
        self.state.front_ultrasound_m = value
        self.state.ultrasound_updated_at = now
        self._front_ultrasound_history.append((now, value))

    def _on_rear_ultrasound(self, msg) -> None:
        self.state.rear_ultrasound_m = float(msg.data)
        self.state.rear_ultrasound_updated_at = time.monotonic()

    def wait_until_ready(self, timeout_s: float, *, require_ultrasound: bool = True) -> None:
        if self.dry_run:
            return
        deadline = time.monotonic() + max(0.0, timeout_s)
        last_error = "state samples have not arrived"
        while time.monotonic() < deadline:
            if self.rclpy is not None and self.node is not None:
                self.rclpy.spin_once(self.node, timeout_sec=min(0.05, max(0.0, deadline - time.monotonic())))
            last_error = self.safety_error(require_ultrasound=require_ultrasound) or ""
            if not last_error:
                return
        raise TimeoutError(f"robot state not ready: {last_error}")

    def sample_age(self, updated_at: Optional[float]) -> float:
        if updated_at is None:
            return math.inf
        return max(0.0, time.monotonic() - updated_at)

    def filtered_front_ultrasound_m(self, window_s: float = 0.8) -> Optional[float]:
        """Return a short-window median without ever consulting the rear sensor."""
        cutoff = time.monotonic() - max(0.0, float(window_s))
        values = [
            value
            for sampled_at, value in self._front_ultrasound_history
            if sampled_at >= cutoff and math.isfinite(value)
        ]
        if not values:
            return self.state.front_ultrasound_m
        values.sort()
        middle = len(values) // 2
        if len(values) % 2:
            return float(values[middle])
        return float((values[middle - 1] + values[middle]) / 2.0)

    def safety_error(self, *, require_ultrasound: bool = False, require_fresh: Optional[bool] = None) -> Optional[str]:
        state = self.poll()
        safety = self.config["safety"]
        if not state.healthy:
            return "robot state is unhealthy"
        if abs(state.roll_deg) > float(safety["max_roll_deg"]):
            return f"roll exceeds limit: {state.roll_deg:.1f}deg"
        if abs(state.pitch_deg) > float(safety["max_pitch_deg"]):
            return f"pitch exceeds limit: {state.pitch_deg:.1f}deg"
        if state.battery_fraction is not None and state.battery_fraction < float(safety["low_battery_fraction"]):
            return f"battery below limit: {state.battery_fraction:.3f}"
        if require_fresh is None:
            require_fresh = bool(safety.get("require_fresh_state", True)) and not self.dry_run
        if require_fresh:
            max_age = float(safety.get("state_max_age_s", 0.75))
            for name, updated_at in (("odometry", state.odom_updated_at), ("imu", state.imu_updated_at)):
                age = self.sample_age(updated_at)
                if age > max_age:
                    return f"{name} sample is stale: age={age:.3f}s limit={max_age:.3f}s"
            if require_ultrasound:
                age = self.sample_age(state.ultrasound_updated_at)
                if age > max_age:
                    return f"ultrasound sample is stale: age={age:.3f}s limit={max_age:.3f}s"
                value = state.front_ultrasound_m
                if value is None or not math.isfinite(float(value)):
                    return "front ultrasound sample is invalid"
        return None

    def pose(self) -> tuple[float, float, float]:
        error = self.safety_error(require_ultrasound=False)
        if error:
            raise RuntimeError(error)
        return self.state.x, self.state.y, self.state.yaw

    def headings(self) -> tuple[float, float]:
        error = self.safety_error(require_ultrasound=False)
        if error:
            raise RuntimeError(error)
        return self.state.odom_yaw, self.state.imu_yaw

    def is_safe(self) -> bool:
        return self.safety_error() is None


def _yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    _, _, yaw = _rpy_from_quaternion(x, y, z, w)
    return yaw


def _rpy_from_quaternion(x: float, y: float, z: float, w: float) -> tuple[float, float, float]:
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw

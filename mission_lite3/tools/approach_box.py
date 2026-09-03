from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

from ..box_approach import ApproachConfig, BoxApproachController
from ..config_loader import load_config
from ..lite3_motion import Lite3MotionController


DEFAULT_SIMULATION_CM = "60,45,40,34,30,28,28,28"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Move Lite3 forward until the front ultrasound reaches the target distance"
    )
    parser.add_argument("--robot", action="store_true", help="Send real motion commands")
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="read live ultrasound samples without sending any motion command",
    )
    parser.add_argument("--yes", action="store_true", help="Confirm real robot movement")
    parser.add_argument("--target-cm", type=float, default=28.0)
    parser.add_argument("--topic", default="/us_publisher/front_distance")
    parser.add_argument("--simulate-cm", default=DEFAULT_SIMULATION_CM)
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument("--udp-fallback", action="store_true")
    parser.add_argument("--axis-fallback", action="store_true")
    return parser


def parse_simulated_distances(value: str) -> list[float]:
    distances = []
    for item in value.split(","):
        text = item.strip()
        if text:
            distances.append(float(text) / 100.0)
    if not distances:
        raise ValueError("simulation distance list is empty")
    return distances


class FrontUltrasoundReader:
    def __init__(self, topic: str):
        self.topic = topic
        self.rclpy = None
        self.node = None
        self._subscription = None
        self._distance_m: Optional[float] = None
        self._updated_at: Optional[float] = None
        self._sequence = 0

    def start(self) -> None:
        import rclpy
        from std_msgs.msg import Float64

        self.rclpy = rclpy
        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = rclpy.create_node("mission_lite3_front_ultrasound")
        self._subscription = self.node.create_subscription(
            Float64,
            self.topic,
            self._on_distance,
            10,
        )

    def close(self) -> None:
        if self.node is not None:
            self.node.destroy_node()
            self.node = None

    def wait_for_first_sample(self, timeout_s: float) -> tuple[float, float, int]:
        deadline = time.monotonic() + max(0.0, timeout_s)
        while self._distance_m is None and time.monotonic() < deadline:
            self._spin(
                timeout_sec=min(0.1, max(0.0, deadline - time.monotonic()))
            )
        if self._distance_m is None:
            raise TimeoutError(f"no front ultrasound data received from {self.topic}")
        return self.latest()

    def poll(self, timeout_sec: float = 0.0) -> tuple[float, float, int]:
        self._spin(timeout_sec)
        return self.latest()

    def _spin(self, timeout_sec: float) -> None:
        if self.rclpy is None or self.node is None:
            raise RuntimeError("front ultrasound reader is not started")
        self.rclpy.spin_once(self.node, timeout_sec=max(0.0, timeout_sec))
        for _ in range(9):
            before = self._updated_at
            self.rclpy.spin_once(self.node, timeout_sec=0.0)
            if self._updated_at == before:
                break

    def latest(self) -> tuple[float, float, int]:
        if self._distance_m is None or self._updated_at is None:
            raise RuntimeError("front ultrasound sample is not available")
        return (
            self._distance_m,
            time.monotonic() - self._updated_at,
            self._sequence,
        )

    def _on_distance(self, msg) -> None:
        self._distance_m = float(msg.data)
        self._updated_at = time.monotonic()
        self._sequence += 1


def run_simulation(controller: BoxApproachController, distances: list[float]) -> int:
    for index, distance_m in enumerate(distances, 1):
        controller.validate_runtime(sample_age_s=0.0, elapsed_s=0.0)
        decision = controller.decide(distance_m)
        print(
            f"[approach-box] sample={index} distance={distance_m * 100:.1f}cm "
            f"mode={decision.mode} vx={decision.vx:.3f} reached={decision.reached}"
        )
        if decision.reached:
            print("[approach-box] dry-run target confirmed")
            return 0
    print("[approach-box] dry-run sequence ended before target confirmation")
    return 2


def settle_with_sensor_poll(reader: FrontUltrasoundReader, duration_s: float) -> None:
    deadline = time.monotonic() + max(0.0, duration_s)
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        reader.poll(timeout_sec=min(0.05, max(0.0, remaining)))


def run_robot(args: argparse.Namespace) -> int:
    if not args.yes:
        raise RuntimeError("real robot control requires --yes")

    target_m = float(args.target_cm) / 100.0
    controller = BoxApproachController(ApproachConfig(target_distance_m=target_m))
    config = load_config(args.config_dir)
    motion = Lite3MotionController(
        config,
        dry_run=False,
        udp_fallback=args.udp_fallback,
        axis_fallback=args.axis_fallback,
    )
    reader = FrontUltrasoundReader(args.topic)
    command_period_s = 1.0 / max(1.0, motion.limits.command_hz)
    started = False
    autonomous_enabled = False
    start_time = time.monotonic()

    try:
        motion.start()
        started = True
        reader.start()
        motion.prepare_walk()
        time.sleep(5.0)
        motion.set_autonomous()
        autonomous_enabled = True
        distance_m, _, last_sequence = reader.wait_for_first_sample(
            timeout_s=3.0
        )
        print(
            f"[approach-box] backend={motion.backend_name} topic={args.topic} "
            f"initial_distance={distance_m * 100:.1f}cm target={args.target_cm:.1f}cm"
        )

        while True:
            distance_m, sample_age_s, sequence = reader.poll(timeout_sec=0.05)
            elapsed_s = time.monotonic() - start_time
            controller.validate_runtime(sample_age_s, elapsed_s)
            if sequence == last_sequence:
                continue
            last_sequence = sequence
            decision = controller.decide(distance_m)
            print(
                f"[approach-box] distance={distance_m * 100:.1f}cm "
                f"mode={decision.mode} vx={decision.vx:.3f}"
            )

            if decision.vx == 0.0:
                motion.stop()
                if not decision.reached:
                    settle_with_sensor_poll(reader, controller.config.settle_s)
            elif decision.mode == "continuous":
                motion.move(decision.vx, 0.0, 0.0)
                time.sleep(command_period_s)
            else:
                motion.hold_velocity(
                    decision.vx,
                    0.0,
                    0.0,
                    float(decision.drive_duration_s or 0.0),
                )
                settle_with_sensor_poll(reader, decision.settle_duration_s)

            if decision.reached:
                print(
                    f"[approach-box] target confirmed at {distance_m * 100:.1f}cm"
                )
                return 0
    finally:
        try:
            if started:
                if autonomous_enabled:
                    motion.stop()
                    motion.set_manual()
                motion.close()
        finally:
            reader.close()


def run_read_only(args: argparse.Namespace) -> int:
    reader = FrontUltrasoundReader(args.topic)
    samples: list[float] = []
    try:
        reader.start()
        distance_m, _, sequence = reader.wait_for_first_sample(timeout_s=3.0)
        samples.append(distance_m)
        deadline = time.monotonic() + 1.0
        while len(samples) < 5 and time.monotonic() < deadline:
            distance_m, _, current_sequence = reader.poll(timeout_sec=0.1)
            if current_sequence == sequence:
                continue
            sequence = current_sequence
            samples.append(distance_m)
    finally:
        reader.close()
    print(
        "[approach-box] read-only distances_cm="
        + ",".join(f"{sample * 100.0:.1f}" for sample in samples)
    )
    return 0


def main() -> int:
    args = build_parser().parse_args()
    target_m = float(args.target_cm) / 100.0
    if not 0.28 <= target_m <= 4.50:
        raise SystemExit("--target-cm must be within the 28cm to 450cm sensor range")

    if args.read_only:
        if not args.robot:
            raise SystemExit("--read-only requires --robot for live sensor input")
        return run_read_only(args)

    if not args.robot:
        controller = BoxApproachController(
            ApproachConfig(target_distance_m=target_m)
        )
        return run_simulation(
            controller,
            parse_simulated_distances(args.simulate_cm),
        )

    try:
        return run_robot(args)
    except KeyboardInterrupt:
        print("[approach-box] interrupted; robot stopped")
        return 130
    except Exception as exc:
        print(f"[approach-box] failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

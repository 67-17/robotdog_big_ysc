from __future__ import annotations

import argparse
import time
from pathlib import Path

from ..config_loader import load_config
from ..lite3_motion import Lite3MotionController


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lite3 motion calibration helper")
    parser.add_argument("--config-dir", type=Path, default=None)
    parser.add_argument("--mode", choices=["forward", "backward", "left", "right", "turn-left", "turn-right"], default="forward")
    parser.add_argument("--speed", type=float, default=0.15, help="Linear m/s or angular rad/s")
    parser.add_argument("--duration", type=float, default=2.0, help="Command duration for each run")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--robot", action="store_true", help="Actually send commands to the robot")
    parser.add_argument("--udp-fallback", action="store_true")
    parser.add_argument("--axis-fallback", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Skip the final hardware confirmation prompt")
    return parser


def command_for_mode(mode: str, speed: float) -> tuple[float, float, float]:
    if mode == "forward":
        return speed, 0.0, 0.0
    if mode == "backward":
        return -speed, 0.0, 0.0
    if mode == "left":
        return 0.0, speed, 0.0
    if mode == "right":
        return 0.0, -speed, 0.0
    if mode == "turn-left":
        return 0.0, 0.0, speed
    if mode == "turn-right":
        return 0.0, 0.0, -speed
    raise ValueError(f"unsupported mode: {mode}")


def main() -> int:
    args = build_parser().parse_args()
    dry_run = not args.robot
    if args.robot and not args.yes:
        input("This will move the Lite3. Place it in a clear area, then press Enter to continue...")

    config = load_config(args.config_dir)
    motion = Lite3MotionController(
        config,
        dry_run=dry_run,
        udp_fallback=args.udp_fallback,
        axis_fallback=args.axis_fallback,
    )
    vx, vy, wz = command_for_mode(args.mode, abs(args.speed))
    print(f"[calibrate] backend={motion.backend_name} mode={args.mode} command=({vx:.3f}, {vy:.3f}, {wz:.3f})")
    motion.start()
    try:
        if args.robot:
            motion.stand_up()
            time.sleep(5.0)
            motion.set_autonomous()
        for idx in range(1, max(1, args.repeat) + 1):
            print(f"[calibrate] run {idx}/{args.repeat}: duration={args.duration:.2f}s")
            motion.hold_velocity(vx, vy, wz, args.duration)
            time.sleep(1.0 if args.robot else 0.1)
    finally:
        motion.close()
    print("[calibrate] done; measure actual distance/angle and update config/robot.yaml if needed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

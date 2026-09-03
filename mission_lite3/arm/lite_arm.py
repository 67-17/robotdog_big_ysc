from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = Path(__file__).resolve().parent / "runtime"
RUNTIME_CONFIG_KEYS = {
    "runtime_config",
    "calibration",
    "grasp_reference",
    "moving_pose",
    "place_reference",
}


@dataclass(frozen=True)
class ArmTaskResult:
    ok: bool
    stage: str
    reason: str = ""
    feedback: str = ""
    object_held: bool = False
    released: bool = False
    hardware_state: str = "UNKNOWN"
    requires_power_cycle: bool = False
    payload: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        default_ok: bool = False,
    ) -> "ArmTaskResult":
        ok = bool(payload.get("ok", default_ok))
        object_held = bool(payload.get("object_held", False))
        # The source runtime reports a successful place as ok=true with a slot
        # and object_held=false; it does not add a separate released field.
        released = bool(
            payload.get(
                "released",
                ok and "slot" in payload and not object_held,
            )
        )
        return cls(
            ok=ok,
            stage=str(payload.get("stage") or "UNKNOWN"),
            reason=str(payload.get("reason") or ""),
            feedback=str(payload.get("feedback") or ""),
            object_held=object_held,
            released=released,
            hardware_state=str(payload.get("hardware_state") or "UNKNOWN"),
            requires_power_cycle=bool(payload.get("requires_power_cycle", False)),
            payload=dict(payload),
        )

    @classmethod
    def success(cls, stage: str, reason: str = "") -> "ArmTaskResult":
        return cls(True, stage, reason=reason)


class LiteArmController:
    def __init__(self, config: dict, dry_run: bool = False, skip_arm: bool = False):
        self.config = config
        self.arm_config = dict(config.get("arm", {}))
        self.camera_config = dict(config.get("camera", {}))
        self.dry_run = dry_run
        self.skip_arm = skip_arm or not self.arm_config.get("enabled", True)
        self.backend = str(self.arm_config.get("backend", "runtime")).lower()
        self.controller = None
        self.last_result: ArmTaskResult | None = None
        self._grasp_ready_prepared = False

    def start(self) -> ArmTaskResult:
        if self.dry_run or self.skip_arm:
            print("[arm] disabled or dry-run")
            return ArmTaskResult.success("DISABLED")
        if self.backend == "legacy":
            return self._start_legacy()
        if self.backend != "runtime":
            return ArmTaskResult(
                False,
                "PREFLIGHT",
                reason=f"unknown arm backend: {self.backend!r}",
            )
        print(
            "[arm] runtime backend "
            f"port={self._port()} camera={self._camera_device()}"
        )
        return self._run_runtime_task(
            "preflight",
            result_name="preflight",
            include_camera=True,
        )

    def close(self) -> None:
        self._grasp_ready_prepared = False
        if self.controller is not None:
            try:
                self.controller.finalize()
            except Exception:
                pass

    def stow(self) -> ArmTaskResult:
        print("[arm] stow")
        self._grasp_ready_prepared = False
        if self.backend == "legacy":
            if self.controller is not None:
                self.controller.set_pose(3)
                return ArmTaskResult.success("STOW")
            return ArmTaskResult(
                False,
                "STOW",
                reason="legacy controller is unavailable",
            )
        command = str(self.arm_config.get("stow_command", "moving-pose")).strip()
        if not command or command.lower() == "none":
            return ArmTaskResult.success("STOW", reason="stow command disabled")
        if command.lower().replace("_", "-") == "moving-pose":
            result = self.moving_pose()
            if result.payload and not self.dry_run and not self.skip_arm:
                self._write_last_result("stow", result.payload)
            return result
        return self._run_runtime_task(command, result_name="stow")

    def moving_pose(self) -> ArmTaskResult:
        print("[arm] moving pose")
        self._grasp_ready_prepared = False
        if self.dry_run or self.skip_arm:
            payload = {
                "ok": True,
                "stage": "MOVING_POSE",
                "reason": "",
                "object_held": False,
            }
            return ArmTaskResult.from_payload(payload, default_ok=True)
        if self.backend == "legacy":
            if self.controller is None:
                return ArmTaskResult(
                    False,
                    "MOVING_POSE",
                    reason="legacy controller is unavailable",
                )
            self.controller.set_pose(3)
            return ArmTaskResult.success("MOVING_POSE")
        return self._run_source_moving_pose()

    def camera_pose(self) -> ArmTaskResult:
        print("[arm] camera pose")
        if self.backend == "legacy":
            if self.controller is not None:
                self.controller.set_pose(2)
                result = ArmTaskResult.success("GRASP_READY")
            else:
                result = ArmTaskResult(
                    False,
                    "GRASP_READY",
                    reason="legacy controller is unavailable",
                )
        else:
            result = self._run_runtime_task(
                "grasp-ready",
                result_name="grasp_ready",
            )
        self._grasp_ready_prepared = bool(result.ok)
        return result

    def grasp_red_bar(self, distance_mm: float) -> ArmTaskResult:
        print(f"[arm] grasp red bar at {distance_mm:.1f} mm")
        if self.dry_run or self.skip_arm:
            self._grasp_ready_prepared = False
            return ArmTaskResult(
                True,
                "GRASP_SIMULATED",
                object_held=True,
                payload={"simulated": True, "object_held": True},
            )
        if self.backend == "legacy":
            ok = self._legacy_grasp_red_bar(distance_mm)
            self._grasp_ready_prepared = False
            return ArmTaskResult(
                ok,
                "GRASP",
                reason="" if ok else "legacy grasp failed",
                object_held=ok,
            )
        prepared = self._grasp_ready_prepared
        result = self._run_runtime_task(
            "grasp",
            result_name="grasp",
            include_camera=True,
            pre_command_args=("--skip-grasp-ready",) if prepared else (),
        )
        self._grasp_ready_prepared = False
        return result

    def place_to_box(self, letter: str) -> ArmTaskResult:
        slot = str(letter).upper()
        print(f"[arm] place to box {slot}")
        if self.dry_run or self.skip_arm:
            return ArmTaskResult(
                True,
                "PLACE_SIMULATED",
                object_held=False,
                released=True,
                payload={
                    "simulated": True,
                    "object_held": False,
                    "slot": slot,
                },
            )
        if self.backend == "legacy":
            ok = self._legacy_place_to_box(slot)
            return ArmTaskResult(
                ok,
                "PLACE",
                reason="" if ok else "legacy place failed",
                released=ok,
            )
        return self._run_runtime_task(
            "place",
            result_name=f"place_{slot}",
            command_args=("--slot", slot, "--object-held"),
        )

    def abort(self) -> ArmTaskResult:
        if self.dry_run or self.skip_arm:
            return ArmTaskResult.success("ABORTED")
        if self.backend == "legacy":
            return ArmTaskResult(
                False,
                "ABORTED",
                reason="legacy backend has no packaged abort command",
            )
        return self._run_runtime_task("abort", result_name="abort")

    def _start_legacy(self) -> ArmTaskResult:
        sample_root = Path.cwd() / "26比赛资料" / "DeepRobotDog-main"
        if str(sample_root) not in sys.path:
            sys.path.insert(0, str(sample_root))
        try:
            from utils.ArmController import ArmController  # type: ignore

            self.controller = ArmController(self.arm_config["port"])
            return ArmTaskResult.success("PREFLIGHT")
        except Exception as exc:
            return ArmTaskResult(
                False,
                "PREFLIGHT",
                reason=f"failed to initialize legacy backend: {exc}",
            )

    def _legacy_grasp_red_bar(self, distance_mm: float) -> bool:
        if self.dry_run or self.skip_arm:
            time.sleep(0.2)
            return True
        if self.controller is None:
            return False
        height = float(self.arm_config["grasp_height_mm"])
        ok = self.controller.grap(distance_mm, height)
        time.sleep(3.0)
        return ok is not False

    def _legacy_place_to_box(self, letter: str) -> bool:
        if self.dry_run or self.skip_arm:
            time.sleep(0.2)
            return True
        if self.controller is None:
            return False
        self.controller.set_pose(4)
        time.sleep(1.0)
        self.controller.set_pose(3)
        return True

    def _run_runtime_task(
        self,
        command: str,
        *,
        result_name: str,
        include_camera: bool = False,
        command_args: Iterable[str] = (),
        pre_command_args: Iterable[str] = (),
    ) -> ArmTaskResult:
        if self.dry_run or self.skip_arm:
            time.sleep(0.2)
            return ArmTaskResult.success(command.upper().replace("-", "_"))

        from .runtime import arm_task

        result_file = self._invocation_result_file(result_name)
        argv = self._runtime_base_argv(result_file, include_camera=include_camera)
        if command == "grasp":
            argv.extend(["--run-log-dir", str(self._new_run_log_dir())])
        argv.extend(str(item) for item in pre_command_args)
        argv.append(command)
        argv.extend(str(item) for item in command_args)

        try:
            exit_code = arm_task.main(argv)
        except Exception as exc:
            self.last_result = ArmTaskResult(
                False,
                "ARM_RUNTIME",
                reason=str(exc),
            )
            print(f"[arm] runtime {command} failed before result: {exc}")
            return self.last_result

        payload = self._load_result(result_file)
        if exit_code != 0:
            payload["ok"] = False
            payload.setdefault("reason", f"arm runtime exited with {exit_code}")
        result = ArmTaskResult.from_payload(payload, default_ok=exit_code == 0)
        self.last_result = result
        self._write_last_result(result_name, payload)
        print(
            f"[arm] runtime {command} ok={result.ok} stage={result.stage} "
            f"feedback={result.feedback} reason={result.reason}"
        )
        return result

    def _run_source_moving_pose(self) -> ArmTaskResult:
        from .runtime import arm_task
        from .runtime import test as arm_test

        pose_path = self._path(
            "ARM_MOVING_POSE",
            "moving_pose",
            RUNTIME_DIR / "moving_pose.json",
        )
        speed = float(arm_test.MOVING_POSE_SPEED)
        acceleration = float(arm_test.MOVING_POSE_ACCELERATION)
        payload: dict[str, Any] = {
            "ok": False,
            "stage": "MOVING_POSE",
            "reason": "moving pose did not start",
            "object_held": False,
            "pose_file": str(pose_path),
            "spd": speed,
            "acc": acceleration,
        }
        try:
            record = arm_test.read_pose_record(pose_path)
            command = arm_test.build_pose_command(
                record,
                spd=speed,
                acc=acceleration,
            )
            payload["plan"] = [
                {
                    "stage": "MOVE_TO_MOVING_POSE",
                    "joints_feedback_deg": dict(record["joints_feedback_deg"]),
                    "joints_command_deg": {
                        joint: float(command[joint])
                        for joint in arm_test.JOINT_KEYS
                    },
                    "gripper_deg": float(record["gripper_deg"]),
                }
            ]
            motion = arm_task.ArmTestSerialMotion(
                port=self._port(),
                baudrate=int(self._env_or_config("ARM_BAUD", "baud", 115200)),
                timeout=float(self._env_or_config("ARM_TIMEOUT", "timeout", 2.0)),
            )
            status = motion._wait_ready(motion._query_status())  # noqa: SLF001
            motion._send(command)  # noqa: SLF001
            motion._remember_command_pose(command)  # noqa: SLF001
            tolerance = float(arm_test.TARGET_TOLERANCE_DEGREES)
            for joint in arm_test.JOINT_KEYS:
                if motion._joint_already_within_target(  # noqa: SLF001
                    status,
                    joint,
                    float(command[joint]),
                    tolerance,
                ):
                    continue
                status = arm_test.wait_for_joint_target(
                    query_status_fn=motion._query_fast_status,  # noqa: SLF001
                    joint=joint,
                    target_degrees=float(command[joint]),
                    initial_status=status,
                    tolerance_degrees=tolerance,
                    timeout_seconds=motion._joint_target_timeout_seconds(  # noqa: SLF001
                        status,
                        joint,
                        float(command[joint]),
                        speed,
                    ),
                )

            gripper_target = float(record["gripper_deg"])
            gripper_deadline = (
                time.monotonic()
                + float(arm_task.GRIPPER_MOTION_WAIT_TIMEOUT_SECONDS)
            )
            while True:
                status = motion._query_fast_status()  # noqa: SLF001
                gripper_angle = float(arm_test.status_to_gripper_degrees(status))
                if (
                    abs(gripper_angle - gripper_target)
                    <= float(arm_task.GRIPPER_TARGET_TOLERANCE_DEGREES)
                ):
                    break
                if time.monotonic() >= gripper_deadline:
                    raise RuntimeError(
                        "moving-pose gripper did not reach target: "
                        f"target={gripper_target:.2f} actual={gripper_angle:.2f}"
                    )
                time.sleep(float(arm_test.MOTION_POLL_SECONDS))

            payload.update(
                {
                    "ok": True,
                    "reason": "",
                    "final_joints_feedback_deg": arm_test.status_to_joint_degrees(
                        status
                    ),
                    "final_gripper_deg": gripper_angle,
                }
            )
        except Exception as exc:
            payload["reason"] = str(exc)

        result = ArmTaskResult.from_payload(payload)
        self.last_result = result
        self._write_last_result("moving_pose", payload)
        print(
            "[arm] source moving-pose "
            f"ok={result.ok} stage={result.stage} reason={result.reason}"
        )
        return result

    def _runtime_base_argv(
        self,
        result_file: Path,
        *,
        include_camera: bool,
    ) -> list[str]:
        argv = [
            "--port",
            self._port(),
            "--baud",
            str(self._env_or_config("ARM_BAUD", "baud", 115200)),
            "--timeout",
            str(self._env_or_config("ARM_TIMEOUT", "timeout", 2.0)),
            "--reference",
            str(
                self._path(
                    "ARM_GRASP_REFERENCE",
                    "grasp_reference",
                    RUNTIME_DIR / "grasp_reference_square_face.json",
                )
            ),
            "--place-reference",
            str(
                self._path(
                    "ARM_PLACE_REFERENCE",
                    "place_reference",
                    RUNTIME_DIR / "place_reference.json",
                )
            ),
            "--config",
            str(
                self._path(
                    "ARM_CONFIG",
                    "runtime_config",
                    RUNTIME_DIR / "strip_detector_grasp_config.json",
                )
            ),
            "--json-result",
            "--result-file",
            str(result_file),
        ]
        if include_camera:
            argv.extend(
                [
                    "--device",
                    self._camera_device(),
                    "--calibration",
                    str(
                        self._path(
                            "ARM_CALIBRATION",
                            "calibration",
                            RUNTIME_DIR / "camera_calibration.json",
                        )
                    ),
                    "--width",
                    str(self._env_or_config("ARM_WIDTH", "camera_width", 1280)),
                    "--height",
                    str(self._env_or_config("ARM_HEIGHT", "camera_height", 720)),
                    "--fps",
                    str(self._env_or_config("ARM_FPS", "camera_fps", 25)),
                    "--frames-per-detect",
                    str(
                        self._env_or_config(
                            "ARM_FRAMES_PER_DETECT",
                            "frames_per_detect",
                            5,
                        )
                    ),
                ]
            )
        for env_name, key, option in (
            ("ARM_MAX_ALIGN_STEPS", "max_align_steps", "--max-align-steps"),
            ("ARM_MAX_JOG_DEG", "max_jog_deg", "--max-jog-deg"),
            ("ARM_SPD", "speed", "--spd"),
            ("ARM_ACC", "acceleration", "--acc"),
            ("ARM_FINAL_SPD", "final_speed", "--final-spd"),
            ("ARM_FINAL_ACC", "final_acceleration", "--final-acc"),
        ):
            value = self._optional_env_or_config(env_name, key)
            if value is not None:
                argv.extend([option, str(value)])
        return argv

    def _port(self) -> str:
        return str(self._env_or_config("ARM_PORT", "port", "/dev/ttyUSB0"))

    def _camera_device(self) -> str:
        default = self.camera_config.get("arm", "/dev/video0")
        return str(self._env_or_config("ARM_CAMERA", "camera_device", default))

    def _env_or_config(self, env_name: str, key: str, default: Any) -> Any:
        env_value = os.environ.get(env_name)
        if env_value not in (None, ""):
            return env_value
        return self.arm_config.get(key, default)

    def _optional_env_or_config(self, env_name: str, key: str) -> Any:
        env_value = os.environ.get(env_name)
        if env_value not in (None, ""):
            return env_value
        return self.arm_config.get(key)

    def _path(self, env_name: str, key: str, default: Path) -> Path:
        raw_value = self._env_or_config(env_name, key, default)
        path = Path(str(raw_value))
        if path.is_absolute():
            return path
        project_path = PROJECT_ROOT / path
        if project_path.exists():
            return project_path
        runtime_path = RUNTIME_DIR / path
        if key in RUNTIME_CONFIG_KEYS and runtime_path.exists():
            return runtime_path
        return project_path

    def _result_file(self, name: str) -> Path:
        result_dir = self._path(
            "ARM_RESULT_DIR",
            "result_dir",
            PROJECT_ROOT / "logs",
        )
        result_dir.mkdir(parents=True, exist_ok=True)
        return result_dir / f"last_{name}_result.json"

    def _invocation_result_file(self, name: str) -> Path:
        result_dir = self._path(
            "ARM_RESULT_DIR",
            "result_dir",
            PROJECT_ROOT / "logs",
        )
        result_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        suffix = f"{time.time_ns() % 1_000_000_000:09d}"
        return result_dir / f"{timestamp}_{suffix}_{name}_result.json"

    def _write_last_result(self, name: str, payload: Mapping[str, Any]) -> None:
        destination = self._result_file(name)
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_text(
            json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)

    def _new_run_log_dir(self) -> Path:
        run_log_root = self._path(
            "ARM_RUN_LOG_DIR",
            "run_log_dir",
            PROJECT_ROOT / "grasp_runs",
        )
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        suffix = f"{time.time_ns() % 1_000_000_000:09d}"
        return run_log_root / f"{timestamp}_{suffix}"

    def _load_result(self, result_file: Path) -> dict[str, Any]:
        if not result_file.exists():
            return {
                "ok": False,
                "stage": "ARM_RUNTIME",
                "reason": "result file missing",
            }
        try:
            return json.loads(result_file.read_text(encoding="utf-8"))
        except Exception as exc:
            return {
                "ok": False,
                "stage": "ARM_RUNTIME",
                "reason": f"failed to read result file: {exc}",
            }

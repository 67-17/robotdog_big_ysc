from __future__ import annotations

import json
import math
import os
import time
import uuid
from collections import Counter, deque
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

from .arm import ArmTaskResult, LiteArmController
from .audio import AudioReporter, build_announcement
from .box_center_alignment import (
    BoxCenterAligner,
    BoxCenterAlignmentResult,
    PlacementLetterFrameResult,
    annotate_placement_letters,
    detect_placement_box_centers,
    detect_placement_letter_candidates,
    strafe_distance_for_box_center,
)
from .box_approach import ApproachConfig, BoxApproachController
from .camera import CameraSource
from .inspection_tags import (
    median_tag_observation,
    plan_inspection_tag_correction,
    station_tag_target,
)
from .lite3_motion import Lite3MotionController
from .pregrasp_red_align import ArmRedObserver, PregraspRedAligner
from .pickup_transfer import PickupTransferController, body_frame_delta
from .placement_letter_navigation import (
    ActionKind,
    LetterCandidate,
    MOTION_MEASUREMENT_TOLERANCE_M,
    NavigationAction,
    NavigationObservation,
    PlacementLetterNavigationConfig,
    PlacementLetterNavigator,
    placement_strafe_completion_tolerance,
)
from .round_result import (
    build_round_result,
    evaluate_round_gate,
    merge_record_into_round,
    records_from_round_result,
    write_json_atomic,
    write_empty_round_result,
    write_latest_stop_result,
)
from .state_reader import StateReader
from .startup_avoidance import StartupAvoidanceRunner
from .vision import InspectionRecord, VisionPipeline
from .wide_box_alignment import WideBoxAligner, detect_placement_row_parallel
from .wide_camera import WideCameraUndistorter


class MissionState(Enum):
    BOOT_CHECK = auto()
    STAND_AND_ARM = auto()
    PASS_OBSTACLE = auto()
    INSPECT_LEFT_OBJECT = auto()
    INSPECT_RIGHT_OBJECT = auto()
    REPORT_RESULTS = auto()
    PICK_RED_BAR = auto()
    PLACE_TO_LETTER_BOX = auto()
    SECOND_PICK_PLACE = auto()
    FINISH_OR_SAFE_STOP = auto()
    FAULT_HOLD = auto()
    ABORT_SAFE = auto()


class MissionAbort(RuntimeError):
    """Stop the state machine without executing any later mission state."""


class InspectionRecognitionFailed(MissionAbort):
    """The current inspection stop produced no usable recognition result."""


class ForwardMotionGuardStop(MissionAbort):
    """A valid front-distance guard stopped an active forward command."""


class PlacementSearchBoundary(MissionAbort):
    """The front echo changed enough to indicate a placement-row boundary."""


@dataclass(frozen=True)
class MissionResult:
    status: str
    state: str
    reason: str
    records: Dict[str, InspectionRecord]
    placed_letters: List[str]
    carried_bar: bool

    @property
    def ok(self) -> bool:
        return self.status == "completed"

    @property
    def exit_code(self) -> int:
        if self.ok:
            return 0
        if self.status == "interrupted":
            return 130
        return 1


@dataclass
class MissionContext:
    records: Dict[str, InspectionRecord] = field(default_factory=dict)
    placed_letters: List[str] = field(default_factory=list)
    reported_letters: List[str] = field(default_factory=list)
    carried_bar: bool = False
    target_letter: Optional[str] = None
    first_outbound_lane_strafe_m: Optional[float] = None
    first_outbound_forward_m: Optional[float] = None
    placement_letter_lateral_m: Dict[str, float] = field(default_factory=dict)
    pickup_target_letter: Optional[str] = None
    pickup_route_name: Optional[str] = None
    pickup_route_action_index: int = 0
    pickup_stage: str = "idle"
    pickup_pregrasp_substage: str = "idle"
    pickup_retreat_progress_m: float = 0.0
    pickup_entry_strafe_progress_m: float = 0.0
    pickup_entry_tag_acquired: bool = False
    pickup_search_origin_pose: Optional[tuple[float, float, float]] = None
    placement_target_letter: Optional[str] = None
    placement_route_action_index: int = 0
    placement_stage: str = "idle"
    placement_visual_approach_complete: bool = False
    placement_ultrasound_approach_complete: bool = False
    placement_letter_centered_complete: bool = False
    placement_search_front_target_m: Optional[float] = None
    placement_post_forward_yaw_attempted: bool = False
    placement_post_forward_yaw_ok: bool = False
    placement_forced_forward_progress_m: float = 0.0
    placement_forced_forward_odom_m: float = 0.0
    placement_navigation_net_lateral_m: float = 0.0
    placement_navigation_lateral_travel_m: float = 0.0
    placement_navigation_search_phase: str = "left"
    placement_navigation_last_lateral_sign: Optional[int] = None
    placement_navigation_last_geometry_sign: Optional[int] = None
    placement_navigation_pending_recovery_sign: Optional[int] = None
    placement_navigation_zero_progress_count: int = 0
    placement_final_approach_complete: bool = False
    placement_final_approach_progress_m: float = 0.0
    placement_legacy_offset_m: float = 0.0
    dry_run: bool = False
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def anomalous_letters(self) -> List[str]:
        letters = [letter for letter, record in self.records.items() if record.state == "异常"]
        return letters[:2]


@dataclass(frozen=True)
class ObstacleCheck:
    close: bool
    reason: str = ""


class LargeQuadrupedMission:
    INSPECTION_WINDOW_NAME = "mission_lite3 inspection"
    INSPECTION_LETTERS = ("A", "B", "C", "D")

    def __init__(
        self,
        config: dict,
        dry_run: bool = False,
        udp_fallback: bool = False,
        axis_fallback: bool = False,
        skip_arm: bool = False,
        ignore_obstacles: bool = False,
        ignore_ultrasound_obstacle: bool = False,
        show_inspection_window: bool = False,
        allow_open_loop: bool = False,
        startup_avoidance_runner_factory=None,
        fault_hold_sleep=None,
        fault_hold_clock=None,
        fault_resume_checker=None,
        fault_hold_max_cycles: Optional[int] = None,
    ):
        self.config = config
        self.context = MissionContext(dry_run=dry_run)
        self.ignore_obstacles = ignore_obstacles
        self.ignore_ultrasound_obstacle = ignore_ultrasound_obstacle
        self._startup_avoidance_runner_factory = (
            startup_avoidance_runner_factory or StartupAvoidanceRunner
        )
        self._fault_hold_sleep = fault_hold_sleep or time.sleep
        self._fault_hold_clock = fault_hold_clock or time.monotonic
        self._fault_resume_checker = fault_resume_checker
        self._fault_hold_max_cycles = fault_hold_max_cycles
        self.show_inspection_window = self._window_available(show_inspection_window)
        self._inspection_window_created = False
        self.motion = Lite3MotionController(config, dry_run=dry_run, udp_fallback=udp_fallback, axis_fallback=axis_fallback)
        self.state_reader = StateReader(config, dry_run=dry_run)
        feedback_required = (
            bool(config.get("navigation", {}).get("feedback_required", True))
            and not allow_open_loop
            and not dry_run
        )
        self.motion.configure_safety(
            self._motion_guard,
            self.state_reader.pose,
            feedback_required=feedback_required,
        )
        self.vision = VisionPipeline(config)
        camera_cfg = config["camera"]
        self.front_camera = CameraSource(
            camera_cfg["front"],
            int(camera_cfg.get("frame_width", 0)) or None,
            int(camera_cfg.get("frame_height", 0)) or None,
            dry_run=dry_run,
            flush_grab_frames=int(camera_cfg.get("flush_grab_frames", 2)),
            stale_frame_reconnect_count=int(camera_cfg.get("stale_frame_reconnect_count", 15)),
            digital_zoom=float(camera_cfg.get("digital_zoom", 1.0)),
            open_timeout_ms=int(camera_cfg.get("open_timeout_ms", 3000)),
            read_timeout_ms=int(camera_cfg.get("read_timeout_ms", 2000)),
            reconnect_backoff_s=float(camera_cfg.get("reconnect_backoff_s", 0.25)),
            persistent_latest=True,
        )
        self.wide_camera = CameraSource(
            camera_cfg["front"],
            int(camera_cfg.get("frame_width", 0)) or None,
            int(camera_cfg.get("frame_height", 0)) or None,
            dry_run=dry_run,
            flush_grab_frames=int(camera_cfg.get("flush_grab_frames", 2)),
            stale_frame_reconnect_count=int(
                camera_cfg.get("stale_frame_reconnect_count", 15)
            ),
            digital_zoom=1.0,
            open_timeout_ms=int(camera_cfg.get("open_timeout_ms", 3000)),
            read_timeout_ms=int(camera_cfg.get("read_timeout_ms", 2000)),
            reconnect_backoff_s=float(camera_cfg.get("reconnect_backoff_s", 0.25)),
            shared_camera=self.front_camera,
        )
        self.inspection_undistorter = None
        inspection_cfg = config.get("inspection", {})
        if bool(inspection_cfg.get("use_wide_undistortion", False)):
            calibration_path = Path(str(camera_cfg["wide_calibration"])).expanduser()
            if not calibration_path.is_absolute():
                calibration_path = Path(__file__).resolve().parent.parent / calibration_path
            self.inspection_undistorter = WideCameraUndistorter.from_file(
                calibration_path
            )
        arm_cfg = config.get("arm", {})
        arm_camera_source = arm_cfg.get("camera_device") or camera_cfg["arm"]
        self.arm_camera = CameraSource(
            arm_camera_source,
            int(arm_cfg.get("camera_width", 0)) or None,
            int(arm_cfg.get("camera_height", 0)) or None,
            dry_run=dry_run,
            flush_grab_frames=int(camera_cfg.get("flush_grab_frames", 2)),
            stale_frame_reconnect_count=int(
                camera_cfg.get("stale_frame_reconnect_count", 15)
            ),
        )
        self.arm = LiteArmController(config, dry_run=dry_run, skip_arm=skip_arm)
        self.pregrasp_aligner = None
        self.pregrasp_box_aligner = None
        self.placement_row_yaw_aligner = None
        self.box_center_aligner = None
        self._placement_undistorter = None
        self._placement_navigation_run_dir: Optional[Path] = None
        self._placement_navigation_events_path: Optional[Path] = None
        self._placement_last_camera_frame_at: Optional[float] = None
        self._placement_last_camera_frame_id: Optional[int] = None
        self._placement_last_camera_signature: Optional[bytes] = None
        self._placement_latest_frame: Optional[tuple[int, Any]] = None
        self._placement_last_sensor_evidence: Dict[str, Any] = {}
        self._placement_front_samples = deque(
            maxlen=int(
                config["placement_letter_navigation"].get(
                    "ultrasound_filter_samples",
                    5,
                )
            )
        )
        self._placement_front_accepted_m: Optional[float] = None
        self._placement_front_jump_candidate_m: Optional[float] = None
        self._placement_front_jump_count = 0
        self.pickup_transfer_controller = None
        self._controlled_box_approach_active = False
        self._placement_forced_forward_active = False
        self._placement_boundary_recovery_active = False
        self._placement_route_active = False
        self._placement_letter_approach_succeeded = False
        self.audio = AudioReporter(config, dry_run=dry_run)
        self.state = MissionState.BOOT_CHECK
        self._round_result_initialized = False

    def run(self) -> MissionResult:
        status = "failed"
        reason = "mission did not start"
        fault_failures: Dict[MissionState, int] = {}
        try:
            motion_started = False
            state_reader_started = False
            arm_started = False
            while not (motion_started and state_reader_started and arm_started):
                try:
                    if not motion_started:
                        self.motion.start()
                        motion_started = True
                    if not state_reader_started:
                        self.state_reader.start()
                        state_reader_started = True
                    if not arm_started:
                        self._require_arm_result(self.arm.start(), "preflight")
                        arm_started = True
                except Exception as exc:
                    self._recover_state_or_raise(
                        MissionState.BOOT_CHECK,
                        exc,
                        fault_failures,
                    )
            fault_failures.pop(MissionState.BOOT_CHECK, None)

            states = [
                MissionState.BOOT_CHECK,
                MissionState.STAND_AND_ARM,
                MissionState.PASS_OBSTACLE,
                MissionState.INSPECT_LEFT_OBJECT,
                MissionState.INSPECT_RIGHT_OBJECT,
                MissionState.REPORT_RESULTS,
                MissionState.PICK_RED_BAR,
                MissionState.PLACE_TO_LETTER_BOX,
                MissionState.SECOND_PICK_PLACE,
            ]
            state_index = 0
            while state_index < len(states):
                state = states[state_index]
                self.state = state
                print(f"\n[mission] state={state.name}")
                try:
                    getattr(self, f"_state_{state.name.lower()}")()
                except Exception as exc:
                    self._recover_state_or_raise(state, exc, fault_failures)
                    continue
                fault_failures.pop(state, None)
                state_index += 1
            placement_retry_error: Exception = MissionAbort(
                "mission reached final checkpoint without two placements: "
                f"placed={self.context.placed_letters}"
            )
            while len(self.context.placed_letters) < 2:
                self._recover_state_or_raise(
                    MissionState.SECOND_PICK_PLACE,
                    placement_retry_error,
                    fault_failures,
                )
                self.state = MissionState.SECOND_PICK_PLACE
                try:
                    self._state_second_pick_place()
                except Exception as exc:
                    placement_retry_error = exc
                    continue
                fault_failures.pop(MissionState.SECOND_PICK_PLACE, None)
                placement_retry_error = MissionAbort(
                    "mission reached final checkpoint without two placements: "
                    f"placed={self.context.placed_letters}"
                )

            while True:
                self.state = MissionState.FINISH_OR_SAFE_STOP
                print(f"\n[mission] state={self.state.name}")
                try:
                    self._state_finish_or_safe_stop()
                    break
                except Exception as exc:
                    self._recover_state_or_raise(
                        MissionState.FINISH_OR_SAFE_STOP,
                        exc,
                        fault_failures,
                    )
            status = "completed"
            reason = ""
        except KeyboardInterrupt:
            self.state = MissionState.ABORT_SAFE
            status = "interrupted"
            reason = "operator interrupted mission"
            print(f"[mission] abort: {reason}")
        except Exception as exc:
            failed_state = self.state.name
            self.state = MissionState.ABORT_SAFE
            status = "failed"
            reason = f"{failed_state}: {exc}"
            print(f"[mission] abort: {reason}")
        finally:
            cleanup_errors = self._cleanup()
            if cleanup_errors:
                cleanup_reason = "; ".join(cleanup_errors)
                print(f"[mission] cleanup errors: {cleanup_reason}")
                if status == "completed":
                    status = "failed"
                    reason = f"cleanup failed: {cleanup_reason}"
        return MissionResult(
            status=status,
            state=self.state.name,
            reason=reason,
            records=dict(self.context.records),
            placed_letters=list(self.context.placed_letters),
            carried_bar=self.context.carried_bar,
        )

    def _fault_hold_enabled(self) -> bool:
        return (
            not self.context.dry_run
            and bool(self.config.get("fault_hold", {}).get("enabled", True))
        )

    def _recover_state_or_raise(
        self,
        failed_state: MissionState,
        exc: Exception,
        failure_counts: Dict[MissionState, int],
    ) -> None:
        if not self._fault_hold_enabled():
            raise exc
        failure_count = failure_counts.get(failed_state, 0) + 1
        failure_counts[failed_state] = failure_count
        max_retries = int(
            self.config.get("fault_hold", {}).get("max_retries_per_state", 2)
        )
        placement_pending = (
            failed_state
            in {
                MissionState.PLACE_TO_LETTER_BOX,
                MissionState.SECOND_PICK_PLACE,
            }
            and self.context.placement_target_letter is not None
            and self.context.placement_stage != "complete"
        )
        if placement_pending:
            max_retries = int(
                self.config.get("fault_hold", {}).get(
                    "max_retries_per_placement_state",
                    8,
                )
            )
        if failure_count > max_retries:
            raise MissionAbort(
                f"{failed_state.name} retry limit reached after "
                f"{failure_count} failures; allowed_retries={max_retries}; "
                f"last_error={exc}"
            ) from exc
        print(
            f"[mission] fault retry state={failed_state.name} "
            f"retry={failure_count}/{max_retries}",
            flush=True,
        )
        self._fault_hold(failed_state, exc)

    def _fault_hold(self, failed_state: MissionState, exc: Exception) -> None:
        config = self.config.get("fault_hold", {})
        poll_interval_s = float(config.get("poll_interval_s", 0.5))
        stable_required = int(config.get("recovery_stable_checks", 5))
        max_wait_s = float(config.get("max_wait_s", 30.0))
        resume_signal_path = str(
            config.get("resume_signal_path", "/tmp/lite3_fault_resume")
        ).strip()
        retry_safe_states = {
            MissionState.BOOT_CHECK,
            MissionState.STAND_AND_ARM,
            MissionState.PASS_OBSTACLE,
            MissionState.REPORT_RESULTS,
            MissionState.PICK_RED_BAR,
            MissionState.PLACE_TO_LETTER_BOX,
            MissionState.SECOND_PICK_PLACE,
            MissionState.FINISH_OR_SAFE_STOP,
        }
        reason = f"{failed_state.name}: {exc}"
        self.state = MissionState.FAULT_HOLD
        print(f"[mission] fault_hold: {reason}", flush=True)
        stable_checks = 0
        cycles = 0
        resume_latched = False
        hold_started_at = self._fault_hold_clock()
        auto_retry = failed_state in retry_safe_states

        while True:
            cycles += 1
            try:
                self.motion.stop()
            except Exception as stop_error:
                if cycles == 1 or cycles % 20 == 0:
                    print(
                        f"[mission] fault_hold stop retry: {stop_error}",
                        flush=True,
                    )

            try:
                require_ultrasound = (
                    not self.ignore_obstacles
                    and not self.ignore_ultrasound_obstacle
                    and bool(
                        self.config.get("safety", {}).get(
                            "use_ultrasound_obstacle", True
                        )
                    )
                )
                safety_error = self.state_reader.safety_error(
                    require_ultrasound=require_ultrasound,
                    require_fresh=True,
                )
            except Exception as state_error:
                safety_error = f"state recovery check failed: {state_error}"

            if safety_error:
                stable_checks = 0
            else:
                stable_checks = min(stable_required, stable_checks + 1)

            resume_latched = (
                self._consume_fault_resume_signal() or resume_latched
            )
            boot_retry = (
                failed_state == MissionState.BOOT_CHECK
                and cycles >= stable_required
            )
            if boot_retry or (
                stable_checks >= stable_required
                and (auto_retry or resume_latched)
            ):
                self.state = failed_state
                mode = "automatic" if auto_retry else "operator-confirmed"
                print(
                    f"[mission] fault_hold recovered mode={mode} "
                    f"state={failed_state.name}",
                    flush=True,
                )
                return

            elapsed_s = max(
                0.0,
                float(self._fault_hold_clock()) - float(hold_started_at),
            )
            if elapsed_s >= max_wait_s:
                raise MissionAbort(
                    f"fault hold timed out state={failed_state.name} "
                    f"after {elapsed_s:.1f}s limit={max_wait_s:.1f}s: {exc}"
                )

            if cycles == 1 or cycles % 20 == 0:
                resume_hint = (
                    "automatic"
                    if auto_retry
                    else f"touch {resume_signal_path}"
                )
                print(
                    "[mission] fault_hold waiting "
                    f"state={failed_state.name} safety={safety_error or 'stable'} "
                    f"stable={stable_checks}/{stable_required} "
                    f"resume_required={not auto_retry} "
                    f"resume={resume_hint!r} "
                    f"remaining_s={max(0.0, max_wait_s - elapsed_s):.1f}",
                    flush=True,
                )

            if (
                self._fault_hold_max_cycles is not None
                and cycles >= self._fault_hold_max_cycles
            ):
                raise MissionAbort(
                    f"fault hold test limit reached after {cycles} cycles: {reason}"
                )
            self._fault_hold_sleep(poll_interval_s)

    def _consume_fault_resume_signal(self) -> bool:
        if self._fault_resume_checker is not None:
            return bool(self._fault_resume_checker())
        path_value = str(
            self.config.get("fault_hold", {}).get(
                "resume_signal_path", "/tmp/lite3_fault_resume"
            )
        ).strip()
        if not path_value:
            return False
        path = Path(path_value)
        if not path.is_file():
            return False
        try:
            path.unlink()
        except OSError as exc:
            print(
                f"[mission] fault_hold resume signal cleanup warning: {exc}",
                flush=True,
            )
        return True

    def _cleanup(self) -> List[str]:
        errors: List[str] = []
        for attempt in range(3):
            try:
                self.motion.stop()
            except Exception as exc:
                errors.append(f"motion stop {attempt + 1}: {exc}")
            if not self.context.dry_run:
                time.sleep(0.03)
        if self.context.carried_bar:
            try:
                result = self.arm.abort()
                if isinstance(result, ArmTaskResult) and not result.ok:
                    errors.append(f"arm abort: {result.reason}")
            except Exception as exc:
                errors.append(f"arm abort: {exc}")
        for name, close in (
            ("front camera", self.front_camera.release),
            ("wide camera", self.wide_camera.release),
            ("arm camera", self.arm_camera.release),
            ("arm", self.arm.close),
            ("state reader", self.state_reader.close),
            ("motion", self.motion.close),
            ("inspection window", self._close_inspection_window),
        ):
            try:
                close()
            except Exception as exc:
                errors.append(f"{name}: {exc}")
        return errors

    def _require_arm_result(self, result: Any, action: str) -> None:
        # Test doubles and the legacy interface may return None for successful no-op actions.
        if result is None:
            return
        if isinstance(result, ArmTaskResult):
            if result.ok:
                return
            detail = result.reason or result.feedback or result.stage
            raise MissionAbort(f"arm {action} failed: {detail}")
        if result is False:
            raise MissionAbort(f"arm {action} failed")

    def _state_boot_check(self) -> None:
        print(f"[mission] motion backend: {self.motion.backend_name}")
        self._initialize_round_result()
        if not self.context.dry_run:
            safety = self.config.get("safety", {})
            require_ultrasound = (
                bool(safety.get("use_ultrasound_obstacle", True))
                and not self.ignore_obstacles
                and not self.ignore_ultrasound_obstacle
            )
            startup_timeout = float(
                self.config.get("navigation", {}).get("startup_sensor_timeout_s", 3.0)
            )
            self.state_reader.wait_until_ready(startup_timeout, require_ultrasound=require_ultrasound)
            if not self.front_camera.open():
                raise MissionAbort(f"front camera failed to open: {self.front_camera.source}")
            first_frame_timeout_s = float(
                self.config["camera"].get("startup_first_frame_timeout_s", 6.0)
            )
            print(
                f"[camera] boot_first_frame_wait timeout_s={first_frame_timeout_s:.3f}",
                flush=True,
            )
            if self.front_camera.read(timeout_s=first_frame_timeout_s) is None:
                raise MissionAbort(f"front camera opened but did not deliver a frame: {self.front_camera.source}")
        audio_error = self.audio.prewarm()
        if audio_error:
            print(f"[audio] prewarm_warning: {audio_error}", flush=True)
        self._check_safety()

    def _state_stand_and_arm(self) -> None:
        if bool(self.config.get("motion", {}).get("assume_standing", True)):
            print("[mission] assume robot is already standing; skip stand_up")
        else:
            self.motion.stand_up()
            time.sleep(1.0 if self.context.dry_run else 5.0)
        self.motion.set_autonomous()
        self._require_arm_result(self.arm.stow(), "stow")

    def _state_pass_obstacle(self) -> None:
        avoidance_config = self.config.get("startup_avoidance", {})
        use_integrated_avoidance = (
            bool(avoidance_config.get("enabled", False))
            and not self.ignore_obstacles
            and not self.ignore_ultrasound_obstacle
        )
        if use_integrated_avoidance:
            runner = self._startup_avoidance_runner_factory(
                self.config,
                self.motion,
                self.state_reader,
                dry_run=self.context.dry_run,
            )
            result = runner.run()
            if not result.ok:
                raise MissionAbort(
                    f"startup avoidance failed: {result.reason}"
                )
            print(
                "[mission] startup avoidance complete "
                f"count={result.avoidance_count} log={result.log_path or '-'}"
            )
            return
        if self._run_scripted_route("pass_obstacle"):
            return
        self._drive_segment("obstacle_entry", distance_m=1.0)
        for _ in range(8):
            obstacle = self._front_obstacle_check()
            if obstacle.close:
                print(f"[mission] obstacle close ({obstacle.reason}), conservative strafe")
                self.motion.strafe_distance(0.35)
                self.motion.go_distance(0.35)
                self.motion.strafe_distance(-0.35)
            else:
                self.motion.go_distance(0.25)

    def _state_inspect_left_object(self) -> None:
        if not self._run_scripted_route("inspect_stop_1_arrive"):
            self._run_scripted_route("inspect_lower_arrive")
        self._collect_inspection_at_route_anchor(
            "inspection_stop_1",
            default_results=[("A", "偏低")],
        )
        self._run_scripted_route("inspect_stop_3_arrive")
        self._collect_inspection_at_route_anchor(
            "inspection_stop_3",
            default_results=[("C", "偏高")],
        )

    def _state_inspect_right_object(self) -> None:
        self._run_scripted_route("inspect_stop_2_arrive")
        self._collect_inspection_at_route_anchor(
            "inspection_stop_2",
            default_results=[("B", "正常")],
        )
        if not self._run_scripted_route("inspect_stop_4_arrive"):
            self._run_scripted_route("inspect_upper_arrive")
        self._collect_fourth_inspection_with_fallback("inspection_stop_4")
        self._run_scripted_route("inspect_stop_4_depart")

    def _state_report_results(self) -> None:
        print(f"[mission] inspection records={list(self.context.records)}")
        if not self.context.anomalous_letters():
            print("[mission] no anomaly detected; pickup will use the lowest-confidence fallback targets")

    def _state_pick_red_bar(self) -> None:
        if not self._round_result_allows_pickup():
            print("[mission] inspection count gate warning; continue first pickup with best available targets")
        target_letter = (
            self.context.pickup_target_letter
            if self.context.pickup_stage not in {"idle", "complete"}
            else self._next_target_letter()
        )
        if target_letter is None:
            print("[mission] no anomaly target for pickup")
            raise MissionAbort("no anomaly target for pickup")
        if not self._pick_target(target_letter):
            raise MissionAbort(f"failed to pick target {target_letter}")

    def _state_place_to_letter_box(self) -> None:
        if not self._place_carried_bar():
            raise MissionAbort("failed to place first carried bar")

    def _state_second_pick_place(self) -> None:
        placement_resume = (
            self.context.placement_target_letter is not None
            and self.context.placement_stage not in {"idle", "complete"}
        )
        if not self.context.carried_bar and not placement_resume:
            if not self._round_result_allows_pickup():
                print("[mission] inspection count gate warning; continue second pickup with best available targets")
            target_letter = (
                self.context.pickup_target_letter
                if self.context.pickup_stage not in {"idle", "complete"}
                else self._next_target_letter()
            )
            if target_letter is None:
                print("[mission] no second anomaly target")
                raise MissionAbort("no second anomaly target")
            if not self._pick_target(target_letter):
                raise MissionAbort(f"failed to pick second target {target_letter}")
        if not self._place_carried_bar():
            raise MissionAbort("failed to place second carried bar")

    def _state_finish_or_safe_stop(self) -> None:
        self.motion.stop()
        if self.context.carried_bar:
            raise MissionAbort("mission cannot stow while an object is still carried")
        self._require_arm_result(self.arm.stow(), "final stow")
        print(f"[mission] placed anomaly bars: {self.context.placed_letters}")

    def _drive_segment(self, waypoint: str, distance_m: float) -> None:
        print(f"[nav] drive toward {waypoint}")
        self._check_safety()
        self.motion.go_distance(distance_m)

    def _run_scripted_route(self, name: str) -> bool:
        route_cfg = self.config.get("scripted_route", {})
        actions = route_cfg.get(name)
        if not isinstance(actions, list):
            return False
        if name == "inspect_stop_1_arrive" and not self.context.dry_run:
            return self._run_tag_guided_first_inspection_arrival(actions)
        print(f"[nav] scripted route {name}")
        for action in actions:
            if not isinstance(action, dict):
                continue
            self._execute_route_action(action)
        return True

    def _run_tag_guided_first_inspection_arrival(
        self,
        actions: List[Dict[str, Any]],
    ) -> bool:
        """Use Tag 0 to replace the remaining blind route to inspection stop 1."""
        tag_cfg = self.config.get("inspection", {}).get("tag_localization", {})
        target = (
            station_tag_target(tag_cfg, "inspection_stop_1")
            if isinstance(tag_cfg, dict) and bool(tag_cfg.get("enabled", False))
            else None
        )
        detector = getattr(self.vision, "inspection_tag_detector", None)
        guided = bool(
            target is not None
            and detector is not None
            and bool(getattr(detector, "available", False))
        )
        if guided:
            ensure_camera = getattr(self.front_camera, "ensure_running", None)
            try:
                guided = not callable(ensure_camera) or ensure_camera(
                    "inspection_stop_1_transit_tag"
                ) is not False
            except Exception as exc:
                guided = False
                print(
                    f"[inspect-tag-transit] camera check failed={exc}; "
                    "use fixed arrival route"
                )

        print(
            "[nav] scripted route inspect_stop_1_arrive "
            f"tag_guidance={'enabled' if guided else 'unavailable'}"
        )
        for action in actions:
            if not isinstance(action, dict):
                continue
            kind = str(action.get("action", "")).strip()
            distance_m = float(action.get("distance_m", 0.0))
            if not guided or kind not in {"forward", "backward", "strafe"}:
                self._execute_route_action(action)
                continue

            remaining_m = abs(distance_m)
            direction = -1.0 if distance_m < 0.0 else 1.0
            while remaining_m > 1e-9:
                observation, _wrong_ids, frame_error = (
                    self._sample_expected_inspection_tag(target.tag_id, 2, 0.30)
                )
                if frame_error is not None:
                    print(
                        f"[inspect-tag-transit] frame failed={frame_error}; "
                        "finish fixed arrival route"
                    )
                    guided = False
                    break
                if observation is not None:
                    print(
                        "[inspect-tag-transit] acquired inspection_stop_1 "
                        f"id={target.tag_id} center_x={observation.center_x_px:.2f} "
                        f"edge={observation.edge_px:.2f}; stop blind route and servo"
                    )
                    self.motion.stop()
                    self._align_inspection_tag(
                        "inspection_stop_1",
                        initial_observation=observation,
                        max_iterations_override=8,
                    )
                    return True

                chunk_m = min(0.10, remaining_m)
                chunk = dict(action)
                chunk["distance_m"] = direction * chunk_m
                chunk.pop("note", None)
                self._execute_route_action(chunk)
                remaining_m -= chunk_m
            if not guided and remaining_m > 1e-9:
                remainder = dict(action)
                remainder["distance_m"] = direction * remaining_m
                self._execute_route_action(remainder)

        if guided:
            observation, _wrong_ids, _frame_error = (
                self._sample_expected_inspection_tag(target.tag_id, 2, 0.30)
            )
            if observation is not None:
                print(
                    "[inspect-tag-transit] acquired inspection_stop_1 at route end; "
                    "servo from current pose"
                )
                self.motion.stop()
                self._align_inspection_tag(
                    "inspection_stop_1",
                    initial_observation=observation,
                    max_iterations_override=8,
                )
        return True

    def _run_resumable_scripted_route(
        self,
        name: str,
        *,
        progress_attr: str,
    ) -> bool:
        route_cfg = self.config.get("scripted_route", {})
        actions = route_cfg.get(name)
        if not isinstance(actions, list):
            return False
        legacy_override = self.__dict__.get("_run_scripted_route")
        if callable(legacy_override):
            completed = bool(legacy_override(name))
            if completed:
                setattr(self.context, progress_attr, len(actions))
            return completed
        action_index = int(getattr(self.context, progress_attr, 0))
        if action_index < 0 or action_index > len(actions):
            raise MissionAbort(
                f"invalid resumable route checkpoint {name}={action_index}"
            )
        print(
            f"[nav] resumable route {name} "
            f"action={action_index}/{len(actions)}"
        )
        while action_index < len(actions):
            action = actions[action_index]
            if isinstance(action, dict):
                self._execute_route_action(action)
            action_index += 1
            setattr(self.context, progress_attr, action_index)
        return True

    def _execute_route_action(self, action: Dict[str, Any]) -> None:
        kind = str(action.get("action", "")).strip()
        note = action.get("note")
        if note:
            print(f"[nav]   {kind}: {note}")
        self._check_safety()
        if kind == "forward":
            distance_m = float(action.get("distance_m", 0.0))
            if action.get("recorded_outbound_restore") is True:
                self._execute_pickup_forward_restore()
            elif self._placement_route_active:
                self._run_placement_forward(distance_m)
            elif action.get("front_stop_is_completion") is True:
                self._run_front_stop_completion_forward(
                    distance_m,
                    speed_mps=None,
                    label="pickup route",
                )
            else:
                self.motion.go_distance(distance_m)
        elif kind == "backward":
            self.motion.go_distance(-abs(float(action.get("distance_m", 0.0))))
        elif kind == "strafe":
            if action.get("pickup_entry_tag_scan") is True:
                self._run_pickup_entry_tag_scan(
                    float(action.get("distance_m", 0.0))
                )
            else:
                self.motion.strafe_distance(float(action.get("distance_m", 0.0)))
        elif kind == "turn":
            self.motion.turn_by(float(action.get("yaw_rad", 0.0)))
        elif kind == "placement_row_yaw_align":
            if self._placement_route_active and self._pickup_transfer_enabled():
                print(
                    "[placement-yaw] defer row alignment until after the "
                    "fixed post-turn forward segment",
                    flush=True,
                )
            else:
                self._align_placement_row_yaw()
        elif kind == "placement_lane_strafe":
            self._execute_placement_lane_strafe()
        elif kind == "placement_letter_approach":
            self._execute_placement_letter_approach()
        elif kind == "pickup_lane_restore":
            self._execute_pickup_lane_restore()
        elif kind == "wait":
            time.sleep(float(action.get("seconds", 0.0)))
        elif kind == "obstacle_forward":
            self._run_obstacle_forward(action)
        else:
            raise MissionAbort(f"unknown route action: {kind!r}")

    def _pickup_tag_centers(self) -> Dict[int, float]:
        boundary = self.config.get("pickup_tag_boundary", {})
        if not isinstance(boundary, dict) or not bool(boundary.get("enabled", False)):
            return {}
        detector = getattr(self.vision, "inspection_tag_detector", None)
        if detector is None or not bool(getattr(detector, "available", False)):
            return {}
        frame = self.front_camera.read_latest(
            timeout_s=float(boundary.get("sample_timeout_s", 0.25))
        )
        if frame is None:
            return {}
        observations = detector.detect(frame)
        centers: Dict[int, float] = {}
        for observation in observations:
            if int(observation.tag_id) not in centers:
                centers[int(observation.tag_id)] = float(observation.center_x_px)
        return centers

    def _run_pickup_entry_tag_scan(self, distance_m: float) -> None:
        boundary = self.config.get("pickup_tag_boundary", {})
        enabled = isinstance(boundary, dict) and bool(boundary.get("enabled", False))
        requested = float(distance_m)
        total_m = abs(requested)
        direction = -1.0 if requested < 0.0 else 1.0
        if self.context.pickup_entry_tag_acquired:
            print("[pickup-tag] entry already acquired; skip remaining blind strafe")
            return
        progress_m = min(
            total_m,
            max(0.0, float(self.context.pickup_entry_strafe_progress_m)),
        )
        step_m = (
            min(
                total_m or 0.05,
                max(0.01, float(boundary.get("entry_scan_step_m", 0.05))),
            )
            if enabled
            else total_m
        )
        entry_id = int(
            boundary.get("entry_tag_id", boundary.get("right_tag_id", 4))
        ) if enabled else 4
        entry_stop_x = float(
            boundary.get(
                "entry_stop_center_x_px",
                boundary.get("right_stop_center_x_px", 620.0),
            )
        )
        entry_tolerance_x = max(
            0.0,
            float(boundary.get("entry_center_tolerance_px", 0.0)),
        )

        def entry_reached(centers: Dict[int, float]) -> bool:
            return (
                entry_id in centers
                and centers[entry_id] >= entry_stop_x - entry_tolerance_x
            )

        while progress_m < total_m - 1e-9:
            centers = self._pickup_tag_centers() if enabled else {}
            if entry_reached(centers):
                self.motion.stop()
                self.context.pickup_entry_tag_acquired = True
                self.context.pickup_search_origin_pose = tuple(
                    float(value) for value in self.state_reader.pose()
                )
                print(
                    f"[pickup-tag] acquired entry ID {entry_id} "
                    f"center_x={centers[entry_id]:.1f} after "
                    f"strafe={progress_m:.3f}m; begin left red search",
                    flush=True,
                )
                return
            if entry_id in centers:
                print(
                    f"[pickup-tag] entry ID {entry_id} visible "
                    f"center_x={centers[entry_id]:.1f} < {entry_stop_x:.1f}; "
                    "continue fixed-direction strafe",
                    flush=True,
                )
            chunk_m = min(step_m, total_m - progress_m)
            self.motion.strafe_distance(direction * chunk_m)
            progress_m += chunk_m
            self.context.pickup_entry_strafe_progress_m = progress_m
        centers = self._pickup_tag_centers() if enabled else {}
        if entry_reached(centers):
            self.context.pickup_entry_tag_acquired = True
            print(
                f"[pickup-tag] acquired entry ID {entry_id} at route endpoint "
                f"center_x={centers[entry_id]:.1f}",
                flush=True,
            )
        else:
            detail = (
                "not visible"
                if entry_id not in centers
                else f"center_x={centers[entry_id]:.1f} < {entry_stop_x:.1f}"
            )
            print(
                f"[pickup-tag] ID {entry_id} entry not reached within "
                f"{total_m:.2f}m ({detail}); "
                "use fixed-route endpoint as search origin",
                flush=True,
            )
        self.context.pickup_search_origin_pose = tuple(
            float(value) for value in self.state_reader.pose()
        )

    def _run_placement_forward(self, distance_m: float) -> None:
        placement_speed = float(
            self.config["safety"].get("placement_forward_speed_mps", 0.08)
        )
        self._run_front_stop_completion_forward(
            distance_m,
            speed_mps=placement_speed,
            label="placement",
        )

    def _run_front_stop_completion_forward(
        self,
        distance_m: float,
        *,
        speed_mps: Optional[float],
        label: str,
    ) -> None:
        """Resume after isolated spikes and accept only a confirmed front stop."""
        requested = float(distance_m)
        remaining = abs(requested)
        if remaining <= 1e-9:
            self.motion.stop()
            return
        direction = math.copysign(1.0, requested)
        resume_deadline = time.monotonic() + 90.0
        resume_count = 0
        self._prime_placement_front_filter()
        while remaining > 1e-9:
            start_pose = None if self.context.dry_run else self.state_reader.pose()
            try:
                if speed_mps is None:
                    self.motion.go_distance(direction * remaining)
                else:
                    self.motion.go_distance(
                        direction * remaining,
                        speed_mps=speed_mps,
                    )
                return
            except ForwardMotionGuardStop as exc:
                self.motion.stop()
                if start_pose is not None:
                    end_pose = self.state_reader.pose()
                    traveled = math.hypot(
                        float(end_pose[0]) - float(start_pose[0]),
                        float(end_pose[1]) - float(start_pose[1]),
                    )
                    remaining = max(0.0, remaining - traveled)
                if self._confirm_placement_front_stop():
                    print(
                        f"[nav] {label} forward reached confirmed ultrasound "
                        "stop distance; continue with next route action"
                    )
                    return
                resume_count += 1
                if time.monotonic() >= resume_deadline:
                    raise MissionAbort(
                        f"{label} forward did not reach a confirmed ultrasound "
                        f"stop within 90s; remaining={remaining:.3f}m"
                    ) from exc
                if resume_count == 1 or resume_count % 10 == 0:
                    print(
                        f"[nav] {label} forward ignored isolated ultrasound "
                        f"samples; resume remaining={remaining:.3f}m "
                        f"count={resume_count}"
                    )

    def _prime_placement_front_filter(
        self,
        warmup_seconds: Optional[float] = None,
    ) -> None:
        if self.context.dry_run or bool(getattr(self.state_reader, "dry_run", False)):
            return
        filtered_reader = getattr(
            self.state_reader,
            "filtered_front_ultrasound_m",
            None,
        )
        if not callable(filtered_reader):
            return
        duration_s = 1.0 if warmup_seconds is None else max(
            0.0,
            float(warmup_seconds),
        )
        deadline = time.monotonic() + duration_s
        while time.monotonic() < deadline:
            error = self.state_reader.safety_error(
                require_ultrasound=True,
                require_fresh=True,
            )
            if error:
                raise MissionAbort(
                    f"placement front filter warmup rejected sensor state: {error}"
                )
            time.sleep(0.02)
        value = filtered_reader(
            float(
                self.config["safety"].get(
                    "placement_front_filter_window_s",
                    0.8,
                )
            )
        )
        print(f"[nav] placement front filter ready: {float(value):.3f}m")

    def _confirm_placement_front_stop(self) -> bool:
        required_samples = 3
        deadline = time.monotonic() + 1.0
        stop_distance = self._front_stop_distance()
        min_valid = float(
            self.config["safety"].get("front_ultrasound_min_valid_m", 0.03)
        )
        last_sample_at = None
        consecutive_close = 0
        while time.monotonic() < deadline:
            state = self.state_reader.poll()
            sample_at = getattr(state, "ultrasound_updated_at", None)
            if sample_at is None or sample_at == last_sample_at:
                time.sleep(0.02)
                continue
            last_sample_at = sample_at
            error = self.state_reader.safety_error(
                require_ultrasound=True,
                require_fresh=True,
            )
            if error:
                raise MissionAbort(
                    f"placement stop confirmation rejected sensor state: {error}"
                )
            filtered_reader = getattr(
                self.state_reader,
                "filtered_front_ultrasound_m",
                None,
            )
            value = float(
                filtered_reader(
                    float(
                        self.config["safety"].get(
                            "placement_front_filter_window_s",
                            0.8,
                        )
                    )
                )
                if callable(filtered_reader)
                else state.front_ultrasound_m
            )
            if not math.isfinite(value) or value < min_valid:
                raise MissionAbort(
                    f"placement stop confirmation received invalid distance: {value!r}"
                )
            if value > stop_distance:
                return False
            consecutive_close += 1
            if consecutive_close >= required_samples:
                return True
        return False

    def _run_obstacle_forward(self, action: Dict[str, Any]) -> None:
        steps = int(action.get("steps", 1))
        clear_step_m = float(action.get("clear_step_m", 0.25))
        avoid_strafe_m = float(action.get("avoid_strafe_m", 0.35))
        avoid_forward_m = float(action.get("avoid_forward_m", 0.30))
        max_avoid_attempts = max(1, int(action.get("max_avoid_attempts", 3)))
        return_after_avoid = bool(action.get("return_after_avoid", True))
        consecutive_avoid_attempts = 0
        for step_index in range(max(0, steps)):
            obstacle = self._front_obstacle_check()
            if obstacle.close:
                consecutive_avoid_attempts += 1
                print(
                    "[mission] obstacle close "
                    f"step={step_index + 1}/{steps} attempt={consecutive_avoid_attempts}/{max_avoid_attempts} "
                    f"reason=({obstacle.reason}) bypass_strafe={avoid_strafe_m:.2f}m bypass_forward={avoid_forward_m:.2f}m"
                )
                self.motion.stop()
                if consecutive_avoid_attempts > max_avoid_attempts:
                    print("[mission] obstacle bypass limit reached, stop obstacle segment")
                    break
                self.motion.strafe_distance(avoid_strafe_m)
                self.motion.go_distance(avoid_forward_m)
                if return_after_avoid:
                    self.motion.strafe_distance(-avoid_strafe_m)
            else:
                consecutive_avoid_attempts = 0
                self.motion.go_distance(clear_step_m)

    def _front_obstacle_close(self) -> bool:
        return self._front_obstacle_check().close

    def _front_stop_distance(self) -> float:
        default_distance = float(self.config["safety"]["front_stop_distance_m"])
        if self.state in {
            MissionState.INSPECT_LEFT_OBJECT,
            MissionState.INSPECT_RIGHT_OBJECT,
        }:
            return float(
                self.config.get("inspection", {}).get(
                    "front_stop_distance_m",
                    default_distance,
                )
            )
        return default_distance

    def _front_obstacle_check(self) -> ObstacleCheck:
        if self.ignore_obstacles or self.context.dry_run:
            return ObstacleCheck(False)
        state = self.state_reader.poll()
        safety = self.config["safety"]
        stop_distance = self._front_stop_distance()
        min_valid_distance = float(safety.get("front_ultrasound_min_valid_m", 0.03))
        use_ultrasound = bool(safety.get("use_ultrasound_obstacle", True)) and not self.ignore_ultrasound_obstacle
        if use_ultrasound:
            error = self.state_reader.safety_error(require_ultrasound=True)
            if error:
                if self.context.dry_run:
                    return ObstacleCheck(False)
                raise MissionAbort(f"cannot check front obstacle: {error}")
            value = float(state.front_ultrasound_m)
            if not math.isfinite(value) or value < min_valid_distance:
                raise MissionAbort(f"invalid front ultrasound distance: {value!r}")
            if value <= stop_distance:
                return ObstacleCheck(True, f"ultrasound={value:.2f}m <= {stop_distance:.2f}m")
        if not bool(safety.get("use_vision_obstacle", True)):
            return ObstacleCheck(False)
        frame = self.front_camera.read()
        if frame is None:
            if self.context.dry_run:
                return ObstacleCheck(False)
            raise MissionAbort("front camera unavailable during obstacle check")
        cones = self.vision.detect_cones(frame)
        if not cones:
            return ObstacleCheck(False)
        width = frame.shape[1]
        for cone in cones:
            if cone.bbox is None:
                continue
            cx, _ = cone.bbox.center
            if abs(cx - width / 2) < width * 0.22 and cone.bbox.area > 2500:
                return ObstacleCheck(True, f"cone area={cone.bbox.area} center_x={cx:.0f}/{width}")
        return ObstacleCheck(False)

    def _collect_inspection_at_route_anchor(
        self,
        stop_name: str,
        default_results: List[tuple[str, str]],
    ) -> None:
        tag_cfg = self.config.get("inspection", {}).get("tag_localization", {})
        restore_enabled = (
            isinstance(tag_cfg, dict)
            and bool(tag_cfg.get("enabled", False))
            and bool(tag_cfg.get("restore_route_anchor", True))
            and station_tag_target(tag_cfg, stop_name) is not None
        )
        if self.context.dry_run or not restore_enabled:
            self._collect_inspection(stop_name, default_results)
            return
        try:
            route_anchor = tuple(float(value) for value in self.state_reader.pose())
        except Exception as exc:
            raise MissionAbort(
                f"{stop_name} cannot record inspection route anchor: {exc}"
            ) from exc
        if len(route_anchor) != 3 or not all(math.isfinite(value) for value in route_anchor):
            raise MissionAbort(
                f"{stop_name} inspection route anchor is invalid: {route_anchor!r}"
            )
        print(
            f"[inspect-anchor] {stop_name} record "
            f"x={route_anchor[0]:.3f} y={route_anchor[1]:.3f} "
            f"yaw_deg={math.degrees(route_anchor[2]):.2f}"
        )
        alignment_moves: List[tuple[str, float]] = []
        inspection_error: Optional[Exception] = None
        try:
            self._collect_inspection(
                stop_name,
                default_results,
                alignment_moves=alignment_moves,
            )
        except Exception as exc:
            inspection_error = exc

        try:
            self._restore_inspection_route_anchor(
                stop_name,
                route_anchor,
                alignment_moves,
            )
        except Exception as restore_exc:
            if inspection_error is not None:
                raise MissionAbort(
                    f"{stop_name} inspection failed ({inspection_error}); "
                    f"route-anchor restore also failed ({restore_exc})"
                ) from restore_exc
            raise
        if inspection_error is not None:
            raise inspection_error

    def _collect_inspection(
        self,
        stop_name: str,
        default_results: List[tuple[str, str]],
        *,
        alignment_moves: Optional[List[tuple[str, float]]] = None,
    ) -> None:
        inspection_cfg = self.config.get("inspection", {})
        dwell_seconds = float(inspection_cfg.get("stop_dwell_seconds", 10.0))
        speak_at_stop = bool(inspection_cfg.get("speak_at_inspection_stop", True))
        print(f"[inspect] {stop_name} stop and observe mode=random-letter dwell={dwell_seconds:.1f}s")
        self.motion.stop()
        if self.context.dry_run:
            for letter, level in self._select_unused_defaults(default_results):
                state = "正常" if level == "正常" else "异常"
                record = InspectionRecord(letter, level, state, 1.0, -1)
                self._store_inspection_record(record)
                if speak_at_stop:
                    self._announce_record(record)
            return
        self._align_inspection_tag(
            stop_name,
            performed_moves=alignment_moves,
        )
        self.motion.stop()
        ensure_camera = getattr(self.front_camera, "ensure_running", None)
        if callable(ensure_camera) and ensure_camera(f"{stop_name}_reuse") is False:
            raise InspectionRecognitionFailed(
                f"{stop_name} failed: front camera is not running"
            )
        self.vision.reset_inspection_votes()
        deadline = time.monotonic() + max(1.0, dwell_seconds)
        frames_seen = 0
        last_frame = None
        while time.monotonic() < deadline:
            frame = self.front_camera.read()
            if frame is None:
                time.sleep(0.05)
                continue
            frame = self._prepare_inspection_frame(frame)
            frames_seen += 1
            last_frame = frame
            source_camera = (
                "front_wide_undistorted"
                if self.inspection_undistorter is not None
                else "front"
            )
            record = self.vision.inspect_frame(
                frame,
                source_camera=source_camera,
            )
            self._show_inspection_preview(stop_name, frame, record)
            if record is not None:
                if self._accept_inspection_record(
                    stop_name,
                    record,
                    frame,
                    speak_at_stop=speak_at_stop,
                ):
                    break
        else:
            if frames_seen == 0:
                self._persist_round_block("camera_failed")
                raise InspectionRecognitionFailed(
                    f"{stop_name} failed: camera_failed"
                )
            used_best_candidate = False
            best_candidate_getter = getattr(
                self.vision,
                "best_inspection_candidate",
                None,
            )
            candidate_bundle = (
                best_candidate_getter() if callable(best_candidate_getter) else None
            )
            if (
                isinstance(candidate_bundle, tuple)
                and len(candidate_bundle) == 2
                and isinstance(candidate_bundle[0], InspectionRecord)
            ):
                candidate_record, candidate_frame = candidate_bundle
                used_best_candidate = self._accept_inspection_record(
                    stop_name,
                    candidate_record,
                    candidate_frame if candidate_frame is not None else last_frame,
                    speak_at_stop=speak_at_stop,
                )
                if used_best_candidate:
                    print(
                        f"[inspect] {stop_name} timeout after frames={frames_seen}; "
                        f"use best real candidate area={candidate_record.letter} "
                        f"level={candidate_record.level}"
                    )
            if not used_best_candidate:
                if not default_results:
                    self._persist_round_block("inspection_not_stable")
                    raise InspectionRecognitionFailed(
                        f"{stop_name} failed: inspection_not_stable"
                    )
                selected_defaults = self._select_unused_defaults(default_results)
                if not selected_defaults:
                    self._persist_round_block("inspection_letters_exhausted")
                    raise MissionAbort(
                        f"{stop_name} failed: no unused inspection letter for fallback"
                    )
                for letter, level in selected_defaults:
                    state = "正常" if level == "正常" else "异常"
                    record = InspectionRecord(
                        letter,
                        level,
                        state,
                        0.0,
                        -1,
                        source_camera="default_fallback",
                        stability_votes={"default_fallback": 1},
                    )
                    evidence_image = self._save_inspection_evidence(
                        stop_name,
                        letter,
                        last_frame,
                    )
                    if evidence_image is not None:
                        record = replace(record, evidence_image=str(evidence_image))
                    print(
                        f"[inspect] {stop_name} no stable recognition after "
                        f"frames={frames_seen}; use default area={letter} level={level}"
                    )
                    self._store_inspection_record(record)
                    if speak_at_stop:
                        self._announce_record(record)
        print(f"[inspect] {stop_name} random-letter observe complete records={sorted(self.context.records)}")

    def _collect_fourth_inspection_with_fallback(self, stop_name: str) -> None:
        if self.context.dry_run:
            self._infer_fourth_inspection(stop_name)
            return
        try:
            self._collect_inspection_at_route_anchor(
                stop_name,
                default_results=[],
            )
        except InspectionRecognitionFailed as exc:
            print(
                f"[inspect-infer] {stop_name} visual recognition failed: {exc}; "
                "use three-result set-difference fallback"
            )
            self._infer_fourth_inspection(stop_name)

    def _infer_fourth_inspection(self, stop_name: str) -> InspectionRecord:
        self.motion.stop()
        records = list(self.context.records.values())
        if len(records) != 3 or len({record.letter for record in records}) != 3:
            raise MissionAbort(
                f"{stop_name} inference requires exactly three distinct known letters"
            )
        if not self.context.dry_run and any(
            record.source_camera == "default_fallback" for record in records
        ):
            raise MissionAbort(
                f"{stop_name} inference requires three real inspection results"
            )

        remaining_letters = set(self.INSPECTION_LETTERS) - {
            record.letter for record in records
        }
        if len(remaining_letters) != 1:
            raise MissionAbort(
                f"{stop_name} letter inference is not unique: "
                f"remaining={sorted(remaining_letters)}"
            )

        expected_levels = Counter({"正常": 2, "偏高": 1, "偏低": 1})
        observed_levels = Counter(record.level for record in records)
        if any(level not in expected_levels for level in observed_levels):
            raise MissionAbort(
                f"{stop_name} level inference has unknown values: "
                f"observed={dict(observed_levels)}"
            )
        remaining_levels: List[str] = []
        for level, expected_count in expected_levels.items():
            observed_count = observed_levels[level]
            if observed_count > expected_count:
                raise MissionAbort(
                    f"{stop_name} level inference exceeds expected count: "
                    f"level={level} observed={observed_count} expected={expected_count}"
                )
            remaining_levels.extend([level] * (expected_count - observed_count))
        if len(remaining_levels) != 1:
            raise MissionAbort(
                f"{stop_name} level inference is not unique: "
                f"remaining={remaining_levels} observed={dict(observed_levels)}"
            )

        letter = remaining_letters.pop()
        level = remaining_levels[0]
        record = InspectionRecord(
            letter,
            level,
            "正常" if level == "正常" else "异常",
            0.0,
            -1,
            source_camera="set_difference_inference",
            stability_votes={"set_difference_inference": 3},
        )
        self._store_inspection_record(record)
        print(
            f"[inspect-infer] {stop_name} area={letter} level={level} "
            "from=three_known_results"
        )
        if bool(
            self.config.get("inspection", {}).get(
                "speak_at_inspection_stop",
                True,
            )
        ):
            self._announce_record(record)
        return record

    def _align_inspection_tag(
        self,
        stop_name: str,
        *,
        performed_moves: Optional[List[tuple[str, float]]] = None,
        initial_observation: Optional[Any] = None,
        max_iterations_override: Optional[int] = None,
    ) -> List[tuple[str, float]]:
        moves = performed_moves if performed_moves is not None else []
        inspection_cfg = self.config.get('inspection', {})
        tag_cfg = inspection_cfg.get('tag_localization', {})
        if not isinstance(tag_cfg, dict) or not bool(tag_cfg.get('enabled', False)):
            return moves
        target = station_tag_target(tag_cfg, stop_name)
        if target is None:
            return moves
        correction_cfg = dict(tag_cfg)
        station_overrides = tag_cfg.get('station_overrides', {})
        if isinstance(station_overrides, dict):
            station_override = station_overrides.get(stop_name, {})
            if isinstance(station_override, dict):
                correction_cfg.update(station_override)
        detector = getattr(self.vision, 'inspection_tag_detector', None)
        if detector is None or not bool(getattr(detector, 'available', False)):
            reason = getattr(detector, 'unavailable_reason', 'detector_missing')
            print(
                f'[inspect-tag] {stop_name} expected_id={target.tag_id} '
                f'unavailable={reason}; continue without correction'
            )
            return moves
        ensure_camera = getattr(self.front_camera, 'ensure_running', None)
        try:
            if callable(ensure_camera) and ensure_camera(f'{stop_name}_tag_align') is False:
                print(
                    f'[inspect-tag] {stop_name} camera unavailable; '
                    'continue without correction'
                )
                return moves
        except Exception as exc:
            print(
                f'[inspect-tag] {stop_name} camera check failed={exc}; '
                'continue without correction'
            )
            return moves

        samples_per_attempt = int(correction_cfg.get('samples_per_attempt', 3))
        sample_timeout_s = float(correction_cfg.get('sample_timeout_s', 0.8))
        max_iterations = int(
            max_iterations_override
            if max_iterations_override is not None
            else correction_cfg.get('max_iterations', 2)
        )
        min_motion_step_m = float(correction_cfg.get('min_motion_step_m', 0.015))
        observed_wrong_ids: set[int] = set()
        for iteration in range(1, max_iterations + 1):
            if iteration == 1 and initial_observation is not None:
                observation, wrong_ids, frame_error = initial_observation, set(), None
            else:
                observation, wrong_ids, frame_error = self._sample_expected_inspection_tag(
                    target.tag_id,
                    samples_per_attempt,
                    sample_timeout_s,
                )
            observed_wrong_ids.update(wrong_ids)
            if frame_error is not None:
                print(
                    f'[inspect-tag] {stop_name} frame failed={frame_error}; '
                    'continue without correction'
                )
                return moves
            if observation is None:
                print(
                    f'[inspect-tag] {stop_name} expected_id={target.tag_id} '
                    f'not found observed_ids={sorted(observed_wrong_ids)}; '
                    'continue without correction'
                )
                return moves
            correction = plan_inspection_tag_correction(
                observation,
                target,
                correction_cfg,
                focal_x_px=getattr(detector, 'focal_x_px', None),
            )
            print(
                f'[inspect-tag] {stop_name} iteration={iteration}/{max_iterations} '
                f'id={observation.tag_id} center_x={observation.center_x_px:.2f} '
                f'edge={observation.edge_px:.2f} action={correction.kind} '
                f'distance={correction.distance_m:.3f} reason={correction.reason}'
            )
            if correction.kind in {'complete', 'fail'}:
                return moves
            if abs(correction.distance_m) < min_motion_step_m:
                print(
                    f'[inspect-tag] {stop_name} correction below minimum step; '
                    'continue inspection'
                )
                return moves
            try:
                if correction.kind == 'forward':
                    self.motion.go_distance(
                        correction.distance_m,
                        speed_mps=float(correction_cfg.get('forward_speed_mps', 0.08)),
                    )
                    moves.append(('forward', correction.distance_m))
                elif correction.kind == 'strafe':
                    self.motion.strafe_distance(
                        correction.distance_m,
                        speed_mps=float(correction_cfg.get('strafe_speed_mps', 0.06)),
                    )
                    moves.append(('strafe', correction.distance_m))
                else:
                    return moves
            except Exception as exc:
                print(
                    f'[inspect-tag] {stop_name} motion failed={exc}; '
                    'continue inspection from current pose'
                )
                return moves
        print(
            f'[inspect-tag] {stop_name} reached bounded correction limit; '
            'continue inspection'
        )
        return moves

    def _restore_inspection_route_anchor(
        self,
        stop_name: str,
        route_anchor: tuple[float, float, float],
        alignment_moves: List[tuple[str, float]],
    ) -> None:
        tag_cfg = self.config.get("inspection", {}).get("tag_localization", {})
        if not isinstance(tag_cfg, dict):
            raise MissionAbort(f"{stop_name} tag localization config is invalid")
        forward_speed = float(tag_cfg.get("forward_speed_mps", 0.08))
        strafe_speed = float(tag_cfg.get("strafe_speed_mps", 0.06))
        motion_threshold = float(tag_cfg.get("return_motion_threshold_m", 0.02))
        position_tolerance = float(tag_cfg.get("return_tolerance_m", 0.06))
        yaw_tolerance = math.radians(
            float(tag_cfg.get("return_yaw_tolerance_deg", 3.0))
        )
        max_passes = int(tag_cfg.get("return_max_correction_passes", 2))

        reverse_error: Optional[Exception] = None
        for kind, distance_m in reversed(alignment_moves):
            try:
                if kind == "forward":
                    self.motion.go_distance(-distance_m, speed_mps=forward_speed)
                elif kind == "strafe":
                    self.motion.strafe_distance(-distance_m, speed_mps=strafe_speed)
                else:
                    raise MissionAbort(
                        f"{stop_name} has unknown inspection alignment move: {kind}"
                    )
            except Exception as exc:
                reverse_error = exc
                print(
                    f"[inspect-anchor] {stop_name} reverse move failed={exc}; "
                    "try odometry residual correction"
                )
                break

        for correction_pass in range(max_passes + 1):
            try:
                current_pose = tuple(
                    float(value) for value in self.state_reader.pose()
                )
            except Exception as exc:
                raise MissionAbort(
                    f"{stop_name} cannot verify inspection route anchor: {exc}"
                ) from exc
            if len(current_pose) != 3 or not all(
                math.isfinite(value) for value in current_pose
            ):
                raise MissionAbort(
                    f"{stop_name} current inspection pose is invalid: {current_pose!r}"
                )
            forward_error, lateral_error, yaw_error = body_frame_delta(
                route_anchor,
                current_pose,
            )
            print(
                f"[inspect-anchor] {stop_name} pass={correction_pass}/{max_passes} "
                f"forward_error={forward_error:.3f} "
                f"lateral_error={lateral_error:.3f} "
                f"yaw_error_deg={math.degrees(yaw_error):.2f}"
            )
            if (
                abs(forward_error) <= position_tolerance
                and abs(lateral_error) <= position_tolerance
                and abs(yaw_error) <= yaw_tolerance
            ):
                print(
                    f"[inspect-anchor] {stop_name} restored "
                    f"moves={len(alignment_moves)}"
                )
                return
            if correction_pass >= max_passes:
                detail = (
                    f"forward={forward_error:.3f}m "
                    f"lateral={lateral_error:.3f}m "
                    f"yaw={math.degrees(yaw_error):.2f}deg"
                )
                if reverse_error is not None:
                    detail += f" reverse_error={reverse_error}"
                raise MissionAbort(
                    f"{stop_name} failed to restore inspection route anchor: {detail}"
                )
            try:
                if abs(yaw_error) > yaw_tolerance:
                    self.motion.turn_by(-yaw_error)
                if abs(forward_error) > motion_threshold:
                    self.motion.go_distance(
                        -forward_error,
                        speed_mps=forward_speed,
                    )
                if abs(lateral_error) > motion_threshold:
                    self.motion.strafe_distance(
                        -lateral_error,
                        speed_mps=strafe_speed,
                    )
            except Exception as exc:
                raise MissionAbort(
                    f"{stop_name} inspection route-anchor correction failed: {exc}"
                ) from exc

    def _sample_expected_inspection_tag(
        self,
        expected_id: int,
        sample_count: int,
        timeout_s: float,
    ) -> tuple[Optional[Any], set[int], Optional[str]]:
        samples = []
        wrong_ids: set[int] = set()
        deadline = time.monotonic() + timeout_s
        while len(samples) < sample_count and time.monotonic() < deadline:
            try:
                frame = self.front_camera.read()
                if frame is None:
                    time.sleep(0.03)
                    continue
                prepared = self._prepare_inspection_frame(frame)
                observations = self.vision.detect_inspection_tags(prepared)
            except Exception as exc:
                return None, wrong_ids, str(exc)
            matching = [item for item in observations if item.tag_id == expected_id]
            wrong_ids.update(
                item.tag_id for item in observations if item.tag_id != expected_id
            )
            if matching:
                samples.append(max(matching, key=lambda item: item.edge_px))
            else:
                time.sleep(0.03)
        return median_tag_observation(samples), wrong_ids, None

    def _accept_inspection_record(
        self,
        stop_name: str,
        record: InspectionRecord,
        frame,
        *,
        speak_at_stop: bool,
    ) -> bool:
        existing = self.context.records.get(record.letter)
        if existing is not None and existing.source_camera != "default_fallback":
            print(
                f"[inspect] duplicate stable area={record.letter}; "
                "keep previous result and continue observing"
            )
            return False
        if existing is not None:
            self._relocate_displaced_default(existing)
        evidence_image = self._save_inspection_evidence(stop_name, record.letter, frame)
        if evidence_image is not None:
            record = replace(record, evidence_image=str(evidence_image))
            self._save_inspection_diagnostics(record, evidence_image)
        self._store_inspection_record(record)
        print(f"[inspect] {record}")
        if speak_at_stop:
            self._announce_record(record)
        return True

    def _prepare_inspection_frame(self, frame):
        if self.inspection_undistorter is None:
            return frame
        try:
            return self.inspection_undistorter.apply(frame)
        except Exception as exc:
            raise MissionAbort(
                f"inspection wide-camera undistortion failed: {exc}"
            ) from exc

    def _select_unused_defaults(
        self,
        default_results: List[tuple[str, str]],
    ) -> List[tuple[str, str]]:
        available = [
            letter
            for letter in self.INSPECTION_LETTERS
            if letter not in self.context.records
        ]
        selected: List[tuple[str, str]] = []
        normal_count = sum(
            1 for record in self.context.records.values() if record.state == "正常"
        )
        abnormal_count = sum(
            1 for record in self.context.records.values() if record.state == "异常"
        )
        for preferred_letter, preferred_level in default_results:
            preferred_letter = str(preferred_letter).upper()
            if preferred_letter in available:
                letter = preferred_letter
            elif available:
                letter = available[0]
            else:
                break
            available.remove(letter)
            if abnormal_count >= 2:
                level = "正常"
            elif normal_count >= 2:
                level = (
                    preferred_level
                    if preferred_level in {"偏低", "偏高"}
                    else "偏低"
                )
            elif abnormal_count > normal_count:
                level = "正常"
            elif normal_count > abnormal_count:
                level = (
                    preferred_level
                    if preferred_level in {"偏低", "偏高"}
                    else "偏低"
                )
            else:
                level = preferred_level
            selected.append((letter, level))
            if level == "正常":
                normal_count += 1
            else:
                abnormal_count += 1
        return selected

    def _relocate_displaced_default(self, record: InspectionRecord) -> None:
        available = [
            letter
            for letter in self.INSPECTION_LETTERS
            if letter not in self.context.records
        ]
        if not available:
            return
        relocated = replace(record, letter=available[0])
        print(
            f"[inspect] stable area={record.letter} replaces fallback; "
            f"move fallback to unused area={relocated.letter}"
        )
        self._store_inspection_record(relocated)

    def _rebalance_default_fallback_records(self) -> List[str]:
        records = self.context.records
        if set(records) != set(self.INSPECTION_LETTERS):
            return []
        if any(record.state not in {"正常", "异常"} for record in records.values()):
            return []

        fallback_letters = [
            letter
            for letter in self.INSPECTION_LETTERS
            if records[letter].source_camera == "default_fallback"
            and records[letter].confidence == 0.0
        ]
        if not fallback_letters:
            return []
        if any(
            records[letter].source_camera == "default_fallback"
            and records[letter].confidence != 0.0
            for letter in self.INSPECTION_LETTERS
        ):
            return []

        real_records = [
            record
            for record in records.values()
            if record.source_camera != "default_fallback"
        ]
        real_normal = sum(record.state == "正常" for record in real_records)
        real_abnormal = sum(record.state == "异常" for record in real_records)
        needed = {
            "正常": 2 - real_normal,
            "异常": 2 - real_abnormal,
        }
        if (
            needed["正常"] < 0
            or needed["异常"] < 0
            or needed["正常"] + needed["异常"] != len(fallback_letters)
        ):
            return []

        assignments: dict[str, str] = {}
        for state in ("正常", "异常"):
            matching = [
                letter
                for letter in fallback_letters
                if records[letter].state == state
            ]
            for letter in matching[: needed[state]]:
                assignments[letter] = state

        unassigned = [letter for letter in fallback_letters if letter not in assignments]
        for state in ("正常", "异常"):
            remaining = needed[state] - sum(
                assigned_state == state for assigned_state in assignments.values()
            )
            for _ in range(remaining):
                assignments[unassigned.pop(0)] = state

        changed: List[str] = []
        for letter in fallback_letters:
            record = records[letter]
            target_state = assignments[letter]
            if record.state == target_state:
                continue
            target_level = "正常"
            if target_state == "异常":
                target_level = record.level if record.level in {"偏低", "偏高"} else "偏低"
            votes = dict(record.stability_votes)
            votes["fallback_count_rebalanced"] = 1
            records[letter] = replace(
                record,
                level=target_level,
                state=target_state,
                stability_votes=votes,
            )
            changed.append(letter)
            print(
                f"[inspect] rebalance zero-confidence fallback area={letter} "
                f"state={record.state}->{target_state}; real visual records preserved"
            )
        return changed

    def _window_available(self, requested: bool) -> bool:
        if not requested:
            return False
        if os.name != "nt" and not os.environ.get("DISPLAY"):
            print("[inspect] DISPLAY is not set; inspection preview window disabled")
            return False
        return True

    def _show_inspection_preview(self, stop_name: str, frame, record: Optional[InspectionRecord]) -> None:
        if not self.show_inspection_window:
            return
        try:
            import cv2 as cv
        except Exception as exc:
            print(f"[inspect] preview window disabled: {exc}")
            self.show_inspection_window = False
            return

        preview = frame.copy()
        if not self._inspection_window_created:
            cv.namedWindow(self.INSPECTION_WINDOW_NAME, cv.WINDOW_NORMAL)
            self._inspection_window_created = True
        self._draw_inspection_overlay(preview, stop_name, record)
        cv.imshow(self.INSPECTION_WINDOW_NAME, preview)
        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            self._close_inspection_window()
            self.show_inspection_window = False

    def _draw_inspection_overlay(self, frame, stop_name: str, record: Optional[InspectionRecord]) -> None:
        import cv2 as cv

        if record is None:
            lines = [f"{stop_name}", "result: detecting..."]
            color = (0, 220, 255)
        else:
            level = {"正常": "normal", "偏低": "low", "偏高": "high"}.get(record.level, record.level)
            state = {"正常": "normal", "异常": "abnormal"}.get(record.state, record.state)
            lines = [
                f"{stop_name}",
                f"result: {record.letter} {level} {state}",
                f"confidence: {record.confidence:.2f}",
            ]
            color = (0, 255, 0) if record.state == "正常" else (0, 0, 255)
        x, y = 18, 34
        line_height = 34
        width = max(360, max(len(line) for line in lines) * 18)
        height = line_height * len(lines) + 18
        cv.rectangle(frame, (8, 8), (8 + width, 8 + height), (0, 0, 0), -1)
        cv.rectangle(frame, (8, 8), (8 + width, 8 + height), color, 2)
        for index, line in enumerate(lines):
            cv.putText(
                frame,
                line,
                (x, y + index * line_height),
                cv.FONT_HERSHEY_SIMPLEX,
                0.85,
                color if index == 1 else (255, 255, 255),
                2,
                cv.LINE_AA,
            )

    def _close_inspection_window(self) -> None:
        if not self._inspection_window_created:
            return
        try:
            import cv2 as cv

            cv.destroyWindow(self.INSPECTION_WINDOW_NAME)
        except Exception:
            pass
        self._inspection_window_created = False

    def _store_inspection_record(self, record: InspectionRecord) -> None:
        self.context.records[record.letter] = record
        self._persist_inspection_record(record)

    def _initialize_round_result(self) -> None:
        if self.context.dry_run or self._round_result_initialized:
            return
        inspection_cfg = self.config.get("inspection", {})
        if bool(inspection_cfg.get("reset_round_result_on_mission_start", True)):
            round_result_path = Path(str(inspection_cfg.get("round_result_path", "round_result.json")))
            write_empty_round_result(
                round_result_path,
                "front",
                block_reason="mission_started",
                run_id=self.context.run_id,
            )
            print(f"[inspect] reset round_result={round_result_path}")
        self._round_result_initialized = True

    def _persist_inspection_record(self, record: InspectionRecord) -> None:
        if self.context.dry_run:
            return
        self._initialize_round_result()
        inspection_cfg = self.config.get("inspection", {})
        latest_result_path = Path(str(inspection_cfg.get("latest_stop_result_path", "latest_stop_result.json")))
        round_result_path = Path(str(inspection_cfg.get("round_result_path", "round_result.json")))
        latest_data = write_latest_stop_result(latest_result_path, record)
        round_data = merge_record_into_round(
            round_result_path,
            record,
            source_camera=record.source_camera or "front",
            run_id=self.context.run_id,
        )
        print(
            "[inspect] result_json "
            f"latest={latest_result_path} round={round_result_path} "
            f"area={latest_data.get('letter')} ready={round_data.get('ready')} "
            f"unknown={round_data.get('unknown_areas')}"
        )

    def _persist_round_block(self, block_reason: str) -> None:
        if self.context.dry_run:
            return
        inspection_cfg = self.config.get("inspection", {})
        round_result_path = Path(str(inspection_cfg.get("round_result_path", "round_result.json")))
        data = build_round_result(
            self.context.records,
            source_camera="front",
            block_reason=block_reason,
            run_id=self.context.run_id,
        )
        write_json_atomic(round_result_path, data)
        print(f"[inspect] round blocked path={round_result_path} reason={block_reason}")

    def _save_inspection_evidence(self, stop_name: str, letter: str, frame) -> Optional[Path]:
        import cv2 as cv

        evidence_dir = Path(str(self.config.get("inspection", {}).get("evidence_dir", "evidence")))
        evidence_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = evidence_dir / f"{timestamp}_{stop_name}_{letter}.png"
        if not cv.imwrite(str(path), frame):
            print(f"[inspect] failed to save evidence image: {path}")
            return None
        print(f"[inspect] evidence_image={path}")
        return path

    def _save_inspection_diagnostics(
        self,
        record: InspectionRecord,
        evidence_path: Path,
    ) -> Optional[Path]:
        getter = getattr(self.vision, "inspection_diagnostics", None)
        if not callable(getter):
            return None
        diagnostics = getter(record.frame_id)
        if not isinstance(diagnostics, dict):
            return None
        diagnostics["accepted_record"] = {
            "letter": record.letter,
            "level": record.level,
            "state": record.state,
            "confidence_is_letter_confidence": record.confidence,
            "stability_votes": dict(record.stability_votes),
        }
        path = evidence_path.with_suffix(".json")
        try:
            serializable = json.loads(
                json.dumps(diagnostics, ensure_ascii=False, default=str)
            )
            write_json_atomic(path, serializable)
        except Exception as exc:
            print(f"[inspect] diagnostics save warning path={path}: {exc}")
            return None
        print(f"[inspect] diagnostics_json={path}")
        return path

    def _announce_record(self, record: InspectionRecord) -> None:
        announcement = build_announcement(record.letter, record.level, record.state)
        if announcement is not None:
            print(f"[mission] 播报内容: {announcement}")
        else:
            print(
                "[mission] 播报内容生成失败: "
                f"letter={record.letter} level={record.level} state={record.state}"
            )
        self.audio.say_record(record)
        self.context.reported_letters.append(record.letter)

    def _estimate_red_bar_distance_mm(self) -> float:
        if self.context.dry_run:
            return 260.0
        self.arm_camera.open()
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            frame = self.arm_camera.read()
            if frame is None:
                continue
            det = self.vision.best_red_bar(frame)
            if det and det.bbox:
                return max(180.0, min(420.0, 50000.0 / max(1, det.bbox.w)))
        return 280.0

    def _pick_target(self, target_letter: str) -> bool:
        second_pickup = bool(self.context.placed_letters)
        pickup_route = "pickup_from_place" if second_pickup else "pickup_from_upper_inspection"
        if self.context.pickup_target_letter != target_letter:
            if self.context.carried_bar:
                raise MissionAbort(
                    "cannot start a new pickup while an object is held"
                )
            if (
                self.context.placement_target_letter is not None
                and self.context.placement_stage != "complete"
            ):
                raise MissionAbort(
                    "cannot start a new pickup while placement recovery is pending"
                )
            self.context.placement_target_letter = None
            self.context.placement_route_action_index = 0
            self.context.placement_stage = "idle"
            self.context.placement_visual_approach_complete = False
            self.context.placement_ultrasound_approach_complete = False
            self.context.placement_letter_centered_complete = False
            self.context.placement_search_front_target_m = None
            self.context.placement_post_forward_yaw_attempted = False
            self.context.placement_post_forward_yaw_ok = False
            self.context.placement_forced_forward_progress_m = 0.0
            self.context.placement_forced_forward_odom_m = 0.0
            self.context.placement_navigation_net_lateral_m = 0.0
            self.context.placement_navigation_lateral_travel_m = 0.0
            self.context.placement_navigation_search_phase = "left"
            self.context.placement_navigation_last_lateral_sign = None
            self.context.placement_navigation_last_geometry_sign = None
            self.context.placement_navigation_pending_recovery_sign = None
            self.context.placement_navigation_zero_progress_count = 0
            self.context.placement_final_approach_complete = False
            self.context.placement_final_approach_progress_m = 0.0
            self.context.placement_legacy_offset_m = 0.0
            self.context.pickup_target_letter = target_letter
            self.context.pickup_route_name = pickup_route
            self.context.pickup_route_action_index = 0
            self.context.pickup_stage = "pre_route"
            self.context.pickup_pregrasp_substage = "idle"
            self.context.pickup_retreat_progress_m = 0.0
            if not second_pickup:
                self.context.pickup_entry_strafe_progress_m = 0.0
                self.context.pickup_entry_tag_acquired = False
                self.context.pickup_search_origin_pose = None
            self.context.target_letter = target_letter
            self.context.carried_bar = False
        elif self.context.pickup_route_name != pickup_route:
            raise MissionAbort(
                "pickup checkpoint route does not match current mission cycle"
            )

        stage = self.context.pickup_stage
        print(
            f"[mission] pickup target letter={target_letter} "
            f"resume_stage={stage}"
        )
        if stage == "pre_route":
            self.motion.stop()
            self._require_arm_result(self.arm.stow(), "pre-pick moving pose")
            self.context.pickup_stage = "route"
            stage = "route"

        if stage == "route":
            if not self._run_resumable_scripted_route(
                pickup_route,
                progress_attr="pickup_route_action_index",
            ):
                self._drive_segment("pickup", distance_m=0.8)
            self.context.pickup_stage = "arrival_alignment"
            stage = "arrival_alignment"

        if stage == "arrival_alignment" and second_pickup:
            if self._pickup_transfer_enabled():
                print(
                    "[pickup-transfer] returned to the previous pickup lane; "
                    "resume red-target search without approaching the box"
                )
            else:
                self._try_align_second_pickup_box_center()
        if stage == "arrival_alignment":
            self.context.pickup_stage = "pregrasp"
            stage = "pregrasp"

        if stage == "pregrasp":
            self.motion.stop()
            try:
                aligned = self._run_pregrasp_base_sequence()
            except Exception as exc:
                print(f"[pregrasp] base sequence failed: {exc}")
                aligned = False
            finally:
                self.motion.stop()
            if not aligned:
                return False
            self.context.pickup_stage = "grasp_ready"
            stage = "grasp_ready"

        if stage == "grasp_ready":
            self.motion.stop()
            self._require_arm_result(self.arm.camera_pose(), "grasp-ready")
            self._settle_after_pregrasp_stop()
            self.context.pickup_stage = "grasp"
            stage = "grasp"

        if stage == "grasp":
            distance_mm = (
                260.0
                if getattr(self.arm, "backend", "runtime") == "runtime"
                else self._estimate_red_bar_distance_mm()
            )
            try:
                self.context.carried_bar = self._retry_grasp(distance_mm)
            except Exception:
                if self.context.carried_bar:
                    self.context.pickup_stage = "retreat"
                raise
            if not self.context.carried_bar:
                self.context.pickup_stage = "pregrasp"
                self.context.pickup_pregrasp_substage = "target_acquisition"
                return False
            self.context.pickup_stage = "retreat"
            stage = "retreat"

        if stage == "retreat" and self._pickup_transfer_enabled():
            self._retreat_from_pickup_box()
        if stage == "retreat":
            self.context.pickup_stage = "departure_yaw"
            stage = "departure_yaw"

        if stage == "departure_yaw" and self._pickup_transfer_enabled():
            self._align_pickup_departure_yaw()
        if stage == "departure_yaw":
            self.context.pickup_stage = "complete"
            stage = "complete"
            print(
                "[pickup-transfer] pickup complete; transfer directly to placement"
            )

        if stage == "departure_center":
            print(
                "[pickup-transfer] skip legacy departure-center checkpoint; "
                "transfer directly to placement"
            )
            self.context.pickup_stage = "complete"

        return self.context.carried_bar

    def _settle_after_pregrasp_stop(self) -> None:
        settle_seconds = max(
            0.0,
            float(
                self.config.get("pregrasp_red_align", {}).get(
                    "post_stop_settle_seconds",
                    0.5,
                )
            ),
        )
        if settle_seconds > 0.0:
            time.sleep(0.0 if self.context.dry_run else settle_seconds)

    def _run_pregrasp_base_sequence(self) -> bool:
        transfer = self.config.get("pickup_transfer", {})
        keep_pre_retreat_yaw = (
            not bool(transfer.get("enabled", False))
            or bool(transfer.get("pre_retreat_yaw_alignment_enabled", True))
        )
        stages = []
        if keep_pre_retreat_yaw:
            stages.append(
                (
                    "initial_yaw",
                    "initial yaw alignment",
                    lambda: self._run_pregrasp_red_alignment(wide_only=True),
                )
            )
        stages.extend(
            [
                (
                    "box_approach",
                    "ultrasound approach to 28cm before target search",
                    self._run_pregrasp_box_approach,
                ),
                (
                    "target_acquisition",
                    "moving-pose red block search and fine lateral alignment",
                    lambda: self._run_pregrasp_red_alignment(
                        skip_wide_parallel=True,
                        acquire_only=True,
                    ),
                ),
            ]
        )
        if keep_pre_retreat_yaw:
            stages.append(
                (
                    "post_lateral_yaw",
                    "post-lateral yaw alignment",
                    lambda: self._run_pregrasp_red_alignment(wide_only=True),
                )
            )
        stages.append(
            (
                "final_distance",
                "final distance check below 30cm",
                self._confirm_pregrasp_box_distance,
            )
        )
        stage_keys = [stage[0] for stage in stages]
        substage = self.context.pickup_pregrasp_substage
        if substage == "lateral_alignment":
            # Resume checkpoints written by the previous two-aligner flow.
            substage = "target_acquisition"
            self.context.pickup_pregrasp_substage = substage
        if substage == "complete":
            return True
        if substage == "idle":
            start_index = 0
            self.context.pickup_pregrasp_substage = stage_keys[0]
        elif substage in stage_keys:
            start_index = stage_keys.index(substage)
        else:
            raise MissionAbort(
                f"invalid pickup pregrasp checkpoint: {substage!r}"
            )
        for stage_index in range(start_index, len(stages)):
            stage_key, stage_name, operation = stages[stage_index]
            self.context.pickup_pregrasp_substage = stage_key
            print(f"[pregrasp] stage={stage_name}")
            if not operation():
                print(f"[pregrasp] stage failed: {stage_name}")
                return False
            self.motion.stop()
            self.context.pickup_pregrasp_substage = (
                stages[stage_index + 1][0]
                if stage_index + 1 < len(stages)
                else "complete"
            )
        return True

    def _run_pregrasp_box_approach(self) -> bool:
        approach = BoxApproachController(ApproachConfig())
        if self.context.dry_run:
            print("[approach-box] dry-run target confirmed at 28cm")
            return True

        command_period_s = 1.0 / max(1.0, self.motion.limits.command_hz)
        started_at = time.monotonic()
        last_sample_at: Optional[float] = None
        self._controlled_box_approach_active = True
        try:
            while True:
                state = self.state_reader.poll()
                sample_at = state.ultrasound_updated_at
                sample_age_s = self.state_reader.sample_age(sample_at)
                approach.validate_runtime(
                    sample_age_s=sample_age_s,
                    elapsed_s=time.monotonic() - started_at,
                )
                if sample_at is None or sample_at == last_sample_at:
                    time.sleep(min(0.02, command_period_s))
                    continue

                error = self.state_reader.safety_error(
                    require_ultrasound=True,
                    require_fresh=True,
                )
                if error:
                    raise MissionAbort(f"box approach state rejected: {error}")
                last_sample_at = sample_at
                distance_m = float(state.front_ultrasound_m)
                decision = approach.decide(distance_m)
                print(
                    f"[approach-box] distance={distance_m * 100.0:.1f}cm "
                    f"mode={decision.mode} vx={decision.vx:.3f}"
                )

                if decision.vx == 0.0:
                    self.motion.stop()
                    if not decision.reached:
                        self._poll_pregrasp_ultrasound(
                            approach.config.settle_s,
                        )
                elif decision.mode == "continuous":
                    self._motion_guard(decision.vx, 0.0, 0.0)
                    self.motion.move(decision.vx, 0.0, 0.0)
                    time.sleep(command_period_s)
                else:
                    self.motion.hold_velocity(
                        decision.vx,
                        0.0,
                        0.0,
                        float(decision.drive_duration_s or 0.0),
                    )
                    self._poll_pregrasp_ultrasound(
                        decision.settle_duration_s,
                    )

                if decision.reached:
                    print(
                        "[approach-box] target confirmed at "
                        f"{distance_m * 100.0:.1f}cm"
                    )
                    return True
        except (TimeoutError, ValueError) as exc:
            print(f"[approach-box] failed: {exc}")
            return False
        finally:
            self._controlled_box_approach_active = False
            self.motion.stop()

    def _poll_pregrasp_ultrasound(self, duration_s: float) -> None:
        deadline = time.monotonic() + max(0.0, float(duration_s))
        while time.monotonic() < deadline:
            error = self.state_reader.safety_error(
                require_ultrasound=True,
                require_fresh=True,
            )
            if error:
                raise MissionAbort(f"box approach settle rejected: {error}")
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

    def _confirm_pregrasp_box_distance(self) -> bool:
        approach_config = ApproachConfig()
        pregrasp_config = self.config.get("pregrasp_red_align", {})
        threshold_m = float(
            pregrasp_config.get("final_distance_max_m", 0.30)
        )
        minimum_m = float(
            pregrasp_config.get("final_distance_min_m", 0.25)
        )
        max_attempts = int(pregrasp_config.get("final_distance_attempts", 3))
        if self.context.dry_run:
            print(
                "[approach-box] dry-run final distance check; if needed, "
                f"up to {max_attempts - 1} corrections to "
                f"{approach_config.target_distance_m * 100.0:.1f}cm across "
                f"{max_attempts} attempts, then require "
                f"{minimum_m * 100.0:.1f}cm <= distance < "
                f"{threshold_m * 100.0:.1f}cm"
            )
            return True

        self.motion.stop()
        state = self.state_reader.poll()
        error = self.state_reader.safety_error(
            require_ultrasound=True,
            require_fresh=True,
        )
        if error:
            print(f"[approach-box] final distance check rejected: {error}")
            return False

        distance_m = float(state.front_ultrasound_m)
        sample_at = getattr(state, "ultrasound_updated_at", None)
        correction_speed_mps = approach_config.near_speed_mps
        for attempt in range(1, max_attempts + 1):
            confirmed = minimum_m <= distance_m < threshold_m
            print(
                f"[approach-box] final distance attempt={attempt}/{max_attempts} "
                f"distance={distance_m * 100.0:.1f}cm "
                f"range=[{minimum_m * 100.0:.1f}, "
                f"{threshold_m * 100.0:.1f})cm confirmed={confirmed}"
            )
            if confirmed:
                return True
            if attempt >= max_attempts:
                break

            correction_distance_m = distance_m - approach_config.target_distance_m
            correction_vx = math.copysign(
                correction_speed_mps,
                correction_distance_m,
            )
            correction_duration_s = (
                abs(correction_distance_m) / correction_speed_mps
            )
            print(
                f"[approach-box] correction={attempt}/{max_attempts - 1} "
                f"distance={correction_distance_m * 100.0:.1f}cm "
                f"target={approach_config.target_distance_m * 100.0:.1f}cm "
                f"vx={correction_vx:.3f} "
                f"duration={correction_duration_s:.2f}s"
            )

            self._controlled_box_approach_active = True
            try:
                self._motion_guard(correction_vx, 0.0, 0.0)
                self.motion.hold_velocity(
                    correction_vx,
                    0.0,
                    0.0,
                    correction_duration_s,
                )
            finally:
                self._controlled_box_approach_active = False
                self.motion.stop()

            new_sample = self._wait_for_new_pregrasp_ultrasound(sample_at)
            if new_sample is None:
                return False
            distance_m, sample_at = new_sample

        print(
            f"[approach-box] final distance failed after {max_attempts} attempts; "
            "no more corrections"
        )
        return False

    def _wait_for_new_pregrasp_ultrasound(
        self,
        previous_sample_at: Optional[float],
        timeout_s: float = 1.0,
    ) -> Optional[tuple[float, float]]:
        deadline = time.monotonic() + max(0.1, float(timeout_s))
        while time.monotonic() < deadline:
            state = self.state_reader.poll()
            sample_at = getattr(state, "ultrasound_updated_at", None)
            if sample_at is None or sample_at == previous_sample_at:
                time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))
                continue
            error = self.state_reader.safety_error(
                require_ultrasound=True,
                require_fresh=True,
            )
            if error:
                print(
                    "[approach-box] post-correction distance rejected: "
                    f"{error}"
                )
                return None
            return float(state.front_ultrasound_m), float(sample_at)
        print("[approach-box] post-correction ultrasound sample timed out")
        return None

    def _pregrasp_ultrasound_ready(self) -> bool:
        gate = self.config.get("pregrasp_red_align", {})
        if not bool(gate.get("ultrasound_gate_enabled", True)):
            print("[pregrasp] ultrasound safety gate disabled")
            return True
        if self.context.dry_run:
            print("[pregrasp] dry-run ultrasound safety gate")
            return True

        error = self.state_reader.safety_error(
            require_ultrasound=True,
            require_fresh=True,
        )
        if error:
            print(f"[pregrasp] ultrasound safety gate rejected stale/invalid state: {error}")
            return False

        try:
            minimum = float(gate.get("ultrasound_min_m", 0.10))
            maximum = float(gate.get("ultrasound_max_m", 2.0))
            value = float(self.state_reader.state.front_ultrasound_m)
        except (TypeError, ValueError):
            print("[pregrasp] ultrasound safety gate rejected non-numeric configuration/sample")
            return False
        if (
            not math.isfinite(minimum)
            or not math.isfinite(maximum)
            or minimum < 0.0
            or maximum <= minimum
            or not math.isfinite(value)
        ):
            print(
                "[pregrasp] ultrasound safety gate rejected invalid range/sample: "
                f"value={value!r} range=({minimum!r}, {maximum!r})"
            )
            return False
        if value < minimum or value > maximum:
            print(
                "[pregrasp] ultrasound safety gate rejected coarse front range: "
                f"{value:.3f}m not in [{minimum:.3f}, {maximum:.3f}]m"
            )
            return False
        print(
            "[pregrasp] ultrasound coarse safety range accepted: "
            f"{value:.3f}m (not used as arm target depth)"
        )
        return True

    def _pregrasp_front_distance_sample(
        self,
    ) -> Optional[tuple[float, float]]:
        state = self.state_reader.poll()
        value = getattr(state, "front_ultrasound_m", None)
        updated_at = getattr(state, "ultrasound_updated_at", None)
        try:
            distance_m = float(value)
            sample_at = float(updated_at)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(distance_m) or not math.isfinite(sample_at):
            return None
        max_age_s = float(
            self.config.get("safety", {}).get("state_max_age_s", 0.75)
        )
        if time.monotonic() - sample_at > max_age_s:
            return None
        return distance_m, sample_at

    def _run_pregrasp_red_alignment(
        self,
        *,
        wide_only: bool = False,
        skip_wide_parallel: bool = False,
        acquire_only: bool = False,
    ) -> bool:
        if wide_only and skip_wide_parallel:
            raise ValueError("wide_only and skip_wide_parallel are mutually exclusive")
        if wide_only and acquire_only:
            raise ValueError("wide_only and acquire_only are mutually exclusive")
        align_config = self.config.get("pregrasp_red_align", {})
        if not bool(align_config.get("enabled", True)):
            print("[pregrasp] lateral alignment disabled")
            return True
        wide_config = align_config.get("wide_parallel", {})
        if bool(wide_config.get("enabled", True)) and not skip_wide_parallel:
            if self.context.dry_run:
                print("[pregrasp] dry-run wide-camera box parallel alignment")
            else:
                project_root = Path(__file__).resolve().parent.parent

                def resolve_wide_path(value: object) -> Path:
                    path = Path(str(value))
                    return path if path.is_absolute() else project_root / path

                if self.pregrasp_box_aligner is None:
                    calibration_path = resolve_wide_path(
                        self.config["camera"]["wide_calibration"]
                    )
                    self.pregrasp_box_aligner = WideBoxAligner(
                        camera=self.wide_camera,
                        undistorter=WideCameraUndistorter.from_file(
                            calibration_path
                        ),
                        motion=self.motion,
                        config=wide_config,
                    )
                box_result = self.pregrasp_box_aligner.run()
                print(
                    "[pregrasp] wide-camera parallel alignment "
                    f"ok={box_result.ok} reason={box_result.reason} "
                    f"error={box_result.final_error_deg}"
                )
                if not box_result.ok:
                    print(
                        "[pregrasp] yaw correction failed; skip correction "
                        f"reason={box_result.reason}"
                    )
        if wide_only:
            print("[pregrasp] wide-camera-only alignment complete")
            return True
        aligner = self.pregrasp_aligner
        if aligner is None:
            if self.context.dry_run:
                stage = "target acquisition" if acquire_only else "lateral alignment"
                print(f"[pregrasp] dry-run {stage}")
                return True
            project_root = Path(__file__).resolve().parent.parent

            def resolve_path(value: object) -> Path:
                path = Path(str(value))
                return path if path.is_absolute() else project_root / path

            arm_config = self.config.get("arm", {})
            observer = ArmRedObserver.from_files(
                resolve_path(
                    arm_config.get(
                        "runtime_config",
                        "mission_lite3/arm/runtime/strip_detector_grasp_config.json",
                    )
                ),
                resolve_path(
                    arm_config.get(
                        "calibration",
                        "mission_lite3/arm/runtime/camera_calibration.json",
                    )
                ),
            )
            runtime_align_config = dict(align_config)
            runtime_align_config["acquire_only"] = bool(acquire_only)
            if acquire_only:
                runtime_align_config["max_strafe_distance_m"] = min(
                    float(runtime_align_config["max_strafe_distance_m"]),
                    float(
                        align_config.get(
                            "acquire_fine_max_strafe_distance_m",
                            0.05,
                        )
                    ),
                )
            runtime_align_config["pickup_tag_boundary"] = dict(
                self.config.get("pickup_tag_boundary", {})
            )
            runtime_align_config["target_search_min_distance_m"] = (
                float(
                    align_config.get(
                        "preapproach_search_min_distance_m",
                        0.0,
                    )
                )
                if acquire_only
                else 0.0
            )
            aligner = PregraspRedAligner(
                camera=self.arm_camera,
                observer=observer,
                motion=self.motion,
                config=runtime_align_config,
                pose_provider=self.state_reader.pose,
                search_origin_pose=self.context.pickup_search_origin_pose,
                tag_boundary_provider=self._pickup_tag_centers,
                front_distance_provider=self._pregrasp_front_distance_sample,
                log_dir=resolve_path(
                    align_config.get(
                        "run_log_dir",
                        "pregrasp_align_runs",
                    )
                ),
            )
        previous_controlled = self._controlled_box_approach_active
        if acquire_only:
            self._controlled_box_approach_active = True
        try:
            result = aligner.run()
        finally:
            self._controlled_box_approach_active = previous_controlled
            self.pregrasp_aligner = None
        print(
            "[pregrasp] "
            + (
                "target acquisition and alignment "
                if acquire_only
                else "lateral alignment "
            )
            + f"ok={result.ok} reason={result.reason}"
        )
        return bool(result.ok)

    def _place_carried_bar(self) -> bool:
        start_new_placement = (
            self.context.carried_bar
            and self.context.target_letter is not None
            and (
                self.context.placement_target_letter is None
                or self.context.placement_stage == "complete"
            )
        )
        if start_new_placement:
            self.context.placement_target_letter = self.context.target_letter
            self.context.placement_route_action_index = 0
            self.context.placement_stage = "route"
            self.context.placement_visual_approach_complete = False
            self.context.placement_ultrasound_approach_complete = False
            self.context.placement_letter_centered_complete = False
            self.context.placement_search_front_target_m = None
            self.context.placement_post_forward_yaw_attempted = False
            self.context.placement_post_forward_yaw_ok = False
            self.context.placement_forced_forward_progress_m = 0.0
            self.context.placement_forced_forward_odom_m = 0.0
            self.context.placement_navigation_net_lateral_m = 0.0
            self.context.placement_navigation_lateral_travel_m = 0.0
            self.context.placement_navigation_search_phase = "left"
            self.context.placement_navigation_last_lateral_sign = None
            self.context.placement_navigation_last_geometry_sign = None
            self.context.placement_navigation_pending_recovery_sign = None
            self.context.placement_navigation_zero_progress_count = 0
            self.context.placement_final_approach_complete = False
            self.context.placement_final_approach_progress_m = 0.0
            self.context.placement_legacy_offset_m = 0.0
        elif self.context.placement_target_letter is None:
            if not self.context.carried_bar or self.context.target_letter is None:
                return False
            self.context.placement_target_letter = self.context.target_letter
            self.context.placement_route_action_index = 0
            self.context.placement_stage = "route"
            self.context.placement_visual_approach_complete = False
            self.context.placement_ultrasound_approach_complete = False
            self.context.placement_letter_centered_complete = False
            self.context.placement_search_front_target_m = None
            self.context.placement_post_forward_yaw_attempted = False
            self.context.placement_post_forward_yaw_ok = False
            self.context.placement_forced_forward_progress_m = 0.0
            self.context.placement_forced_forward_odom_m = 0.0
            self.context.placement_navigation_net_lateral_m = 0.0
            self.context.placement_navigation_lateral_travel_m = 0.0
            self.context.placement_navigation_search_phase = "left"
            self.context.placement_navigation_last_lateral_sign = None
            self.context.placement_navigation_last_geometry_sign = None
            self.context.placement_navigation_pending_recovery_sign = None
            self.context.placement_navigation_zero_progress_count = 0
            self.context.placement_final_approach_complete = False
            self.context.placement_final_approach_progress_m = 0.0
            self.context.placement_legacy_offset_m = 0.0
        target_letter = self.context.placement_target_letter
        if (
            self.context.target_letter not in (None, target_letter)
            or target_letter is None
        ):
            raise MissionAbort("placement checkpoint target mismatch")
        stage = self.context.placement_stage
        if (
            not self.context.carried_bar
            and stage not in {"post_place_stow", "pause", "restore_offset", "complete"}
        ):
            return False
        pickup_transfer_enabled = self._pickup_transfer_enabled()
        print(
            f"[mission] place carried bar to letter={target_letter} "
            f"resume_stage={stage}"
        )

        if stage == "route":
            self._placement_route_active = True
            self._placement_letter_approach_succeeded = (
                self.context.placement_visual_approach_complete
            )
            try:
                route_completed = self._run_placement_route(pickup_transfer_enabled)
                if not route_completed:
                    if pickup_transfer_enabled:
                        raise MissionAbort("visual placement route is unavailable")
                    self._drive_segment("place", distance_m=1.0)
                if (
                    pickup_transfer_enabled
                    and not self.context.placement_visual_approach_complete
                ):
                    raise MissionAbort("visual placement route is unavailable")
            finally:
                self._placement_route_active = False
            self.context.placement_stage = "align_box"
            stage = "align_box"

        if stage == "align_box":
            self.motion.stop()
            self.context.placement_legacy_offset_m = (
                0.0
                if pickup_transfer_enabled
                else self._align_to_letter_box(target_letter)
            )
            self.context.placement_stage = "release"
            stage = "release"

        if stage == "release":
            result = self.arm.place_to_box(target_letter)
            released = (
                result.released and not result.object_held
                if isinstance(result, ArmTaskResult)
                else result is not False
            )
            if not released:
                detail = (
                    result.reason
                    if isinstance(result, ArmTaskResult)
                    else "place command failed"
                )
                if isinstance(result, ArmTaskResult) and result.requires_power_cycle:
                    detail = f"{detail}; arm power cycle required"
                print(
                    "[mission] place failed; preserve release checkpoint: "
                    f"{detail}"
                )
                return False
            if isinstance(result, ArmTaskResult) and not result.ok:
                print(
                    "[mission] release verified despite post-release error; "
                    f"continue recovery: {result.reason}"
                )
            if target_letter not in self.context.placed_letters:
                self.context.placed_letters.append(target_letter)
            self.context.carried_bar = False
            self.context.placement_stage = "post_place_stow"
            stage = "post_place_stow"

        if stage == "post_place_stow":
            self.motion.stop()
            self._require_arm_result(
                self.arm.stow(),
                "post-place moving pose",
            )
            self.context.placement_stage = "pause"
            stage = "pause"

        if stage == "pause":
            pause_seconds = float(
                self.config.get("inspection", {}).get(
                    "place_pause_seconds",
                    3.0,
                )
            )
            if pause_seconds > 0:
                print(f"[mission] place pause {pause_seconds:.1f}s")
                time.sleep(0.2 if self.context.dry_run else pause_seconds)
            self.context.placement_stage = "restore_offset"
            stage = "restore_offset"

        if stage == "restore_offset":
            offset = self.context.placement_legacy_offset_m
            if abs(offset) > 1e-6:
                strafe_distance_for_box_center(
                    self.motion,
                    -offset,
                    self.config.get("box_center_alignment", {}),
                )
                self.context.placement_legacy_offset_m = 0.0
            self.context.placement_stage = "complete"

        self.context.target_letter = None
        return True

    def _run_placement_route(self, pickup_transfer_enabled: bool) -> bool:
        route_cfg = self.config.get("scripted_route", {})
        actions = route_cfg.get("place_from_pickup")
        if pickup_transfer_enabled or not self._route_has_visual_placement(actions):
            return self._run_resumable_scripted_route(
                "place_from_pickup",
                progress_attr="placement_route_action_index",
            )

        print("[nav] scripted route place_from_pickup (legacy placement mode)")
        legacy_actions = (
            {"action": "turn", "yaw_rad": -3.1416},
            {"action": "placement_row_yaw_align"},
            {"action": "placement_lane_strafe"},
            {"action": "forward", "distance_m": 1.38},
        )
        action_index = self.context.placement_route_action_index
        while action_index < len(legacy_actions):
            self._execute_route_action(legacy_actions[action_index])
            action_index += 1
            self.context.placement_route_action_index = action_index
        return True

    @staticmethod
    def _route_has_visual_placement(actions: Any) -> bool:
        return isinstance(actions, list) and any(
            isinstance(action, dict)
            and str(action.get("action", "")).strip()
            == "placement_letter_approach"
            for action in actions
        )

    def _align_to_letter_box(self, target_letter: str) -> float:
        center_config = self.config.get("box_center_alignment", {})
        if bool(center_config.get("enabled", False)):
            result = self._run_box_center_alignment("placement", target_letter)
            print(
                "[box-center] placement alignment "
                f"ok={result.ok} reason={result.reason} "
                f"visual_strafe={result.visual_strafe_m:.3f}m "
                f"net_strafe={result.net_strafe_m:.3f}m"
            )
            if result.ok:
                return float(result.net_strafe_m)
            if not result.rollback_ok:
                raise MissionAbort(
                    "placement box-center alignment failed and visual strafe "
                    f"rollback was not verified: {result.reason}"
                )
            print(
                "[box-center] placement recognition failed after visual "
                "rollback; use fixed fallback"
            )
        if not bool(center_config.get("fallback_enabled", True)):
            raise MissionAbort(
                f"placement box-center unavailable and fallback disabled: {target_letter}"
            )
        fallback = center_config.get("fallback_offsets_m", {})
        route_cfg = self.config.get("scripted_route", {})
        offsets = fallback if isinstance(fallback, dict) else route_cfg.get("placement_letter_strafe_m", {})
        if not isinstance(offsets, dict):
            return 0.0
        offset = float(offsets.get(target_letter, 0.0) or 0.0)
        if abs(offset) > 1e-6:
            print(
                f"[nav] fixed fallback align to letter box {target_letter}: "
                f"strafe {offset:.2f}m"
            )
            strafe_distance_for_box_center(
                self.motion,
                offset,
                center_config,
            )
        return offset

    def _run_box_center_alignment(
        self,
        mode: str,
        target_letter: Optional[str] = None,
        *,
        tolerance_fraction: Optional[float] = None,
    ) -> BoxCenterAlignmentResult:
        if self.box_center_aligner is None:
            project_root = Path(__file__).resolve().parent.parent
            calibration_path = Path(str(self.config["camera"]["wide_calibration"]))
            if not calibration_path.is_absolute():
                calibration_path = project_root / calibration_path
            center_config = dict(self.config.get("box_center_alignment", {}))
            if mode == "pickup" and self._pickup_transfer_enabled():
                center_config["enabled"] = True
            for key in ("alignment_run_log_dir",):
                path = Path(str(center_config.get(key, "box_center_alignment_runs")))
                if not path.is_absolute():
                    center_config[key] = str(project_root / path)
            self.box_center_aligner = BoxCenterAligner(
                camera=self.wide_camera,
                undistorter=WideCameraUndistorter.from_file(calibration_path),
                motion=self.motion,
                config=center_config,
            )
        return self.box_center_aligner.run(
            mode,
            target_letter,
            tolerance_fraction=tolerance_fraction,
        )

    def _pickup_transfer_enabled(self) -> bool:
        return bool(self.config.get("pickup_transfer", {}).get("enabled", False))

    def _get_pickup_transfer_controller(self) -> PickupTransferController:
        if self.pickup_transfer_controller is None:
            self.pickup_transfer_controller = PickupTransferController(
                self.motion,
                self.state_reader,
                self.config.get("pickup_transfer", {}),
                dry_run=self.context.dry_run,
            )
        return self.pickup_transfer_controller

    def _retreat_from_pickup_box(self) -> None:
        result = self._get_pickup_transfer_controller().retreat_to_front_distance(
            initial_odom_retreat_m=self.context.pickup_retreat_progress_m,
        )
        self.context.pickup_retreat_progress_m = max(
            self.context.pickup_retreat_progress_m,
            result.odom_retreat_m,
        )
        print(
            "[pickup-transfer] retreat "
            f"ok={result.ok} reason={result.reason} "
            f"front={result.start_front_m}->{result.final_front_m} "
            f"odom={result.odom_retreat_m:.3f}m"
        )
        if not result.ok:
            raise MissionAbort(f"pickup retreat failed: {result.reason}")

    def _align_pickup_departure_yaw(self) -> bool:
        transfer = self.config.get("pickup_transfer", {})
        if not bool(transfer.get("post_retreat_yaw_alignment_enabled", True)):
            print("[pickup-transfer] post-retreat pickup-box yaw alignment disabled")
            return True
        self.motion.stop()
        try:
            aligned = self._run_pregrasp_red_alignment(wide_only=True)
        except Exception as exc:
            print(
                "[pickup-transfer] warning: post-retreat pickup-box yaw alignment "
                f"raised {exc}; continue direct placement transfer"
            )
            return False
        finally:
            self.motion.stop()
        if not aligned:
            print(
                "[pickup-transfer] warning: post-retreat pickup-box yaw alignment "
                "unavailable; continue direct placement transfer"
            )
        return bool(aligned)

    def _align_pickup_box_center_strict(
        self,
        *,
        stage: str,
        tolerance_fraction: float,
    ) -> None:
        if self.context.dry_run:
            print(
                f"[box-center] dry-run pickup stage={stage} "
                f"tolerance={tolerance_fraction:.3f}"
            )
            return
        try:
            result = self._run_box_center_alignment(
                "pickup",
                tolerance_fraction=tolerance_fraction,
            )
        except Exception as exc:
            raise MissionAbort(
                f"pickup box-center alignment raised at {stage}: {exc}"
            ) from exc
        print(
            f"[box-center] pickup stage={stage} ok={result.ok} "
            f"reason={result.reason} tolerance={tolerance_fraction:.3f} "
            f"net_strafe={result.net_strafe_m:.3f}m"
        )
        if not result.ok:
            rollback = "verified" if result.rollback_ok else "failed"
            raise MissionAbort(
                f"pickup box-center alignment failed at {stage}: "
                f"{result.reason}; rollback={rollback}"
            )

    def _placement_letter_navigation_config(
        self,
    ) -> PlacementLetterNavigationConfig:
        configured = self.config["placement_letter_navigation"]
        search_step_m = float(configured["search_step_m"])
        return PlacementLetterNavigationConfig(
            letter_order=tuple(str(value) for value in configured["letter_order"]),
            min_confidence=float(configured["letter_min_confidence"]),
            forward_speed_mps=float(configured["forward_speed_mps"]),
            lateral_speed_mps=float(configured["lateral_speed_mps"]),
            front_stop_distance_m=float(configured["front_stop_distance_m"]),
            forward_budget_m=float(configured["forward_budget_m"]),
            forward_step_m=search_step_m,
            lateral_search_step_m=search_step_m,
            min_center_correction_m=float(
                configured["min_center_correction_m"]
            ),
            max_center_correction_m=float(
                configured["max_center_correction_m"]
            ),
            center_gain_m_per_fraction=float(
                configured["center_gain_m_per_fraction"]
            ),
            max_lateral_search_m=float(configured["max_lateral_search_m"]),
            bilateral_search_enabled=bool(
                configured["bilateral_search_enabled"]
            ),
            lateral_search_each_side_m=float(
                configured["lateral_search_each_side_m"]
            ),
            immediate_complete_on_target_detection=bool(
                configured["immediate_complete_on_target_detection"]
            ),
            acquisition_center_band=tuple(
                float(value) for value in configured["acquisition_center_band"]
            ),
            center_tolerance_fraction=float(
                configured["center_tolerance_fraction"]
            ),
            final_approach_distance_m=float(
                configured["final_approach_distance_m"]
            ),
            final_approach_step_m=float(
                configured["final_approach_step_m"]
            ),
            letter_spacing_m=float(configured["letter_spacing_m"]),
            max_anchor_jump_m=float(configured["max_anchor_jump_m"]),
            target_vote_window=int(configured["target_vote_window"]),
            target_min_votes=int(configured["target_min_votes"]),
            target_memory_max_misses=int(
                configured["target_memory_max_misses"]
            ),
            target_memory_max_lateral_m=float(
                configured["target_memory_max_lateral_m"]
            ),
            target_memory_max_forward_m=float(
                configured["target_memory_max_forward_m"]
            ),
            target_memory_fraction_per_m=float(
                configured["target_memory_fraction_per_m"]
            ),
            required_center_frames=int(configured["required_center_frames"]),
            capture_retries=int(configured["capture_retries"]),
            strafe_min_progress_m=float(
                configured.get("motion_stall_min_progress_m", 0.01)
            ),
            strafe_zero_progress_reverse_count=int(
                configured.get("strafe_zero_progress_reverse_count", 2)
            ),
            image_timeout_s=float(configured["image_timeout_s"]),
            total_timeout_s=float(configured["total_timeout_s"]),
        )

    def _placement_navigation_undistorter(self) -> WideCameraUndistorter:
        if self._placement_undistorter is None:
            project_root = Path(__file__).resolve().parent.parent
            calibration_path = Path(str(self.config["camera"]["wide_calibration"]))
            if not calibration_path.is_absolute():
                calibration_path = project_root / calibration_path
            self._placement_undistorter = WideCameraUndistorter.from_file(
                calibration_path
            )
        return self._placement_undistorter

    def _detect_placement_letters(self, frame: Any) -> PlacementLetterFrameResult:
        detector_config = dict(self.config.get("box_center_alignment", {}))
        detector_config["placement_letter_min_confidence"] = float(
            self.config["placement_letter_navigation"]["letter_min_confidence"]
        )
        return detect_placement_letter_candidates(frame, detector_config)

    def _placement_front_distance(self) -> tuple[float, Dict[str, Any]]:
        error = self.state_reader.safety_error(
            require_ultrasound=True,
            require_fresh=True,
        )
        if error:
            raise MissionAbort(f"placement navigation sensor rejected: {error}")
        state = self.state_reader.poll()
        updated_at = getattr(state, "ultrasound_updated_at", None)
        if (
            isinstance(updated_at, bool)
            or not isinstance(updated_at, (int, float))
            or not math.isfinite(float(updated_at))
        ):
            raise MissionAbort("placement navigation ultrasound timestamp is invalid")
        age_s = max(0.0, time.monotonic() - float(updated_at))
        max_age_s = float(self.config["safety"].get("state_max_age_s", 0.75))
        if age_s > max_age_s:
            raise MissionAbort(
                "placement navigation ultrasound sample is stale: "
                f"age={age_s:.3f}s limit={max_age_s:.3f}s"
            )
        try:
            value = float(
                self.state_reader.filtered_front_ultrasound_m(
                    float(
                        self.config["safety"].get(
                            "placement_front_filter_window_s",
                            0.8,
                        )
                    )
                )
            )
        except (TypeError, ValueError) as exc:
            raise MissionAbort(
                "placement navigation front ultrasound is unavailable"
            ) from exc
        minimum = float(
            self.config["safety"].get("front_ultrasound_min_valid_m", 0.03)
        )
        if not math.isfinite(value) or value < minimum or value > 4.50:
            raise MissionAbort(
                f"placement navigation front ultrasound is invalid: {value!r}"
            )
        self._placement_front_samples.append(value)
        ordered = sorted(self._placement_front_samples)
        middle = len(ordered) // 2
        candidate = (
            ordered[middle]
            if len(ordered) % 2
            else (ordered[middle - 1] + ordered[middle]) / 2.0
        )
        configured = self.config["placement_letter_navigation"]
        jump_limit = float(configured.get("ultrasound_jump_reject_m", 0.25))
        jump_rejected = False
        if self._placement_front_accepted_m is None:
            self._placement_front_accepted_m = candidate
        elif abs(candidate - self._placement_front_accepted_m) > jump_limit:
            same_jump = (
                self._placement_front_jump_candidate_m is not None
                and abs(candidate - self._placement_front_jump_candidate_m)
                <= jump_limit / 2.0
            )
            if same_jump:
                self._placement_front_jump_count += 1
            else:
                self._placement_front_jump_candidate_m = candidate
                self._placement_front_jump_count = 1
            if self._placement_front_jump_count >= int(
                configured.get("ultrasound_jump_confirm_samples", 3)
            ):
                self._placement_front_accepted_m = candidate
                self._placement_front_jump_candidate_m = None
                self._placement_front_jump_count = 0
            else:
                jump_rejected = True
        else:
            self._placement_front_accepted_m = candidate
            self._placement_front_jump_candidate_m = None
            self._placement_front_jump_count = 0
        accepted = float(self._placement_front_accepted_m)
        return accepted, {
            "front_distance_m": accepted,
            "front_candidate_m": candidate,
            "front_sample_count": len(self._placement_front_samples),
            "jump_rejected": jump_rejected,
            "jump_confirmation_count": self._placement_front_jump_count,
            "ultrasound_updated_at": float(updated_at),
            "ultrasound_age_s": age_s,
        }

    def _wait_for_placement_front_distance(
        self,
        label: str,
    ) -> tuple[float, Dict[str, Any]]:
        """Wait for fresh placement ultrasound without abandoning the substage."""
        attempt = 0
        while True:
            try:
                return self._placement_front_distance()
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as exc:
                attempt += 1
                try:
                    self.motion.stop()
                except Exception as stop_exc:
                    if attempt == 1 or attempt % 20 == 0:
                        print(
                            "[placement-nav] sensor recovery stop warning "
                            f"label={label} error={stop_exc}",
                            flush=True,
                        )
                if attempt == 1 or attempt % 20 == 0:
                    print(
                        "[placement-nav] wait for fresh front ultrasound "
                        f"label={label} attempt={attempt} reason={exc}",
                        flush=True,
                    )
                self._reset_placement_front_filter()
                time.sleep(0.10)

    def _placement_strafe_front_velocity(self) -> float:
        configured = self.config["placement_letter_navigation"]
        distance_m, details = self._placement_front_distance()
        target_m = self.context.placement_search_front_target_m
        if (
            target_m is None
            or not math.isfinite(float(target_m))
            or float(target_m) <= 0.0
        ):
            raise MissionAbort(
                "placement lateral search front-distance target is unavailable"
            )
        target_m = float(target_m)
        boundary_delta_m = float(
            configured.get("search_hold_boundary_delta_m", 0.20)
        )
        if (
            not self._placement_boundary_recovery_active
            and abs(distance_m - target_m) >= boundary_delta_m
        ):
            raise PlacementSearchBoundary(
                "placement lateral search reached front-echo boundary: "
                f"front={distance_m:.3f}m target={target_m:.3f}m "
                f"delta={distance_m - target_m:+.3f}m "
                f"sensor={details}"
            )
        deadband_m = max(
            0.0,
            float(configured.get("strafe_forward_deadband_m", 0.03)),
        )
        error_m = distance_m - target_m
        if abs(error_m) <= deadband_m:
            return 0.0
        correction = float(
            configured.get("strafe_forward_hold_kp_s", 0.8)
        ) * error_m
        limit = max(
            0.0,
            float(configured.get("strafe_max_vx_correction_mps", 0.025)),
        )
        return max(-limit, min(limit, correction))

    def _reset_placement_front_filter(self) -> None:
        self._placement_front_samples.clear()
        self._placement_front_accepted_m = None
        self._placement_front_jump_candidate_m = None
        self._placement_front_jump_count = 0

    def _capture_placement_search_front_target(self) -> float:
        existing = self.context.placement_search_front_target_m
        if (
            existing is not None
            and math.isfinite(float(existing))
            and float(existing) > 0.0
        ):
            return float(existing)
        if self.context.dry_run:
            target_m = max(
                0.28,
                float(
                    self.config["placement_letter_navigation"].get(
                        "front_stop_distance_m",
                        0.28,
                    )
                ),
            )
            evidence: Dict[str, Any] = {"dry_run": True}
        else:
            self._reset_placement_front_filter()
            configured = self.config["placement_letter_navigation"]
            required_samples = int(
                configured.get("search_hold_capture_samples", 5)
            )
            values: List[float] = []
            last_sample_at: Optional[float] = None
            evidence = {}
            while True:
                value, sample_evidence = self._wait_for_placement_front_distance(
                    "capture_search_distance"
                )
                sample_at = sample_evidence.get("ultrasound_updated_at")
                if sample_at == last_sample_at:
                    time.sleep(0.02)
                    continue
                last_sample_at = sample_at
                values.append(float(value))
                evidence = dict(sample_evidence)
                if len(values) < required_samples:
                    time.sleep(0.02)
                    continue
                ordered = sorted(values)
                target_m = ordered[len(ordered) // 2]
                spread_m = ordered[-1] - ordered[0]
                max_spread_m = float(
                    configured.get("search_hold_capture_max_spread_m", 0.06)
                )
                minimum_m = float(
                    configured.get("search_hold_capture_min_m", 0.20)
                )
                maximum_m = float(
                    configured.get("search_hold_capture_max_m", 1.20)
                )
                if spread_m <= max_spread_m and minimum_m <= target_m <= maximum_m:
                    break
                print(
                    "[placement-nav] retry search-distance capture "
                    f"values={values} spread={spread_m:.3f}m "
                    f"target={target_m:.3f}m",
                    flush=True,
                )
                values.clear()
                last_sample_at = None
                self._reset_placement_front_filter()
                time.sleep(0.10)
            evidence.update(
                {
                    "capture_values_m": values,
                    "capture_spread_m": spread_m,
                    "capture_sample_count": len(values),
                }
            )
        self.context.placement_search_front_target_m = float(target_m)
        self._placement_last_sensor_evidence = dict(evidence)
        self._append_placement_navigation_event(
            {
                "event": "placement_search_front_target_captured",
                "target_distance_m": float(target_m),
                "deadband_m": float(
                    self.config["placement_letter_navigation"].get(
                        "strafe_forward_deadband_m",
                        0.03,
                    )
                ),
                "sensor": dict(evidence),
            }
        )
        print(
            "[placement-nav] lateral search front-distance hold "
            f"target={float(target_m):.3f}m "
            "deadband=+/-0.030m",
            flush=True,
        )
        return float(target_m)

    def _restore_placement_search_front_distance(self) -> bool:
        """Remove residual forward drift after a lateral placement move."""
        if self.context.dry_run:
            return False
        configured = self.config["placement_letter_navigation"]
        target_m = self.context.placement_search_front_target_m
        if (
            target_m is None
            or not math.isfinite(float(target_m))
            or float(target_m) <= 0.0
        ):
            raise MissionAbort(
                "placement lateral search front-distance target is unavailable"
            )
        target_m = float(target_m)
        deadband_m = float(configured.get("strafe_forward_deadband_m", 0.03))
        boundary_delta_m = float(
            configured.get("search_hold_boundary_delta_m", 0.20)
        )
        attempts = int(configured.get("search_hold_restore_attempts", 2))
        speed_mps = float(
            configured.get("search_hold_restore_speed_mps", 0.03)
        )
        max_step_m = float(
            configured.get("search_hold_restore_max_step_m", 0.10)
        )
        min_step_m = float(
            configured.get("search_hold_restore_min_step_m", 0.04)
        )
        moved = False

        for attempt in range(1, attempts + 1):
            distance_m, sensor = self._wait_for_placement_front_distance(
                "restore_search_distance"
            )
            error_m = distance_m - target_m
            if abs(error_m) <= max(deadband_m, min_step_m):
                return moved
            if abs(error_m) >= boundary_delta_m:
                raise PlacementSearchBoundary(
                    "placement front-distance restore reached echo boundary: "
                    f"front={distance_m:.3f}m target={target_m:.3f}m "
                    f"delta={error_m:+.3f}m sensor={sensor}"
                )
            command_m = max(-max_step_m, min(max_step_m, error_m))
            start_pose = self.state_reader.pose()
            motion_error: Optional[BaseException] = None
            stop_error: Optional[Exception] = None
            try:
                self.motion.set_autonomous()
                self.motion.go_distance(command_m, speed_mps=speed_mps)
            except BaseException as exc:
                motion_error = exc
            finally:
                try:
                    self.motion.stop()
                except Exception as exc:
                    stop_error = exc
            measured_forward_m = 0.0
            try:
                measured_forward_m, _lateral_m = self._project_placement_motion(
                    start_pose,
                    self.state_reader.pose(),
                )
            except Exception as exc:
                if motion_error is None:
                    motion_error = exc
            self._append_placement_navigation_event(
                {
                    "event": "placement_search_front_restore",
                    "attempt": attempt,
                    "target_distance_m": target_m,
                    "distance_before_m": distance_m,
                    "error_before_m": error_m,
                    "commanded_distance_m": command_m,
                    "measured_forward_m": measured_forward_m,
                    "sensor": dict(sensor),
                    "result": "error" if motion_error is not None else "complete",
                }
            )
            if isinstance(motion_error, (KeyboardInterrupt, SystemExit)):
                raise motion_error
            if motion_error is not None:
                print(
                    "[placement-nav] front-distance restore will retry after "
                    f"motion warning: {motion_error}",
                    flush=True,
                )
            if stop_error is not None:
                print(
                    "[placement-nav] front-distance restore stop warning: "
                    f"{stop_error}",
                    flush=True,
                )
            if self.context.first_outbound_forward_m is not None:
                self.context.first_outbound_forward_m = max(
                    0.20,
                    min(
                        1.60,
                        float(self.context.first_outbound_forward_m)
                        + measured_forward_m,
                    ),
                )
            moved = moved or abs(measured_forward_m) > 1e-4
            self._reset_placement_front_filter()
            if motion_error is not None:
                time.sleep(float(configured.get("motion_recovery_pause_s", 0.30)))
                return moved
        return moved

    def _placement_label_row_distance_m(self) -> Optional[float]:
        """Estimate row distance from calibrated focal length and box spacing."""
        try:
            raw, timed_out, read_error = self._read_placement_camera_latest(
                float(self.config["placement_letter_navigation"]["image_timeout_s"])
            )
            if raw is None or timed_out or read_error is not None:
                return None
            undistorter = self._placement_navigation_undistorter()
            undistorted = undistorter.apply(raw)
            result = detect_placement_box_centers(
                undistorted,
                self.config.get("box_center_alignment", {}),
            )
            if not (
                result.ok
                and len(result.centers) == 4
                and len(result.spacing_px) == 3
            ):
                return None
            spacings_px = sorted(
                abs(float(value))
                for value in result.spacing_px
                if math.isfinite(float(value)) and abs(float(value)) > 1.0
            )
            if not spacings_px:
                return None
            middle = len(spacings_px) // 2
            spacing_px = (
                spacings_px[middle]
                if len(spacings_px) % 2
                else (spacings_px[middle - 1] + spacings_px[middle]) / 2.0
            )
            calibration = getattr(undistorter, "calibration", {})
            focal_x_px = float(calibration["new_camera_matrix"][0][0])
            physical_spacing_m = float(
                self.config["placement_letter_navigation"]["letter_spacing_m"]
            )
            distance_m = focal_x_px * physical_spacing_m / spacing_px
            if not math.isfinite(distance_m) or distance_m <= 0.0:
                return None
            print(
                "[placement-nav] label-row range estimate "
                f"distance={distance_m:.3f}m spacing={spacing_px:.1f}px",
                flush=True,
            )
            return distance_m
        except Exception as exc:
            print(
                f"[placement-nav] label-row range warning: {exc}",
                flush=True,
            )
            return None

    def _read_placement_camera_latest(
        self,
        timeout_s: float,
    ) -> tuple[Any, bool, Optional[BaseException]]:
        started_at = time.monotonic()
        read_error: Optional[BaseException] = None
        try:
            latest_reader = getattr(self.wide_camera, "read_latest", None)
            if callable(latest_reader):
                raw = latest_reader(timeout_s=max(0.0, timeout_s))
            else:
                raw = self.wide_camera.read()
        except BaseException as exc:
            raw = None
            read_error = exc
        elapsed = time.monotonic() - started_at
        timed_out = raw is None and elapsed >= max(0.0, timeout_s) * 0.90
        if timed_out:
            try:
                self.motion.stop()
            except Exception as exc:
                print(
                    "[placement-nav] stop warning after camera timeout: "
                    f"{exc}",
                    flush=True,
                )
        return raw, timed_out, read_error

    def _capture_placement_navigation_frame(
        self,
        *,
        target_letter: str,
        frame_sequence: int,
        started_at: float,
    ) -> tuple[NavigationObservation, PlacementLetterFrameResult]:
        configured = self.config["placement_letter_navigation"]
        retries = int(configured["capture_retries"])
        timeout_s = float(configured["image_timeout_s"])
        last_reason = "camera_read_failed"
        raw = None
        accepted_frame_at = None
        accepted_signature = None
        attempt = 0
        while True:
            attempt += 1
            read_started = time.monotonic()
            raw, timed_out, read_error = self._read_placement_camera_latest(
                timeout_s
            )
            read_elapsed = time.monotonic() - read_started
            if read_error is not None:
                if isinstance(read_error, (KeyboardInterrupt, SystemExit)):
                    raise read_error
                raw = None
                last_reason = f"camera_read_error:{read_error}"
            camera_frame_at = getattr(self.wide_camera, "last_frame_at", None)
            marker_is_valid = (
                not isinstance(camera_frame_at, bool)
                and isinstance(camera_frame_at, (int, float))
                and math.isfinite(float(camera_frame_at))
            )
            accepted_frame_at = (
                float(camera_frame_at) if marker_is_valid else read_started
            )
            try:
                frame_signature = (
                    None
                    if raw is None
                    else cv2.resize(raw, (32, 18), interpolation=cv2.INTER_AREA).tobytes()
                )
            except Exception:
                frame_signature = None
                raw = None
                last_reason = "invalid_camera_frame"
            stale = False
            if self._placement_last_camera_frame_at is not None:
                stale = (
                    accepted_frame_at < self._placement_last_camera_frame_at
                    or frame_signature == self._placement_last_camera_signature
                    or (
                        accepted_frame_at == self._placement_last_camera_frame_at
                        and frame_signature == self._placement_last_camera_signature
                    )
                    or (
                        accepted_frame_at == self._placement_last_camera_frame_at
                        and id(raw) == self._placement_last_camera_frame_id
                    )
                )
            if raw is not None and not timed_out and not stale:
                accepted_signature = frame_signature
                break
            if raw is None and not last_reason.startswith("camera_read_error"):
                last_reason = "camera_read_failed"
            elif timed_out:
                last_reason = f"camera_read_timeout:{read_elapsed:.3f}s"
            elif stale:
                last_reason = "stale_cached_frame"
            if not timed_out:
                try:
                    self.motion.stop()
                except Exception as exc:
                    print(
                        "[placement-nav] capture-retry stop warning: "
                        f"{exc}",
                        flush=True,
                    )
            reconnect = getattr(self.wide_camera, "reconnect", None)
            if callable(reconnect):
                try:
                    reconnect(
                        f"placement_capture_retry_{attempt}:{last_reason}"
                    )
                except Exception as exc:
                    print(
                        "[placement-nav] camera reconnect warning: "
                        f"{exc}",
                        flush=True,
                    )
            if attempt == 1 or attempt % max(1, retries) == 0:
                print(
                    "[placement-nav] camera unavailable; keep current substage "
                    f"attempt={attempt} reason={last_reason}",
                    flush=True,
                )

        self._placement_last_camera_frame_at = accepted_frame_at
        self._placement_last_camera_frame_id = id(raw)
        self._placement_last_camera_signature = accepted_signature
        try:
            self.motion.stop()
        except Exception as exc:
            print(
                "[placement-nav] pre-processing stop warning: "
                f"{exc}",
                flush=True,
            )
        try:
            undistorted = self._placement_navigation_undistorter().apply(raw)
            detected = self._detect_placement_letters(undistorted)
        except MissionAbort:
            raise
        except Exception as exc:
            raise MissionAbort(f"placement letter detection failed: {exc}") from exc
        sequence = frame_sequence + 1
        front_distance_m, sensor_evidence = self._wait_for_placement_front_distance(
            "capture_letter_frame"
        )
        converted = tuple(
            LetterCandidate(
                candidate.recognized_letter,
                float(candidate.center[0]),
                float(candidate.confidence),
            )
            for candidate in detected.candidates
        )
        observation = NavigationObservation(
            sequence,
            int(detected.frame_width),
            converted,
            front_distance_m,
            time.monotonic() - started_at,
        )
        self._placement_latest_frame = (sequence, undistorted)
        self._placement_latest_detection = detected
        self._placement_last_sensor_evidence = sensor_evidence
        self._save_placement_capture_images(sequence, raw, undistorted)
        return observation, detected

    def _save_placement_capture_images(
        self,
        sequence: int,
        raw: Any,
        undistorted: Any,
    ) -> None:
        run_dir = self._placement_navigation_run_dir
        if run_dir is None:
            return
        prefix = f"frame_{sequence:06d}"
        for suffix, frame in (("raw", raw), ("undistorted", undistorted)):
            path = run_dir / f"{prefix}_{suffix}.jpg"
            try:
                written = cv2.imwrite(str(path), frame)
            except Exception as exc:
                written = False
                print(
                    f"[placement-nav] evidence image warning path={path}: {exc}",
                    flush=True,
                )
            if not written:
                print(
                    f"[placement-nav] evidence image warning path={path}",
                    flush=True,
                )

    def _append_placement_navigation_event(self, event: Dict[str, Any]) -> None:
        if self._placement_navigation_events_path is None:
            return
        try:
            with self._placement_navigation_events_path.open(
                "a",
                encoding="utf-8",
            ) as stream:
                stream.write(
                    json.dumps(event, ensure_ascii=False, default=str) + "\n"
                )
        except Exception as exc:
            print(
                f"[placement-nav] evidence event warning: {exc}",
                flush=True,
            )

    def _wide_camera_status_snapshot(self) -> Dict[str, Any]:
        reader = getattr(self.wide_camera, "status_snapshot", None)
        if not callable(reader):
            return {"status": "not_reported"}
        try:
            snapshot = reader()
        except Exception as exc:
            return {"status": "status_error", "error": str(exc)}
        return dict(snapshot) if isinstance(snapshot, dict) else {"status": str(snapshot)}

    def _append_placement_terminal_event(
        self,
        primary_error: BaseException,
    ) -> None:
        try:
            self._append_placement_navigation_event(
                {
                    "event": "terminal",
                    "result": "error",
                    "error": str(primary_error),
                    "camera": self._wide_camera_status_snapshot(),
                }
            )
        except Exception as log_error:
            if primary_error is None:
                raise MissionAbort(
                    f"placement terminal evidence failed: {log_error}"
                ) from log_error

    def _write_placement_frame_evidence(
        self,
        observation: NavigationObservation,
        detected: PlacementLetterFrameResult,
        target_letter: str,
        action: NavigationAction,
        navigator: PlacementLetterNavigator,
        *,
        result: str = "in_progress",
    ) -> None:
        run_dir = self._placement_navigation_run_dir
        latest = self._placement_latest_frame
        if run_dir is not None:
            if latest is None or latest[0] != observation.frame_sequence:
                print(
                    "[placement-nav] evidence frame sequence warning: "
                    f"expected={observation.frame_sequence} "
                    f"actual={None if latest is None else latest[0]}",
                    flush=True,
                )
            else:
                annotated_path = run_dir / (
                    f"frame_{observation.frame_sequence:06d}_annotated.jpg"
                )
                try:
                    annotated = annotate_placement_letters(
                        latest[1],
                        detected,
                        target_letter,
                        action.kind.value,
                    )
                    written = cv2.imwrite(str(annotated_path), annotated)
                except Exception as exc:
                    written = False
                    print(
                        "[placement-nav] annotated evidence warning "
                        f"path={annotated_path}: {exc}",
                        flush=True,
                    )
                if not written:
                    print(
                        "[placement-nav] annotated evidence warning "
                        f"path={annotated_path}",
                        flush=True,
                    )
        self._append_placement_navigation_event(
            {
                "sequence": observation.frame_sequence,
                "target_letter": target_letter,
                "candidates": [
                    {
                        "letter": candidate.letter,
                        "center_x_px": candidate.center_x_px,
                        "confidence": candidate.confidence,
                    }
                    for candidate in observation.candidates
                ],
                "sensor": dict(self._placement_last_sensor_evidence),
                "camera": self._wide_camera_status_snapshot(),
                "action": {
                    "kind": action.kind.value,
                    "reason": action.reason,
                    "distance_m": action.distance_m,
                    "vx_mps": action.vx_mps,
                    "vy_mps": action.vy_mps,
                },
                "measured_distance_m": None,
                "cumulative": {
                    "forward_m": navigator.forward_travel_m,
                    "final_approach_m": navigator.final_approach_travel_m,
                    "lateral_m": navigator.lateral_travel_m,
                    "net_lateral_m": navigator.net_lateral_m,
                },
                "elapsed_s": observation.elapsed_s,
                "result": result,
            }
        )

    def _write_placement_motion_event(
        self,
        action: NavigationAction,
        measured_distance_m: float,
        navigator: PlacementLetterNavigator,
        source_observation: NavigationObservation,
        result_observation: NavigationObservation,
    ) -> None:
        def candidates(observation: NavigationObservation) -> List[Dict[str, Any]]:
            return [
                {
                    "letter": candidate.letter,
                    "center_x_px": candidate.center_x_px,
                    "confidence": candidate.confidence,
                }
                for candidate in observation.candidates
            ]

        self._append_placement_navigation_event(
            {
                "event": "motion",
                "sequence": result_observation.frame_sequence,
                "source_sequence": source_observation.frame_sequence,
                "sensor": {
                    **dict(self._placement_last_sensor_evidence),
                    "front_distance_m": result_observation.front_distance_m,
                },
                "candidates": candidates(result_observation),
                "action": {
                    "kind": action.kind.value,
                    "reason": action.reason,
                    "requested_distance_m": action.distance_m,
                    "centering": action.centering,
                },
                "measured_distance_m": measured_distance_m,
                "cumulative": {
                    "forward_m": navigator.forward_travel_m,
                    "final_approach_m": navigator.final_approach_travel_m,
                    "lateral_m": navigator.lateral_travel_m,
                    "net_lateral_m": navigator.net_lateral_m,
                },
                "source_frame": {
                    "sequence": source_observation.frame_sequence,
                    "front_distance_m": source_observation.front_distance_m,
                    "candidates": candidates(source_observation),
                },
                "result_frame": {
                    "sequence": result_observation.frame_sequence,
                    "front_distance_m": result_observation.front_distance_m,
                    "candidates": candidates(result_observation),
                },
                "result": "motion_complete",
            }
        )

    @staticmethod
    def _project_placement_motion(
        start_pose: tuple[float, float, float],
        end_pose: tuple[float, float, float],
    ) -> tuple[float, float]:
        dx = float(end_pose[0]) - float(start_pose[0])
        dy = float(end_pose[1]) - float(start_pose[1])
        start_yaw = float(start_pose[2])
        forward = dx * math.cos(start_yaw) + dy * math.sin(start_yaw)
        lateral = -dx * math.sin(start_yaw) + dy * math.cos(start_yaw)
        return forward, lateral

    def _execute_placement_navigation_motion(
        self,
        action: NavigationAction,
        *,
        navigator: Optional[PlacementLetterNavigator] = None,
    ) -> float:
        if action.kind != ActionKind.STRAFE:
            raise MissionAbort(
                "placement fixed-distance executor accepts only strafe actions"
            )
        configured = self.config["placement_letter_navigation"]
        command_distance = action.distance_m * int(
            configured["physical_left_strafe_sign"]
        )
        completion_tolerance_m = placement_strafe_completion_tolerance(
            command_distance,
            float(configured.get("fine_strafe_distance_tolerance_m", 0.015)),
            float(configured.get("motion_stall_min_progress_m", 0.01)),
        )
        start_pose = None if self.context.dry_run else self.state_reader.pose()
        motion_error: Optional[BaseException] = None
        stop_error: Optional[Exception] = None
        previous_controlled_approach = self._controlled_box_approach_active
        previous_boundary_recovery = self._placement_boundary_recovery_active
        try:
            self._controlled_box_approach_active = True
            self._placement_boundary_recovery_active = (
                action.reason
                in {
                    "reverse_after_repeated_zero_progress",
                    "reverse_after_front_echo_boundary",
                }
            )
            pose_held_strafe = getattr(
                self.motion,
                "strafe_distance_pose_hold",
                None,
            )
            if not callable(pose_held_strafe):
                raise MissionAbort(
                    "placement pure-lateral strafe requires pose feedback"
                )
            pose_held_strafe(
                command_distance,
                speed_mps=float(configured["lateral_speed_mps"]),
                completion_tolerance_m=completion_tolerance_m,
                forward_hold_kp_s=float(
                    configured.get("strafe_forward_hold_kp_s", 0.0)
                ),
                max_vx_correction_mps=float(
                    configured.get("strafe_max_vx_correction_mps", 0.025)
                ),
                forward_deadband_m=float(
                    configured.get("strafe_forward_deadband_m", 0.03)
                ),
                forward_velocity_provider=self._placement_strafe_front_velocity,
                max_forward_drift_m=float(
                    configured.get("strafe_max_forward_drift_m", 0.15)
                ),
                yaw_hold_kp_s=float(
                    configured.get("strafe_yaw_hold_kp_s", 1.2)
                ),
                max_wz_correction_rad_s=float(
                    configured.get(
                        "strafe_max_wz_correction_rad_s",
                        0.12,
                    )
                ),
                yaw_deadband_deg=float(
                    configured.get("strafe_yaw_deadband_deg", 0.30)
                ),
                max_yaw_drift_deg=float(
                    configured.get("strafe_max_yaw_drift_deg", 5.0)
                ),
            )
        except BaseException as exc:
            motion_error = exc
        finally:
            self._controlled_box_approach_active = previous_controlled_approach
            self._placement_boundary_recovery_active = previous_boundary_recovery
            try:
                self.motion.stop()
            except Exception as exc:
                stop_error = exc
            self._reset_placement_front_filter()

        if self.context.dry_run:
            measured = action.distance_m
        else:
            try:
                _forward, command_lateral = self._project_placement_motion(
                    start_pose,
                    self.state_reader.pose(),
                )
                measured = command_lateral * int(
                    configured["physical_left_strafe_sign"]
                )
            except Exception as exc:
                if motion_error is not None:
                    raise motion_error
                raise MissionAbort(
                    f"placement strafe odometry failed: {exc}"
                ) from exc
        if navigator is not None:
            try:
                navigator.record_motion(action, measured)
            except Exception as exc:
                if motion_error is not None:
                    raise MissionAbort(
                        "placement strafe failed and partial odometry was "
                        f"rejected: motion={motion_error}; odometry={exc}"
                    ) from motion_error
                raise MissionAbort(
                    f"placement strafe odometry rejected: {exc}"
                ) from exc
            if isinstance(motion_error, PlacementSearchBoundary):
                navigator.request_lateral_recovery(action)
            self.context.placement_navigation_net_lateral_m = (
                navigator.net_lateral_m
            )
            self.context.placement_navigation_lateral_travel_m = (
                navigator.lateral_travel_m
            )
            self.context.placement_navigation_search_phase = (
                navigator.bilateral_search_phase
            )
            self.context.placement_navigation_last_lateral_sign = (
                navigator.last_lateral_sign
            )
            self.context.placement_navigation_last_geometry_sign = (
                navigator.last_geometry_lateral_sign
            )
            self.context.placement_navigation_pending_recovery_sign = (
                navigator.pending_recovery_lateral_sign
            )
            self.context.placement_navigation_zero_progress_count = (
                navigator.zero_progress_strafe_count
            )
        if isinstance(motion_error, PlacementSearchBoundary):
            print(
                "[placement-nav] front-echo boundary reached; stop and "
                "reverse the next lateral step: "
                f"{motion_error}",
                flush=True,
            )
            motion_error = None
        if motion_error is not None:
            if isinstance(
                motion_error,
                (MissionAbort, KeyboardInterrupt, SystemExit),
            ):
                raise motion_error
            raise MissionAbort(
                f"placement strafe failed after moving {measured:.3f}m: "
                f"{motion_error}"
            ) from motion_error
        if stop_error is not None:
            raise MissionAbort(
                f"placement strafe stop failed after moving {measured:.3f}m: "
                f"{stop_error}"
            ) from stop_error
        return measured

    def _execute_placement_final_approach(
        self,
        action: NavigationAction,
        *,
        navigator: PlacementLetterNavigator,
    ) -> float:
        raise MissionAbort(
            "placement final approach is disabled after the ultrasound stop"
        )
        # Unreachable legacy implementation retained for old evidence readers.
        if action.kind != ActionKind.FINAL_APPROACH:
            raise MissionAbort(
                "placement final approach executor requires final_approach action"
            )
        requested_m = abs(float(action.distance_m))
        direction = math.copysign(1.0, float(action.distance_m))
        configured = self.config["placement_letter_navigation"]
        total_before_m = float(self.context.placement_final_approach_progress_m)
        final_total_m = float(configured["final_approach_distance_m"])
        if (
            not math.isfinite(total_before_m)
            or total_before_m < 0.0
            or total_before_m > final_total_m + 0.03
        ):
            raise MissionAbort(
                "placement final approach checkpoint is invalid: "
                f"{total_before_m!r}"
            )
        if self.context.dry_run:
            navigator.record_motion(action, requested_m)
            total_after_m = min(final_total_m, total_before_m + requested_m)
            self.context.placement_final_approach_progress_m = total_after_m
            self.context.placement_final_approach_complete = (
                total_after_m
                >= final_total_m - MOTION_MEASUREMENT_TOLERANCE_M
            )
            return requested_m

        segment_progress_m = 0.0
        recovery_count = 0
        recovery_limit = int(configured.get("motion_stall_retries", 2))
        minimum_progress_m = float(
            configured.get("motion_stall_min_progress_m", 0.01)
        )
        active_speed_mps = float(action.vx_mps)
        degraded_reason: Optional[str] = None
        while (
            requested_m - segment_progress_m
            > MOTION_MEASUREMENT_TOLERANCE_M
        ):
            remaining_m = requested_m - segment_progress_m
            start_pose = self.state_reader.pose()
            motion_error: Optional[BaseException] = None
            stop_error: Optional[Exception] = None
            previous_controlled = self._controlled_box_approach_active
            self._controlled_box_approach_active = True
            try:
                self.motion.go_distance(
                    direction * remaining_m,
                    speed_mps=active_speed_mps,
                )
            except BaseException as exc:
                motion_error = exc
            finally:
                self._controlled_box_approach_active = previous_controlled
                try:
                    self.motion.stop()
                except Exception as exc:
                    stop_error = exc

            if isinstance(motion_error, (KeyboardInterrupt, SystemExit)):
                raise motion_error
            try:
                projected_forward_m, _lateral = self._project_placement_motion(
                    start_pose,
                    self.state_reader.pose(),
                )
                measured_step_m = max(
                    0.0,
                    direction * projected_forward_m,
                )
            except Exception as exc:
                measured_step_m = 0.0
                motion_error = motion_error or exc
            measured_step_m = min(remaining_m, measured_step_m)
            segment_progress_m += measured_step_m
            total_progress_m = min(
                final_total_m,
                total_before_m + segment_progress_m,
            )
            self.context.placement_final_approach_progress_m = total_progress_m
            print(
                "[placement-nav] segmented final approach "
                f"step={measured_step_m:.3f}m "
                f"segment={segment_progress_m:.3f}/{requested_m:.3f}m "
                f"total={total_progress_m:.3f}/{final_total_m:.3f}m",
                flush=True,
            )
            needs_recovery = (
                motion_error is not None
                or stop_error is not None
                or measured_step_m < minimum_progress_m
            )
            if not needs_recovery:
                continue
            recovery_count += 1
            if recovery_count > recovery_limit:
                degraded_reason = (
                    "final_approach_stalled_after_recovery "
                    f"progress={segment_progress_m:.3f}/{requested_m:.3f}m "
                    f"motion={motion_error} stop={stop_error}"
                )
                break
            try:
                self.motion.set_autonomous()
            except Exception as exc:
                print(
                    f"[placement-nav] final approach recovery warning: {exc}",
                    flush=True,
                )
            time.sleep(float(configured.get("motion_recovery_pause_s", 0.30)))
            active_speed_mps = min(
                active_speed_mps,
                float(configured.get("motion_recovery_speed_mps", 0.06)),
            )
            print(
                "[placement-nav] final approach recovery "
                f"attempt={recovery_count}/{recovery_limit} "
                f"speed={active_speed_mps:.3f}m/s",
                flush=True,
            )

        if degraded_reason is None:
            navigator.record_motion(action, segment_progress_m)
        else:
            navigator.complete_final_approach_degraded(
                segment_progress_m,
                degraded_reason,
            )
            print(
                "[placement-nav] final approach degraded; "
                "continue with visual confirmation instead of aborting: "
                f"{degraded_reason}",
                flush=True,
            )
        total_after_m = min(
            final_total_m,
            total_before_m + segment_progress_m,
        )
        self.context.placement_final_approach_progress_m = total_after_m
        self.context.placement_final_approach_complete = (
            navigator.final_approach_completed
        )
        return segment_progress_m

    def _return_placement_search_to_origin(
        self,
        navigator: PlacementLetterNavigator,
    ) -> None:
        net_lateral_m = float(navigator.net_lateral_m)
        if self.context.dry_run or abs(net_lateral_m) <= 0.02:
            return
        configured = self.config["placement_letter_navigation"]
        requested_m = -net_lateral_m
        action = NavigationAction(
            ActionKind.STRAFE,
            "return_failed_search_to_origin",
            distance_m=requested_m,
            vy_mps=math.copysign(
                float(configured["lateral_speed_mps"]),
                requested_m,
            ),
        )
        measured_m = self._execute_placement_navigation_motion(action)
        if abs(measured_m - requested_m) > 0.05:
            raise MissionAbort(
                "placement search origin return odometry mismatch: "
                f"requested={requested_m:.3f}m measured={measured_m:.3f}m"
            )
        print(
            "[placement-nav] failed search returned to attempt origin "
            f"requested={requested_m:.3f}m measured={measured_m:.3f}m",
            flush=True,
        )

    def _require_placement_navigation_time_remaining(self, started_at: float) -> None:
        del started_at
        return None

    def _run_placement_forward_search_step(
        self,
        *,
        target_letter: str,
        frame_sequence: int,
        maximum_distance_m: float,
        started_at: float,
        action: Optional[NavigationAction] = None,
        navigator: Optional[PlacementLetterNavigator] = None,
    ) -> NavigationObservation:
        configured = self.config["placement_letter_navigation"]
        forward_action = action or NavigationAction(
            ActionKind.FORWARD,
            "placement_forward_search",
            distance_m=maximum_distance_m,
            vx_mps=float(configured["forward_speed_mps"]),
        )
        start_pose = None if self.context.dry_run else self.state_reader.pose()
        measured_forward_m = 0.0
        primary_error: Optional[BaseException] = None
        stall_timeout_s = float(configured.get("motion_stall_timeout_s", 2.0))
        stall_min_progress_m = float(
            configured.get("motion_stall_min_progress_m", 0.01)
        )
        stall_recovery_limit = int(configured.get("motion_stall_retries", 2))
        recovery_pause_s = float(configured.get("motion_recovery_pause_s", 0.30))
        active_speed_mps = float(configured["forward_speed_mps"])
        stall_checkpoint_at = time.monotonic()
        stall_checkpoint_m = 0.0
        recovery_count = 0
        try:
            while True:
                observation, detected = self._capture_placement_navigation_frame(
                    target_letter=target_letter,
                    frame_sequence=frame_sequence,
                    started_at=started_at,
                )
                frame_sequence = observation.frame_sequence
                if self.context.dry_run:
                    measured_forward_m = maximum_distance_m
                else:
                    projected_forward_m, _lateral = self._project_placement_motion(
                        start_pose,
                        self.state_reader.pose(),
                    )
                    measured_forward_m = abs(projected_forward_m)
                now = time.monotonic()
                if measured_forward_m - stall_checkpoint_m >= stall_min_progress_m:
                    stall_checkpoint_m = measured_forward_m
                    stall_checkpoint_at = now
                elif now - stall_checkpoint_at >= stall_timeout_s:
                    self.motion.stop()
                    recovery_count += 1
                    if recovery_count > stall_recovery_limit:
                        print(
                            "[placement-nav] forward search stalled; "
                            "return to visual decision without aborting "
                            f"progress={measured_forward_m:.3f}m",
                            flush=True,
                        )
                        return observation
                    try:
                        self.motion.set_autonomous()
                    except Exception as exc:
                        print(
                            f"[placement-nav] forward recovery warning: {exc}",
                            flush=True,
                        )
                    time.sleep(recovery_pause_s)
                    active_speed_mps = min(
                        active_speed_mps,
                        float(configured.get("motion_recovery_speed_mps", 0.06)),
                    )
                    stall_checkpoint_at = time.monotonic()
                    stall_checkpoint_m = measured_forward_m
                if (
                    observation.front_distance_m
                    <= float(configured["front_stop_distance_m"])
                ):
                    return observation
                if measured_forward_m >= maximum_distance_m:
                    return observation
                if navigator is not None:
                    self._write_placement_frame_evidence(
                        observation,
                        detected,
                        target_letter,
                        forward_action,
                        navigator,
                        result="forward_active",
                    )
                resume_front_m, sensor_evidence = self._placement_front_distance()
                self._placement_last_sensor_evidence = sensor_evidence
                if resume_front_m <= float(configured["front_stop_distance_m"]):
                    return NavigationObservation(
                        observation.frame_sequence,
                        observation.frame_width,
                        observation.candidates,
                        resume_front_m,
                        time.monotonic() - started_at,
                    )
                self.motion.move(active_speed_mps, 0.0, 0.0)
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            stop_error: Optional[Exception] = None
            pose_error: Optional[Exception] = None
            record_error: Optional[Exception] = None
            try:
                self.motion.stop()
            except Exception as exc:
                stop_error = exc
            if not self.context.dry_run:
                try:
                    projected_forward_m, _lateral = self._project_placement_motion(
                        start_pose,
                        self.state_reader.pose(),
                    )
                    measured_forward_m = abs(projected_forward_m)
                except Exception as exc:
                    pose_error = exc
            if navigator is not None and abs(measured_forward_m) > 1e-9:
                try:
                    navigator.record_motion(forward_action, measured_forward_m)
                except Exception as exc:
                    record_error = exc
            if primary_error is None:
                if stop_error is not None:
                    raise MissionAbort(
                        f"placement forward stop failed: {stop_error}"
                    ) from stop_error
                if pose_error is not None:
                    raise MissionAbort(
                        f"placement forward final odometry failed: {pose_error}"
                    ) from pose_error
                if record_error is not None:
                    raise MissionAbort(
                        f"placement forward odometry rejected: {record_error}"
                    ) from record_error

    def _create_placement_navigation_run_dir(self) -> Path:
        project_root = Path(__file__).resolve().parent.parent
        configured = self.config["placement_letter_navigation"]
        root = Path(str(configured["run_log_dir"])).expanduser()
        if not root.is_absolute():
            root = project_root / root
        run_dir = root / (
            datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            + f"_{uuid.uuid4().hex[:8]}"
        )
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    def _run_forced_placement_forward(
        self,
        *,
        target_letter: str,
        navigator: PlacementLetterNavigator,
        started_at: float,
    ) -> None:
        del started_at
        target_distance_m = 1.30
        speed_mps = 0.08
        completed_before_m = max(
            0.0,
            float(self.context.placement_forced_forward_progress_m),
        )
        remaining_m = max(0.0, target_distance_m - completed_before_m)
        print(
            "[placement-nav] forced post-turn forward "
            f"target={target_distance_m:.3f}m "
            f"completed={completed_before_m:.3f}m "
            f"remaining={remaining_m:.3f}m "
            "ultrasound_ignored=True",
            flush=True,
        )
        if remaining_m <= MOTION_MEASUREMENT_TOLERANCE_M:
            self.context.placement_forced_forward_progress_m = target_distance_m
            return
        if self.context.dry_run:
            self.context.placement_forced_forward_progress_m = target_distance_m
            return

        action = NavigationAction(
            ActionKind.FORWARD,
            "forced_post_turn_forward_1p3m",
            distance_m=remaining_m,
            vx_mps=speed_mps,
        )
        expected_duration_s = remaining_m / speed_mps
        start_pose: Optional[tuple[float, float, float]] = None
        try:
            start_pose = self.state_reader.pose()
        except Exception as exc:
            raise MissionAbort(
                f"forced post-turn forward requires odometry: {exc}"
            ) from exc
        command_started_at = time.monotonic()
        motion_error: Optional[BaseException] = None
        stop_error: Optional[Exception] = None
        record_error: Optional[Exception] = None
        previous_forced_forward = self._placement_forced_forward_active
        try:
            self._placement_forced_forward_active = True
            self.motion.set_autonomous()
            self.motion.go_distance(remaining_m, speed_mps=speed_mps)
        except BaseException as exc:
            motion_error = exc
        finally:
            self._placement_forced_forward_active = previous_forced_forward
            try:
                self.motion.stop()
            except Exception as exc:
                stop_error = exc

        elapsed_s = max(0.0, time.monotonic() - command_started_at)
        odom_increment_m = 0.0
        try:
            odom_increment_m, _lateral_m = self._project_placement_motion(
                start_pose,
                self.state_reader.pose(),
            )
            odom_increment_m = max(0.0, min(remaining_m, odom_increment_m))
        except Exception as exc:
            if motion_error is None:
                motion_error = exc
        self.context.placement_forced_forward_odom_m = min(
            target_distance_m + 0.20,
            max(
                0.0,
                float(self.context.placement_forced_forward_odom_m)
                + odom_increment_m,
            ),
        )

        total_progress_m = min(
            target_distance_m,
            completed_before_m + odom_increment_m,
        )
        self.context.placement_forced_forward_progress_m = total_progress_m
        if odom_increment_m > 1e-9:
            try:
                navigator.record_motion(action, odom_increment_m)
            except Exception as exc:
                record_error = exc
        self._append_placement_navigation_event(
            {
                "event": "forced_post_turn_forward",
                "target_letter": target_letter,
                "target_distance_m": target_distance_m,
                "completed_before_m": completed_before_m,
                "commanded_remaining_m": remaining_m,
                "expected_duration_s": expected_duration_s,
                "elapsed_s": elapsed_s,
                "time_based_estimate_m": min(
                    remaining_m,
                    elapsed_s * speed_mps,
                ),
                "odometry_increment_m": odom_increment_m,
                "odometry_cumulative_m": (
                    self.context.placement_forced_forward_odom_m
                ),
                "total_progress_m": total_progress_m,
                "speed_mps": speed_mps,
                "ultrasound_ignored": True,
                "odometry_primary": True,
                "result": "error" if motion_error is not None else "complete",
            }
        )
        if isinstance(motion_error, (KeyboardInterrupt, SystemExit)):
            raise motion_error
        if motion_error is not None:
            raise MissionAbort(
                "forced post-turn forward failed after "
                f"{total_progress_m:.3f}/{target_distance_m:.3f}m: "
                f"{motion_error}"
            ) from motion_error
        if stop_error is not None:
            raise MissionAbort(
                f"forced post-turn forward stop failed: {stop_error}"
            ) from stop_error
        if record_error is not None:
            raise MissionAbort(
                f"forced post-turn forward progress rejected: {record_error}"
            ) from record_error
        if total_progress_m < (
            target_distance_m - MOTION_MEASUREMENT_TOLERANCE_M
        ):
            raise MissionAbort(
                "forced post-turn forward incomplete: "
                f"{total_progress_m:.3f}/{target_distance_m:.3f}m"
            )
        self.context.placement_forced_forward_progress_m = target_distance_m
        if not self.context.placed_letters:
            measured_outbound_m = float(
                self.context.placement_forced_forward_odom_m
            )
            if not (0.20 <= measured_outbound_m <= target_distance_m + 0.20):
                print(
                    "[placement-nav] outbound odometry unavailable; "
                    "return distance falls back to fixed 1.300m "
                    f"odom={measured_outbound_m:.3f}m",
                    flush=True,
                )
                measured_outbound_m = target_distance_m
            self.context.first_outbound_forward_m = measured_outbound_m
        print(
            "[placement-nav] forced post-turn forward complete "
            f"distance={target_distance_m:.3f}m "
            f"recorded_return={self.context.first_outbound_forward_m}; "
            "capture lateral-search front distance next",
            flush=True,
        )

    def _run_placement_ultrasound_approach(
        self,
        *,
        target_letter: str,
        navigator: PlacementLetterNavigator,
        started_at: float,
    ) -> None:
        configured = self.config["placement_letter_navigation"]
        stop_distance_m = float(configured["front_stop_distance_m"])
        forward_budget_m = float(configured["forward_budget_m"])
        forward_speed_mps = float(configured["forward_speed_mps"])
        action = NavigationAction(
            ActionKind.FORWARD,
            "continuous_ultrasound_approach",
            distance_m=forward_budget_m,
            vx_mps=forward_speed_mps,
        )
        limits = getattr(self.motion, "limits", None)
        raw_command_hz = getattr(limits, "command_hz", 20.0)
        command_hz = (
            float(raw_command_hz)
            if (
                not isinstance(raw_command_hz, bool)
                and isinstance(raw_command_hz, (int, float))
                and math.isfinite(float(raw_command_hz))
            )
            else 20.0
        )
        command_hz = max(1.0, command_hz)
        command_period_s = 1.0 / command_hz
        start_pose = None if self.context.dry_run else self.state_reader.pose()
        start_front_m: Optional[float] = None
        initial_ultrasound_m: Optional[float] = None
        initial_visual_distance_m: Optional[float] = None
        odometry_stop_distance_m: Optional[float] = None
        measured_forward_m = 0.0
        primary_error: Optional[BaseException] = None
        required_close = int(configured.get("ultrasound_stable_samples", 3))
        consecutive_close = 0
        last_close_sample_at: Optional[float] = None
        stall_timeout_s = float(configured.get("motion_stall_timeout_s", 2.0))
        stall_min_progress_m = float(
            configured.get("motion_stall_min_progress_m", 0.01)
        )
        stall_recovery_limit = int(configured.get("motion_stall_retries", 3))
        recovery_pause_s = float(
            configured.get("motion_recovery_pause_s", 0.30)
        )
        recovery_speed_mps = float(
            configured.get("motion_recovery_speed_mps", 0.05)
        )
        slow_distance_m = float(
            configured.get("approach_slow_distance_m", 0.40)
        )
        creep_distance_m = float(
            configured.get("approach_creep_distance_m", 0.33)
        )
        slow_speed_mps = float(
            configured.get("approach_slow_speed_mps", 0.05)
        )
        creep_speed_mps = float(
            configured.get("approach_creep_speed_mps", 0.025)
        )
        odometry_stop_guard_margin_m = float(
            configured.get("odometry_stop_guard_margin_m", 0.02)
        )
        echo_loss_margin_m = float(
            configured.get("ultrasound_echo_loss_margin_m", 0.35)
        )
        echo_loss_min_progress_m = float(
            configured.get("ultrasound_echo_loss_min_progress_m", 0.05)
        )
        echo_loss_fallback_speed_mps = float(
            configured.get("echo_loss_fallback_speed_mps", 0.03)
        )
        visual_odom_fallback_speed_mps = float(
            configured.get("visual_odom_fallback_speed_mps", 0.05)
        )
        active_speed_mps = forward_speed_mps
        echo_loss_active = False
        ultrasound_stuck_fallback_active = False
        command_started_at: Optional[float] = None
        stall_checkpoint_at: Optional[float] = None
        stall_checkpoint_m = 0.0
        stall_recovery_count = 0
        ultrasound_progress_m = 0.0
        progress_evidence_m = 0.0
        self._reset_placement_front_filter()
        self._prime_placement_front_filter(
            float(configured.get("approach_filter_warmup_s", 0.30))
        )
        print(
            "[placement-nav] continuous approach until front ultrasound "
            f"<= {stop_distance_m:.2f}m",
            flush=True,
        )
        previous_controlled_approach = self._controlled_box_approach_active
        self._controlled_box_approach_active = True
        try:
            while True:
                front_distance_m, sensor_evidence = (
                    self._wait_for_placement_front_distance(
                        "final_28cm_approach"
                    )
                )
                if start_front_m is None:
                    initial_ultrasound_m = front_distance_m
                    start_front_m = initial_ultrasound_m
                    if not self.context.dry_run:
                        require_visual_preflight = bool(
                            configured.get(
                                "require_visual_row_before_forward",
                                False,
                            )
                        )
                        visual_trigger_m = float(
                            configured.get(
                                "visual_row_preflight_trigger_m",
                                0.33,
                            )
                        )
                        run_visual_preflight = (
                            require_visual_preflight
                            or initial_ultrasound_m <= visual_trigger_m
                        )
                        if run_visual_preflight:
                            visual_attempts = int(
                                configured.get("visual_row_preflight_attempts", 1)
                            )
                            if not require_visual_preflight:
                                visual_attempts = 1
                            for _attempt in range(max(1, visual_attempts)):
                                initial_visual_distance_m = (
                                    self._placement_label_row_distance_m()
                                )
                                if initial_visual_distance_m is not None:
                                    break
                                time.sleep(command_period_s)
                        else:
                            print(
                                "[placement-nav] skip visual row preflight; "
                                f"ultrasound={initial_ultrasound_m:.3f}m "
                                f"trigger={visual_trigger_m:.3f}m",
                                flush=True,
                            )
                        if (
                            run_visual_preflight
                            and initial_visual_distance_m is None
                            and require_visual_preflight
                        ):
                            raise MissionAbort(
                                "placement row is not visible before forward approach"
                            )
                        if initial_visual_distance_m is not None:
                            preflight_tolerance_m = float(
                                configured.get(
                                    "visual_ultrasound_start_tolerance_m",
                                    0.35,
                                )
                            )
                            stuck_value_m = float(
                                configured.get("ultrasound_stuck_value_m", 0.28)
                            )
                            stuck_tolerance_m = float(
                                configured.get(
                                    "ultrasound_stuck_tolerance_m",
                                    0.01,
                                )
                            )
                            ultrasound_is_stuck_close = (
                                abs(initial_ultrasound_m - stuck_value_m)
                                <= stuck_tolerance_m
                                and initial_visual_distance_m
                                > initial_ultrasound_m + preflight_tolerance_m
                            )
                            if ultrasound_is_stuck_close:
                                ultrasound_stuck_fallback_active = True
                                start_front_m = initial_visual_distance_m
                                active_speed_mps = min(
                                    forward_speed_mps,
                                    visual_odom_fallback_speed_mps,
                                )
                                self.motion.stop()
                                print(
                                    "[placement-nav] front ultrasound frozen "
                                    "near pickup distance; use visual range and "
                                    "odometry hard stop "
                                    f"ultrasound={initial_ultrasound_m:.3f}m "
                                    f"visual={initial_visual_distance_m:.3f}m "
                                    f"speed={active_speed_mps:.3f}m/s",
                                    flush=True,
                                )
                            elif abs(
                                initial_visual_distance_m
                                - initial_ultrasound_m
                            ) > preflight_tolerance_m:
                                print(
                                    "[placement-nav] visual/ultrasound range warning; "
                                    "continue with filtered ultrasound and odometry: "
                                    f"visual={initial_visual_distance_m:.3f}m "
                                    f"ultrasound={initial_ultrasound_m:.3f}m "
                                    f"tolerance={preflight_tolerance_m:.3f}m",
                                    flush=True,
                                )
                    odometry_stop_distance_m = min(
                        forward_budget_m,
                        max(
                            0.0,
                            start_front_m
                            - stop_distance_m
                            + odometry_stop_guard_margin_m,
                        ),
                    )
                    print(
                        "[placement-nav] approach preflight "
                        f"ultrasound={initial_ultrasound_m:.3f}m "
                        f"visual={initial_visual_distance_m} "
                        f"reference={start_front_m:.3f}m "
                        f"odom_hard_stop={odometry_stop_distance_m:.3f}m",
                        flush=True,
                    )
                self._placement_last_sensor_evidence = sensor_evidence
                if self.context.dry_run:
                    measured_forward_m = 0.0
                else:
                    measured_forward_m, _lateral_m = (
                        self._project_placement_motion(
                            start_pose,
                            self.state_reader.pose(),
                        )
                    )
                    measured_forward_m = max(0.0, measured_forward_m)
                ultrasound_progress_m = max(
                    0.0,
                    float(initial_ultrasound_m) - front_distance_m,
                )
                ultrasound_progress_valid = (
                    not bool(sensor_evidence.get("jump_rejected", False))
                    and not echo_loss_active
                    and not ultrasound_stuck_fallback_active
                )
                progress_evidence_m = max(
                    measured_forward_m,
                    ultrasound_progress_m if ultrasound_progress_valid else 0.0,
                )
                expected_front_m = max(
                    stop_distance_m,
                    float(start_front_m) - measured_forward_m,
                )
                if (
                    not echo_loss_active
                    and not ultrasound_stuck_fallback_active
                    and measured_forward_m >= echo_loss_min_progress_m
                    and front_distance_m
                    > expected_front_m + echo_loss_margin_m
                ):
                    echo_loss_active = True
                    active_speed_mps = min(
                        forward_speed_mps,
                        echo_loss_fallback_speed_mps,
                    )
                    self.motion.stop()
                    print(
                        "[placement-nav] ultrasound echo lost; "
                        "continue only to odometry hard stop "
                        f"front={front_distance_m:.3f}m "
                        f"expected={expected_front_m:.3f}m "
                        f"speed={active_speed_mps:.3f}m/s",
                        flush=True,
                    )
                final_min_m = float(
                    configured.get("final_ultrasound_min_m", 0.27)
                )
                final_max_m = float(
                    configured.get("final_ultrasound_max_m", 0.30)
                )
                in_final_window = (
                    final_min_m <= front_distance_m <= final_max_m
                    and not ultrasound_stuck_fallback_active
                )
                if in_final_window:
                    if self.context.dry_run:
                        consistency_ok = True
                        visual_arrival_ok = True
                    else:
                        observed_drop_m = max(
                            0.0,
                            float(start_front_m) - front_distance_m,
                        )
                        tolerance_m = float(
                            configured.get(
                                "ultrasound_odom_consistency_tolerance_m",
                                0.15,
                            )
                        )
                        consistency_ok = (
                            abs(observed_drop_m - progress_evidence_m)
                            <= tolerance_m
                        )
                        visual_arrival_ok = False
                        if start_front_m <= stop_distance_m:
                            if initial_visual_distance_m is None:
                                initial_visual_distance_m = (
                                    self._placement_label_row_distance_m()
                                )
                            if initial_visual_distance_m is not None:
                                visual_required_forward_m = max(
                                    0.0,
                                    initial_visual_distance_m - stop_distance_m,
                                )
                                visual_arrival_ok = (
                                    measured_forward_m
                                    >= visual_required_forward_m
                                )
                                if initial_visual_distance_m > (
                                    stop_distance_m + tolerance_m
                                ):
                                    consistency_ok = False
                    if not consistency_ok and not visual_arrival_ok:
                        consecutive_close = 0
                        print(
                            "[placement-nav] reject premature close ultrasound "
                            f"front={front_distance_m:.3f}m "
                            f"forward_odom={measured_forward_m:.3f}m "
                            f"visual_start={initial_visual_distance_m}",
                            flush=True,
                        )
                    else:
                        sample_at = sensor_evidence.get("ultrasound_updated_at")
                        if sample_at == last_close_sample_at:
                            self.motion.stop()
                            time.sleep(command_period_s)
                            continue
                        last_close_sample_at = sample_at
                        consecutive_close += 1
                        self.motion.stop()
                        if consecutive_close < required_close:
                            time.sleep(command_period_s)
                            continue
                        print(
                            "[placement-nav] approach reached "
                            f"front={front_distance_m:.3f}m "
                            f"stable={consecutive_close}/{required_close} "
                            f"odom_consistent={consistency_ok} "
                            f"visual_arrival={visual_arrival_ok}",
                            flush=True,
                        )
                        return
                else:
                    consecutive_close = 0
                if (
                    odometry_stop_distance_m is not None
                    and measured_forward_m >= odometry_stop_distance_m
                ):
                    self.motion.stop()
                    estimated_front_m = max(
                        0.0,
                        float(start_front_m) - measured_forward_m,
                    )
                    print(
                        "[placement-nav] odometry guard reached before stable "
                        "28cm confirmation; hold and keep sampling: "
                        f"forward={measured_forward_m:.3f}m "
                        f"estimated_front={estimated_front_m:.3f}m "
                        f"ultrasound={front_distance_m:.3f}m "
                        f"required=[{final_min_m:.3f},{final_max_m:.3f}]m",
                        flush=True,
                    )
                    self._reset_placement_front_filter()
                    time.sleep(recovery_pause_s)
                    continue
                if self.context.dry_run:
                    return
                if measured_forward_m >= forward_budget_m:
                    self.motion.stop()
                    print(
                        "[placement-nav] forward budget reached; hold position "
                        "and wait for stable 28cm evidence "
                        f"measured={measured_forward_m:.3f}m "
                        f"front={front_distance_m:.3f}m",
                        flush=True,
                    )
                    self._reset_placement_front_filter()
                    time.sleep(recovery_pause_s)
                    continue
                now = time.monotonic()
                if command_started_at is not None:
                    if (
                        progress_evidence_m - stall_checkpoint_m
                        >= stall_min_progress_m
                    ):
                        stall_checkpoint_m = progress_evidence_m
                        stall_checkpoint_at = now
                    elif (
                        stall_checkpoint_at is not None
                        and now - stall_checkpoint_at >= stall_timeout_s
                    ):
                        self.motion.stop()
                        stall_recovery_count += 1
                        if stall_recovery_count > stall_recovery_limit:
                            print(
                                "[placement-nav] approach still stalled; "
                                "restart recovery cycle without leaving substage "
                                f"attempts={stall_recovery_count} "
                                f"odom={measured_forward_m:.3f}m "
                                f"ultrasound_drop={ultrasound_progress_m:.3f}m",
                                flush=True,
                            )
                            stall_recovery_count = 0
                        print(
                            "[placement-nav] approach progress stalled; "
                            "reassert autonomous mode and retry "
                            f"attempt={stall_recovery_count}/"
                            f"{stall_recovery_limit} "
                            f"odom={measured_forward_m:.3f}m "
                            f"ultrasound_drop={ultrasound_progress_m:.3f}m",
                            flush=True,
                        )
                        try:
                            self.motion.set_autonomous()
                        except Exception as exc:
                            print(
                                "[placement-nav] recovery autonomous-mode "
                                f"warning: {exc}",
                                flush=True,
                            )
                        time.sleep(recovery_pause_s)
                        active_speed_mps = min(
                            active_speed_mps,
                            recovery_speed_mps,
                        )
                        command_started_at = None
                        stall_checkpoint_at = None
                        stall_checkpoint_m = progress_evidence_m
                        continue
                command_speed_mps = active_speed_mps
                if front_distance_m <= creep_distance_m:
                    command_speed_mps = min(
                        command_speed_mps,
                        creep_speed_mps,
                    )
                elif front_distance_m <= slow_distance_m:
                    command_speed_mps = min(
                        command_speed_mps,
                        slow_speed_mps,
                    )
                self.motion.move(command_speed_mps, 0.0, 0.0)
                if command_started_at is None:
                    command_started_at = time.monotonic()
                    stall_checkpoint_at = command_started_at
                    stall_checkpoint_m = progress_evidence_m
                time.sleep(command_period_s)
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            self._controlled_box_approach_active = previous_controlled_approach
            stop_error: Optional[Exception] = None
            pose_error: Optional[Exception] = None
            record_error: Optional[Exception] = None
            try:
                self.motion.stop()
            except Exception as exc:
                stop_error = exc
            if not self.context.dry_run:
                try:
                    measured_forward_m, _lateral_m = (
                        self._project_placement_motion(
                            start_pose,
                            self.state_reader.pose(),
                        )
                    )
                    measured_forward_m = max(0.0, measured_forward_m)
                except Exception as exc:
                    pose_error = exc
            if measured_forward_m > 1e-9:
                try:
                    navigator.record_motion(action, measured_forward_m)
                except Exception as exc:
                    record_error = exc
            try:
                self._append_placement_navigation_event(
                    {
                        "event": "continuous_ultrasound_approach",
                        "target_letter": target_letter,
                        "sensor": dict(self._placement_last_sensor_evidence),
                        "measured_distance_m": measured_forward_m,
                        "ultrasound_drop_m": ultrasound_progress_m,
                        "progress_evidence_m": progress_evidence_m,
                        "stall_recovery_count": stall_recovery_count,
                        "initial_front_m": start_front_m,
                        "initial_ultrasound_m": initial_ultrasound_m,
                        "initial_visual_distance_m": initial_visual_distance_m,
                        "odometry_stop_distance_m": odometry_stop_distance_m,
                        "echo_loss_fallback": echo_loss_active,
                        "ultrasound_stuck_fallback": (
                            ultrasound_stuck_fallback_active
                        ),
                        "cumulative": {
                            "forward_m": navigator.forward_travel_m,
                            "lateral_m": navigator.lateral_travel_m,
                            "net_lateral_m": navigator.net_lateral_m,
                        },
                        "result": (
                            "error" if primary_error is not None else "complete"
                        ),
                    }
                )
            except Exception as log_error:
                if primary_error is None:
                    raise MissionAbort(
                        f"placement approach evidence failed: {log_error}"
                    ) from log_error
            if primary_error is None:
                if stop_error is not None:
                    raise MissionAbort(
                        f"placement continuous approach stop failed: {stop_error}"
                    ) from stop_error
                if pose_error is not None:
                    raise MissionAbort(
                        "placement continuous approach final odometry failed: "
                        f"{pose_error}"
                    ) from pose_error
                if record_error is not None:
                    raise MissionAbort(
                        "placement continuous approach odometry rejected: "
                        f"{record_error}"
                    ) from record_error

    def _ensure_post_forward_placement_yaw_alignment(self) -> None:
        if self.context.placement_post_forward_yaw_attempted:
            return
        self.context.placement_post_forward_yaw_attempted = True
        aligned = False
        failure_reason: Optional[str] = None
        try:
            self.motion.stop()
            aligned = bool(self._align_placement_row_yaw())
        except Exception as exc:
            failure_reason = f"{type(exc).__name__}: {exc}"
            print(
                "[placement-yaw] warning: post-forward alignment raised "
                f"{failure_reason}; continue with IMU yaw hold",
                flush=True,
            )
        finally:
            self.context.placement_post_forward_yaw_ok = aligned
            try:
                self.motion.stop()
            except Exception as exc:
                print(
                    "[placement-yaw] warning: post-forward stop failed: "
                    f"{exc}",
                    flush=True,
                )
        self.context.placement_search_front_target_m = None
        self._reset_placement_front_filter()
        self._append_placement_navigation_event(
            {
                "event": "post_forward_yaw_alignment",
                "ok": aligned,
                "failure_reason": failure_reason,
            }
        )

    def _run_placement_letter_navigation_loop(
        self,
        target_letter: str,
        navigator: PlacementLetterNavigator,
    ) -> float:
        configured = self.config["placement_letter_navigation"]
        started_at = time.monotonic()
        if not self.context.placement_ultrasound_approach_complete:
            self._run_forced_placement_forward(
                target_letter=target_letter,
                navigator=navigator,
                started_at=started_at,
            )
            self._ensure_post_forward_placement_yaw_alignment()
            self._capture_placement_search_front_target()
        if self.context.placement_letter_centered_complete:
            if not self.context.placement_ultrasound_approach_complete:
                self.motion.stop()
                try:
                    self._run_placement_ultrasound_approach(
                        target_letter=target_letter,
                        navigator=navigator,
                        started_at=time.monotonic(),
                    )
                except BaseException:
                    self.context.placement_letter_centered_complete = False
                    self.context.placement_search_front_target_m = None
                    raise
                self.context.placement_ultrasound_approach_complete = True
            return navigator.net_lateral_m
        observation, detected = self._capture_placement_navigation_frame(
            target_letter=target_letter,
            frame_sequence=0,
            started_at=started_at,
        )
        while True:
            action = navigator.decide(observation)
            result = (
                "complete"
                if action.kind == ActionKind.COMPLETE
                else "fail"
                if action.kind == ActionKind.FAIL
                else "in_progress"
            )
            self._write_placement_frame_evidence(
                observation,
                detected,
                target_letter,
                action,
                navigator,
                result=result,
            )
            if action.kind == ActionKind.COMPLETE:
                self.motion.stop()
                self.context.placement_letter_centered_complete = True
                try:
                    self._run_placement_ultrasound_approach(
                        target_letter=target_letter,
                        navigator=navigator,
                        started_at=time.monotonic(),
                    )
                except BaseException:
                    self.context.placement_letter_centered_complete = False
                    self.context.placement_search_front_target_m = None
                    raise
                self.context.placement_ultrasound_approach_complete = True
                return navigator.net_lateral_m
            if action.kind == ActionKind.FAIL:
                print(
                    "[placement-nav] navigator requested recovery; keep current "
                    f"substage reason={action.reason}",
                    flush=True,
                )
                observation, detected = self._capture_placement_navigation_frame(
                    target_letter=target_letter,
                    frame_sequence=observation.frame_sequence,
                    started_at=started_at,
                )
                continue
            if action.kind == ActionKind.RETRY:
                observation, detected = self._capture_placement_navigation_frame(
                    target_letter=target_letter,
                    frame_sequence=observation.frame_sequence,
                    started_at=started_at,
                )
                continue
            if action.kind == ActionKind.STRAFE:
                self._require_placement_navigation_time_remaining(started_at)
            source_observation = observation
            if action.kind == ActionKind.STRAFE:
                try:
                    measured = self._execute_placement_navigation_motion(
                        action,
                        navigator=navigator,
                    )
                except (KeyboardInterrupt, SystemExit):
                    raise
                except BaseException as exc:
                    try:
                        self.motion.stop()
                    except Exception as stop_exc:
                        print(
                            "[placement-nav] strafe-recovery stop warning: "
                            f"{stop_exc}",
                            flush=True,
                        )
                    print(
                        "[placement-nav] strafe warning; preserve checkpoint "
                        f"and retry current search substage: {exc}",
                        flush=True,
                    )
                    try:
                        self.motion.set_autonomous()
                    except Exception as mode_exc:
                        print(
                            "[placement-nav] strafe-recovery autonomous warning: "
                            f"{mode_exc}",
                            flush=True,
                        )
                    time.sleep(
                        float(configured.get("motion_recovery_pause_s", 0.30))
                    )
                    observation, detected = (
                        self._capture_placement_navigation_frame(
                            target_letter=target_letter,
                            frame_sequence=source_observation.frame_sequence,
                            started_at=started_at,
                        )
                    )
                    continue
                observation, detected = self._capture_placement_navigation_frame(
                    target_letter=target_letter,
                    frame_sequence=source_observation.frame_sequence,
                    started_at=started_at,
                )
                self._write_placement_motion_event(
                    action,
                    measured,
                    navigator,
                    source_observation,
                    observation,
                )
                configured = self.config["placement_letter_navigation"]
                front_target_m = self.context.placement_search_front_target_m
                boundary_delta_m = float(
                    configured.get("search_hold_boundary_delta_m", 0.20)
                )
                boundary_reached = (
                    front_target_m is not None
                    and abs(
                        observation.front_distance_m - float(front_target_m)
                    )
                    >= boundary_delta_m
                )
                front_restored = False
                if not boundary_reached:
                    try:
                        front_restored = (
                            self._restore_placement_search_front_distance()
                        )
                    except PlacementSearchBoundary:
                        boundary_reached = True
                if boundary_reached:
                    if abs(measured) >= float(configured.get(
                        "motion_stall_min_progress_m",
                        0.01,
                    )):
                        recovery_distance_m = -measured
                        recovery_action = NavigationAction(
                            ActionKind.STRAFE,
                            "reverse_after_front_echo_boundary",
                            distance_m=recovery_distance_m,
                            vy_mps=(
                                (1.0 if recovery_distance_m > 0.0 else -1.0)
                                * float(configured["lateral_speed_mps"])
                            ),
                        )
                        recovery_source = observation
                        recovery_measured = (
                            self._execute_placement_navigation_motion(
                                recovery_action,
                                navigator=navigator,
                            )
                        )
                        observation, detected = (
                            self._capture_placement_navigation_frame(
                                target_letter=target_letter,
                                frame_sequence=recovery_source.frame_sequence,
                                started_at=started_at,
                            )
                        )
                        self._write_placement_motion_event(
                            recovery_action,
                            recovery_measured,
                            navigator,
                            recovery_source,
                            observation,
                        )
                    else:
                        navigator.request_lateral_recovery(action)
                    continue
                if front_restored:
                    observation, detected = (
                        self._capture_placement_navigation_frame(
                            target_letter=target_letter,
                            frame_sequence=observation.frame_sequence,
                            started_at=started_at,
                        )
                    )
                continue
            if action.kind in (ActionKind.FORWARD, ActionKind.FINAL_APPROACH):
                raise MissionAbort(
                    "placement navigator requested forbidden forward motion "
                    "during the lateral-search stage after the fixed 1.30m "
                    "post-turn forward segment"
                )
            raise MissionAbort(f"unsupported placement action: {action.kind}")

    def _run_placement_letter_navigator(self, target_letter: str) -> float:
        primary_error: Optional[BaseException] = None
        navigator: Optional[PlacementLetterNavigator] = None

        def rollback_error() -> Optional[str]:
            if navigator is not None:
                print(
                    "[placement-nav] preserve placement substage and current "
                    "lateral position for retry",
                    flush=True,
                )
            return None

        try:
            configured = self.config["placement_letter_navigation"]
            if not bool(configured.get("enabled", False)):
                raise MissionAbort("placement letter navigation is disabled")
            navigator = PlacementLetterNavigator(
                target_letter,
                self._placement_letter_navigation_config(),
                preferred_target_lateral_m=(
                    self._cached_placement_target_lateral_m(target_letter)
                ),
                final_approach_completed=(
                    self.context.placement_final_approach_complete
                ),
                final_approach_progress_m=(
                    self.context.placement_final_approach_progress_m
                ),
            )
            navigator.net_lateral_m = float(
                self.context.placement_navigation_net_lateral_m
            )
            navigator.lateral_travel_m = float(
                self.context.placement_navigation_lateral_travel_m
            )
            navigator.bilateral_search_phase = str(
                self.context.placement_navigation_search_phase
            )
            navigator.last_lateral_sign = (
                self.context.placement_navigation_last_lateral_sign
            )
            navigator.last_geometry_lateral_sign = (
                self.context.placement_navigation_last_geometry_sign
            )
            navigator.pending_recovery_lateral_sign = (
                self.context.placement_navigation_pending_recovery_sign
            )
            navigator.zero_progress_strafe_count = int(
                self.context.placement_navigation_zero_progress_count
            )
            if not math.isfinite(navigator.lateral_travel_m) or navigator.lateral_travel_m < 0.0:
                raise MissionAbort(
                    "placement lateral checkpoint is invalid"
                )
            if navigator.bilateral_search_phase not in {"left", "right"}:
                raise MissionAbort("placement search-phase checkpoint is invalid")
            if self.context.dry_run:
                action = None
                for sequence in range(1, 8):
                    action = navigator.decide(
                        NavigationObservation(
                            sequence,
                            1000,
                            (LetterCandidate(target_letter, 500.0, 1.0),),
                            float(configured["front_stop_distance_m"]),
                            sequence * 0.01,
                        )
                    )
                    if action.kind == ActionKind.COMPLETE:
                        break
                if action is None or action.kind != ActionKind.COMPLETE:
                    raise MissionAbort(
                        "dry-run placement target did not pass center confirmation"
                    )
                return navigator.net_lateral_m

            try:
                self._placement_navigation_run_dir = (
                    self._create_placement_navigation_run_dir()
                )
            except Exception as exc:
                self._placement_navigation_run_dir = None
                print(
                    f"[placement-nav] evidence directory warning: {exc}",
                    flush=True,
                )
            self._placement_navigation_events_path = (
                None
                if self._placement_navigation_run_dir is None
                else self._placement_navigation_run_dir / "events.jsonl"
            )
            self._placement_last_camera_frame_at = None
            self._placement_last_camera_frame_id = None
            self._placement_last_camera_signature = None
            ensure_camera = getattr(self.wide_camera, "ensure_running", None)
            camera_attempt = 0
            while callable(ensure_camera):
                try:
                    if ensure_camera("placement_navigation_reuse") is not False:
                        break
                except (KeyboardInterrupt, SystemExit):
                    raise
                except Exception as exc:
                    camera_attempt += 1
                    if camera_attempt == 1 or camera_attempt % 10 == 0:
                        print(
                            "[placement-nav] camera startup recovery "
                            f"attempt={camera_attempt} reason={exc}",
                            flush=True,
                        )
                else:
                    camera_attempt += 1
                    if camera_attempt == 1 or camera_attempt % 10 == 0:
                        print(
                            "[placement-nav] camera not running; retry in place "
                            f"attempt={camera_attempt}",
                            flush=True,
                        )
                try:
                    self.motion.stop()
                except Exception as exc:
                    print(
                        f"[placement-nav] camera-recovery stop warning: {exc}",
                        flush=True,
                    )
                time.sleep(float(configured.get("motion_recovery_pause_s", 0.30)))
            return self._run_placement_letter_navigation_loop(
                target_letter,
                navigator,
            )

        except MissionAbort as exc:
            rollback_failure = rollback_error()
            converted = (
                exc
                if rollback_failure is None
                else MissionAbort(
                    f"{exc}; placement_search_origin_return_failed:"
                    f"{rollback_failure}"
                )
            )
            primary_error = converted
            self._append_placement_terminal_event(converted)
            if converted is exc:
                raise
            raise converted from exc
        except Exception as exc:
            rollback_failure = rollback_error()
            converted = MissionAbort(
                f"placement navigation failed: {exc}"
                + (
                    ""
                    if rollback_failure is None
                    else "; placement_search_origin_return_failed:"
                    f"{rollback_failure}"
                )
            )
            primary_error = converted
            self._append_placement_terminal_event(converted)
            raise converted from exc
        except BaseException as exc:
            primary_error = exc
            self._append_placement_terminal_event(exc)
            raise
        finally:
            try:
                self.motion.stop()
            except Exception as stop_error:
                if primary_error is None:
                    raise MissionAbort(
                        f"placement navigation stop failed: {stop_error}"
                    ) from stop_error

    def _cached_placement_target_lateral_m(
        self,
        target_letter: str,
    ) -> Optional[float]:
        configured = self.config["placement_letter_navigation"]
        if not bool(configured.get("cached_geometry_enabled", True)):
            return None
        cached = self.context.placement_letter_lateral_m
        if target_letter in cached:
            return float(cached[target_letter])
        if not cached:
            return None
        reference_letter, reference_lateral_m = next(iter(cached.items()))
        letter_order = tuple(str(value) for value in configured["letter_order"])
        spacing_m = float(configured["letter_spacing_m"])
        predicted = float(reference_lateral_m) + (
            letter_order.index(reference_letter) - letter_order.index(target_letter)
        ) * spacing_m
        search_limit_m = float(configured["lateral_search_each_side_m"])
        if abs(predicted) > search_limit_m + 1e-9:
            return None
        print(
            "[placement-nav] cached row geometry "
            f"reference={reference_letter}@{reference_lateral_m:.3f}m "
            f"target={target_letter}@{predicted:.3f}m",
            flush=True,
        )
        return predicted

    def _execute_placement_letter_approach(self) -> None:
        target_letter = self.context.target_letter
        primary_error: Optional[BaseException] = None
        try:
            if target_letter not in self.INSPECTION_LETTERS:
                raise MissionAbort(
                    "placement letter approach has no valid target letter"
                )
            try:
                net_lateral_m = self._run_placement_letter_navigator(target_letter)
            except MissionAbort:
                raise
            except Exception as exc:
                raise MissionAbort(
                    f"placement letter approach failed: {exc}"
                ) from exc
            if not self.context.placed_letters:
                self.context.first_outbound_lane_strafe_m = float(net_lateral_m)
            self.context.placement_letter_lateral_m[target_letter] = float(
                net_lateral_m
            )
            self._placement_letter_approach_succeeded = True
            self.context.placement_visual_approach_complete = True
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            try:
                self.motion.stop()
            except Exception as stop_error:
                if primary_error is None:
                    raise MissionAbort(
                        f"placement approach stop failed: {stop_error}"
                    ) from stop_error

    def _execute_placement_lane_strafe(self) -> None:
        target_letter = self.context.target_letter
        if target_letter is None:
            raise MissionAbort("placement lane strafe has no target letter")
        transfer = self.config.get("pickup_transfer", {})
        offsets = transfer.get("lane_offsets_m", {})
        if not isinstance(offsets, dict) or target_letter not in offsets:
            raise MissionAbort(f"placement lane offset is missing for {target_letter}")
        requested = float(offsets[target_letter])
        result = self._get_pickup_transfer_controller().move_lane(requested)
        print(
            f"[pickup-transfer] placement lane={target_letter} "
            f"requested={requested:.3f}m measured={result.measured_distance_m:.3f}m "
            f"ok={result.ok} reason={result.reason}"
        )
        if not result.ok:
            raise MissionAbort(
                f"placement lane strafe failed for {target_letter}: {result.reason}"
            )
        if not self.context.placed_letters:
            physical_left_sign = int(
                self.config.get("placement_letter_navigation", {}).get(
                    "physical_left_strafe_sign",
                    1,
                )
            )
            self.context.first_outbound_lane_strafe_m = (
                float(result.measured_distance_m) * physical_left_sign
            )

    def _align_placement_row_yaw(self) -> bool:
        yaw_config = dict(self.config.get("placement_yaw_alignment", {}))
        if not bool(yaw_config.get("enabled", False)):
            print("[placement-yaw] disabled")
            return True
        if self.context.dry_run:
            print("[placement-yaw] dry-run visible-letter row alignment")
            return True
        if self.placement_row_yaw_aligner is None:
            project_root = Path(__file__).resolve().parent.parent
            calibration_path = Path(str(self.config["camera"]["wide_calibration"]))
            if not calibration_path.is_absolute():
                calibration_path = project_root / calibration_path
            run_path = Path(
                str(yaw_config.get("run_log_dir", "placement_yaw_alignment_runs"))
            )
            if not run_path.is_absolute():
                yaw_config["run_log_dir"] = str(project_root / run_path)
            box_config = dict(self.config.get("box_center_alignment", {}))
            min_span = float(yaw_config.get("min_row_span_fraction", 0.38))

            def detector(frame):
                return detect_placement_row_parallel(
                    frame,
                    box_config,
                    min_row_span_fraction=min_span,
                )

            self.placement_row_yaw_aligner = WideBoxAligner(
                camera=self.wide_camera,
                undistorter=WideCameraUndistorter.from_file(calibration_path),
                motion=self.motion,
                config=yaw_config,
                detector=detector,
            )
        result = self.placement_row_yaw_aligner.run()
        print(
            "[placement-yaw] visible-letter row alignment "
            f"ok={result.ok} reason={result.reason} "
            f"error={result.initial_error_deg}->{result.final_error_deg} "
            f"corrections={result.correction_count}"
        )
        if not result.ok:
            print(
                "[placement-yaw] warning: row yaw alignment unavailable; "
                f"continue placement lane selection reason={result.reason}"
            )
            return False
        return True

    def _execute_pickup_lane_restore(self) -> None:
        recorded = self.context.first_outbound_lane_strafe_m
        if recorded is None or not math.isfinite(float(recorded)):
            raise MissionAbort("first outbound lane strafe record is unavailable")
        limit = float(
            self.config.get("pickup_transfer", {}).get(
                "max_recorded_lane_strafe_m",
                1.05,
            )
        )
        if abs(float(recorded)) > limit + 1e-9:
            raise MissionAbort(
                "first outbound lane strafe record exceeds limit: "
                f"{float(recorded):.3f}m > {limit:.3f}m"
            )
        physical_recorded = float(recorded)
        physical_left_sign = int(
            self.config.get("placement_letter_navigation", {}).get(
                "physical_left_strafe_sign",
                1,
            )
        )
        command = physical_recorded * physical_left_sign
        result = self._get_pickup_transfer_controller().move_lane(command)
        print(
            "[pickup-transfer] restore pickup lane "
            f"physical_recorded={physical_recorded:.3f}m "
            f"command={command:.3f}m "
            f"command_measured={result.measured_distance_m:.3f}m "
            f"ok={result.ok} reason={result.reason}"
        )
        if not result.ok:
            raise MissionAbort(f"pickup lane restore failed: {result.reason}")

    def _execute_pickup_forward_restore(self) -> None:
        recorded = self.context.first_outbound_forward_m
        if recorded is None or not math.isfinite(float(recorded)):
            raise MissionAbort("first outbound forward record is unavailable")
        target_m = float(recorded)
        if not (0.20 <= target_m <= 1.60):
            raise MissionAbort(
                "first outbound forward record is invalid: "
                f"{target_m:.3f}m"
            )
        speed_mps = 0.08
        duration_s = target_m / speed_mps
        print(
            "[pickup-transfer] restore pickup forward "
            f"recorded={target_m:.3f}m duration={duration_s:.2f}s",
            flush=True,
        )
        if self.context.dry_run:
            return
        start_pose = self.state_reader.pose()
        started_at = time.monotonic()
        elapsed_s = 0.0
        measured_m = 0.0
        stop_reason = "timed_odometry_fallback"
        primary_error: Optional[BaseException] = None
        try:
            self.motion.set_autonomous()
            while elapsed_s < duration_s:
                try:
                    measured_m, _lateral_m = self._project_placement_motion(
                        start_pose,
                        self.state_reader.pose(),
                    )
                    measured_m = max(0.0, measured_m)
                except Exception:
                    measured_m = 0.0
                if measured_m >= target_m - MOTION_MEASUREMENT_TOLERANCE_M:
                    stop_reason = "target_reached_odometry"
                    break
                self.motion.move(speed_mps, 0.0, 0.0)
                time.sleep(min(0.05, max(0.0, duration_s - elapsed_s)))
                elapsed_s = max(0.0, time.monotonic() - started_at)
        except BaseException as exc:
            primary_error = exc
            raise
        finally:
            try:
                self.motion.stop()
            except Exception as exc:
                if primary_error is None:
                    raise MissionAbort(
                        f"pickup forward restore stop failed: {exc}"
                    ) from exc
        print(
            "[pickup-transfer] restore pickup forward complete "
            f"reason={stop_reason} odom={measured_m:.3f}m "
            f"elapsed={elapsed_s:.2f}s",
            flush=True,
        )

    def _try_align_second_pickup_box_center(self) -> bool:
        center_config = self.config.get("box_center_alignment", {})
        if not bool(center_config.get("enabled", False)):
            return True
        try:
            result = self._run_box_center_alignment("pickup")
        except Exception as exc:
            print(
                "[box-center] warning: pickup box-center alignment raised "
                f"{exc}; continue existing red-bar alignment"
            )
            return False
        print(
            "[box-center] pickup alignment "
            f"ok={result.ok} reason={result.reason} "
            f"net_strafe={result.net_strafe_m:.3f}m"
        )
        if not result.ok:
            print(
                "[box-center] warning: pickup box center unavailable; "
                "continue existing yaw, 28cm approach and red-bar alignment"
            )
        return bool(result.ok)

    def _retry_grasp(self, distance_mm: float) -> bool:
        retries = int(self.config["arm"]["max_retries"])
        for attempt in range(1, retries + 2):
            print(f"[mission] grasp attempt {attempt}")
            if not self._pregrasp_ultrasound_ready():
                print("[mission] grasp blocked by pregrasp ultrasound safety gate")
                break
            result = self.arm.grasp_red_bar(distance_mm)
            ok = (
                result.ok and result.object_held
                if isinstance(result, ArmTaskResult)
                else result is not False
            )
            if ok:
                time.sleep(3.0 if not self.context.dry_run else 0.2)
                return True
            if isinstance(result, ArmTaskResult):
                print(f"[mission] grasp failed stage={result.stage} reason={result.reason}")
                if result.object_held:
                    self.context.carried_bar = True
                    raise MissionAbort(
                        "grasp transport pose failed after gripper closed; "
                        "object is held and manual recovery is required"
                    )
                if result.requires_power_cycle:
                    print("[mission] grasp retry blocked: arm power cycle required")
                    break
                if (
                    result.stage == "VISUAL_ALIGN"
                    and result.feedback in {"target_left", "target_right"}
                    and attempt <= retries
                ):
                    print("[mission] horizontal grasp rejection; realign and retry")
                    self.motion.stop()
                    self._require_arm_result(
                        self.arm.stow(),
                        "grasp retry moving pose",
                    )
                    try:
                        aligned = self._run_pregrasp_base_sequence()
                    except Exception as exc:
                        print(f"[pregrasp] retry lateral alignment failed: {exc}")
                        aligned = False
                    finally:
                        self.motion.stop()
                    if not aligned:
                        break
                    self.motion.stop()
                    self._require_arm_result(
                        self.arm.camera_pose(),
                        "grasp retry ready pose",
                    )
                    self._settle_after_pregrasp_stop()
        return False

    def _next_target_letter(self) -> Optional[str]:
        for letter in self.context.anomalous_letters():
            if letter not in self.context.placed_letters:
                return letter
        fallback_candidates = sorted(
            (
                record
                for letter, record in self.context.records.items()
                if letter not in self.context.placed_letters
            ),
            key=lambda record: (
                record.source_camera != "default_fallback",
                record.confidence,
                record.letter,
            ),
        )
        if fallback_candidates:
            selected = fallback_candidates[0]
            print(
                "[mission] no remaining anomaly target; continue with "
                f"fallback area={selected.letter} confidence={selected.confidence:.3f} "
                f"source={selected.source_camera or 'unknown'}"
            )
            return selected.letter
        return None

    def _round_result_allows_pickup(self) -> bool:
        inspection_cfg = self.config.get("inspection", {})
        if not bool(inspection_cfg.get("gate_pickup_on_round_result", True)):
            return True
        round_result_path = Path(str(inspection_cfg.get("round_result_path", "round_result.json")))
        if self.context.records:
            rebalanced_letters = self._rebalance_default_fallback_records()
            data = build_round_result(
                self.context.records,
                source_camera="front",
                run_id=self.context.run_id,
            )
            if rebalanced_letters and not self.context.dry_run:
                write_json_atomic(round_result_path, data)
                print(
                    "[inspect] persisted fallback rebalance "
                    f"areas={rebalanced_letters} round={round_result_path}"
                )
            if not data.get("ready"):
                print(
                    "[mission] current inspection result blocks pickup: "
                    f"block_reason={data.get('block_reason')} unknown_areas={data.get('unknown_areas')}"
                )
                return False
            return True
        if not round_result_path.exists():
            print(f"[mission] round_result blocks pickup: missing path={round_result_path}")
            return False
        gate = evaluate_round_gate(round_result_path, expected_run_id=self.context.run_id)
        if not gate.allowed:
            print(
                "[mission] round_result blocks pickup: "
                f"block_reason={gate.block_reason} unknown_areas={gate.unknown_areas}"
            )
            return False
        records = records_from_round_result(gate.data)
        if records:
            self.context.records.update(records)
        for letter in gate.abnormal_areas:
            self.context.records.setdefault(letter, InspectionRecord(letter, "", "异常", 0.0, -1))
        return True

    def _check_safety(self) -> None:
        error = self.state_reader.safety_error(require_fresh=not self.context.dry_run)
        if error:
            try:
                self.motion.stop()
            except Exception as stop_error:
                print(
                    f"[mission] safety hold stop failed: {stop_error}",
                    flush=True,
                )
            raise MissionAbort(f"robot state failed safety check: {error}")

    def _motion_guard(self, vx: float, vy: float, wz: float) -> None:
        if self.context.dry_run:
            return
        safety = self.config["safety"]
        require_ultrasound = (
            vx > 0.0
            and not self.ignore_obstacles
            and not self.ignore_ultrasound_obstacle
            and bool(safety.get("use_ultrasound_obstacle", True))
        )
        error = self.state_reader.safety_error(require_ultrasound=require_ultrasound, require_fresh=True)
        if error:
            raise MissionAbort(f"motion guard rejected command: {error}")
        if require_ultrasound:
            value = self.state_reader.state.front_ultrasound_m
            if self._placement_route_active:
                filtered_reader = getattr(
                    self.state_reader,
                    "filtered_front_ultrasound_m",
                    None,
                )
                if callable(filtered_reader):
                    value = filtered_reader(
                        float(
                            safety.get(
                                "placement_front_filter_window_s",
                                0.8,
                            )
                        )
                    )
            stop_distance = self._front_stop_distance()
            min_valid = float(safety.get("front_ultrasound_min_valid_m", 0.03))
            if value is None or not math.isfinite(float(value)) or float(value) < min_valid:
                raise MissionAbort(f"motion guard received invalid ultrasound value: {value!r}")
            if (
                not self._controlled_box_approach_active
                and not self._placement_forced_forward_active
                and float(value) <= stop_distance
            ):
                raise ForwardMotionGuardStop(
                    "motion guard stopped forward command: "
                    f"ultrasound={float(value):.2f}m "
                    f"threshold={stop_distance:.2f}m state={self.state.name}"
                )

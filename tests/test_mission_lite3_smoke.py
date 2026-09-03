from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import math
import struct
import sys
import tempfile
import time
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import mission_lite3.config_loader as config_loader
from mission_lite3.audio import AudioReporter, build_announcement, clip_name_for_result, resolve_audio_clip
from mission_lite3.arm.lite_arm import ArmTaskResult, LiteArmController
from mission_lite3.arm.runtime import arm_task
from mission_lite3.arm.runtime import test as arm_test
from mission_lite3.camera import CameraSource
from mission_lite3.config_loader import ConfigError, load_config, validate_config
from mission_lite3.lite3_motion import HEARTBEAT, VEL_FORWARD, Lite3MotionController, pack_simple_command, pack_velocity_command
from mission_lite3.mission import (
    ForwardMotionGuardStop,
    LargeQuadrupedMission,
    MissionAbort,
    MissionContext,
    MissionState,
    ObstacleCheck,
)
from mission_lite3.persistent_camera import PersistentLatestFrameReader
from mission_lite3.route_validation import (
    RoutePose,
    route_boundary_errors,
    simulate_route_actions,
    simulate_route_sequence,
)
from mission_lite3.round_result import build_round_result, evaluate_round_gate, load_round_result
from mission_lite3.state_reader import StateReader
from mission_lite3.vision import InspectionRecord
from mission_lite3.vision.common import StableVote
from mission_lite3.vision.pipeline import VisionPipeline, runtime_result_to_record_fields
from run_live_inspection import LiveInspectionRunner, build_parser as build_live_parser


class MissionLite3SmokeTest(unittest.TestCase):
    def test_full_mission_launcher_holds_shared_motion_lock(self) -> None:
        attributes = Path(".gitattributes").read_text(encoding="utf-8")
        self.assertIn("scripts/run_full_mission.sh text eol=lf", attributes)
        raw_script = Path("scripts/run_full_mission.sh").read_bytes()
        self.assertTrue(raw_script.startswith(b"#!/usr/bin/env bash\n"))
        self.assertNotIn(b"\r\n", raw_script)
        script = raw_script.decode("utf-8")
        self.assertIn(
            'MOTION_LOCK="${LITE3_MOTION_LOCK:-/tmp/lite3_motion_test.lock}"',
            script,
        )
        self.assertIn(
            'MOTION_LOCK_WAIT_S="${LITE3_MOTION_LOCK_WAIT_S:-5}"',
            script,
        )
        open_lock = 'exec 9>"${MOTION_LOCK}"'
        acquire_lock = 'if ! flock -w "${MOTION_LOCK_WAIT_S}" 9; then'
        lock_failure = (
            'if ! flock -w "${MOTION_LOCK_WAIT_S}" 9; then\n'
            '  echo "[full-mission] motion lock timeout: ${MOTION_LOCK}" >&2\n'
            "  exit 3\n"
            "fi"
        )
        run_mission = 'python3 -u -m mission_lite3.run_mission --robot "$@"'
        self.assertIn(open_lock, script)
        self.assertIn(acquire_lock, script)
        self.assertIn(lock_failure, script)
        self.assertIn(run_mission, script)
        self.assertIn('| tee -a "${MISSION_LOG_PATH}"', script)
        self.assertIn('PIPE_RESULTS=("${PIPESTATUS[@]}")', script)
        self.assertIn("MISSION_STATUS=${PIPE_RESULTS[0]}", script)
        self.assertIn("LOG_STATUS=${PIPE_RESULTS[1]}", script)
        self.assertIn('if [[ "${LOG_STATUS}" -ne 0 ]]; then', script)
        self.assertIn('exit "${MISSION_STATUS}"', script)
        self.assertLess(script.index(open_lock), script.index(acquire_lock))
        self.assertLess(script.index(acquire_lock), script.index(run_mission))

    def test_pickup_transfer_launcher_holds_shared_motion_lock(self) -> None:
        attributes = Path(".gitattributes").read_text(encoding="utf-8")
        self.assertIn(
            "scripts/run_pickup_transfer_mission.sh text eol=lf",
            attributes,
        )
        raw_script = Path("scripts/run_pickup_transfer_mission.sh").read_bytes()
        self.assertTrue(raw_script.startswith(b"#!/usr/bin/env bash\n"))
        self.assertNotIn(b"\r\n", raw_script)
        script = raw_script.decode("utf-8")
        self.assertIn(
            'MOTION_LOCK="${LITE3_MOTION_LOCK:-/tmp/lite3_motion_test.lock}"',
            script,
        )
        self.assertIn(
            'MOTION_LOCK_WAIT_S="${LITE3_MOTION_LOCK_WAIT_S:-5}"',
            script,
        )
        open_lock = 'exec 9>"${MOTION_LOCK}"'
        acquire_lock = 'if ! flock -w "${MOTION_LOCK_WAIT_S}" 9; then'
        lock_failure = (
            'if ! flock -w "${MOTION_LOCK_WAIT_S}" 9; then\n'
            '    echo "[pickup-transfer] motion lock timeout: ${MOTION_LOCK}" >&2\n'
            "    exit 3\n"
            "  fi"
        )
        run_mission = (
            'exec python3 -u -m mission_lite3.tools.pickup_transfer_mission "$@"'
        )
        self.assertIn(open_lock, script)
        self.assertIn(acquire_lock, script)
        self.assertIn(lock_failure, script)
        self.assertIn(run_mission, script)
        self.assertLess(script.index(open_lock), script.index(acquire_lock))
        self.assertLess(script.index(acquire_lock), script.index(run_mission))

    def test_readme_real_mission_commands_use_locked_launchers(self) -> None:
        for path in (Path("README.md"), Path("mission_lite3/README.md")):
            with self.subTest(path=path):
                readme = path.read_text(encoding="utf-8")
                self.assertNotIn(
                    "python -m mission_lite3.run_mission --robot",
                    readme,
                )
                self.assertNotIn(
                    "python3 -m mission_lite3.run_mission --robot",
                    readme,
                )
                self.assertIn("scripts/run_full_mission.sh --skip-arm", readme)

        mission_readme = Path("mission_lite3/README.md").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "scripts/run_pickup_transfer_mission.sh --robot --yes",
            mission_readme,
        )

    def test_config_loads_required_sections(self) -> None:
        config = load_config()
        for section in (
            "network",
            "ros2",
            "camera",
            "motion",
            "safety",
            "fault_hold",
            "startup_avoidance",
            "vision",
            "inspection",
            "arm",
            "box_center_alignment",
            "placement_letter_navigation",
            "pickup_transfer",
            "audio",
        ):
            self.assertIn(section, config)
        self.assertEqual(config["network"]["motion_port"], 43893)
        self.assertEqual(config["ros2"]["cmd_vel_topic"], "/cmd_vel")
        self.assertEqual(config["ros2"]["odom_topic"], "/leg_odom2")
        self.assertEqual(config["safety"]["front_stop_distance_m"], 0.35)
        self.assertTrue(config["safety"]["use_ultrasound_obstacle"])
        self.assertFalse(config["safety"]["use_vision_obstacle"])
        self.assertEqual(
            config["fault_hold"],
            {
                "enabled": True,
                "poll_interval_s": 0.50,
                "recovery_stable_checks": 5,
                "resume_signal_path": "/tmp/lite3_fault_resume",
                "max_wait_s": 30.0,
                "max_retries_per_state": 2,
            },
        )
        self.assertEqual(
            config["startup_avoidance"]["fault_hold_retry_s"],
            0.50,
        )
        self.assertEqual(
            config["startup_avoidance"]["fault_hold_max_s"],
            30.0,
        )
        self.assertEqual(
            config["startup_avoidance"]["distance"],
            {
                "emergency_stop_m": 0.15,
                "front_trigger_m": 0.40,
                "side_trigger_m": 0.40,
                "ultrasound_window": 20,
            },
        )
        self.assertEqual(
            config["startup_avoidance"]["decision"]["finish_forward_m"],
            2.40,
        )
        self.assertEqual(config["inspection"]["front_stop_distance_m"], 0.28)
        self.assertEqual(
            config["ros2"]["ultrasound_topic"],
            "/us_publisher/front_distance",
        )
        self.assertEqual(
            config["ros2"]["rear_ultrasound_topic"],
            "/us_publisher/rear_distance",
        )
        self.assertEqual(config["vision"]["inspection_backend"], "runtime_meter_anchor")
        self.assertTrue(config["motion"]["assume_standing"])
        self.assertEqual(config["inspection"]["stop_dwell_seconds"], 8.0)
        self.assertEqual(config["vision"]["runtime_fast_accept_confidence"], 0.84)
        self.assertEqual(config["vision"]["runtime_fast_accept_margin"], 0.20)
        self.assertEqual(config["vision"]["runtime_best_candidate_confidence"], 0.82)
        self.assertTrue(config["inspection"]["use_wide_undistortion"])
        self.assertEqual(config["camera"]["frame_width"], 1280)
        self.assertEqual(config["camera"]["frame_height"], 720)
        self.assertEqual(config["camera"]["digital_zoom"], 1.0)
        self.assertEqual(config["arm"]["backend"], "runtime")
        self.assertEqual(
            config["arm"]["camera_device"],
            "/dev/v4l/by-id/usb-SXW_USB_Camera_200901010001-video-index0",
        )
        self.assertTrue(config["arm"]["calibration"].endswith("camera_calibration.json"))
        self.assertIn("pregrasp_red_align", config)
        self.assertEqual(
            config["pregrasp_red_align"]["strict_motion_min_linear_size_ratio"],
            0.70,
        )
        self.assertFalse(config["box_center_alignment"]["enabled"])
        self.assertTrue(config["pickup_transfer"]["enabled"])
        self.assertEqual(
            config["pickup_transfer"]["lane_offsets_m"],
            {"A": 1.0, "B": 0.5, "C": 0.0, "D": -0.5},
        )
        self.assertEqual(
            config["box_center_alignment"]["fallback_offsets_m"],
            {"A": 0.15, "B": -0.15, "C": -0.45, "D": -0.75},
        )
        self.assertEqual(config["arm"]["stow_command"], "moving-pose")
        self.assertIn("strip_detector_grasp_config.json", config["arm"]["runtime_config"])
        self.assertIn("moving_pose.json", config["arm"]["moving_pose"])
        self.assertEqual(config["audio"]["remote_gain_db"], 3.0)
        self.assertTrue(config["navigation"]["feedback_required"])
        self.assertTrue(config["navigation"]["translation_path_hold_enabled"])
        self.assertEqual(
            config["navigation"]["translation_max_cross_track_correction_mps"],
            0.04,
        )
        self.assertEqual(
            config["navigation"]["translation_max_wz_correction_rad_s"],
            0.12,
        )
        self.assertEqual(
            config["pregrasp_red_align"]["final_distance_min_m"],
            0.25,
        )
        self.assertTrue(
            config["pregrasp_red_align"]["target_search_enabled"]
        )
        self.assertEqual(
            config["pregrasp_red_align"]["target_search_max_distance_m"],
            3.00,
        )
        self.assertTrue(
            config["pregrasp_red_align"]["target_search_bilateral_enabled"]
        )
        self.assertTrue(
            config["pregrasp_red_align"]["target_search_until_found"]
        )
        self.assertEqual(
            config["pregrasp_red_align"]["target_search_each_side_m"],
            0.25,
        )
        self.assertEqual(
            config["pregrasp_red_align"]["target_search_center_band"],
            [0.3333333333, 0.6666666667],
        )
        self.assertEqual(
            config["pregrasp_red_align"]["preapproach_search_min_distance_m"],
            0.00,
        )
        self.assertEqual(
            config["pregrasp_red_align"]["target_search_speed_mps"],
            0.08,
        )
        self.assertEqual(
            config["pregrasp_red_align"]["target_search_step_seconds"],
            1.00,
        )
        self.assertEqual(
            config["pregrasp_red_align"]["target_search_settle_seconds"],
            0.00,
        )
        self.assertTrue(
            config["pregrasp_red_align"][
                "target_search_require_odom_progress"
            ]
        )
        self.assertEqual(
            config["pregrasp_red_align"]["target_search_min_progress_m"],
            0.015,
        )
        self.assertEqual(
            config["pregrasp_red_align"]["target_search_max_stalled_pulses"],
            3,
        )
        self.assertEqual(
            config["pregrasp_red_align"]["target_search_max_net_lateral_m"],
            0.25,
        )
        self.assertEqual(
            config["pregrasp_red_align"]["max_vx_correction_mps"],
            0.04,
        )

    def test_translation_path_hold_config_rejects_unsafe_values(self) -> None:
        invalid_values = (
            ("translation_path_hold_enabled", 1),
            ("translation_cross_track_kp_s", math.nan),
            ("translation_cross_track_deadband_m", 0.16),
            ("translation_max_cross_track_correction_mps", 0.21),
            ("translation_max_wz_correction_rad_s", 0.56),
            ("translation_yaw_deadband_deg", 5.1),
        )
        for key, value in invalid_values:
            with self.subTest(key=key, value=value):
                config = load_config()
                config["navigation"][key] = value
                with self.assertRaisesRegex(ConfigError, key):
                    validate_config(config)

    def test_fault_hold_config_rejects_invalid_values(self) -> None:
        invalid_values = (
            ("enabled", 1),
            ("poll_interval_s", True),
            ("poll_interval_s", -0.01),
            ("poll_interval_s", 60.01),
            ("poll_interval_s", math.nan),
            ("recovery_stable_checks", True),
            ("recovery_stable_checks", 0),
            ("recovery_stable_checks", 1.5),
            ("recovery_stable_checks", 1001),
            ("resume_signal_path", ""),
            ("resume_signal_path", "   "),
            ("resume_signal_path", None),
            ("max_wait_s", True),
            ("max_wait_s", 0.09),
            ("max_wait_s", 3600.01),
            ("max_wait_s", math.nan),
            ("max_retries_per_state", True),
            ("max_retries_per_state", -1),
            ("max_retries_per_state", 1.5),
            ("max_retries_per_state", 101),
        )
        for key, value in invalid_values:
            with self.subTest(key=key, value=value):
                config = load_config()
                config["fault_hold"][key] = value
                with self.assertRaisesRegex(ConfigError, f"fault_hold.{key}"):
                    validate_config(config)

        for value in (True, -0.01, 60.01, math.nan):
            with self.subTest(fault_hold_retry_s=value):
                config = load_config()
                config["startup_avoidance"]["fault_hold_retry_s"] = value
                with self.assertRaisesRegex(
                    ConfigError,
                    "startup_avoidance.fault_hold_retry_s",
                ):
                    validate_config(config)

        for value in (True, 0.09, 3600.01, math.nan):
            with self.subTest(fault_hold_max_s=value):
                config = load_config()
                config["startup_avoidance"]["fault_hold_max_s"] = value
                with self.assertRaisesRegex(
                    ConfigError,
                    "startup_avoidance.fault_hold_max_s",
                ):
                    validate_config(config)

    def test_pregrasp_target_search_config_rejects_invalid_values(self) -> None:
        invalid_values = (
            ("target_search_enabled", 1),
            ("target_search_bilateral_enabled", 1),
            ("target_search_until_found", 1),
            ("target_search_require_odom_progress", 1),
            ("target_search_return_to_origin_on_failure", 1),
            ("target_search_speed_mps", True),
            ("target_search_speed_mps", 0.0),
            ("target_search_speed_mps", 0.21),
            ("target_search_step_seconds", 0.0),
            ("target_search_settle_seconds", -0.01),
            ("target_search_each_side_m", 0.0),
            ("target_search_max_distance_m", 0.0),
            ("target_search_min_progress_m", 0.0),
            ("target_search_max_stalled_pulses", 0),
            ("target_search_max_net_lateral_m", 0.99),
            ("preapproach_search_min_distance_m", 3.01),
            ("target_search_center_band", [0.5]),
            ("target_search_center_band", [0.8, 0.2]),
            ("target_search_center_band", [True, 0.7]),
        )
        for key, value in invalid_values:
            with self.subTest(key=key, value=value):
                config = load_config()
                config["pregrasp_red_align"][key] = value
                with self.assertRaisesRegex(ConfigError, key):
                    validate_config(config)

    def test_until_found_search_accepts_bilateral_boundary_patrol(self) -> None:
        config = load_config()
        config["pregrasp_red_align"]["target_search_until_found"] = True
        config["pregrasp_red_align"]["target_search_bilateral_enabled"] = True

        validate_config(config)

    def test_pickup_entry_ignores_tag_at_left_edge_until_threshold(self) -> None:
        mission = object.__new__(LargeQuadrupedMission)
        mission.config = {
            "pickup_tag_boundary": {
                "enabled": True,
                "right_tag_id": 4,
                "entry_stop_center_x_px": 620,
                "entry_scan_step_m": 0.05,
            }
        }
        mission.context = MissionContext()
        mission.motion = mock.Mock()
        mission.state_reader = mock.Mock()
        mission.state_reader.pose.return_value = (1.0, 2.0, 0.0)
        mission._pickup_tag_centers = mock.Mock(
            side_effect=({4: 155.9}, {4: 410.0}, {4: 625.0})
        )

        with contextlib.redirect_stdout(io.StringIO()):
            mission._run_pickup_entry_tag_scan(1.30)

        self.assertEqual(
            mission.motion.strafe_distance.call_args_list,
            [mock.call(0.05), mock.call(0.05)],
        )
        self.assertTrue(mission.context.pickup_entry_tag_acquired)
        self.assertEqual(mission.context.pickup_entry_strafe_progress_m, 0.10)
        self.assertEqual(mission.context.pickup_search_origin_pose, (1.0, 2.0, 0.0))

    def test_pregrasp_final_distance_minimum_must_be_inside_window(self) -> None:
        for value in (0.10, 0.30, 0.31, math.nan):
            with self.subTest(value=value):
                config = load_config()
                config["pregrasp_red_align"]["final_distance_min_m"] = value
                with self.assertRaisesRegex(ConfigError, "final_distance_min_m"):
                    validate_config(config)

    def test_placement_letter_navigation_defaults_match_approved_values(self) -> None:
        expected = {
            "enabled": True,
            "letter_order": ["A", "B", "C", "D"],
            "letter_min_confidence": 0.60,
            "forward_speed_mps": 0.08,
            "front_stop_distance_m": 0.80,
            "forward_budget_m": 1.80,
            "search_step_m": 0.20,
            "min_center_correction_m": 0.03,
            "max_center_correction_m": 0.10,
            "center_gain_m_per_fraction": 0.45,
            "lateral_speed_mps": 0.08,
            "max_lateral_search_m": 3.10,
            "bilateral_search_enabled": True,
            "lateral_search_each_side_m": 1.00,
            "immediate_complete_on_target_detection": False,
            "acquisition_center_band": [1.0 / 3.0, 2.0 / 3.0],
            "center_tolerance_fraction": 0.05,
            "final_approach_distance_m": 0.30,
            "final_approach_step_m": 0.10,
            "letter_spacing_m": 0.50,
            "max_anchor_jump_m": 0.20,
            "target_vote_window": 3,
            "target_min_votes": 2,
            "target_memory_max_misses": 6,
            "target_memory_max_lateral_m": 0.25,
            "target_memory_max_forward_m": 0.40,
            "target_memory_fraction_per_m": 0.50,
            "ultrasound_filter_samples": 5,
            "ultrasound_stable_samples": 3,
            "ultrasound_jump_reject_m": 0.25,
            "ultrasound_jump_confirm_samples": 3,
            "motion_stall_timeout_s": 2.0,
            "motion_stall_min_progress_m": 0.01,
            "motion_stall_retries": 2,
            "motion_recovery_pause_s": 0.30,
            "motion_recovery_speed_mps": 0.06,
            "cached_geometry_enabled": True,
            "required_center_frames": 3,
            "capture_retries": 3,
            "image_timeout_s": 0.50,
            "total_timeout_s": 90.0,
            "physical_left_strafe_sign": 1,
            "run_log_dir": "placement_letter_navigation_runs",
        }
        robot_data = json.loads(
            (config_loader.ROOT / "config" / "robot.yaml").read_text(encoding="utf-8")
        )

        self.assertEqual(
            config_loader.DEFAULT_ROBOT["placement_letter_navigation"], expected
        )
        self.assertEqual(robot_data["placement_letter_navigation"], expected)
        self.assertEqual(load_config()["placement_letter_navigation"], expected)
        for config in (config_loader.DEFAULT_ROBOT, robot_data):
            box_center = config["box_center_alignment"]
            self.assertAlmostEqual(
                box_center["placement_label_min_area_fraction"], 0.00035
            )
            self.assertAlmostEqual(
                box_center["placement_label_max_area_fraction"], 0.12
            )

    def test_placement_letter_navigation_is_required_in_validation_and_robot_yaml(self) -> None:
        config = load_config()
        del config["placement_letter_navigation"]
        with self.assertRaisesRegex(ConfigError, "placement_letter_navigation"):
            validate_config(config)

        source_config = config_loader.ROOT / "config"
        robot_data = json.loads(
            (source_config / "robot.yaml").read_text(encoding="utf-8")
        )
        del robot_data["placement_letter_navigation"]
        with tempfile.TemporaryDirectory() as tmp:
            temp_config = Path(tmp)
            (temp_config / "field.yaml").write_text(
                (source_config / "field.yaml").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (temp_config / "robot.yaml").write_text(
                json.dumps(robot_data),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ConfigError, "placement_letter_navigation"):
                load_config(temp_config)

    def test_placement_letter_navigation_rejects_non_boolean_enabled(self) -> None:
        config = load_config()
        config["placement_letter_navigation"]["enabled"] = 1
        with self.assertRaisesRegex(ConfigError, "enabled"):
            validate_config(config)

    def test_placement_letter_navigation_rejects_invalid_letter_order(self) -> None:
        for value in (
            ["A", "B", "D", "C"],
            "A,B,C,D",
            ["A", "B", "C"],
            123,
        ):
            with self.subTest(value=value):
                config = load_config()
                config["placement_letter_navigation"]["letter_order"] = value
                with self.assertRaisesRegex(ConfigError, "letter_order"):
                    validate_config(config)

    def test_placement_letter_navigation_rejects_integer_and_boolean_traps(self) -> None:
        for key, value in (
            ("required_center_frames", True),
            ("required_center_frames", 1.5),
            ("capture_retries", False),
            ("capture_retries", 0),
        ):
            with self.subTest(key=key, value=value):
                config = load_config()
                config["placement_letter_navigation"][key] = value
                with self.assertRaisesRegex(ConfigError, key):
                    validate_config(config)

    def test_placement_letter_navigation_rejects_invalid_direction_sign(self) -> None:
        for value in (True, False, 0, 2, -2, 1.0, "1"):
            with self.subTest(value=value):
                config = load_config()
                config["placement_letter_navigation"]["physical_left_strafe_sign"] = value
                with self.assertRaisesRegex(ConfigError, "physical_left_strafe_sign"):
                    validate_config(config)

        config = load_config()
        config["placement_letter_navigation"]["physical_left_strafe_sign"] = -1
        validate_config(config)

    def test_placement_letter_navigation_rejects_non_finite_and_out_of_range_numbers(self) -> None:
        invalid_values = (
            ("letter_min_confidence", math.nan),
            ("letter_min_confidence", 1.01),
            ("forward_speed_mps", 0.0),
            ("front_stop_distance_m", 0.279),
            ("front_stop_distance_m", 4.501),
            ("forward_budget_m", math.inf),
            ("search_step_m", -0.1),
            ("max_center_correction_m", 0.0),
            ("lateral_speed_mps", -math.inf),
            ("max_lateral_search_m", 0.0),
            ("lateral_search_each_side_m", 0.0),
            ("center_tolerance_fraction", 0.501),
            ("final_approach_distance_m", 0.0),
            ("letter_spacing_m", math.inf),
            ("max_anchor_jump_m", -0.1),
            ("image_timeout_s", 0.0),
            ("total_timeout_s", -1.0),
        )
        for key, value in invalid_values:
            with self.subTest(key=key, value=value):
                config = load_config()
                config["placement_letter_navigation"][key] = value
                with self.assertRaisesRegex(ConfigError, key):
                    validate_config(config)

    def test_placement_letter_navigation_rejects_boolean_numbers(self) -> None:
        numeric_fields = (
            "letter_min_confidence",
            "forward_speed_mps",
            "front_stop_distance_m",
            "forward_budget_m",
            "search_step_m",
            "max_center_correction_m",
            "lateral_speed_mps",
            "max_lateral_search_m",
            "lateral_search_each_side_m",
            "center_tolerance_fraction",
            "final_approach_distance_m",
            "letter_spacing_m",
            "max_anchor_jump_m",
            "image_timeout_s",
            "total_timeout_s",
        )
        for key in numeric_fields:
            for value in (True, False):
                with self.subTest(key=key, value=value):
                    config = load_config()
                    config["placement_letter_navigation"][key] = value
                    with self.assertRaisesRegex(
                        ConfigError,
                        f"placement_letter_navigation.{key}",
                    ):
                        validate_config(config)

    def test_placement_letter_navigation_rejects_invalid_bilateral_search(self) -> None:
        for key in (
            "bilateral_search_enabled",
            "immediate_complete_on_target_detection",
            "cached_geometry_enabled",
        ):
            for value in (0, 1, None, "true"):
                with self.subTest(key=key, value=value):
                    config = load_config()
                    config["placement_letter_navigation"][key] = value
                    with self.assertRaisesRegex(ConfigError, key):
                        validate_config(config)

        config = load_config()
        config["placement_letter_navigation"]["max_lateral_search_m"] = 2.99
        with self.assertRaisesRegex(ConfigError, "max_lateral_search_m"):
            validate_config(config)

    def test_placement_letter_navigation_rejects_invalid_center_band(self) -> None:
        for value in (
            [0.5],
            [0.7, 0.3],
            [True, 0.7],
            [-0.1, 0.7],
            [0.3, 1.1],
        ):
            with self.subTest(value=value):
                config = load_config()
                config["placement_letter_navigation"][
                    "acquisition_center_band"
                ] = value
                with self.assertRaisesRegex(ConfigError, "acquisition_center_band"):
                    validate_config(config)

    def test_placement_letter_navigation_rejects_empty_log_directory(self) -> None:
        for value in (None, "", "   ", 123):
            with self.subTest(value=value):
                config = load_config()
                config["placement_letter_navigation"]["run_log_dir"] = value
                with self.assertRaisesRegex(ConfigError, "run_log_dir"):
                    validate_config(config)

    def test_transfer_enabled_rejects_mixed_or_repeated_visual_place_routes(
        self,
    ) -> None:
        invalid_routes = (
            [
                {"action": "turn", "yaw_rad": -math.pi},
                {"action": "placement_row_yaw_align"},
                {"action": "placement_lane_strafe"},
                {"action": "placement_letter_approach"},
            ],
            [
                {"action": "turn", "yaw_rad": -math.pi},
                {"action": "placement_row_yaw_align"},
                {"action": "placement_letter_approach"},
                {"action": "placement_letter_approach"},
            ],
            [
                {"action": "turn", "yaw_rad": -math.pi},
                {"action": "placement_row_yaw_align"},
                {"action": "forward", "distance_m": 1.38},
                {"action": "placement_letter_approach"},
            ],
            [
                {"action": "turn", "yaw_rad": -math.pi},
                {"action": "placement_row_yaw_align"},
                {"action": "forward", "distance_m": 0.25},
            ],
        )
        for route in invalid_routes:
            with self.subTest(actions=[action["action"] for action in route]):
                config = load_config()
                config["scripted_route"]["place_from_pickup"] = route
                with self.assertRaisesRegex(ConfigError, "place_from_pickup"):
                    validate_config(config)

    def test_transfer_disabled_accepts_a_custom_nonvisual_place_route(self) -> None:
        config = load_config()
        config["pickup_transfer"]["enabled"] = False
        config["scripted_route"]["place_from_pickup"] = [
            {"action": "turn", "yaw_rad": -0.75},
            {"action": "forward", "distance_m": 0.60},
        ]

        validate_config(config)

    def test_route_front_stop_completion_requires_forward_boolean(self) -> None:
        config = load_config()
        pickup_forward = config["scripted_route"]["pickup_from_upper_inspection"][1]
        pickup_forward["front_stop_is_completion"] = 1
        with self.assertRaisesRegex(ConfigError, "must be a boolean"):
            validate_config(config)

        config = load_config()
        pickup_strafe = config["scripted_route"]["pickup_from_upper_inspection"][2]
        pickup_strafe["front_stop_is_completion"] = True
        with self.assertRaisesRegex(ConfigError, "valid only for forward"):
            validate_config(config)

    def test_transfer_enabled_rejects_non_boolean_values(self) -> None:
        for value in (0, 1, "true", None):
            with self.subTest(value=value):
                config = load_config()
                config["pickup_transfer"]["enabled"] = value
                with self.assertRaisesRegex(ConfigError, "pickup_transfer.enabled"):
                    validate_config(config)

    def test_transfer_enabled_requires_placement_letter_navigation(self) -> None:
        config = load_config()
        config["placement_letter_navigation"]["enabled"] = False

        with self.assertRaisesRegex(
            ConfigError,
            "pickup_transfer.enabled requires placement_letter_navigation.enabled",
        ):
            validate_config(config)

        config["pickup_transfer"]["enabled"] = False
        validate_config(config)

    def test_placement_label_area_fraction_bounds_are_strict(self) -> None:
        invalid_values = (
            (math.nan, 0.12),
            (0.0, 0.12),
            (0.12, 0.12),
            (0.13, 0.12),
            (0.00035, math.inf),
            (0.00035, 1.01),
        )
        for minimum, maximum in invalid_values:
            with self.subTest(minimum=minimum, maximum=maximum):
                config = load_config()
                box_center = config["box_center_alignment"]
                box_center["placement_label_min_area_fraction"] = minimum
                box_center["placement_label_max_area_fraction"] = maximum
                with self.assertRaisesRegex(ConfigError, "placement_label"):
                    validate_config(config)

    def test_placement_label_area_fractions_reject_boolean_numbers(self) -> None:
        for key in (
            "placement_label_min_area_fraction",
            "placement_label_max_area_fraction",
        ):
            for value in (True, False):
                with self.subTest(key=key, value=value):
                    config = load_config()
                    config["box_center_alignment"][key] = value
                    with self.assertRaisesRegex(
                        ConfigError,
                        f"box_center_alignment.{key}",
                    ):
                        validate_config(config)

    def test_live_inspection_applies_configured_wide_undistortion(self) -> None:
        runner = LiveInspectionRunner.__new__(LiveInspectionRunner)
        runner.inspection_undistorter = mock.Mock()
        frame = object()
        runner.inspection_undistorter.apply.return_value = "undistorted"

        prepared = runner._prepare_inspection_frame(frame)

        self.assertEqual(prepared, "undistorted")
        runner.inspection_undistorter.apply.assert_called_once_with(frame)

    def test_live_inspection_accepts_real_time_limit(self) -> None:
        args = build_live_parser().parse_args(["--max-seconds", "8"])
        self.assertEqual(args.max_seconds, 8.0)

    def test_config_rejects_missing_files_and_unknown_route_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ConfigError):
                load_config(Path(tmp))
        config = load_config()
        config["scripted_route"] = dict(config["scripted_route"])
        config["scripted_route"]["bad"] = [{"action": "teleport", "distance_m": 1.0}]
        with self.assertRaisesRegex(ConfigError, "unknown"):
            validate_config(config)

    def test_arm_runtime_argv_uses_env_device_overrides(self) -> None:
        config = load_config()
        controller = LiteArmController(config)
        result_file = Path("logs/test_arm_result.json")
        with mock.patch.dict(
            "os.environ",
            {
                "ARM_PORT": "/dev/ttyUSB9",
                "ARM_CAMERA": "/dev/video9",
                "ARM_WIDTH": "800",
                "ARM_HEIGHT": "600",
            },
        ):
            argv = controller._runtime_base_argv(result_file, include_camera=True)  # noqa: SLF001
        self.assertIn("/dev/ttyUSB9", argv)
        self.assertIn("/dev/video9", argv)
        self.assertIn("800", argv)
        self.assertIn("600", argv)
        self.assertNotIn("--moving-pose", argv)
        self.assertNotIn("--transport-spd", argv)

    def test_arm_transport_uses_source_grasp_ready_sequence(self) -> None:
        task = arm_task.ArmTask(dry_run=True)
        result = task.transport()
        self.assertTrue(result["ok"])
        self.assertEqual(result["stage"], "GRASP_READY")
        self.assertEqual(result["plan"][0]["stage"], "OPEN_GRIPPER")
        self.assertEqual(result["plan"][1]["stage"], "MOVE_TO_GRASP_READY_POSE")

    @staticmethod
    def _arm_status(w_degrees: float, move: int) -> dict:
        return {
            "b": 0.0,
            "s": 0.0,
            "e": 0.0,
            "w": math.radians(w_degrees),
            "t": 0.0,
            "move": move,
        }

    def test_joint_target_accepts_stable_feedback_with_move_one(self) -> None:
        statuses = iter(
            [
                self._arm_status(0.8, 1),
                self._arm_status(0.8, 1),
                self._arm_status(0.8, 1),
            ]
        )
        final = arm_test.wait_for_joint_target(
            query_status_fn=lambda: next(statuses),
            joint="w",
            target_degrees=1.0,
            initial_status=self._arm_status(0.0, 0),
            sleep_fn=lambda _seconds: None,
        )
        self.assertEqual(final["move"], 1)

    def test_jog_limits_match_cli_safety_contract(self) -> None:
        status = self._arm_status(0.0, 0)
        command = arm_test.build_jog_command("b", 20.0, status)
        self.assertEqual(command["b"], 20.0)
        with self.assertRaisesRegex(ValueError, "b关节单次微动不得超过20度"):
            arm_test.build_jog_command("b", 20.1, status)

    def test_motion_completion_uses_stable_feedback_not_move_flag(self) -> None:
        statuses = iter(
            [
                self._arm_status(0.0, 1),
                self._arm_status(0.0, 1),
                self._arm_status(0.0, 1),
            ]
        )
        final = arm_test.wait_for_motion_idle(
            query_status_fn=lambda: next(statuses),
            initial_status=self._arm_status(0.0, 1),
            sleep_fn=lambda _seconds: None,
        )
        self.assertEqual(final["move"], 1)

    def test_arm_preflight_reads_source_yaml_calibration_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            calibration = Path(tmp) / "ost.yaml"
            calibration.write_text(
                "image_width: 1280\nimage_height: 720\n",
                encoding="utf-8",
            )
            self.assertEqual(
                arm_task._read_calibration_image_size(calibration),  # noqa: SLF001
                (1280, 720),
            )

    def test_arm_preflight_reads_runtime_json_calibration_size(self) -> None:
        calibration = Path(
            "mission_lite3/arm/runtime/camera_calibration.json"
        )
        self.assertEqual(
            arm_task._read_calibration_image_size(calibration),  # noqa: SLF001
            (1280, 720),
        )

    def test_arm_camera_source_prefers_arm_camera_device(self) -> None:
        config = load_config()
        config["camera"] = dict(config["camera"])
        config["arm"] = dict(config["arm"])
        config["camera"]["arm"] = "/dev/video-old"
        config["arm"]["camera_device"] = "/dev/video-new"
        mission = LargeQuadrupedMission(config, dry_run=True, skip_arm=True)
        self.assertEqual(mission.arm_camera.source, "/dev/video-new")

    def test_inspection_audio_clip_mapping(self) -> None:
        self.assertEqual(clip_name_for_result("A", "正常", "正常"), "A_normal.wav")
        self.assertEqual(clip_name_for_result("a", "偏低", "异常"), "A_low.wav")
        self.assertEqual(clip_name_for_result("C", "偏高", "异常"), "C_high.wav")
        self.assertIsNone(clip_name_for_result("X", "偏高", "异常"))
        self.assertEqual(build_announcement("B", "正常", "正常"), "B区域仪表盘显示正常，状态正常")

    def test_project_audio_assets_are_available(self) -> None:
        audio_dir = Path("mission_lite3/inspection_audio")
        expected = {
            f"{letter}_{suffix}.wav"
            for letter in ("A", "B", "C", "D")
            for suffix in ("normal", "low", "high")
        }
        self.assertEqual({path.name for path in audio_dir.glob("*.wav")}, expected)
        clip_path = resolve_audio_clip("A", "偏低", "异常", audio_dir)
        self.assertIsNotNone(clip_path)
        self.assertTrue(clip_path.is_file())

    def test_audio_reporter_dry_run_accepts_structured_result(self) -> None:
        reporter = AudioReporter(load_config(), dry_run=True)
        with contextlib.redirect_stdout(io.StringIO()):
            error = reporter.say_result("A", "偏低", "异常")
        self.assertIsNone(error)

    def test_audio_reporter_tts_fallback_clears_wav_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config()
            config["audio"] = dict(config["audio"])
            config["audio"].update(
                {
                    "mode": "wav_first",
                    "audio_dir": tmp,
                    "fallback_to_tts_on_audio_failure": True,
                    "command": sys.executable,
                    "args": ["-c", "pass"],
                    "prepare_pulse": False,
                }
            )
            reporter = AudioReporter(config)
            with contextlib.redirect_stdout(io.StringIO()):
                error = reporter.say_result("A", "偏低", "异常")
        self.assertIsNone(error)

    def test_audio_reporter_remote_udp_sends_clip_and_accepts_ack(self) -> None:
        sent: dict[str, object] = {}

        class FakeSocket:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def settimeout(self, timeout: float) -> None:
                sent["timeout"] = timeout

            def sendto(self, payload: bytes, address) -> None:
                sent["payload"] = payload
                sent["address"] = address

            def recvfrom(self, size: int):
                request = json.loads(sent["payload"].decode("utf-8"))
                response = {
                    "ok": True,
                    "request_id": request["request_id"],
                    "clip": request["clip"],
                    "applied_gain_db": request["gain_db"],
                }
                return json.dumps(response).encode("utf-8"), ("127.0.0.1", 43910)

        config = load_config()
        config["audio"] = dict(config["audio"])
        config["audio"].update(
            {
                "mode": "remote_udp",
                "remote_host": "127.0.0.1",
                "remote_port": 43910,
                "remote_timeout_seconds": 1.0,
                "remote_retries": 0,
                "prepare_pulse": False,
            }
        )
        with mock.patch("mission_lite3.audio.socket.socket", return_value=FakeSocket()):
            reporter = AudioReporter(config)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                error = reporter.say_result("B", "正常", "正常")
        self.assertIsNone(error)
        self.assertEqual(sent["address"], ("127.0.0.1", 43910))
        self.assertEqual(sent["timeout"], 1.0)
        received = json.loads(sent["payload"].decode("utf-8"))
        self.assertEqual(received["command"], "play")
        self.assertEqual(received["clip"], "B_normal.wav")
        self.assertEqual(received["gain_db"], 3.0)
        self.assertTrue(received["request_id"])
        self.assertIn("[audio-service] response", output.getvalue())
        self.assertIn(received["request_id"], output.getvalue())

    def test_audio_reporter_boot_prewarm_logs_service_response(self) -> None:
        sent: dict[str, object] = {}

        class FakeSocket:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def settimeout(self, timeout: float) -> None:
                pass

            def sendto(self, payload: bytes, address) -> None:
                sent["payload"] = payload

            def recvfrom(self, size: int):
                request = json.loads(sent["payload"].decode("utf-8"))
                response = {
                    "ok": True,
                    "request_id": request["request_id"],
                    "command": "warmup",
                    "warmup_duration_s": request["duration_s"],
                }
                return json.dumps(response).encode("utf-8"), ("127.0.0.1", 43910)

        config = load_config()
        config["audio"] = dict(config["audio"])
        config["audio"].update(
            {
                "remote_host": "127.0.0.1",
                "remote_retries": 0,
                "prewarm_enabled": True,
                "prewarm_duration_s": 0.8,
            }
        )
        with mock.patch("mission_lite3.audio.socket.socket", return_value=FakeSocket()):
            reporter = AudioReporter(config)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertIsNone(reporter.prewarm())
        request = json.loads(sent["payload"].decode("utf-8"))
        self.assertEqual(request["command"], "warmup")
        self.assertAlmostEqual(request["duration_s"], 0.8)
        self.assertIn("[audio-service] response", output.getvalue())
        self.assertIn("[audio] prewarm_ok", output.getvalue())

    def test_boot_check_invokes_audio_prewarm(self) -> None:
        mission = LargeQuadrupedMission(load_config(), dry_run=True, skip_arm=True)
        mission._initialize_round_result = mock.Mock()
        mission._check_safety = mock.Mock()
        mission.audio.prewarm = mock.Mock(return_value=None)

        mission._state_boot_check()

        mission.audio.prewarm.assert_called_once_with()

    def test_boot_check_uses_dedicated_first_frame_timeout(self) -> None:
        config = load_config()
        self.assertEqual(config["camera"]["startup_first_frame_timeout_s"], 6.0)
        mission = LargeQuadrupedMission(config, dry_run=True, skip_arm=True)
        mission.context.dry_run = False
        mission._initialize_round_result = mock.Mock()
        mission._check_safety = mock.Mock()
        mission.state_reader.wait_until_ready = mock.Mock()
        mission.front_camera.open = mock.Mock(return_value=True)
        mission.front_camera.read = mock.Mock(return_value=object())
        mission.audio.prewarm = mock.Mock(return_value=None)

        mission._state_boot_check()

        mission.front_camera.read.assert_called_once_with(timeout_s=6.0)

    def test_camera_rejects_invalid_startup_first_frame_timeout(self) -> None:
        config = load_config()
        config["camera"]["startup_first_frame_timeout_s"] = 0.0

        with self.assertRaisesRegex(ConfigError, "startup_first_frame_timeout_s"):
            validate_config(config)

    def test_audio_reporter_remote_udp_timeout_returns_error(self) -> None:
        class TimeoutSocket:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def settimeout(self, timeout: float) -> None:
                pass

            def sendto(self, payload: bytes, address) -> None:
                pass

            def recvfrom(self, size: int):
                raise TimeoutError("timed out")

        config = load_config()
        config["audio"] = dict(config["audio"])
        config["audio"].update(
            {
                "mode": "remote_udp",
                "remote_host": "127.0.0.1",
                "remote_port": 43910,
                "remote_timeout_seconds": 0.05,
                "remote_retries": 0,
                "prepare_pulse": False,
            }
        )
        with mock.patch("mission_lite3.audio.socket.socket", return_value=TimeoutSocket()):
            reporter = AudioReporter(config)
            with contextlib.redirect_stdout(io.StringIO()):
                error = reporter.say_result("C", "偏高", "异常")
        self.assertIsNotNone(error)
        self.assertIn("timed out", error)

    def test_camera_source_reconnects_on_repeated_stale_frame(self) -> None:
        class FakeCap:
            def read(self):
                return True, object()

        camera = CameraSource("fake", stale_frame_reconnect_count=1)
        camera.cap = FakeCap()
        camera._frame_signature = mock.Mock(return_value=b"same")  # type: ignore[method-assign]
        camera._reopen = mock.Mock()  # type: ignore[method-assign]

        first = camera.read()
        second = camera.read()

        self.assertIsNotNone(first)
        self.assertIsNone(second)
        camera._reopen.assert_called_once_with("stale_frame")

    def test_camera_source_applies_center_digital_zoom(self) -> None:
        import numpy as np

        frame = np.zeros((4, 6, 3), dtype=np.uint8)
        frame[1:3, 1:5] = 100

        class FakeCap:
            def read(self):
                return True, frame

        camera = CameraSource("fake", digital_zoom=1.5)
        camera.cap = FakeCap()

        zoomed = camera.read()

        self.assertIsNotNone(zoomed)
        self.assertEqual(zoomed.shape, frame.shape)
        self.assertTrue((zoomed == 100).all())

    def test_camera_views_share_latest_frame_reader_without_view_release(self) -> None:
        import numpy as np

        owner = CameraSource("fake", dry_run=True)
        reader = mock.Mock()
        frame = np.zeros((4, 6, 3), dtype=np.uint8)
        reader.read_latest.side_effect = [(frame, 1, 10.0), (frame, 1, 10.0)]
        owner._persistent_reader = reader
        view = CameraSource("fake", dry_run=True, shared_camera=owner)

        self.assertIsNotNone(owner.read())
        self.assertIsNotNone(view.read())
        self.assertEqual(view._last_frame_sequence, 1)
        view.release()
        reader.stop.assert_not_called()
        self.assertEqual(view._last_frame_sequence, 1)
        owner.release()
        reader.stop.assert_called_once_with()

    def test_persistent_reader_terminates_blocked_worker(self) -> None:
        reader = PersistentLatestFrameReader("fake", 4, 4)
        process = mock.Mock()
        process.is_alive.side_effect = [True, False]
        process.exitcode = -15
        reader._process = process

        with contextlib.redirect_stdout(io.StringIO()):
            reader._stop_worker("blocked_read")

        process.terminate.assert_called_once_with()
        self.assertEqual(process.join.call_count, 2)
        self.assertIsNone(reader._process)

    def test_persistent_reader_does_not_return_pre_reconnect_buffer(self) -> None:
        reader = PersistentLatestFrameReader("fake", 4, 4)
        reader.start = mock.Mock(return_value=True)
        reader._sequence.value = 5
        reader._captured_at.value = 0.0

        with contextlib.redirect_stdout(io.StringIO()):
            result = reader.read_latest(0, timeout_s=0.0)

        self.assertIsNone(result)

    def test_persistent_reader_double_buffer_rejects_copy_changed_during_publish(self) -> None:
        import numpy as np

        reader = PersistentLatestFrameReader('fake', 2, 2)
        reader._sequence.value = 1
        reader._captured_at.value = 10.0
        reader._active_slot.value = 0

        class PublishingNumpy:
            uint8 = np.uint8

            @staticmethod
            def frombuffer(*args, **kwargs):
                result = np.frombuffer(*args, **kwargs)
                reader._sequence.value += 1
                reader._active_slot.value = 1
                return result

        self.assertIsNone(reader._copy_published_frame(0, PublishingNumpy))

    def test_persistent_reader_ignores_status_from_old_generation(self) -> None:
        import queue

        reader = PersistentLatestFrameReader('fake', 4, 4)
        reader._generation = 2
        reader._last_status = 'starting'
        reader._consecutive_failures = 3
        reader._status_queue = queue.Queue()
        reader._status_queue.put_nowait(
            {
                'event': 'connected',
                'event_time_unix': time.time(),
                'generation': 1,
            }
        )

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            reader._drain_status()

        self.assertEqual(reader._last_status, 'starting')
        self.assertEqual(reader._consecutive_failures, 3)
        self.assertIn('event=stale_worker_status', output.getvalue())
        self.assertIn('event_generation', output.getvalue())

    def test_remote_audio_server_rejects_non_whitelisted_clip(self) -> None:
        module_path = Path("tools/remote_audio_server.py")
        spec = importlib.util.spec_from_file_location("remote_audio_server", module_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(spec.name, None)

        response = module.handle_request(
            b'{"command":"play","clip":"../../etc/passwd","request_id":"test"}',
            Path("/tmp"),
            "dummy",
            1.0,
        )
        self.assertFalse(response["ok"])
        self.assertIn("not allowed", response["error"])

    def test_remote_audio_server_accepts_bounded_warmup(self) -> None:
        module_path = Path("tools/remote_audio_server.py")
        spec = importlib.util.spec_from_file_location("remote_audio_server_warmup", module_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
            with mock.patch.object(module, "play_warmup") as play_warmup:
                response = module.handle_request(
                    b'{"command":"warmup","request_id":"warm","duration_s":0.8}',
                    Path("/tmp"),
                    "dummy",
                    1.0,
                )
        finally:
            sys.modules.pop(spec.name, None)
        self.assertTrue(response["ok"])
        self.assertEqual(response["command"], "warmup")
        self.assertAlmostEqual(response["warmup_duration_s"], 0.8)
        play_warmup.assert_called_once_with("dummy", 1.0, 0.8)

    def test_remote_audio_server_applies_three_db_without_clipping(self) -> None:
        module_path = Path("tools/remote_audio_server.py")
        spec = importlib.util.spec_from_file_location("remote_audio_server_gain", module_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
            with tempfile.TemporaryDirectory() as tmp:
                source = Path(tmp) / "source.wav"
                destination = Path(tmp) / "amplified.wav"
                samples = (1000, -1000, 2000, -2000)
                with wave.open(str(source), "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(22050)
                    wav_file.writeframes(struct.pack("<4h", *samples))
                clipped = module.apply_gain_to_wav(source, destination, 3.0)
                with wave.open(str(destination), "rb") as wav_file:
                    amplified = struct.unpack("<4h", wav_file.readframes(4))
        finally:
            sys.modules.pop(spec.name, None)
        self.assertEqual(clipped, 0)
        expected_multiplier = 10.0 ** (3.0 / 20.0)
        self.assertAlmostEqual(amplified[0], round(samples[0] * expected_multiplier), delta=1)
        self.assertAlmostEqual(amplified[3], round(samples[3] * expected_multiplier), delta=1)

    def test_remote_audio_server_rejects_unsafe_gain(self) -> None:
        module_path = Path("tools/remote_audio_server.py")
        spec = importlib.util.spec_from_file_location("remote_audio_server_invalid_gain", module_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
            response = module.handle_request(
                b'{"command":"play","clip":"A_low.wav","request_id":"test","gain_db":20}',
                Path("mission_lite3/inspection_audio"),
                "dummy",
                1.0,
            )
        finally:
            sys.modules.pop(spec.name, None)
        self.assertFalse(response["ok"])
        self.assertIn("gain_db", response["error"])

    def test_standalone_arm_script_dry_run_sequence(self) -> None:
        module_path = Path("tools/test_arm_standalone.py")
        spec = importlib.util.spec_from_file_location("test_arm_standalone", module_path)
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(spec.name, None)

        with tempfile.TemporaryDirectory() as tmp:
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                code = module.main(["--scenario", "dry-run", "--result-dir", tmp])

        self.assertEqual(code, 0)
        self.assertIn("scenario=dry-run", output.getvalue())

    def test_runtime_inspection_result_maps_to_local_record_fields(self) -> None:
        result = {
            "ok": True,
            "letter_detection": {"label": "C", "confidence": 0.86},
            "meter_detection": {"state": "abnormal", "description": "偏高"},
        }
        self.assertEqual(runtime_result_to_record_fields(result), ("C", "偏高", "异常", 0.86))

    def test_runtime_unknown_result_is_not_valid_record(self) -> None:
        result = {
            "ok": True,
            "letter_detection": {"label": "B", "confidence": 0.99},
            "meter_detection": {"state": "unknown", "description": "无法确认"},
        }
        self.assertIsNone(runtime_result_to_record_fields(result))

    @staticmethod
    def _runtime_result(
        *,
        letter: str = "A",
        confidence: float = 0.90,
        level: str = "偏低",
        hit_ratio: float = 0.80,
        run_ratio: float = 0.60,
    ) -> dict:
        state = "normal" if level == "正常" else "abnormal"
        return {
            "ok": True,
            "geometry_source": "letter_anchor",
            "letter_detection": {"label": letter, "confidence": confidence},
            "meter_detection": {
                "state": state,
                "description": level,
                "pointer_support": {
                    "hit_ratio": hit_ratio,
                    "longest_run_ratio": run_ratio,
                },
            },
        }

    def _runtime_pipeline_for_test(self) -> VisionPipeline:
        with mock.patch.object(VisionPipeline, "_load_runtime_backend"):
            pipeline = VisionPipeline(load_config())
        pipeline.runtime_frame_pipeline = mock.Mock()
        return pipeline

    def test_runtime_high_quality_letter_anchor_requires_three_frames(self) -> None:
        pipeline = self._runtime_pipeline_for_test()
        pipeline._analyze_runtime_frame = mock.Mock(  # type: ignore[method-assign]
            return_value=self._runtime_result()
        )
        self.assertIsNone(pipeline.inspect_frame(object(), source_camera="front"))
        self.assertIsNone(pipeline.inspect_frame(object(), source_camera="front"))
        record = pipeline.inspect_frame(object(), source_camera="front")
        self.assertIsNotNone(record)
        self.assertEqual(record.stability_votes["runtime"], 3)

    def test_runtime_clear_margin_does_not_bypass_three_frame_vote(self) -> None:
        pipeline = self._runtime_pipeline_for_test()
        result = self._runtime_result(confidence=0.737)
        result["letter_detection"]["margin"] = 0.289
        pipeline._analyze_runtime_frame = mock.Mock(return_value=result)  # type: ignore[method-assign]

        self.assertIsNone(pipeline.inspect_frame(object(), source_camera="front"))
        self.assertIsNone(pipeline.inspect_frame(object(), source_camera="front"))
        record = pipeline.inspect_frame(object(), source_camera="front")

        self.assertIsNotNone(record)
        self.assertEqual(record.stability_votes["runtime"], 3)

    def test_runtime_strong_pointer_with_weaker_letter_uses_stable_vote(self) -> None:
        pipeline = self._runtime_pipeline_for_test()
        pipeline._analyze_runtime_frame = mock.Mock(  # type: ignore[method-assign]
            return_value=self._runtime_result(confidence=0.80)
        )
        self.assertIsNone(pipeline.inspect_frame(object(), source_camera="front"))
        self.assertIsNone(pipeline.inspect_frame(object(), source_camera="front"))
        record = pipeline.inspect_frame(object(), source_camera="front")
        self.assertIsNotNone(record)
        self.assertEqual(record.stability_votes["runtime"], 3)

    def test_runtime_short_false_pointer_cannot_enter_stable_vote(self) -> None:
        pipeline = self._runtime_pipeline_for_test()
        pipeline._analyze_runtime_frame = mock.Mock(  # type: ignore[method-assign]
            return_value=self._runtime_result(hit_ratio=0.57, run_ratio=0.32)
        )
        self.assertIsNone(pipeline.inspect_frame(object(), source_camera="front"))
        self.assertIsNone(pipeline.inspect_frame(object(), source_camera="front"))
        self.assertIsNone(pipeline.best_inspection_candidate())

    def test_runtime_disagreeing_status_methods_cannot_enter_vote(self) -> None:
        pipeline = self._runtime_pipeline_for_test()
        result = self._runtime_result()
        result["meter_detection"]["status_evidence"] = {
            "status_agreement": False,
        }
        pipeline._analyze_runtime_frame = mock.Mock(return_value=result)  # type: ignore[method-assign]
        self.assertIsNone(pipeline.inspect_frame(object(), source_camera="front"))
        self.assertIsNone(pipeline.inspect_frame(object(), source_camera="front"))
        self.assertIsNone(pipeline.best_inspection_candidate())

    def test_runtime_best_candidate_keeps_matching_frame_and_resets(self) -> None:
        pipeline = self._runtime_pipeline_for_test()
        pipeline._analyze_runtime_frame = mock.Mock(  # type: ignore[method-assign]
            return_value=self._runtime_result(confidence=0.83)
        )
        frame = mock.Mock()
        frame.copy.return_value = "saved-frame"
        self.assertIsNone(pipeline.inspect_frame(frame, source_camera="front"))
        candidate, candidate_frame = pipeline.best_inspection_candidate()
        self.assertEqual(candidate.letter, "A")
        self.assertEqual(candidate_frame, "saved-frame")
        pipeline.reset_inspection_votes()
        self.assertIsNone(pipeline.best_inspection_candidate())

    def test_runtime_threshold_config_must_be_probability(self) -> None:
        config = load_config()
        config["vision"] = dict(config["vision"])
        config["vision"]["runtime_fast_min_pointer_hit_ratio"] = 1.01
        with self.assertRaisesRegex(ConfigError, "runtime_fast_min_pointer_hit_ratio"):
            validate_config(config)

    def test_remote_audio_systemd_service_uses_persistent_audio_dir(self) -> None:
        service_text = Path("systemd/lite3-remote-audio.service").read_text(encoding="utf-8")
        self.assertIn("/opt/robot_competition/remote_audio_server.py", service_text)
        self.assertIn("--audio-dir /opt/robot_competition/inspection_audio_test", service_text)
        self.assertIn("--device plughw:CARD=rockchipes8388c,DEV=0", service_text)
        self.assertIn("--port 43910", service_text)
        self.assertIn("--default-gain-db 0", service_text)

    def test_udp_packet_packing_matches_lite3_examples(self) -> None:
        self.assertEqual(pack_simple_command(HEARTBEAT), struct.pack("<3i", 0x21040001, 0, 0))
        packet = pack_velocity_command(VEL_FORWARD, 0.18)
        code, length, msg_type = struct.unpack("<3I", packet[:12])
        value = struct.unpack("<d", packet[12:])[0]
        self.assertEqual(code, VEL_FORWARD)
        self.assertEqual(length, 8)
        self.assertEqual(msg_type, 1)
        self.assertAlmostEqual(value, 0.18)

    def test_dry_run_mission_completes_two_anomaly_places(self) -> None:
        config = load_config()
        mission = LargeQuadrupedMission(config, dry_run=True, skip_arm=True)
        with contextlib.redirect_stdout(io.StringIO()):
            result = mission.run()
        self.assertTrue(result.ok)
        self.assertEqual(mission.context.placed_letters, ["A", "C"])

    def test_round_gate_rejects_a_different_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "round.json"
            data = build_round_result(
                {
                    "A": InspectionRecord("A", "偏低", "异常", 0.9, 1),
                    "B": InspectionRecord("B", "正常", "正常", 0.9, 2),
                    "C": InspectionRecord("C", "偏高", "异常", 0.9, 3),
                    "D": InspectionRecord("D", "正常", "正常", 0.9, 4),
                },
                run_id="old-run",
            )
            path.write_text(json.dumps(data), encoding="utf-8")
            gate = evaluate_round_gate(path, expected_run_id="new-run")
        self.assertFalse(gate.allowed)
        self.assertEqual(gate.block_reason, "round_result_run_id_mismatch")

    def test_stable_vote_requires_the_current_frame_to_be_valid(self) -> None:
        vote = StableVote[str](window=3, votes=2)
        self.assertIsNone(vote.add("A"))
        self.assertEqual(vote.add("A"), "A")
        self.assertIsNone(vote.add(None))

    def test_state_reader_rejects_stale_real_robot_samples(self) -> None:
        reader = StateReader(load_config(), dry_run=False)
        reader.state.odom_updated_at = time.monotonic() - 10.0
        reader.state.imu_updated_at = time.monotonic()
        error = reader.safety_error(require_fresh=True)
        self.assertIsNotNone(error)
        self.assertIn("odometry sample is stale", error)

    def test_state_reader_drains_queued_callbacks_before_safety_check(self) -> None:
        reader = StateReader(load_config(), dry_run=False)
        reader.node = object()
        reader.state.odom_updated_at = time.monotonic() - 10.0
        reader.state.imu_updated_at = time.monotonic() - 10.0
        callbacks = [
            lambda: setattr(reader.state, "imu_updated_at", time.monotonic()),
            lambda: setattr(reader.state, "odom_updated_at", time.monotonic()),
        ]

        class FakeRclpy:
            @staticmethod
            def spin_once(_node, *, timeout_sec: float) -> None:
                self.assertEqual(timeout_sec, 0.0)
                if callbacks:
                    callbacks.pop(0)()

        reader.rclpy = FakeRclpy()

        self.assertIsNone(reader.safety_error(require_fresh=True))
        self.assertEqual(callbacks, [])

    def test_state_reader_keeps_odom_and_imu_yaw_in_separate_frames(self) -> None:
        reader = StateReader(load_config(), dry_run=False)

        def quaternion(yaw: float) -> SimpleNamespace:
            return SimpleNamespace(
                x=0.0,
                y=0.0,
                z=math.sin(yaw / 2.0),
                w=math.cos(yaw / 2.0),
            )

        odom = SimpleNamespace(
            pose=SimpleNamespace(
                pose=SimpleNamespace(
                    position=SimpleNamespace(x=1.0, y=2.0),
                    orientation=quaternion(0.25),
                )
            )
        )
        imu = SimpleNamespace(orientation=quaternion(-0.75))

        reader._on_odom(odom)  # noqa: SLF001
        reader._on_imu(imu)  # noqa: SLF001

        self.assertAlmostEqual(reader.state.yaw, 0.25)
        self.assertAlmostEqual(reader.state.odom_yaw, 0.25)
        self.assertAlmostEqual(reader.state.imu_yaw, -0.75)
        self.assertEqual(reader.pose()[:2], (1.0, 2.0))
        self.assertAlmostEqual(reader.pose()[2], 0.25)

    def test_state_reader_front_filter_rejects_short_minimum_range_echo(self) -> None:
        reader = StateReader(load_config(), dry_run=False)
        for value in [1.12] * 7 + [0.28] * 2:
            reader._on_ultrasound(SimpleNamespace(data=value))  # noqa: SLF001

        self.assertAlmostEqual(reader.filtered_front_ultrasound_m(), 1.12)

        close_reader = StateReader(load_config(), dry_run=False)
        for value in [0.34] * 7:
            close_reader._on_ultrasound(SimpleNamespace(data=value))  # noqa: SLF001
        self.assertAlmostEqual(close_reader.filtered_front_ultrasound_m(), 0.34)

    def test_route_simulator_integrates_robot_frame_actions(self) -> None:
        pose = simulate_route_actions(
            RoutePose(0.0, 0.0, math.pi),
            [
                {"action": "forward", "distance_m": 1.0},
                {"action": "strafe", "distance_m": 0.5},
                {"action": "turn", "yaw_rad": -math.pi / 2.0},
            ],
        )
        self.assertAlmostEqual(pose.x, -1.0, places=6)
        self.assertAlmostEqual(pose.y, -0.5, places=6)
        self.assertAlmostEqual(pose.yaw, math.pi / 2.0, places=6)

    def test_route_simulator_preserves_legacy_and_dynamic_actions(self) -> None:
        start = RoutePose(-1.0, 2.0, 0.5)

        pose = simulate_route_actions(
            start,
            [
                {"action": "placement_lane_strafe"},
                {"action": "placement_letter_approach"},
            ],
        )

        self.assertEqual(pose, start)

    def test_route_boundary_check_reports_out_of_field_pose(self) -> None:
        errors = route_boundary_errors(load_config(), {"bad_route": RoutePose(-6.0, 1.0, 0.0)})
        self.assertEqual(len(errors), 1)
        self.assertIn("bad_route leaves field bounds", errors[0])

    def test_motion_guard_failure_always_sends_stop(self) -> None:
        class RecordingBackend:
            name = "recording"

            def __init__(self) -> None:
                self.velocities = []

            def send_velocity(self, vx, vy, wz) -> None:
                self.velocities.append((vx, vy, wz))

            def send_simple(self, code, value=0, msg_type=0) -> None:
                return None

            def close(self) -> None:
                return None

        controller = Lite3MotionController(load_config(), dry_run=True)
        controller.dry_run = False
        backend = RecordingBackend()
        controller.backend = backend

        def reject(vx, vy, wz) -> None:
            raise RuntimeError("unsafe")

        controller.configure_safety(reject, lambda: (0.0, 0.0, 0.0), feedback_required=False)
        with self.assertRaisesRegex(RuntimeError, "unsafe"):
            controller.hold_velocity(0.1, 0.0, 0.0, 0.1)
        self.assertEqual(backend.velocities[-1], (0.0, 0.0, 0.0))

    def test_placement_stop_distance_ends_forward_route_and_continues(self) -> None:
        mission = LargeQuadrupedMission(load_config(), dry_run=True, skip_arm=True)
        mission.context.dry_run = False
        mission.state = MissionState.PLACE_TO_LETTER_BOX
        mission._placement_route_active = True
        mission._check_safety = lambda: None
        events = []

        def go_distance(_distance, *, speed_mps=None):
            self.assertEqual(speed_mps, 0.08)
            events.append("forward")
            raise ForwardMotionGuardStop(
                "motion guard stopped forward command: "
                "ultrasound=0.34m threshold=0.35m state=PLACE_TO_LETTER_BOX"
            )

        mission.motion.go_distance = go_distance
        mission.motion.turn_by = lambda _yaw: events.append("turn")
        mission._confirm_placement_front_stop = mock.Mock(return_value=True)
        mission._execute_route_action({"action": "forward", "distance_m": 2.70})
        mission._execute_route_action({"action": "turn", "yaw_rad": 1.57})

        self.assertEqual(events, ["forward", "turn"])

    def test_placement_strafe_front_hold_corrects_both_directions(self) -> None:
        mission = LargeQuadrupedMission(load_config(), dry_run=True, skip_arm=True)
        for distance_m, expected_vx in ((0.33, 0.025), (0.23, -0.025), (0.29, 0.0)):
            with self.subTest(distance_m=distance_m):
                mission._placement_front_distance = mock.Mock(  # noqa: SLF001
                    return_value=(distance_m, {})
                )
                self.assertAlmostEqual(
                    mission._placement_strafe_front_velocity(),  # noqa: SLF001
                    expected_vx,
                )

    def test_non_placement_stop_distance_still_aborts_route(self) -> None:
        mission = LargeQuadrupedMission(load_config(), dry_run=True, skip_arm=True)
        mission.context.dry_run = False
        mission.state = MissionState.PASS_OBSTACLE
        mission._check_safety = lambda: None
        mission.motion.go_distance = mock.Mock(
            side_effect=MissionAbort(
                "motion guard stopped forward command: "
                "ultrasound=0.34m threshold=0.35m state=PASS_OBSTACLE"
            )
        )

        with self.assertRaises(MissionAbort):
            mission._execute_route_action({"action": "forward", "distance_m": 1.0})

    def test_pickup_route_confirmed_front_stop_completes_forward_action(self) -> None:
        mission = LargeQuadrupedMission(load_config(), dry_run=True, skip_arm=True)
        mission.context.dry_run = False
        mission.state = MissionState.PICK_RED_BAR
        mission._check_safety = mock.Mock()
        mission._prime_placement_front_filter = mock.Mock()
        mission._confirm_placement_front_stop = mock.Mock(return_value=True)
        mission.state_reader = mock.Mock()
        mission.state_reader.pose.side_effect = [
            (1.0, 2.0, 0.0),
            (1.0, 2.0, 0.0),
        ]
        mission.motion = mock.Mock()
        mission.motion.go_distance.side_effect = ForwardMotionGuardStop(
            "motion guard stopped forward command: "
            "ultrasound=0.28m threshold=0.35m state=PICK_RED_BAR"
        )

        mission._execute_route_action(
            {
                "action": "forward",
                "distance_m": 1.35,
                "front_stop_is_completion": True,
            }
        )

        mission.motion.go_distance.assert_called_once_with(1.35)
        mission.motion.stop.assert_called_once_with()
        mission._confirm_placement_front_stop.assert_called_once_with()

    def test_inspection_motion_guard_uses_28cm_override(self) -> None:
        mission = LargeQuadrupedMission(load_config(), dry_run=True, skip_arm=True)
        mission.context.dry_run = False
        mission.state = MissionState.INSPECT_LEFT_OBJECT
        mission.state_reader = SimpleNamespace(
            state=SimpleNamespace(front_ultrasound_m=0.30),
            safety_error=lambda **_kwargs: None,
        )

        mission._motion_guard(0.05, 0.0, 0.0)
        mission.state_reader.state.front_ultrasound_m = 0.28
        with self.assertRaisesRegex(
            MissionAbort,
            r"threshold=0\.28m state=INSPECT_LEFT_OBJECT",
        ):
            mission._motion_guard(0.05, 0.0, 0.0)

        mission.state = MissionState.PASS_OBSTACLE
        mission.state_reader.state.front_ultrasound_m = 0.30
        with self.assertRaisesRegex(
            MissionAbort,
            r"threshold=0\.35m state=PASS_OBSTACLE",
        ):
            mission._motion_guard(0.05, 0.0, 0.0)

    def test_feedback_velocity_refreshes_command_and_stops(self) -> None:
        class RecordingBackend:
            name = "recording"

            def __init__(self) -> None:
                self.velocities = []

            def send_velocity(self, vx, vy, wz) -> None:
                self.velocities.append((vx, vy, wz))

        controller = Lite3MotionController(load_config(), dry_run=True)
        controller.dry_run = False
        backend = RecordingBackend()
        controller.backend = backend
        calls = 0

        def provide_velocity():
            nonlocal calls
            calls += 1
            return (-0.01 * calls, 0.08, -0.02 * calls)

        controller.hold_velocity_feedback(provide_velocity, 0.11)

        nonzero = [velocity for velocity in backend.velocities if velocity != (0.0, 0.0, 0.0)]
        self.assertGreaterEqual(calls, 2)
        self.assertGreaterEqual(len(set(nonzero)), 2)
        self.assertEqual(backend.velocities[-1], (0.0, 0.0, 0.0))

    def test_invalid_feedback_velocity_still_stops(self) -> None:
        class RecordingBackend:
            name = "recording"

            def __init__(self) -> None:
                self.velocities = []

            def send_velocity(self, vx, vy, wz) -> None:
                self.velocities.append((vx, vy, wz))

        controller = Lite3MotionController(load_config(), dry_run=True)
        controller.dry_run = False
        backend = RecordingBackend()
        controller.backend = backend

        with self.assertRaisesRegex(RuntimeError, "three finite values"):
            controller.hold_velocity_feedback(
                lambda: (0.0, float("nan"), 0.0),
                0.05,
            )

        self.assertEqual(backend.velocities[-1], (0.0, 0.0, 0.0))

    def test_closed_loop_distance_stops_from_odometry_progress(self) -> None:
        class RecordingBackend:
            name = "recording"

            def __init__(self) -> None:
                self.velocities = []

            def send_velocity(self, vx, vy, wz) -> None:
                self.velocities.append((vx, vy, wz))

        controller = Lite3MotionController(load_config(), dry_run=True)
        controller.dry_run = False
        backend = RecordingBackend()
        controller.backend = backend
        poses = iter(((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.08, 0.0, 0.0), (0.18, 0.0, 0.0)))
        controller.configure_safety(lambda vx, vy, wz: None, lambda: next(poses), feedback_required=True)

        controller.go_distance(0.20, speed_mps=0.10)

        self.assertTrue(any(vx > 0.0 for vx, _, _ in backend.velocities))
        self.assertEqual(backend.velocities[-1], (0.0, 0.0, 0.0))

    def test_forward_translation_corrects_cross_track_and_yaw_drift(self) -> None:
        backend = mock.Mock()
        backend.name = "recording"
        backend.velocities = []
        backend.send_velocity.side_effect = lambda vx, vy, wz: backend.velocities.append(
            (vx, vy, wz)
        )
        controller = Lite3MotionController(load_config(), dry_run=True)
        controller.dry_run = False
        controller.backend = backend
        controller.translation_max_cross_track_drift_m = 0.30
        poses = iter(
            (
                (0.0, 0.0, 0.0),
                (0.0, 0.02, 0.02),
                (0.0, 0.18, 0.0),
                (0.18, 0.18, 0.0),
            )
        )
        controller.configure_safety(
            lambda vx, vy, wz: None,
            lambda: next(poses),
            feedback_required=True,
        )

        controller.go_distance(0.20, speed_mps=0.10)

        nonzero = [item for item in backend.velocities if item != (0.0, 0.0, 0.0)]
        self.assertEqual(len(nonzero), 2)
        self.assertGreater(nonzero[0][0], 0.0)
        self.assertLess(nonzero[0][1], 0.0)
        self.assertLess(nonzero[0][2], 0.0)

    def test_reverse_translation_uses_direction_correct_cross_track_hold(self) -> None:
        backend = mock.Mock()
        backend.name = "recording"
        backend.velocities = []
        backend.send_velocity.side_effect = lambda vx, vy, wz: backend.velocities.append(
            (vx, vy, wz)
        )
        controller = Lite3MotionController(load_config(), dry_run=True)
        controller.dry_run = False
        controller.backend = backend
        poses = iter(
            (
                (0.0, 0.0, 0.0),
                (-0.04, 0.02, -0.02),
                (-0.18, 0.0, 0.0),
            )
        )
        controller.configure_safety(
            lambda vx, vy, wz: None,
            lambda: next(poses),
            feedback_required=True,
        )

        controller.go_distance(-0.20, speed_mps=0.10)

        command = next(item for item in backend.velocities if item != (0.0, 0.0, 0.0))
        self.assertLess(command[0], 0.0)
        self.assertLess(command[1], 0.0)
        self.assertGreater(command[2], 0.0)

    def test_strafe_translation_corrects_forward_and_yaw_drift(self) -> None:
        backend = mock.Mock()
        backend.name = "recording"
        backend.velocities = []
        backend.send_velocity.side_effect = lambda vx, vy, wz: backend.velocities.append(
            (vx, vy, wz)
        )
        controller = Lite3MotionController(load_config(), dry_run=True)
        controller.dry_run = False
        controller.backend = backend
        poses = iter(
            (
                (0.0, 0.0, 0.0),
                (0.02, 0.04, 0.02),
                (0.0, 0.18, 0.0),
            )
        )
        controller.configure_safety(
            lambda vx, vy, wz: None,
            lambda: next(poses),
            feedback_required=True,
        )

        controller.strafe_distance(0.20, speed_mps=0.10)

        command = next(item for item in backend.velocities if item != (0.0, 0.0, 0.0))
        self.assertLess(command[0], 0.0)
        self.assertGreater(command[1], 0.0)
        self.assertLess(command[2], 0.0)

    def test_stop_failure_does_not_replace_motion_guard_error(self) -> None:
        class StopFailingBackend:
            name = "stop-failing"

            def send_velocity(self, vx, vy, wz) -> None:
                if (vx, vy, wz) == (0.0, 0.0, 0.0):
                    raise OSError("stop link down")

        controller = Lite3MotionController(load_config(), dry_run=True)
        controller.dry_run = False
        controller.backend = StopFailingBackend()

        def reject(vx, vy, wz) -> None:
            raise RuntimeError("original safety failure")

        controller.configure_safety(reject, lambda: (0.0, 0.0, 0.0), feedback_required=False)
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "original safety failure"):
            controller.hold_velocity(0.1, 0.0, 0.0, 0.1)

    def test_place_failure_preserves_carried_bar_and_retries_placement_only(self) -> None:
        config = load_config()
        mission = LargeQuadrupedMission(config, dry_run=True, skip_arm=True)
        mission.context.carried_bar = True
        mission.context.target_letter = "A"
        mission._run_placement_letter_navigator = mock.Mock(return_value=0.0)
        mission._run_scripted_route = mock.Mock(  # type: ignore[method-assign]
            side_effect=lambda _name: (
                mission._execute_placement_letter_approach() is None
            )
        )
        mission._align_to_letter_box = mock.Mock(return_value=0.2)  # type: ignore[method-assign]
        mission.arm = mock.Mock()
        mission.arm.place_to_box.return_value = False
        mission.motion = mock.Mock()

        with contextlib.redirect_stdout(io.StringIO()):
            placed = mission._place_carried_bar()

        self.assertFalse(placed)
        self.assertTrue(mission.context.carried_bar)
        self.assertEqual(mission.context.target_letter, "A")
        mission.motion.strafe_distance.assert_not_called()
        mission._pick_target = mock.Mock()
        with self.assertRaisesRegex(
            MissionAbort,
            "failed to place second carried bar",
        ):
            mission._state_second_pick_place()
        mission._pick_target.assert_not_called()

    def test_grasp_transport_failure_blocks_retry_and_preserves_carried_bar(self) -> None:
        mission = LargeQuadrupedMission(load_config(), dry_run=True, skip_arm=True)
        mission.context.carried_bar = False
        mission._pregrasp_ultrasound_ready = mock.Mock(return_value=True)
        mission.arm = mock.Mock()
        mission.arm.grasp_red_bar.return_value = ArmTaskResult(
            False,
            "CARGO_POSE",
            reason="cargo pose failed",
            object_held=True,
        )

        with self.assertRaisesRegex(MissionAbort, "object is held"):
            mission._retry_grasp(260.0)

        self.assertTrue(mission.context.carried_bar)
        mission.arm.grasp_red_bar.assert_called_once_with(260.0)

    def test_successful_place_enters_moving_pose_before_continuing(self) -> None:
        config = load_config()
        config["pickup_transfer"] = dict(config["pickup_transfer"])
        config["pickup_transfer"]["enabled"] = False
        config["inspection"] = dict(config["inspection"])
        config["inspection"]["place_pause_seconds"] = 0.0
        mission = LargeQuadrupedMission(config, dry_run=True, skip_arm=True)
        mission.context.carried_bar = True
        mission.context.target_letter = "A"
        mission._run_scripted_route = mock.Mock(return_value=True)  # type: ignore[method-assign]
        mission._align_to_letter_box = mock.Mock(return_value=0.2)  # type: ignore[method-assign]
        events: list[str] = []
        mission.arm = mock.Mock()
        mission.arm.place_to_box.side_effect = lambda _slot: (
            events.append("place")
            or ArmTaskResult(
                True,
                "DONE",
                object_held=False,
                released=True,
            )
        )
        mission.arm.stow.side_effect = lambda: (
            events.append("moving_pose")
            or ArmTaskResult.success("MOVING_POSE")
        )
        mission.motion = mock.Mock()
        mission.motion.stop.side_effect = lambda: events.append("stop")
        mission.motion.strafe_distance.side_effect = (
            lambda _distance, **_kwargs: events.append("continue")
        )

        with contextlib.redirect_stdout(io.StringIO()):
            placed = mission._place_carried_bar()

        self.assertTrue(placed)
        self.assertEqual(events, ["stop", "place", "stop", "moving_pose", "continue"])
        self.assertFalse(mission.context.carried_bar)
        self.assertEqual(mission.context.placed_letters, ["A"])

    def test_mission_exception_returns_failed_result_and_nonzero_exit(self) -> None:
        mission = LargeQuadrupedMission(load_config(), dry_run=True, skip_arm=True)
        mission._state_pass_obstacle = mock.Mock(side_effect=RuntimeError("route failure"))  # type: ignore[method-assign]
        with contextlib.redirect_stdout(io.StringIO()):
            result = mission.run()
        self.assertFalse(result.ok)
        self.assertEqual(result.exit_code, 1)
        self.assertEqual(result.state, "ABORT_SAFE")
        self.assertIn("route failure", result.reason)

    def test_real_mode_fault_hold_retries_safe_state_without_exiting(self) -> None:
        config = load_config()
        config["fault_hold"] = dict(config["fault_hold"])
        config["fault_hold"].update(
            {"poll_interval_s": 0.0, "recovery_stable_checks": 1}
        )
        mission = LargeQuadrupedMission(
            config,
            dry_run=True,
            skip_arm=True,
            fault_hold_sleep=lambda _seconds: None,
            fault_hold_max_cycles=5,
        )
        mission.context.dry_run = False
        mission.context.placed_letters = ["A", "B"]
        mission.motion = mock.Mock()
        mission.state_reader = mock.Mock()
        mission.state_reader.safety_error.return_value = None
        mission.arm = mock.Mock()
        mission.arm.start.return_value = None
        for state in (
            MissionState.BOOT_CHECK,
            MissionState.STAND_AND_ARM,
            MissionState.INSPECT_LEFT_OBJECT,
            MissionState.INSPECT_RIGHT_OBJECT,
            MissionState.REPORT_RESULTS,
            MissionState.PICK_RED_BAR,
            MissionState.PLACE_TO_LETTER_BOX,
            MissionState.SECOND_PICK_PLACE,
            MissionState.FINISH_OR_SAFE_STOP,
        ):
            setattr(mission, f"_state_{state.name.lower()}", mock.Mock())
        pass_state = mock.Mock(side_effect=[RuntimeError("temporary state loss"), None])
        mission._state_pass_obstacle = pass_state

        with contextlib.redirect_stdout(io.StringIO()):
            result = mission.run()

        self.assertTrue(result.ok)
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(pass_state.call_count, 2)
        self.assertNotEqual(result.state, "ABORT_SAFE")

    def test_unsafe_state_hold_requires_resume_signal_and_stable_state(self) -> None:
        config = load_config()
        config["fault_hold"] = dict(config["fault_hold"])
        config["fault_hold"].update(
            {"poll_interval_s": 0.0, "recovery_stable_checks": 2}
        )
        resume_checks = iter((True, False))
        mission = LargeQuadrupedMission(
            config,
            dry_run=True,
            skip_arm=True,
            fault_hold_sleep=lambda _seconds: None,
            fault_resume_checker=lambda: next(resume_checks, False),
            fault_hold_max_cycles=3,
        )
        mission.motion = mock.Mock()
        mission.state_reader = mock.Mock()
        mission.state_reader.safety_error.return_value = None

        with contextlib.redirect_stdout(io.StringIO()):
            mission._fault_hold(
                MissionState.PICK_RED_BAR,
                RuntimeError("arm state unknown"),
            )

        self.assertEqual(mission.state, MissionState.PICK_RED_BAR)
        self.assertEqual(mission.motion.stop.call_count, 2)

    def test_pick_fault_retries_automatically_then_exits_at_retry_limit(self) -> None:
        class AdvancingClock:
            def __init__(self) -> None:
                self.now = 0.0

            def __call__(self) -> float:
                return self.now

            def sleep(self, seconds: float) -> None:
                self.now += seconds

        config = load_config()
        config["fault_hold"] = dict(config["fault_hold"])
        config["fault_hold"].update(
            {
                "poll_interval_s": 0.1,
                "recovery_stable_checks": 1,
                "max_wait_s": 0.5,
            }
        )
        clock = AdvancingClock()
        mission = LargeQuadrupedMission(
            config,
            dry_run=True,
            skip_arm=True,
            fault_hold_sleep=clock.sleep,
            fault_hold_clock=clock,
            fault_resume_checker=lambda: False,
        )
        mission.context.dry_run = False
        mission.motion = mock.Mock()
        mission.state_reader = mock.Mock()
        mission.state_reader.safety_error.return_value = None
        mission.arm = mock.Mock()
        mission.arm.start.return_value = None
        for state in (
            MissionState.BOOT_CHECK,
            MissionState.STAND_AND_ARM,
            MissionState.PASS_OBSTACLE,
            MissionState.INSPECT_LEFT_OBJECT,
            MissionState.INSPECT_RIGHT_OBJECT,
            MissionState.REPORT_RESULTS,
            MissionState.PLACE_TO_LETTER_BOX,
            MissionState.SECOND_PICK_PLACE,
            MissionState.FINISH_OR_SAFE_STOP,
        ):
            setattr(mission, f"_state_{state.name.lower()}", mock.Mock())
        mission._state_pick_red_bar = mock.Mock(
            side_effect=RuntimeError("red target unavailable")
        )

        with contextlib.redirect_stdout(io.StringIO()):
            result = mission.run()

        self.assertFalse(result.ok)
        self.assertEqual(result.exit_code, 1)
        self.assertEqual(result.state, "ABORT_SAFE")
        self.assertIn("retry limit reached after 3 failures", result.reason)
        self.assertEqual(mission._state_pick_red_bar.call_count, 3)
        self.assertIn("red target unavailable", result.reason)

    def test_fault_hold_stable_counter_is_capped(self) -> None:
        config = load_config()
        config["fault_hold"] = dict(config["fault_hold"])
        config["fault_hold"].update(
            {
                "poll_interval_s": 0.0,
                "recovery_stable_checks": 5,
                "max_wait_s": 30.0,
            }
        )
        mission = LargeQuadrupedMission(
            config,
            dry_run=True,
            skip_arm=True,
            fault_hold_sleep=lambda _seconds: None,
            fault_hold_clock=lambda: 0.0,
            fault_resume_checker=lambda: False,
            fault_hold_max_cycles=20,
        )
        mission.motion = mock.Mock()
        mission.state_reader = mock.Mock()
        mission.state_reader.safety_error.return_value = None
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            with self.assertRaisesRegex(MissionAbort, "test limit"):
                mission._fault_hold(
                    MissionState.INSPECT_LEFT_OBJECT,
                    RuntimeError("arm state unknown"),
                )

        self.assertIn("stable=5/5", output.getvalue())
        self.assertNotIn("stable=20/5", output.getvalue())

    def test_permanently_failing_safe_state_exits_after_two_retries(self) -> None:
        config = load_config()
        config["fault_hold"] = dict(config["fault_hold"])
        config["fault_hold"].update(
            {
                "poll_interval_s": 0.0,
                "recovery_stable_checks": 1,
                "max_retries_per_state": 2,
            }
        )
        mission = LargeQuadrupedMission(
            config,
            dry_run=True,
            skip_arm=True,
            fault_hold_sleep=lambda _seconds: None,
        )
        mission.context.dry_run = False
        mission.context.placed_letters = ["A", "B"]
        mission.motion = mock.Mock()
        mission.state_reader = mock.Mock()
        mission.state_reader.safety_error.return_value = None
        mission.arm = mock.Mock()
        mission.arm.start.return_value = None
        for state in (
            MissionState.BOOT_CHECK,
            MissionState.STAND_AND_ARM,
            MissionState.INSPECT_LEFT_OBJECT,
            MissionState.INSPECT_RIGHT_OBJECT,
            MissionState.REPORT_RESULTS,
            MissionState.PICK_RED_BAR,
            MissionState.PLACE_TO_LETTER_BOX,
            MissionState.SECOND_PICK_PLACE,
            MissionState.FINISH_OR_SAFE_STOP,
        ):
            setattr(mission, f"_state_{state.name.lower()}", mock.Mock())
        pass_state = mock.Mock(side_effect=RuntimeError("persistent odom fault"))
        mission._state_pass_obstacle = pass_state

        with contextlib.redirect_stdout(io.StringIO()):
            result = mission.run()

        self.assertFalse(result.ok)
        self.assertEqual(result.exit_code, 1)
        self.assertEqual(result.state, "ABORT_SAFE")
        self.assertEqual(pass_state.call_count, 3)
        self.assertIn("retry limit reached after 3 failures", result.reason)
        self.assertIn("allowed_retries=2", result.reason)

    def test_final_placement_retry_consumes_one_resume_per_attempt(self) -> None:
        config = load_config()
        config["fault_hold"] = dict(config["fault_hold"])
        config["fault_hold"].update(
            {"poll_interval_s": 0.0, "recovery_stable_checks": 1}
        )
        resume_calls = 0

        def resume_checker() -> bool:
            nonlocal resume_calls
            resume_calls += 1
            return True

        mission = LargeQuadrupedMission(
            config,
            dry_run=True,
            skip_arm=True,
            fault_hold_sleep=lambda _seconds: None,
            fault_resume_checker=resume_checker,
            fault_hold_max_cycles=2,
        )
        mission.context.dry_run = False
        mission.context.placed_letters = ["A"]
        mission.motion = mock.Mock()
        mission.state_reader = mock.Mock()
        mission.state_reader.safety_error.return_value = None
        mission.arm = mock.Mock()
        mission.arm.start.return_value = None
        for state in (
            MissionState.BOOT_CHECK,
            MissionState.STAND_AND_ARM,
            MissionState.PASS_OBSTACLE,
            MissionState.INSPECT_LEFT_OBJECT,
            MissionState.INSPECT_RIGHT_OBJECT,
            MissionState.REPORT_RESULTS,
            MissionState.PICK_RED_BAR,
            MissionState.PLACE_TO_LETTER_BOX,
            MissionState.FINISH_OR_SAFE_STOP,
        ):
            setattr(mission, f"_state_{state.name.lower()}", mock.Mock())

        second_attempts = 0

        def second_pick_place() -> None:
            nonlocal second_attempts
            second_attempts += 1
            if second_attempts == 1:
                return
            if second_attempts == 2:
                raise RuntimeError("transient second placement failure")
            mission.context.placed_letters.append("B")

        mission._state_second_pick_place = mock.Mock(side_effect=second_pick_place)

        with contextlib.redirect_stdout(io.StringIO()):
            result = mission.run()

        self.assertTrue(result.ok)
        self.assertEqual(mission._state_second_pick_place.call_count, 3)
        self.assertEqual(resume_calls, 2)

    def test_recoverable_safety_error_stops_without_soft_estop(self) -> None:
        mission = LargeQuadrupedMission(load_config(), dry_run=True, skip_arm=True)
        mission.context.dry_run = False
        mission.motion = mock.Mock()
        mission.state_reader = mock.Mock()
        mission.state_reader.safety_error.return_value = "battery below limit"

        with self.assertRaisesRegex(MissionAbort, "battery below limit"):
            mission._check_safety()

        mission.motion.stop.assert_called_once_with()
        mission.motion.soft_estop.assert_not_called()

    def test_report_results_does_not_sort_and_rebroadcast(self) -> None:
        config = load_config()
        mission = LargeQuadrupedMission(config, dry_run=True, skip_arm=True)
        mission.context.records["C"] = InspectionRecord("C", "偏高", "异常", 0.9, 1)
        mission.context.records["A"] = InspectionRecord("A", "偏低", "异常", 0.9, 2)
        mission.audio.say_record = mock.Mock()  # type: ignore[method-assign]
        with contextlib.redirect_stdout(io.StringIO()):
            mission._state_report_results()
        self.assertEqual(mission.context.anomalous_letters(), ["C", "A"])
        mission.audio.say_record.assert_not_called()

    def test_collect_inspection_speaks_detected_letter_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config()
            config["inspection"] = dict(config["inspection"])
            config["inspection"].update(
                {
                    "stop_dwell_seconds": 0.01,
                    "round_result_path": str(Path(tmp) / "round_result.json"),
                    "latest_stop_result_path": str(Path(tmp) / "latest_stop_result.json"),
                }
            )
            mission = LargeQuadrupedMission(config, dry_run=True, skip_arm=True)
            mission.context.dry_run = False
            mission.motion = mock.Mock()
            mission.front_camera = mock.Mock()
            raw_frame = object()
            prepared_frame = object()
            mission.front_camera.read.return_value = raw_frame
            mission.inspection_undistorter = mock.Mock()
            mission.inspection_undistorter.apply.return_value = prepared_frame
            mission.vision = mock.Mock()
            mission.vision.inspect_frame.return_value = InspectionRecord("D", "偏高", "异常", 0.88, 7, source_camera="front")
            mission.vision.reset_inspection_votes.return_value = None
            mission.audio.say_record = mock.Mock()  # type: ignore[method-assign]
            mission._save_inspection_evidence = mock.Mock(return_value=None)  # type: ignore[method-assign]
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                mission._collect_inspection("inspection_stop_test", default_results=[("A", "偏低")])
        self.assertEqual(list(mission.context.records), ["D"])
        mission.audio.say_record.assert_called_once()
        spoken = mission.audio.say_record.call_args.args[0]
        self.assertEqual(spoken.letter, "D")
        self.assertEqual(mission.context.reported_letters, ["D"])
        self.assertIn("[mission] 播报内容: D区域仪表盘显示偏高，状态异常", output.getvalue())
        mission.front_camera.read.assert_called_once()
        mission.inspection_undistorter.apply.assert_called_once_with(raw_frame)
        mission.vision.inspect_frame.assert_called_once_with(
            prepared_frame,
            source_camera="front_wide_undistorted",
        )

    def test_collect_inspection_blocks_without_fabricating_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config()
            config["inspection"] = dict(config["inspection"])
            config["inspection"].update(
                {
                    "stop_dwell_seconds": 0.01,
                    "round_result_path": str(Path(tmp) / "round_result.json"),
                    "latest_stop_result_path": str(Path(tmp) / "latest_stop_result.json"),
                }
            )
            mission = LargeQuadrupedMission(config, dry_run=True, skip_arm=True)
            mission.context.dry_run = False
            mission.motion = mock.Mock()
            mission.front_camera = mock.Mock()
            mission.front_camera.read.return_value = None
            mission.vision = mock.Mock()
            mission.vision.reset_inspection_votes.return_value = None
            mission.audio.say_record = mock.Mock()  # type: ignore[method-assign]
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(MissionAbort):
                mission._collect_inspection("inspection_stop_test", default_results=[("A", "偏低")])
            round_data = load_round_result(Path(tmp) / "round_result.json")
        self.assertEqual(mission.context.records, {})
        self.assertFalse(round_data["ready"])
        self.assertEqual(round_data["block_reason"], "camera_failed")
        mission.audio.say_record.assert_not_called()

    def test_collect_inspection_uses_default_when_frames_never_stabilize(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config()
            config["inspection"] = dict(config["inspection"])
            config["inspection"].update(
                {
                    "stop_dwell_seconds": 0.01,
                    "round_result_path": str(Path(tmp) / "round_result.json"),
                    "latest_stop_result_path": str(Path(tmp) / "latest_stop_result.json"),
                }
            )
            mission = LargeQuadrupedMission(config, dry_run=True, skip_arm=True)
            mission.context.dry_run = False
            mission.motion = mock.Mock()
            mission.front_camera = mock.Mock()
            mission.front_camera.read.return_value = object()
            mission.inspection_undistorter = None
            mission.vision = mock.Mock()
            mission.vision.inspect_frame.return_value = None
            mission.vision.reset_inspection_votes.return_value = None
            mission.audio.say_record = mock.Mock()  # type: ignore[method-assign]
            mission._save_inspection_evidence = mock.Mock(return_value=None)  # type: ignore[method-assign]
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                mission._collect_inspection(
                    "inspection_stop_test",
                    default_results=[("A", "偏低")],
                )
            round_data = load_round_result(Path(tmp) / "round_result.json")

        record = mission.context.records["A"]
        self.assertEqual(record.level, "偏低")
        self.assertEqual(record.state, "异常")
        self.assertEqual(record.confidence, 0.0)
        self.assertEqual(record.source_camera, "default_fallback")
        self.assertEqual(record.stability_votes, {"default_fallback": 1})
        self.assertEqual(round_data["records"]["A"]["source_camera"], "default_fallback")
        self.assertIn("use default area=A level=偏低", output.getvalue())
        mission.audio.say_record.assert_called_once()

    def test_last_default_balances_two_normal_two_abnormal(self) -> None:
        mission = LargeQuadrupedMission(load_config(), dry_run=True, skip_arm=True)
        mission.context.records.update(
            {
                "A": InspectionRecord("A", "正常", "正常", 0.9, 1),
                "B": InspectionRecord("B", "正常", "正常", 0.9, 2),
                "C": InspectionRecord("C", "偏高", "异常", 0.9, 3),
            }
        )

        selected = mission._select_unused_defaults([("D", "正常")])

        self.assertEqual(selected, [("D", "偏低")])

    def test_collect_inspection_uses_best_real_candidate_before_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config()
            config["inspection"] = dict(config["inspection"])
            config["inspection"].update(
                {
                    "stop_dwell_seconds": 0.01,
                    "speak_at_inspection_stop": False,
                    "round_result_path": str(Path(tmp) / "round_result.json"),
                    "latest_stop_result_path": str(Path(tmp) / "latest_stop_result.json"),
                }
            )
            mission = LargeQuadrupedMission(config, dry_run=True, skip_arm=True)
            mission.context.dry_run = False
            mission.motion = mock.Mock()
            mission.front_camera = mock.Mock()
            observed_frame = object()
            candidate_frame = object()
            mission.front_camera.read.return_value = observed_frame
            mission.inspection_undistorter = None
            mission.vision = mock.Mock()
            mission.vision.inspect_frame.return_value = None
            candidate = InspectionRecord(
                "D",
                "偏低",
                "异常",
                0.83,
                9,
                source_camera="front",
                stability_votes={"runtime_best": 1},
            )
            mission.vision.best_inspection_candidate.return_value = (
                candidate,
                candidate_frame,
            )
            mission._save_inspection_evidence = mock.Mock(return_value=None)  # type: ignore[method-assign]
            output = io.StringIO()
            with mock.patch(
                "mission_lite3.mission.time.monotonic",
                side_effect=[0.0, 0.0, 2.0],
            ):
                with contextlib.redirect_stdout(output):
                    mission._collect_inspection(
                        "inspection_stop_test",
                        default_results=[("A", "正常")],
                    )

        self.assertEqual(list(mission.context.records), ["D"])
        self.assertEqual(mission.context.records["D"].source_camera, "front")
        mission._save_inspection_evidence.assert_called_once_with(
            "inspection_stop_test",
            "D",
            candidate_frame,
        )
        self.assertIn("use best real candidate area=D level=偏低", output.getvalue())

    def test_duplicate_stable_letter_uses_an_unused_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config()
            config["inspection"] = dict(config["inspection"])
            config["inspection"].update(
                {
                    "stop_dwell_seconds": 0.01,
                    "round_result_path": str(Path(tmp) / "round_result.json"),
                    "latest_stop_result_path": str(Path(tmp) / "latest_stop_result.json"),
                }
            )
            mission = LargeQuadrupedMission(config, dry_run=True, skip_arm=True)
            mission.context.dry_run = False
            mission.context.records["A"] = InspectionRecord(
                "A", "偏低", "异常", 0.0, -1, source_camera="default_fallback"
            )
            original_c = InspectionRecord("C", "正常", "正常", 0.9, 3, source_camera="front")
            mission.context.records["C"] = original_c
            mission.motion = mock.Mock()
            mission.front_camera = mock.Mock()
            mission.front_camera.read.side_effect = [object()] + [None] * 30
            mission.inspection_undistorter = None
            mission.vision = mock.Mock()
            mission.vision.inspect_frame.return_value = InspectionRecord(
                "C", "偏高", "异常", 0.8, 4, source_camera="front"
            )
            mission.audio.say_record = mock.Mock()  # type: ignore[method-assign]
            mission._save_inspection_evidence = mock.Mock(return_value=None)  # type: ignore[method-assign]
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                mission._collect_inspection(
                    "inspection_stop_test",
                    default_results=[("C", "偏高")],
                )

        self.assertIs(mission.context.records["C"], original_c)
        self.assertEqual(mission.context.records["B"].level, "偏高")
        self.assertEqual(
            mission.context.records["B"].source_camera,
            "default_fallback",
        )
        self.assertIn("duplicate stable area=C", output.getvalue())
        self.assertIn("use default area=B level=偏高", output.getvalue())

    def test_stable_detection_replaces_and_relocates_prior_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config()
            config["inspection"] = dict(config["inspection"])
            config["inspection"].update(
                {
                    "round_result_path": str(Path(tmp) / "round_result.json"),
                    "latest_stop_result_path": str(Path(tmp) / "latest_stop_result.json"),
                }
            )
            mission = LargeQuadrupedMission(config, dry_run=True, skip_arm=True)
            mission.context.dry_run = False
            fallback = InspectionRecord(
                "A", "偏低", "异常", 0.0, -1, source_camera="default_fallback"
            )
            mission.context.records["A"] = fallback
            mission.motion = mock.Mock()
            mission.front_camera = mock.Mock()
            mission.front_camera.read.return_value = object()
            mission.inspection_undistorter = None
            mission.vision = mock.Mock()
            detected = InspectionRecord(
                "A", "正常", "正常", 0.9, 5, source_camera="front"
            )
            mission.vision.inspect_frame.return_value = detected
            mission.audio.say_record = mock.Mock()  # type: ignore[method-assign]
            mission._save_inspection_evidence = mock.Mock(return_value=None)  # type: ignore[method-assign]
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                mission._collect_inspection(
                    "inspection_stop_test",
                    default_results=[("D", "正常")],
                )

        self.assertIs(mission.context.records["A"], detected)
        self.assertEqual(mission.context.records["B"].level, "偏低")
        self.assertEqual(
            mission.context.records["B"].source_camera,
            "default_fallback",
        )
        self.assertIn("move fallback to unused area=B", output.getvalue())

    def test_shifted_start_route_matches_current_stop_1_tuning(self) -> None:
        config = load_config()
        default_config = config_loader.DEFAULT_FIELD
        default_route = default_config["scripted_route"]
        tracked_route = config["scripted_route"]
        self.assertEqual(default_config["waypoints"]["start"], config["waypoints"]["start"])
        self.assertEqual(default_route["pass_obstacle"], tracked_route["pass_obstacle"])
        self.assertEqual(default_route["inspect_stop_1_arrive"], tracked_route["inspect_stop_1_arrive"])
        self.assertEqual(
            default_route["pickup_from_upper_inspection"],
            tracked_route["pickup_from_upper_inspection"],
        )
        pass_obstacle = config["scripted_route"]["pass_obstacle"][0]
        pickup = config["scripted_route"]["pickup_from_upper_inspection"]
        stop_1 = config["scripted_route"]["inspect_stop_1_arrive"]
        self.assertAlmostEqual(config["waypoints"]["start"]["y"], 0.40)
        self.assertAlmostEqual(config["waypoints"]["start"]["yaw"], 3.1416)
        self.assertEqual(pass_obstacle["steps"], 8)
        self.assertAlmostEqual(pass_obstacle["clear_step_m"], 0.30)
        self.assertAlmostEqual(
            config["startup_avoidance"]["decision"]["finish_forward_m"],
            pass_obstacle["steps"] * pass_obstacle["clear_step_m"],
        )
        self.assertEqual(pickup[0]["action"], "turn")
        self.assertAlmostEqual(pickup[0]["yaw_rad"], 1.5708)
        self.assertEqual(pickup[0]["note"], "face map-down for pickup and placement")
        self.assertEqual(pickup[1]["action"], "forward")
        self.assertAlmostEqual(pickup[1]["distance_m"], 1.35)
        self.assertNotIn("front_stop_is_completion", pickup[1])
        self.assertEqual(pickup[1]["note"], "move map-down from inspection stop 4 toward pickup row")
        self.assertEqual(pickup[2]["action"], "strafe")
        self.assertAlmostEqual(pickup[2]["distance_m"], 1.30)
        self.assertEqual(pickup[2]["note"], "move map-right into pickup lane")
        pickup_from_place = config["scripted_route"]["pickup_from_place"]
        self.assertIs(pickup_from_place[1]["front_stop_is_completion"], True)
        self.assertEqual(stop_1[0]["action"], "strafe")
        self.assertAlmostEqual(stop_1[0]["distance_m"], -0.65)
        self.assertEqual(stop_1[1]["action"], "forward")
        self.assertAlmostEqual(stop_1[1]["distance_m"], 0.40)
        poses = simulate_route_sequence(
            config,
            (
                "pass_obstacle",
                "inspect_stop_1_arrive",
                "inspect_stop_2_arrive",
                "inspect_stop_3_arrive",
                "inspect_stop_4_arrive",
                "inspect_stop_4_depart",
                "pickup_from_upper_inspection",
            ),
        )
        self.assertAlmostEqual(poses["inspect_stop_1_arrive"].x, -2.80, places=3)
        self.assertAlmostEqual(poses["inspect_stop_1_arrive"].y, 1.05, places=3)
        self.assertAlmostEqual(poses["pickup_from_upper_inspection"].x, -1.50, places=3)
        self.assertAlmostEqual(poses["pickup_from_upper_inspection"].y, 1.83, places=3)
        self.assertAlmostEqual(poses["pickup_from_upper_inspection"].yaw, -math.pi / 2, places=3)
        default_poses = simulate_route_sequence(
            default_config,
            ("pass_obstacle", "inspect_stop_1_arrive"),
        )
        self.assertAlmostEqual(default_poses["inspect_stop_1_arrive"].x, -2.80, places=3)
        self.assertAlmostEqual(default_poses["inspect_stop_1_arrive"].y, 1.05, places=3)

    def test_stand_state_skips_stand_up_when_robot_assumed_standing(self) -> None:
        class FakeMotion:
            def __init__(self) -> None:
                self.calls: list[str] = []

            def stand_up(self) -> None:
                self.calls.append("stand_up")

            def set_autonomous(self) -> None:
                self.calls.append("set_autonomous")

        class FakeArm:
            def __init__(self) -> None:
                self.calls: list[str] = []

            def stow(self) -> None:
                self.calls.append("stow")

        config = load_config()
        mission = LargeQuadrupedMission(config, dry_run=True, skip_arm=True)
        fake_motion = FakeMotion()
        fake_arm = FakeArm()
        mission.motion = fake_motion  # type: ignore[assignment]
        mission.arm = fake_arm  # type: ignore[assignment]
        with contextlib.redirect_stdout(io.StringIO()):
            mission._state_stand_and_arm()
        self.assertEqual(fake_motion.calls, ["set_autonomous"])
        self.assertEqual(fake_arm.calls, ["stow"])

    def test_corrupt_round_result_recovers_as_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "round_result.json"
            path.write_text("{bad-json", encoding="utf-8")
            data = load_round_result(path)
        self.assertFalse(data["ready"])
        self.assertEqual(data["block_reason"], "round_result_invalid")
        self.assertEqual(data["unknown_areas"], ["A", "B", "C", "D"])

    def test_round_result_ready_for_two_normal_two_abnormal(self) -> None:
        data = build_round_result(
            {
                "A": InspectionRecord("A", "偏低", "异常", 0.9, 10, stability_votes={"letter": 3, "dashboard": 3}),
                "B": InspectionRecord("B", "正常", "正常", 0.9, 11, stability_votes={"letter": 3, "dashboard": 3}),
                "C": InspectionRecord("C", "偏高", "异常", 0.9, 12, stability_votes={"letter": 3, "dashboard": 3}),
                "D": InspectionRecord("D", "正常", "正常", 0.9, 13, stability_votes={"letter": 3, "dashboard": 3}),
            },
            source_camera="front",
        )
        self.assertTrue(data["ready"])
        self.assertIsNone(data["block_reason"])
        self.assertEqual(data["abnormal_areas"], ["A", "C"])
        self.assertEqual(data["unknown_areas"], [])
        self.assertTrue(data["count_check"]["passed"])
        self.assertEqual(data["source_camera"], "front")
        self.assertEqual(data["stability_votes"]["A"], {"letter": 3, "dashboard": 3})

    def test_round_result_blocks_bad_count_check(self) -> None:
        data = build_round_result(
            {
                "A": InspectionRecord("A", "偏低", "异常", 0.9, 10),
                "B": InspectionRecord("B", "正常", "正常", 0.9, 11),
                "C": InspectionRecord("C", "正常", "正常", 0.9, 12),
                "D": InspectionRecord("D", "正常", "正常", 0.9, 13),
            }
        )
        self.assertFalse(data["ready"])
        self.assertEqual(data["block_reason"], "count_check_fail")
        self.assertFalse(data["count_check"]["passed"])

    def test_pickup_gate_rebalances_only_zero_confidence_fallback_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config()
            config["inspection"] = dict(config["inspection"])
            round_path = Path(tmp) / "round_result.json"
            config["inspection"]["round_result_path"] = str(round_path)
            mission = LargeQuadrupedMission(config, dry_run=True, skip_arm=True)
            mission.context.dry_run = False
            real_a = InspectionRecord(
                "A", "偏高", "异常", 0.841, 3, source_camera="front_wide_undistorted"
            )
            real_d = InspectionRecord(
                "D", "偏低", "异常", 0.825, 5, source_camera="front_wide_undistorted"
            )
            mission.context.records.update(
                {
                    "A": real_a,
                    "B": InspectionRecord(
                        "B", "偏低", "异常", 0.0, -1, source_camera="default_fallback"
                    ),
                    "C": InspectionRecord(
                        "C", "正常", "正常", 0.0, -1, source_camera="default_fallback"
                    ),
                    "D": real_d,
                }
            )

            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                allowed = mission._round_result_allows_pickup()
            data = load_round_result(round_path)

        self.assertTrue(allowed)
        self.assertIs(mission.context.records["A"], real_a)
        self.assertIs(mission.context.records["D"], real_d)
        self.assertEqual(mission.context.records["B"].state, "正常")
        self.assertEqual(mission.context.records["B"].level, "正常")
        self.assertEqual(mission.context.records["B"].confidence, 0.0)
        self.assertEqual(mission.context.records["B"].source_camera, "default_fallback")
        self.assertEqual(
            mission.context.records["B"].stability_votes["fallback_count_rebalanced"],
            1,
        )
        self.assertEqual(data["abnormal_areas"], ["A", "D"])
        self.assertTrue(data["ready"])
        self.assertIn("real visual records preserved", output.getvalue())

    def test_pickup_gate_does_not_rebalance_three_real_anomalies(self) -> None:
        mission = LargeQuadrupedMission(load_config(), dry_run=True, skip_arm=True)
        real_records = {
            letter: InspectionRecord(
                letter,
                "偏低",
                "异常",
                0.9,
                index,
                source_camera="front",
            )
            for index, letter in enumerate(("A", "B", "D"), start=1)
        }
        mission.context.records.update(real_records)
        mission.context.records["C"] = InspectionRecord(
            "C", "正常", "正常", 0.0, -1, source_camera="default_fallback"
        )

        allowed = mission._round_result_allows_pickup()

        self.assertFalse(allowed)
        for letter, record in real_records.items():
            self.assertIs(mission.context.records[letter], record)
        self.assertEqual(mission.context.records["C"].state, "正常")

    def test_count_gate_warning_does_not_abort_pickup_state(self) -> None:
        mission = LargeQuadrupedMission(load_config(), dry_run=True, skip_arm=True)
        mission._round_result_allows_pickup = mock.Mock(return_value=False)
        mission._next_target_letter = mock.Mock(return_value="A")
        mission._pick_target = mock.Mock(return_value=True)

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            mission._state_pick_red_bar()

        mission._pick_target.assert_called_once_with("A")
        self.assertIn("count gate warning", output.getvalue())

    def test_next_target_uses_low_confidence_fallback_when_no_anomaly_remains(self) -> None:
        mission = LargeQuadrupedMission(load_config(), dry_run=True, skip_arm=True)
        mission.context.records.update(
            {
                "A": InspectionRecord("A", "正常", "正常", 0.9, 1, source_camera="front"),
                "B": InspectionRecord(
                    "B", "正常", "正常", 0.0, -1, source_camera="default_fallback"
                ),
                "C": InspectionRecord("C", "正常", "正常", 0.8, 2, source_camera="front"),
                "D": InspectionRecord(
                    "D", "正常", "正常", 0.0, -1, source_camera="default_fallback"
                ),
            }
        )

        first = mission._next_target_letter()
        mission.context.placed_letters.append(first)
        second = mission._next_target_letter()

        self.assertEqual((first, second), ("B", "D"))

    def test_area_only_round_result_interface_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "round_result.json"
            path.write_text(
                json.dumps(
                    {
                        "ready": True,
                        "abnormal_areas": ["A", "C"],
                        "unknown_areas": [],
                        "count_check": {"normal": 2, "abnormal": 2, "unknown": 0, "passed": True},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            gate = evaluate_round_gate(path)
        self.assertTrue(gate.allowed)
        self.assertEqual(gate.abnormal_areas, ["A", "C"])
        self.assertEqual(gate.unknown_areas, [])

    def test_obstacle_forward_stops_after_configured_bypass_limit(self) -> None:
        class FakeMotion:
            def __init__(self) -> None:
                self.calls: list[tuple[str, float | None]] = []

            def stop(self) -> None:
                self.calls.append(("stop", None))

            def strafe_distance(self, distance_m: float) -> None:
                self.calls.append(("strafe", distance_m))

            def go_distance(self, distance_m: float) -> None:
                self.calls.append(("go", distance_m))

        config = load_config()
        mission = LargeQuadrupedMission(config, dry_run=True, skip_arm=True)
        fake_motion = FakeMotion()
        mission.motion = fake_motion  # type: ignore[assignment]
        mission._front_obstacle_check = lambda: ObstacleCheck(True, "test")  # type: ignore[method-assign]
        with contextlib.redirect_stdout(io.StringIO()):
            mission._run_obstacle_forward(
                {
                    "steps": 5,
                    "clear_step_m": 0.28,
                    "avoid_strafe_m": -0.35,
                    "avoid_forward_m": 0.30,
                    "max_avoid_attempts": 2,
                    "return_after_avoid": True,
                }
            )
        self.assertEqual(fake_motion.calls.count(("go", 0.30)), 2)
        self.assertEqual(fake_motion.calls.count(("strafe", -0.35)), 2)
        self.assertEqual(fake_motion.calls.count(("strafe", 0.35)), 2)
        self.assertEqual(fake_motion.calls.count(("stop", None)), 3)


if __name__ == "__main__":
    unittest.main()

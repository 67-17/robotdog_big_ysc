from __future__ import annotations

import contextlib
import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from mission_lite3.arm import ArmTaskResult
from mission_lite3.box_center_alignment import (
    BoxCenterAlignmentResult,
    PlacementLetterCandidate,
    PlacementLetterFrameResult,
)
from mission_lite3.config_loader import load_config
from mission_lite3.mission import (
    ForwardMotionGuardStop,
    LargeQuadrupedMission,
    MissionAbort,
    MissionState,
)
from mission_lite3.placement_letter_navigation import (
    ActionKind,
    LetterCandidate,
    NavigationAction,
    NavigationObservation,
    PlacementLetterNavigator,
)


def alignment_failure(*, rollback_ok: bool = True) -> BoxCenterAlignmentResult:
    return BoxCenterAlignmentResult(
        False,
        "missing_boxes" if rollback_ok else "missing_boxes;rollback_failed",
        "placement",
        "D",
        1,
        2,
        2,
        300.0,
        None,
        -0.25,
        0.0 if rollback_ok else -0.25,
        True,
        rollback_ok,
        None,
    )


def placement_detection(
    letter: str | None,
    *,
    center_x: float = 500.0,
    confidence: float = 0.90,
) -> PlacementLetterFrameResult:
    candidates = ()
    if letter is not None:
        candidates = (
            PlacementLetterCandidate(
                center=(center_x, 400.0),
                label_bbox=(450, 300, 100, 100),
                recognized_letter=letter,
                confidence=confidence,
            ),
        )
    return PlacementLetterFrameResult(
        bool(candidates),
        "" if candidates else "no_recognized_letter",
        1000,
        720,
        candidates,
    )


class FreshPlacementStateReader:
    def __init__(
        self,
        front_m: float = 0.60,
        positions: tuple[tuple[float, float, float], ...] = (
            (0.00, 0.00, 0.0),
            (0.00, 0.00, 0.0),
            (0.03, 0.00, 0.0),
        ),
    ) -> None:
        self.front_m = front_m
        self.positions = iter(positions)
        self.last_pose = positions[0]
        self.safety_calls: list[dict[str, object]] = []

    def safety_error(self, **kwargs):
        self.safety_calls.append(kwargs)
        return None

    def poll(self):
        return SimpleNamespace(
            front_ultrasound_m=self.front_m,
            ultrasound_updated_at=time.monotonic(),
        )

    def filtered_front_ultrasound_m(self, _window_s: float):
        return self.front_m

    def pose(self):
        try:
            self.last_pose = next(self.positions)
        except StopIteration:
            pass
        return self.last_pose


class RecordingPlacementMotion:
    def __init__(self) -> None:
        self.commands: list[tuple[float, float, float]] = []
        self.strafes: list[tuple[float, float]] = []
        self.forward_distances: list[tuple[float, float]] = []
        self.autonomous_count = 0

    def set_autonomous(self) -> None:
        self.autonomous_count += 1

    def move(self, vx: float, vy: float, wz: float) -> None:
        self.commands.append((vx, vy, wz))

    def stop(self) -> None:
        self.commands.append((0.0, 0.0, 0.0))

    def strafe_distance(self, distance_m: float, *, speed_mps: float) -> None:
        self.strafes.append((distance_m, speed_mps))

    def go_distance(self, distance_m: float, *, speed_mps: float) -> None:
        self.forward_distances.append((distance_m, speed_mps))


class PlacementCamera:
    def __init__(self, frames) -> None:
        self.frames = iter(frames)
        self.read_timeout_ms = 2000
        self.open_timeout_ms = 3000
        self.last_frame_at = None
        self.release_count = 0
        self.read_count = 0

    def read(self):
        self.read_count += 1
        frame = next(self.frames)
        self.last_frame_at = time.monotonic()
        return frame

    def release(self) -> None:
        self.release_count += 1


class BoxCenterMissionTest(unittest.TestCase):
    def test_final_approach_retry_commands_only_unfinished_distance(self) -> None:
        mission = LargeQuadrupedMission(
            load_config(),
            dry_run=True,
            skip_arm=True,
        )
        mission.context.dry_run = False
        action = NavigationAction(
            ActionKind.FINAL_APPROACH,
            "test_final_approach",
            distance_m=0.30,
            vx_mps=0.08,
        )
        first_motion = RecordingPlacementMotion()
        first_motion.go_distance = mock.Mock(
            side_effect=RuntimeError("temporary drive failure")
        )
        mission.motion = first_motion
        mission.state_reader = FreshPlacementStateReader(
            positions=((0.00, 0.0, 0.0), (0.18, 0.0, 0.0))
        )
        first_navigator = PlacementLetterNavigator(
            "C",
            mission._placement_letter_navigation_config(),
        )

        with self.assertRaisesRegex(MissionAbort, "temporary drive failure"):
            mission._execute_placement_final_approach(
                action,
                navigator=first_navigator,
            )

        self.assertAlmostEqual(
            mission.context.placement_final_approach_progress_m,
            0.18,
        )
        self.assertFalse(mission.context.placement_final_approach_complete)

        second_motion = RecordingPlacementMotion()
        mission.motion = second_motion
        mission.state_reader = FreshPlacementStateReader(
            positions=((0.18, 0.0, 0.0), (0.30, 0.0, 0.0))
        )
        second_navigator = PlacementLetterNavigator(
            "C",
            mission._placement_letter_navigation_config(),
        )

        measured = mission._execute_placement_final_approach(
            action,
            navigator=second_navigator,
        )

        self.assertAlmostEqual(measured, 0.30)
        self.assertEqual(len(second_motion.forward_distances), 1)
        self.assertAlmostEqual(second_motion.forward_distances[0][0], 0.12)
        self.assertTrue(mission.context.placement_final_approach_complete)
        self.assertFalse(mission._controlled_box_approach_active)

    def test_failed_placement_strafe_records_partial_odometry_for_rollback(
        self,
    ) -> None:
        mission = LargeQuadrupedMission(
            load_config(),
            dry_run=True,
            skip_arm=True,
        )
        mission.context.dry_run = False
        motion = RecordingPlacementMotion()
        motion.strafe_distance = mock.Mock(
            side_effect=RuntimeError("temporary strafe failure")
        )
        mission.motion = motion
        mission.state_reader = FreshPlacementStateReader(
            positions=((0.00, 0.00, 0.0), (0.00, 0.04, 0.0))
        )
        navigator = PlacementLetterNavigator(
            "C",
            mission._placement_letter_navigation_config(),
        )
        action = NavigationAction(
            ActionKind.STRAFE,
            "test_partial_strafe",
            distance_m=0.10,
            vy_mps=0.08,
        )

        with self.assertRaisesRegex(
            MissionAbort,
            "failed after moving 0.040m",
        ):
            mission._execute_placement_navigation_motion(
                action,
                navigator=navigator,
            )

        self.assertAlmostEqual(navigator.net_lateral_m, 0.04)
        self.assertAlmostEqual(navigator.lateral_travel_m, 0.04)

    def test_camera_latest_reader_timeout_stops_before_retry(self) -> None:
        config = load_config()
        config["placement_letter_navigation"] = dict(
            config["placement_letter_navigation"]
        )
        config["placement_letter_navigation"]["image_timeout_s"] = 0.02
        mission = LargeQuadrupedMission(config, dry_run=True, skip_arm=True)
        mission.context.dry_run = False
        events: list[str] = []

        class TimeoutThenFreshCamera(PlacementCamera):
            def read_latest(self, timeout_s=None):
                self.read_count += 1
                if self.read_count == 1:
                    events.append("timeout_read")
                    time.sleep(float(timeout_s))
                    return None
                events.append("retry_read")
                self.last_frame_at = time.monotonic()
                return np.ones((720, 1000, 3), dtype=np.uint8)

        camera = TimeoutThenFreshCamera(())
        mission.wide_camera = camera
        mission._placement_undistorter = SimpleNamespace(apply=lambda value: value)
        mission._detect_placement_letters = mock.Mock(
            return_value=placement_detection(None)
        )
        mission.state_reader = FreshPlacementStateReader()
        mission.motion = mock.Mock()

        def stop() -> None:
            events.append("stop")

        mission.motion.stop.side_effect = stop

        observation, _ = mission._capture_placement_navigation_frame(
            target_letter="D",
            frame_sequence=0,
            started_at=time.monotonic(),
        )

        self.assertEqual(observation.frame_sequence, 1)
        self.assertEqual(camera.read_count, 2)
        self.assertLess(events.index("stop"), events.index("retry_read"))

    def test_camera_latest_reader_exhausts_bounded_retries(self) -> None:
        config = load_config()
        config["placement_letter_navigation"] = dict(
            config["placement_letter_navigation"]
        )
        config["placement_letter_navigation"]["image_timeout_s"] = 0.02
        mission = LargeQuadrupedMission(config, dry_run=True, skip_arm=True)
        mission.context.dry_run = False

        class TimeoutCamera(PlacementCamera):
            def read_latest(self, timeout_s=None):
                self.read_count += 1
                time.sleep(float(timeout_s))
                return None

        camera = TimeoutCamera(())
        mission.wide_camera = camera
        mission.motion = mock.Mock()
        started_at = time.monotonic()
        with self.assertRaisesRegex(MissionAbort, "placement camera timeout"):
            mission._capture_placement_navigation_frame(
                target_letter="C",
                frame_sequence=0,
                started_at=started_at,
            )
        self.assertLess(time.monotonic() - started_at, 0.20)
        self.assertEqual(
            camera.read_count,
            int(config["placement_letter_navigation"]["capture_retries"]),
        )
        self.assertEqual(camera.release_count, 0)

    def test_keyboard_interrupt_from_latest_reader_is_propagated(self) -> None:
        mission = LargeQuadrupedMission(load_config(), dry_run=True, skip_arm=True)
        mission.context.dry_run = False

        class InterruptingCamera(PlacementCamera):
            def read_latest(self, timeout_s=None):
                raise KeyboardInterrupt("operator interrupt")

        mission.wide_camera = InterruptingCamera(())
        mission.motion = mock.Mock()
        with self.assertRaisesRegex(KeyboardInterrupt, "operator interrupt"):
            mission._capture_placement_navigation_frame(
                target_letter="C",
                frame_sequence=0,
                started_at=time.monotonic(),
            )

    def test_final_cleanup_releases_all_camera_views(self) -> None:
        mission = LargeQuadrupedMission(load_config(), dry_run=True, skip_arm=True)
        mission.motion = mock.Mock()
        mission.front_camera = mock.Mock()
        mission.wide_camera = mock.Mock()
        mission.arm_camera = mock.Mock()
        mission.arm = mock.Mock()
        mission.state_reader = mock.Mock()

        errors = mission._cleanup()
        self.assertEqual(errors, [])
        mission.front_camera.release.assert_called_once_with()
        mission.arm_camera.release.assert_called_once_with()
        mission.wide_camera.release.assert_called_once_with()

    def test_placement_navigation_config_maps_search_step_to_both_axes(self) -> None:
        config = load_config()
        config["placement_letter_navigation"] = dict(
            config["placement_letter_navigation"]
        )
        config["placement_letter_navigation"]["search_step_m"] = 0.17
        mission = LargeQuadrupedMission(config, dry_run=True, skip_arm=True)

        navigation = mission._placement_letter_navigation_config()

        self.assertAlmostEqual(navigation.forward_step_m, 0.17)
        self.assertAlmostEqual(navigation.lateral_search_step_m, 0.17)
        self.assertAlmostEqual(navigation.image_timeout_s, 0.50)
        self.assertEqual(navigation.capture_retries, 3)

    def test_placement_navigation_maps_physical_left_and_right_to_configured_strafe_sign(self) -> None:
        config = load_config()
        config["placement_letter_navigation"] = dict(
            config["placement_letter_navigation"]
        )
        config["placement_letter_navigation"]["physical_left_strafe_sign"] = -1
        mission = LargeQuadrupedMission(config, dry_run=True, skip_arm=True)
        mission.motion = mock.Mock()

        mission._execute_placement_navigation_motion(
            NavigationAction(
                ActionKind.STRAFE,
                "left",
                distance_m=0.10,
                vy_mps=0.08,
            )
        )
        mission._execute_placement_navigation_motion(
            NavigationAction(
                ActionKind.STRAFE,
                "right",
                distance_m=-0.10,
                vy_mps=-0.08,
            )
        )

        self.assertEqual(
            [call.args[0] for call in mission.motion.strafe_distance.call_args_list],
            [-0.10, 0.10],
        )
        for call in mission.motion.strafe_distance.call_args_list:
            self.assertEqual(call.kwargs["speed_mps"], 0.08)
        mission.motion.go_distance.assert_not_called()

    def test_forced_approach_ignores_visible_letter_until_step_distance(self) -> None:
        mission = LargeQuadrupedMission(load_config(), dry_run=True, skip_arm=True)
        mission.context.dry_run = False
        frames = [
            np.zeros((720, 1000, 3), dtype=np.uint8),
            np.ones((720, 1000, 3), dtype=np.uint8),
            np.full((720, 1000, 3), 2, dtype=np.uint8),
        ]
        mission.wide_camera = PlacementCamera(frames)
        mission._placement_undistorter = SimpleNamespace(apply=lambda frame: frame)
        mission._detect_placement_letters = mock.Mock(
            side_effect=[
                placement_detection(None),
                placement_detection("D"),
                placement_detection("D"),
            ]
        )
        mission.state_reader = FreshPlacementStateReader(
            front_m=1.20,
            positions=(
                (0.00, 0.00, 0.0),
                (0.00, 0.00, 0.0),
                (0.03, 0.00, 0.0),
                (0.10, 0.00, 0.0),
            ),
        )
        mission.motion = RecordingPlacementMotion()
        navigator = PlacementLetterNavigator(
            "D",
            mission._placement_letter_navigation_config(),
        )
        action = NavigationAction(
            ActionKind.FORWARD,
            "search",
            distance_m=0.10,
            vx_mps=0.08,
        )

        observation = mission._run_placement_forward_search_step(
            target_letter="D",
            frame_sequence=0,
            maximum_distance_m=0.10,
            started_at=time.monotonic(),
            action=action,
            navigator=navigator,
        )

        self.assertEqual(observation.candidates[0].letter, "D")
        self.assertEqual(mission.motion.commands[-1], (0.0, 0.0, 0.0))
        self.assertTrue(
            all(vy == 0.0 and wz == 0.0 for _, vy, wz in mission.motion.commands)
        )
        self.assertEqual(mission.motion.strafes, [])
        self.assertEqual(mission.wide_camera.read_count, 3)
        self.assertAlmostEqual(navigator.forward_travel_m, 0.10)

    def test_lateral_action_records_odometry_in_physical_left_coordinates(self) -> None:
        config = load_config()
        config["placement_letter_navigation"] = dict(
            config["placement_letter_navigation"]
        )
        config["placement_letter_navigation"]["physical_left_strafe_sign"] = -1
        mission = LargeQuadrupedMission(config, dry_run=True, skip_arm=True)
        mission.context.dry_run = False
        mission.motion = RecordingPlacementMotion()
        mission.state_reader = FreshPlacementStateReader(
            positions=(
                (2.0, 3.00, 0.0),
                (2.0, 2.92, 0.0),
                (2.0, 2.92, 0.0),
                (2.0, 3.00, 0.0),
            )
        )
        navigator = PlacementLetterNavigator(
            "D",
            mission._placement_letter_navigation_config(),
        )
        action = NavigationAction(
            ActionKind.STRAFE,
            "left",
            distance_m=0.10,
            vy_mps=0.08,
        )

        measured_left = mission._execute_placement_navigation_motion(
            action,
            navigator=navigator,
        )
        measured_right = mission._execute_placement_navigation_motion(
            NavigationAction(
                ActionKind.STRAFE,
                "right",
                distance_m=-0.10,
                vy_mps=-0.08,
            ),
            navigator=navigator,
        )

        self.assertEqual(
            mission.motion.strafes,
            [(-0.10, 0.08), (0.10, 0.08)],
        )
        self.assertEqual(mission.motion.forward_distances, [])
        self.assertAlmostEqual(measured_left, 0.08)
        self.assertAlmostEqual(measured_right, -0.08)
        self.assertAlmostEqual(navigator.lateral_travel_m, 0.16)
        self.assertAlmostEqual(navigator.net_lateral_m, 0.0)

    def test_placement_continuously_approaches_until_ultrasound_reaches_28cm(
        self,
    ) -> None:
        mission = LargeQuadrupedMission(load_config(), dry_run=True, skip_arm=True)
        mission.context.dry_run = False
        mission.motion = RecordingPlacementMotion()
        mission.state_reader = mock.Mock()
        mission.state_reader.pose.side_effect = (
            (0.00, 0.0, 0.0),
            (0.00, 0.0, 0.0),
            (0.40, 0.0, 0.0),
            (1.05, 0.0, 0.0),
            (1.08, 0.0, 0.0),
            (1.09, 0.0, 0.0),
            (1.09, 0.0, 0.0),
        )
        mission._prime_placement_front_filter = mock.Mock()
        mission._placement_label_row_distance_m = mock.Mock(return_value=1.35)
        mission._placement_front_distance = mock.Mock(
            side_effect=(
                (1.40, {"front_distance_m": 1.40}),
                (1.00, {"front_distance_m": 1.00}),
                (0.28, {"front_distance_m": 0.28}),
                (0.28, {"front_distance_m": 0.28}),
                (0.28, {"front_distance_m": 0.28}),
            )
        )
        mission._append_placement_navigation_event = mock.Mock()
        navigator = PlacementLetterNavigator(
            "C",
            mission._placement_letter_navigation_config(),
        )

        with mock.patch("mission_lite3.mission.time.sleep"):
            mission._run_placement_ultrasound_approach(
                target_letter="C",
                navigator=navigator,
                started_at=time.monotonic(),
            )

        self.assertEqual(
            [command for command in mission.motion.commands if command[0] > 0.0],
            [(0.08, 0.0, 0.0), (0.08, 0.0, 0.0)],
        )
        self.assertAlmostEqual(navigator.forward_travel_m, 1.09)
        self.assertEqual(mission._placement_front_distance.call_count, 5)

    def test_placement_echo_loss_uses_odometry_hard_stop_without_collision(self) -> None:
        mission = LargeQuadrupedMission(load_config(), dry_run=True, skip_arm=True)
        mission.context.dry_run = False
        mission.motion = RecordingPlacementMotion()
        mission.state_reader = mock.Mock()
        mission.state_reader.pose.side_effect = (
            (0.00, 0.0, 0.0),
            (0.00, 0.0, 0.0),
            (0.40, 0.0, 0.0),
            (1.00, 0.0, 0.0),
            (1.68, 0.0, 0.0),
            (1.68, 0.0, 0.0),
        )
        mission._prime_placement_front_filter = mock.Mock()
        mission._placement_label_row_distance_m = mock.Mock(return_value=1.888)
        mission._placement_front_distance = mock.Mock(
            side_effect=(
                (1.974, {"front_distance_m": 1.974}),
                (1.60, {"front_distance_m": 1.60}),
                (4.50, {"front_distance_m": 4.50}),
                (4.50, {"front_distance_m": 4.50}),
            )
        )
        mission._append_placement_navigation_event = mock.Mock()
        navigator = PlacementLetterNavigator(
            "D",
            mission._placement_letter_navigation_config(),
        )

        with mock.patch("mission_lite3.mission.time.sleep"):
            mission._run_placement_ultrasound_approach(
                target_letter="D",
                navigator=navigator,
                started_at=time.monotonic(),
            )

        positive_commands = [
            command for command in mission.motion.commands if command[0] > 0.0
        ]
        self.assertEqual(
            positive_commands,
            [
                (0.08, 0.0, 0.0),
                (0.08, 0.0, 0.0),
                (0.03, 0.0, 0.0),
            ],
        )
        self.assertNotIn((0.06, 0.0, 0.0), mission.motion.commands)
        self.assertAlmostEqual(navigator.forward_travel_m, 1.68)
        event = mission._append_placement_navigation_event.call_args.args[0]
        self.assertTrue(event["echo_loss_fallback"])
        self.assertAlmostEqual(event["odometry_stop_distance_m"], 1.674)

    def test_frozen_pickup_ultrasound_uses_visual_odometry_hard_stop(self) -> None:
        mission = LargeQuadrupedMission(load_config(), dry_run=True, skip_arm=True)
        mission.context.dry_run = False
        mission.motion = RecordingPlacementMotion()
        mission.state_reader = mock.Mock()
        mission.state_reader.safety_error.return_value = None
        mission.state_reader.state.front_ultrasound_m = 0.28
        mission.state_reader.pose.side_effect = (
            (0.00, 0.0, 0.0),
            (0.00, 0.0, 0.0),
            (0.50, 0.0, 0.0),
            (1.20, 0.0, 0.0),
            (1.46, 0.0, 0.0),
            (1.46, 0.0, 0.0),
        )
        unguarded_move = mission.motion.move

        def guarded_move(vx: float, vy: float, wz: float) -> None:
            mission._motion_guard(vx, vy, wz)
            unguarded_move(vx, vy, wz)

        mission.motion.move = mock.Mock(side_effect=guarded_move)
        mission._prime_placement_front_filter = mock.Mock()
        mission._placement_label_row_distance_m = mock.Mock(return_value=1.758)
        mission._placement_front_distance = mock.Mock(
            side_effect=(
                (0.28, {"front_distance_m": 0.28}),
                (0.28, {"front_distance_m": 0.28}),
                (0.28, {"front_distance_m": 0.28}),
                (0.28, {"front_distance_m": 0.28}),
            )
        )
        mission._append_placement_navigation_event = mock.Mock()
        navigator = PlacementLetterNavigator(
            "D",
            mission._placement_letter_navigation_config(),
        )

        with mock.patch("mission_lite3.mission.time.sleep"):
            mission._run_placement_ultrasound_approach(
                target_letter="D",
                navigator=navigator,
                started_at=time.monotonic(),
            )

        positive_commands = [
            command for command in mission.motion.commands if command[0] > 0.0
        ]
        self.assertEqual(
            positive_commands,
            [
                (0.05, 0.0, 0.0),
                (0.05, 0.0, 0.0),
                (0.05, 0.0, 0.0),
            ],
        )
        self.assertNotIn((0.08, 0.0, 0.0), mission.motion.commands)
        self.assertNotIn((0.06, 0.0, 0.0), mission.motion.commands)
        self.assertAlmostEqual(navigator.forward_travel_m, 1.46)
        event = mission._append_placement_navigation_event.call_args.args[0]
        self.assertTrue(event["ultrasound_stuck_fallback"])
        self.assertFalse(event["echo_loss_fallback"])
        self.assertAlmostEqual(event["initial_ultrasound_m"], 0.28)
        self.assertAlmostEqual(event["initial_visual_distance_m"], 1.758)
        self.assertAlmostEqual(event["odometry_stop_distance_m"], 1.458)
        self.assertFalse(mission._controlled_box_approach_active)

    def test_post_turn_forward_is_fixed_1p4m_and_ignores_ultrasound(self) -> None:
        mission = LargeQuadrupedMission(load_config(), dry_run=True, skip_arm=True)
        mission.context.dry_run = False
        mission.motion = RecordingPlacementMotion()
        mission.state_reader = mock.Mock()
        mission.state_reader.pose.side_effect = (
            (0.00, 0.0, 0.0),
            (1.37, 0.0, 0.0),
        )
        mission._append_placement_navigation_event = mock.Mock()
        navigator = PlacementLetterNavigator(
            "D",
            mission._placement_letter_navigation_config(),
        )

        with mock.patch(
            "mission_lite3.mission.time.monotonic",
            side_effect=(0.0, 17.5),
        ), mock.patch("mission_lite3.mission.time.sleep"):
            mission._run_forced_placement_forward(
                target_letter="D",
                navigator=navigator,
                started_at=99.0,
            )

        self.assertEqual(mission.motion.autonomous_count, 1)
        self.assertEqual(
            [command for command in mission.motion.commands if command[0] > 0.0],
            [(0.08, 0.0, 0.0)],
        )
        self.assertEqual(mission.motion.forward_distances, [])
        self.assertAlmostEqual(
            mission.context.placement_forced_forward_progress_m,
            1.40,
        )
        self.assertAlmostEqual(navigator.forward_travel_m, 1.40)
        self.assertAlmostEqual(mission.context.first_outbound_forward_m, 1.37)
        self.assertAlmostEqual(
            mission.context.placement_forced_forward_odom_m,
            1.37,
        )
        self.assertFalse(mission.ignore_ultrasound_obstacle)
        self.assertFalse(mission._controlled_box_approach_active)
        event = mission._append_placement_navigation_event.call_args.args[0]
        self.assertTrue(event["ultrasound_ignored"])
        self.assertTrue(event["odometry_stop_ignored"])
        self.assertTrue(event["odometry_recorded"])
        self.assertAlmostEqual(event["target_distance_m"], 1.40)
        self.assertAlmostEqual(event["command_duration_s"], 17.5)

    def test_post_turn_forward_retry_only_completes_remaining_distance(self) -> None:
        mission = LargeQuadrupedMission(load_config(), dry_run=True, skip_arm=True)
        mission.context.dry_run = False
        mission.motion = RecordingPlacementMotion()
        mission.motion.move = mock.Mock(
            side_effect=RuntimeError("temporary drive failure")
        )
        mission.state_reader = mock.Mock()
        mission.state_reader.pose.side_effect = (
            (0.00, 0.0, 0.0),
            (0.58, 0.0, 0.0),
        )
        mission._append_placement_navigation_event = mock.Mock()
        navigator = PlacementLetterNavigator(
            "D",
            mission._placement_letter_navigation_config(),
        )

        with self.assertRaisesRegex(
            MissionAbort,
            "0.600/1.400m",
        ), mock.patch(
            "mission_lite3.mission.time.monotonic",
            side_effect=(0.0, 7.5),
        ):
            mission._run_forced_placement_forward(
                target_letter="D",
                navigator=navigator,
                started_at=99.0,
            )

        self.assertAlmostEqual(
            mission.context.placement_forced_forward_progress_m,
            0.60,
        )
        second_motion = RecordingPlacementMotion()
        mission.motion = second_motion
        mission.state_reader.pose.side_effect = (
            (0.58, 0.0, 0.0),
            (1.38, 0.0, 0.0),
        )

        with mock.patch(
            "mission_lite3.mission.time.monotonic",
            side_effect=(0.0, 10.0),
        ), mock.patch("mission_lite3.mission.time.sleep"):
            mission._run_forced_placement_forward(
                target_letter="D",
                navigator=navigator,
                started_at=99.0,
            )

        self.assertEqual(second_motion.autonomous_count, 1)
        self.assertEqual(
            [command for command in second_motion.commands if command[0] > 0.0],
            [(0.08, 0.0, 0.0)],
        )
        self.assertEqual(second_motion.forward_distances, [])
        self.assertAlmostEqual(
            mission.context.placement_forced_forward_progress_m,
            1.40,
        )
        self.assertAlmostEqual(navigator.forward_travel_m, 1.40)
        self.assertAlmostEqual(mission.context.first_outbound_forward_m, 1.38)

    def test_pickup_return_reuses_first_outbound_forward_odometry(self) -> None:
        mission = LargeQuadrupedMission(load_config(), dry_run=True, skip_arm=True)
        mission.context.dry_run = False
        mission.context.first_outbound_forward_m = 1.37
        mission.motion = RecordingPlacementMotion()
        mission.state_reader = mock.Mock()
        mission.state_reader.pose.side_effect = (
            (0.00, 0.0, 0.0),
            (0.00, 0.0, 0.0),
            (1.37, 0.0, 0.0),
        )

        with mock.patch(
            "mission_lite3.mission.time.monotonic",
            side_effect=(0.0, 1.0),
        ), mock.patch("mission_lite3.mission.time.sleep"):
            mission._execute_pickup_forward_restore()

        self.assertEqual(mission.motion.autonomous_count, 1)
        self.assertEqual(
            [command for command in mission.motion.commands if command[0] > 0.0],
            [(0.08, 0.0, 0.0)],
        )
        self.assertEqual(mission.motion.forward_distances, [])

    def test_placement_odometry_stall_never_commands_forward_recovery(self) -> None:
        config = load_config()
        config["placement_letter_navigation"]["motion_stall_timeout_s"] = 0.0
        mission = LargeQuadrupedMission(config, dry_run=True, skip_arm=True)
        mission.context.dry_run = False
        mission.motion = RecordingPlacementMotion()
        mission.state_reader = mock.Mock()
        mission.state_reader.pose.return_value = (0.00, 0.0, 0.0)
        mission._prime_placement_front_filter = mock.Mock()
        mission._placement_label_row_distance_m = mock.Mock(return_value=1.90)
        mission._placement_front_distance = mock.Mock(
            return_value=(1.974, {"front_distance_m": 1.974})
        )
        mission._append_placement_navigation_event = mock.Mock()
        navigator = PlacementLetterNavigator(
            "D",
            mission._placement_letter_navigation_config(),
        )

        with self.assertRaisesRegex(MissionAbort, "forward retry disabled"):
            mission._run_placement_ultrasound_approach(
                target_letter="D",
                navigator=navigator,
                started_at=time.monotonic(),
            )

        self.assertFalse(
            any(command[0] > 0.0 for command in mission.motion.commands)
        )

    def test_forward_records_final_pose_after_stop(self) -> None:
        mission = LargeQuadrupedMission(load_config(), dry_run=True, skip_arm=True)
        mission.context.dry_run = False
        mission.wide_camera = PlacementCamera(
            (
                np.zeros((720, 1000, 3), dtype=np.uint8),
                np.ones((720, 1000, 3), dtype=np.uint8),
            )
        )
        mission._placement_undistorter = SimpleNamespace(apply=lambda value: value)
        mission._detect_placement_letters = mock.Mock(
            return_value=placement_detection("D")
        )
        mission.state_reader = FreshPlacementStateReader(
            front_m=1.20,
            positions=(
                (0.00, 0.0, 0.0),
                (0.08, 0.0, 0.0),
                (0.10, 0.0, 0.0),
            )
        )
        mission.motion = RecordingPlacementMotion()
        navigator = PlacementLetterNavigator(
            "D",
            mission._placement_letter_navigation_config(),
        )

        mission._run_placement_forward_search_step(
            target_letter="D",
            frame_sequence=0,
            maximum_distance_m=0.10,
            started_at=time.monotonic(),
            action=NavigationAction(
                ActionKind.FORWARD,
                "search",
                distance_m=0.10,
                vx_mps=0.08,
            ),
            navigator=navigator,
        )

        self.assertAlmostEqual(navigator.forward_travel_m, 0.10)

    def test_forward_primary_error_is_preserved_after_final_pose_recording(self) -> None:
        mission = LargeQuadrupedMission(load_config(), dry_run=True, skip_arm=True)
        mission.context.dry_run = False
        mission.wide_camera = PlacementCamera(
            (np.zeros((720, 1000, 3), dtype=np.uint8),)
        )
        mission._placement_undistorter = SimpleNamespace(apply=lambda value: value)
        mission._detect_placement_letters = mock.Mock(
            return_value=placement_detection(None)
        )
        mission.state_reader = FreshPlacementStateReader(
            front_m=1.20,
            positions=(
                (0.00, 0.0, 0.0),
                (0.08, 0.0, 0.0),
                (0.10, 0.0, 0.0),
            )
        )
        mission.motion = RecordingPlacementMotion()
        mission.motion.move = mock.Mock(side_effect=MissionAbort("primary motion"))
        navigator = PlacementLetterNavigator(
            "D",
            mission._placement_letter_navigation_config(),
        )

        with self.assertRaisesRegex(MissionAbort, "primary motion"):
            mission._run_placement_forward_search_step(
                target_letter="D",
                frame_sequence=0,
                maximum_distance_m=0.10,
                started_at=time.monotonic(),
                action=NavigationAction(
                    ActionKind.FORWARD,
                    "search",
                    distance_m=0.10,
                    vx_mps=0.08,
                ),
                navigator=navigator,
            )

        self.assertAlmostEqual(navigator.forward_travel_m, 0.10)

    def test_forward_search_counts_backward_odometry_as_progress(self) -> None:
        mission = LargeQuadrupedMission(load_config(), dry_run=True, skip_arm=True)
        mission.context.dry_run = False
        mission.wide_camera = PlacementCamera(
            (
                np.zeros((720, 1000, 3), dtype=np.uint8),
                np.ones((720, 1000, 3), dtype=np.uint8),
            )
        )
        mission._placement_undistorter = SimpleNamespace(apply=lambda value: value)
        mission._detect_placement_letters = mock.Mock(
            return_value=placement_detection("D")
        )
        mission.state_reader = FreshPlacementStateReader(
            front_m=1.20,
            positions=(
                (0.0, 0.0, 0.0),
                (-0.02, 0.0, 0.0),
                (-0.10, 0.0, 0.0),
            ),
        )
        mission.motion = RecordingPlacementMotion()
        navigator = PlacementLetterNavigator(
            "D",
            mission._placement_letter_navigation_config(),
        )
        action = NavigationAction(
            ActionKind.FORWARD,
            "search",
            distance_m=0.10,
            vx_mps=0.08,
        )

        mission._run_placement_forward_search_step(
            target_letter="D",
            frame_sequence=0,
            maximum_distance_m=0.10,
            started_at=time.monotonic(),
            action=action,
            navigator=navigator,
        )

        self.assertEqual(
            mission.motion.commands.count((0.08, 0.0, 0.0)),
            1,
        )
        self.assertEqual(mission.motion.commands[-1], (0.0, 0.0, 0.0))
        self.assertAlmostEqual(navigator.forward_travel_m, 0.10)

    def test_forward_search_does_not_abort_on_precommand_negative_odometry(self) -> None:
        mission = LargeQuadrupedMission(load_config(), dry_run=True, skip_arm=True)
        mission.context.dry_run = False
        mission.wide_camera = PlacementCamera(
            (
                np.zeros((720, 1000, 3), dtype=np.uint8),
                np.ones((720, 1000, 3), dtype=np.uint8),
            )
        )
        mission._placement_undistorter = SimpleNamespace(apply=lambda value: value)
        mission._detect_placement_letters = mock.Mock(
            return_value=placement_detection("D")
        )
        mission.state_reader = FreshPlacementStateReader(
            front_m=1.20,
            positions=(
                (0.0, 0.0, 0.0),
                (-0.02, 0.0, 0.0),
                (0.10, 0.0, 0.0),
            ),
        )
        mission.motion = RecordingPlacementMotion()
        navigator = PlacementLetterNavigator(
            "D",
            mission._placement_letter_navigation_config(),
        )

        mission._run_placement_forward_search_step(
            target_letter="D",
            frame_sequence=0,
            maximum_distance_m=0.10,
            started_at=time.monotonic(),
            action=NavigationAction(
                ActionKind.FORWARD,
                "search",
                distance_m=0.10,
                vx_mps=0.08,
            ),
            navigator=navigator,
        )

        self.assertIn((0.08, 0.0, 0.0), mission.motion.commands)
        self.assertEqual(mission.motion.commands[-1], (0.0, 0.0, 0.0))
        self.assertAlmostEqual(navigator.forward_travel_m, 0.10)

    def test_placement_navigation_camera_failure_stops_and_aborts_before_arm(self) -> None:
        mission = LargeQuadrupedMission(load_config(), dry_run=True, skip_arm=True)
        mission.context.dry_run = False
        mission.context.carried_bar = True
        mission.context.target_letter = "C"
        mission.motion = mock.Mock()
        mission._run_placement_letter_navigator = mock.Mock(
            side_effect=TimeoutError("placement camera timeout")
        )
        mission.arm = mock.Mock()

        with self.assertRaisesRegex(MissionAbort, "placement camera timeout"):
            mission._execute_placement_letter_approach()

        mission.motion.stop.assert_called()
        mission.arm.place_to_box.assert_not_called()

    def test_primary_navigation_error_wins_over_stop_error(self) -> None:
        mission = LargeQuadrupedMission(load_config(), dry_run=True, skip_arm=True)
        mission.context.target_letter = "C"
        mission._run_placement_letter_navigator = mock.Mock(
            side_effect=MissionAbort("primary camera error")
        )
        mission.motion = mock.Mock()
        mission.motion.stop.side_effect = RuntimeError("stop failed")

        with self.assertRaisesRegex(MissionAbort, "primary camera error"):
            mission._execute_placement_letter_approach()

    def test_terminal_log_failure_never_overrides_primary_exception(self) -> None:
        for primary in (
            MissionAbort("primary mission abort"),
            KeyboardInterrupt("primary interrupt"),
        ):
            with self.subTest(primary=type(primary).__name__):
                with tempfile.TemporaryDirectory() as temporary:
                    config = load_config()
                    config["placement_letter_navigation"] = dict(
                        config["placement_letter_navigation"]
                    )
                    config["placement_letter_navigation"]["run_log_dir"] = temporary
                    mission = LargeQuadrupedMission(
                        config,
                        dry_run=True,
                        skip_arm=True,
                    )
                    mission.context.dry_run = False
                    mission.wide_camera = PlacementCamera(())
                    mission.front_camera = mock.Mock()
                    mission.motion = mock.Mock()
                    mission._run_placement_letter_navigation_loop = mock.Mock(
                        side_effect=primary
                    )
                    mission._append_placement_navigation_event = mock.Mock(
                        side_effect=OSError("terminal json failed")
                    )

                    with self.assertRaisesRegex(type(primary), str(primary)):
                        mission._run_placement_letter_navigator("C")

                    mission._append_placement_navigation_event.assert_called_once()

    def test_evidence_failure_without_primary_is_converted_to_mission_abort(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = load_config()
            config["placement_letter_navigation"] = dict(
                config["placement_letter_navigation"]
            )
            config["placement_letter_navigation"]["run_log_dir"] = temporary
            mission = LargeQuadrupedMission(config, dry_run=True, skip_arm=True)
            mission.context.dry_run = False
            mission.wide_camera = PlacementCamera(())
            mission.front_camera = mock.Mock()
            mission.motion = mock.Mock()
            mission._run_placement_letter_navigation_loop = mock.Mock(
                side_effect=OSError("frame json failed")
            )
            mission._append_placement_navigation_event = mock.Mock(
                side_effect=OSError("terminal json failed")
            )

            with self.assertRaisesRegex(MissionAbort, "frame json failed"):
                mission._run_placement_letter_navigator("C")

    def test_navigation_reuses_camera_without_release_or_timeout_mutation(self) -> None:
        mission = LargeQuadrupedMission(load_config(), dry_run=True, skip_arm=True)
        mission.context.dry_run = False
        camera = PlacementCamera(())
        camera.release = mock.Mock()
        mission.wide_camera = camera
        mission.motion = mock.Mock()
        mission._create_placement_navigation_run_dir = mock.Mock(
            return_value=Path(".")
        )
        mission._run_placement_letter_navigation_loop = mock.Mock(return_value=0.0)

        self.assertEqual(mission._run_placement_letter_navigator("C"), 0.0)
        camera.release.assert_not_called()
        self.assertEqual(camera.read_timeout_ms, 2000)
        self.assertEqual(camera.open_timeout_ms, 3000)

    def test_first_placement_records_measured_net_lateral_but_second_does_not_overwrite(self) -> None:
        mission = LargeQuadrupedMission(load_config(), dry_run=True, skip_arm=True)
        mission.context.target_letter = "A"
        mission._run_placement_letter_navigator = mock.Mock(return_value=0.42)
        mission._execute_placement_letter_approach()
        self.assertAlmostEqual(mission.context.first_outbound_lane_strafe_m, 0.42)
        self.assertAlmostEqual(mission.context.placement_letter_lateral_m["A"], 0.42)

        mission.context.placed_letters = ["A"]
        mission.context.target_letter = "D"
        mission._run_placement_letter_navigator.return_value = -0.31
        mission._execute_placement_letter_approach()
        self.assertAlmostEqual(mission.context.first_outbound_lane_strafe_m, 0.42)
        self.assertAlmostEqual(mission.context.placement_letter_lateral_m["D"], -0.31)
        self.assertEqual(mission._run_placement_letter_navigator.call_count, 2)

    def test_cached_letter_geometry_predicts_next_target_lane(self) -> None:
        mission = LargeQuadrupedMission(load_config(), dry_run=True, skip_arm=True)
        mission.context.placement_letter_lateral_m["C"] = 0.0

        self.assertAlmostEqual(mission._cached_placement_target_lateral_m("A"), 1.0)
        self.assertAlmostEqual(mission._cached_placement_target_lateral_m("B"), 0.5)
        self.assertAlmostEqual(mission._cached_placement_target_lateral_m("C"), 0.0)
        self.assertAlmostEqual(mission._cached_placement_target_lateral_m("D"), -0.5)

    def test_dry_run_uses_two_stage_confirmation_without_camera_or_state(self) -> None:
        mission = LargeQuadrupedMission(load_config(), dry_run=True, skip_arm=True)
        mission.wide_camera = mock.Mock()
        mission.state_reader = mock.Mock()
        mission.motion = mock.Mock()

        measured = mission._run_placement_letter_navigator("D")

        self.assertEqual(measured, 0.0)
        mission.wide_camera.read.assert_not_called()
        mission.state_reader.poll.assert_not_called()
        mission.state_reader.pose.assert_not_called()
        mission.motion.stop.assert_called_once_with()

    def test_capture_retries_and_converts_only_real_ocr_candidates(self) -> None:
        mission = LargeQuadrupedMission(load_config(), dry_run=True, skip_arm=True)
        mission.context.dry_run = False
        frame = np.zeros((720, 1000, 3), dtype=np.uint8)
        mission.wide_camera = PlacementCamera((None, None, frame))
        mission._placement_undistorter = SimpleNamespace(apply=lambda value: value)
        mission._detect_placement_letters = mock.Mock(
            return_value=placement_detection("B", center_x=420.0)
        )
        mission.state_reader = FreshPlacementStateReader(front_m=0.55)
        mission.motion = mock.Mock()

        observation, detected = mission._capture_placement_navigation_frame(
            target_letter="D",
            frame_sequence=0,
            started_at=time.monotonic(),
        )

        self.assertEqual(detected.candidates[0].recognized_letter, "B")
        self.assertEqual(
            observation.candidates,
            (LetterCandidate("B", 420.0, 0.90),),
        )
        self.assertEqual(mission.wide_camera.read_count, 3)
        mission._detect_placement_letters.assert_called_once()
        self.assertEqual(mission.motion.stop.call_count, 3)
        self.assertEqual(observation.frame_sequence, 1)
        self.assertIn(
            {"require_ultrasound": True, "require_fresh": True},
            mission.state_reader.safety_calls,
        )

    def test_recognized_candidate_stops_before_sensor_and_image_evidence(self) -> None:
        mission = LargeQuadrupedMission(load_config(), dry_run=True, skip_arm=True)
        mission.context.dry_run = False
        mission.wide_camera = PlacementCamera(
            (np.zeros((720, 1000, 3), dtype=np.uint8),)
        )
        mission._placement_undistorter = SimpleNamespace(apply=lambda value: value)
        events: list[str] = []

        def detect(_frame):
            events.append("detect")
            return placement_detection("C")

        class OrderedStateReader(FreshPlacementStateReader):
            def safety_error(self, **kwargs):
                events.append("sensor")
                return super().safety_error(**kwargs)

        mission._detect_placement_letters = detect
        mission.state_reader = OrderedStateReader()
        mission.motion = mock.Mock()
        mission.motion.stop.side_effect = lambda: events.append("stop")
        mission._save_placement_capture_images = mock.Mock(
            side_effect=lambda *_args: events.append("save")
        )

        mission._capture_placement_navigation_frame(
            target_letter="C",
            frame_sequence=0,
            started_at=time.monotonic(),
        )

        self.assertLess(events.index("stop"), events.index("detect"))
        self.assertLess(events.index("stop"), events.index("sensor"))
        self.assertLess(events.index("stop"), events.index("save"))

    def test_no_candidate_stops_before_processing_and_image_evidence(self) -> None:
        mission = LargeQuadrupedMission(load_config(), dry_run=True, skip_arm=True)
        mission.context.dry_run = False
        mission.wide_camera = PlacementCamera(
            (np.zeros((720, 1000, 3), dtype=np.uint8),)
        )
        events: list[str] = []

        def undistort(frame):
            events.append("undistort")
            return frame

        def detect(_frame):
            events.append("detect")
            return placement_detection(None)

        mission._placement_undistorter = SimpleNamespace(apply=undistort)
        mission._detect_placement_letters = detect
        mission.state_reader = FreshPlacementStateReader(front_m=0.60)
        mission.motion = mock.Mock()
        mission.motion.stop.side_effect = lambda: events.append("stop")
        mission._save_placement_capture_images = mock.Mock(
            side_effect=lambda *_args: events.append("save")
        )

        mission._capture_placement_navigation_frame(
            target_letter="D",
            frame_sequence=0,
            started_at=time.monotonic(),
        )

        self.assertLess(events.index("stop"), events.index("undistort"))
        self.assertLess(events.index("stop"), events.index("detect"))
        self.assertLess(events.index("stop"), events.index("save"))

    def test_front_stop_distance_stops_before_image_evidence_without_candidate(self) -> None:
        mission = LargeQuadrupedMission(load_config(), dry_run=True, skip_arm=True)
        mission.context.dry_run = False
        mission.wide_camera = PlacementCamera(
            (np.zeros((720, 1000, 3), dtype=np.uint8),)
        )
        mission._placement_undistorter = SimpleNamespace(apply=lambda value: value)
        mission._detect_placement_letters = mock.Mock(
            return_value=placement_detection(None)
        )
        events: list[str] = []

        class OrderedStateReader(FreshPlacementStateReader):
            def safety_error(self, **kwargs):
                events.append("sensor")
                return super().safety_error(**kwargs)

        mission.state_reader = OrderedStateReader(front_m=0.35)
        mission.motion = mock.Mock()
        mission.motion.stop.side_effect = lambda: events.append("stop")
        mission._save_placement_capture_images = mock.Mock(
            side_effect=lambda *_args: events.append("save")
        )

        mission._capture_placement_navigation_frame(
            target_letter="C",
            frame_sequence=0,
            started_at=time.monotonic(),
        )

        self.assertLess(events.index("stop"), events.index("sensor"))
        self.assertLess(events.index("stop"), events.index("save"))

    def test_evidence_write_time_blocks_strafe_after_total_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = load_config()
            config["placement_letter_navigation"] = dict(
                config["placement_letter_navigation"]
            )
            config["placement_letter_navigation"]["run_log_dir"] = temporary
            config["placement_letter_navigation"]["total_timeout_s"] = 1.0
            config["placement_letter_navigation"][
                "bilateral_search_enabled"
            ] = False
            config["placement_letter_navigation"][
                "immediate_complete_on_target_detection"
            ] = False
            mission = LargeQuadrupedMission(config, dry_run=True, skip_arm=True)
            mission.context.dry_run = False
            mission.wide_camera = PlacementCamera(())
            mission.front_camera = mock.Mock()
            mission.state_reader = FreshPlacementStateReader()
            mission.motion = mock.Mock()
            mission._prime_placement_front_filter = mock.Mock()
            observation = NavigationObservation(
                1,
                1000,
                (LetterCandidate("C", 200.0, 0.90),),
                0.60,
                0.10,
            )
            detected = placement_detection("C", center_x=200.0)
            mission._capture_placement_navigation_frame = mock.Mock(
                side_effect=[
                    (observation, detected),
                    MissionAbort("unexpected second capture"),
                ]
            )
            clock = [0.0]
            mission._write_placement_frame_evidence = mock.Mock(
                side_effect=lambda *_args, **_kwargs: clock.__setitem__(0, 2.0)
            )

            with mock.patch(
                "mission_lite3.mission.time.monotonic",
                side_effect=lambda: clock[0],
            ):
                with self.assertRaisesRegex(
                    MissionAbort,
                    "placement navigation total timeout",
                ):
                    mission._run_placement_letter_navigator("C")

            mission.motion.strafe_distance.assert_not_called()
            mission.motion.go_distance.assert_not_called()
            mission.motion.move.assert_not_called()
            mission.motion.stop.assert_called()

    def test_forward_evidence_write_time_blocks_next_move_after_total_timeout(self) -> None:
        config = load_config()
        config["placement_letter_navigation"] = dict(
            config["placement_letter_navigation"]
        )
        config["placement_letter_navigation"]["total_timeout_s"] = 1.0
        mission = LargeQuadrupedMission(config, dry_run=True, skip_arm=True)
        mission.context.dry_run = False
        mission.state_reader = FreshPlacementStateReader(front_m=1.20)
        mission.motion = mock.Mock()
        observation = NavigationObservation(1, 1000, (), 1.20, 0.10)
        detected = placement_detection(None)
        mission._capture_placement_navigation_frame = mock.Mock(
            side_effect=[
                (observation, detected),
                MissionAbort("unexpected second capture"),
            ]
        )
        clock = [0.0]
        mission._write_placement_frame_evidence = mock.Mock(
            side_effect=lambda *_args, **_kwargs: clock.__setitem__(0, 2.0)
        )
        navigator = mock.Mock()
        action = NavigationAction(
            ActionKind.FORWARD,
            "search",
            distance_m=0.10,
            vx_mps=0.08,
        )

        with mock.patch(
            "mission_lite3.mission.time.monotonic",
            side_effect=lambda: clock[0],
        ):
            with self.assertRaisesRegex(
                MissionAbort,
                "placement navigation total timeout",
            ):
                mission._run_placement_forward_search_step(
                    target_letter="C",
                    frame_sequence=0,
                    maximum_distance_m=0.10,
                    started_at=0.0,
                    action=action,
                    navigator=navigator,
                )

        mission.motion.move.assert_not_called()
        mission.motion.stop.assert_called_once_with()

    def test_forward_rechecks_sensor_freshness_after_evidence_before_move(self) -> None:
        mission = LargeQuadrupedMission(load_config(), dry_run=True, skip_arm=True)
        mission.context.dry_run = False
        mission.motion = mock.Mock()
        mission.state_reader = FreshPlacementStateReader(front_m=1.20)
        observation = NavigationObservation(1, 1000, (), 1.20, 0.10)
        detected = placement_detection(None)
        mission._capture_placement_navigation_frame = mock.Mock(
            side_effect=[
                (observation, detected),
                MissionAbort("unexpected second capture"),
            ]
        )
        mission._write_placement_frame_evidence = mock.Mock(
            side_effect=lambda *_args, **_kwargs: setattr(
                mission.state_reader,
                "safety_error",
                mock.Mock(return_value="state became stale during evidence write"),
            )
        )

        with self.assertRaisesRegex(
            MissionAbort,
            "state became stale during evidence write",
        ):
            mission._run_placement_forward_search_step(
                target_letter="D",
                frame_sequence=0,
                maximum_distance_m=0.10,
                started_at=time.monotonic(),
                navigator=mock.Mock(),
            )

        mission.motion.move.assert_not_called()
        mission.motion.stop.assert_called()

    def test_stale_ultrasound_aborts_capture(self) -> None:
        mission = LargeQuadrupedMission(load_config(), dry_run=True, skip_arm=True)
        mission.context.dry_run = False
        mission.wide_camera = PlacementCamera(
            (np.zeros((720, 1000, 3), dtype=np.uint8),)
        )
        mission._placement_undistorter = SimpleNamespace(apply=lambda value: value)
        mission._detect_placement_letters = mock.Mock(
            return_value=placement_detection(None)
        )
        mission.state_reader = FreshPlacementStateReader()
        mission.state_reader.safety_error = mock.Mock(return_value="ultrasound stale")
        mission.motion = mock.Mock()

        with self.assertRaisesRegex(MissionAbort, "ultrasound stale"):
            mission._capture_placement_navigation_frame(
                target_letter="D",
                frame_sequence=0,
                started_at=time.monotonic(),
            )

    def test_identical_cached_frames_with_new_timestamps_never_increment_sequence(self) -> None:
        mission = LargeQuadrupedMission(load_config(), dry_run=True, skip_arm=True)
        mission.context.dry_run = False
        first = np.zeros((720, 1000, 3), dtype=np.uint8)
        fresh = np.ones((720, 1000, 3), dtype=np.uint8)

        class IncrementingTimestampCamera(PlacementCamera):
            def __init__(self) -> None:
                super().__init__((first, first.copy(), first.copy(), first.copy()))
                self.frame_times = iter((10.0, 11.0, 12.0, 13.0))

            def read(self):
                self.read_count += 1
                frame = next(self.frames)
                self.last_frame_at = next(self.frame_times)
                return frame

        mission.wide_camera = IncrementingTimestampCamera()
        mission._placement_undistorter = SimpleNamespace(apply=lambda value: value)
        mission._detect_placement_letters = mock.Mock(
            return_value=placement_detection(None)
        )
        mission.state_reader = FreshPlacementStateReader()
        mission.motion = mock.Mock()

        first_observation, _ = mission._capture_placement_navigation_frame(
            target_letter="D",
            frame_sequence=0,
            started_at=time.monotonic(),
        )
        with self.assertRaisesRegex(MissionAbort, "stale_cached_frame"):
            mission._capture_placement_navigation_frame(
                target_letter="D",
                frame_sequence=first_observation.frame_sequence,
                started_at=time.monotonic(),
            )

        self.assertEqual(mission.wide_camera.read_count, 4)
        self.assertEqual(mission._detect_placement_letters.call_count, 1)
        mission.wide_camera = PlacementCamera((fresh,))
        second_observation, _ = mission._capture_placement_navigation_frame(
            target_letter="D",
            frame_sequence=first_observation.frame_sequence,
            started_at=time.monotonic(),
        )
        self.assertEqual(second_observation.frame_sequence, 2)

    def test_invalid_placement_target_still_attempts_final_stop(self) -> None:
        mission = LargeQuadrupedMission(load_config(), dry_run=True, skip_arm=True)
        mission.context.target_letter = None
        mission.motion = mock.Mock()

        with self.assertRaisesRegex(MissionAbort, "no valid target"):
            mission._execute_placement_letter_approach()

        mission.motion.stop.assert_called_once_with()

    def test_real_navigation_writes_unique_frame_evidence_and_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = load_config()
            config["placement_letter_navigation"] = dict(
                config["placement_letter_navigation"]
            )
            config["placement_letter_navigation"]["run_log_dir"] = temporary
            mission = LargeQuadrupedMission(config, dry_run=True, skip_arm=True)
            mission.context.dry_run = False
            frames = tuple(
                np.full((720, 1000, 3), value, dtype=np.uint8)
                for value in (10, 20, 30, 40, 50, 60)
            )
            mission.wide_camera = PlacementCamera(frames)
            mission.front_camera = mock.Mock()
            mission._placement_undistorter = SimpleNamespace(
                apply=lambda frame: frame.copy()
            )
            mission._detect_placement_letters = mock.Mock(
                return_value=placement_detection("C")
            )
            mission.state_reader = FreshPlacementStateReader(
                front_m=0.35,
                positions=(
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0),
                    (0.30, 0.0, 0.0),
                ),
            )
            mission.motion = RecordingPlacementMotion()

            measured = mission._run_placement_letter_navigator("C")

            self.assertEqual(measured, 0.0)
            run_dirs = [path for path in Path(temporary).iterdir() if path.is_dir()]
            self.assertEqual(len(run_dirs), 1)
            run_dir = run_dirs[0]
            self.assertEqual(len(list(run_dir.glob("*_raw.jpg"))), 6)
            self.assertEqual(len(list(run_dir.glob("*_undistorted.jpg"))), 6)
            self.assertEqual(len(list(run_dir.glob("*_annotated.jpg"))), 6)
            events = [
                json.loads(line)
                for line in (run_dir / "events.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            frame_events = [
                event
                for event in events
                if "sequence" in event and event.get("event") != "motion"
            ]
            self.assertEqual(
                [event["sequence"] for event in frame_events],
                [1, 2, 3, 4, 5, 6],
            )
            self.assertEqual(frame_events[-1]["result"], "complete")
            self.assertEqual(frame_events[-1]["action"]["kind"], "complete")
            self.assertIn("front_distance_m", frame_events[-1]["sensor"])
            self.assertIn("net_lateral_m", frame_events[-1]["cumulative"])
            mission.front_camera.release.assert_not_called()

    def test_motion_jsonl_event_contains_complete_result_frame_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = load_config()
            config["placement_letter_navigation"] = dict(
                config["placement_letter_navigation"]
            )
            config["placement_letter_navigation"]["run_log_dir"] = temporary
            config["placement_letter_navigation"][
                "bilateral_search_enabled"
            ] = False
            config["placement_letter_navigation"][
                "immediate_complete_on_target_detection"
            ] = False
            config["placement_letter_navigation"]["required_center_frames"] = 1
            mission = LargeQuadrupedMission(config, dry_run=True, skip_arm=True)
            mission.context.dry_run = False
            mission.wide_camera = PlacementCamera(
                tuple(
                    np.full((720, 1000, 3), value, dtype=np.uint8)
                    for value in (10, 20)
                )
            )
            mission.front_camera = mock.Mock()
            mission._placement_undistorter = SimpleNamespace(
                apply=lambda frame: frame.copy()
            )
            mission._detect_placement_letters = mock.Mock(
                side_effect=[placement_detection("C"), placement_detection("C")]
            )
            mission.state_reader = FreshPlacementStateReader(
                front_m=0.35,
                positions=(
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0),
                    (0.30, 0.0, 0.0),
                ),
            )
            mission.motion = RecordingPlacementMotion()

            measured = mission._run_placement_letter_navigator("C")

            self.assertAlmostEqual(measured, 0.0)
            run_dir = next(Path(temporary).iterdir())
            events = [
                json.loads(line)
                for line in (run_dir / "events.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            motion_event = next(
                event for event in events if event.get("event") == "motion"
            )
            self.assertEqual(motion_event["sequence"], 2)
            self.assertEqual(motion_event["source_sequence"], 1)
            self.assertEqual(motion_event["sensor"]["front_distance_m"], 0.35)
            self.assertEqual(
                motion_event["candidates"],
                [{"letter": "C", "center_x_px": 500.0, "confidence": 0.9}],
            )
            self.assertEqual(motion_event["action"]["kind"], "final_approach")
            self.assertAlmostEqual(motion_event["measured_distance_m"], 0.30)
            self.assertAlmostEqual(motion_event["cumulative"]["forward_m"], 0.0)
            self.assertAlmostEqual(
                motion_event["cumulative"]["final_approach_m"],
                0.30,
            )
            self.assertAlmostEqual(motion_event["cumulative"]["lateral_m"], 0.0)
            self.assertAlmostEqual(motion_event["cumulative"]["net_lateral_m"], 0.0)

    def test_second_placement_completes_visual_route_before_arm(self) -> None:
        config = load_config()
        config["inspection"] = dict(config["inspection"])
        config["inspection"]["place_pause_seconds"] = 0.0
        mission = LargeQuadrupedMission(config, dry_run=True, skip_arm=True)
        mission.context.dry_run = False
        mission.context.carried_bar = True
        mission.context.target_letter = "C"
        mission.context.placed_letters = ["A"]
        mission.state = MissionState.SECOND_PICK_PLACE
        mission._check_safety = lambda: None
        mission._align_to_letter_box = mock.Mock(return_value=0.0)
        mission._align_placement_row_yaw = mock.Mock()
        mission._run_placement_letter_navigator = mock.Mock(return_value=-0.12)
        mission.arm = mock.Mock()
        mission.arm.place_to_box.return_value = ArmTaskResult(
            True,
            "DONE",
            object_held=False,
            released=True,
        )
        mission.arm.stow.return_value = ArmTaskResult.success("MOVING_POSE")
        mission.motion = mock.Mock()

        with contextlib.redirect_stdout(io.StringIO()):
            placed = mission._place_carried_bar()

        self.assertTrue(placed)
        self.assertFalse(mission._placement_route_active)
        mission._run_placement_letter_navigator.assert_called_once_with("C")
        mission.motion.go_distance.assert_not_called()
        mission.arm.place_to_box.assert_called_once_with("C")

    def test_placement_forward_resumes_after_unconfirmed_ultrasound_spike(self) -> None:
        mission = LargeQuadrupedMission(load_config(), dry_run=True, skip_arm=True)
        mission.context.dry_run = False
        mission.state = MissionState.PLACE_TO_LETTER_BOX
        mission._placement_route_active = True
        mission._check_safety = mock.Mock()
        mission.motion = mock.Mock()
        mission.motion.go_distance.side_effect = [
            ForwardMotionGuardStop(
                "motion guard stopped forward command: ultrasound=0.28m "
                "threshold=0.35m state=PLACE_TO_LETTER_BOX"
            ),
            None,
        ]
        mission.state_reader = mock.Mock()
        mission.state_reader.pose.side_effect = [
            (1.0, 2.0, 0.0),
            (1.0, 2.0, 0.0),
            (1.0, 2.0, 0.0),
        ]
        mission._confirm_placement_front_stop = mock.Mock(return_value=False)

        with contextlib.redirect_stdout(io.StringIO()):
            mission._execute_route_action({"action": "forward", "distance_m": 1.38})

        self.assertEqual(mission.motion.go_distance.call_count, 2)
        self.assertEqual(
            [call.args[0] for call in mission.motion.go_distance.call_args_list],
            [1.38, 1.38],
        )
        self.assertEqual(
            [call.kwargs["speed_mps"] for call in mission.motion.go_distance.call_args_list],
            [0.08, 0.08],
        )
        mission._confirm_placement_front_stop.assert_called_once_with()

    def test_second_pickup_center_failure_continues_existing_red_alignment(self) -> None:
        config = load_config()
        config["box_center_alignment"] = dict(config["box_center_alignment"])
        config["box_center_alignment"]["enabled"] = True
        config["pickup_transfer"] = dict(config["pickup_transfer"])
        config["pickup_transfer"]["enabled"] = False
        mission = LargeQuadrupedMission(config, dry_run=True, skip_arm=True)
        mission.context.placed_letters = ["A"]
        mission.motion = mock.Mock()
        mission.arm = mock.Mock()
        mission.arm.stow.return_value = ArmTaskResult.success("MOVING_POSE")
        mission.arm.camera_pose.return_value = ArmTaskResult.success("GRASP_READY")
        mission._run_scripted_route = mock.Mock(return_value=True)
        mission._run_box_center_alignment = mock.Mock(
            return_value=alignment_failure()
        )
        mission._run_pregrasp_base_sequence = mock.Mock(return_value=True)
        mission._settle_after_pregrasp_stop = mock.Mock()
        mission._retry_grasp = mock.Mock(return_value=True)

        with contextlib.redirect_stdout(io.StringIO()):
            picked = mission._pick_target("C")

        self.assertTrue(picked)
        mission._run_box_center_alignment.assert_called_once_with("pickup")
        mission._run_pregrasp_base_sequence.assert_called_once_with()

    def test_placement_fallback_runs_only_after_verified_visual_rollback(self) -> None:
        config = load_config()
        config["box_center_alignment"] = dict(config["box_center_alignment"])
        config["box_center_alignment"]["enabled"] = True
        mission = LargeQuadrupedMission(config, dry_run=True, skip_arm=True)
        events: list[str] = []
        mission._run_box_center_alignment = mock.Mock(
            side_effect=lambda *_args: events.append("visual_rollback") or alignment_failure()
        )
        mission.motion = mock.Mock()
        mission.motion.strafe_distance.side_effect = lambda *_args, **_kwargs: events.append("fixed_fallback")

        with contextlib.redirect_stdout(io.StringIO()):
            offset = mission._align_to_letter_box("D")

        self.assertEqual(events, ["visual_rollback", "fixed_fallback"])
        self.assertAlmostEqual(offset, -0.75)

    def test_unverified_visual_rollback_blocks_fixed_fallback(self) -> None:
        config = load_config()
        config["box_center_alignment"] = dict(config["box_center_alignment"])
        config["box_center_alignment"]["enabled"] = True
        mission = LargeQuadrupedMission(config, dry_run=True, skip_arm=True)
        mission._run_box_center_alignment = mock.Mock(
            return_value=alignment_failure(rollback_ok=False)
        )
        mission.motion = mock.Mock()

        with self.assertRaisesRegex(MissionAbort, "rollback was not verified"):
            mission._align_to_letter_box("D")
        mission.motion.strafe_distance.assert_not_called()


if __name__ == "__main__":
    unittest.main()

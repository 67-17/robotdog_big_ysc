from __future__ import annotations

import contextlib
import io
import math
import unittest
from types import SimpleNamespace
from unittest import mock

from mission_lite3.arm import ArmTaskResult
from mission_lite3.box_center_alignment import BoxCenterAlignmentResult
from mission_lite3.config_loader import DEFAULT_FIELD, load_config
from mission_lite3.mission import LargeQuadrupedMission, MissionAbort
from mission_lite3.pickup_transfer import (
    LaneMovementResult,
    PickupRetreatResult,
    PickupTransferController,
    body_frame_delta,
)
from mission_lite3.wide_box_alignment import WideBoxAlignmentResult


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class FakeMotion:
    def __init__(self) -> None:
        self.limits = SimpleNamespace(command_hz=20.0)
        self.moves: list[tuple[float, float, float]] = []
        self.strafes: list[tuple[float, dict[str, float]]] = []
        self.stop_count = 0
        self.on_strafe = None

    def move(self, vx: float, vy: float, wz: float) -> None:
        self.moves.append((vx, vy, wz))

    def stop(self) -> None:
        self.stop_count += 1

    def strafe_distance_pose_hold(
        self,
        distance_m: float,
        speed_mps: float | None = None,
        **kwargs: float,
    ) -> None:
        values = dict(kwargs)
        values["speed_mps"] = float(speed_mps or 0.0)
        self.strafes.append((distance_m, values))
        if self.on_strafe is not None:
            self.on_strafe(distance_m)


class SequenceStateReader:
    def __init__(self, samples: list[tuple[float, float, float, float]]) -> None:
        self.samples = list(samples)
        self.index = 0
        self.state = SimpleNamespace(
            x=0.0,
            y=0.0,
            yaw=0.0,
            front_ultrasound_m=None,
        )
        self.error: str | None = None

    def safety_error(self, **_kwargs: object) -> str | None:
        if self.error:
            return self.error
        sample = self.samples[min(self.index, len(self.samples) - 1)]
        self.index += 1
        self.state.front_ultrasound_m, self.state.x, self.state.y, self.state.yaw = sample
        return None


class ReadForbiddenMapping(dict):
    def __getitem__(self, key: object) -> object:
        raise AssertionError(f"fixed lane offset was read: {key}")

    def get(self, key: object, default: object = None) -> object:
        raise AssertionError(f"fixed lane offset was read: {key}")


def transfer_config(**overrides: float) -> dict[str, object]:
    config = dict(load_config()["pickup_transfer"])
    config.update(overrides)
    return config


def alignment_result(ok: bool, reason: str = "aligned") -> BoxCenterAlignmentResult:
    return BoxCenterAlignmentResult(
        ok,
        reason,
        "pickup",
        "pickup",
        0,
        0,
        1,
        0.0,
        0.0,
        0.0,
        0.0,
        False,
        True,
        None,
    )


class PickupTransferControllerTest(unittest.TestCase):
    def test_body_frame_delta_uses_reference_heading(self) -> None:
        forward, lateral, yaw = body_frame_delta(
            (0.0, 0.0, math.pi / 2.0),
            (-0.5, 0.2, math.pi / 2.0 + 0.1),
        )
        self.assertAlmostEqual(forward, 0.2)
        self.assertAlmostEqual(lateral, 0.5)
        self.assertAlmostEqual(yaw, 0.1)

    def test_retreat_from_grasp_distance_to_80cm(self) -> None:
        reader = SequenceStateReader(
            [
                (0.285, 0.0, 0.0, 0.0),
                (0.50, -0.20, 0.0, 0.0),
                (0.78, -0.49, 0.0, 0.0),
            ]
        )
        motion = FakeMotion()
        clock = FakeClock()
        controller = PickupTransferController(
            motion,
            reader,
            transfer_config(),
            clock=clock,
            sleep=clock.sleep,
        )

        result = controller.retreat_to_front_distance()

        self.assertTrue(result.ok)
        self.assertEqual(result.reason, "target_reached")
        self.assertAlmostEqual(result.start_front_m or 0.0, 0.285)
        self.assertAlmostEqual(result.final_front_m or 0.0, 0.78)
        self.assertAlmostEqual(result.odom_retreat_m, 0.49)
        self.assertEqual(result.motion_command_count, 2)
        self.assertTrue(all(vx < 0.0 for vx, _, _ in motion.moves))
        self.assertGreaterEqual(motion.stop_count, 1)

    def test_retreat_already_at_target_sends_no_motion(self) -> None:
        reader = SequenceStateReader([(0.80, 0.0, 0.0, 0.0)])
        motion = FakeMotion()
        controller = PickupTransferController(motion, reader, transfer_config())

        result = controller.retreat_to_front_distance()

        self.assertTrue(result.ok)
        self.assertEqual(result.motion_command_count, 0)
        self.assertEqual(motion.moves, [])

    def test_retreat_rejects_stale_front_sample_without_motion(self) -> None:
        reader = SequenceStateReader([(0.30, 0.0, 0.0, 0.0)])
        reader.error = "ultrasound sample is stale"
        motion = FakeMotion()
        controller = PickupTransferController(motion, reader, transfer_config())

        result = controller.retreat_to_front_distance()

        self.assertFalse(result.ok)
        self.assertIn("stale", result.reason)
        self.assertEqual(motion.moves, [])

        invalid_reader = SequenceStateReader([(float("nan"), 0.0, 0.0, 0.0)])
        invalid_motion = FakeMotion()
        invalid = PickupTransferController(
            invalid_motion,
            invalid_reader,
            transfer_config(),
        ).retreat_to_front_distance()
        self.assertFalse(invalid.ok)
        self.assertEqual(invalid.reason, "invalid_front_ultrasound_sample")
        self.assertEqual(invalid_motion.moves, [])

    def test_retreat_stops_at_timeout_and_odometry_cap(self) -> None:
        clock = FakeClock()
        timeout_reader = SequenceStateReader([(0.30, 0.0, 0.0, 0.0)])
        timeout_motion = FakeMotion()
        timeout = PickupTransferController(
            timeout_motion,
            timeout_reader,
            transfer_config(retreat_timeout_s=0.10),
            clock=clock,
            sleep=clock.sleep,
        ).retreat_to_front_distance()
        self.assertFalse(timeout.ok)
        self.assertEqual(timeout.reason, "retreat_timeout")

        cap_reader = SequenceStateReader(
            [(0.30, 0.0, 0.0, 0.0), (0.50, -0.56, 0.0, 0.0)]
        )
        cap_motion = FakeMotion()
        cap = PickupTransferController(
            cap_motion,
            cap_reader,
            transfer_config(),
            clock=FakeClock(),
            sleep=lambda _seconds: None,
        ).retreat_to_front_distance()
        self.assertFalse(cap.ok)
        self.assertEqual(cap.reason, "retreat_odometry_limit")

    def test_retreat_uses_odometry_when_front_ultrasound_is_stuck_at_minimum(self) -> None:
        reader = SequenceStateReader(
            [
                (0.28, 0.00, 0.0, 0.0),
                (0.28, -0.10, 0.0, 0.0),
                (0.28, -0.20, 0.0, 0.0),
                (0.28, -0.30, 0.0, 0.0),
                (0.28, -0.44, 0.0, 0.0),
            ]
        )
        motion = FakeMotion()
        controller = PickupTransferController(
            motion,
            reader,
            transfer_config(),
            sleep=lambda _seconds: None,
        )

        result = controller.retreat_to_front_distance()

        self.assertTrue(result.ok)
        self.assertEqual(result.reason, "target_reached_odom_fallback")
        self.assertAlmostEqual(result.final_front_m or 0.0, 0.28)
        self.assertAlmostEqual(result.odom_retreat_m, 0.44)
        self.assertEqual(result.motion_command_count, 4)
        self.assertGreaterEqual(motion.stop_count, 1)

    def test_retreat_retry_counts_prior_odometry_with_other_stuck_value(self) -> None:
        reader = SequenceStateReader(
            [
                (0.31, 0.00, 0.0, 0.0),
                (0.31, -0.04, 0.0, 0.0),
                (0.31, -0.08, 0.0, 0.0),
                (0.31, -0.12, 0.0, 0.0),
                (0.31, -0.14, 0.0, 0.0),
            ]
        )
        motion = FakeMotion()
        controller = PickupTransferController(
            motion,
            reader,
            transfer_config(),
            sleep=lambda _seconds: None,
        )

        result = controller.retreat_to_front_distance(
            initial_odom_retreat_m=0.30,
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.reason, "target_reached_odom_fallback")
        self.assertAlmostEqual(result.odom_retreat_m, 0.44)

    def test_retreat_corrects_lateral_and_yaw_drift_and_enforces_limit(self) -> None:
        reader = SequenceStateReader(
            [
                (0.30, 0.0, 0.0, 0.0),
                (0.50, -0.20, 0.02, 0.02),
                (0.78, -0.48, 0.0, 0.0),
            ]
        )
        motion = FakeMotion()
        controller = PickupTransferController(
            motion,
            reader,
            transfer_config(),
            sleep=lambda _seconds: None,
        )
        result = controller.retreat_to_front_distance()
        self.assertTrue(result.ok)
        self.assertLess(motion.moves[1][1], 0.0)
        self.assertLess(motion.moves[1][2], 0.0)

        drift_reader = SequenceStateReader(
            [(0.30, 0.0, 0.0, 0.0), (0.50, -0.20, 0.11, 0.0)]
        )
        drift_motion = FakeMotion()
        drift = PickupTransferController(
            drift_motion,
            drift_reader,
            transfer_config(),
            sleep=lambda _seconds: None,
        ).retreat_to_front_distance()
        self.assertFalse(drift.ok)
        self.assertEqual(drift.reason, "retreat_lateral_drift_limit")

    def test_lane_move_records_signed_odometry_and_center_lane_sends_no_strafe(self) -> None:
        reader = SequenceStateReader([(0.80, 0.0, 0.0, math.pi / 2.0)])
        motion = FakeMotion()

        def update_pose(_distance: float) -> None:
            reader.samples = [(0.80, -0.48, 0.01, math.pi / 2.0 + 0.01)]
            reader.index = 0

        motion.on_strafe = update_pose
        controller = PickupTransferController(motion, reader, transfer_config())
        moved = controller.move_lane(0.50)
        self.assertTrue(moved.ok)
        self.assertAlmostEqual(moved.measured_distance_m, 0.48, places=3)
        self.assertEqual(len(motion.strafes), 1)

        centered = controller.move_lane(0.0)
        self.assertTrue(centered.ok)
        self.assertEqual(len(motion.strafes), 1)

    def test_controller_does_not_require_a_second_ultrasound_source(self) -> None:
        reader = SequenceStateReader([(0.80, 0.0, 0.0, 0.0)])
        self.assertFalse(hasattr(reader.state, "rear_ultrasound_m"))
        result = PickupTransferController(
            FakeMotion(),
            reader,
            transfer_config(),
        ).retreat_to_front_distance()
        self.assertTrue(result.ok)

    def test_operator_interrupt_stops_motion_and_propagates(self) -> None:
        reader = SequenceStateReader([(0.30, 0.0, 0.0, 0.0)])
        reader.safety_error = mock.Mock(side_effect=KeyboardInterrupt())
        motion = FakeMotion()
        controller = PickupTransferController(motion, reader, transfer_config())

        with self.assertRaises(KeyboardInterrupt):
            controller.retreat_to_front_distance()

        self.assertEqual(motion.stop_count, 1)


class PickupTransferMissionTest(unittest.TestCase):
    def make_mission(self) -> LargeQuadrupedMission:
        config = load_config()
        config["inspection"] = dict(config["inspection"])
        config["inspection"]["place_pause_seconds"] = 0.0
        return LargeQuadrupedMission(config, dry_run=True, skip_arm=True)

    def test_lane_directions_record_measured_value_and_restore_same_sign(self) -> None:
        for physical_left_sign in (1, -1):
            for letter, requested in {
                "A": 1.0,
                "B": 0.5,
                "C": 0.0,
                "D": -0.5,
            }.items():
                with self.subTest(
                    physical_left_sign=physical_left_sign,
                    letter=letter,
                ):
                    mission = self.make_mission()
                    mission.config["placement_letter_navigation"][
                        "physical_left_strafe_sign"
                    ] = physical_left_sign
                    mission.context.target_letter = letter
                    measured = requested * 0.96
                    controller = mock.Mock()
                    controller.move_lane.return_value = LaneMovementResult(
                        True,
                        "completed",
                        requested,
                        measured,
                        0.0,
                        0.0,
                        0 if requested == 0.0 else 1,
                    )
                    mission.pickup_transfer_controller = controller

                    with contextlib.redirect_stdout(io.StringIO()):
                        mission._execute_placement_lane_strafe()
                        mission._execute_pickup_lane_restore()

                    self.assertEqual(
                        mission.context.first_outbound_lane_strafe_m,
                        measured * physical_left_sign,
                    )
                    self.assertEqual(
                        [
                            call.args[0]
                            for call in controller.move_lane.call_args_list
                        ],
                        [requested, measured],
                    )

    def test_pickup_lane_restore_maps_physical_distance_to_command_sign(self) -> None:
        for physical_left_sign in (1, -1):
            for recorded in (0.2, -0.2):
                with self.subTest(
                    physical_left_sign=physical_left_sign,
                    recorded=recorded,
                ):
                    mission = self.make_mission()
                    mission.config["placement_letter_navigation"][
                        "physical_left_strafe_sign"
                    ] = physical_left_sign
                    mission.context.first_outbound_lane_strafe_m = recorded
                    command = recorded * physical_left_sign
                    controller = mock.Mock()
                    controller.move_lane.return_value = LaneMovementResult(
                        True,
                        "completed",
                        command,
                        command,
                        0.0,
                        0.0,
                        1,
                    )
                    mission.pickup_transfer_controller = controller

                    output = io.StringIO()
                    with contextlib.redirect_stdout(output):
                        mission._execute_pickup_lane_restore()

                    controller.move_lane.assert_called_once_with(command)
                    self.assertIn(
                        f"physical_recorded={recorded:.3f}m",
                        output.getvalue(),
                    )
                    self.assertIn(f"command={command:.3f}m", output.getvalue())

    def test_configured_transfer_routes_use_visual_placement_action(self) -> None:
        for routes in (
            DEFAULT_FIELD["scripted_route"],
            load_config()["scripted_route"],
        ):
            self.assertEqual(
                [action["action"] for action in routes["place_from_pickup"]],
                ["turn", "placement_row_yaw_align", "placement_letter_approach"],
            )
            self.assertAlmostEqual(
                routes["place_from_pickup"][0]["yaw_rad"],
                -math.pi,
                places=3,
            )
            self.assertNotIn(
                "placement_lane_strafe",
                [action["action"] for action in routes["place_from_pickup"]],
            )
            self.assertFalse(
                any(
                    action["action"] == "forward"
                    and action.get("distance_m") == 1.38
                    for action in routes["place_from_pickup"]
                )
            )
            self.assertEqual(
                [action["action"] for action in routes["pickup_from_place"]],
                ["turn", "forward", "pickup_lane_restore"],
            )
            self.assertAlmostEqual(
                routes["pickup_from_place"][0]["yaw_rad"],
                math.pi,
                places=3,
            )
            self.assertEqual(routes["pickup_from_place"][1]["distance_m"], 1.38)

    def test_placement_route_retry_does_not_repeat_completed_turns(self) -> None:
        mission = self.make_mission()
        events: list[str] = []
        fail_approach = True

        def execute(action: dict[str, object]) -> None:
            nonlocal fail_approach
            kind = str(action["action"])
            events.append(kind)
            if kind == "placement_letter_approach" and fail_approach:
                fail_approach = False
                raise MissionAbort("temporary placement camera failure")
            if kind == "placement_letter_approach":
                mission.context.placement_visual_approach_complete = True

        mission._execute_route_action = execute

        with self.assertRaisesRegex(MissionAbort, "camera failure"):
            mission._run_placement_route(True)
        self.assertEqual(mission.context.placement_route_action_index, 2)

        self.assertTrue(mission._run_placement_route(True))
        self.assertEqual(
            events,
            [
                "turn",
                "placement_row_yaw_align",
                "placement_letter_approach",
                "placement_letter_approach",
            ],
        )
        self.assertEqual(mission.context.placement_route_action_index, 3)

    def test_release_success_then_stow_retry_does_not_release_twice(self) -> None:
        mission = self.make_mission()
        mission.context.carried_bar = True
        mission.context.target_letter = "B"
        mission.context.placement_target_letter = "B"
        mission.context.placement_stage = "release"
        mission.motion = mock.Mock()
        mission.arm = mock.Mock()
        mission.arm.place_to_box.return_value = ArmTaskResult(
            True,
            "DONE",
            object_held=False,
            released=True,
        )
        mission.arm.stow.side_effect = [
            RuntimeError("temporary stow failure"),
            ArmTaskResult.success("MOVING_POSE"),
        ]

        with self.assertRaisesRegex(RuntimeError, "temporary stow failure"):
            mission._place_carried_bar()
        self.assertEqual(mission.context.placement_stage, "post_place_stow")
        self.assertFalse(mission.context.carried_bar)

        self.assertTrue(mission._place_carried_bar())
        mission.arm.place_to_box.assert_called_once_with("B")
        self.assertEqual(mission.arm.stow.call_count, 2)

    def test_each_placement_runs_visual_approach_and_never_reads_fixed_lane_offsets(self) -> None:
        mission = self.make_mission()
        mission.config["pickup_transfer"]["lane_offsets_m"] = ReadForbiddenMapping()
        mission._check_safety = mock.Mock()
        events: list[str] = []
        mission.motion = mock.Mock()
        mission.motion.turn_by.side_effect = lambda _yaw: events.append("turn")
        mission._align_placement_row_yaw = mock.Mock(
            side_effect=lambda: events.append("yaw_align") or True
        )
        mission._run_placement_letter_navigator = mock.Mock(
            side_effect=lambda letter: events.append(f"vision:{letter}")
            or ({"A": 0.42, "D": -0.31}[letter])
        )
        mission.arm = mock.Mock()
        mission.arm.place_to_box.side_effect = lambda letter: (
            events.append(f"arm:{letter}")
            or ArmTaskResult(True, "DONE", object_held=False, released=True)
        )
        mission.arm.stow.return_value = ArmTaskResult.success("MOVING_POSE")

        mission.context.carried_bar = True
        mission.context.target_letter = "A"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(mission._place_carried_bar())
        self.assertAlmostEqual(mission.context.first_outbound_lane_strafe_m, 0.42)

        mission.context.carried_bar = True
        mission.context.target_letter = "D"
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(mission._place_carried_bar())

        self.assertEqual(
            mission._run_placement_letter_navigator.call_args_list,
            [mock.call("A"), mock.call("D")],
        )
        self.assertAlmostEqual(mission.context.first_outbound_lane_strafe_m, 0.42)
        self.assertEqual(
            events,
            [
                "turn",
                "yaw_align",
                "vision:A",
                "arm:A",
                "turn",
                "yaw_align",
                "vision:D",
                "arm:D",
            ],
        )
        mission.motion.go_distance.assert_not_called()

    def test_legacy_mode_replaces_default_visual_route_before_fixed_alignment(self) -> None:
        mission = self.make_mission()
        mission.config["pickup_transfer"]["enabled"] = False
        mission.context.carried_bar = True
        mission.context.target_letter = "B"
        events: list[str] = []
        mission.motion = mock.Mock()
        mission.motion.turn_by.side_effect = lambda yaw: events.append(f"turn:{yaw:.4f}")
        mission._align_placement_row_yaw = mock.Mock(
            side_effect=lambda: events.append("yaw_align") or True
        )
        mission._execute_placement_lane_strafe = mock.Mock(
            side_effect=lambda: events.append("legacy_lane")
        )
        mission._run_placement_forward = mock.Mock(
            side_effect=lambda distance: events.append(f"legacy_forward:{distance:.2f}")
        )
        mission._run_placement_letter_navigator = mock.Mock()
        mission._align_to_letter_box = mock.Mock(
            side_effect=lambda letter: events.append(f"fixed_align:{letter}") or 0.0
        )
        mission.arm = mock.Mock()
        mission.arm.place_to_box.side_effect = lambda letter: (
            events.append(f"arm:{letter}")
            or ArmTaskResult(True, "DONE", object_held=False, released=True)
        )
        mission.arm.stow.return_value = ArmTaskResult.success("MOVING_POSE")

        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(mission._place_carried_bar())

        self.assertEqual(
            events,
            [
                "turn:-3.1416",
                "yaw_align",
                "legacy_lane",
                "legacy_forward:1.38",
                "fixed_align:B",
                "arm:B",
            ],
        )
        mission._run_placement_letter_navigator.assert_not_called()

    def test_legacy_mode_preserves_custom_nonvisual_route(self) -> None:
        mission = self.make_mission()
        mission.config["pickup_transfer"]["enabled"] = False
        mission.config["scripted_route"]["place_from_pickup"] = [
            {"action": "turn", "yaw_rad": -0.75},
            {"action": "forward", "distance_m": 0.60},
        ]
        mission.context.carried_bar = True
        mission.context.target_letter = "C"
        mission.motion = mock.Mock()
        mission._run_placement_forward = mock.Mock()
        mission._execute_placement_lane_strafe = mock.Mock()
        mission._run_placement_letter_navigator = mock.Mock()
        mission._align_to_letter_box = mock.Mock(return_value=0.0)
        mission.arm = mock.Mock()
        mission.arm.place_to_box.return_value = ArmTaskResult(
            True,
            "DONE",
            object_held=False,
            released=True,
        )
        mission.arm.stow.return_value = ArmTaskResult.success("MOVING_POSE")

        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(mission._place_carried_bar())

        mission.motion.turn_by.assert_called_once_with(-0.75)
        mission._run_placement_forward.assert_called_once_with(0.60)
        mission._execute_placement_lane_strafe.assert_not_called()
        mission._run_placement_letter_navigator.assert_not_called()

    def test_legacy_mode_keeps_drive_fallback_when_route_is_missing(self) -> None:
        mission = self.make_mission()
        mission.config["pickup_transfer"]["enabled"] = False
        mission.config["scripted_route"].pop("place_from_pickup")
        mission.context.carried_bar = True
        mission.context.target_letter = "A"
        mission.motion = mock.Mock()
        mission._drive_segment = mock.Mock()
        mission._run_placement_letter_navigator = mock.Mock()
        mission._align_to_letter_box = mock.Mock(return_value=0.0)
        mission.arm = mock.Mock()
        mission.arm.place_to_box.return_value = ArmTaskResult(
            True,
            "DONE",
            object_held=False,
            released=True,
        )
        mission.arm.stow.return_value = ArmTaskResult.success("MOVING_POSE")

        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(mission._place_carried_bar())

        mission._drive_segment.assert_called_once_with("place", distance_m=1.0)
        mission._run_placement_letter_navigator.assert_not_called()

    def test_missing_visual_placement_route_aborts_before_arm(self) -> None:
        mission = self.make_mission()
        mission.context.carried_bar = True
        mission.context.target_letter = "C"
        mission._run_scripted_route = mock.Mock(return_value=False)
        mission._drive_segment = mock.Mock()
        mission.arm = mock.Mock()

        with self.assertRaisesRegex(
            MissionAbort,
            "visual placement route is unavailable",
        ):
            mission._place_carried_bar()

        mission._drive_segment.assert_not_called()
        mission.arm.place_to_box.assert_not_called()
        self.assertTrue(mission.context.carried_bar)

    def test_route_without_successful_visual_action_aborts_before_arm(self) -> None:
        mission = self.make_mission()
        mission.context.carried_bar = True
        mission.context.target_letter = "B"
        mission._run_scripted_route = mock.Mock(return_value=True)
        mission.arm = mock.Mock()

        with self.assertRaisesRegex(
            MissionAbort,
            "visual placement route is unavailable",
        ):
            mission._place_carried_bar()

        mission.arm.place_to_box.assert_not_called()
        self.assertTrue(mission.context.carried_bar)

    def test_visual_placement_failure_preserves_carried_bar_and_blocks_arm(self) -> None:
        mission = self.make_mission()
        mission.context.carried_bar = True
        mission.context.target_letter = "D"
        mission._check_safety = mock.Mock()
        mission.motion = mock.Mock()
        mission._align_placement_row_yaw = mock.Mock(return_value=True)
        mission._run_placement_letter_navigator = mock.Mock(
            side_effect=MissionAbort("placement vision failed")
        )
        mission.arm = mock.Mock()

        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(
            MissionAbort,
            "placement vision failed",
        ):
            mission._place_carried_bar()

        mission.arm.place_to_box.assert_not_called()
        self.assertTrue(mission.context.carried_bar)
        self.assertEqual(mission.context.target_letter, "D")

    def test_placement_yaw_failure_warns_and_allows_lane_selection(self) -> None:
        mission = self.make_mission()
        mission.context.dry_run = False
        mission.config["placement_yaw_alignment"]["enabled"] = True
        mission.front_camera = mock.Mock()
        mission.placement_row_yaw_aligner = mock.Mock()
        mission.placement_row_yaw_aligner.run.return_value = WideBoxAlignmentResult(
            False,
            "no_valid_parallel_frames",
            0,
            0,
            None,
            None,
            None,
            None,
        )

        with contextlib.redirect_stdout(io.StringIO()) as output:
            aligned = mission._align_placement_row_yaw()

        self.assertFalse(aligned)
        self.assertIn("continue placement lane selection", output.getvalue())

    def test_first_cycle_transfers_directly_after_grasp_retreat(self) -> None:
        mission = self.make_mission()
        mission.context.dry_run = False
        events: list[str] = []
        mission.motion = mock.Mock()
        mission.arm = mock.Mock()
        mission.arm.stow.return_value = ArmTaskResult.success("MOVING_POSE")
        mission.arm.camera_pose.return_value = ArmTaskResult.success("GRASP_READY")
        mission._run_scripted_route = mock.Mock(
            side_effect=lambda name: events.append(f"route:{name}") or True
        )
        mission._run_pregrasp_base_sequence = mock.Mock(
            side_effect=lambda: events.append("pregrasp") or True
        )
        mission._settle_after_pregrasp_stop = mock.Mock()
        mission._retry_grasp = mock.Mock(
            side_effect=lambda _distance: events.append("grasp") or True
        )
        mission._retreat_from_pickup_box = mock.Mock(
            side_effect=lambda: events.append("retreat")
        )
        mission._align_pickup_departure_yaw = mock.Mock(
            side_effect=lambda: events.append("departure_yaw") or True
        )
        mission._align_pickup_box_center_strict = mock.Mock()

        with contextlib.redirect_stdout(io.StringIO()):
            picked = mission._pick_target("A")

        self.assertTrue(picked)
        self.assertEqual(
            events,
            [
                "route:pickup_from_upper_inspection",
                "pregrasp",
                "grasp",
                "retreat",
                "departure_yaw",
            ],
        )
        mission._align_pickup_box_center_strict.assert_not_called()

    def test_missing_or_out_of_range_lane_record_aborts_return(self) -> None:
        mission = self.make_mission()
        with self.assertRaisesRegex(MissionAbort, "unavailable"):
            mission._execute_pickup_lane_restore()
        mission.context.first_outbound_lane_strafe_m = 1.06
        with self.assertRaisesRegex(MissionAbort, "exceeds limit"):
            mission._execute_pickup_lane_restore()

    def test_second_cycle_returns_then_resumes_red_search_without_box_approach(self) -> None:
        mission = self.make_mission()
        mission.context.dry_run = False
        mission.context.placed_letters = ["A"]
        mission.context.first_outbound_lane_strafe_m = 1.0
        events: list[str] = []
        mission.motion = mock.Mock()
        mission.arm = mock.Mock()
        mission.arm.stow.return_value = ArmTaskResult.success("MOVING_POSE")
        mission.arm.camera_pose.return_value = ArmTaskResult.success("GRASP_READY")
        mission._run_scripted_route = mock.Mock(
            side_effect=lambda name: events.append(f"route:{name}") or True
        )
        mission._align_pickup_box_center_strict = mock.Mock()
        mission._run_pregrasp_base_sequence = mock.Mock(
            side_effect=lambda: events.append("pregrasp") or True
        )
        mission._settle_after_pregrasp_stop = mock.Mock()
        mission._retry_grasp = mock.Mock(
            side_effect=lambda _distance: events.append("grasp") or True
        )
        mission._retreat_from_pickup_box = mock.Mock(
            side_effect=lambda: events.append("retreat")
        )
        mission._align_pickup_departure_yaw = mock.Mock(
            side_effect=lambda: events.append("departure_yaw") or True
        )

        with contextlib.redirect_stdout(io.StringIO()):
            picked = mission._pick_target("D")

        self.assertTrue(picked)
        self.assertEqual(
            events,
            [
                "route:pickup_from_place",
                "pregrasp",
                "grasp",
                "retreat",
                "departure_yaw",
            ],
        )
        mission._align_pickup_box_center_strict.assert_not_called()

    def test_legacy_departure_center_checkpoint_skips_box_approach(self) -> None:
        mission = self.make_mission()
        mission.context.dry_run = False
        mission.context.carried_bar = True
        mission.context.target_letter = "A"
        mission.context.pickup_target_letter = "A"
        mission.context.pickup_route_name = "pickup_from_upper_inspection"
        mission.context.pickup_stage = "departure_center"
        mission._align_pickup_box_center_strict = mock.Mock(
            side_effect=AssertionError("box centering must not run")
        )

        with contextlib.redirect_stdout(io.StringIO()) as output:
            picked = mission._pick_target("A")

        self.assertTrue(picked)
        self.assertEqual(mission.context.pickup_stage, "complete")
        self.assertIn("skip legacy departure-center checkpoint", output.getvalue())
        mission._align_pickup_box_center_strict.assert_not_called()

    def test_mission_cycle_places_each_bar_before_returning_for_next(self) -> None:
        mission = self.make_mission()
        events: list[str] = []
        targets = iter(("D", "A"))
        mission._round_result_allows_pickup = mock.Mock(return_value=True)
        mission._next_target_letter = mock.Mock(side_effect=lambda: next(targets))

        def pick(letter: str) -> bool:
            events.append(f"pick:{letter}")
            mission.context.target_letter = letter
            mission.context.carried_bar = True
            mission.context.pickup_stage = "complete"
            return True

        def place() -> bool:
            letter = mission.context.target_letter
            events.append(f"place:{letter}")
            mission.context.placed_letters.append(str(letter))
            mission.context.carried_bar = False
            mission.context.target_letter = None
            mission.context.placement_stage = "complete"
            return True

        mission._pick_target = mock.Mock(side_effect=pick)
        mission._place_carried_bar = mock.Mock(side_effect=place)

        mission._state_pick_red_bar()
        mission._state_place_to_letter_box()
        mission._state_second_pick_place()

        self.assertEqual(
            events,
            ["pick:D", "place:D", "pick:A", "place:A"],
        )
        self.assertEqual(mission.context.placed_letters, ["D", "A"])

    def test_post_retreat_yaw_failure_warns_and_continues(self) -> None:
        mission = self.make_mission()
        mission.context.dry_run = False
        mission.motion = mock.Mock()
        mission._run_pregrasp_red_alignment = mock.Mock(
            side_effect=RuntimeError("no_valid_parallel_frames")
        )

        with contextlib.redirect_stdout(io.StringIO()) as output:
            aligned = mission._align_pickup_departure_yaw()

        self.assertFalse(aligned)
        self.assertIn("continue direct placement transfer", output.getvalue())
        self.assertGreaterEqual(mission.motion.stop.call_count, 2)

    def test_second_pickup_skips_box_center_and_continues_pregrasp_search(self) -> None:
        mission = self.make_mission()
        mission.context.dry_run = False
        mission.context.placed_letters = ["A"]
        mission.motion = mock.Mock()
        mission.arm = mock.Mock()
        mission.arm.stow.return_value = ArmTaskResult.success("MOVING_POSE")
        mission._run_scripted_route = mock.Mock(return_value=True)
        mission._run_box_center_alignment = mock.Mock(
            side_effect=AssertionError("box centering must not run")
        )
        mission._run_pregrasp_base_sequence = mock.Mock(return_value=True)
        mission._settle_after_pregrasp_stop = mock.Mock()
        mission._retry_grasp = mock.Mock(return_value=True)
        mission._retreat_from_pickup_box = mock.Mock()
        mission._align_pickup_departure_yaw = mock.Mock(return_value=True)

        with contextlib.redirect_stdout(io.StringIO()):
            picked = mission._pick_target("B")

        self.assertTrue(picked)
        mission._run_box_center_alignment.assert_not_called()
        mission._run_pregrasp_base_sequence.assert_called_once_with()

    def test_transfer_placement_skips_old_letter_alignment_and_reverse(self) -> None:
        mission = self.make_mission()
        mission.context.carried_bar = True
        mission.context.target_letter = "D"
        mission._run_placement_letter_navigator = mock.Mock(return_value=-0.31)
        mission._run_scripted_route = mock.Mock(
            side_effect=lambda _name: (
                mission._execute_placement_letter_approach() is None
            )
        )
        mission._align_to_letter_box = mock.Mock(return_value=-0.75)
        mission.motion = mock.Mock()
        mission.arm = mock.Mock()
        mission.arm.place_to_box.return_value = ArmTaskResult(
            True,
            "DONE",
            object_held=False,
            released=True,
        )
        mission.arm.stow.return_value = ArmTaskResult.success("MOVING_POSE")

        with contextlib.redirect_stdout(io.StringIO()):
            placed = mission._place_carried_bar()

        self.assertTrue(placed)
        mission._align_to_letter_box.assert_not_called()
        mission.motion.strafe_distance.assert_not_called()

    def test_strict_alignment_passes_per_stage_tolerance_to_aligner(self) -> None:
        mission = self.make_mission()
        mission.context.dry_run = False
        mission._run_box_center_alignment = mock.Mock(
            return_value=alignment_result(True)
        )

        with contextlib.redirect_stdout(io.StringIO()):
            mission._align_pickup_box_center_strict(
                stage="arrival",
                tolerance_fraction=0.06,
            )
            mission._align_pickup_box_center_strict(
                stage="departure",
                tolerance_fraction=0.03,
            )

        self.assertEqual(
            mission._run_box_center_alignment.call_args_list,
            [
                mock.call("pickup", tolerance_fraction=0.06),
                mock.call("pickup", tolerance_fraction=0.03),
            ],
        )

    def test_retreat_failure_aborts_before_departure_alignment(self) -> None:
        mission = self.make_mission()
        mission.pickup_transfer_controller = mock.Mock()
        mission.pickup_transfer_controller.retreat_to_front_distance.return_value = (
            PickupRetreatResult(False, "retreat_timeout", 0.285, 0.50, 0.20, 12.0, 10)
        )
        mission._align_pickup_box_center_strict = mock.Mock()

        with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(
            MissionAbort,
            "retreat_timeout",
        ):
            mission._retreat_from_pickup_box()
        mission._align_pickup_box_center_strict.assert_not_called()


if __name__ == "__main__":
    unittest.main()

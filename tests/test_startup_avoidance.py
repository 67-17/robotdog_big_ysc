from __future__ import annotations

import copy
import io
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from mission_lite3.config_loader import ConfigError, load_config, validate_config
from mission_lite3.mission import LargeQuadrupedMission
from mission_lite3.startup_avoidance import (
    StartupAvoidanceResult,
    StartupAvoidanceRunner,
)
from mission_lite3.startup_avoidance.controller import AvoidanceController
from mission_lite3.startup_avoidance.model import (
    BBox,
    Decision,
    Detection,
    SensorFrame,
    TrackUpdate,
    TrackView,
)
from mission_lite3.startup_avoidance.runtime import CameraSnapshot


def controller_config() -> dict:
    config = copy.deepcopy(load_config()["startup_avoidance"])
    config["decision"]["stable_frames"] = 2
    config["freshness"] = {
        "image_s": 1.0,
        "ultrasound_s": 1.0,
        "odom_s": 1.0,
    }
    return config


def frame(
    now: float,
    *,
    tracks=(),
    cleared_ids=(),
    ultrasound_m: float = 0.8,
    odom_x: float = 0.0,
    odom_y: float = 0.0,
    ambiguous: bool = False,
    image_age_s: float = 0.0,
    ultrasound_age_s: float = 0.0,
    odom_age_s: float = 0.0,
) -> SensorFrame:
    return SensorFrame(
        now,
        list(tracks),
        list(cleared_ids),
        ambiguous,
        ultrasound_m,
        odom_x,
        odom_y,
        0.0,
        image_age_s,
        ultrasound_age_s,
        odom_age_s,
    )


def track(track_id: int, zone: str) -> TrackView:
    x_by_zone = {
        "safe_left": 50,
        "front": 600,
        "side": 800,
        "safe_right": 1100,
    }
    detection = Detection(BBox(x_by_zone[zone], 300, 100, 200), 5000.0)
    return TrackView(track_id, detection, zone, 0)


class AvoidanceControllerTest(unittest.TestCase):
    def test_pass_speed_is_17_times_the_original_value(self) -> None:
        self.assertAlmostEqual(
            controller_config()["speed"]["pass_vx"],
            0.102,
        )

    def test_emergency_distance_enters_hold_then_resumes_after_stable_recovery(self) -> None:
        controller = AvoidanceController(controller_config())
        decision = controller.step(frame(1.0, ultrasound_m=0.15))

        self.assertFalse(decision.fault)
        self.assertEqual(decision.state, "HOLD")
        self.assertEqual(decision.reason, "emergency distance floor")
        self.assertEqual((decision.vx, decision.vy, decision.wz), (0.0, 0.0, 0.0))
        recovering = controller.step(frame(1.1, ultrasound_m=0.8))
        resumed = controller.step(frame(1.2, ultrasound_m=0.8))
        self.assertEqual(recovering.state, "HOLD")
        self.assertIn("recovering 1/2", recovering.reason)
        self.assertEqual(resumed.state, "CRUISE")
        self.assertGreater(resumed.vx, 0.0)

    def test_pass_hold_preserves_distance_origin_and_resumes_pass(self) -> None:
        config = controller_config()
        controller = AvoidanceController(config)
        controller.step(frame(1.0, ultrasound_m=0.8))
        controller.state = "PASS"
        controller.pass_origin_odom_x = 0.0
        controller.pass_origin_odom_y = 0.0
        controller.pass_progress_m = 0.0
        controller.return_line_target_m = 0.0

        held = controller.step(frame(1.1, ultrasound_m=0.15, odom_x=0.75))
        recovering = controller.step(frame(1.2, ultrasound_m=0.8, odom_x=0.75))
        resumed = controller.step(frame(1.3, ultrasound_m=0.8, odom_x=0.75))

        self.assertEqual(held.state, "HOLD")
        self.assertEqual(recovering.state, "HOLD")
        self.assertEqual(resumed.state, "PASS")
        self.assertEqual(resumed.reason, "pass_distance")
        self.assertAlmostEqual(controller.pass_origin_odom_x, 0.0)
        self.assertAlmostEqual(controller.pass_progress_m, 0.75)

    def test_perception_and_sensor_faults_hold_instead_of_exit(self) -> None:
        cases = (
            ("stale image", {"image_age_s": 1.1}),
            ("stale ultrasound", {"ultrasound_age_s": 1.1}),
            ("stale odometry", {"odom_age_s": 1.1}),
            ("ambiguous tracking", {"ambiguous": True}),
            ("unknown obstacle", {"ultrasound_m": 0.40}),
            ("invalid ultrasound_m", {"ultrasound_m": float("nan")}),
        )
        for expected_reason, overrides in cases:
            with self.subTest(reason=expected_reason):
                controller = AvoidanceController(controller_config())
                decision = controller.step(frame(1.0, **overrides))
                self.assertEqual(decision.state, "HOLD")
                self.assertFalse(decision.fault)
                self.assertIn(expected_reason, decision.reason)
                self.assertEqual(
                    (decision.vx, decision.vy, decision.wz),
                    (0.0, 0.0, 0.0),
                )

    def test_front_and_side_targets_start_confirmation_at_40cm(self) -> None:
        for zone in ("front", "side"):
            with self.subTest(zone=zone):
                config = controller_config()
                controller = AvoidanceController(config)
                target = track(1, zone)

                cruising = controller.step(
                    frame(1.0, tracks=[target], ultrasound_m=0.401)
                )
                confirming = controller.step(
                    frame(1.1, tracks=[target], ultrasound_m=0.40)
                )

                self.assertEqual(cruising.state, "CRUISE")
                self.assertEqual(confirming.state, "CONFIRM")
                self.assertEqual(
                    (confirming.vx, confirming.vy),
                    (config["speed"]["cruise_vx"], 0.0),
                )

    def test_zero_obstacles_finishes_when_zone_distance_is_crossed(self) -> None:
        controller = AvoidanceController(controller_config())

        cruising = controller.step(
            frame(1.0, ultrasound_m=0.8, odom_x=0.0)
        )
        almost_done = controller.step(
            frame(1.1, ultrasound_m=0.8, odom_x=2.39)
        )
        finished = controller.step(
            frame(1.2, ultrasound_m=0.8, odom_x=2.40)
        )

        self.assertEqual(cruising.state, "CRUISE")
        self.assertEqual(almost_done.state, "CRUISE")
        self.assertEqual(finished.state, "FINISHED")
        self.assertTrue(finished.finished)
        self.assertEqual(controller.avoidance_count, 0)

    def test_one_obstacle_keeps_total_forward_distance_at_two_point_four_metres(self) -> None:
        controller = AvoidanceController(controller_config())
        front = track(1, "front")
        safe_right = track(1, "safe_right")

        controller.step(frame(0.9, ultrasound_m=0.8, odom_x=0.0))
        controller.step(
            frame(1.0, tracks=[front], ultrasound_m=0.20, odom_x=0.30)
        )
        controller.step(
            frame(1.1, tracks=[front], ultrasound_m=0.20, odom_x=0.30)
        )
        controller.step(
            frame(
                1.2,
                tracks=[safe_right],
                ultrasound_m=0.45,
                odom_x=0.30,
                odom_y=0.20,
            )
        )
        pass_start = controller.step(
            frame(
                1.3,
                tracks=[safe_right],
                ultrasound_m=0.45,
                odom_x=0.30,
                odom_y=0.20,
            )
        )
        almost_done = controller.step(
            frame(
                1.4,
                cleared_ids=[1],
                ultrasound_m=0.8,
                odom_x=2.39,
                odom_y=0.20,
            )
        )
        return_start = controller.step(
            frame(1.5, ultrasound_m=0.8, odom_x=2.40, odom_y=0.20)
        )
        controller.step(
            frame(1.6, ultrasound_m=0.8, odom_x=2.40, odom_y=0.0)
        )
        finished = controller.step(
            frame(1.7, ultrasound_m=0.8, odom_x=2.40, odom_y=0.0)
        )

        self.assertEqual(pass_start.state, "PASS")
        self.assertEqual(almost_done.state, "PASS")
        self.assertAlmostEqual(controller.forward_progress_m, 2.4)
        self.assertAlmostEqual(controller.pass_progress_m, 2.1)
        self.assertEqual(return_start.state, "RETURN_LINE")
        self.assertEqual(return_start.reason, "return_right")
        self.assertEqual(finished.state, "FINISHED")
        self.assertTrue(finished.finished)
        self.assertEqual(controller.avoidance_count, 1)

    def test_second_obstacle_during_pass_preserves_total_distance_and_returns_to_first_line(self) -> None:
        config = controller_config()
        controller = AvoidanceController(config)
        first_front = track(1, "front")
        first_safe = track(1, "safe_right")
        second_front = track(2, "front")
        second_safe = track(2, "safe_right")

        controller.step(frame(1.0, tracks=[first_front], ultrasound_m=0.20))
        controller.step(frame(1.1, tracks=[first_front], ultrasound_m=0.20))
        controller.step(
            frame(1.2, tracks=[first_safe], ultrasound_m=0.45, odom_y=0.20)
        )
        controller.step(
            frame(1.3, tracks=[first_safe], ultrasound_m=0.45, odom_y=0.20)
        )
        self.assertEqual(controller.state, "PASS")
        self.assertAlmostEqual(controller.pass_origin_odom_x, 0.0)
        self.assertAlmostEqual(controller.return_line_target_m, 0.0)

        confirming_second = controller.step(
            frame(1.4, tracks=[second_front], ultrasound_m=0.20, odom_x=0.50, odom_y=0.20)
        )
        avoiding_second = controller.step(
            frame(1.5, tracks=[second_front], ultrasound_m=0.20, odom_x=0.51, odom_y=0.20)
        )
        self.assertEqual(confirming_second.state, "CONFIRM")
        self.assertEqual(confirming_second.vx, config["speed"]["pass_vx"])
        self.assertEqual(avoiding_second.state, "AVOID")
        self.assertIsNone(controller.pass_progress_m)
        self.assertAlmostEqual(controller.return_line_target_m, 0.0)

        controller.step(
            frame(1.6, tracks=[second_safe], ultrasound_m=0.45, odom_x=0.51, odom_y=0.40)
        )
        controller.step(
            frame(1.7, tracks=[second_safe], ultrasound_m=0.45, odom_x=0.51, odom_y=0.40)
        )
        self.assertEqual(controller.state, "PASS")
        self.assertAlmostEqual(controller.pass_origin_odom_x, 0.51)

        almost_done = controller.step(
            frame(1.8, ultrasound_m=0.8, odom_x=2.39, odom_y=0.40)
        )
        returning = controller.step(
            frame(1.9, ultrasound_m=0.8, odom_x=2.40, odom_y=0.40)
        )
        controller.step(frame(2.0, ultrasound_m=0.8, odom_x=2.40, odom_y=0.0))
        finished = controller.step(frame(2.1, ultrasound_m=0.8, odom_x=2.40, odom_y=0.0))

        self.assertEqual(almost_done.state, "PASS")
        self.assertEqual(returning.state, "RETURN_LINE")
        self.assertEqual(returning.reason, "return_right")
        self.assertEqual(finished.state, "FINISHED")
        self.assertTrue(finished.finished)
        self.assertEqual(controller.avoidance_count, 2)
        self.assertAlmostEqual(controller.forward_progress_m, 2.4)

    def test_pass_does_not_end_when_target_disappears_early(self) -> None:
        controller = AvoidanceController(controller_config())
        controller.step(frame(1.0, ultrasound_m=0.8))
        controller.state = "PASS"
        controller.pass_origin_odom_x = 0.0
        controller.pass_origin_odom_y = 0.0
        controller.pass_progress_m = 0.0
        controller.return_line_target_m = 0.0

        early_clear = controller.step(
            frame(1.1, cleared_ids=[1], ultrasound_m=0.8, odom_x=0.12)
        )
        almost_done = controller.step(frame(1.2, ultrasound_m=0.8, odom_x=2.39))
        at_distance = controller.step(frame(1.3, ultrasound_m=0.8, odom_x=2.40))

        self.assertEqual(early_clear.state, "PASS")
        self.assertEqual(early_clear.reason, "pass_distance")
        self.assertEqual(almost_done.state, "PASS")
        self.assertEqual(at_distance.state, "RETURN_LINE")

    def test_second_obstacle_at_two_point_four_metres_restarts_before_return(self) -> None:
        config = controller_config()
        controller = AvoidanceController(config)
        controller.step(frame(1.0, ultrasound_m=0.8))
        controller.state = "PASS"
        controller.pass_origin_odom_x = 0.0
        controller.pass_origin_odom_y = 0.0
        controller.pass_progress_m = 0.0
        controller.return_line_target_m = 0.0

        decision = controller.step(
            frame(
                1.1,
                tracks=[track(2, "front")],
                ultrasound_m=0.20,
                odom_x=2.40,
            )
        )

        self.assertEqual(decision.state, "CONFIRM")
        self.assertEqual(decision.vx, config["speed"]["pass_vx"])
        self.assertFalse(decision.finished)


class FakeMotion:
    def __init__(self) -> None:
        self.commands = []
        self.stop_calls = 0

    def move(self, vx, vy, wz) -> None:
        self.commands.append((vx, vy, wz))

    def stop(self) -> None:
        self.stop_calls += 1
        self.commands.append((0.0, 0.0, 0.0))


class FlakyMoveMotion(FakeMotion):
    def __init__(self) -> None:
        super().__init__()
        self.move_calls = 0

    def move(self, vx, vy, wz) -> None:
        self.move_calls += 1
        if self.move_calls == 1:
            raise RuntimeError("motion interface temporarily unavailable")
        super().move(vx, vy, wz)


class InspectableLog(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.closed_by_runner = False

    def close(self) -> None:
        self.closed_by_runner = True


class FakeCamera:
    def __init__(self) -> None:
        self.open_calls = 0
        self.release_calls = 0
        self.snapshot = CameraSnapshot(object(), 10.0, 1)

    def open(self) -> None:
        self.open_calls += 1

    def read(self):
        return self.snapshot

    def release(self) -> None:
        self.release_calls += 1


class FlakyOpenCamera(FakeCamera):
    def open(self) -> None:
        self.open_calls += 1
        if self.open_calls == 1:
            raise RuntimeError("camera temporarily unavailable")


class AdvancingCamera(FakeCamera):
    def read(self):
        sequence = self.snapshot.sequence + 1
        self.snapshot = CameraSnapshot(object(), 10.0, sequence)
        return self.snapshot


class AlwaysFailOpenCamera(FakeCamera):
    def open(self) -> None:
        self.open_calls += 1
        raise RuntimeError("camera unavailable")


class AdvancingClock:
    def __init__(self, start: float = 10.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


class FakeTracker:
    def __init__(self) -> None:
        self.active_id = None

    def update(self, _detections):
        return TrackUpdate([], [], False)

    def set_active(self, track_id) -> None:
        self.active_id = track_id

    def clear_active(self) -> None:
        self.active_id = None


class FakeDetector:
    def detect(self, _image):
        return []


class FakeController:
    def __init__(self, decision: Decision) -> None:
        self.decision = decision
        self.active_track_id = None
        self.avoidance_count = 2
        self.forward_progress_m = 2.00
        self.pass_progress_m = 2.00
        self.return_line_error_m = None
        self.hold_reasons = []

    def step(self, _frame):
        return self.decision

    def check_safety(self, _frame):
        return None

    def force_hold(self, reason):
        self.hold_reasons.append(str(reason))
        return Decision("HOLD", 0.0, 0.0, 0.0, str(reason), False, False)


def fake_state_reader():
    state = SimpleNamespace(
        x=0.0,
        y=0.0,
        yaw=0.0,
        front_ultrasound_m=0.8,
        ultrasound_updated_at=10.0,
        odom_updated_at=10.0,
    )
    return SimpleNamespace(
        state=state,
        safety_error=lambda **_kwargs: None,
    )


def recovering_state_reader():
    reader = fake_state_reader()
    errors = iter(("roll exceeds limit", None))
    reader.safety_error = lambda **_kwargs: next(errors, None)
    return reader


class StartupAvoidanceRunnerTest(unittest.TestCase):
    def _runner(self, decision: Decision):
        motion = FakeMotion()
        camera = FakeCamera()
        log_stream = InspectableLog()
        runner = StartupAvoidanceRunner(
            load_config(),
            motion,
            fake_state_reader(),
            camera=camera,
            detector=FakeDetector(),
            tracker=FakeTracker(),
            controller=FakeController(decision),
            clock=lambda: 10.0,
            sleep=lambda _seconds: None,
            log_root=Path("I:/tmp"),
            log_opener=lambda _path: log_stream,
            max_hold_retries=0,
        )
        return runner, motion, camera, log_stream

    def test_success_always_stops_releases_camera_and_writes_log(self) -> None:
        decision = Decision(
            "FINISHED", 0.0, 0.0, 0.0, "finished", True, False
        )
        runner, motion, camera, log_stream = self._runner(decision)

        result = runner.run()

        self.assertTrue(result.ok)
        self.assertEqual(result.avoidance_count, 2)
        self.assertEqual(motion.stop_calls, 1)
        self.assertEqual(camera.release_calls, 1)
        self.assertTrue(log_stream.closed_by_runner)
        self.assertIn('"event": "decision"', log_stream.getvalue())

    def test_fault_raises_and_still_stops_and_releases_camera(self) -> None:
        decision = Decision(
            "FAULT", 0.0, 0.0, 0.0, "test fault", False, True
        )
        runner, motion, camera, log_stream = self._runner(decision)

        with self.assertRaisesRegex(RuntimeError, "test fault"):
            runner.run()

        self.assertEqual(motion.stop_calls, 2)
        self.assertEqual(camera.release_calls, 2)
        self.assertTrue(log_stream.closed_by_runner)
        self.assertIn('"event": "error"', log_stream.getvalue())

    def test_camera_open_error_holds_reconnects_and_finishes(self) -> None:
        decision = Decision(
            "FINISHED", 0.0, 0.0, 0.0, "finished", True, False
        )
        motion = FakeMotion()
        camera = FlakyOpenCamera()
        log_stream = InspectableLog()
        controller = FakeController(decision)
        runner = StartupAvoidanceRunner(
            load_config(),
            motion,
            fake_state_reader(),
            camera=camera,
            detector=FakeDetector(),
            tracker=FakeTracker(),
            controller=controller,
            clock=lambda: 10.0,
            sleep=lambda _seconds: None,
            log_root=Path("I:/tmp"),
            log_opener=lambda _path: log_stream,
            max_hold_retries=2,
        )

        result = runner.run()

        self.assertTrue(result.ok)
        self.assertEqual(camera.open_calls, 2)
        self.assertIn("camera temporarily unavailable", controller.hold_reasons[0])
        self.assertIn('"event": "fault_hold"', log_stream.getvalue())
        self.assertNotIn('"event": "error"', log_stream.getvalue())

    def test_motion_error_holds_reconnects_and_finishes(self) -> None:
        decision = Decision(
            "FINISHED", 0.0, 0.0, 0.0, "finished", True, False
        )
        motion = FlakyMoveMotion()
        controller = FakeController(decision)
        runner = StartupAvoidanceRunner(
            load_config(),
            motion,
            fake_state_reader(),
            camera=FakeCamera(),
            detector=FakeDetector(),
            tracker=FakeTracker(),
            controller=controller,
            clock=lambda: 10.0,
            sleep=lambda _seconds: None,
            log_root=Path("I:/tmp"),
            log_opener=lambda _path: InspectableLog(),
            max_hold_retries=2,
        )

        result = runner.run()

        self.assertTrue(result.ok)
        self.assertEqual(motion.move_calls, 2)
        self.assertIn("motion interface temporarily unavailable", controller.hold_reasons[0])

    def test_robot_state_error_holds_then_continues_without_runner_exit(self) -> None:
        decision = Decision(
            "FINISHED", 0.0, 0.0, 0.0, "finished", True, False
        )
        controller = FakeController(decision)
        motion = FakeMotion()
        runner = StartupAvoidanceRunner(
            load_config(),
            motion,
            recovering_state_reader(),
            camera=AdvancingCamera(),
            detector=FakeDetector(),
            tracker=FakeTracker(),
            controller=controller,
            clock=lambda: 10.0,
            sleep=lambda _seconds: None,
            log_root=Path("I:/tmp"),
            log_opener=lambda _path: InspectableLog(),
            max_hold_retries=0,
        )

        result = runner.run()

        self.assertTrue(result.ok)
        self.assertIn("roll exceeds limit", controller.hold_reasons[0])
        self.assertEqual(motion.commands[0], (0.0, 0.0, 0.0))

    def test_continuous_controller_hold_times_out_and_cleans_up(self) -> None:
        config = load_config()
        config["startup_avoidance"] = copy.deepcopy(
            config["startup_avoidance"]
        )
        config["startup_avoidance"]["fault_hold_max_s"] = 0.20
        clock = AdvancingClock()
        motion = FakeMotion()
        camera = FakeCamera()
        runner = StartupAvoidanceRunner(
            config,
            motion,
            fake_state_reader(),
            camera=camera,
            detector=FakeDetector(),
            tracker=FakeTracker(),
            controller=FakeController(
                Decision("HOLD", 0.0, 0.0, 0.0, "persistent hold", False, False)
            ),
            clock=clock,
            sleep=clock.sleep,
            log_root=Path("I:/tmp"),
            log_opener=lambda _path: InspectableLog(),
        )

        with self.assertRaisesRegex(RuntimeError, "hold timed out"):
            runner.run()

        self.assertGreaterEqual(motion.stop_calls, 2)
        self.assertGreaterEqual(camera.release_calls, 2)

    def test_repeated_camera_open_failure_times_out_and_cleans_up(self) -> None:
        config = load_config()
        config["startup_avoidance"] = copy.deepcopy(
            config["startup_avoidance"]
        )
        config["startup_avoidance"]["fault_hold_retry_s"] = 0.10
        config["startup_avoidance"]["fault_hold_max_s"] = 0.25
        clock = AdvancingClock()
        motion = FakeMotion()
        camera = AlwaysFailOpenCamera()
        runner = StartupAvoidanceRunner(
            config,
            motion,
            fake_state_reader(),
            camera=camera,
            detector=FakeDetector(),
            tracker=FakeTracker(),
            controller=FakeController(
                Decision("FINISHED", 0.0, 0.0, 0.0, "finished", True, False)
            ),
            clock=clock,
            sleep=clock.sleep,
            log_root=Path("I:/tmp"),
            log_opener=lambda _path: InspectableLog(),
        )

        with self.assertRaisesRegex(RuntimeError, "fault hold timed out"):
            runner.run()

        self.assertGreaterEqual(camera.open_calls, 3)
        self.assertGreaterEqual(motion.stop_calls, 3)
        self.assertGreaterEqual(camera.release_calls, 3)

    def test_dry_run_does_not_open_camera_or_read_robot_state(self) -> None:
        motion = FakeMotion()
        camera = FakeCamera()
        state_reader = mock.Mock()
        runner = StartupAvoidanceRunner(
            load_config(),
            motion,
            state_reader,
            dry_run=True,
            camera=camera,
        )

        result = runner.run()

        self.assertTrue(result.ok)
        self.assertEqual(result.avoidance_count, 0)
        self.assertEqual(camera.open_calls, 0)
        self.assertEqual(camera.release_calls, 0)
        self.assertEqual(motion.commands, [])
        state_reader.assert_not_called()

    def test_ultrasound_window_does_not_duplicate_same_timestamp(self) -> None:
        runner = StartupAvoidanceRunner(
            load_config(),
            FakeMotion(),
            fake_state_reader(),
            dry_run=True,
            camera=FakeCamera(),
        )
        state = fake_state_reader().state
        self.assertEqual(runner._median_ultrasound(state), 0.8)
        state.front_ultrasound_m = 0.2
        self.assertEqual(runner._median_ultrasound(state), 0.8)
        state.ultrasound_updated_at = 10.1
        self.assertEqual(runner._median_ultrasound(state), 0.5)

    def test_runtime_requires_fresh_main_robot_state(self) -> None:
        decision = Decision(
            "FINISHED", 0.0, 0.0, 0.0, "finished", True, False
        )
        reader = fake_state_reader()
        reader.safety_error = mock.Mock(return_value=None)
        runner = StartupAvoidanceRunner(
            load_config(),
            FakeMotion(),
            reader,
            camera=FakeCamera(),
            detector=FakeDetector(),
            tracker=FakeTracker(),
            controller=FakeController(decision),
            clock=lambda: 10.0,
            sleep=lambda _seconds: None,
            log_root=Path("I:/tmp"),
            log_opener=lambda _path: InspectableLog(),
        )

        runner.run()

        reader.safety_error.assert_called_once_with(
            require_ultrasound=True,
            require_fresh=True,
        )


class StartupAvoidanceMissionIntegrationTest(unittest.TestCase):
    def test_pass_obstacle_uses_integrated_runner_then_preserves_stop_1_route(self) -> None:
        fake_runner = mock.Mock()
        fake_runner.run.return_value = StartupAvoidanceResult(
            True, 2, "finished", "avoidance.jsonl"
        )
        factory = mock.Mock(return_value=fake_runner)
        mission = LargeQuadrupedMission(
            load_config(),
            dry_run=True,
            skip_arm=True,
            startup_avoidance_runner_factory=factory,
        )
        mission._run_scripted_route = mock.Mock(return_value=True)

        mission._state_pass_obstacle()

        fake_runner.run.assert_called_once_with()
        mission._run_scripted_route.assert_not_called()
        mission._collect_inspection = mock.Mock()
        mission._state_inspect_left_object()
        self.assertEqual(
            mission._run_scripted_route.call_args_list[0],
            mock.call("inspect_stop_1_arrive"),
        )

    def test_runner_fault_does_not_fall_back_to_scripted_avoidance(self) -> None:
        fake_runner = mock.Mock()
        fake_runner.run.side_effect = RuntimeError("camera fault")
        mission = LargeQuadrupedMission(
            load_config(),
            dry_run=True,
            skip_arm=True,
            startup_avoidance_runner_factory=mock.Mock(
                return_value=fake_runner
            ),
        )
        mission._run_scripted_route = mock.Mock(return_value=True)

        with self.assertRaisesRegex(RuntimeError, "camera fault"):
            mission._state_pass_obstacle()

        mission._run_scripted_route.assert_not_called()


class StartupAvoidanceConfigTest(unittest.TestCase):
    def test_rejects_invalid_boolean_distance_zone_and_speed(self) -> None:
        mutations = [
            (
                lambda config: config["startup_avoidance"].__setitem__(
                    "enabled", 1
                ),
                "enabled must be a boolean",
            ),
            (
                lambda config: config["startup_avoidance"]["distance"].__setitem__(
                    "emergency_stop_m", 0.40
                ),
                "front_trigger_m must be greater than emergency_stop_m",
            ),
            (
                lambda config: config["startup_avoidance"]["zones"].__setitem__(
                    "front_center_min", 200
                ),
                "zones must be ordered",
            ),
            (
                lambda config: config["startup_avoidance"]["speed"].__setitem__(
                    "avoid_vy", config["motion"]["max_vy"] + 0.01
                ),
                "avoid_vy must be <=",
            ),
            (
                lambda config: config["startup_avoidance"]["decision"].__setitem__(
                    "finish_forward_m", 0.0
                ),
                "finish_forward_m must be >= 0.1",
            ),
            (
                lambda config: config["startup_avoidance"]["hsv"].__setitem__(
                    "lower", [19, 110, 80]
                ),
                "hsv.lower must not exceed hsv.upper",
            ),
            (
                lambda config: config["startup_avoidance"]["hsv"].__setitem__(
                    "upper", [180, 255, 255]
                ),
                r"hsv.upper\[0\] must be <= 179",
            ),
        ]
        for mutate, message in mutations:
            with self.subTest(message=message):
                config = copy.deepcopy(load_config())
                mutate(config)
                with self.assertRaisesRegex(ConfigError, message):
                    validate_config(config)


if __name__ == "__main__":
    unittest.main()

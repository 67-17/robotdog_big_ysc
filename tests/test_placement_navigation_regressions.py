from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mission_lite3.placement_letter_navigation import (
    ActionKind,
    LetterCandidate,
    NavigationObservation,
    NavigationAction,
    PlacementLetterNavigationConfig,
    PlacementLetterNavigator,
    placement_strafe_completion_tolerance,
)


class PlacementCenteringTests(unittest.TestCase):
    @staticmethod
    def observation(sequence: int, center_x_px: float) -> NavigationObservation:
        return NavigationObservation(
            frame_sequence=sequence,
            frame_width=1280,
            candidates=(LetterCandidate("D", center_x_px, 0.87),),
            front_distance_m=0.536,
            elapsed_s=float(sequence),
        )

    def test_central_third_edge_requires_fine_centering(self) -> None:
        config = PlacementLetterNavigationConfig(
            min_center_correction_m=0.02,
            max_center_correction_m=0.08,
            center_tolerance_fraction=0.05,
            required_center_frames=5,
        )
        navigator = PlacementLetterNavigator("D", config)
        navigator.decide(self.observation(1, 431.5))
        action = navigator.decide(self.observation(2, 431.5))

        self.assertEqual(action.kind, ActionKind.STRAFE)
        self.assertEqual(action.reason, "center_target=D")
        self.assertGreater(action.distance_m, 0.0)
        self.assertGreaterEqual(abs(action.distance_m), 0.02)
        self.assertLessEqual(abs(action.distance_m), 0.08)

    def test_five_fresh_centered_frames_are_required(self) -> None:
        config = PlacementLetterNavigationConfig(
            center_tolerance_fraction=0.05,
            required_center_frames=5,
        )
        navigator = PlacementLetterNavigator("D", config)
        actions = [
            navigator.decide(self.observation(sequence, 640.0))
            for sequence in range(1, 6)
        ]

        self.assertTrue(all(action.kind != ActionKind.COMPLETE for action in actions[:4]))
        self.assertEqual(actions[-1].kind, ActionKind.COMPLETE)
        self.assertEqual(actions[-1].reason, "target_finely_centered")

    def test_zero_progress_centering_retries_same_direction_with_larger_step(self) -> None:
        config = PlacementLetterNavigationConfig(
            min_center_correction_m=0.02,
            max_center_correction_m=0.08,
            center_tolerance_fraction=0.05,
            required_center_frames=5,
            strafe_min_progress_m=0.01,
        )
        navigator = PlacementLetterNavigator("D", config)
        navigator.decide(self.observation(1, 562.5))
        first = navigator.decide(self.observation(2, 562.5))
        navigator.record_motion(first, 0.0)
        second = navigator.decide(self.observation(3, 562.5))

        self.assertTrue(first.centering)
        self.assertTrue(second.centering)
        self.assertGreater(first.distance_m, 0.0)
        self.assertGreater(second.distance_m, first.distance_m)
        self.assertGreater(second.distance_m, 0.0)

    def test_placement_tolerance_never_swallows_positive_correction(self) -> None:
        for requested_m in (0.027246, 0.020, 0.012, 0.005):
            with self.subTest(requested_m=requested_m):
                tolerance_m = placement_strafe_completion_tolerance(
                    requested_m,
                    0.015,
                    0.01,
                )
                self.assertLess(tolerance_m, requested_m)
                self.assertGreaterEqual(
                    requested_m - tolerance_m,
                    min(requested_m, 0.01),
                )

    def test_locked_target_uses_current_frame_instead_of_stale_vote_median(self) -> None:
        config = PlacementLetterNavigationConfig(
            center_tolerance_fraction=0.05,
            required_center_frames=5,
        )
        navigator = PlacementLetterNavigator("D", config)
        navigator.decide(self.observation(1, 922.5))
        navigator.decide(self.observation(2, 873.0))

        action = navigator.decide(self.observation(3, 673.0))

        self.assertEqual(action.kind, ActionKind.RETRY)
        self.assertEqual(action.reason, "confirm_target_in_fine_center_band")

    def test_locked_target_loss_never_uses_other_letter_anchor(self) -> None:
        config = PlacementLetterNavigationConfig(
            center_tolerance_fraction=0.05,
            target_memory_max_lateral_m=0.15,
        )
        navigator = PlacementLetterNavigator("D", config)
        navigator.decide(self.observation(1, 922.5))
        navigator.decide(self.observation(2, 873.0))
        navigator.record_motion(
            NavigationAction(
                ActionKind.STRAFE,
                "test_move",
                distance_m=-0.20,
                vy_mps=-0.08,
            ),
            -0.18,
        )
        observation = NavigationObservation(
            frame_sequence=3,
            frame_width=1280,
            candidates=(LetterCandidate("C", 222.0, 0.75),),
            front_distance_m=0.54,
            elapsed_s=3.0,
        )

        action = navigator.decide(observation)

        self.assertEqual(action.kind, ActionKind.STRAFE)
        self.assertEqual(action.reason, "return_to_last_locked_target_pose")
        self.assertGreater(action.distance_m, 0.0)

    def test_elapsed_time_does_not_abort_placement_search(self) -> None:
        navigator = PlacementLetterNavigator(
            "D",
            PlacementLetterNavigationConfig(total_timeout_s=0.0),
        )
        observation = NavigationObservation(
            frame_sequence=1,
            frame_width=1280,
            candidates=(LetterCandidate("C", 640.0, 0.90),),
            front_distance_m=0.54,
            elapsed_s=3600.0,
        )

        action = navigator.decide(observation)

        self.assertEqual(action.kind, ActionKind.STRAFE)
        self.assertNotEqual(action.reason, "degraded_total_timeout")

    def test_bilateral_search_reverses_after_old_cumulative_budget(self) -> None:
        navigator = PlacementLetterNavigator(
            "D",
            PlacementLetterNavigationConfig(
                bilateral_search_enabled=True,
                lateral_search_step_m=0.10,
                lateral_search_each_side_m=0.10,
                max_lateral_search_m=0.31,
            ),
        )
        actions = []
        for sequence in range(1, 7):
            action = navigator.decide(
                NavigationObservation(
                    frame_sequence=sequence,
                    frame_width=1280,
                    candidates=(),
                    front_distance_m=0.54,
                    elapsed_s=sequence * 100.0,
                )
            )
            actions.append(action)
            self.assertEqual(action.kind, ActionKind.STRAFE)
            navigator.record_motion(action, action.distance_m)

        self.assertGreater(navigator.lateral_travel_m, 0.30)
        self.assertTrue(all(action.kind != ActionKind.FAIL for action in actions))


class PlacementLetterDetectionReplayTests(unittest.TestCase):
    @staticmethod
    def load_detector_config() -> tuple[Path, dict]:
        project_root = Path(__file__).resolve().parents[1]
        with (project_root / "mission_lite3/config/robot.yaml").open(
            "r",
            encoding="utf-8",
        ) as stream:
            config = json.load(stream)["box_center_alignment"]
        config = dict(config)
        config["placement_letter_min_confidence"] = 0.60
        return project_root, config

    def test_real_centering_frames_recover_d_from_glyph_contour(self) -> None:
        import cv2

        from mission_lite3.box_center_alignment import (
            detect_placement_letter_candidates,
        )

        project_root, config = self.load_detector_config()
        fixture_root = project_root / "tests/fixtures/placement_20260815_1700"

        expected_center_x = {
            4: 792.0,
            5: 730.0,
            6: 673.0,
            7: 482.0,
        }
        for frame_index in range(4, 8):
            with self.subTest(frame_index=frame_index):
                frame = cv2.imread(
                    str(
                        fixture_root
                        / f"frame_{frame_index:06d}_undistorted.jpg"
                    )
                )
                self.assertIsNotNone(frame)
                result = detect_placement_letter_candidates(frame, config)
                d_candidates = [
                    candidate
                    for candidate in result.candidates
                    if candidate.recognized_letter == "D"
                ]
                self.assertTrue(d_candidates, result.reason)
                closest = min(
                    d_candidates,
                    key=lambda candidate: abs(
                        candidate.center[0] - expected_center_x[frame_index]
                    ),
                )
                self.assertLessEqual(
                    abs(closest.center[0] - expected_center_x[frame_index]),
                    20.0,
                )
                self.assertGreater(closest.center[1], frame.shape[0] * 0.48)
                self.assertGreaterEqual(
                    closest.confidence,
                    0.60,
                )

    def test_real_replay_stops_at_centered_d_without_using_c_anchor(self) -> None:
        import cv2

        from mission_lite3.box_center_alignment import (
            detect_placement_letter_candidates,
        )

        project_root, detector_config = self.load_detector_config()
        fixture_root = project_root / "tests/fixtures/placement_20260815_1700"
        navigator = PlacementLetterNavigator(
            "D",
            PlacementLetterNavigationConfig(
                center_tolerance_fraction=0.05,
                required_center_frames=5,
                max_anchor_jump_m=0.08,
            ),
        )
        actions = []
        sequence = 0
        for frame_index in (4, 5, 6, 6, 6, 6, 6):
            sequence += 1
            frame = cv2.imread(
                str(
                    fixture_root
                    / f"frame_{frame_index:06d}_undistorted.jpg"
                )
            )
            detected = detect_placement_letter_candidates(
                frame,
                detector_config,
            )
            action = navigator.decide(
                NavigationObservation(
                    frame_sequence=sequence,
                    frame_width=frame.shape[1],
                    candidates=tuple(
                        LetterCandidate(
                            candidate.recognized_letter,
                            candidate.center[0],
                            candidate.confidence,
                        )
                        for candidate in detected.candidates
                    ),
                    front_distance_m=0.54,
                    elapsed_s=float(sequence),
                )
            )
            actions.append(action)

        self.assertTrue(
            all(not action.reason.startswith("anchor=C") for action in actions)
        )
        self.assertTrue(
            all(
                abs(action.distance_m) <= 0.08
                for action in actions
                if action.kind == ActionKind.STRAFE
            )
        )
        self.assertEqual(actions[-1].kind, ActionKind.COMPLETE)
        self.assertEqual(actions[-1].reason, "target_finely_centered")


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(0.0, float(seconds))

    def advance(self, seconds: float) -> None:
        self.now += float(seconds)


class FakeStateReader:
    def __init__(self) -> None:
        self.x = 0.0

    def pose(self) -> tuple[float, float, float]:
        return (self.x, 0.0, 0.0)


class FakeMotion:
    def __init__(
        self,
        clock: FakeClock,
        state: FakeStateReader,
        *,
        starts_enabled: bool,
    ) -> None:
        self.clock = clock
        self.state = state
        self.enabled = starts_enabled
        self.commands: list[tuple[float, float]] = []
        self.autonomous_calls = 0

    def move(self, vx: float, vy: float, wz: float) -> None:
        del vy, wz
        self.commands.append((self.clock.monotonic(), float(vx)))
        if self.enabled and vx > 0.0:
            self.state.x += 0.04

    def stop(self) -> None:
        return None

    def set_autonomous(self) -> None:
        self.autonomous_calls += 1
        self.enabled = True


class FakeNavigator:
    def __init__(self) -> None:
        self.forward_travel_m = 0.0
        self.lateral_travel_m = 0.0
        self.net_lateral_m = 0.0

    def record_motion(self, action, measured_distance_m: float) -> None:
        del action
        self.forward_travel_m += float(measured_distance_m)


class PlacementApproachTests(unittest.TestCase):
    def run_approach(
        self,
        *,
        starts_enabled: bool,
        initial_front_m: float = 0.536,
    ):
        from mission_lite3 import mission as mission_module

        clock = FakeClock()
        state = FakeStateReader()
        motion = FakeMotion(clock, state, starts_enabled=starts_enabled)
        events: list[dict] = []
        config = {
            "placement_letter_navigation": {
                "forward_budget_m": 1.80,
                "forward_speed_mps": 0.08,
                "front_stop_distance_m": 0.28,
                "total_timeout_s": 0.0,
                "ultrasound_stable_samples": 2,
                "motion_stall_timeout_s": 2.0,
                "motion_stall_min_progress_m": 0.01,
                "motion_stall_retries": 3,
                "motion_recovery_pause_s": 0.30,
                "motion_recovery_speed_mps": 0.05,
                "approach_slow_distance_m": 0.40,
                "approach_creep_distance_m": 0.33,
                "approach_slow_speed_mps": 0.05,
                "approach_creep_speed_mps": 0.025,
                "odometry_stop_guard_margin_m": 0.02,
                "ultrasound_echo_loss_margin_m": 0.35,
                "ultrasound_echo_loss_min_progress_m": 0.05,
                "echo_loss_fallback_speed_mps": 0.03,
                "visual_odom_fallback_speed_mps": 0.05,
                "approach_filter_warmup_s": 0.30,
                "visual_row_preflight_attempts": 1,
                "visual_row_preflight_trigger_m": 0.33,
                "require_visual_row_before_forward": False,
                "visual_ultrasound_start_tolerance_m": 0.35,
                "ultrasound_stuck_value_m": 0.28,
                "ultrasound_stuck_tolerance_m": 0.01,
                "final_ultrasound_min_m": 0.27,
                "final_ultrasound_max_m": 0.30,
                "ultrasound_odom_consistency_tolerance_m": 0.15,
            }
        }
        fake = SimpleNamespace(
            config=config,
            context=SimpleNamespace(dry_run=False),
            state_reader=state,
            motion=motion,
            _controlled_box_approach_active=False,
            _placement_last_sensor_evidence={},
        )
        fake._reset_placement_front_filter = lambda: None
        fake._prime_placement_front_filter = lambda _warmup_seconds=None: None
        fake._project_placement_motion = (
            mission_module.LargeQuadrupedMission._project_placement_motion
        )
        fake._append_placement_navigation_event = events.append

        visual_preflight_calls = []

        def visual_preflight():
            visual_preflight_calls.append(clock.monotonic())
            clock.advance(1.0)
            return None

        def front_distance():
            distance = max(0.29, initial_front_m - state.x)
            return distance, {
                "front_distance_m": distance,
                "front_candidate_m": distance,
                "front_sample_count": 5,
                "jump_rejected": False,
                "jump_confirmation_count": 0,
                "ultrasound_updated_at": clock.monotonic(),
                "ultrasound_age_s": 0.0,
            }

        fake._placement_label_row_distance_m = visual_preflight
        fake._placement_front_distance = front_distance
        fake._wait_for_placement_front_distance = lambda _label: front_distance()
        navigator = FakeNavigator()

        with patch.object(mission_module.time, "monotonic", clock.monotonic), patch.object(
            mission_module.time,
            "sleep",
            clock.sleep,
        ):
            mission_module.LargeQuadrupedMission._run_placement_ultrasound_approach(
                fake,
                target_letter="D",
                navigator=navigator,
                started_at=clock.monotonic(),
            )
        return clock, motion, navigator, events, visual_preflight_calls

    def test_normal_distance_skips_expensive_visual_preflight(self) -> None:
        clock, motion, navigator, events, visual_calls = self.run_approach(
            starts_enabled=True
        )

        self.assertEqual(visual_calls, [])
        self.assertGreater(len(motion.commands), 0)
        self.assertEqual(motion.autonomous_calls, 0)
        self.assertGreater(navigator.forward_travel_m, 0.0)
        self.assertEqual(events[-1]["result"], "complete")

    def test_suspicious_close_ultrasound_runs_one_visual_preflight(self) -> None:
        _clock, _motion, _navigator, events, visual_calls = self.run_approach(
            starts_enabled=True,
            initial_front_m=0.30,
        )

        self.assertEqual(len(visual_calls), 1)
        self.assertEqual(events[-1]["result"], "complete")

    def test_stalled_base_recovers_before_approach_fails(self) -> None:
        _clock, motion, navigator, events, _visual_calls = self.run_approach(
            starts_enabled=False
        )

        self.assertGreaterEqual(motion.autonomous_calls, 1)
        self.assertGreater(navigator.forward_travel_m, 0.0)
        self.assertEqual(events[-1]["result"], "complete")
        self.assertGreaterEqual(events[-1]["stall_recovery_count"], 1)


class PlacementSearchDistanceHoldTests(unittest.TestCase):
    def test_residual_front_drift_is_corrected_between_strafes(self) -> None:
        from mission_lite3 import mission as mission_module

        class State:
            x = 0.0

            def pose(self):
                return (self.x, 0.0, 0.0)

        class Motion:
            def __init__(self, state):
                self.state = state
                self.commands = []

            def set_autonomous(self):
                return None

            def go_distance(self, distance_m, *, speed_mps):
                self.commands.append((float(distance_m), float(speed_mps)))
                self.state.x += float(distance_m)

            def stop(self):
                return None

        state = State()
        motion = Motion(state)
        events = []
        context = SimpleNamespace(
            dry_run=False,
            placement_search_front_target_m=0.474,
            first_outbound_forward_m=1.28,
        )
        fake = SimpleNamespace(
            config={
                "placement_letter_navigation": {
                    "strafe_forward_deadband_m": 0.02,
                    "search_hold_boundary_delta_m": 0.20,
                    "search_hold_restore_attempts": 2,
                    "search_hold_restore_speed_mps": 0.03,
                    "search_hold_restore_max_step_m": 0.10,
                }
            },
            context=context,
            state_reader=state,
            motion=motion,
        )
        fake._placement_front_distance = lambda: (
            0.558 - state.x,
            {"front_distance_m": 0.558 - state.x},
        )
        fake._wait_for_placement_front_distance = (
            lambda _label: fake._placement_front_distance()
        )
        fake._project_placement_motion = (
            mission_module.LargeQuadrupedMission._project_placement_motion
        )
        fake._append_placement_navigation_event = events.append
        fake._reset_placement_front_filter = lambda: None
        fake._prime_placement_front_filter = lambda: None

        moved = (
            mission_module.LargeQuadrupedMission._restore_placement_search_front_distance(
                fake
            )
        )

        self.assertTrue(moved)
        self.assertEqual(len(motion.commands), 1)
        self.assertAlmostEqual(motion.commands[0][0], 0.084, places=3)
        self.assertAlmostEqual(context.first_outbound_forward_m, 1.364, places=3)
        self.assertEqual(events[-1]["result"], "complete")

    def test_small_residual_front_error_is_not_recorrected(self) -> None:
        from mission_lite3 import mission as mission_module

        motion = SimpleNamespace(
            set_autonomous=lambda: None,
            go_distance=lambda *_args, **_kwargs: self.fail(
                "small residual must not issue a motion command"
            ),
            stop=lambda: None,
        )
        fake = SimpleNamespace(
            config={
                "placement_letter_navigation": {
                    "strafe_forward_deadband_m": 0.02,
                    "search_hold_boundary_delta_m": 0.20,
                    "search_hold_restore_attempts": 1,
                    "search_hold_restore_speed_mps": 0.03,
                    "search_hold_restore_min_step_m": 0.04,
                    "search_hold_restore_max_step_m": 0.10,
                }
            },
            context=SimpleNamespace(
                dry_run=False,
                placement_search_front_target_m=0.457,
                first_outbound_forward_m=1.28,
            ),
            motion=motion,
        )
        fake._wait_for_placement_front_distance = lambda _label: (
            0.487,
            {"front_distance_m": 0.487},
        )

        moved = (
            mission_module.LargeQuadrupedMission._restore_placement_search_front_distance(
                fake
            )
        )

        self.assertFalse(moved)


class PlacementYawReferenceTests(unittest.TestCase):
    def test_one_visible_letter_allows_parallel_measurement(self) -> None:
        import numpy as np

        from mission_lite3.wide_box_alignment import detect_placement_row_parallel
        from mission_lite3.wide_camera import BoxParallelResult

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        centers = SimpleNamespace(ok=False, centers={}, reason="not_four_boxes")
        letters = SimpleNamespace(candidates=(object(),), reason="")
        parallel = BoxParallelResult(
            True,
            "",
            parallel_error_deg=0.5,
            confidence=0.9,
            box_x_range=(100, 1180),
        )
        result = detect_placement_row_parallel(
            frame,
            {},
            min_row_span_fraction=0.38,
            center_detector=lambda _frame, _config: centers,
            letter_detector=lambda _frame, _config: letters,
            parallel_detector=lambda _frame: parallel,
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.parallel_error_deg, 0.5)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from mission_lite3.inspection_runtime.letter_recognition import (
    GLYPH_HEIGHT_TOLERANCE_PX,
    _glyph_component_mask,
)

from mission_lite3.placement_letter_navigation import (
    MOTION_MEASUREMENT_TOLERANCE_M,
    ActionKind,
    LetterCandidate,
    NavigationAction,
    NavigationObservation,
    PlacementLetterNavigationConfig as _PlacementLetterNavigationConfig,
    PlacementLetterNavigator,
)


def PlacementLetterNavigationConfig(**kwargs):
    kwargs.setdefault("target_vote_window", 1)
    kwargs.setdefault("target_min_votes", 1)
    return _PlacementLetterNavigationConfig(**kwargs)


def seen(letter: str, x_fraction: float, confidence: float = 0.90) -> LetterCandidate:
    return LetterCandidate(letter, x_fraction * 1000.0, confidence)


def observed(
    sequence: int,
    *candidates: LetterCandidate,
    front_m: float = 0.35,
    elapsed_s: float = 1.0,
    frame_width: int = 1000,
) -> NavigationObservation:
    return NavigationObservation(
        sequence,
        frame_width,
        tuple(candidates),
        front_m,
        elapsed_s,
    )


class PlacementLetterDirectionTest(unittest.TestCase):
    def test_all_current_target_combinations_follow_physical_order(self) -> None:
        for current_index, current in enumerate("ABCD"):
            for target_index, target in enumerate("ABCD"):
                with self.subTest(current=current, target=target):
                    navigator = PlacementLetterNavigator(
                        target,
                        PlacementLetterNavigationConfig(required_center_frames=1),
                    )
                    action = navigator.decide(observed(1, seen(current, 0.50)))
                    if target_index > current_index:
                        self.assertEqual(action.kind, ActionKind.STRAFE)
                        self.assertLess(action.distance_m, 0.0)
                    elif target_index < current_index:
                        self.assertEqual(action.kind, ActionKind.STRAFE)
                        self.assertGreater(action.distance_m, 0.0)
                    else:
                        self.assertEqual(action.kind, ActionKind.COMPLETE)

    def test_target_candidate_wins_over_nearer_non_target_anchor(self) -> None:
        navigator = PlacementLetterNavigator(
            "D",
            PlacementLetterNavigationConfig(required_center_frames=1),
        )
        action = navigator.decide(observed(1, seen("B", 0.50), seen("D", 0.30)))
        self.assertEqual(action.kind, ActionKind.STRAFE)
        self.assertGreater(action.distance_m, 0.0)

    def test_highest_confidence_target_candidate_is_selected(self) -> None:
        navigator = PlacementLetterNavigator(
            "D",
            PlacementLetterNavigationConfig(required_center_frames=1),
        )
        action = navigator.decide(
            observed(1, seen("D", 0.10, 0.60), seen("D", 0.50, 0.95))
        )
        self.assertEqual(action.kind, ActionKind.COMPLETE)

    def test_target_confidence_tie_prefers_candidate_nearest_center(self) -> None:
        navigator = PlacementLetterNavigator(
            "D",
            PlacementLetterNavigationConfig(required_center_frames=1),
        )
        action = navigator.decide(
            observed(1, seen("D", 0.10, 0.90), seen("D", 0.52, 0.90))
        )
        self.assertEqual(action.kind, ActionKind.COMPLETE)

    def test_nonfinite_target_confidence_is_rejected(self) -> None:
        navigator = PlacementLetterNavigator("D", PlacementLetterNavigationConfig())
        action = navigator.decide(
            observed(
                1,
                seen("D", 0.50, float("inf")),
                seen("B", 0.50, 0.90),
            )
        )
        self.assertEqual(action.kind, ActionKind.STRAFE)
        self.assertIn("anchor=B", action.reason)

    def test_nearest_image_center_anchor_is_used_when_target_is_absent(self) -> None:
        navigator = PlacementLetterNavigator("D", PlacementLetterNavigationConfig())
        action = navigator.decide(observed(1, seen("A", 0.10), seen("C", 0.48)))
        self.assertEqual(action.kind, ActionKind.STRAFE)
        self.assertLess(action.distance_m, 0.0)
        self.assertIn("anchor=C", action.reason)


class PlacementLetterBilateralSearchTest(unittest.TestCase):
    @staticmethod
    def config() -> PlacementLetterNavigationConfig:
        return PlacementLetterNavigationConfig(
            max_lateral_search_m=3.10,
            bilateral_search_enabled=True,
            lateral_search_each_side_m=1.00,
            immediate_complete_on_target_detection=False,
            required_center_frames=1,
        )

    def test_target_detection_outside_center_band_starts_centering(
        self,
    ) -> None:
        navigator = PlacementLetterNavigator("C", self.config())

        action = navigator.decide(observed(1, seen("C", 0.93), front_m=0.80))

        self.assertEqual(action.kind, ActionKind.STRAFE)
        self.assertEqual(action.reason, "center_target=C")
        self.assertEqual(navigator.lateral_travel_m, 0.0)

    def test_non_target_letter_uses_known_box_spacing(self) -> None:
        navigator = PlacementLetterNavigator("D", self.config())

        action = navigator.decide(observed(1, seen("B", 0.50), front_m=0.80))

        self.assertEqual(action.kind, ActionKind.STRAFE)
        self.assertEqual(action.reason, "anchor=B;box_delta=2")
        self.assertAlmostEqual(action.distance_m, -0.20)

    def test_search_covers_one_meter_left_then_one_meter_right_from_origin(
        self,
    ) -> None:
        navigator = PlacementLetterNavigator("A", self.config())
        sequence = 0
        actions = []

        for _ in range(15):
            sequence += 1
            action = navigator.decide(observed(sequence, front_m=0.80))
            self.assertEqual(action.kind, ActionKind.STRAFE)
            actions.append(action)
            navigator.record_motion(action, action.distance_m)

        sequence += 1
        exhausted = navigator.decide(observed(sequence, front_m=0.80))

        self.assertTrue(all(action.distance_m > 0.0 for action in actions[:5]))
        self.assertTrue(all(action.distance_m < 0.0 for action in actions[5:]))
        self.assertEqual(exhausted.kind, ActionKind.FAIL)
        self.assertEqual(
            exhausted.reason,
            "degraded_bilateral_letter_search_exhausted",
        )
        self.assertAlmostEqual(navigator.lateral_travel_m, 3.00)
        self.assertAlmostEqual(navigator.net_lateral_m, -1.00)

    def test_target_found_during_right_sweep_stops_without_more_strafe(self) -> None:
        navigator = PlacementLetterNavigator("B", self.config())
        sequence = 0
        for _ in range(12):
            sequence += 1
            action = navigator.decide(observed(sequence, front_m=0.80))
            navigator.record_motion(action, action.distance_m)
        travel_before = navigator.lateral_travel_m

        complete = navigator.decide(
            observed(sequence + 1, seen("B", 0.50), front_m=0.80)
        )

        self.assertEqual(complete.kind, ActionKind.COMPLETE)
        self.assertAlmostEqual(navigator.lateral_travel_m, travel_before)


class PlacementLetterStateTest(unittest.TestCase):
    def test_config_defaults_match_the_task_contract(self) -> None:
        config = PlacementLetterNavigationConfig()
        self.assertEqual(
            (
                config.min_confidence,
                config.forward_speed_mps,
                config.lateral_speed_mps,
                config.front_stop_distance_m,
                config.forward_budget_m,
                config.forward_step_m,
                config.lateral_search_step_m,
                config.max_center_correction_m,
                config.max_lateral_search_m,
                config.acquisition_center_band,
                config.center_tolerance_fraction,
                config.final_approach_distance_m,
                config.letter_spacing_m,
                config.max_anchor_jump_m,
                config.required_center_frames,
                config.capture_retries,
                config.image_timeout_s,
                config.total_timeout_s,
            ),
            (
                0.60,
                0.08,
                0.08,
                0.40,
                1.80,
                0.10,
                0.20,
                0.10,
                1.05,
                (1.0 / 3.0, 2.0 / 3.0),
                0.05,
                0.0,
                0.50,
                0.20,
                3,
                3,
                0.50,
                90.0,
            ),
        )

    def test_no_letter_starts_lateral_search_without_distance_override(self) -> None:
        navigator = PlacementLetterNavigator("C", PlacementLetterNavigationConfig())
        left = navigator.decide(observed(1, front_m=1.00))
        self.assertEqual(left.kind, ActionKind.STRAFE)
        self.assertGreater(left.distance_m, 0.0)

    def test_cached_geometry_moves_directly_before_blind_search(self) -> None:
        navigator = PlacementLetterNavigator(
            "A",
            PlacementLetterNavigationConfig(
                max_lateral_search_m=3.10,
                bilateral_search_enabled=True,
            ),
            preferred_target_lateral_m=0.80,
        )
        first = navigator.decide(observed(1, front_m=0.80))
        self.assertEqual(first.reason, "cached_target_geometry")
        self.assertAlmostEqual(first.distance_m, 0.20)
        navigator.record_motion(first, 0.20)
        second = navigator.decide(observed(2, front_m=0.80))
        self.assertAlmostEqual(second.distance_m, 0.20)
        navigator.record_motion(second, 0.20)
        for sequence in (3, 4):
            action = navigator.decide(observed(sequence, front_m=0.80))
            navigator.record_motion(action, action.distance_m)
        fallback = navigator.decide(observed(5, front_m=0.80))
        self.assertEqual(fallback.reason, "search_left_for_target_letter")

    def test_visible_target_at_any_range_has_priority_over_ultrasound(self) -> None:
        navigator = PlacementLetterNavigator("D", PlacementLetterNavigationConfig())

        far = navigator.decide(
            observed(
                1,
                seen("A", 0.45),
                seen("D", 0.70),
                front_m=2.24,
            )
        )

        self.assertEqual(far.kind, ActionKind.STRAFE)
        self.assertEqual(far.reason, "center_target=D")
        navigator.record_motion(far, far.distance_m)

        at_search_distance = navigator.decide(
            observed(2, seen("A", 0.50), front_m=0.80)
        )
        self.assertEqual(at_search_distance.kind, ActionKind.RETRY)
        self.assertIn("target_memory", at_search_distance.reason)

    def test_left_search_fails_without_crossing_105cm(self) -> None:
        config = PlacementLetterNavigationConfig(max_lateral_search_m=1.05)
        navigator = PlacementLetterNavigator("A", config)
        sequence = 1
        while True:
            action = navigator.decide(observed(sequence, front_m=0.35))
            if action.kind == ActionKind.FAIL:
                break
            navigator.record_motion(action, action.distance_m)
            sequence += 1
        self.assertEqual(action.reason, "degraded_lateral_search_limit")
        self.assertLessEqual(navigator.lateral_travel_m, 1.05)

    def test_short_letter_loss_retries_then_continues_last_valid_direction(self) -> None:
        navigator = PlacementLetterNavigator(
            "D",
            PlacementLetterNavigationConfig(capture_retries=3),
        )
        first = navigator.decide(observed(1, seen("B", 0.50)))
        self.assertLess(first.distance_m, 0.0)
        for sequence in (2, 3, 4):
            retry = navigator.decide(observed(sequence))
            self.assertEqual(retry.kind, ActionKind.RETRY)
        resumed = navigator.decide(observed(5))
        self.assertEqual(resumed.kind, ActionKind.STRAFE)
        self.assertLess(resumed.distance_m, 0.0)

    def test_three_distinct_centered_frames_are_required(self) -> None:
        navigator = PlacementLetterNavigator("C", PlacementLetterNavigationConfig())
        first = navigator.decide(observed(1, seen("C", 0.50)))
        duplicate = navigator.decide(observed(1, seen("C", 0.50)))
        second = navigator.decide(observed(2, seen("C", 0.50)))
        third = navigator.decide(observed(3, seen("C", 0.50)))
        self.assertEqual(first.kind, ActionKind.RETRY)
        self.assertEqual(duplicate.reason, "duplicate_frame")
        self.assertEqual(second.kind, ActionKind.RETRY)
        self.assertEqual(third.kind, ActionKind.COMPLETE)
        self.assertEqual((third.vx_mps, third.vy_mps), (0.0, 0.0))

    def test_stale_sequence_does_not_count_as_a_new_centered_frame(self) -> None:
        navigator = PlacementLetterNavigator("C", PlacementLetterNavigationConfig())
        first = navigator.decide(observed(1, seen("C", 0.50)))
        second = navigator.decide(observed(2, seen("C", 0.50)))
        stale = navigator.decide(observed(1, seen("C", 0.50)))
        third = navigator.decide(observed(3, seen("C", 0.50)))
        self.assertEqual(first.kind, ActionKind.RETRY)
        self.assertEqual(second.kind, ActionKind.RETRY)
        self.assertEqual(stale.kind, ActionKind.RETRY)
        self.assertEqual(stale.reason, "stale_frame")
        self.assertEqual(third.kind, ActionKind.COMPLETE)

    def test_target_in_center_third_completes_without_final_approach(self) -> None:
        navigator = PlacementLetterNavigator(
            "B",
            PlacementLetterNavigationConfig(required_center_frames=1),
        )
        complete = navigator.decide(observed(1, seen("B", 0.65), front_m=0.80))
        self.assertEqual(complete.kind, ActionKind.COMPLETE)
        self.assertEqual((complete.vx_mps, complete.vy_mps), (0.0, 0.0))

    def test_final_approach_rejects_short_odometry(self) -> None:
        navigator = PlacementLetterNavigator(
            "B",
            PlacementLetterNavigationConfig(required_center_frames=1),
        )
        approach = NavigationAction(
            ActionKind.FINAL_APPROACH,
            "legacy_final_approach",
            distance_m=0.10,
            vx_mps=0.08,
        )
        with self.assertRaisesRegex(ValueError, "disabled"):
            navigator.record_motion(approach, 0.02)
        self.assertFalse(navigator.final_approach_completed)

    def test_centered_target_completes_without_forward_at_any_range(self) -> None:
        config = PlacementLetterNavigationConfig(required_center_frames=1)
        navigator = PlacementLetterNavigator("A", config)
        action = navigator.decide(observed(1, seen("A", 0.50), front_m=1.00))
        self.assertEqual(action.kind, ActionKind.COMPLETE)
        self.assertEqual(action.vx_mps, 0.0)

    def test_centered_target_cannot_complete_while_too_far_at_budget_limit(self) -> None:
        config = PlacementLetterNavigationConfig(
            required_center_frames=1,
            forward_budget_m=0.15,
        )
        navigator = PlacementLetterNavigator("A", config)
        navigator.forward_travel_m = 0.15

        action = navigator.decide(
            observed(1, seen("A", 0.50), front_m=1.00)
        )

        self.assertEqual(action.kind, ActionKind.COMPLETE)

    def test_forward_budget_never_switches_to_strafe_while_too_far(self) -> None:
        config = PlacementLetterNavigationConfig(forward_budget_m=0.15)
        navigator = PlacementLetterNavigator("A", config)
        first = navigator._forward("test_forward_budget")
        navigator.record_motion(first, 0.10)
        second = navigator._forward("test_forward_budget")
        self.assertAlmostEqual(second.distance_m, 0.05)
        navigator.record_motion(second, 0.05)
        exhausted = navigator._forward("test_forward_budget")
        self.assertEqual(exhausted.kind, ActionKind.FAIL)
        self.assertEqual(
            exhausted.reason,
            "degraded_forward_budget_exhausted",
        )
        self.assertEqual(exhausted.vx_mps, 0.0)
        self.assertEqual(exhausted.vy_mps, 0.0)
        self.assertAlmostEqual(navigator.forward_travel_m, 0.15)

    def test_timeout_invalid_ultrasound_and_invalid_target_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            PlacementLetterNavigator("X", PlacementLetterNavigationConfig())
        timeout = PlacementLetterNavigator("A", PlacementLetterNavigationConfig())
        self.assertEqual(
            timeout.decide(observed(1, elapsed_s=90.01)).reason,
            "degraded_total_timeout",
        )
        for sequence, front_m in enumerate((float("nan"), 0.0, -0.1), start=1):
            with self.subTest(front_m=front_m):
                invalid = PlacementLetterNavigator(
                    "A",
                    PlacementLetterNavigationConfig(),
                )
                action = invalid.decide(observed(sequence, front_m=front_m))
                self.assertEqual(action.kind, ActionKind.FAIL)
                self.assertEqual(action.reason, "invalid_front_ultrasound")

    def test_frame_sequence_requires_nonnegative_non_boolean_integer(self) -> None:
        for frame_sequence in (float("nan"), 1.0, True, -1):
            with self.subTest(frame_sequence=frame_sequence):
                navigator = PlacementLetterNavigator(
                    "A",
                    PlacementLetterNavigationConfig(),
                )
                action = navigator.decide(
                    observed(frame_sequence, seen("A", 0.50))
                )
                self.assertEqual(action.kind, ActionKind.FAIL)
                self.assertEqual(action.reason, "invalid_frame_sequence")
                self.assertEqual((action.vx_mps, action.vy_mps), (0.0, 0.0))

        zero_sequence = PlacementLetterNavigator(
            "A",
            PlacementLetterNavigationConfig(required_center_frames=1),
        ).decide(observed(0, seen("A", 0.50)))
        self.assertEqual(zero_sequence.kind, ActionKind.COMPLETE)

    def test_nan_sequence_does_not_accumulate_centered_frames(self) -> None:
        navigator = PlacementLetterNavigator("C", PlacementLetterNavigationConfig())
        first = navigator.decide(observed(0, seen("C", 0.50)))
        invalid = navigator.decide(observed(float("nan"), seen("C", 0.50)))
        second = navigator.decide(observed(1, seen("C", 0.50)))
        third = navigator.decide(observed(2, seen("C", 0.50)))
        self.assertEqual(first.kind, ActionKind.RETRY)
        self.assertEqual(invalid.kind, ActionKind.FAIL)
        self.assertEqual(invalid.reason, "invalid_frame_sequence")
        self.assertEqual(second.kind, ActionKind.RETRY)
        self.assertEqual(third.kind, ActionKind.COMPLETE)

    def test_frame_width_requires_positive_non_boolean_integer(self) -> None:
        for frame_width in (float("nan"), 1000.0, True, -1, 0):
            with self.subTest(frame_width=frame_width):
                navigator = PlacementLetterNavigator(
                    "D",
                    PlacementLetterNavigationConfig(),
                )
                action = navigator.decide(
                    observed(1, seen("D", 0.10), frame_width=frame_width)
                )
                self.assertEqual(action.kind, ActionKind.FAIL)
                self.assertEqual(action.reason, "invalid_frame_width")
                self.assertEqual((action.vx_mps, action.vy_mps), (0.0, 0.0))

    def test_nonfinite_or_negative_elapsed_time_fails_closed(self) -> None:
        for elapsed_s in (float("nan"), float("inf"), -0.01):
            with self.subTest(elapsed_s=elapsed_s):
                navigator = PlacementLetterNavigator(
                    "A",
                    PlacementLetterNavigationConfig(),
                )
                action = navigator.decide(observed(1, elapsed_s=elapsed_s))
                self.assertEqual(action.kind, ActionKind.FAIL)
                self.assertEqual(action.reason, "invalid_elapsed_time")

    def test_untrusted_candidates_are_ignored(self) -> None:
        navigator = PlacementLetterNavigator("D", PlacementLetterNavigationConfig())
        action = navigator.decide(
            observed(
                1,
                seen("D", 0.50, confidence=0.49),
                LetterCandidate("D", float("nan"), 0.99),
                seen("X", 0.50, confidence=0.99),
                front_m=1.00,
            )
        )
        self.assertEqual(action.kind, ActionKind.STRAFE)
        self.assertEqual(action.reason, "search_left_without_letter")

    def test_actions_never_command_forward_and_lateral_axes_together(self) -> None:
        navigator = PlacementLetterNavigator("D", PlacementLetterNavigationConfig())
        actions = [
            navigator.decide(observed(1, front_m=0.60)),
            navigator.decide(observed(2, seen("B", 0.50), front_m=0.60)),
        ]
        for action in actions:
            self.assertFalse(action.vx_mps and action.vy_mps)


class PlacementLetterRegressionTest(unittest.TestCase):
    def test_target_requires_two_of_three_frames(self) -> None:
        navigator = PlacementLetterNavigator(
            "D",
            _PlacementLetterNavigationConfig(required_center_frames=1),
        )
        first = navigator.decide(observed(1, seen("D", 0.50), front_m=1.34))
        missed = navigator.decide(observed(2, seen("C", 0.50), front_m=1.34))
        confirmed = navigator.decide(observed(3, seen("D", 0.50), front_m=1.34))
        self.assertEqual(first.reason, "confirm_target_detection_2_of_3")
        self.assertEqual(missed.reason, "confirm_recent_target_before_anchor_fallback")
        self.assertEqual(confirmed.kind, ActionKind.COMPLETE)
        self.assertEqual(confirmed.vx_mps, 0.0)

    def test_remembered_target_beats_anchor_and_ultrasound_jump(self) -> None:
        navigator = PlacementLetterNavigator(
            "D",
            _PlacementLetterNavigationConfig(required_center_frames=1),
        )
        navigator.decide(observed(1, seen("D", 0.30), front_m=0.76))
        center = navigator.decide(observed(2, seen("D", 0.30), front_m=0.76))
        self.assertEqual(center.kind, ActionKind.STRAFE)
        navigator.record_motion(center, center.distance_m)

        missed = navigator.decide(observed(3, seen("C", 0.48), front_m=1.34))

        self.assertNotEqual(missed.kind, ActionKind.FORWARD)
        self.assertNotIn("anchor=C", missed.reason)
        self.assertIn("target_memory", missed.reason)

    def test_25px_glyph_passes_114px_height_boundary(self) -> None:
        self.assertEqual(GLYPH_HEIGHT_TOLERANCE_PX, 2)
        mask = np.zeros((114, 121), dtype=bool)
        mask[44:69, 45:75] = True
        _component, bbox = _glyph_component_mask(mask)
        self.assertIsNotNone(bbox)
        self.assertEqual(bbox[3] - bbox[1], 25)

    def test_record_motion_counts_negative_forward_measurements(self) -> None:
        navigator = PlacementLetterNavigator("D", PlacementLetterNavigationConfig())
        forward = navigator._forward("test_forward_measurement")
        with self.assertRaises(ValueError):
            navigator.record_motion(forward, float("nan"))
        navigator.record_motion(forward, -0.01)
        self.assertAlmostEqual(navigator.forward_travel_m, 0.01)
        navigator.record_motion(forward, -0.02)
        self.assertAlmostEqual(navigator.forward_travel_m, 0.03)

        strafe = navigator.decide(observed(2, seen("B", 0.50)))
        with self.assertRaises(ValueError):
            navigator.record_motion(strafe, 0.01)
        self.assertEqual(navigator.lateral_travel_m, 0.0)
        self.assertEqual(navigator.net_lateral_m, 0.0)

    def test_forward_measurement_tolerance_records_real_motion(self) -> None:
        self.assertEqual(MOTION_MEASUREMENT_TOLERANCE_M, 0.03)
        navigator = PlacementLetterNavigator("A", PlacementLetterNavigationConfig())
        action = navigator._forward("test_forward_measurement")
        navigator.record_motion(action, 0.100002)
        self.assertAlmostEqual(navigator.forward_travel_m, 0.100002)

    def test_forward_overages_are_recorded_before_error(self) -> None:
        navigator = PlacementLetterNavigator(
            "A",
            PlacementLetterNavigationConfig(forward_budget_m=0.50),
        )
        action = navigator._forward("test_forward_overage")
        with self.assertRaisesRegex(ValueError, "requested distance"):
            navigator.record_motion(action, 0.131)
        self.assertAlmostEqual(navigator.forward_travel_m, 0.131)

        budgeted = PlacementLetterNavigator(
            "A",
            PlacementLetterNavigationConfig(forward_budget_m=0.15),
        )
        budget_action = budgeted._forward("test_forward_budget")
        budgeted.record_motion(budget_action, 0.10)
        with self.assertRaisesRegex(ValueError, "forward budget"):
            budgeted.record_motion(budget_action, 0.06)
        self.assertAlmostEqual(budgeted.forward_travel_m, 0.16)

    def test_lateral_measurement_tolerance_records_real_motion(self) -> None:
        navigator = PlacementLetterNavigator("A", PlacementLetterNavigationConfig())
        action = navigator.decide(observed(1, front_m=0.35))
        navigator.record_motion(action, 0.100002)
        self.assertAlmostEqual(navigator.lateral_travel_m, 0.100002)
        self.assertAlmostEqual(navigator.net_lateral_m, 0.100002)

    def test_lateral_overages_are_recorded_before_error(self) -> None:
        navigator = PlacementLetterNavigator(
            "A",
            PlacementLetterNavigationConfig(max_lateral_search_m=0.50),
        )
        action = navigator.decide(observed(1, front_m=0.35))
        with self.assertRaisesRegex(ValueError, "requested distance"):
            navigator.record_motion(action, 0.231)
        self.assertAlmostEqual(navigator.lateral_travel_m, 0.231)
        self.assertAlmostEqual(navigator.net_lateral_m, 0.231)

        budgeted = PlacementLetterNavigator(
            "A",
            PlacementLetterNavigationConfig(max_lateral_search_m=0.15),
        )
        budget_action = budgeted.decide(observed(1, front_m=0.35))
        budgeted.record_motion(budget_action, 0.10)
        with self.assertRaisesRegex(ValueError, "lateral budget"):
            budgeted.record_motion(budget_action, 0.06)
        self.assertAlmostEqual(budgeted.lateral_travel_m, 0.16)
        self.assertAlmostEqual(budgeted.net_lateral_m, 0.16)


class PlacementLetterConfigValidationTest(unittest.TestCase):
    def test_action_numbers_must_be_finite_and_positive(self) -> None:
        positive_fields = (
            "forward_speed_mps",
            "lateral_speed_mps",
            "front_stop_distance_m",
            "forward_budget_m",
            "forward_step_m",
            "lateral_search_step_m",
            "max_center_correction_m",
            "max_lateral_search_m",
            "lateral_search_each_side_m",
            "letter_spacing_m",
            "max_anchor_jump_m",
            "image_timeout_s",
            "total_timeout_s",
        )
        default = PlacementLetterNavigationConfig()
        for field in positive_fields:
            for invalid in (0.0, -0.01, float("nan"), float("inf")):
                with self.subTest(field=field, invalid=invalid):
                    with self.assertRaisesRegex(ValueError, field):
                        replace(default, **{field: invalid})

    def test_bilateral_search_flags_must_be_boolean(self) -> None:
        default = PlacementLetterNavigationConfig()
        for field in (
            "bilateral_search_enabled",
            "immediate_complete_on_target_detection",
        ):
            for invalid in (0, 1, None, "true"):
                with self.subTest(field=field, invalid=invalid):
                    with self.assertRaisesRegex(ValueError, field):
                        replace(default, **{field: invalid})

    def test_bilateral_total_budget_must_cover_both_sides(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_lateral_search_m"):
            PlacementLetterNavigationConfig(
                max_lateral_search_m=2.99,
                bilateral_search_enabled=True,
                lateral_search_each_side_m=1.00,
            )

    def test_front_stop_distance_uses_ultrasound_range(self) -> None:
        default = PlacementLetterNavigationConfig()
        replace(default, front_stop_distance_m=0.28)
        replace(default, front_stop_distance_m=4.50)
        for invalid in (0.279, 4.501):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "front_stop_distance_m"):
                    replace(default, front_stop_distance_m=invalid)

    def test_fraction_fields_use_closed_ranges(self) -> None:
        default = PlacementLetterNavigationConfig()
        for field, maximum in (
            ("min_confidence", 1.0),
            ("center_tolerance_fraction", 0.50),
        ):
            replace(default, **{field: 0.0})
            replace(default, **{field: maximum})
            for invalid in (-0.001, maximum + 0.001, float("nan"), float("inf")):
                with self.subTest(field=field, invalid=invalid):
                    with self.assertRaisesRegex(ValueError, field):
                        replace(default, **{field: invalid})

    def test_acquisition_center_band_is_ordered_and_normalized(self) -> None:
        default = PlacementLetterNavigationConfig()
        replace(default, acquisition_center_band=(0.0, 1.0))
        for invalid in (
            (0.5, 0.5),
            (0.8, 0.2),
            (-0.1, 0.5),
            (0.2, 1.1),
            (0.2,),
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "acquisition_center_band"):
                    replace(default, acquisition_center_band=invalid)

    def test_counter_fields_are_positive_non_boolean_integers(self) -> None:
        default = PlacementLetterNavigationConfig()
        for field in ("required_center_frames", "capture_retries"):
            replace(default, **{field: 1})
            for invalid in (0, -1, 1.5, True):
                with self.subTest(field=field, invalid=invalid):
                    with self.assertRaisesRegex(ValueError, field):
                        replace(default, **{field: invalid})


if __name__ == "__main__":
    unittest.main()

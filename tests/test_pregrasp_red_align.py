from __future__ import annotations

import contextlib
import io
import math
import unittest
from contextlib import ExitStack
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import mission_lite3.pregrasp_red_align as pregrasp_red_align
from mission_lite3 import run_pregrasp_align
from mission_lite3.arm.lite_arm import ArmTaskResult
from mission_lite3.arm.runtime import strip_detection
from mission_lite3.config_loader import load_config
from mission_lite3.mission import LargeQuadrupedMission, MissionAbort
from mission_lite3.pregrasp_red_align import (
    AlignAction,
    DEFAULT_ROI,
    FrameObservation,
    REFERENCE_LINEAR_SIZE_PX,
    RedTarget,
    choose_alignment_action,
    plan_strafe_pose_correction,
    select_loose_target,
    select_strict_target,
)


FRAME_SIZE = (1280, 720)
ACTION_CONFIG = {
    "roi": DEFAULT_ROI,
    "reference_linear_size_px": REFERENCE_LINEAR_SIZE_PX,
    "linear_size_tolerance": 0.30,
    "strafe_speed_mps": 0.08,
    "min_pulse_seconds": 0.15,
    "max_pulse_seconds": 1.00,
    "horizontal_error_strafe_gain_m_per_px": 0.00080,
}
OBSERVER_CONFIG = {
    "colors": {
        "red": [
            {"lower": [0, 80, 60], "upper": [8, 255, 255]},
            {"lower": [170, 80, 60], "upper": [179, 255, 255]},
        ]
    },
    "geometry": {"min_area_px": 1200},
}


def make_target(
    *,
    center_px: tuple[float, float] = (640.0, 520.0),
    linear_size_px: float = REFERENCE_LINEAR_SIZE_PX,
    area_px: float | None = None,
    track_id: int | None = None,
    stable: bool = True,
    confidence: float = 0.0,
    source: str = "strict",
) -> RedTarget:
    return RedTarget(
        source=source,
        center_px=center_px,
        area_px=linear_size_px**2 if area_px is None else area_px,
        frame_size=FRAME_SIZE,
        track_id=track_id,
        stable=stable,
        confidence=confidence,
    )


def make_tracked_strip(
    *,
    color: str = "red",
    track_id: int = 7,
    center_px: tuple[float, float] = (4.0, 3.0),
    area_px: float = 16.0,
    stable: bool = True,
    confidence: float = 0.9,
    box: tuple[tuple[float, float], ...] = (
        (2.0, 1.0),
        (6.0, 1.0),
        (6.0, 5.0),
        (2.0, 5.0),
    ),
) -> SimpleNamespace:
    return SimpleNamespace(
        color=color,
        track_id=track_id,
        center_px=center_px,
        area_px=area_px,
        stable=stable,
        confidence=confidence,
        box=box,
    )


class PregraspTargetSelectionTests(unittest.TestCase):
    def test_default_roi_matches_pregrasp_alignment_region(self) -> None:
        self.assertEqual(DEFAULT_ROI, (0.42, 0.55, 0.58, 0.85))

    def test_red_target_is_immutable_and_exposes_linear_size(self) -> None:
        target = make_target(area_px=144.0)

        self.assertEqual(target.linear_size_px, 12.0)
        with self.assertRaises(FrozenInstanceError):
            target.area_px = 225.0  # type: ignore[misc]

    def test_red_target_rejects_invalid_center(self) -> None:
        invalid_centers = (
            (math.nan, 100.0),
            (100.0, math.nan),
            (math.inf, 100.0),
            (100.0, -math.inf),
            (100.0,),
            (100.0, 200.0, 300.0),
            None,
            (True, 100.0),
            (100.0, False),
            (-0.1, 100.0),
            (100.0, -0.1),
            (float(FRAME_SIZE[0]), 100.0),
            (100.0, float(FRAME_SIZE[1])),
        )
        for invalid_center in invalid_centers:
            with self.subTest(center_px=invalid_center):
                with self.assertRaises(ValueError):
                    RedTarget(
                        source="strict",
                        center_px=invalid_center,  # type: ignore[arg-type]
                        area_px=144.0,
                        frame_size=FRAME_SIZE,
                    )

    def test_red_target_rejects_invalid_source(self) -> None:
        for invalid_source in ("", "red", "STRICT", None, True):
            with self.subTest(source=invalid_source):
                with self.assertRaises(ValueError):
                    RedTarget(
                        source=invalid_source,  # type: ignore[arg-type]
                        center_px=(640.0, 520.0),
                        area_px=144.0,
                        frame_size=FRAME_SIZE,
                    )

    def test_red_target_rejects_invalid_area(self) -> None:
        for invalid_area in (0.0, -1.0, math.nan, math.inf, True):
            with self.subTest(area_px=invalid_area):
                with self.assertRaises(ValueError):
                    RedTarget(
                        source="strict",
                        center_px=(640.0, 520.0),
                        area_px=invalid_area,
                        frame_size=FRAME_SIZE,
                    )

    def test_red_target_rejects_invalid_frame_size(self) -> None:
        invalid_sizes = (
            (0, 720),
            (1280, 0),
            (-1, 720),
            (1280, -1),
            (1280.0, 720),
            (1280,),
            "1280x720",
        )
        for invalid_size in invalid_sizes:
            with self.subTest(frame_size=invalid_size):
                with self.assertRaises(ValueError):
                    RedTarget(
                        source="strict",
                        center_px=(640.0, 520.0),
                        area_px=144.0,
                        frame_size=invalid_size,  # type: ignore[arg-type]
                    )

    def test_selects_strict_target_inside_roi_before_outside_target(self) -> None:
        outside = make_target(
            center_px=(250.0, 520.0),
            track_id=1,
            stable=True,
            confidence=1.0,
        )
        inside = make_target(
            center_px=(630.0, 535.0),
            linear_size_px=105.0,
            track_id=2,
        )

        selected = select_strict_target([outside, inside], DEFAULT_ROI)

        self.assertIs(selected, inside)

    def test_strict_roi_ranking_prefers_reference_linear_size_first(self) -> None:
        closest_size = make_target(
            linear_size_px=REFERENCE_LINEAR_SIZE_PX,
            track_id=1,
            stable=False,
            confidence=0.1,
        )
        otherwise_stronger = make_target(
            linear_size_px=REFERENCE_LINEAR_SIZE_PX + 1.0,
            track_id=2,
            stable=True,
            confidence=1.0,
        )

        selected = select_strict_target([otherwise_stronger, closest_size], DEFAULT_ROI)

        self.assertIs(selected, closest_size)

    def test_strict_roi_ranking_preserves_subnanopixel_size_difference(self) -> None:
        closer = make_target(
            linear_size_px=REFERENCE_LINEAR_SIZE_PX + 1e-10,
            track_id=1,
            stable=False,
            confidence=0.1,
        )
        farther_but_stronger = make_target(
            linear_size_px=REFERENCE_LINEAR_SIZE_PX + 4e-10,
            track_id=2,
            stable=True,
            confidence=1.0,
        )
        closer_error = abs(closer.linear_size_px - REFERENCE_LINEAR_SIZE_PX)
        farther_error = abs(farther_but_stronger.linear_size_px - REFERENCE_LINEAR_SIZE_PX)
        self.assertLess(closer_error, farther_error)
        self.assertLess(farther_error - closer_error, 1e-9)

        selected = select_strict_target([farther_but_stronger, closer], DEFAULT_ROI)

        self.assertIs(selected, closer)

    def test_strict_roi_ranking_uses_stability_then_confidence_then_area(self) -> None:
        size_below = 90.0
        size_above = 2.0 * REFERENCE_LINEAR_SIZE_PX - size_below
        stable = make_target(
            linear_size_px=size_below,
            track_id=1,
            stable=True,
            confidence=0.1,
        )
        unstable = make_target(
            linear_size_px=size_above,
            track_id=2,
            stable=False,
            confidence=1.0,
        )
        self.assertIs(select_strict_target([unstable, stable], DEFAULT_ROI), stable)

        high_confidence = make_target(
            linear_size_px=size_below,
            track_id=3,
            stable=True,
            confidence=0.9,
        )
        low_confidence = make_target(
            linear_size_px=size_above,
            track_id=4,
            stable=True,
            confidence=0.2,
        )
        self.assertIs(
            select_strict_target([low_confidence, high_confidence], DEFAULT_ROI),
            high_confidence,
        )

        smaller_area = make_target(
            linear_size_px=size_below,
            track_id=5,
            stable=True,
            confidence=0.9,
        )
        larger_area = make_target(
            linear_size_px=size_above,
            track_id=6,
            stable=True,
            confidence=0.9,
        )
        self.assertGreater(larger_area.area_px, smaller_area.area_px)
        self.assertIs(
            select_strict_target([smaller_area, larger_area], DEFAULT_ROI),
            larger_area,
        )

    def test_selects_target_nearest_roi_when_all_targets_are_outside(self) -> None:
        nearer_left = make_target(center_px=(550.0, 500.0), area_px=4000.0, track_id=1)
        farther_below = make_target(center_px=(640.0, 660.0), area_px=16000.0, track_id=2)

        selected = select_strict_target([farther_below, nearer_left], DEFAULT_ROI)

        self.assertIs(selected, nearer_left)

    def test_keeps_locked_track_when_it_is_still_visible(self) -> None:
        locked_outside = make_target(center_px=(100.0, 100.0), track_id=7)
        preferred_inside = make_target(center_px=(640.0, 520.0), track_id=8)

        selected = select_strict_target(
            [preferred_inside, locked_outside],
            DEFAULT_ROI,
            locked_track_id=7,
        )

        self.assertIs(selected, locked_outside)

    def test_missing_locked_track_falls_back_to_roi_selection(self) -> None:
        outside = make_target(center_px=(100.0, 100.0), track_id=7)
        inside = make_target(center_px=(640.0, 520.0), track_id=8)

        selected = select_strict_target(
            [outside, inside],
            DEFAULT_ROI,
            locked_track_id=99,
        )

        self.assertIs(selected, inside)

    def test_loose_recovery_selects_region_nearest_roi_not_largest(self) -> None:
        larger_left = make_target(
            center_px=(72.3, 516.6),
            area_px=12999.0,
            source="loose",
        )
        smaller_central = make_target(
            center_px=(632.4, 534.1),
            area_px=9086.0,
            source="loose",
        )
        self.assertGreater(larger_left.area_px, smaller_central.area_px)

        selected = select_loose_target([larger_left, smaller_central], DEFAULT_ROI)

        self.assertIs(selected, smaller_central)

    def test_selectors_return_none_for_no_targets(self) -> None:
        self.assertIsNone(select_strict_target([], DEFAULT_ROI))
        self.assertIsNone(select_loose_target([], DEFAULT_ROI))

    def test_reference_linear_size_matches_measured_area(self) -> None:
        self.assertAlmostEqual(REFERENCE_LINEAR_SIZE_PX, math.sqrt(8881.0), places=3)


class PregraspObserverTests(unittest.TestCase):
    def test_grasp_mask_rejects_low_hue_obstacle_and_keeps_high_hue_block(
        self,
    ) -> None:
        detector_config = strip_detection.load_config(
            "mission_lite3/arm/runtime/strip_detector_grasp_config.json"
        )
        hsv = np.zeros((120, 240, 3), dtype=np.uint8)
        hsv[20:100, 10:110] = (5, 190, 180)
        hsv[20:100, 130:230] = (175, 115, 180)
        frame = pregrasp_red_align.cv2.cvtColor(
            hsv,
            pregrasp_red_align.cv2.COLOR_HSV2BGR,
        )

        masks = strip_detection.build_color_masks(frame, detector_config)

        self.assertEqual(
            pregrasp_red_align.cv2.countNonZero(masks["red"][:, :120]),
            0,
        )
        self.assertGreater(
            pregrasp_red_align.cv2.countNonZero(masks["red"][:, 120:]),
            0,
        )

    def observer_class(self):
        self.assertTrue(
            hasattr(pregrasp_red_align, "ArmRedObserver"),
            "ArmRedObserver must be implemented",
        )
        return pregrasp_red_align.ArmRedObserver

    def observation_class(self):
        self.assertTrue(
            hasattr(pregrasp_red_align, "FrameObservation"),
            "FrameObservation must be implemented",
        )
        return pregrasp_red_align.FrameObservation

    def test_frame_observation_is_immutable(self) -> None:
        observation_class = self.observation_class()
        observation = observation_class(
            mode="none",
            strict_targets=[],
            loose_targets=[],
            undistorted_frame=np.zeros((4, 5, 3), dtype=np.uint8),
        )

        self.assertEqual(observation.strict_targets, ())
        self.assertEqual(observation.loose_targets, ())
        with self.assertRaises(FrozenInstanceError):
            observation.mode = "strict"

    def test_frame_observation_rejects_mode_target_mismatches(self) -> None:
        observation_class = self.observation_class()
        frame = np.zeros((4, 5, 3), dtype=np.uint8)
        strict_target = RedTarget(
            source="strict",
            center_px=(2.0, 2.0),
            area_px=4.0,
            frame_size=(5, 4),
        )
        loose_target = RedTarget(
            source="loose",
            center_px=(2.0, 2.0),
            area_px=4.0,
            frame_size=(5, 4),
        )
        invalid_cases = (
            ("strict", (), ()),
            ("strict", (), (loose_target,)),
            ("strict", (strict_target,), (loose_target,)),
            ("loose", (), ()),
            ("loose", (strict_target,), ()),
            ("loose", (strict_target,), (loose_target,)),
            ("none", (strict_target,), ()),
            ("none", (), (loose_target,)),
            ("none", (strict_target,), (loose_target,)),
        )

        for mode, strict_targets, loose_targets in invalid_cases:
            with self.subTest(
                mode=mode,
                strict_count=len(strict_targets),
                loose_count=len(loose_targets),
            ):
                with self.assertRaises(ValueError):
                    observation_class(
                        mode=mode,
                        strict_targets=strict_targets,
                        loose_targets=loose_targets,
                        undistorted_frame=frame,
                    )

    def test_frame_observation_rejects_non_target_elements(self) -> None:
        observation_class = self.observation_class()
        frame = np.zeros((4, 5, 3), dtype=np.uint8)
        invalid_cases = (
            ("strict", (object(),), ()),
            ("loose", (), (object(),)),
        )

        for mode, strict_targets, loose_targets in invalid_cases:
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError):
                    observation_class(
                        mode=mode,
                        strict_targets=strict_targets,  # type: ignore[arg-type]
                        loose_targets=loose_targets,  # type: ignore[arg-type]
                        undistorted_frame=frame,
                    )

    def test_frame_observation_rejects_target_source_mismatch(self) -> None:
        observation_class = self.observation_class()
        frame = np.zeros((4, 5, 3), dtype=np.uint8)
        strict_target = RedTarget(
            source="strict",
            center_px=(2.0, 2.0),
            area_px=4.0,
            frame_size=(5, 4),
        )
        loose_target = RedTarget(
            source="loose",
            center_px=(2.0, 2.0),
            area_px=4.0,
            frame_size=(5, 4),
        )
        invalid_cases = (
            ("strict", (loose_target,), ()),
            ("loose", (), (strict_target,)),
        )

        for mode, strict_targets, loose_targets in invalid_cases:
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError):
                    observation_class(
                        mode=mode,
                        strict_targets=strict_targets,
                        loose_targets=loose_targets,
                        undistorted_frame=frame,
                    )

    def test_frame_observation_rejects_target_frame_size_mismatch(self) -> None:
        observation_class = self.observation_class()
        frame = np.zeros((4, 5, 3), dtype=np.uint8)
        strict_target = RedTarget(
            source="strict",
            center_px=(2.0, 2.0),
            area_px=4.0,
            frame_size=(6, 4),
        )
        loose_target = RedTarget(
            source="loose",
            center_px=(2.0, 2.0),
            area_px=4.0,
            frame_size=(5, 5),
        )
        invalid_cases = (
            ("strict", (strict_target,), ()),
            ("loose", (), (loose_target,)),
        )

        for mode, strict_targets, loose_targets in invalid_cases:
            with self.subTest(mode=mode):
                with self.assertRaises(ValueError):
                    observation_class(
                        mode=mode,
                        strict_targets=strict_targets,
                        loose_targets=loose_targets,
                        undistorted_frame=frame,
                    )

    def test_undistortion_runs_before_strict_detection(self) -> None:
        observer_class = self.observer_class()
        raw_frame = np.zeros((6, 8, 3), dtype=np.uint8)
        undistorted_frame = np.full_like(raw_frame, 17)
        call_order = []

        def apply(frame):
            self.assertIs(frame, raw_frame)
            call_order.append("undistort")
            return undistorted_frame

        def detect_candidates(frame, config):
            self.assertIs(frame, undistorted_frame)
            self.assertIs(config, OBSERVER_CONFIG)
            call_order.append("detect")
            return ["candidate"], {"red": np.zeros((6, 8), dtype=np.uint8)}

        def update(candidates):
            self.assertEqual(candidates, ["candidate"])
            call_order.append("track")
            return [make_tracked_strip()]

        observer = observer_class(
            detector_config=OBSERVER_CONFIG,
            undistorter=SimpleNamespace(apply=apply),
            detect_candidates_fn=detect_candidates,
            tracker=SimpleNamespace(update=update),
            loose_detector_fn=lambda *_args: self.fail("loose detector must not run"),
        )

        observation = observer.observe(raw_frame)

        self.assertEqual(call_order, ["undistort", "detect", "track"])
        self.assertIs(observation.undistorted_frame, undistorted_frame)
        self.assertEqual(observation.mode, "strict")

    def test_strict_empty_runs_loose_on_undistorted_frame(self) -> None:
        observer_class = self.observer_class()
        raw_frame = np.zeros((6, 8, 3), dtype=np.uint8)
        undistorted_frame = np.full_like(raw_frame, 23)
        loose_target = RedTarget(
            source="loose",
            center_px=(4.0, 3.0),
            area_px=16.0,
            frame_size=(8, 6),
            bbox_px=(2, 1, 4, 4),
        )
        call_order = []

        def apply(_frame):
            call_order.append("undistort")
            return undistorted_frame

        def detect_candidates(frame, _config):
            self.assertIs(frame, undistorted_frame)
            call_order.append("detect")
            return [], {}

        def update(candidates):
            self.assertEqual(candidates, [])
            call_order.append("track")
            return []

        def detect_loose(frame, red_ranges, min_area_px):
            self.assertIs(frame, undistorted_frame)
            self.assertEqual(red_ranges, OBSERVER_CONFIG["colors"]["red"])
            self.assertEqual(min_area_px, 300.0)
            call_order.append("loose")
            return [loose_target]

        observer = observer_class(
            detector_config=OBSERVER_CONFIG,
            undistorter=SimpleNamespace(apply=apply),
            detect_candidates_fn=detect_candidates,
            tracker=SimpleNamespace(update=update),
            loose_detector_fn=detect_loose,
        )

        observation = observer.observe(raw_frame)

        self.assertEqual(call_order, ["undistort", "detect", "track", "loose"])
        self.assertEqual(observation.mode, "loose")
        self.assertEqual(observation.strict_targets, ())
        self.assertEqual(observation.loose_targets, (loose_target,))

    def test_far_upper_strict_target_does_not_hide_near_loose_target(self) -> None:
        observer_class = self.observer_class()
        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        far_strip = make_tracked_strip(center_px=(100.0, 20.0))
        near_target = RedTarget(
            source="loose",
            center_px=(80.0, 75.0),
            area_px=3600.0,
            frame_size=(120, 100),
            bbox_px=(50, 50, 60, 50),
        )
        observer = observer_class(
            detector_config=OBSERVER_CONFIG,
            undistorter=SimpleNamespace(apply=lambda current: current),
            detect_candidates_fn=lambda _frame, _config: (["candidate"], {}),
            tracker=SimpleNamespace(update=lambda _candidates: [far_strip]),
            loose_detector_fn=lambda _frame, _ranges, _area: [near_target],
        )

        observation = observer.observe(frame)

        self.assertEqual(observation.mode, "loose")
        self.assertEqual(observation.strict_targets, ())
        self.assertEqual(observation.loose_targets, (near_target,))

    def test_strict_targets_skip_loose_and_preserve_tracking_fields(self) -> None:
        observer_class = self.observer_class()
        frame = np.zeros((6, 8, 3), dtype=np.uint8)
        red_strip = make_tracked_strip()
        green_strip = make_tracked_strip(color="green", track_id=8)
        loose_detector = Mock(side_effect=AssertionError("loose detector must not run"))
        observer = observer_class(
            detector_config=OBSERVER_CONFIG,
            undistorter=SimpleNamespace(apply=lambda current: current),
            detect_candidates_fn=lambda _frame, _config: (["candidate"], {}),
            tracker=SimpleNamespace(update=lambda _candidates: [green_strip, red_strip]),
            loose_detector_fn=loose_detector,
        )

        observation = observer.observe(frame)

        self.assertEqual(observation.mode, "strict")
        self.assertEqual(observation.loose_targets, ())
        self.assertEqual(len(observation.strict_targets), 1)
        target = observation.strict_targets[0]
        self.assertEqual(target.source, "strict")
        self.assertEqual(target.track_id, red_strip.track_id)
        self.assertEqual(target.stable, red_strip.stable)
        self.assertEqual(target.confidence, red_strip.confidence)
        self.assertEqual(target.box, red_strip.box)
        self.assertEqual(target.frame_size, (8, 6))
        loose_detector.assert_not_called()

    def test_loose_mode_does_not_masquerade_as_strict(self) -> None:
        observer_class = self.observer_class()
        frame = np.zeros((6, 8, 3), dtype=np.uint8)
        loose_target = RedTarget(
            source="loose",
            center_px=(4.0, 3.0),
            area_px=16.0,
            frame_size=(8, 6),
        )
        observer = observer_class(
            detector_config=OBSERVER_CONFIG,
            undistorter=SimpleNamespace(apply=lambda current: current),
            detect_candidates_fn=lambda _frame, _config: ([], {}),
            tracker=SimpleNamespace(update=lambda _candidates: []),
            loose_detector_fn=lambda _frame, _ranges, _area: [loose_target],
        )

        observation = observer.observe(frame)

        self.assertEqual(observation.mode, "loose")
        self.assertEqual(observation.strict_targets, ())
        self.assertEqual(observation.loose_targets, (loose_target,))

    def test_empty_strict_and_loose_targets_return_none_mode(self) -> None:
        observer_class = self.observer_class()
        frame = np.zeros((6, 8, 3), dtype=np.uint8)
        observer = observer_class(
            detector_config=OBSERVER_CONFIG,
            undistorter=SimpleNamespace(apply=lambda current: current),
            detect_candidates_fn=lambda _frame, _config: ([], {}),
            tracker=SimpleNamespace(update=lambda _candidates: []),
            loose_detector_fn=lambda _frame, _ranges, _area: [],
        )

        observation = observer.observe(frame)

        self.assertEqual(observation.mode, "none")
        self.assertEqual(observation.strict_targets, ())
        self.assertEqual(observation.loose_targets, ())

    def test_calibration_frame_mismatch_value_error_propagates(self) -> None:
        observer_class = self.observer_class()

        def raise_mismatch(_frame):
            raise ValueError("frame resolution does not match calibration")

        observer = observer_class(
            detector_config=OBSERVER_CONFIG,
            undistorter=SimpleNamespace(apply=raise_mismatch),
            detect_candidates_fn=lambda *_args: self.fail(
                "strict detection must not run after undistortion failure"
            ),
            tracker=SimpleNamespace(update=lambda *_args: []),
        )

        with self.assertRaisesRegex(
            ValueError,
            "frame resolution does not match calibration",
        ):
            observer.observe(np.zeros((6, 8, 3), dtype=np.uint8))

    def test_real_loose_hsv_detection_returns_two_connected_regions(self) -> None:
        observer_class = self.observer_class()
        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        frame[10:30, 10:30] = (0, 0, 255)
        frame[50:75, 70:95] = (0, 0, 255)
        observer = observer_class(
            detector_config=OBSERVER_CONFIG,
            undistorter=SimpleNamespace(apply=lambda current: current),
            detect_candidates_fn=lambda _frame, _config: ([], {}),
            tracker=SimpleNamespace(update=lambda _candidates: []),
        )

        observation = observer.observe(frame)

        self.assertEqual(observation.mode, "loose")
        self.assertEqual(len(observation.loose_targets), 2)
        self.assertEqual(
            sorted(target.area_px for target in observation.loose_targets),
            [400.0, 625.0],
        )
        self.assertEqual(
            sorted(target.bbox_px for target in observation.loose_targets),
            [(10, 10, 20, 20), (70, 50, 25, 25)],
        )
        for target in observation.loose_targets:
            self.assertEqual(target.source, "loose")
            self.assertEqual(target.frame_size, (120, 100))

    def test_annotation_does_not_modify_undistorted_frame(self) -> None:
        observation_class = self.observation_class()
        self.assertTrue(
            hasattr(pregrasp_red_align, "annotate_observation"),
            "annotate_observation must be implemented",
        )
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        target = RedTarget(
            source="loose",
            center_px=(20.0, 20.0),
            area_px=400.0,
            frame_size=(100, 80),
            bbox_px=(10, 10, 20, 20),
        )
        observation = observation_class(
            mode="loose",
            strict_targets=(),
            loose_targets=(target,),
            undistorted_frame=frame,
        )
        before = frame.copy()

        annotated = pregrasp_red_align.annotate_observation(
            observation,
            DEFAULT_ROI,
            selected=target,
            action=AlignAction("strafe_left", vy=0.08),
        )

        np.testing.assert_array_equal(frame, before)
        self.assertIsNot(annotated, frame)
        self.assertTrue(np.any(annotated != before))

    def test_from_files_reuses_runtime_loaders_and_factories(self) -> None:
        observer_class = self.observer_class()
        calibration = {"image_size": [12, 10]}
        undistorter = SimpleNamespace(apply=lambda frame: frame)
        tracker = SimpleNamespace(update=Mock(return_value=[]))
        frame = np.zeros((10, 12, 3), dtype=np.uint8)

        with ExitStack() as stack:
            load_config = stack.enter_context(
                patch(
                    "mission_lite3.arm.runtime.strip_detection.load_config",
                    return_value=OBSERVER_CONFIG,
                )
            )
            detect_candidates = stack.enter_context(
                patch(
                    "mission_lite3.arm.runtime.strip_detection.detect_candidates",
                    return_value=([], {}),
                )
            )
            tracker_class = stack.enter_context(
                patch(
                    "mission_lite3.arm.runtime.strip_detection.StripTracker",
                    return_value=tracker,
                )
            )
            load_calibration = stack.enter_context(
                patch(
                    "mission_lite3.arm.runtime.camera_calibration.load_calibration",
                    return_value=calibration,
                )
            )
            undistorter_class = stack.enter_context(
                patch(
                    "mission_lite3.arm.runtime.camera_calibration.FrameUndistorter",
                    return_value=undistorter,
                )
            )
            observer = observer_class.from_files("detector.json", "calibration.json")
            observation = observer.observe(frame)

        load_config.assert_called_once_with("detector.json")
        load_calibration.assert_called_once_with("calibration.json")
        tracker_class.assert_called_once_with(OBSERVER_CONFIG)
        undistorter_class.assert_called_once_with(calibration)
        detect_candidates.assert_called_once_with(frame, OBSERVER_CONFIG)
        tracker.update.assert_called_once_with([])
        self.assertEqual(observation.mode, "none")


class PregraspActionPlanningTests(unittest.TestCase):
    def test_align_action_is_immutable(self) -> None:
        action = AlignAction("hold", reason="target_horizontally_aligned")

        with self.assertRaises(FrozenInstanceError):
            action.vy = 1.0  # type: ignore[misc]

    def test_align_action_has_no_forward_axis(self) -> None:
        action = AlignAction("hold")

        self.assertFalse(hasattr(action, "vx"))

    def test_pose_correction_cancels_forward_and_yaw_drift(self) -> None:
        config = dict(pregrasp_red_align.DEFAULT_ALIGN_CONFIG)
        correction = plan_strafe_pose_correction(
            (0.0, 0.0, math.pi / 2.0),
            (0.01, 0.02, math.pi / 2.0 + math.radians(2.0)),
            config,
        )

        self.assertAlmostEqual(correction.forward_drift_m, 0.02)
        self.assertAlmostEqual(correction.vx, -0.02)
        self.assertLess(correction.wz, 0.0)
        self.assertAlmostEqual(correction.wz, -1.2 * math.radians(2.0))

    def test_robot_near_box_strafe_corrects_forward_drift(self) -> None:
        config = load_config()["pregrasp_red_align"]
        correction = plan_strafe_pose_correction(
            (0.0, 0.0, 0.0),
            (-0.02, 0.0, math.radians(2.0)),
            config,
        )

        self.assertAlmostEqual(correction.forward_drift_m, -0.02)
        self.assertAlmostEqual(correction.vx, 0.02)
        self.assertLess(correction.wz, 0.0)

    def test_horizontal_error_is_corrected_regardless_of_size(self) -> None:
        target = make_target(center_px=(300.0, 520.0), linear_size_px=50.0)

        action = choose_alignment_action(target, ACTION_CONFIG)

        self.assertEqual(action.name, "strafe_left")
        self.assertGreater(action.vy, 0.0)

    def test_horizontal_error_is_corrected_regardless_of_vertical_position(self) -> None:
        target = make_target(
            center_px=(300.0, 650.0),
            linear_size_px=60.0,
        )

        action = choose_alignment_action(target, ACTION_CONFIG)

        self.assertEqual(action.name, "strafe_left")
        self.assertGreater(action.vy, 0.0)

    def test_left_target_commands_left_strafe(self) -> None:
        target = make_target(center_px=(300.0, 520.0))

        action = choose_alignment_action(target, ACTION_CONFIG)

        self.assertEqual(action.name, "strafe_left")
        self.assertAlmostEqual(action.vy, 0.08)
        self.assertGreater(action.pulse_seconds, 0.15)
        self.assertLessEqual(action.pulse_seconds, 1.0)

    def test_right_target_commands_right_strafe(self) -> None:
        target = make_target(center_px=(950.0, 520.0))

        action = choose_alignment_action(target, ACTION_CONFIG)

        self.assertEqual(action.name, "strafe_right")
        self.assertAlmostEqual(action.vy, -0.08)
        self.assertGreater(action.pulse_seconds, 0.15)
        self.assertLessEqual(action.pulse_seconds, 1.0)

    def test_larger_horizontal_error_uses_longer_pulse(self) -> None:
        near = make_target(center_px=(520.0, 520.0))
        far = make_target(center_px=(200.0, 520.0))

        near_action = choose_alignment_action(near, ACTION_CONFIG)
        far_action = choose_alignment_action(far, ACTION_CONFIG)

        self.assertEqual(near_action.name, "strafe_left")
        self.assertEqual(far_action.name, "strafe_left")
        self.assertGreater(far_action.pulse_seconds, near_action.pulse_seconds)

    def test_horizontal_error_gain_maps_pixel_delta_to_strafe_distance(self) -> None:
        near = make_target(center_px=(510.0, 520.0))
        farther = make_target(center_px=(460.0, 520.0))

        near_action = choose_alignment_action(near, ACTION_CONFIG)
        farther_action = choose_alignment_action(farther, ACTION_CONFIG)

        speed = ACTION_CONFIG["strafe_speed_mps"]
        gain = ACTION_CONFIG["horizontal_error_strafe_gain_m_per_px"]
        self.assertAlmostEqual(
            near_action.pulse_seconds * speed,
            (0.42 * FRAME_SIZE[0] - 510.0) * gain,
        )
        self.assertAlmostEqual(
            farther_action.pulse_seconds * speed,
            (0.42 * FRAME_SIZE[0] - 460.0) * gain,
        )
        self.assertGreater(
            farther_action.pulse_seconds * speed,
            near_action.pulse_seconds * speed,
        )

    def test_small_target_does_not_command_fore_aft_motion(self) -> None:
        target = make_target(center_px=(640.0, 520.0), linear_size_px=65.0)

        action = choose_alignment_action(target, ACTION_CONFIG)

        self.assertEqual(action.name, "hold")
        self.assertEqual(action.vy, 0.0)
        self.assertEqual(action.reason, "target_horizontally_aligned")

    def test_large_target_does_not_command_fore_aft_motion(self) -> None:
        target = make_target(center_px=(640.0, 520.0), linear_size_px=123.0)

        action = choose_alignment_action(target, ACTION_CONFIG)

        self.assertEqual(action.name, "hold")
        self.assertEqual(action.vy, 0.0)
        self.assertEqual(action.reason, "target_horizontally_aligned")

    def test_vertical_position_does_not_command_motion(self) -> None:
        above = make_target(center_px=(640.0, 380.0))
        below = make_target(center_px=(640.0, 630.0))

        above_action = choose_alignment_action(above, ACTION_CONFIG)
        below_action = choose_alignment_action(below, ACTION_CONFIG)

        self.assertEqual(above_action.name, "hold")
        self.assertEqual(below_action.name, "hold")
        self.assertEqual(above_action.vy, 0.0)
        self.assertEqual(below_action.vy, 0.0)

    def test_matching_target_returns_hold(self) -> None:
        target = make_target(center_px=(640.0, 520.0))

        action = choose_alignment_action(target, ACTION_CONFIG)

        self.assertEqual(action.name, "hold")
        self.assertEqual(action.vy, 0.0)
        self.assertEqual(action.reason, "target_horizontally_aligned")

    def test_horizontal_alignment_boundaries_are_inclusive(self) -> None:
        left, _top, right, _bottom = DEFAULT_ROI
        boundary_targets = (
            make_target(
                center_px=(left * FRAME_SIZE[0], 10.0),
                linear_size_px=40.0,
            ),
            make_target(
                center_px=(right * FRAME_SIZE[0], FRAME_SIZE[1] - 10.0),
                linear_size_px=180.0,
            ),
        )

        actions = [choose_alignment_action(target, ACTION_CONFIG) for target in boundary_targets]

        self.assertEqual([action.name for action in actions], ["hold", "hold"])

    def test_size_and_vertical_geometry_conflict_does_not_block_lateral_alignment(self) -> None:
        too_small_but_below_roi = make_target(
            center_px=(640.0, 650.0),
            linear_size_px=60.0,
        )

        action = choose_alignment_action(too_small_but_below_roi, ACTION_CONFIG)

        self.assertEqual(action.name, "hold")
        self.assertEqual(action.reason, "target_horizontally_aligned")
        self.assertEqual(action.vy, 0.0)

    def test_opposite_size_and_vertical_geometry_conflict_is_also_observation_only(self) -> None:
        too_large_but_above_roi = make_target(
            center_px=(640.0, 350.0),
            linear_size_px=130.0,
        )

        action = choose_alignment_action(too_large_but_above_roi, ACTION_CONFIG)

        self.assertEqual(action.name, "hold")
        self.assertEqual(action.reason, "target_horizontally_aligned")
        self.assertEqual(action.vy, 0.0)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleep_calls = []

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)
        self.advance(seconds)


class SequenceCamera:
    def __init__(self, frames) -> None:
        self.frames = list(frames)
        self.read_count = 0
        self.release_count = 0
        self.reconnect_reasons = []

    def read(self):
        self.read_count += 1
        if not self.frames:
            return None
        frame = self.frames.pop(0)
        if isinstance(frame, BaseException):
            raise frame
        return frame

    def release(self) -> None:
        self.release_count += 1

    def reconnect(self, reason: str) -> None:
        self.reconnect_reasons.append(reason)


class SequenceObserver:
    def __init__(self, observations) -> None:
        self.observations = list(observations)
        self.frames = []

    def observe(self, frame):
        self.frames.append(frame)
        observation = self.observations.pop(0)
        if isinstance(observation, BaseException):
            raise observation
        return observation


class RecordingMotion:
    def __init__(
        self,
        clock: FakeClock,
        hold_error: Exception | None = None,
        stop_errors=(),
    ) -> None:
        self.clock = clock
        self.hold_error = hold_error
        self.stop_errors = list(stop_errors)
        self.events = []

    def hold_velocity(
        self,
        vx: float,
        vy: float,
        wz: float,
        duration: float,
    ) -> None:
        self.events.append(("hold_velocity", vx, vy, wz, duration))
        self.clock.advance(duration)
        if self.hold_error is not None:
            raise self.hold_error

    def stop(self) -> None:
        self.events.append(("stop",))
        if self.stop_errors:
            raise self.stop_errors.pop(0)


class OdomRecordingMotion(RecordingMotion):
    def __init__(self, clock: FakeClock, lateral_scale: float) -> None:
        super().__init__(clock)
        self.pose = [0.0, 0.0, 0.0]
        self.lateral_scale = lateral_scale

    def hold_velocity(
        self,
        vx: float,
        vy: float,
        wz: float,
        duration: float,
    ) -> None:
        super().hold_velocity(vx, vy, wz, duration)
        self.pose[0] += vx * duration
        self.pose[1] += vy * duration * self.lateral_scale
        self.pose[2] += wz * duration


class RecordingWriter:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.created = []
        self.json_writes = []
        self.image_writes = []

    def create_run_dir(self, root, run_name):
        if self.fail:
            raise OSError("log directory unavailable")
        run_dir = root / run_name
        self.created.append(run_dir)
        return run_dir

    def write_json(self, path, payload) -> None:
        if self.fail:
            raise OSError("json write failed")
        self.json_writes.append((path, payload))

    def write_image(self, path, image) -> None:
        if self.fail:
            raise OSError("image write failed")
        self.image_writes.append((path, image))


class DeadlineAdvancingWriter(RecordingWriter):
    def __init__(self, clock: FakeClock, seconds: float) -> None:
        super().__init__()
        self.clock = clock
        self.seconds = seconds

    def write_json(self, path, payload) -> None:
        super().write_json(path, payload)
        if path.name.startswith("decision_"):
            self.clock.advance(self.seconds)


def make_observation(
    mode: str,
    targets: tuple[RedTarget, ...] = (),
) -> FrameObservation:
    frame = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), dtype=np.uint8)
    return FrameObservation(
        mode=mode,
        strict_targets=targets if mode == "strict" else (),
        loose_targets=targets if mode == "loose" else (),
        undistorted_frame=frame,
    )


class PregraspRedAlignerLoopTests(unittest.TestCase):
    def aligner_class(self):
        aligner_class = getattr(pregrasp_red_align, "PregraspRedAligner", None)
        self.assertIsNotNone(
            aligner_class,
            "PregraspRedAligner must be implemented",
        )
        self.assertTrue(
            hasattr(aligner_class, "run"),
            "PregraspRedAligner.run must be implemented",
        )
        return aligner_class

    def build_aligner(
        self,
        observations,
        *,
        config=None,
        camera_frames=None,
        motion=None,
        writer=None,
        pose_provider=None,
        search_origin_pose=None,
        tag_boundary_provider=None,
        front_distance_provider=None,
    ):
        clock = FakeClock()
        observation_list = list(observations)
        frames = (
            [object() for _ in observation_list]
            if camera_frames is None
            else camera_frames
        )
        camera = SequenceCamera(frames)
        observer = SequenceObserver(observation_list)
        motion = RecordingMotion(clock) if motion is None else motion
        writer = RecordingWriter() if writer is None else writer
        align_config = {"target_search_enabled": False}
        if config is not None:
            align_config.update(config)
        aligner = self.aligner_class()(
            camera=camera,
            observer=observer,
            motion=motion,
            config=align_config,
            clock=clock,
            sleep=clock.sleep,
            pose_provider=pose_provider,
            search_origin_pose=search_origin_pose,
            tag_boundary_provider=tag_boundary_provider,
            front_distance_provider=front_distance_provider,
            log_dir="unused-test-log-root",
            writer=writer,
        )
        return aligner, camera, observer, motion, clock, writer

    def test_pregrasp_aligner_exposes_run(self) -> None:
        aligner_class = getattr(pregrasp_red_align, "PregraspRedAligner", object)

        self.assertTrue(
            hasattr(aligner_class, "run"),
            "PregraspRedAligner.run must be implemented",
        )

    def test_three_strict_horizontally_aligned_frames_succeed_without_motion(
        self,
    ) -> None:
        observations = [
            make_observation(
                "strict",
                (make_target(track_id=7, center_px=(640.0, 520.0)),),
            )
            for _ in range(3)
        ]
        aligner, camera, observer, motion, _clock, _writer = self.build_aligner(
            observations
        )

        result = aligner.run()

        self.assertTrue(result.ok)
        self.assertEqual(result.reason, "aligned")
        self.assertEqual(result.pulse_count, 0)
        self.assertEqual(result.strafe_distance_m, 0.0)
        self.assertEqual(result.selected_track_id, 7)
        self.assertEqual(camera.read_count, 3)
        self.assertEqual(len(observer.frames), 3)
        self.assertFalse(
            any(event[0] == "hold_velocity" for event in motion.events)
        )

    def test_finish_stop_failure_returns_stop_failed_and_logs_result(self) -> None:
        observations = [
            make_observation(
                "strict",
                (make_target(track_id=7, center_px=(640.0, 520.0)),),
            )
            for _ in range(3)
        ]
        motion = RecordingMotion(
            FakeClock(),
            stop_errors=[RuntimeError("stop failed")],
        )
        writer = RecordingWriter()
        aligner, camera, _observer, motion, _clock, writer = (
            self.build_aligner(
                observations,
                motion=motion,
                writer=writer,
            )
        )

        result = aligner.run()

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "stop_failed")
        self.assertEqual(
            sum(event[0] == "stop" for event in motion.events),
            2,
        )
        self.assertEqual(camera.release_count, 1)
        result_payloads = [
            payload
            for path, payload in writer.json_writes
            if path.name == "result.json"
        ]
        self.assertEqual(len(result_payloads), 1)
        self.assertFalse(result_payloads[0]["ok"])
        self.assertEqual(result_payloads[0]["reason"], "stop_failed")

    def test_small_strict_target_never_drives_motion(
        self,
    ) -> None:
        observations = [
            make_observation(
                "strict",
                (
                    make_target(
                        track_id=4,
                        center_px=(640.0, 520.0),
                        linear_size_px=50.0,
                    ),
                ),
            )
            for _ in range(5)
        ]
        aligner, _camera, _observer, motion, _clock, _writer = (
            self.build_aligner(
                observations,
                config={"target_not_found_retries": 0},
            )
        )

        result = aligner.run()

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "target_not_found")
        self.assertIsNone(result.selected_track_id)
        self.assertFalse(
            any(event[0] == "hold_velocity" for event in motion.events)
        )

    def test_replay_sized_false_target_does_not_end_left_search(self) -> None:
        replay_linear_size = REFERENCE_LINEAR_SIZE_PX * 0.5855
        observations = [
            make_observation("none"),
            make_observation(
                "strict",
                (
                    make_target(
                        track_id=41,
                        center_px=(640.0, 520.0),
                        linear_size_px=replay_linear_size,
                    ),
                ),
            ),
            make_observation(
                "strict",
                (
                    make_target(
                        track_id=42,
                        center_px=(640.0, 520.0),
                        linear_size_px=REFERENCE_LINEAR_SIZE_PX * 0.80,
                    ),
                ),
            ),
        ]
        aligner, _camera, _observer, motion, _clock, _writer = (
            self.build_aligner(
                observations,
                config={
                    "acquire_only": True,
                    "no_red_frame_limit": 1,
                    "target_not_found_retries": 0,
                    "target_search_enabled": True,
                    "target_search_bilateral_enabled": False,
                    "target_search_max_distance_m": 1.0,
                    "target_search_settle_seconds": 0.0,
                },
            )
        )

        result = aligner.run()

        self.assertTrue(result.ok)
        self.assertEqual(result.reason, "target_acquired")
        self.assertEqual(result.selected_track_id, 42)
        pulses = [event for event in motion.events if event[0] == "hold_velocity"]
        self.assertEqual(len(pulses), 1)
        self.assertGreater(pulses[0][2], 0.0)

    def test_near_field_strict_target_can_be_smaller_than_loose_threshold(
        self,
    ) -> None:
        observations = [
            make_observation(
                "strict",
                (
                    make_target(
                        track_id=4,
                        center_px=(640.0, 520.0),
                        linear_size_px=70.0,
                    ),
                ),
            )
            for _ in range(3)
        ]
        aligner, _camera, _observer, motion, _clock, _writer = (
            self.build_aligner(observations)
        )

        result = aligner.run()

        self.assertTrue(result.ok)
        self.assertEqual(result.reason, "aligned")
        self.assertFalse(
            any(event[0] == "hold_velocity" for event in motion.events)
        )

    def test_locked_strict_target_uses_lower_tracking_size_threshold(self) -> None:
        observations = [
            make_observation(
                "strict",
                (
                    make_target(
                        track_id=4,
                        center_px=(320.0, 520.0),
                        linear_size_px=70.0,
                    ),
                ),
            ),
            make_observation(
                "strict",
                (
                    make_target(
                        track_id=4,
                        center_px=(540.0, 520.0),
                        linear_size_px=65.0,
                    ),
                ),
            ),
            *[
                make_observation(
                    "strict",
                    (
                        make_target(
                            track_id=4,
                            center_px=(640.0, 520.0),
                            linear_size_px=60.0,
                        ),
                    ),
                )
                for _ in range(3)
            ],
        ]
        aligner, _camera, _observer, motion, _clock, _writer = (
            self.build_aligner(observations)
        )

        result = aligner.run()

        self.assertTrue(result.ok)
        self.assertEqual(result.reason, "aligned")
        self.assertEqual(result.selected_track_id, 4)
        self.assertEqual(
            len([event for event in motion.events if event[0] == "hold_velocity"]),
            1,
        )

    def test_loose_red_moves_laterally_but_cannot_complete_alignment(self) -> None:
        observations = [
            *[
                make_observation(
                    "loose",
                    (
                        make_target(
                            source="loose",
                            center_px=(200.0, 520.0),
                        ),
                    ),
                )
                for _ in range(3)
            ],
            *[
                make_observation(
                    "strict",
                    (make_target(track_id=9, center_px=(640.0, 520.0)),),
                )
                for _ in range(3)
            ],
        ]
        aligner, camera, _observer, motion, _clock, _writer = (
            self.build_aligner(observations)
        )

        result = aligner.run()

        self.assertTrue(result.ok)
        self.assertEqual(camera.read_count, 6)
        self.assertEqual(result.pulse_count, 1)
        self.assertEqual(result.selected_track_id, 9)
        pulses = [event for event in motion.events if event[0] == "hold_velocity"]
        self.assertEqual(len(pulses), 1)
        self.assertEqual(pulses[0][:4], ("hold_velocity", 0.0, 0.08, 0.0))
        self.assertAlmostEqual(
            pulses[0][4],
            choose_alignment_action(
                make_target(source="loose", center_px=(200.0, 520.0)),
                ACTION_CONFIG,
            ).pulse_seconds,
        )

    def test_strafe_pulse_holds_reference_forward_position_and_yaw(self) -> None:
        observations = [
            make_observation(
                "strict",
                (make_target(track_id=9, center_px=(200.0, 520.0)),),
            ),
            *[
                make_observation(
                    "strict",
                    (make_target(track_id=9, center_px=(640.0, 520.0)),),
                )
                for _ in range(3)
            ],
        ]
        poses = iter(
            [
                (0.0, 0.0, 0.0),
                (0.02, 0.0, math.radians(2.0)),
            ]
        )
        aligner, _camera, _observer, motion, _clock, _writer = self.build_aligner(
            observations,
            config={"strafe_pose_hold_enabled": True},
            pose_provider=lambda: next(poses),
        )

        result = aligner.run()

        self.assertTrue(result.ok)
        pulses = [event for event in motion.events if event[0] == "hold_velocity"]
        self.assertEqual(len(pulses), 1)
        self.assertAlmostEqual(pulses[0][1], -0.02)
        self.assertEqual(pulses[0][2], 0.08)
        self.assertAlmostEqual(pulses[0][3], -1.2 * math.radians(2.0))
        self.assertAlmostEqual(result.max_abs_forward_drift_m, 0.02)
        self.assertAlmostEqual(result.max_abs_yaw_error_deg, 2.0)

    def test_pose_drift_limit_blocks_pulse(self) -> None:
        observation = make_observation(
            "strict",
            (make_target(track_id=9, center_px=(200.0, 520.0)),),
        )
        poses = iter([(0.0, 0.0, 0.0), (0.16, 0.0, 0.0)])
        aligner, _camera, _observer, motion, _clock, _writer = self.build_aligner(
            [observation],
            config={"strafe_pose_hold_enabled": True},
            pose_provider=lambda: next(poses),
        )

        result = aligner.run()

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "forward_drift_limit")
        self.assertEqual(result.pulse_count, 0)
        self.assertFalse(any(event[0] == "hold_velocity" for event in motion.events))

    def test_left_search_corrects_forward_drift_instead_of_exiting(self) -> None:
        observations = [
            make_observation("none"),
            make_observation("none"),
            make_observation(
                "strict",
                (
                    make_target(
                        track_id=8,
                        center_px=(640.0, 520.0),
                        linear_size_px=REFERENCE_LINEAR_SIZE_PX * 0.80,
                    ),
                ),
            ),
        ]
        poses = iter([(0.0, 0.0, 0.0), (0.16, 0.0, 0.0)])
        aligner, _camera, _observer, motion, _clock, _writer = (
            self.build_aligner(
                observations,
                config={
                    "acquire_only": True,
                    "no_red_frame_limit": 1,
                    "target_not_found_retries": 0,
                    "target_search_enabled": True,
                    "target_search_bilateral_enabled": False,
                    "target_search_max_distance_m": 1.0,
                    "target_search_settle_seconds": 0.0,
                    "strafe_pose_hold_enabled": True,
                    "max_vx_correction_mps": 0.04,
                },
                pose_provider=lambda: next(poses),
            )
        )

        result = aligner.run()

        self.assertTrue(result.ok)
        self.assertEqual(result.reason, "target_acquired")
        self.assertAlmostEqual(result.max_abs_forward_drift_m, 0.16)
        pulses = [event for event in motion.events if event[0] == "hold_velocity"]
        self.assertEqual(len(pulses), 1)
        self.assertAlmostEqual(pulses[0][1], -0.04)
        self.assertGreater(pulses[0][2], 0.0)

    def test_acquisition_waits_for_configured_left_search_dead_zone(self) -> None:
        target = make_observation(
            "strict",
            (
                make_target(
                    track_id=8,
                    center_px=(640.0, 520.0),
                    linear_size_px=REFERENCE_LINEAR_SIZE_PX * 0.80,
                ),
            ),
        )
        aligner, _camera, _observer, motion, _clock, _writer = (
            self.build_aligner(
                [make_observation("none"), target, target, target],
                config={
                    "acquire_only": True,
                    "no_red_frame_limit": 1,
                    "target_not_found_retries": 0,
                    "target_search_enabled": True,
                    "target_search_bilateral_enabled": False,
                    "target_search_max_distance_m": 1.0,
                    "target_search_min_distance_m": 0.16,
                    "target_search_settle_seconds": 0.0,
                },
            )
        )

        result = aligner.run()

        self.assertTrue(result.ok)
        self.assertEqual(result.reason, "target_acquired")
        pulses = [event for event in motion.events if event[0] == "hold_velocity"]
        self.assertEqual(len(pulses), 2)
        self.assertTrue(all(pulse[2] > 0.0 for pulse in pulses))
        self.assertAlmostEqual(result.strafe_distance_m, 0.16)

    def test_small_loose_false_positive_never_drives_motion(self) -> None:
        observations = [
            make_observation(
                "loose",
                (
                    make_target(
                        source="loose",
                        center_px=(740.0, 680.0),
                        linear_size_px=60.0,
                    ),
                ),
            )
            for _ in range(5)
        ]
        aligner, _camera, _observer, motion, _clock, _writer = (
            self.build_aligner(
                observations,
                config={"target_not_found_retries": 0},
            )
        )

        result = aligner.run()

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "target_not_found")
        self.assertEqual(result.pulse_count, 0)
        self.assertFalse(
            any(event[0] == "hold_velocity" for event in motion.events)
        )

    def test_no_red_frames_retry_three_times_before_target_not_found(self) -> None:
        aligner, camera, _observer, motion, _clock, _writer = self.build_aligner(
            [make_observation("none") for _ in range(20)]
        )

        result = aligner.run()

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "target_not_found")
        self.assertEqual(camera.read_count, 20)
        self.assertEqual(
            camera.reconnect_reasons,
            [
                "pregrasp_target_not_found_retry_1",
                "pregrasp_target_not_found_retry_2",
                "pregrasp_target_not_found_retry_3",
            ],
        )
        self.assertEqual(result.pulse_count, 0)
        self.assertGreaterEqual(
            sum(event[0] == "stop" for event in motion.events),
            2,
        )

    def test_missing_target_searches_left_until_target_enters_center_third(
        self,
    ) -> None:
        centered = make_observation(
            "strict",
            (make_target(track_id=8, center_px=(640.0, 520.0)),),
        )
        observations = [
            *[make_observation("none") for _ in range(20)],
            make_observation("none"),
            make_observation(
                "strict",
                (make_target(track_id=8, center_px=(300.0, 520.0)),),
            ),
            centered,
            centered,
            centered,
            centered,
        ]
        aligner, camera, _observer, motion, _clock, _writer = (
            self.build_aligner(
                observations,
                config={
                    "target_search_enabled": True,
                    "target_search_speed_mps": 0.04,
                    "target_search_step_seconds": 0.50,
                    "target_search_settle_seconds": 0.0,
                },
            )
        )

        result = aligner.run()

        self.assertTrue(result.ok)
        self.assertEqual(result.reason, "aligned")
        self.assertEqual(
            camera.reconnect_reasons,
            [
                "pregrasp_target_not_found_retry_1",
                "pregrasp_target_not_found_retry_2",
                "pregrasp_target_not_found_retry_3",
            ],
        )
        search_events = [
            event for event in motion.events if event[0] == "hold_velocity"
        ]
        self.assertEqual(len(search_events), 3)
        self.assertTrue(all(event[1] == 0.0 for event in search_events))
        self.assertTrue(all(event[2] == 0.04 for event in search_events))
        self.assertTrue(all(event[3] == 0.0 for event in search_events))
        self.assertTrue(all(event[4] == 0.50 for event in search_events))
        self.assertAlmostEqual(result.strafe_distance_m, 0.04)

    def test_pickup_search_holds_front_distance_in_both_directions(self) -> None:
        for distance_m, expected_vx in ((0.33, 0.025), (0.23, -0.025), (0.29, 0.0)):
            with self.subTest(distance_m=distance_m):
                aligner, _camera, _observer, motion, _clock, _writer = (
                    self.build_aligner(
                        [make_observation("none") for _ in range(3)],
                        config={
                            "no_red_frame_limit": 1,
                            "target_not_found_retries": 0,
                            "target_search_enabled": True,
                            "target_search_max_distance_m": 0.10,
                            "target_search_speed_mps": 0.04,
                            "target_search_step_seconds": 0.50,
                            "target_search_front_hold_enabled": True,
                            "target_search_front_target_m": 0.28,
                            "target_search_front_deadband_m": 0.02,
                            "target_search_front_hold_kp_s": 0.8,
                            "target_search_front_max_vx_mps": 0.025,
                        },
                        front_distance_provider=lambda value=distance_m: (value, 1.0),
                    )
                )

                aligner.run()

                search = next(
                    event for event in motion.events if event[0] == "hold_velocity"
                )
                self.assertAlmostEqual(search[1], expected_vx)

    def test_until_found_search_ignores_distance_and_time_limits(self) -> None:
        centered = make_observation(
            "strict",
            (make_target(track_id=8, center_px=(640.0, 520.0)),),
        )
        aligner, _camera, _observer, motion, clock, _writer = (
            self.build_aligner(
                [
                    make_observation("none"),
                    make_observation("none"),
                    make_observation("none"),
                    centered,
                ],
                config={
                    "acquire_only": True,
                    "no_red_frame_limit": 1,
                    "target_not_found_retries": 0,
                    "target_search_enabled": True,
                    "target_search_speed_mps": 0.08,
                    "target_search_step_seconds": 1.00,
                    "target_search_settle_seconds": 0.00,
                    "target_search_bilateral_enabled": False,
                    "target_search_until_found": True,
                    "target_search_max_distance_m": 0.05,
                    "max_seconds": 0.50,
                },
            )
        )

        result = aligner.run()

        self.assertTrue(result.ok)
        self.assertEqual(result.reason, "target_acquired")
        self.assertGreater(result.strafe_distance_m, 0.05)
        self.assertGreater(clock(), 0.50)
        search_events = [
            event for event in motion.events if event[0] == "hold_velocity"
        ]
        self.assertEqual(len(search_events), 3)
        self.assertTrue(all(event[2] == 0.08 for event in search_events))
        self.assertTrue(all(event[4] == 1.00 for event in search_events))

    def test_pickup_tags_reverse_search_without_aborting(self) -> None:
        clock = FakeClock()
        motion = OdomRecordingMotion(clock, lateral_scale=1.0)
        tag_frames = iter(({5: 710.0}, {4: 600.0}, {}))
        centered = make_observation(
            "strict",
            (make_target(track_id=8, center_px=(640.0, 520.0)),),
        )
        aligner, _camera, _observer, _motion, _clock, _writer = (
            self.build_aligner(
                [make_observation("none"), make_observation("none"), centered],
                config={
                    "acquire_only": True,
                    "target_search_enabled": True,
                    "target_search_bilateral_enabled": True,
                    "target_search_until_found": True,
                    "target_search_each_side_m": 0.25,
                    "target_search_max_net_lateral_m": 0.25,
                    "target_search_speed_mps": 0.10,
                    "target_search_step_seconds": 0.50,
                    "target_search_settle_seconds": 0.0,
                    "target_search_require_odom_progress": True,
                    "target_search_return_to_origin_on_failure": False,
                    "pickup_tag_boundary": {
                        "enabled": True,
                        "left_tag_id": 5,
                        "right_tag_id": 4,
                        "left_stop_center_x_px": 700,
                        "right_stop_center_x_px": 620,
                    },
                },
                motion=motion,
                pose_provider=lambda: tuple(motion.pose),
                search_origin_pose=(0.0, 0.0, 0.0),
                tag_boundary_provider=lambda: next(tag_frames),
            )
        )

        result = aligner.run()

        self.assertTrue(result.ok)
        search_events = [event for event in motion.events if event[0] == "hold_velocity"]
        self.assertEqual([event[2] for event in search_events], [-0.10, 0.10])

    def test_until_found_odom_boundary_ping_pongs(self) -> None:
        clock = FakeClock()
        motion = OdomRecordingMotion(clock, lateral_scale=1.0)
        centered = make_observation(
            "strict",
            (make_target(track_id=8, center_px=(640.0, 520.0)),),
        )
        aligner, _camera, _observer, _motion, _clock, _writer = (
            self.build_aligner(
                [*[make_observation("none") for _ in range(5)], centered],
                config={
                    "acquire_only": True,
                    "target_search_enabled": True,
                    "target_search_bilateral_enabled": True,
                    "target_search_until_found": True,
                    "target_search_each_side_m": 0.10,
                    "target_search_max_net_lateral_m": 0.10,
                    "target_search_speed_mps": 0.10,
                    "target_search_step_seconds": 1.0,
                    "target_search_settle_seconds": 0.0,
                    "target_search_require_odom_progress": True,
                    "target_search_return_to_origin_on_failure": False,
                },
                motion=motion,
                pose_provider=lambda: tuple(motion.pose),
                search_origin_pose=(0.0, 0.0, 0.0),
            )
        )

        result = aligner.run()

        self.assertTrue(result.ok)
        search_events = [event for event in motion.events if event[0] == "hold_velocity"]
        self.assertEqual(
            [math.copysign(1.0, event[2]) for event in search_events],
            [1.0, -1.0, -1.0, 1.0, 1.0],
        )

    def test_bilateral_search_stops_after_covering_both_sides(self) -> None:
        aligner, _camera, _observer, motion, _clock, _writer = (
            self.build_aligner(
                [make_observation("none") for _ in range(5)],
                config={
                    "no_red_frame_limit": 1,
                    "target_not_found_retries": 0,
                    "target_search_enabled": True,
                    "target_search_speed_mps": 0.10,
                    "target_search_step_seconds": 1.0,
                    "target_search_settle_seconds": 0.0,
                    "target_search_bilateral_enabled": True,
                    "target_search_each_side_m": 0.10,
                    "target_search_max_distance_m": 0.30,
                },
            )
        )

        result = aligner.run()

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "target_not_found_after_bilateral_search")
        self.assertEqual(result.pulse_count, 3)
        self.assertAlmostEqual(result.strafe_distance_m, 0.30)
        search_events = [
            event for event in motion.events if event[0] == "hold_velocity"
        ]
        self.assertEqual(len(search_events), 3)
        self.assertGreater(search_events[0][2], 0.0)
        self.assertLess(search_events[1][2], 0.0)
        self.assertLess(search_events[2][2], 0.0)

    def test_search_fails_after_consecutive_odometry_stalls(self) -> None:
        clock = FakeClock()
        motion = OdomRecordingMotion(clock, lateral_scale=0.0)
        aligner, _camera, _observer, _motion, _clock, _writer = (
            self.build_aligner(
                [make_observation("none") for _ in range(3)],
                config={
                    "no_red_frame_limit": 1,
                    "target_not_found_retries": 0,
                    "target_search_enabled": True,
                    "target_search_bilateral_enabled": False,
                    "target_search_max_distance_m": 1.0,
                    "target_search_require_odom_progress": True,
                    "target_search_max_stalled_pulses": 2,
                    "target_search_return_to_origin_on_failure": False,
                },
                motion=motion,
                pose_provider=lambda: tuple(motion.pose),
            )
        )

        result = aligner.run()

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "target_search_odometry_stalled")
        self.assertEqual(result.pulse_count, 2)

    def test_search_rejects_odometry_motion_opposite_command(self) -> None:
        clock = FakeClock()
        motion = OdomRecordingMotion(clock, lateral_scale=-1.0)
        aligner, _camera, _observer, _motion, _clock, _writer = (
            self.build_aligner(
                [make_observation("none") for _ in range(2)],
                config={
                    "no_red_frame_limit": 1,
                    "target_not_found_retries": 0,
                    "target_search_enabled": True,
                    "target_search_bilateral_enabled": False,
                    "target_search_max_distance_m": 1.0,
                    "target_search_require_odom_progress": True,
                    "target_search_return_to_origin_on_failure": False,
                },
                motion=motion,
                pose_provider=lambda: tuple(motion.pose),
            )
        )

        result = aligner.run()

        self.assertFalse(result.ok)
        self.assertEqual(
            result.reason,
            "target_search_odometry_wrong_direction",
        )
        self.assertEqual(result.pulse_count, 1)

    def test_search_enforces_net_lateral_field_boundary(self) -> None:
        clock = FakeClock()
        motion = OdomRecordingMotion(clock, lateral_scale=2.0)
        aligner, _camera, _observer, _motion, _clock, _writer = (
            self.build_aligner(
                [make_observation("none") for _ in range(2)],
                config={
                    "no_red_frame_limit": 1,
                    "target_not_found_retries": 0,
                    "target_search_enabled": True,
                    "target_search_bilateral_enabled": False,
                    "target_search_max_distance_m": 1.0,
                    "target_search_require_odom_progress": True,
                    "target_search_max_net_lateral_m": 0.10,
                    "target_search_return_to_origin_on_failure": False,
                },
                motion=motion,
                pose_provider=lambda: tuple(motion.pose),
            )
        )

        result = aligner.run()

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "target_search_field_boundary")
        self.assertEqual(result.pulse_count, 1)

    def test_strict_track_lock_is_kept_while_visible(self) -> None:
        observations = [
            make_observation(
                "strict",
                (make_target(track_id=1, center_px=(640.0, 520.0)),),
            ),
            make_observation(
                "strict",
                (
                    make_target(track_id=1, center_px=(200.0, 520.0)),
                    make_target(track_id=2, center_px=(640.0, 520.0)),
                ),
            ),
            *[
                make_observation(
                    "strict",
                    (make_target(track_id=1, center_px=(640.0, 520.0)),),
                )
                for _ in range(3)
            ],
        ]
        aligner, camera, _observer, _motion, _clock, _writer = (
            self.build_aligner(observations)
        )

        result = aligner.run()

        self.assertTrue(result.ok)
        self.assertEqual(result.selected_track_id, 1)
        self.assertEqual(result.pulse_count, 1)
        self.assertEqual(camera.read_count, 5)

    def test_every_pulse_is_followed_by_stop_and_uses_only_lateral_axis(
        self,
    ) -> None:
        observations = [
            make_observation(
                "strict",
                (make_target(track_id=3, center_px=(200.0, 520.0)),),
            ),
            make_observation(
                "strict",
                (make_target(track_id=3, center_px=(1000.0, 520.0)),),
            ),
            *[
                make_observation(
                    "strict",
                    (make_target(track_id=3, center_px=(640.0, 520.0)),),
                )
                for _ in range(3)
            ],
        ]
        aligner, _camera, _observer, motion, _clock, _writer = (
            self.build_aligner(observations)
        )

        result = aligner.run()

        self.assertTrue(result.ok)
        hold_indexes = [
            index
            for index, event in enumerate(motion.events)
            if event[0] == "hold_velocity"
        ]
        self.assertEqual(len(hold_indexes), 2)
        expected_pulse_seconds = (
            choose_alignment_action(
                make_target(center_px=(200.0, 520.0)),
                ACTION_CONFIG,
            ).pulse_seconds,
            choose_alignment_action(
                make_target(center_px=(1000.0, 520.0)),
                ACTION_CONFIG,
            ).pulse_seconds,
        )
        for pulse_number, index in enumerate(hold_indexes):
            event = motion.events[index]
            self.assertEqual(event[1], 0.0)
            self.assertEqual(event[3], 0.0)
            self.assertAlmostEqual(event[4], expected_pulse_seconds[pulse_number])
            self.assertEqual(motion.events[index + 1], ("stop",))

    def test_pulse_time_and_strafe_distance_limits_stop(self) -> None:
        limit_cases = (
            (
                {"max_pulses": 1},
                [
                    make_observation(
                        "strict",
                        (make_target(track_id=1, center_px=(200.0, 520.0)),),
                    ),
                    make_observation(
                        "strict",
                        (make_target(track_id=1, center_px=(200.0, 520.0)),),
                    ),
                ],
                "max_pulses",
                1,
            ),
            (
                {"max_seconds": 0.20},
                [
                    make_observation(
                        "strict",
                        (make_target(track_id=1, center_px=(200.0, 520.0)),),
                    )
                ],
                "max_seconds",
                0,
            ),
            (
                {"max_strafe_distance_m": 0.01},
                [
                    make_observation(
                        "strict",
                        (make_target(track_id=1, center_px=(200.0, 520.0)),),
                    )
                ],
                "max_strafe_distance",
                0,
            ),
        )

        for config, observations, reason, pulse_count in limit_cases:
            with self.subTest(reason=reason):
                aligner, _camera, _observer, motion, _clock, _writer = (
                    self.build_aligner(observations, config=config)
                )

                result = aligner.run()

                self.assertFalse(result.ok)
                self.assertEqual(result.reason, reason)
                self.assertEqual(result.pulse_count, pulse_count)
                self.assertEqual(motion.events[-1], ("stop",))

    def test_settle_wait_is_capped_by_remaining_max_seconds(self) -> None:
        observation = make_observation(
            "strict",
            (make_target(track_id=1, center_px=(200.0, 520.0)),),
        )
        pulse_seconds = choose_alignment_action(
            observation.strict_targets[0],
            ACTION_CONFIG,
        ).pulse_seconds
        aligner, camera, _observer, _motion, clock, _writer = (
            self.build_aligner(
                [observation],
                config={"max_seconds": pulse_seconds + 0.15},
            )
        )

        result = aligner.run()

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "max_seconds")
        self.assertEqual(camera.read_count, 1)
        self.assertEqual(len(clock.sleep_calls), 1)
        self.assertAlmostEqual(clock.sleep_calls[0], 0.15)
        self.assertLessEqual(clock.now, pulse_seconds + 0.15)
        self.assertLessEqual(result.elapsed_seconds, pulse_seconds + 0.15)

    def test_frame_processing_deadline_precedes_alignment_success(self) -> None:
        observation = make_observation(
            "strict",
            (make_target(track_id=1, center_px=(640.0, 520.0)),),
        )

        for deadline_stage in ("camera", "observer", "logging"):
            with self.subTest(deadline_stage=deadline_stage):
                clock = FakeClock()
                frame = object()

                def read():
                    if deadline_stage == "camera":
                        clock.advance(0.60)
                    return frame

                def observe(current_frame):
                    self.assertIs(current_frame, frame)
                    if deadline_stage == "observer":
                        clock.advance(0.60)
                    return observation

                camera = SimpleNamespace(read=read, release=Mock())
                observer = SimpleNamespace(observe=observe)
                motion = RecordingMotion(clock)
                writer = (
                    DeadlineAdvancingWriter(clock, 0.60)
                    if deadline_stage == "logging"
                    else RecordingWriter()
                )
                aligner = self.aligner_class()(
                    camera=camera,
                    observer=observer,
                    motion=motion,
                    config={
                        "max_seconds": 0.50,
                        "success_stable_frames": 1,
                    },
                    clock=clock,
                    sleep=clock.sleep,
                    log_dir="unused-test-log-root",
                    writer=writer,
                )

                result = aligner.run()

                self.assertFalse(result.ok)
                self.assertEqual(result.reason, "max_seconds")
                self.assertFalse(
                    any(
                        event[0] == "hold_velocity"
                        for event in motion.events
                    )
                )
                camera.release.assert_called_once_with()

    def test_exception_and_interrupt_always_stop_and_release_camera(self) -> None:
        for error in (RuntimeError("observer failed"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                observation = make_observation("none")
                aligner, camera, observer, motion, _clock, _writer = (
                    self.build_aligner([observation])
                )
                observer.observations = [error]

                with self.assertRaises(type(error)):
                    aligner.run()

                self.assertGreaterEqual(
                    sum(event[0] == "stop" for event in motion.events),
                    1,
                )
                self.assertEqual(camera.release_count, 1)

    def test_cleanup_stop_failure_does_not_replace_original_exception(
        self,
    ) -> None:
        observation = make_observation("none")
        motion = RecordingMotion(
            FakeClock(),
            stop_errors=[RuntimeError("stop failed")],
        )
        aligner, camera, observer, _motion, _clock, _writer = (
            self.build_aligner(
                [observation],
                motion=motion,
            )
        )
        observer.observations = [ValueError("observer failed")]

        with self.assertRaisesRegex(ValueError, "observer failed"):
            aligner.run()

        self.assertEqual(camera.release_count, 1)

    def test_initial_clock_exception_and_interrupt_stop_and_release_camera(
        self,
    ) -> None:
        for error in (RuntimeError("clock failed"), KeyboardInterrupt()):
            with self.subTest(error=type(error).__name__):
                clock = Mock(side_effect=error)
                camera = SequenceCamera([])
                observer = SequenceObserver([])
                motion = RecordingMotion(FakeClock())
                aligner = self.aligner_class()(
                    camera=camera,
                    observer=observer,
                    motion=motion,
                    config={},
                    clock=clock,
                    sleep=lambda _seconds: None,
                    log_dir="unused-test-log-root",
                    writer=RecordingWriter(),
                )

                with self.assertRaises(type(error)):
                    aligner.run()

                clock.assert_called_once_with()
                self.assertGreaterEqual(
                    sum(event[0] == "stop" for event in motion.events),
                    1,
                )
                self.assertEqual(camera.release_count, 1)

    def test_camera_read_failure_stops_and_releases(self) -> None:
        aligner, camera, _observer, motion, _clock, _writer = self.build_aligner(
            [],
            camera_frames=[None],
        )

        result = aligner.run()

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "camera_read_failed")
        self.assertGreaterEqual(
            sum(event[0] == "stop" for event in motion.events),
            2,
        )
        self.assertEqual(camera.release_count, 1)

    def test_run_logs_request_result_decisions_and_both_frame_images(self) -> None:
        observations = [
            make_observation(
                "strict",
                (make_target(track_id=7, center_px=(640.0, 520.0)),),
            )
            for _ in range(3)
        ]
        writer = RecordingWriter()
        aligner, _camera, _observer, _motion, _clock, writer = (
            self.build_aligner(observations, writer=writer)
        )

        result = aligner.run()

        self.assertTrue(result.ok)
        self.assertEqual(len(writer.created), 1)
        json_names = [path.name for path, _payload in writer.json_writes]
        self.assertEqual(json_names.count("request.json"), 1)
        self.assertEqual(json_names.count("result.json"), 1)
        self.assertEqual(
            sorted(name for name in json_names if name.startswith("decision_")),
            [
                "decision_0001.json",
                "decision_0002.json",
                "decision_0003.json",
            ],
        )
        image_names = [path.name for path, _image in writer.image_writes]
        self.assertEqual(
            sorted(image_names),
            [
                "annotated_0001.jpg",
                "annotated_0002.jpg",
                "annotated_0003.jpg",
                "undistorted_0001.jpg",
                "undistorted_0002.jpg",
                "undistorted_0003.jpg",
            ],
        )

    def test_logging_failure_does_not_mask_motion_exception(self) -> None:
        observation = make_observation(
            "strict",
            (make_target(track_id=2, center_px=(200.0, 520.0)),),
        )
        clock = FakeClock()
        motion = RecordingMotion(clock, hold_error=RuntimeError("motion failed"))
        aligner, camera, _observer, motion, _clock, _writer = self.build_aligner(
            [observation],
            motion=motion,
            writer=RecordingWriter(fail=True),
        )

        with self.assertRaisesRegex(RuntimeError, "motion failed"):
            aligner.run()

        self.assertGreaterEqual(
            sum(event[0] == "stop" for event in motion.events),
            2,
        )
        self.assertEqual(camera.release_count, 1)


class PregraspMissionIntegrationTests(unittest.TestCase):
    def test_resume_lateral_is_explicit_and_uses_tracking_threshold(self):
        config = load_config()
        self.assertEqual(
            config["pregrasp_red_align"]["strict_motion_min_linear_size_ratio"],
            0.70,
        )

        run_pregrasp_align._enable_lateral_resume(config)  # noqa: SLF001

        self.assertEqual(
            config["pregrasp_red_align"]["strict_motion_min_linear_size_ratio"],
            0.50,
        )

    def test_resume_lateral_requires_lateral_only_mode(self):
        with contextlib.redirect_stdout(io.StringIO()):
            exit_code = run_pregrasp_align.main(["--resume-lateral"])

        self.assertEqual(exit_code, 2)

    def build_mission(self, *, align_ok=True, align_error=None):
        config = load_config()
        config["pregrasp_red_align"] = dict(config["pregrasp_red_align"])
        config["pregrasp_red_align"]["enabled"] = True
        config["pregrasp_red_align"]["post_stop_settle_seconds"] = 0.0
        mission = LargeQuadrupedMission(
            config,
            dry_run=True,
            skip_arm=True,
        )
        events = []

        class FakeMotion:
            def stop(self):
                events.append("stop")

        class FakeArm:
            def stow(self):
                events.append("moving_pose")

            def camera_pose(self):
                events.append("ready_pose")

        class FakeAligner:
            def __init__(self):
                self.run_count = 0

            def run(self):
                self.run_count += 1
                if align_error is not None:
                    raise align_error
                return SimpleNamespace(
                    ok=align_ok,
                    reason="aligned" if align_ok else "target_not_found",
                )

        aligner = FakeAligner()
        mission.motion = FakeMotion()
        mission.arm = FakeArm()
        mission.pregrasp_aligner = aligner

        def run_alignment(
            *,
            wide_only=False,
            skip_wide_parallel=False,
            acquire_only=False,
        ):
            if wide_only:
                events.append("yaw")
                return True
            self.assertTrue(skip_wide_parallel)
            events.append("acquire" if acquire_only else "lateral")
            return bool(aligner.run().ok)

        mission._run_pregrasp_red_alignment = run_alignment
        mission._run_pregrasp_box_approach = (
            lambda: events.append("approach") or True
        )
        mission._confirm_pregrasp_box_distance = (
            lambda: events.append("distance_check") or True
        )
        mission._run_scripted_route = lambda _name: events.append("route") or True
        mission._estimate_red_bar_distance_mm = (
            lambda: events.append("distance") or 260.0
        )
        mission._retry_grasp = lambda _distance: events.append("grasp") or True
        return mission, aligner, events

    def test_pickup_keeps_moving_pose_through_alignment_then_enters_ready(self):
        mission, _aligner, events = self.build_mission()

        self.assertTrue(mission._pick_target("A"))

        self.assertEqual(
            events,
            [
                "stop",
                "moving_pose",
                "route",
                "stop",
                "approach",
                "stop",
                "acquire",
                "stop",
                "lateral",
                "stop",
                "distance_check",
                "stop",
                "stop",
                "stop",
                "ready_pose",
                "grasp",
                "stop",
                "yaw",
                "stop",
            ],
        )

    def test_alignment_failure_after_ready_pose_blocks_grasp(self):
        mission, _aligner, events = self.build_mission(align_ok=False)

        self.assertFalse(mission._pick_target("A"))

        self.assertEqual(
            events,
            [
                "stop",
                "moving_pose",
                "route",
                "stop",
                "approach",
                "stop",
                "acquire",
                "stop",
            ],
        )

    def test_second_pickup_runs_alignment_again(self):
        mission, aligner, _events = self.build_mission()

        self.assertTrue(mission._pick_target("A"))
        mission.context.carried_bar = False
        mission.context.placed_letters = ["A"]
        mission.context.target_letter = None
        self.assertTrue(mission._pick_target("C"))

        self.assertEqual(aligner.run_count, 4)

    def test_wide_only_does_not_run_arm_camera_aligner(self):
        mission, aligner, _events = self.build_mission()

        self.assertTrue(
            LargeQuadrupedMission._run_pregrasp_red_alignment(
                mission,
                wide_only=True,
            )
        )

        self.assertEqual(aligner.run_count, 0)

    def test_wide_only_skips_yaw_when_no_parallel_frames_exist(self):
        mission, _aligner, _events = self.build_mission()
        mission.context.dry_run = False
        mission.front_camera = SimpleNamespace(release=lambda: None)
        mission.pregrasp_box_aligner = SimpleNamespace(
            run=lambda: SimpleNamespace(
                ok=False,
                reason="no_valid_parallel_frames",
                final_error_deg=None,
            )
        )

        self.assertTrue(
            LargeQuadrupedMission._run_pregrasp_red_alignment(
                mission,
                wide_only=True,
            )
        )

    def test_wide_only_skips_partial_parallel_frame_failure(self):
        mission, _aligner, _events = self.build_mission()
        mission.context.dry_run = False
        mission.front_camera = SimpleNamespace(release=lambda: None)
        mission.pregrasp_box_aligner = SimpleNamespace(
            run=lambda: SimpleNamespace(
                ok=False,
                reason="insufficient_valid_parallel_frames",
                final_error_deg=3.0,
            )
        )

        self.assertTrue(
            LargeQuadrupedMission._run_pregrasp_red_alignment(
                mission,
                wide_only=True,
            )
        )

    def test_final_distance_failure_blocks_grasp_ready(self):
        mission, _aligner, events = self.build_mission()
        mission._confirm_pregrasp_box_distance = (
            lambda: events.append("distance_check_failed") or False
        )

        self.assertFalse(mission._pick_target("A"))

        self.assertIn("distance_check_failed", events)
        self.assertNotIn("ready_pose", events)
        self.assertNotIn("grasp", events)

    def test_final_distance_check_accepts_27cm_without_correction(self):
        mission, _aligner, _events = self.build_mission()
        mission.context.dry_run = False
        mission._confirm_pregrasp_box_distance = (
            LargeQuadrupedMission._confirm_pregrasp_box_distance.__get__(mission)
        )
        mission._run_pregrasp_box_approach = (
            lambda: self.fail("final distance check must not move the robot")
        )
        state = SimpleNamespace(front_ultrasound_m=0.27)
        mission.state_reader = SimpleNamespace(
            poll=lambda: state,
            safety_error=lambda **_kwargs: None,
        )
        with contextlib.redirect_stdout(io.StringIO()):
            confirmed = mission._confirm_pregrasp_box_distance()

        self.assertTrue(confirmed)

    def test_final_distance_check_backs_away_from_too_close_24cm(self):
        mission, _aligner, _events = self.build_mission()
        mission.context.dry_run = False
        mission._confirm_pregrasp_box_distance = (
            LargeQuadrupedMission._confirm_pregrasp_box_distance.__get__(mission)
        )
        mission.motion = Mock()
        state = SimpleNamespace(
            front_ultrasound_m=0.24,
            ultrasound_updated_at=1.0,
        )
        mission.state_reader = SimpleNamespace(
            poll=lambda: state,
            safety_error=lambda **_kwargs: None,
        )
        mission._wait_for_new_pregrasp_ultrasound = Mock(
            return_value=(0.28, 2.0)
        )

        with contextlib.redirect_stdout(io.StringIO()):
            confirmed = mission._confirm_pregrasp_box_distance()

        self.assertTrue(confirmed)
        mission.motion.hold_velocity.assert_called_once()
        correction = mission.motion.hold_velocity.call_args.args
        self.assertAlmostEqual(correction[0], -0.05)
        self.assertEqual(correction[1:3], (0.0, 0.0))
        self.assertAlmostEqual(correction[3], 0.8)

    def test_pickup_retry_resumes_failed_pregrasp_substage(self):
        mission, _aligner, events = self.build_mission()
        acquisition_calls = 0

        def run_alignment(
            *,
            wide_only=False,
            skip_wide_parallel=False,
            acquire_only=False,
        ):
            nonlocal acquisition_calls
            if wide_only:
                events.append("yaw")
                return True
            self.assertTrue(skip_wide_parallel)
            if acquire_only:
                acquisition_calls += 1
                events.append("acquire")
                return acquisition_calls > 1
            events.append("lateral")
            return True

        mission._run_pregrasp_red_alignment = run_alignment

        self.assertFalse(mission._pick_target("A"))
        self.assertTrue(mission._pick_target("A"))

        self.assertEqual(events.count("route"), 1)
        self.assertEqual(events.count("approach"), 1)
        self.assertEqual(events.count("acquire"), 2)
        self.assertEqual(events.count("lateral"), 1)

    def test_final_distance_check_allows_two_corrections_across_three_attempts(self):
        mission, _aligner, _events = self.build_mission()
        mission.context.dry_run = False
        mission._confirm_pregrasp_box_distance = (
            LargeQuadrupedMission._confirm_pregrasp_box_distance.__get__(mission)
        )
        states = iter(
            (
                SimpleNamespace(
                    front_ultrasound_m=0.31,
                    ultrasound_updated_at=1.0,
                ),
                SimpleNamespace(
                    front_ultrasound_m=0.302,
                    ultrasound_updated_at=2.0,
                ),
                SimpleNamespace(
                    front_ultrasound_m=0.289,
                    ultrasound_updated_at=3.0,
                ),
            )
        )
        reader = SimpleNamespace(state=None)

        def poll():
            try:
                reader.state = next(states)
            except StopIteration:
                pass
            return reader.state

        reader.poll = poll
        reader.safety_error = lambda **_kwargs: None
        mission.state_reader = reader
        correction_calls = []
        mission.motion.hold_velocity = lambda vx, vy, wz, duration: (
            correction_calls.append((vx, vy, wz, duration))
        )

        with contextlib.redirect_stdout(io.StringIO()):
            confirmed = mission._confirm_pregrasp_box_distance()

        self.assertTrue(confirmed)
        self.assertEqual(len(correction_calls), 2)
        first_vx, first_vy, first_wz, first_duration = correction_calls[0]
        second_vx, second_vy, second_wz, second_duration = correction_calls[1]
        self.assertAlmostEqual(first_vx, 0.05)
        self.assertEqual((first_vy, first_wz), (0.0, 0.0))
        self.assertAlmostEqual(first_duration, 0.60)
        self.assertAlmostEqual(second_vx, 0.05)
        self.assertEqual((second_vy, second_wz), (0.0, 0.0))
        self.assertAlmostEqual(second_duration, 0.44)

    def test_controlled_approach_does_not_disable_global_front_guard_afterward(self):
        mission, _aligner, _events = self.build_mission()
        mission.context.dry_run = False
        mission.state_reader = SimpleNamespace(
            state=SimpleNamespace(front_ultrasound_m=0.30),
            safety_error=lambda **_kwargs: None,
        )

        mission._controlled_box_approach_active = True
        mission._motion_guard(0.05, 0.0, 0.0)
        mission._controlled_box_approach_active = False

        with self.assertRaisesRegex(MissionAbort, "stopped forward command"):
            mission._motion_guard(0.05, 0.0, 0.0)

    def test_box_approach_uses_existing_controller_and_restores_guard(self):
        mission, _aligner, _events = self.build_mission()
        mission.context.dry_run = False
        mission._run_pregrasp_box_approach = (
            LargeQuadrupedMission._run_pregrasp_box_approach.__get__(mission)
        )

        class SequenceStateReader:
            def __init__(self):
                self.state = SimpleNamespace(
                    front_ultrasound_m=None,
                    ultrasound_updated_at=None,
                )
                self.distances = iter((0.60, 0.280, 0.279, 0.278))
                self.sequence = 0

            def poll(self):
                self.sequence += 1
                self.state.front_ultrasound_m = next(self.distances)
                self.state.ultrasound_updated_at = float(self.sequence)
                return self.state

            def sample_age(self, _updated_at):
                return 0.0

            def safety_error(self, **_kwargs):
                return None

        class RecordingMotion:
            def __init__(self):
                self.limits = SimpleNamespace(command_hz=20.0)
                self.events = []

            def move(self, vx, vy, wz):
                self.events.append(("move", vx, vy, wz))

            def hold_velocity(self, vx, vy, wz, duration):
                self.events.append(("hold", vx, vy, wz, duration))

            def stop(self):
                self.events.append(("stop",))

        motion = RecordingMotion()
        mission.motion = motion
        mission.state_reader = SequenceStateReader()
        mission._poll_pregrasp_ultrasound = lambda _duration: None

        with contextlib.redirect_stdout(io.StringIO()):
            reached = mission._run_pregrasp_box_approach()

        self.assertTrue(reached)
        self.assertIn(("move", 0.08, 0.0, 0.0), motion.events)
        self.assertFalse(mission._controlled_box_approach_active)

    def test_alignment_exception_after_ready_pose_stops_and_blocks_grasp(self):
        mission, _aligner, events = self.build_mission(
            align_error=RuntimeError("camera failed"),
        )

        self.assertFalse(mission._pick_target("A"))

        self.assertEqual(
            events,
            [
                "stop",
                "moving_pose",
                "route",
                "stop",
                "approach",
                "stop",
                "acquire",
                "stop",
            ],
        )

    def test_moving_pose_failure_blocks_pickup_route(self):
        mission, aligner, events = self.build_mission()

        def fail_moving_pose():
            events.append("moving_pose_failed")
            return False

        mission.arm.stow = fail_moving_pose

        with self.assertRaisesRegex(MissionAbort, "pre-pick moving pose"):
            mission._pick_target("A")

        self.assertEqual(events, ["stop", "moving_pose_failed"])
        self.assertEqual(aligner.run_count, 0)

    def test_horizontal_retry_realigns_entirely_from_moving_pose(self):
        config = load_config()
        config["arm"] = dict(config["arm"])
        config["arm"]["max_retries"] = 1
        config["pregrasp_red_align"] = dict(config["pregrasp_red_align"])
        config["pregrasp_red_align"]["post_stop_settle_seconds"] = 0.0
        mission = LargeQuadrupedMission(config, dry_run=True, skip_arm=True)
        events: list[str] = []

        class FakeMotion:
            def stop(self):
                events.append("stop")

        class FakeArm:
            def __init__(self):
                self.results = iter(
                    (
                        ArmTaskResult(
                            False,
                            "VISUAL_ALIGN",
                            feedback="target_left",
                        ),
                        ArmTaskResult(True, "DONE", object_held=True),
                    )
                )

            def grasp_red_bar(self, _distance):
                events.append("grasp")
                return next(self.results)

            def stow(self):
                events.append("moving_pose")
                return ArmTaskResult.success("MOVING_POSE")

            def camera_pose(self):
                events.append("ready_pose")
                return ArmTaskResult.success("GRASP_READY")

        mission.motion = FakeMotion()
        mission.arm = FakeArm()
        mission._pregrasp_ultrasound_ready = lambda: True
        mission._run_pregrasp_base_sequence = (
            lambda: events.append("align") or True
        )

        self.assertTrue(mission._retry_grasp(260.0))

        self.assertEqual(
            events,
            [
                "grasp",
                "stop",
                "moving_pose",
                "align",
                "stop",
                "stop",
                "ready_pose",
                "grasp",
            ],
        )

    def test_task_config_uses_stable_arm_camera_and_runtime_calibration_json(self):
        config = load_config()
        align_config = config["pregrasp_red_align"]

        self.assertEqual(
            config["arm"]["camera_device"],
            "/dev/v4l/by-id/usb-SXW_USB_Camera_200901010001-video-index0",
        )
        self.assertTrue(config["arm"]["calibration"].endswith("camera_calibration.json"))
        self.assertTrue(align_config["enabled"])
        self.assertEqual(align_config["roi"], [0.42, 0.55, 0.58, 0.85])
        self.assertNotIn("forward_speed_mps", align_config)
        self.assertNotIn("max_forward_distance_m", align_config)
        self.assertTrue(align_config["ultrasound_gate_enabled"])
        self.assertFalse(
            config["pickup_transfer"]["pre_retreat_yaw_alignment_enabled"]
        )
        self.assertEqual(align_config["ultrasound_min_m"], 0.10)
        self.assertEqual(align_config["ultrasound_max_m"], 2.0)
        self.assertEqual(align_config["loose_motion_min_linear_size_ratio"], 0.75)
        self.assertEqual(
            align_config["strict_tracking_min_linear_size_ratio"],
            0.50,
        )
        self.assertEqual(align_config["max_vx_correction_mps"], 0.04)


class PregraspDetectOnlyTests(unittest.TestCase):
    def test_detect_only_saves_ten_undistorted_and_annotated_frames(self):
        observations = [
            make_observation(
                "strict",
                (make_target(track_id=7, center_px=(640.0, 520.0)),),
            )
            for _ in range(10)
        ]
        frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(10)]
        camera = SequenceCamera(frames)
        observer = SequenceObserver(observations)
        writer = RecordingWriter()

        result = pregrasp_red_align.run_detect_only(
            camera=camera,
            observer=observer,
            config=ACTION_CONFIG,
            frame_count=10,
            log_dir="detect-only-test",
            writer=writer,
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.frames_saved, 10)
        self.assertEqual(result.motion_command_count, 0)
        self.assertEqual(result.selected_sources, ("strict",) * 10)
        image_names = [path.name for path, _image in writer.image_writes]
        self.assertEqual(
            len([name for name in image_names if name.startswith("undistorted_")]),
            10,
        )
        self.assertEqual(
            len([name for name in image_names if name.startswith("annotated_")]),
            10,
        )
        self.assertEqual(camera.release_count, 1)


if __name__ == "__main__":
    unittest.main()

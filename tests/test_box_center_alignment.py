from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from mission_lite3.box_center_alignment import (
    BoxCenterAligner,
    BoxCenterFrameResult,
    BoxCenterMeasurement,
    PlacementCandidate,
    PlacementLetterCandidate,
    PlacementLetterFrameResult,
    annotate_placement_letters,
    detect_pickup_box_center,
    detect_labeled_placement_candidates,
    detect_placement_box_centers,
    detect_placement_letter_candidates,
    map_placement_candidates,
    strafe_distance_for_box_center,
    strafe_correction_for_measurement,
    summarize_center_frames,
)
from mission_lite3.config_loader import load_config
from mission_lite3.lite3_motion import Lite3MotionController
from mission_lite3.wide_camera import BoxParallelResult


def candidate(x: float, y: float = 500.0, letter: str = "A") -> PlacementCandidate:
    return PlacementCandidate(
        center=(x, y),
        label_bbox=(int(x - 20), int(y - 40), 40, 25),
        box_bbox=(int(x - 45), int(y - 60), 90, 120),
        recognized_letter=letter,
        confidence=0.8,
    )


def placement_frame(
    centers: tuple[float, float, float, float],
    *,
    target: str = "C",
    width: int = 1000,
) -> BoxCenterFrameResult:
    mapped = {letter: (x, 500.0) for letter, x in zip("ABCD", centers)}
    return BoxCenterFrameResult(
        True,
        "",
        "placement",
        width,
        720,
        centers=mapped,
        target_label=target,
        target_center=mapped[target],
        spacing_px=tuple(centers[index + 1] - centers[index] for index in range(3)),
        confidence=0.9,
    )


def cardboard_row_frame(
    boundaries: tuple[int, int, int, int, int],
    *,
    width: int = 1000,
) -> np.ndarray:
    frame = np.full((720, width, 3), 55, dtype=np.uint8)
    cardboard = (95, 145, 175)
    cv2.rectangle(
        frame,
        (max(0, boundaries[0]), 350),
        (min(width - 1, boundaries[-1]), 650),
        cardboard,
        -1,
    )
    for x in boundaries[1:-1]:
        if 0 <= x < width:
            cv2.line(frame, (x, 350), (x, 650), (25, 35, 40), 7)
    return frame


def measurement(error_px: float, *, width: int = 1000) -> BoxCenterMeasurement:
    centers = {
        "A": (400.0 + error_px, 500.0),
        "B": (500.0 + error_px, 500.0),
        "C": (600.0 + error_px, 500.0),
        "D": (700.0 + error_px, 500.0),
    }
    # Target A is placed exactly error_px from the image center for the test.
    centers["A"] = (width / 2.0 + error_px, 500.0)
    centers["B"] = (centers["A"][0] + 100.0, 500.0)
    centers["C"] = (centers["A"][0] + 200.0, 500.0)
    centers["D"] = (centers["A"][0] + 300.0, 500.0)
    return BoxCenterMeasurement(
        True,
        "stable",
        "placement",
        centers,
        "A",
        centers["A"],
        width,
        720,
        7,
        7,
        0.9,
        (100.0, 100.0, 100.0),
        {key: center[0] - width / 2.0 for key, center in centers.items()},
        error_px,
        error_px / width,
    )


class RecordingMotion:
    def __init__(self) -> None:
        self.strafes: list[float] = []
        self.stop_count = 0

    def strafe_distance(self, distance_m: float, speed_mps=None) -> None:
        self.strafes.append(distance_m)

    def stop(self) -> None:
        self.stop_count += 1


class PoseHeldRecordingMotion(RecordingMotion):
    def __init__(self) -> None:
        super().__init__()
        self.pose_held_strafes: list[tuple[float, dict[str, float]]] = []

    def strafe_distance_pose_hold(self, distance_m: float, **kwargs) -> None:
        self.pose_held_strafes.append((distance_m, kwargs))


class RecordingBackend:
    name = "recording"

    def __init__(self) -> None:
        self.velocities: list[tuple[float, float, float]] = []

    def send_velocity(self, vx: float, vy: float, wz: float) -> None:
        self.velocities.append((vx, vy, wz))


class NullCamera:
    def release(self) -> None:
        return None


class SequenceCamera(NullCamera):
    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = iter(frames)

    def read(self):
        try:
            return next(self.frames).copy()
        except StopIteration:
            return None


class NullUndistorter:
    def apply(self, frame):
        return frame


class BoxCenterRecognitionTest(unittest.TestCase):
    def test_anchorless_white_label_row_uses_fixed_abcd_order(self) -> None:
        frame = np.zeros((720, 1000, 3), dtype=np.uint8)
        for center_x in (200, 350, 500, 650):
            cv2.rectangle(
                frame,
                (center_x - 40, 450),
                (center_x + 40, 520),
                (255, 255, 255),
                -1,
            )

        result = detect_placement_letter_candidates(
            frame,
            {
                "placement_label_row_anchorless_order_enabled": True,
                "placement_label_row_anchorless_min_confidence": 0.75,
            },
        )

        self.assertTrue(result.ok, result.reason)
        self.assertEqual(
            [candidate.recognized_letter for candidate in result.candidates],
            list("ABCD"),
        )
        self.assertEqual(
            [candidate.center[0] for candidate in result.candidates],
            [200.5, 350.5, 500.5, 650.5],
        )

    def test_single_visible_d_is_exposed_as_a_genuine_letter_candidate(self) -> None:
        image_path = Path(__file__).parent / "data" / "placement_letter_d.jpg"
        frame = cv2.imread(str(image_path))
        self.assertIsNotNone(frame)

        result = detect_labeled_placement_candidates(frame, {})

        self.assertTrue(result.ok, result.reason)
        self.assertEqual(len(result.candidates), 1)
        detected = result.candidates[0]
        self.assertEqual(detected.recognized_letter, "D")
        self.assertGreaterEqual(detected.confidence, 0.50)
        self.assertIsInstance(detected.label_bbox, tuple)
        self.assertEqual(len(detected.label_bbox), 4)
        x, y, width, height = detected.label_bbox
        self.assertGreater(width * height, 0)
        self.assertLessEqual(x, detected.center[0])
        self.assertLessEqual(y, detected.center[1])
        self.assertLessEqual(detected.center[0], x + width)
        self.assertLessEqual(detected.center[1], y + height)

    def test_genuine_letter_detector_does_not_infer_separator_labels(self) -> None:
        frame = cardboard_row_frame((150, 325, 500, 675, 850))

        result = detect_labeled_placement_candidates(frame, {})

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "no_recognized_letter")
        self.assertEqual(result.candidates, ())

    def test_genuine_letter_detector_rejects_invalid_frames_safely(self) -> None:
        invalid_frames = (
            None,
            np.zeros((20, 20), dtype=np.uint8),
            np.zeros((20, 20, 4), dtype=np.uint8),
        )

        for frame in invalid_frames:
            with self.subTest(shape=getattr(frame, "shape", None)):
                result = detect_labeled_placement_candidates(frame, {})
                self.assertFalse(result.ok)
                self.assertEqual(result.reason, "invalid_frame")
                self.assertEqual(result.candidates, ())

    def test_placement_letter_annotation_preserves_input_frame(self) -> None:
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        original = frame.copy()
        result = PlacementLetterFrameResult(
            True,
            "",
            160,
            120,
            candidates=(
                PlacementLetterCandidate(
                    center=(82.0, 58.0),
                    label_bbox=(60, 35, 44, 40),
                    recognized_letter="D",
                    confidence=0.87,
                ),
            ),
        )

        annotated = annotate_placement_letters(
            frame,
            result,
            target_letter="D",
            action="strafe_left",
        )

        self.assertEqual(annotated.shape, frame.shape)
        self.assertIsNot(annotated, frame)
        np.testing.assert_array_equal(frame, original)

    def test_unlabelled_cardboard_row_uses_three_vertical_separators(self) -> None:
        frame = np.full((720, 1000, 3), 55, dtype=np.uint8)
        cardboard = (95, 145, 175)
        cv2.rectangle(frame, (150, 350), (850, 650), cardboard, -1)
        for x in (325, 500, 675):
            cv2.line(frame, (x, 350), (x, 650), (25, 35, 40), 7)

        result = detect_placement_box_centers(frame, {}, target_letter="C")

        self.assertTrue(result.ok, result.reason)
        self.assertEqual(set(result.centers), set("ABCD"))
        self.assertAlmostEqual(result.centers["A"][0], 237.5, delta=12.0)
        self.assertAlmostEqual(result.centers["D"][0], 762.5, delta=12.0)
        self.assertEqual(result.target_center, result.centers["C"])
        self.assertEqual(result.detector_reason, "cardboard_vertical_separators")

    def test_distant_four_box_row_uses_calibrated_minimum_span(self) -> None:
        frame = cardboard_row_frame((350, 430, 510, 590, 670))

        result = detect_placement_box_centers(frame, {}, target_letter="D")

        self.assertTrue(result.ok, result.reason)
        self.assertEqual(result.detector_reason, "cardboard_vertical_separators")
        self.assertAlmostEqual(result.centers["A"][0], 390.0, delta=8.0)
        self.assertAlmostEqual(result.centers["D"][0], 630.0, delta=8.0)

    def test_cropped_left_row_tracks_d_without_remapping_visible_boxes(self) -> None:
        reference = {
            letter: (x, 500.0)
            for letter, x in zip("ABCD", (200.0, 350.0, 500.0, 650.0))
        }
        frame = cardboard_row_frame((0, 150, 300, 450, 600))

        result = detect_placement_box_centers(
            frame,
            {},
            target_letter="D",
            reference_centers=reference,
            expected_target_x=525.0,
        )

        self.assertTrue(result.ok, result.reason)
        self.assertEqual(result.detector_reason, "cardboard_separator_tracking")
        self.assertAlmostEqual(result.centers["A"][0], 75.0, delta=8.0)
        self.assertAlmostEqual(result.centers["D"][0], 525.0, delta=8.0)

    def test_cropped_right_row_tracks_a_without_remapping_visible_boxes(self) -> None:
        reference = {
            letter: (x, 500.0)
            for letter, x in zip("ABCD", (200.0, 350.0, 500.0, 650.0))
        }
        frame = cardboard_row_frame((400, 550, 700, 850, 1000))

        result = detect_placement_box_centers(
            frame,
            {},
            target_letter="A",
            reference_centers=reference,
            expected_target_x=475.0,
        )

        self.assertTrue(result.ok, result.reason)
        self.assertEqual(result.detector_reason, "cardboard_separator_tracking")
        self.assertAlmostEqual(result.centers["A"][0], 475.0, delta=8.0)
        self.assertAlmostEqual(result.centers["D"][0], 925.0, delta=8.0)

    def test_tracking_rejects_geometry_opposite_expected_motion(self) -> None:
        reference = {
            letter: (x, 500.0)
            for letter, x in zip("ABCD", (200.0, 350.0, 500.0, 650.0))
        }
        frame = cardboard_row_frame((0, 150, 300, 450, 600))

        result = detect_placement_box_centers(
            frame,
            {},
            target_letter="D",
            reference_centers=reference,
            expected_target_x=775.0,
        )

        self.assertFalse(result.ok)
        self.assertNotEqual(result.detector_reason, "cardboard_separator_tracking")

    def test_narrow_pickup_cardboard_uses_center_only_fallback(self) -> None:
        frame = np.full((720, 1000, 3), 55, dtype=np.uint8)
        cv2.rectangle(frame, (390, 360), (610, 650), (100, 145, 170), -1)

        def too_narrow(_frame):
            return BoxParallelResult(False, "cardboard_span_too_small")

        result = detect_pickup_box_center(frame, {}, detector=too_narrow)

        self.assertTrue(result.ok, result.reason)
        self.assertAlmostEqual(result.target_center[0], 500.0, delta=8.0)
        self.assertIn("narrow_cardboard_center_fallback", result.detector_reason)

    def test_four_boxes_are_mapped_by_image_order_not_recognized_letter(self) -> None:
        result = map_placement_candidates(
            [
                candidate(700, letter="A"),
                candidate(100, letter="D"),
                candidate(500, letter="B"),
                candidate(300, letter="C"),
            ],
            frame_width=1000,
            frame_height=720,
            target_letter="B",
        )

        self.assertTrue(result.ok)
        self.assertEqual([result.centers[key][0] for key in "ABCD"], [100.0, 300.0, 500.0, 700.0])
        self.assertEqual(result.target_center, (300.0, 500.0))

    def test_upper_background_letter_is_excluded_by_placement_roi(self) -> None:
        result = map_placement_candidates(
            [candidate(600, y=100), candidate(100), candidate(300), candidate(500), candidate(700)],
            frame_width=1000,
            frame_height=720,
            placement_roi=(0.0, 0.28, 1.0, 1.0),
        )

        self.assertTrue(result.ok)
        self.assertEqual(len(result.candidates), 4)

    def test_missing_and_duplicate_boxes_are_rejected(self) -> None:
        missing = map_placement_candidates(
            [candidate(100), candidate(300), candidate(500)],
            frame_width=1000,
            frame_height=720,
        )
        duplicate = map_placement_candidates(
            [candidate(100), candidate(300), candidate(320), candidate(700)],
            frame_width=1000,
            frame_height=720,
        )

        self.assertEqual(missing.reason, "missing_boxes")
        self.assertEqual(duplicate.reason, "duplicate_boxes")

    def test_pickup_center_uses_box_x_range_even_when_seam_is_unavailable(self) -> None:
        frame = np.zeros((720, 1000, 3), dtype=np.uint8)

        def detector(_frame):
            return BoxParallelResult(
                False,
                "cardboard_seam_not_found",
                confidence=0.0,
                box_x_range=(200, 600),
                top_line=(200, 300, 600, 320),
            )

        result = detect_pickup_box_center(frame, {}, detector=detector)

        self.assertTrue(result.ok)
        self.assertEqual(result.target_center, (400.0, 310.0))
        self.assertEqual(result.detector_reason, "cardboard_seam_not_found")

    def test_seven_frame_summary_accepts_four_stable_frames(self) -> None:
        results = [
            placement_frame((100 + dx, 300 + dx, 500 + dx, 700 + dx))
            for dx in (0, 4, -3, 5)
        ]
        results.extend(
            BoxCenterFrameResult(False, "missing_boxes", "placement", 1000, 720)
            for _ in range(3)
        )

        summary = summarize_center_frames(
            results,
            mode="placement",
            requested_frames=7,
            min_valid_frames=4,
            max_center_range_fraction=0.03,
            target_letter="C",
        )

        self.assertTrue(summary.ok)
        self.assertEqual(summary.stable_frames, 4)
        self.assertAlmostEqual(summary.target_error_px, 2.0, places=6)

    def test_center_range_over_three_percent_is_unstable(self) -> None:
        results = [
            placement_frame((100 + dx, 300, 500, 700))
            for dx in (0, 5, 10, 40)
        ]
        summary = summarize_center_frames(
            results,
            mode="placement",
            requested_frames=7,
            min_valid_frames=4,
            max_center_range_fraction=0.03,
            target_letter="A",
        )

        self.assertFalse(summary.ok)
        self.assertEqual(summary.reason, "center_measurement_unstable")

    def test_positive_image_error_commands_negative_strafe(self) -> None:
        result = strafe_correction_for_measurement(
            measurement(100.0),
            {
                "positive_error_strafe_sign": -1,
                "adjacent_box_spacing_m": 0.30,
            },
        )

        self.assertAlmostEqual(result, -0.30, places=6)

    def test_box_center_strafe_prefers_pose_held_controller(self) -> None:
        motion = PoseHeldRecordingMotion()

        strafe_distance_for_box_center(motion, -0.10, {})

        self.assertEqual(motion.strafes, [])
        self.assertEqual(len(motion.pose_held_strafes), 1)
        distance, options = motion.pose_held_strafes[0]
        self.assertEqual(distance, -0.10)
        self.assertEqual(options["speed_mps"], 0.08)
        self.assertEqual(options["max_vx_correction_mps"], 0.04)

    def test_pose_held_strafe_corrects_forward_and_yaw_drift(self) -> None:
        controller = Lite3MotionController(load_config(), dry_run=True)
        controller.dry_run = False
        backend = RecordingBackend()
        controller.backend = backend
        poses = iter(
            (
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.04, 0.05, np.deg2rad(2.0)),
                (0.0, 0.18, 0.0),
            )
        )
        controller.configure_safety(
            lambda _vx, _vy, _wz: None,
            lambda: next(poses),
            feedback_required=True,
        )

        controller.strafe_distance_pose_hold(0.20, speed_mps=0.08)

        corrections = [
            velocity
            for velocity in backend.velocities
            if velocity[0] < 0.0 and velocity[2] < 0.0
        ]
        self.assertEqual(len(corrections), 1)
        self.assertAlmostEqual(corrections[0][0], -0.04)
        self.assertAlmostEqual(corrections[0][1], 0.08)
        self.assertAlmostEqual(corrections[0][2], -1.2 * np.deg2rad(2.0))
        self.assertEqual(backend.velocities[-1], (0.0, 0.0, 0.0))

    def test_pose_held_strafe_rejects_excessive_forward_drift(self) -> None:
        controller = Lite3MotionController(load_config(), dry_run=True)
        controller.dry_run = False
        backend = RecordingBackend()
        controller.backend = backend
        poses = iter(((0.0, 0.0, 0.0), (0.16, 0.0, 0.0)))
        controller.configure_safety(
            lambda _vx, _vy, _wz: None,
            lambda: next(poses),
            feedback_required=True,
        )

        with self.assertRaisesRegex(RuntimeError, "forward drift limit"):
            controller.strafe_distance_pose_hold(0.20, speed_mps=0.08)

        self.assertEqual(backend.velocities[-1], (0.0, 0.0, 0.0))

    def test_three_correction_limit_rolls_back_visual_displacement(self) -> None:
        motion = RecordingMotion()
        with tempfile.TemporaryDirectory() as tmp:
            aligner = BoxCenterAligner(
                camera=NullCamera(),
                undistorter=NullUndistorter(),
                motion=motion,
                config={
                    "enabled": True,
                    "tolerance_fraction": 0.05,
                    "max_corrections": 3,
                    "max_single_strafe_m": 0.25,
                    "max_total_strafe_m": 0.75,
                    "adjacent_box_spacing_m": 0.30,
                    "positive_error_strafe_sign": -1,
                    "strafe_speed_mps": 0.08,
                    "settle_seconds": 0.0,
                    "alignment_run_log_dir": tmp,
                },
                measurement_provider=lambda _mode, _target: measurement(200.0),
                sleep=lambda _seconds: None,
            )

            result = aligner.run("placement", "A")

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "max_corrections")
        self.assertEqual(result.correction_count, 3)
        self.assertEqual(result.motion_command_count, 4)
        self.assertEqual(motion.strafes, [-0.25, -0.25, -0.25, 0.75])
        self.assertTrue(result.rollback_attempted)
        self.assertTrue(result.rollback_ok)
        self.assertAlmostEqual(result.net_strafe_m, 0.0)

    def test_aligner_accepts_per_run_tolerance_override(self) -> None:
        motion = RecordingMotion()
        with tempfile.TemporaryDirectory() as tmp:
            aligner = BoxCenterAligner(
                camera=NullCamera(),
                undistorter=NullUndistorter(),
                motion=motion,
                config={
                    "enabled": True,
                    "tolerance_fraction": 0.03,
                    "max_corrections": 3,
                    "alignment_run_log_dir": tmp,
                },
                measurement_provider=lambda _mode, _target: measurement(40.0),
                sleep=lambda _seconds: None,
            )

            result = aligner.run(
                "placement",
                "A",
                tolerance_fraction=0.05,
            )

        self.assertTrue(result.ok)
        self.assertEqual(result.correction_count, 0)
        self.assertEqual(motion.strafes, [])

    def test_aligner_tracks_cropped_row_after_first_correction(self) -> None:
        motion = RecordingMotion()
        initial = cardboard_row_frame((125, 275, 425, 575, 725))
        cropped = cardboard_row_frame((0, 150, 300, 450, 600))
        camera = SequenceCamera([initial] * 4 + [cropped] * 4)
        with tempfile.TemporaryDirectory() as tmp:
            aligner = BoxCenterAligner(
                camera=camera,
                undistorter=NullUndistorter(),
                motion=motion,
                config={
                    "enabled": True,
                    "frames_per_measurement": 4,
                    "min_valid_frames": 4,
                    "max_center_range_fraction": 0.03,
                    "tolerance_fraction": 0.05,
                    "max_corrections": 3,
                    "max_single_strafe_m": 0.25,
                    "max_total_strafe_m": 0.75,
                    "adjacent_box_spacing_m": 0.30,
                    "positive_error_strafe_sign": -1,
                    "strafe_speed_mps": 0.08,
                    "settle_seconds": 0.0,
                    "alignment_run_log_dir": tmp,
                },
                sleep=lambda _seconds: None,
            )

            result = aligner.run("placement", "D")

        self.assertTrue(result.ok, result.reason)
        self.assertEqual(result.reason, "aligned")
        self.assertEqual(result.correction_count, 1)
        self.assertEqual(result.measurement_count, 2)
        self.assertEqual(len(motion.strafes), 1)
        self.assertAlmostEqual(motion.strafes[0], -0.25)
        self.assertLessEqual(abs(result.final_error_px), 50.0)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import numpy as np

from mission_lite3.wide_box_alignment import (
    WideBoxAligner,
    detect_placement_row_parallel,
    summarize_parallel_results,
)
from mission_lite3.wide_camera import BoxParallelResult


class FakeCamera:
    def __init__(self, frame_count: int) -> None:
        self.frames = [object() for _ in range(frame_count)]
        self.release_count = 0

    def read(self):
        return self.frames.pop(0) if self.frames else None

    def release(self) -> None:
        self.release_count += 1


class FakeUndistorter:
    calibration = {"validated_for_control": True}

    def apply(self, frame):
        return frame


class FakeMotion:
    def __init__(self) -> None:
        self.events = []

    def hold_velocity(self, vx, vy, wz, duration) -> None:
        self.events.append(("hold", vx, vy, wz, duration))

    def stop(self) -> None:
        self.events.append(("stop",))


def results(error: float, count: int = 8):
    return [BoxParallelResult(True, "", parallel_error_deg=error) for _ in range(count)]


class WideBoxAlignmentTests(unittest.TestCase):
    def test_recorded_placement_row_spread_passes_calibrated_limit(self) -> None:
        values = [
            4.8244,
            4.6667,
            4.7254,
            3.1485,
            5.0330,
            2.9044,
            4.3763,
            3.3746,
            3.3525,
            4.4581,
            4.9510,
            3.4451,
        ]
        measurement = summarize_parallel_results(
            [
                BoxParallelResult(True, "", parallel_error_deg=value)
                for value in values
            ],
            requested_frames=12,
            min_valid_frames=8,
            max_range_deg=2.0,
        )

        self.assertTrue(measurement.ok)
        self.assertAlmostEqual(measurement.median_error_deg, 4.4172, places=3)
        self.assertLess(measurement.error_range_deg, 2.0)

    def test_placement_row_requires_four_boxes_and_full_row_span(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        four_boxes = SimpleNamespace(
            ok=True,
            reason="",
            centers={letter: (index * 100.0, 500.0) for index, letter in enumerate("ABCD")},
        )
        full_row = BoxParallelResult(
            True,
            "",
            parallel_error_deg=2.0,
            box_x_range=(200, 1000),
        )
        accepted = detect_placement_row_parallel(
            frame,
            {},
            center_detector=lambda *_args, **_kwargs: four_boxes,
            parallel_detector=lambda _frame: full_row,
        )
        self.assertTrue(accepted.ok)

        missing = detect_placement_row_parallel(
            frame,
            {},
            center_detector=lambda *_args, **_kwargs: SimpleNamespace(
                ok=False,
                reason="missing_boxes",
                centers={},
            ),
            parallel_detector=lambda _frame: full_row,
        )
        self.assertFalse(missing.ok)
        self.assertIn("four_box_reference_unavailable", missing.reason)

        partial = detect_placement_row_parallel(
            frame,
            {},
            center_detector=lambda *_args, **_kwargs: four_boxes,
            parallel_detector=lambda _frame: BoxParallelResult(
                True,
                "",
                parallel_error_deg=5.0,
                box_x_range=(700, 1050),
            ),
        )
        self.assertFalse(partial.ok)
        self.assertEqual(partial.reason, "placement_row_span_too_small")

    def test_robust_summary_ignores_one_outlier(self) -> None:
        samples = results(2.0, 9) + results(3.5, 1)
        measurement = summarize_parallel_results(
            samples,
            requested_frames=10,
            min_valid_frames=8,
            max_range_deg=0.8,
        )
        self.assertTrue(measurement.ok)
        self.assertLess(measurement.error_range_deg, measurement.full_error_range_deg)

    def test_recorded_point_eight_degree_spread_is_stable(self) -> None:
        values = [
            2.5551,
            2.8823,
            2.1346,
            2.0808,
            2.2154,
            1.7524,
            2.2984,
            2.3324,
            1.9865,
            2.2377,
            2.8219,
            2.5740,
        ]
        measurement = summarize_parallel_results(
            [
                BoxParallelResult(True, "", parallel_error_deg=value)
                for value in values
            ],
            requested_frames=12,
            min_valid_frames=8,
            max_range_deg=1.0,
        )

        self.assertTrue(measurement.ok)
        self.assertGreater(measurement.error_range_deg, 0.8)

    def test_default_budget_allows_fifth_correction_then_alignment(self) -> None:
        samples = []
        for error in (8.5, 6.7, 5.1, 3.8, 2.27, 1.2):
            samples.extend(results(error))
        motion = FakeMotion()
        aligner = WideBoxAligner(
            camera=FakeCamera(len(samples)),
            undistorter=FakeUndistorter(),
            motion=motion,
            detector=lambda _frame: samples.pop(0),
            sleep=lambda _seconds: None,
            config={
                "frames_per_measurement": 8,
                "min_valid_frames": 8,
                "run_log_dir": "/proc/not-writable",
            },
        )

        result = aligner.run()

        self.assertTrue(result.ok)
        self.assertEqual(result.reason, "aligned")
        self.assertEqual(result.correction_count, 5)

    def test_aligned_measurement_sends_no_motion(self) -> None:
        samples = results(0.7)
        detector = lambda _frame: samples.pop(0)
        motion = FakeMotion()
        aligner = WideBoxAligner(
            camera=FakeCamera(8),
            undistorter=FakeUndistorter(),
            motion=motion,
            detector=detector,
            config={
                "frames_per_measurement": 8,
                "min_valid_frames": 8,
                "run_log_dir": "/proc/not-writable",
            },
        )

        result = aligner.run()

        self.assertTrue(result.ok)
        self.assertEqual(result.reason, "aligned")
        self.assertEqual(result.motion_command_count, 0)
        self.assertFalse(any(event[0] == "hold" for event in motion.events))

    def test_default_accepts_error_within_one_point_five_degrees(self) -> None:
        samples = results(1.4)
        detector = lambda _frame: samples.pop(0)
        motion = FakeMotion()
        aligner = WideBoxAligner(
            camera=FakeCamera(8),
            undistorter=FakeUndistorter(),
            motion=motion,
            detector=detector,
            config={
                "frames_per_measurement": 8,
                "min_valid_frames": 8,
                "run_log_dir": "/proc/not-writable",
            },
        )

        result = aligner.run()

        self.assertTrue(result.ok)
        self.assertEqual(result.reason, "aligned")
        self.assertEqual(result.motion_command_count, 0)

    def test_positive_error_commands_negative_wz_then_remeasures(self) -> None:
        samples = results(2.5) + results(0.8)
        detector = lambda _frame: samples.pop(0)
        motion = FakeMotion()
        aligner = WideBoxAligner(
            camera=FakeCamera(16),
            undistorter=FakeUndistorter(),
            motion=motion,
            detector=detector,
            sleep=lambda _seconds: None,
            config={
                "frames_per_measurement": 8,
                "min_valid_frames": 8,
                "run_log_dir": "/proc/not-writable",
            },
        )

        result = aligner.run()

        self.assertTrue(result.ok)
        self.assertEqual(result.correction_count, 1)
        holds = [event for event in motion.events if event[0] == "hold"]
        expected_duration = math.radians(2.5 * 0.5) / (0.1 * 0.5)
        self.assertEqual(len(holds), 1)
        self.assertEqual(holds[0][:4], ("hold", 0.0, 0.0, -0.1))
        self.assertAlmostEqual(holds[0][4], expected_duration)

    def test_unstable_measurement_blocks_motion(self) -> None:
        values = [0.0, 0.2, 0.4, 0.6, 1.2, 1.4, 1.6, 1.8]
        samples = [
            BoxParallelResult(True, "", parallel_error_deg=value)
            for value in values
        ]
        motion = FakeMotion()
        aligner = WideBoxAligner(
            camera=FakeCamera(8),
            undistorter=FakeUndistorter(),
            motion=motion,
            detector=lambda _frame: samples.pop(0),
            config={
                "frames_per_measurement": 8,
                "min_valid_frames": 8,
                "run_log_dir": "/proc/not-writable",
            },
        )

        result = aligner.run()

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "parallel_measurement_unstable")
        self.assertFalse(any(event[0] == "hold" for event in motion.events))


if __name__ == "__main__":
    unittest.main()

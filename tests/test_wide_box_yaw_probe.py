from __future__ import annotations

import unittest

from mission_lite3.tools.wide_box_yaw_probe import (
    normalize_angle_rad,
    summarize_samples,
    validate_probe_parameters,
)


class WideBoxYawProbeTests(unittest.TestCase):
    def test_accepts_recommended_small_probe(self) -> None:
        validate_probe_parameters(0.08, 0.25)
        validate_probe_parameters(-0.08, 0.25)

    def test_rejects_large_or_too_short_probe(self) -> None:
        with self.assertRaises(ValueError):
            validate_probe_parameters(0.13, 0.25)
        with self.assertRaises(ValueError):
            validate_probe_parameters(0.08, 0.09)

    def test_summarizes_only_valid_samples(self) -> None:
        summary = summarize_samples(
            [
                {"ok": True, "parallel_error_deg": 2.0},
                {"ok": False, "parallel_error_deg": None},
                {"ok": True, "parallel_error_deg": 2.4},
            ]
        )
        self.assertEqual(summary["successful_frames"], 2)
        self.assertAlmostEqual(summary["median_parallel_error_deg"], 2.2)
        self.assertAlmostEqual(summary["error_range_deg"], 0.4)

    def test_uses_robust_range_for_a_long_window(self) -> None:
        summary = summarize_samples(
            [
                {"ok": True, "parallel_error_deg": value}
                for value in [2.0, 2.1, 2.1, 2.2, 2.2, 2.2, 2.3, 2.3, 2.4, 3.5]
            ]
        )
        self.assertLess(summary["error_range_deg"], summary["full_error_range_deg"])
        self.assertLess(summary["error_range_deg"], 0.8)

    def test_normalizes_wrapped_yaw_delta(self) -> None:
        self.assertAlmostEqual(normalize_angle_rad(-6.2), 0.08318530717958605)


if __name__ == "__main__":
    unittest.main()

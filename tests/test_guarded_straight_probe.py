from __future__ import annotations

import math
import unittest

from mission_lite3.tools.guarded_straight_probe import (
    body_delta,
    correction_velocity,
)


class GuardedStraightProbeTests(unittest.TestCase):
    def test_body_delta_uses_reference_heading(self) -> None:
        forward, lateral, yaw = body_delta(
            (0.0, 0.0, math.pi / 2.0),
            (-0.02, -0.05, math.pi / 2.0 + math.radians(1.0)),
        )
        self.assertAlmostEqual(forward, -0.05)
        self.assertAlmostEqual(lateral, 0.02)
        self.assertAlmostEqual(math.degrees(yaw), 1.0)

    def test_correction_opposes_lateral_and_yaw_drift(self) -> None:
        vx, vy, wz = correction_velocity(
            (0.0, 0.0, 0.0),
            (-0.01, 0.02, math.radians(2.0)),
            base_vx=-0.05,
        )
        self.assertEqual(vx, -0.05)
        self.assertLess(vy, 0.0)
        self.assertLess(wz, 0.0)

    def test_small_drift_is_inside_deadband(self) -> None:
        _vx, vy, wz = correction_velocity(
            (0.0, 0.0, 0.0),
            (0.0, 0.002, math.radians(0.2)),
            base_vx=-0.05,
        )
        self.assertEqual(vy, 0.0)
        self.assertEqual(wz, 0.0)


if __name__ == "__main__":
    unittest.main()

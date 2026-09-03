from __future__ import annotations

import unittest

from mission_lite3.box_approach import ApproachConfig, BoxApproachController


class BoxApproachTests(unittest.TestCase):
    def test_far_distance_uses_continuous_motion(self) -> None:
        controller = BoxApproachController(ApproachConfig())
        decision = controller.decide(0.662)
        self.assertEqual(decision.mode, "continuous")
        self.assertAlmostEqual(decision.vx, 0.08)
        self.assertIsNone(decision.drive_duration_s)
        self.assertEqual(decision.settle_duration_s, 0.0)

    def test_middle_distance_uses_continuous_motion(self) -> None:
        controller = BoxApproachController(ApproachConfig())
        decision = controller.decide(0.385)
        self.assertEqual(decision.mode, "continuous")
        self.assertAlmostEqual(decision.vx, 0.05)
        self.assertIsNone(decision.drive_duration_s)
        self.assertEqual(decision.settle_duration_s, 0.0)

    def test_near_distance_uses_continuous_motion(self) -> None:
        controller = BoxApproachController(ApproachConfig())
        decision = controller.decide(0.311)
        self.assertEqual(decision.mode, "continuous")
        self.assertAlmostEqual(decision.vx, 0.05)
        self.assertIsNone(decision.drive_duration_s)
        self.assertEqual(decision.settle_duration_s, 0.0)

    def test_target_requires_three_consecutive_stop_samples(self) -> None:
        controller = BoxApproachController(ApproachConfig())
        self.assertFalse(controller.decide(0.280).reached)
        self.assertFalse(controller.decide(0.279).reached)
        self.assertTrue(controller.decide(0.278).reached)

    def test_small_approach_overshoot_remains_a_valid_stop(self) -> None:
        controller = BoxApproachController(ApproachConfig())
        controller.decide(0.27)
        controller.decide(0.27)
        decision = controller.decide(0.27)
        self.assertTrue(decision.reached)
        self.assertEqual(decision.vx, 0.0)

    def test_sensor_floor_just_above_target_uses_stop_tolerance(self) -> None:
        controller = BoxApproachController(ApproachConfig())
        self.assertFalse(controller.decide(0.284).reached)
        self.assertFalse(controller.decide(0.284).reached)
        decision = controller.decide(0.284)
        self.assertTrue(decision.reached)
        self.assertEqual(decision.mode, "stop")
        self.assertEqual(decision.vx, 0.0)

    def test_distance_above_target_resets_stop_confirmation(self) -> None:
        controller = BoxApproachController(ApproachConfig())
        self.assertFalse(controller.decide(0.280).reached)
        self.assertFalse(controller.decide(0.30).reached)
        self.assertFalse(controller.decide(0.280).reached)

    def test_too_close_ultrasound_value_stops_for_backaway_correction(self) -> None:
        controller = BoxApproachController(ApproachConfig())
        controller.decide(0.24)
        controller.decide(0.24)
        decision = controller.decide(0.24)
        self.assertTrue(decision.reached)
        self.assertEqual(decision.vx, 0.0)

    def test_physically_invalid_ultrasound_value_is_rejected(self) -> None:
        controller = BoxApproachController(ApproachConfig())
        with self.assertRaisesRegex(ValueError, "outside"):
            controller.decide(0.02)


if __name__ == "__main__":
    unittest.main()

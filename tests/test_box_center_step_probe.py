from __future__ import annotations

import contextlib
import io
import json
import unittest

from mission_lite3.box_center_alignment import (
    BoxCenterAlignmentResult,
    BoxCenterMeasurement,
)
from mission_lite3.tools.box_center_step_probe import (
    build_parser,
    evaluate_probe_result,
    main,
)


def alignment_result(
    *,
    initial: float = 164.0,
    final: float = 132.0,
    rollback_ok: bool = True,
) -> BoxCenterAlignmentResult:
    return BoxCenterAlignmentResult(
        False,
        "max_corrections" if rollback_ok else "max_corrections;rollback_failed",
        "placement",
        "C",
        1,
        2,
        2,
        initial,
        final,
        -0.05,
        0.0 if rollback_ok else -0.05,
        True,
        rollback_ok,
        "box_center_step_runs/test",
    )


def full_alignment_result() -> BoxCenterAlignmentResult:
    return BoxCenterAlignmentResult(
        True,
        "aligned",
        "placement",
        "D",
        2,
        2,
        3,
        216.0,
        30.0,
        -0.38,
        -0.38,
        False,
        True,
        "box_center_full_runs/test",
    )


def rollback_measurement(error: float = 170.0) -> BoxCenterMeasurement:
    return BoxCenterMeasurement(
        True,
        "stable",
        "placement",
        {"D": (640.0 + error, 530.0)},
        "D",
        (640.0 + error, 530.0),
        1280,
        720,
        7,
        7,
        0.9,
        (180.0, 180.0, 180.0),
        {"D": error},
        error,
        error / 1280.0,
    )


class BoxCenterStepProbeTest(unittest.TestCase):
    def test_defaults_to_dry_run_c_target(self) -> None:
        args = build_parser().parse_args([])
        self.assertFalse(args.robot)
        self.assertEqual(args.target_letter, "C")
        self.assertEqual(args.profile, "step")

    def test_direction_and_rollback_must_both_be_verified(self) -> None:
        accepted = evaluate_probe_result(
            alignment_result(),
            rollback_measurement=rollback_measurement(170.0),
            odom_return_error_m=0.02,
            odom_return_yaw_deg=0.8,
        )
        wrong_direction = evaluate_probe_result(
            alignment_result(initial=164.0, final=190.0),
            rollback_measurement=rollback_measurement(164.0),
            odom_return_error_m=0.02,
            odom_return_yaw_deg=0.8,
        )
        rollback_failed = evaluate_probe_result(
            alignment_result(rollback_ok=False),
            rollback_measurement=rollback_measurement(164.0),
            odom_return_error_m=0.02,
            odom_return_yaw_deg=0.8,
        )

        self.assertTrue(accepted["ok"])
        self.assertFalse(wrong_direction["ok"])
        self.assertFalse(rollback_failed["ok"])

    def test_visual_or_odometry_return_outside_limit_is_rejected(self) -> None:
        visual_bad = evaluate_probe_result(
            alignment_result(),
            rollback_measurement=rollback_measurement(210.0),
            odom_return_error_m=0.02,
            odom_return_yaw_deg=0.8,
        )
        odom_bad = evaluate_probe_result(
            alignment_result(),
            rollback_measurement=rollback_measurement(164.0),
            odom_return_error_m=0.05,
            odom_return_yaw_deg=0.8,
        )

        self.assertFalse(visual_bad["ok"])
        self.assertFalse(odom_bad["ok"])

    def test_full_profile_requires_alignment_and_external_return(self) -> None:
        accepted = evaluate_probe_result(
            full_alignment_result(),
            rollback_measurement=rollback_measurement(220.0),
            odom_return_error_m=0.03,
            odom_return_yaw_deg=1.0,
            profile="full",
            external_return_attempted=True,
            external_return_ok=True,
        )
        not_aligned = evaluate_probe_result(
            alignment_result(),
            rollback_measurement=rollback_measurement(164.0),
            odom_return_error_m=0.02,
            odom_return_yaw_deg=0.8,
            profile="full",
        )

        self.assertTrue(accepted["ok"])
        self.assertFalse(not_aligned["ok"])

    def test_dry_run_has_no_motion_commands(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main([])
        payload = json.loads(output.getvalue())

        self.assertEqual(code, 0)
        self.assertTrue(payload["dry_run"])
        self.assertEqual(payload["motion_command_count"], 0)
        self.assertEqual(payload["max_single_strafe_m"], 0.05)

    def test_real_probe_requires_explicit_confirmation(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(["--robot"])

        self.assertEqual(code, 2)
        self.assertIn("--robot --yes", output.getvalue())

    def test_full_dry_run_uses_production_limits_without_motion(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(["--profile", "full", "--target-letter", "D"])
        payload = json.loads(output.getvalue())

        self.assertEqual(code, 0)
        self.assertEqual(payload["motion_command_count"], 0)
        self.assertEqual(payload["max_corrections"], 3)
        self.assertEqual(payload["max_single_strafe_m"], 0.25)
        self.assertEqual(payload["max_total_strafe_m"], 0.75)


if __name__ == "__main__":
    unittest.main()

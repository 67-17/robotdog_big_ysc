import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

import local_cartesian_jog as jog


class LocalCartesianMathTests(unittest.TestCase):
    def test_endpoint_rz_uses_radial_xy_distance(self):
        self.assertEqual(
            jog.endpoint_rz({"x": 3.0, "y": 4.0, "z": 12.0}),
            (5.0, 12.0),
        )

    def test_build_local_jacobian_uses_actual_joint_feedback(self):
        samples = {
            "s": {
                "actual_joint_delta_deg": 0.8,
                "delta_r_mm": 4.0,
                "delta_z_mm": 1.6,
            },
            "e": {
                "actual_joint_delta_deg": 1.25,
                "delta_r_mm": 2.5,
                "delta_z_mm": -5.0,
            },
        }

        matrix = jog.build_local_jacobian(samples)

        self.assertEqual(matrix, ((5.0, 2.0), (2.0, -4.0)))

    def test_build_local_jacobian_rejects_probe_without_actual_motion(self):
        samples = {
            "s": {
                "actual_joint_delta_deg": 0.2,
                "delta_r_mm": 1.0,
                "delta_z_mm": 0.0,
            },
            "e": {
                "actual_joint_delta_deg": 1.0,
                "delta_r_mm": 1.0,
                "delta_z_mm": 1.0,
            },
        }

        with self.assertRaisesRegex(jog.SafetyStop, "actual movement"):
            jog.build_local_jacobian(samples, min_actual_joint_delta_deg=0.3)

    def test_solve_joint_delta_returns_s_and_e_values(self):
        solution = jog.solve_joint_delta(
            ((5.0, 2.0), (1.0, -4.0)),
            (20.0, 0.0),
        )

        self.assertAlmostEqual(solution["s"], 40.0 / 11.0)
        self.assertAlmostEqual(solution["e"], 10.0 / 11.0)

    def test_solve_joint_delta_rejects_near_parallel_columns(self):
        with self.assertRaisesRegex(jog.SafetyStop, "singular"):
            jog.solve_joint_delta(
                ((1.0, 2.0), (2.0, 4.0001)),
                (20.0, 0.0),
            )

    def test_distance_is_limited_to_one_twenty_millimetre_segment(self):
        self.assertEqual(jog.validate_distance_mm(20.0), 20.0)
        with self.assertRaisesRegex(ValueError, "20"):
            jog.validate_distance_mm(20.1)
        with self.assertRaisesRegex(ValueError, "positive"):
            jog.validate_distance_mm(0.0)

    def test_runtime_parameters_reject_unsafe_probe_speed_and_timeout(self):
        self.assertEqual(
            jog.validate_runtime_parameters(
                probe_delta_deg=2.0,
                spd=3.0,
                acc=3.0,
                timeout_seconds=12.0,
            ),
            {
                "probe_delta_deg": 2.0,
                "spd": 3.0,
                "acc": 3.0,
                "timeout_seconds": 12.0,
            },
        )
        invalid_cases = [
            ({"probe_delta_deg": 3.1}, "probe"),
            ({"probe_delta_deg": 1.9}, "probe"),
            ({"spd": 5.1}, "speed"),
            ({"spd": 0.0}, "speed"),
            ({"acc": 5.1}, "acceleration"),
            ({"timeout_seconds": 0.9}, "timeout"),
            ({"timeout_seconds": 60.1}, "timeout"),
        ]
        defaults = {
            "probe_delta_deg": 2.0,
            "spd": 3.0,
            "acc": 3.0,
            "timeout_seconds": 12.0,
        }
        for overrides, message in invalid_cases:
            with self.subTest(overrides=overrides):
                values = {**defaults, **overrides}
                with self.assertRaisesRegex(ValueError, message):
                    jog.validate_runtime_parameters(**values)

    def test_joint_solution_rejects_more_than_six_degrees(self):
        jog.validate_joint_solution(
            {"s": 6.0, "e": -6.0},
            max_joint_delta_deg=6.0,
        )
        with self.assertRaisesRegex(jog.SafetyStop, "6"):
            jog.validate_joint_solution(
                {"s": 6.1, "e": 0.5},
                max_joint_delta_deg=6.0,
            )

    def test_build_joint_command_preserves_base_wrist_and_gripper(self):
        command = jog.build_joint_command(
            {"b": 1.0, "s": 2.0, "e": 3.0, "w": 4.0, "h": -45.0},
            {"s": 2.5, "e": 2.0},
            spd=3.0,
            acc=3.0,
        )

        self.assertEqual(
            command,
            {
                "T": 122,
                "b": 1.0,
                "s": 2.5,
                "e": 2.0,
                "w": 4.0,
                "h": -45.0,
                "spd": 3.0,
                "acc": 3.0,
            },
        )

    def test_negative_e_target_uses_five_degree_command_compensation(self):
        command_targets, compensated = jog.command_targets_for_expected(
            {"s": 10.0, "e": 22.0},
            {"s": 10.0, "e": 20.0},
            negative_e_compensation_deg=5.0,
        )

        self.assertTrue(compensated)
        self.assertEqual(command_targets, {"s": 10.0, "e": 15.0})

    def test_positive_e_target_is_not_compensated(self):
        command_targets, compensated = jog.command_targets_for_expected(
            {"s": 10.0, "e": 20.0},
            {"s": 10.0, "e": 22.0},
            negative_e_compensation_deg=5.0,
        )

        self.assertFalse(compensated)
        self.assertEqual(command_targets, {"s": 10.0, "e": 22.0})

    def test_negative_e_compensation_rejects_command_below_controller_limit(self):
        with self.assertRaisesRegex(jog.SafetyStop, "controller limit"):
            jog.command_targets_for_expected(
                {"s": 10.0, "e": -83.0},
                {"s": 10.0, "e": -86.0},
                negative_e_compensation_deg=5.0,
                command_joint_limits={
                    "s": (-90.0, 90.0),
                    "e": (-90.0, 90.0),
                },
            )

    def test_validate_final_displacement_uses_centimetre_scale_limits(self):
        accepted = jog.validate_final_displacement(
            requested_distance_mm=20.0,
            delta_r_mm=18.0,
            delta_z_mm=8.0,
            endpoint_step_mm=19.7,
        )

        self.assertTrue(accepted["accepted"])

        with self.assertRaisesRegex(jog.SafetyStop, "height"):
            jog.validate_final_displacement(
                requested_distance_mm=20.0,
                delta_r_mm=20.0,
                delta_z_mm=12.1,
                endpoint_step_mm=23.4,
            )

    def test_fifty_millimetres_is_split_into_twenty_twenty_ten(self):
        self.assertEqual(
            jog.plan_segment_distances(50.0),
            [20.0, 20.0, 10.0],
        )
        self.assertEqual(jog.plan_segment_distances(20.0), [20.0])
        with self.assertRaisesRegex(ValueError, "50"):
            jog.plan_segment_distances(50.1)

    def test_total_displacement_requires_thirty_to_seventy_millimetres(self):
        accepted = jog.validate_total_displacement(
            requested_distance_mm=50.0,
            delta_r_mm=48.0,
            delta_z_mm=12.0,
            endpoint_step_mm=49.5,
        )
        self.assertTrue(accepted["accepted"])

        with self.assertRaisesRegex(jog.SafetyStop, "total radial"):
            jog.validate_total_displacement(
                requested_distance_mm=50.0,
                delta_r_mm=12.0,
                delta_z_mm=0.0,
                endpoint_step_mm=12.0,
            )

    def test_status_invariants_reject_closed_gripper_and_preserved_joint_drift(self):
        baseline = {
            "pose_deg": {
                "b": 1.0,
                "s": 10.0,
                "e": 20.0,
                "w": 4.0,
                "h": -45.0,
            },
            "endpoint_xyz_mm": [0.0, 100.0, 50.0],
            "move": 0,
            "feedback_stable": True,
        }
        closed = copy.deepcopy(baseline)
        closed["pose_deg"]["h"] = 25.0
        with self.assertRaisesRegex(jog.SafetyStop, "gripper"):
            jog.validate_final_status(baseline, closed)

        drifted = copy.deepcopy(baseline)
        drifted["pose_deg"]["b"] += 1.0
        with self.assertRaisesRegex(jog.SafetyStop, "preserved joint"):
            jog.validate_final_status(baseline, drifted)

    def test_move_one_requires_explicit_stable_feedback(self):
        baseline = {
            "pose_deg": {
                "b": 1.0,
                "s": 10.0,
                "e": 20.0,
                "w": 4.0,
                "h": -45.0,
            },
            "endpoint_xyz_mm": [0.0, 100.0, 50.0],
            "move": 0,
            "feedback_stable": True,
        }
        stable_move_one = copy.deepcopy(baseline)
        stable_move_one["move"] = 1
        stable_move_one["feedback_stable"] = True
        jog.validate_final_status(baseline, stable_move_one)

        unstable_move_one = copy.deepcopy(stable_move_one)
        unstable_move_one["feedback_stable"] = False
        with self.assertRaisesRegex(jog.SafetyStop, "stable"):
            jog.validate_final_status(baseline, unstable_move_one)

    def test_restore_rejects_half_degree_and_two_point_five_mm_residual(self):
        baseline = {
            "pose_deg": {
                "b": 1.0,
                "s": 10.0,
                "e": 20.0,
                "w": 4.0,
                "h": -45.0,
            },
            "endpoint_xyz_mm": [0.0, 100.0, 50.0],
            "move": 0,
            "feedback_stable": True,
        }
        restored = copy.deepcopy(baseline)
        restored["pose_deg"]["s"] = 10.5
        restored["endpoint_xyz_mm"][1] = 102.5

        with self.assertRaisesRegex(jog.SafetyStop, "endpoint"):
            jog.validate_restored_status(baseline, restored)


class LinearFakeMotion:
    def __init__(self):
        self.actions = []
        self.pose = {
            "b": 1.0,
            "s": 10.0,
            "e": 20.0,
            "w": 4.0,
            "h": -45.0,
        }
        self.endpoint = [0.0, 100.0, 50.0]
        self.force_final_height_error = False
        self.height_error_injected = False

    def _status(self):
        return {
            "pose_deg": dict(self.pose),
            "endpoint_xyz_mm": list(self.endpoint),
            "move": 0,
            "feedback_stable": True,
            "raw": {
                "x": self.endpoint[0],
                "y": self.endpoint[1],
                "z": self.endpoint[2],
                "move": 0,
            },
        }

    def read_stable_status(self):
        self.actions.append(("read",))
        return copy.deepcopy(self._status())

    def move_joint_targets(self, targets, *, spd, acc):
        targets = {key: float(value) for key, value in targets.items()}
        before = dict(self.pose)
        delta_s = targets.get("s", before["s"]) - before["s"]
        delta_e = targets.get("e", before["e"]) - before["e"]
        if abs(delta_s) <= 0.01 and abs(delta_e) <= 0.01:
            return copy.deepcopy(self._status())
        self.actions.append(("move", dict(targets), float(spd), float(acc)))
        self.pose.update(targets)

        delta_r = 5.0 * delta_s + 2.0 * delta_e
        delta_z = 1.0 * delta_s - 4.0 * delta_e
        both_changed = abs(delta_s) > 0.01 and abs(delta_e) > 0.01
        radius = math.hypot(self.endpoint[0], self.endpoint[1]) + delta_r
        angle = math.atan2(self.endpoint[1], self.endpoint[0])
        self.endpoint[0] = radius * math.cos(angle)
        self.endpoint[1] = radius * math.sin(angle)
        self.endpoint[2] += delta_z
        result = self._status()
        if (
            self.force_final_height_error
            and not self.height_error_injected
            and both_changed
        ):
            self.height_error_injected = True
            result["endpoint_xyz_mm"][2] += 20.0
            result["raw"]["z"] += 20.0
        return copy.deepcopy(result)


class OversizedProbeMotion(LinearFakeMotion):
    def move_joint_targets(self, targets, *, spd, acc):
        result = super().move_joint_targets(targets, spd=spd, acc=acc)
        if len(self.actions) == 2:
            result["endpoint_xyz_mm"][1] += 35.0
            result["raw"]["y"] += 35.0
        return copy.deepcopy(result)


class StalledShoulderProbeMotion(LinearFakeMotion):
    def move_joint_targets(self, targets, *, spd, acc):
        targets = {key: float(value) for key, value in targets.items()}
        if (
            abs(targets.get("s", self.pose["s"]) - self.pose["s"]) > 0.01
            and abs(targets.get("e", self.pose["e"]) - self.pose["e"]) <= 0.01
        ):
            self.actions.append(("move", dict(targets), float(spd), float(acc)))
            return copy.deepcopy(self._status())
        return super().move_joint_targets(targets, spd=spd, acc=acc)


class PartialRestoreMotion(LinearFakeMotion):
    def move_joint_targets(self, targets, *, spd, acc):
        targets = {key: float(value) for key, value in targets.items()}
        restoring = (
            len([action for action in self.actions if action[0] == "move"]) >= 1
            and abs(targets.get("s", self.pose["s"]) - 10.0) <= 0.01
            and abs(targets.get("e", self.pose["e"]) - 20.0) <= 0.01
        )
        result = super().move_joint_targets(targets, spd=spd, acc=acc)
        if restoring:
            self.pose["s"] = 11.2
            result = self._status()
        return copy.deepcopy(result)


class UnstableFinalMotion(LinearFakeMotion):
    def __init__(self):
        super().__init__()
        self.unstable_injected = False

    def move_joint_targets(self, targets, *, spd, acc):
        before = dict(self.pose)
        result = super().move_joint_targets(targets, spd=spd, acc=acc)
        delta_s = float(targets.get("s", before["s"])) - before["s"]
        delta_e = float(targets.get("e", before["e"])) - before["e"]
        if (
            not self.unstable_injected
            and abs(delta_s) > 0.01
            and abs(delta_e) > 0.01
        ):
            self.unstable_injected = True
            result["move"] = 1
            result["feedback_stable"] = False
            result["raw"]["move"] = 1
        return copy.deepcopy(result)


class OSErrorAfterForwardMotion(LinearFakeMotion):
    def __init__(self):
        super().__init__()
        self.forward_error_raised = False

    def move_joint_targets(self, targets, *, spd, acc):
        before = dict(self.pose)
        result = super().move_joint_targets(targets, spd=spd, acc=acc)
        delta_s = float(targets.get("s", before["s"])) - before["s"]
        delta_e = float(targets.get("e", before["e"])) - before["e"]
        if (
            not self.forward_error_raised
            and abs(delta_s) > 0.01
            and abs(delta_e) > 0.01
        ):
            self.forward_error_raised = True
            raise OSError("serial disconnected after command")
        return result


class LocalCartesianRunnerTests(unittest.TestCase):
    def runner(self, motion):
        return jog.LocalCartesianJogRunner(
            motion,
            probe_delta_deg=2.0,
            max_probe_endpoint_mm=30.0,
            max_probe_z_mm=20.0,
            max_joint_delta_deg=6.0,
            spd=3.0,
            acc=3.0,
        )

    def test_dry_run_sends_no_motion_commands(self):
        motion = LinearFakeMotion()

        result = self.runner(motion).run(distance_mm=20.0, execute=False)

        self.assertTrue(result["ok"])
        self.assertEqual(result["stage"], "DRY_RUN")
        self.assertEqual(motion.actions, [])

    def test_probe_only_changes_one_joint_at_a_time_and_restores_baseline(self):
        motion = LinearFakeMotion()
        baseline_pose = dict(motion.pose)

        result = self.runner(motion).run(
            distance_mm=20.0,
            execute_probes=True,
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["stage"], "PROBES_COMPLETE")
        self.assertEqual(motion.pose, baseline_pose)
        self.assertEqual(set(result["samples"]), {"s", "e"})
        moves = [action[1] for action in motion.actions if action[0] == "move"]
        self.assertEqual(moves[0], {"s": 12.0, "e": 20.0})
        self.assertEqual(moves[1], {"s": 10.0, "e": 20.0})
        self.assertEqual(moves[2], {"s": 10.0, "e": 22.0})
        self.assertEqual(moves[3], {"s": 10.0, "e": 20.0})

    def test_oversized_probe_is_rejected_and_baseline_is_restored(self):
        motion = OversizedProbeMotion()
        baseline_pose = dict(motion.pose)

        result = self.runner(motion).run(
            distance_mm=20.0,
            execute_probes=True,
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["stage"], "PROBE")
        self.assertTrue(result["baseline_restored"])
        self.assertEqual(motion.pose, baseline_pose)

    def test_stalled_s_probe_stops_before_e_probe(self):
        motion = StalledShoulderProbeMotion()

        result = self.runner(motion).run(
            distance_mm=20.0,
            execute_probes=True,
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["stage"], "PROBE")
        moves = [action[1] for action in motion.actions if action[0] == "move"]
        self.assertEqual(
            moves,
            [
                {"s": 12.0, "e": 20.0},
            ],
        )

    def test_partial_baseline_restore_is_rejected(self):
        motion = PartialRestoreMotion()

        result = self.runner(motion).run(
            distance_mm=20.0,
            execute_probes=True,
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["stage"], "PROBE")
        self.assertFalse(result["baseline_restored"])
        self.assertIn("restore", result["reason"])

    def test_execute_moves_s_and_e_together_to_reach_forward_target(self):
        motion = LinearFakeMotion()
        start_radius = math.hypot(motion.endpoint[0], motion.endpoint[1])

        result = self.runner(motion).run(distance_mm=20.0, execute=True)

        self.assertTrue(result["ok"])
        self.assertEqual(result["stage"], "COMPLETE")
        self.assertAlmostEqual(result["actual"]["delta_r_mm"], 20.0, places=6)
        self.assertAlmostEqual(result["actual"]["delta_z_mm"], 0.0, places=6)
        self.assertAlmostEqual(
            math.hypot(motion.endpoint[0], motion.endpoint[1]) - start_radius,
            20.0,
            places=6,
        )
        final_move = [action for action in motion.actions if action[0] == "move"][-1]
        self.assertEqual(set(final_move[1]), {"s", "e"})
        self.assertNotEqual(final_move[1]["s"], 10.0)
        self.assertNotEqual(final_move[1]["e"], 20.0)

    def test_failed_final_validation_restores_baseline(self):
        motion = LinearFakeMotion()
        baseline_pose = dict(motion.pose)
        baseline_endpoint = list(motion.endpoint)
        motion.force_final_height_error = True

        result = self.runner(motion).run(distance_mm=20.0, execute=True)

        self.assertFalse(result["ok"])
        self.assertEqual(result["stage"], "FINAL_VALIDATION")
        self.assertTrue(result["baseline_restored"])
        self.assertEqual(motion.pose, baseline_pose)
        self.assertAlmostEqual(motion.endpoint[0], baseline_endpoint[0])
        self.assertAlmostEqual(motion.endpoint[1], baseline_endpoint[1])
        self.assertAlmostEqual(motion.endpoint[2], baseline_endpoint[2])

    def test_unstable_final_feedback_is_rejected_and_restored(self):
        motion = UnstableFinalMotion()
        baseline_pose = dict(motion.pose)

        result = self.runner(motion).run(distance_mm=20.0, execute=True)

        self.assertFalse(result["ok"])
        self.assertEqual(result["stage"], "FINAL_VALIDATION")
        self.assertIn("stable", result["reason"])
        self.assertTrue(result["baseline_restored"])
        self.assertEqual(motion.pose, baseline_pose)

    def test_oserror_after_forward_motion_restores_baseline(self):
        motion = OSErrorAfterForwardMotion()
        baseline_pose = dict(motion.pose)
        baseline_endpoint = list(motion.endpoint)

        result = self.runner(motion).run(distance_mm=20.0, execute=True)

        self.assertFalse(result["ok"])
        self.assertEqual(result["stage"], "FORWARD_MOVE")
        self.assertIn("serial disconnected", result["reason"])
        self.assertTrue(result["baseline_restored"])
        self.assertEqual(motion.pose, baseline_pose)
        for actual, expected in zip(motion.endpoint, baseline_endpoint):
            self.assertAlmostEqual(actual, expected)

    def test_sequence_runner_completes_twenty_twenty_ten_and_checks_total(self):
        motion = LinearFakeMotion()
        segment_runner = self.runner(motion)
        sequence = jog.LocalCartesianSequenceRunner(
            motion,
            segment_runner,
        )

        result = sequence.run(total_distance_mm=50.0, execute=True)

        self.assertTrue(result["ok"])
        self.assertEqual(result["stage"], "SEQUENCE_COMPLETE")
        self.assertEqual(
            [segment["requested_distance_mm"] for segment in result["segments"]],
            [20.0, 20.0, 10.0],
        )
        self.assertAlmostEqual(
            result["actual"]["delta_r_mm"],
            50.0,
            places=6,
        )
        self.assertAlmostEqual(
            result["actual"]["delta_z_mm"],
            0.0,
            places=6,
        )

    def test_sequence_stops_and_restores_current_segment_on_cumulative_overshoot(
        self,
    ):
        class SequenceMotion:
            def __init__(self):
                self.current = {
                    "pose_deg": {
                        "b": 1.0,
                        "s": 10.0,
                        "e": 20.0,
                        "w": 4.0,
                        "h": -45.0,
                    },
                    "endpoint_xyz_mm": [0.0, 100.0, 50.0],
                    "move": 0,
                    "feedback_stable": True,
                    "raw": {},
                }

            def read_stable_status(self):
                return copy.deepcopy(self.current)

        class OvershootSegmentRunner:
            def __init__(self, motion):
                self.motion = motion
                self.calls = []
                self.restore_calls = []

            def run(self, *, distance_mm, execute):
                self.calls.append(float(distance_mm))
                baseline = copy.deepcopy(self.motion.current)
                final_status = copy.deepcopy(baseline)
                final_status["endpoint_xyz_mm"][1] += 40.0
                self.motion.current = copy.deepcopy(final_status)
                return {
                    "ok": True,
                    "stage": "COMPLETE",
                    "requested_distance_mm": float(distance_mm),
                    "baseline": baseline,
                    "final_status": final_status,
                    "actual": {
                        "delta_r_mm": 40.0,
                        "delta_z_mm": 0.0,
                        "endpoint_step_mm": 40.0,
                    },
                    "baseline_restored": False,
                }

            def _restore_baseline(self, baseline):
                self.restore_calls.append(copy.deepcopy(baseline))
                self.motion.current = copy.deepcopy(baseline)
                return True, copy.deepcopy(baseline), ""

        motion = SequenceMotion()
        segment_runner = OvershootSegmentRunner(motion)
        sequence = jog.LocalCartesianSequenceRunner(
            motion,
            segment_runner,
        )

        result = sequence.run(total_distance_mm=50.0, execute=True)

        self.assertFalse(result["ok"])
        self.assertEqual(result["stage"], "SEQUENCE_VALIDATION_2")
        self.assertEqual(segment_runner.calls, [20.0, 20.0])
        self.assertEqual(len(segment_runner.restore_calls), 1)
        self.assertTrue(result["baseline_restored"])
        self.assertAlmostEqual(
            motion.current["endpoint_xyz_mm"][1],
            140.0,
        )


class SerialMotionAdapterTests(unittest.TestCase):
    class FakeArmModule:
        @staticmethod
        def status_to_command_degrees(status):
            return {
                "b": math.degrees(float(status["b"])),
                "s": -math.degrees(float(status["s"])),
                "e": math.degrees(float(status["e"])),
                "w": math.degrees(float(status["w"])),
            }

        @staticmethod
        def status_to_gripper_degrees(status):
            return math.degrees(float(status["t"]))

    class FakeSerialMotion:
        def __init__(self):
            self.arm = SerialMotionAdapterTests.FakeArmModule()
            self.event_log = []
            self.remembered = []
            self.status = {
                "x": 0.0,
                "y": 100.0,
                "z": 50.0,
                "b": math.radians(1.0),
                "s": math.radians(10.0),
                "e": math.radians(22.0),
                "w": math.radians(4.0),
                "t": math.radians(-45.0),
                "move": 0,
            }

        def _query_status(self):
            return dict(self.status)

        def _query_fast_status(self):
            return dict(self.status)

        def _wait_ready(self, status, *, timeout_seconds):
            return dict(status)

        def _send(self, command):
            self.event_log.append({"type": "command", "command": dict(command)})
            self.status["s"] = math.radians(-float(command["s"]))
            expected_e = (
                float(command["e"]) + 5.0
                if float(command["e"]) < 22.0
                else float(command["e"])
            )
            self.status["e"] = math.radians(expected_e)
            self.status["y"] += 3.0

        def _remember_command_pose(self, command):
            self.remembered.append(dict(command))

    class CreepingSerialMotion(FakeSerialMotion):
        def __init__(self):
            super().__init__()
            self.fast_statuses = []
            self.fast_query_count = 0

        def _send(self, command):
            self.event_log.append({"type": "command", "command": dict(command)})
            samples = [
                (10.20, 100.20),
                (10.21, 100.21),
                (10.22, 100.22),
                (10.23, 100.23),
                (12.00, 102.00),
                (12.00, 102.00),
                (12.00, 102.00),
                (12.00, 102.00),
            ]
            self.fast_statuses = []
            for feedback_s, y in samples:
                status = dict(self.status)
                status["s"] = math.radians(feedback_s)
                status["y"] = y
                status["move"] = 1
                self.fast_statuses.append(status)

        def _query_fast_status(self):
            self.fast_query_count += 1
            if not self.fast_statuses:
                return dict(self.status)
            if len(self.fast_statuses) > 1:
                return dict(self.fast_statuses.pop(0))
            return dict(self.fast_statuses[0])

    class MovingIdleSerialMotion(FakeSerialMotion):
        def __init__(self):
            super().__init__()
            self.fast_query_count = 0
            self.fast_statuses = []
            for feedback_s, y in (
                (10.20, 100.20),
                (11.00, 101.00),
                (12.00, 102.00),
                (12.00, 102.00),
                (12.00, 102.00),
                (12.00, 102.00),
            ):
                status = dict(self.status)
                status["s"] = math.radians(feedback_s)
                status["y"] = y
                status["move"] = 0
                self.fast_statuses.append(status)

        def _query_fast_status(self):
            self.fast_query_count += 1
            if len(self.fast_statuses) > 1:
                return dict(self.fast_statuses.pop(0))
            return dict(self.fast_statuses[0])

    def test_adapter_sends_compensated_t122_and_returns_feedback_pose(self):
        serial_motion = self.FakeSerialMotion()
        adapter = jog.SerialMotionAdapter(
            serial_motion,
            spd=3.0,
            acc=3.0,
            timeout_seconds=2.0,
            poll_seconds=0.0,
        )

        result = adapter.move_joint_targets(
            {"s": -10.0, "e": 20.0},
            spd=3.0,
            acc=3.0,
        )

        command = serial_motion.event_log[0]["command"]
        self.assertEqual(command["T"], 122)
        self.assertEqual(command["s"], -10.0)
        self.assertEqual(command["e"], 15.0)
        self.assertEqual(command["b"], 1.0)
        self.assertEqual(command["w"], 4.0)
        self.assertEqual(command["h"], -45.0)
        self.assertAlmostEqual(result["pose_deg"]["s"], -10.0)
        self.assertAlmostEqual(result["pose_deg"]["e"], 20.0)
        self.assertTrue(result["feedback_stable"])

    def test_adapter_does_not_treat_slow_move_one_creep_as_stable(self):
        serial_motion = self.CreepingSerialMotion()
        adapter = jog.SerialMotionAdapter(
            serial_motion,
            spd=3.0,
            acc=3.0,
            timeout_seconds=2.0,
            poll_seconds=0.0,
        )

        result = adapter.move_joint_targets(
            {"s": -12.0, "e": 22.0},
            spd=3.0,
            acc=3.0,
        )

        self.assertGreaterEqual(serial_motion.fast_query_count, 6)
        self.assertAlmostEqual(result["pose_deg"]["s"], -12.0)
        self.assertEqual(result["move"], 1)
        self.assertTrue(result["feedback_stable"])

    def test_adapter_does_not_trust_move_zero_while_feedback_changes(self):
        serial_motion = self.MovingIdleSerialMotion()
        adapter = jog.SerialMotionAdapter(
            serial_motion,
            spd=3.0,
            acc=3.0,
            timeout_seconds=2.0,
            poll_seconds=0.0,
        )

        result = adapter.read_stable_status()

        self.assertGreaterEqual(serial_motion.fast_query_count, 6)
        self.assertAlmostEqual(result["pose_deg"]["s"], -12.0)
        self.assertEqual(result["move"], 0)
        self.assertTrue(result["feedback_stable"])


class LocalCartesianCliTests(unittest.TestCase):
    def test_cli_defaults_to_twenty_millimetres_and_dry_run(self):
        args = jog.build_parser().parse_args([])

        self.assertEqual(args.distance_mm, 20.0)
        self.assertEqual(args.probe_delta_deg, 2.0)
        self.assertFalse(args.execute)
        self.assertFalse(args.execute_probes)
        self.assertIsNone(args.total_distance_mm)

    def test_write_outputs_creates_json_and_summary(self):
        payload = {
            "ok": True,
            "stage": "DRY_RUN",
            "requested_distance_mm": 20.0,
            "samples": {},
        }

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            jog.write_outputs(output_dir, payload)
            written = json.loads(
                (output_dir / "result.json").read_text(encoding="utf-8")
            )

            self.assertEqual(written["stage"], "DRY_RUN")
            self.assertTrue((output_dir / "summary.md").exists())

    def test_run_cli_records_sent_joint_targets(self):
        motion = LinearFakeMotion()

        with tempfile.TemporaryDirectory() as directory:
            args = jog.build_parser().parse_args(
                [
                    "--execute",
                    "--distance-mm",
                    "20",
                    "--output-dir",
                    directory,
                ]
            )

            exit_code, payload, _ = jog.run_cli(args, motion=motion)

        self.assertEqual(exit_code, 0)
        self.assertGreaterEqual(len(payload["commands_sent"]), 5)
        self.assertEqual(
            set(payload["commands_sent"][-1]),
            {"s", "e"},
        )

    def test_run_cli_can_execute_fifty_millimetre_sequence(self):
        motion = LinearFakeMotion()

        with tempfile.TemporaryDirectory() as directory:
            args = jog.build_parser().parse_args(
                [
                    "--execute",
                    "--total-distance-mm",
                    "50",
                    "--output-dir",
                    directory,
                ]
            )

            exit_code, payload, _ = jog.run_cli(args, motion=motion)

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["stage"], "SEQUENCE_COMPLETE")
        self.assertEqual(len(payload["segments"]), 3)
        self.assertAlmostEqual(payload["actual"]["delta_r_mm"], 50.0)


if __name__ == "__main__":
    unittest.main()

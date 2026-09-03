from __future__ import annotations

import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mission_lite3.arm.lite_arm import ArmTaskResult, LiteArmController
from mission_lite3.arm.runtime import arm_grasp, arm_task, place_controller
from mission_lite3.arm.runtime import test as arm_test
from mission_lite3.config_loader import load_config


RUNTIME_DIR = Path("mission_lite3/arm/runtime")


class RecordingReadyMotion:
    def __init__(self, current_e: float = 60.0):
        self.current_e = current_e
        self.calls = []

    def set_abort_checker(self, _checker):
        return None

    def open_gripper(self, *, angle, spd, acc):
        self.calls.append(("open_gripper", angle, spd, acc))

    def current_pose_degrees(self):
        return {"b": 0.0, "s": 0.0, "e": self.current_e, "w": 0.0}

    def move_joints_with_expected_targets(
        self,
        command_targets,
        expected_targets,
        *,
        spd,
        acc,
        tolerance_degrees,
    ):
        self.calls.append(
            (
                "move_joints_with_expected_targets",
                dict(command_targets),
                dict(expected_targets),
                spd,
                acc,
                tolerance_degrees,
            )
        )

    def move_joints(self, joints, *, spd, acc, tolerance_degrees):
        self.calls.append(
            ("move_joints", dict(joints), spd, acc, tolerance_degrees)
        )


class ArmRuntimeSourceContractTest(unittest.TestCase):
    def test_runtime_manifest_files_are_packaged(self):
        manifest = {
            "arm_grasp.py",
            "arm_task.py",
            "build_final_view_references.py",
            "capture_grasp_samples.py",
            "local_cartesian_jog.py",
            "save_grasp_reference_image.py",
            "strip_detection.py",
            "strip_detector.py",
            "teach_grasp_pose.py",
            "test.py",
            "camera_calibration.json",
            "grasp_reference_square_face.json",
            "place_reference.json",
            "strip_detector_grasp_config.json",
        }
        self.assertEqual(
            sorted(manifest),
            sorted(path.name for path in RUNTIME_DIR.iterdir() if path.name in manifest),
        )

    def test_source_parser_defaults_and_commands_are_preserved(self):
        parser = arm_task.build_parser()
        args = parser.parse_args(["grasp"])
        arm_task._resolve_alignment_defaults(args)  # noqa: SLF001
        self.assertEqual(args.spd, arm_grasp.DEFAULT_SPEED)
        self.assertEqual(args.acc, arm_grasp.DEFAULT_ACCELERATION)
        self.assertEqual(args.max_align_steps, arm_task.AUTO_ALIGN_STEPS)
        self.assertEqual(args.max_jog_deg, arm_task.AUTO_MAX_JOG_DEG)
        self.assertFalse(hasattr(args, "hardware_state_file"))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["moving-pose"])

    def test_source_grasp_parameters_are_preserved(self):
        reference = json.loads(
            (RUNTIME_DIR / "grasp_reference_square_face.json").read_text(
                encoding="utf-8"
            )
        )
        detector = json.loads(
            (RUNTIME_DIR / "strip_detector_grasp_config.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(arm_grasp.GRASP_READY_NEGATIVE_E_COMPENSATION_DEG, 5.0)
        self.assertEqual(reference["square_face_target"]["center_px"], [590.0, 339.5])
        self.assertTrue(reference["initial_red_search"]["enabled"])
        self.assertEqual(
            reference["visual_servo"]["square_face"]["horizontal_tolerance_px"],
            20.0,
        )
        self.assertEqual(detector["roi"], [0.0, 0.0, 1.0, 0.95])
        self.assertEqual(
            detector["colors"]["red"],
            [{"lower": [168, 45, 50], "upper": [179, 255, 255]}],
        )

    def test_transport_is_the_source_grasp_ready_sequence(self):
        task = arm_task.ArmTask(dry_run=True)
        result = task.transport()
        self.assertTrue(result["ok"])
        self.assertEqual(result["stage"], "GRASP_READY")
        self.assertEqual(result["plan"][0]["stage"], "OPEN_GRIPPER")
        self.assertEqual(
            result["plan"][1]["joints_deg"],
            arm_task.DEFAULT_GRASP_READY_POSE,
        )

    def test_negative_elbow_ready_move_uses_source_five_degree_compensation(self):
        motion = RecordingReadyMotion(current_e=60.0)
        task = arm_task.ArmTask(motion=motion, dry_run=False)
        result = task.grasp_ready()
        ready = result["plan"][1]
        self.assertEqual(ready["e_negative_compensation_deg"], 5.0)
        self.assertAlmostEqual(
            ready["command_joints_deg"]["e"],
            arm_task.DEFAULT_GRASP_READY_POSE["e"] - 5.0,
        )
        self.assertEqual(motion.calls[1][0], "move_joints_with_expected_targets")

    def test_source_place_slots_and_dry_run_hold_state_are_preserved(self):
        reference = place_controller.load_place_reference(
            RUNTIME_DIR / "place_reference.json"
        )
        controller = place_controller.PlaceController(reference, motion=mock.Mock())
        result_a = controller.place("A", object_held=False, dry_run=True)
        result_b = controller.place("B", object_held=True, dry_run=True)
        self.assertTrue(result_a.ok)
        self.assertTrue(result_a.object_held)
        self.assertNotEqual(
            result_a.plan[1]["joints_deg"]["b"],
            result_b.plan[1]["joints_deg"]["b"],
        )
        self.assertEqual(
            [step["stage"] for step in result_a.plan[3:]],
            [
                "MOVE_S_TO_RETREAT_CLEARANCE",
                "RETRACT_SHOULDER_TO_HALF",
                "RETRACT_SHOULDER_AND_ELBOW_TO_MOVING_POSE",
                "COMPLETE_RETREAT_TO_MOVING_POSE",
            ],
        )
        retreat = reference["slots"]["A"]["retreat_joints_deg"]
        moving_record = arm_test.read_pose_record(RUNTIME_DIR / "moving_pose.json")
        moving_command = arm_test.build_pose_command(
            moving_record,
            spd=arm_test.MOVING_POSE_SPEED,
            acc=arm_test.MOVING_POSE_ACCELERATION,
        )
        for joint in arm_test.JOINT_KEYS:
            self.assertAlmostEqual(retreat[joint], moving_command[joint])
        expected_half_s = 50.0 + (retreat["s"] - 50.0) * 0.5
        self.assertEqual(result_a.plan[4]["joints_deg"], {"s": expected_half_s})
        self.assertEqual(
            result_a.plan[5]["joints_deg"],
            {"s": retreat["s"], "e": retreat["e"]},
        )
        self.assertEqual(
            result_a.plan[6]["joints_deg"],
            {"b": retreat["b"], "w": retreat["w"]},
        )

    def test_place_starts_elbow_after_shoulder_reaches_half_retreat(self):
        reference = place_controller.load_place_reference(
            RUNTIME_DIR / "place_reference.json"
        )
        motion = mock.Mock()
        motion.current_pose_degrees.return_value = {
            "b": -12.0,
            "s": 50.0,
            "e": -66.0,
            "w": -2.0,
        }
        controller = place_controller.PlaceController(reference, motion=motion)

        result = controller.place("A", object_held=True)

        self.assertTrue(result.ok)
        retreat = reference["slots"]["A"]["retreat_joints_deg"]
        expected_half_s = 50.0 + (retreat["s"] - 50.0) * 0.5
        retreat_calls = motion.move_joints.call_args_list[2:]
        self.assertEqual(retreat_calls[0].args[0], {"s": 50.0})
        self.assertEqual(retreat_calls[1].args[0], {"s": expected_half_s})
        self.assertEqual(
            retreat_calls[2].args[0],
            {"s": retreat["s"], "e": retreat["e"]},
        )
        self.assertEqual(
            retreat_calls[3].args[0],
            {"b": retreat["b"], "w": retreat["w"]},
        )
        self.assertEqual(retreat_calls[1].kwargs, {"spd": 30.0, "acc": 30.0})
        self.assertEqual(retreat_calls[2].kwargs, {"spd": 30.0, "acc": 30.0})

    def test_place_never_retracts_elbow_when_shoulder_half_retreat_fails(self):
        reference = place_controller.load_place_reference(
            RUNTIME_DIR / "place_reference.json"
        )
        motion = mock.Mock()
        motion.current_pose_degrees.return_value = {
            "b": -12.0,
            "s": 50.0,
            "e": -66.0,
            "w": -2.0,
        }
        motion.move_joints.side_effect = [
            None,
            None,
            None,
            RuntimeError("shoulder retreat blocked"),
        ]
        controller = place_controller.PlaceController(reference, motion=motion)

        result = controller.place("A", object_held=True)

        self.assertFalse(result.ok)
        self.assertEqual(result.stage, "RETRACT_SHOULDER_TO_HALF")
        self.assertIn("shoulder retreat blocked", result.reason)
        self.assertTrue(result.released)
        self.assertFalse(result.object_held)
        self.assertEqual(len(motion.move_joints.call_args_list), 4)
        last_joints = motion.move_joints.call_args_list[-1].args[0]
        self.assertEqual(
            last_joints,
            {
                "s": 50.0
                + (
                    reference["slots"]["A"]["retreat_joints_deg"]["s"]
                    - 50.0
                )
                * 0.5
            },
        )

    def test_grasp_cargo_failure_preserves_object_held(self):
        machine = object.__new__(arm_grasp.ArmGraspStateMachine)
        machine.motion = mock.Mock()
        machine.final_spd = 40.0
        machine.final_acc = 40.0
        machine._execute_transport_pose = mock.Mock(  # noqa: SLF001
            side_effect=arm_grasp.TerminalStepError(
                arm_grasp.CARGO_POSE_STAGE,
                "cargo pose failed: shoulder blocked",
            )
        )
        terminal_plan = [
            {"stage": "CLOSE_GRIPPER", "close_gripper_h": 45.0},
            {
                "stage": arm_grasp.CARGO_POSE_STAGE,
                "joints_deg": {"s": -20.0, "e": 30.0},
            },
        ]

        result = machine._finish_no_red_fallback_to_cargo(  # noqa: SLF001
            [],
            {"center_x": 320.0},
            terminal_plan,
        )

        self.assertFalse(result.ok)
        self.assertEqual(result.stage, arm_grasp.CARGO_POSE_STAGE)
        self.assertTrue(result.object_held)
        machine.motion.close_gripper.assert_called_once()

    def test_square_face_near_top_edge_corrects_vertical_before_horizontal(self):
        machine = object.__new__(arm_grasp.ArmGraspStateMachine)
        machine.reference = arm_grasp.default_grasp_reference()
        machine.max_jog_deg = 6.0
        machine.grasp_window_size_ratio_tolerance = 0.2

        step = machine._square_face_alignment_step(  # noqa: SLF001
            {
                "center_px": [445.0, 81.0],
                "size_px": [230.0, 162.0],
                "angle_deg": 0.0,
                "angle_reliable": True,
            }
        )

        self.assertEqual(step["stage"], "VISUAL_ALIGN")
        self.assertEqual(step["joint"], "s")
        self.assertEqual(step["feedback"], "square_face_above")
        self.assertLess(step["delta_deg"], 0.0)

    def test_real_abort_reports_manual_recovery_instead_of_false_success(self):
        motion = mock.Mock()
        task = arm_task.ArmTask(motion=motion, dry_run=False)
        task.object_held = True

        result = task.abort()

        self.assertFalse(result["ok"])
        self.assertEqual(result["stage"], "MANUAL_RECOVERY_REQUIRED")
        self.assertTrue(result["object_held"])
        self.assertIn("manual recovery required", result["reason"])
        motion.abort.assert_not_called()

    def test_adapter_passes_only_source_runtime_options(self):
        controller = LiteArmController(load_config())
        argv = controller._runtime_base_argv(  # noqa: SLF001
            Path("/tmp/arm-result.json"),
            include_camera=True,
        )
        for unsupported in (
            "--hardware-state-file",
            "--moving-pose",
            "--transport-spd",
            "--transport-acc",
        ):
            self.assertNotIn(unsupported, argv)
        arm_task.build_parser().parse_args([*argv, "preflight"])

    def test_adapter_derives_release_from_source_place_result(self):
        result = ArmTaskResult.from_payload(
            {"ok": True, "stage": "DONE", "slot": "C", "object_held": False}
        )
        self.assertTrue(result.released)
        self.assertFalse(result.object_held)

    def test_stow_uses_saved_moving_pose_instead_of_transport(self):
        controller = LiteArmController(load_config())
        moving = ArmTaskResult.success("MOVING_POSE")
        with mock.patch.object(
            controller,
            "moving_pose",
            return_value=moving,
        ) as run_moving, mock.patch.object(
            controller,
            "_run_runtime_task",
        ) as run_runtime:
            result = controller.stow()

        self.assertIs(result, moving)
        run_moving.assert_called_once_with()
        run_runtime.assert_not_called()

    def test_project_cli_exposes_moving_pose_without_extending_source_parser(self):
        from mission_lite3.arm import run_arm_task

        args = run_arm_task.build_parser().parse_args(["moving-pose"])
        self.assertEqual(args.command, "moving-pose")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            arm_task.build_parser().parse_args(["moving-pose"])

    def test_project_cli_honors_moving_pose_dry_run(self):
        from mission_lite3.arm import run_arm_task

        with mock.patch.object(
            run_arm_task,
            "LiteArmController",
        ) as controller_type, contextlib.redirect_stdout(io.StringIO()):
            controller_type.return_value.moving_pose.return_value = (
                ArmTaskResult.success("MOVING_POSE")
            )
            exit_code = run_arm_task.main(["moving-pose", "--dry-run"])

        self.assertEqual(exit_code, 0)
        self.assertTrue(controller_type.call_args.kwargs["dry_run"])
        controller_type.return_value.moving_pose.assert_called_once_with()

    def test_moving_pose_adapter_uses_source_record_and_source_speed(self):
        record = arm_test.read_pose_record(RUNTIME_DIR / "moving_pose.json")
        status = {
            joint: math.radians(float(record["joints_feedback_deg"][joint]))
            for joint in arm_test.JOINT_KEYS
        }
        status.update(
            {
                "t": math.radians(float(record["gripper_deg"])),
                "move": 0,
            }
        )

        class FakeMovingMotion:
            instances = []

            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.commands = []
                self.__class__.instances.append(self)

            def _query_status(self):
                return dict(status)

            def _query_fast_status(self):
                return dict(status)

            def _wait_ready(self, current):
                return current

            def _send(self, command):
                self.commands.append(dict(command))

            def _remember_command_pose(self, _command):
                return None

            def _joint_already_within_target(
                self,
                _status,
                _joint,
                _target,
                _tolerance,
            ):
                return True

        controller = LiteArmController(load_config())
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            "os.environ",
            {"ARM_RESULT_DIR": tmp},
        ), mock.patch.object(
            arm_task,
            "ArmTestSerialMotion",
            FakeMovingMotion,
        ):
            result = controller._run_source_moving_pose()  # noqa: SLF001

        self.assertTrue(result.ok)
        command = FakeMovingMotion.instances[-1].commands[0]
        self.assertEqual(command["spd"], arm_test.MOVING_POSE_SPEED)
        self.assertEqual(command["acc"], arm_test.MOVING_POSE_ACCELERATION)
        self.assertAlmostEqual(command["h"], float(record["gripper_deg"]))
        self.assertAlmostEqual(
            command["s"],
            -float(record["joints_feedback_deg"]["s"]),
        )

    def test_prepared_pose_uses_source_skip_flag_once(self):
        controller = LiteArmController(load_config())
        ready = ArmTaskResult.success("GRASP_READY")
        grasped = ArmTaskResult(True, "DONE", object_held=True)
        with mock.patch.object(
            controller,
            "_run_runtime_task",
            side_effect=[ready, grasped],
        ) as run:
            controller.camera_pose()
            controller.grasp_red_bar(260.0)
        self.assertEqual(
            run.call_args_list[1].kwargs["pre_command_args"],
            ("--skip-grasp-ready",),
        )
        self.assertFalse(controller._grasp_ready_prepared)  # noqa: SLF001


if __name__ == "__main__":
    unittest.main()

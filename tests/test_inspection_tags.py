from __future__ import annotations

import contextlib
import io
import math
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from mission_lite3.config_loader import ConfigError, load_config, validate_config
from mission_lite3.inspection_tags import (
    InspectionTagDetector,
    InspectionTagObservation,
    median_tag_observation,
    plan_inspection_tag_correction,
    station_tag_target,
)
from mission_lite3.mission import LargeQuadrupedMission, MissionAbort


def observation(
    tag_id: int,
    center_x_px: float,
    edge_px: float,
) -> InspectionTagObservation:
    return InspectionTagObservation(
        tag_id=tag_id,
        center_x_px=center_x_px,
        center_y_px=430.0,
        edge_px=edge_px,
        distance_m=None,
        corners=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
    )


class InspectionTagTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config()
        self.tag_config = self.config['inspection']['tag_localization']

    def test_station_ids_are_fixed_and_do_not_encode_letters(self) -> None:
        self.assertEqual(
            [
                station_tag_target(
                    self.tag_config,
                    f'inspection_stop_{index}',
                ).tag_id
                for index in range(1, 5)
            ],
            [0, 1, 2, 3],
        )
        self.assertNotIn('B', self.tag_config['station_tag_ids'])

    def test_config_rejects_changed_station_id_mapping(self) -> None:
        config = load_config()
        config['inspection']['tag_localization']['station_tag_ids'][
            'inspection_stop_1'
        ] = 3
        with self.assertRaisesRegex(ConfigError, 'station_tag_ids'):
            validate_config(config)

    def test_median_observation_filters_frame_jitter(self) -> None:
        result = median_tag_observation(
            [
                observation(2, 708.0, 43.0),
                observation(2, 710.0, 44.0),
                observation(2, 740.0, 60.0),
            ]
        )
        self.assertEqual(result.tag_id, 2)
        self.assertEqual(result.center_x_px, 710.0)
        self.assertEqual(result.edge_px, 44.0)

    def test_wrong_id_is_never_used_for_correction(self) -> None:
        target = station_tag_target(self.tag_config, 'inspection_stop_1')
        result = plan_inspection_tag_correction(
            observation(3, target.center_x_px, target.edge_px),
            target,
            self.tag_config,
            focal_x_px=600.0,
        )
        self.assertEqual((result.kind, result.reason), ('fail', 'unexpected_tag_id'))

    def test_forward_and_strafe_corrections_are_bounded(self) -> None:
        target = station_tag_target(self.tag_config, 'inspection_stop_1')
        forward = plan_inspection_tag_correction(
            observation(0, target.center_x_px, 24.0),
            target,
            self.tag_config,
            focal_x_px=600.0,
        )
        self.assertEqual(forward.kind, 'forward')
        self.assertGreater(forward.distance_m, 0.0)
        self.assertLessEqual(forward.distance_m, 0.10)

        strafe = plan_inspection_tag_correction(
            observation(0, target.center_x_px + 60.0, target.edge_px),
            target,
            self.tag_config,
            focal_x_px=600.0,
        )
        self.assertEqual(strafe.kind, 'strafe')
        self.assertLess(strafe.distance_m, 0.0)
        self.assertLessEqual(abs(strafe.distance_m), 0.10)

    def test_large_horizontal_error_is_corrected_before_tag_scale(self) -> None:
        target = station_tag_target(self.tag_config, 'inspection_stop_3')
        correction = plan_inspection_tag_correction(
            observation(2, target.center_x_px - 500.0, target.edge_px + 40.0),
            target,
            {
                **self.tag_config,
                **self.tag_config['station_overrides']['inspection_stop_3'],
            },
            focal_x_px=600.0,
        )

        self.assertEqual(correction.kind, 'strafe')
        self.assertEqual(correction.reason, 'horizontal_center_error')
        self.assertAlmostEqual(correction.distance_m, 0.20)

    def test_stop_3_uses_station_specific_twenty_centimetre_bound(self) -> None:
        self.assertEqual(
            self.tag_config['station_overrides']['inspection_stop_3'][
                'max_iterations'
            ],
            5,
        )
        mission = object.__new__(LargeQuadrupedMission)
        mission.config = self.config
        mission.vision = SimpleNamespace(
            inspection_tag_detector=SimpleNamespace(
                available=True,
                focal_x_px=600.0,
                unavailable_reason='',
            )
        )
        mission.front_camera = mock.Mock()
        mission.front_camera.ensure_running.return_value = True
        mission.motion = mock.Mock()
        target = station_tag_target(self.tag_config, 'inspection_stop_3')
        mission._sample_expected_inspection_tag = mock.Mock(
            side_effect=[
                (
                    observation(2, target.center_x_px - 500.0, target.edge_px + 40.0),
                    set(),
                    None,
                ),
                (
                    observation(2, target.center_x_px, target.edge_px),
                    set(),
                    None,
                ),
            ]
        )

        moves = mission._align_inspection_tag('inspection_stop_3')

        mission.motion.strafe_distance.assert_called_once_with(
            0.20,
            speed_mps=0.06,
        )
        mission.motion.go_distance.assert_not_called()
        self.assertEqual(moves, [('strafe', 0.20)])

    def test_inspection_restores_alignment_moves_in_reverse_order(self) -> None:
        mission = LargeQuadrupedMission(
            self.config,
            dry_run=True,
            skip_arm=True,
        )
        mission.context.dry_run = False
        mission.motion = mock.Mock()
        mission.state_reader = mock.Mock()
        anchor = (1.0, 2.0, 0.25)
        mission.state_reader.pose.side_effect = [anchor, anchor]

        def collect(_stop_name, _default_results, *, alignment_moves):
            alignment_moves.extend([
                ('forward', 0.10),
                ('strafe', -0.20),
            ])

        mission._collect_inspection = mock.Mock(side_effect=collect)

        mission._collect_inspection_at_route_anchor(
            'inspection_stop_1',
            [('A', '偏低')],
        )

        self.assertEqual(
            mission.motion.method_calls,
            [
                mock.call.strafe_distance(0.20, speed_mps=0.06),
                mock.call.go_distance(-0.10, speed_mps=0.08),
            ],
        )
        self.assertEqual(mission.state_reader.pose.call_count, 2)

    def test_inspection_exception_still_restores_route_anchor(self) -> None:
        mission = LargeQuadrupedMission(
            self.config,
            dry_run=True,
            skip_arm=True,
        )
        mission.context.dry_run = False
        mission.motion = mock.Mock()
        mission.state_reader = mock.Mock()
        anchor = (1.0, 2.0, 0.25)
        mission.state_reader.pose.side_effect = [anchor, anchor]

        def collect(_stop_name, _default_results, *, alignment_moves):
            alignment_moves.append(('forward', 0.10))
            raise RuntimeError('camera failed')

        mission._collect_inspection = mock.Mock(side_effect=collect)

        with self.assertRaisesRegex(RuntimeError, 'camera failed'):
            mission._collect_inspection_at_route_anchor(
                'inspection_stop_2',
                [('B', '正常')],
            )

        mission.motion.go_distance.assert_called_once_with(
            -0.10,
            speed_mps=0.08,
        )
        self.assertEqual(mission.state_reader.pose.call_count, 2)

    def test_route_anchor_residual_is_corrected_and_verified(self) -> None:
        mission = LargeQuadrupedMission(
            self.config,
            dry_run=True,
            skip_arm=True,
        )
        mission.context.dry_run = False
        mission.motion = mock.Mock()
        mission.state_reader = mock.Mock()
        anchor = (1.0, 2.0, 0.0)
        current = (1.08, 1.90, math.radians(6.0))
        mission.state_reader.pose.side_effect = [anchor, current, anchor]
        mission._collect_inspection = mock.Mock()

        mission._collect_inspection_at_route_anchor(
            'inspection_stop_4',
            [('D', '正常')],
        )

        mission.motion.turn_by.assert_called_once()
        self.assertAlmostEqual(
            mission.motion.turn_by.call_args.args[0],
            -math.radians(6.0),
        )
        mission.motion.go_distance.assert_called_once()
        self.assertAlmostEqual(
            mission.motion.go_distance.call_args.args[0],
            -0.08,
        )
        self.assertEqual(
            mission.motion.go_distance.call_args.kwargs,
            {'speed_mps': 0.08},
        )
        mission.motion.strafe_distance.assert_called_once()
        self.assertAlmostEqual(
            mission.motion.strafe_distance.call_args.args[0],
            0.10,
        )
        self.assertEqual(
            mission.motion.strafe_distance.call_args.kwargs,
            {'speed_mps': 0.06},
        )
        self.assertEqual(mission.state_reader.pose.call_count, 3)

    def test_route_anchor_verification_rejects_persistent_residual(self) -> None:
        mission = LargeQuadrupedMission(
            self.config,
            dry_run=True,
            skip_arm=True,
        )
        mission.context.dry_run = False
        mission.motion = mock.Mock()
        mission.state_reader = mock.Mock()
        anchor = (0.0, 0.0, 0.0)
        current = (0.10, 0.0, 0.0)
        mission.state_reader.pose.side_effect = [
            anchor,
            current,
            current,
            current,
        ]
        mission._collect_inspection = mock.Mock()

        with self.assertRaisesRegex(
            MissionAbort,
            'failed to restore inspection route anchor',
        ):
            mission._collect_inspection_at_route_anchor(
                'inspection_stop_3',
                [('C', '偏高')],
            )

        self.assertEqual(mission.motion.go_distance.call_count, 2)

    def test_dry_run_and_unknown_stop_do_not_require_odometry(self) -> None:
        mission = LargeQuadrupedMission(
            self.config,
            dry_run=True,
            skip_arm=True,
        )
        mission.state_reader = mock.Mock()
        mission._collect_inspection = mock.Mock()

        mission._collect_inspection_at_route_anchor(
            'inspection_stop_1',
            [('A', '偏低')],
        )
        mission.context.dry_run = False
        mission._collect_inspection_at_route_anchor(
            'inspection_stop_test',
            [('A', '偏低')],
        )

        mission.state_reader.pose.assert_not_called()

    def test_tag_scale_lowers_letter_anchor_threshold(self) -> None:
        self.assertEqual(
            InspectionTagDetector.recommended_letter_min_height_px(
                [observation(2, 710.0, 44.0)]
            ),
            21,
        )

    def test_missing_expected_tag_does_not_move_or_raise(self) -> None:
        mission = object.__new__(LargeQuadrupedMission)
        mission.config = self.config
        mission.vision = SimpleNamespace(
            inspection_tag_detector=SimpleNamespace(
                available=True,
                focal_x_px=600.0,
                unavailable_reason='',
            )
        )
        mission.front_camera = mock.Mock()
        mission.front_camera.ensure_running.return_value = True
        mission.motion = mock.Mock()
        mission._sample_expected_inspection_tag = mock.Mock(
            return_value=(None, {3}, None)
        )

        with contextlib.redirect_stdout(io.StringIO()) as output:
            mission._align_inspection_tag('inspection_stop_1')

        mission.motion.go_distance.assert_not_called()
        mission.motion.strafe_distance.assert_not_called()
        self.assertIn('observed_ids=[3]', output.getvalue())
        self.assertIn('continue without correction', output.getvalue())

    def test_correction_motion_error_is_downgraded(self) -> None:
        mission = object.__new__(LargeQuadrupedMission)
        mission.config = self.config
        mission.vision = SimpleNamespace(
            inspection_tag_detector=SimpleNamespace(
                available=True,
                focal_x_px=600.0,
                unavailable_reason='',
            )
        )
        mission.front_camera = mock.Mock()
        mission.front_camera.ensure_running.return_value = True
        mission.motion = mock.Mock()
        mission.motion.go_distance.side_effect = RuntimeError('motion rejected')
        target = station_tag_target(self.tag_config, 'inspection_stop_1')
        mission._sample_expected_inspection_tag = mock.Mock(
            return_value=(
                observation(0, target.center_x_px, 24.0),
                set(),
                None,
            )
        )

        with contextlib.redirect_stdout(io.StringIO()) as output:
            mission._align_inspection_tag('inspection_stop_1')

        mission.motion.go_distance.assert_called_once()
        self.assertIn('continue inspection from current pose', output.getvalue())

    def test_config_rejects_non_integer_route_anchor_pass_count(self) -> None:
        config = load_config()
        config['inspection']['tag_localization'][
            'return_max_correction_passes'
        ] = 1.5

        with self.assertRaisesRegex(ConfigError, 'must be an integer'):
            validate_config(config)

    def test_detector_reads_apriltag_36h11_with_two_border_bits(self) -> None:
        detector = InspectionTagDetector(self.config)
        if not detector.available:
            self.skipTest(detector.unavailable_reason)
        cv2 = detector._cv2
        aruco = cv2.aruco
        dictionary = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)
        marker = aruco.generateImageMarker(dictionary, 2, 120, borderBits=2)
        frame = np.full((720, 1280, 3), 255, dtype=np.uint8)
        frame[300:420, 580:700] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)

        detected = detector.detect(frame)

        self.assertEqual([item.tag_id for item in detected], [2])
        self.assertGreater(detected[0].edge_px, 100.0)


if __name__ == '__main__':
    unittest.main()

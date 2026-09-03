from __future__ import annotations

import unittest

import numpy as np

from mission_lite3.tools.wide_camera_calibration import (
    board_view,
    is_diverse_view,
    parse_pattern,
    straightness_error,
)


def synthetic_corners(offset_x: float = 300.0, offset_y: float = 140.0) -> np.ndarray:
    columns, rows = (8, 11)
    points = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2).astype(np.float64)
    points[:, 0] = offset_x + points[:, 0] * 32.0
    points[:, 1] = offset_y + points[:, 1] * 28.0
    return points.reshape(-1, 1, 2)


class WideCameraCalibrationTests(unittest.TestCase):
    def test_parse_pattern_accepts_common_separators(self) -> None:
        self.assertEqual(parse_pattern("8x11"), (8, 11))
        self.assertEqual(parse_pattern("8*11"), (8, 11))
        self.assertEqual(parse_pattern("8,11"), (8, 11))

    def test_duplicate_view_is_rejected_but_shifted_view_is_accepted(self) -> None:
        first = board_view(
            synthetic_corners(),
            (8, 11),
            (1280, 720),
            100.0,
        )
        duplicate = board_view(
            synthetic_corners(302.0, 141.0),
            (8, 11),
            (1280, 720),
            105.0,
        )
        shifted = board_view(
            synthetic_corners(520.0, 300.0),
            (8, 11),
            (1280, 720),
            110.0,
        )

        self.assertFalse(is_diverse_view(duplicate, [first]))
        self.assertTrue(is_diverse_view(shifted, [first]))

    def test_zero_distortion_preserves_straight_grid_lines(self) -> None:
        camera_matrix = np.array(
            [[600.0, 0.0, 640.0], [0.0, 600.0, 360.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        error = straightness_error(
            [synthetic_corners()],
            (8, 11),
            camera_matrix,
            np.zeros(5, dtype=np.float64),
            fisheye=False,
        )
        self.assertLess(error, 1e-6)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import cv2
import numpy as np

from mission_lite3.wide_camera import (
    _select_widest_line_cluster,
    detect_box_parallel,
)


class WideBoxParallelTests(unittest.TestCase):
    def test_prefers_full_box_seam_over_many_short_right_side_fragments(self) -> None:
        candidates = [
            (220.0, 1.2, (310, 550, 530, 555), 558.0),
            (340.0, 1.2, (529, 555, 869, 562), 558.0),
            (168.0, 0.0, (692, 580, 860, 580), 580.0),
            (165.0, 1.0, (695, 578, 860, 581), 580.0),
            (160.0, -1.0, (700, 582, 860, 579), 580.0),
            (158.0, 0.5, (702, 579, 860, 581), 580.0),
        ]

        selected = _select_widest_line_cluster(
            candidates,
            max_reference_y_gap=14.0,
        )

        selected_x = [
            x
            for _length, _angle, (x1, _y1, x2, _y2), _reference_y in selected
            for x in (x1, x2)
        ]
        self.assertLessEqual(min(selected_x), 310)
        self.assertGreaterEqual(max(selected_x), 869)

    def test_rejects_frame_without_cardboard(self) -> None:
        frame = np.full((720, 1280, 3), 255, dtype=np.uint8)
        result = detect_box_parallel(frame)
        self.assertFalse(result.ok)
        self.assertIn("cardboard", result.reason)

    def test_detects_parallel_lines_on_synthetic_cardboard(self) -> None:
        frame = np.full((720, 1280, 3), 245, dtype=np.uint8)
        cardboard = (115, 155, 195)
        polygon = np.asarray(
            [[420, 390], [900, 390], [930, 719], [390, 719]],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(frame, polygon, cardboard)
        cv2.line(frame, (405, 555), (915, 555), (35, 45, 55), 5)

        result = detect_box_parallel(frame)

        self.assertTrue(result.ok, result.reason)
        self.assertIsNotNone(result.parallel_error_deg)
        self.assertLess(abs(float(result.parallel_error_deg)), 1.0)

    def test_ignores_disconnected_warm_background_run(self) -> None:
        frame = np.full((720, 1280, 3), 245, dtype=np.uint8)
        cardboard = (115, 155, 195)
        cv2.rectangle(frame, (420, 390), (850, 719), cardboard, -1)
        cv2.line(frame, (420, 555), (850, 555), (35, 45, 55), 5)
        cv2.rectangle(frame, (915, 390), (950, 610), cardboard, -1)

        result = detect_box_parallel(frame)

        self.assertTrue(result.ok, result.reason)
        self.assertIsNotNone(result.box_x_range)
        self.assertLess(result.box_x_range[1], 900)

    def test_rejects_short_partial_seam(self) -> None:
        frame = np.full((720, 1280, 3), 245, dtype=np.uint8)
        cardboard = (115, 155, 195)
        cv2.rectangle(frame, (420, 390), (850, 719), cardboard, -1)
        cv2.line(frame, (420, 555), (700, 555), (35, 45, 55), 5)

        result = detect_box_parallel(frame)

        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "cardboard_seam_span_too_small")

    def test_detects_close_box_with_low_seam(self) -> None:
        frame = np.full((720, 1280, 3), 245, dtype=np.uint8)
        cardboard = (115, 155, 195)
        polygon = np.asarray(
            [[220, 370], [890, 370], [940, 719], [170, 719]],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(frame, polygon, cardboard)
        cv2.line(frame, (190, 600), (920, 600), (35, 45, 55), 5)

        result = detect_box_parallel(frame)

        self.assertTrue(result.ok, result.reason)
        self.assertLess(result.box_x_range[0], 300)
        self.assertLess(abs(float(result.parallel_error_deg)), 1.0)

    def test_detects_very_close_box_with_seam_near_bottom(self) -> None:
        frame = np.full((720, 1280, 3), 245, dtype=np.uint8)
        cardboard = (115, 155, 195)
        polygon = np.asarray(
            [[130, 325], [1100, 325], [1180, 719], [40, 719]],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(frame, polygon, cardboard)
        cv2.line(frame, (65, 675), (1155, 675), (35, 45, 55), 5)

        result = detect_box_parallel(frame)

        self.assertTrue(result.ok, result.reason)
        self.assertIsNotNone(result.seam_line)
        self.assertGreater(result.seam_line[1], 650)
        self.assertLess(abs(float(result.parallel_error_deg)), 1.0)

    def test_detects_sloped_seam_clipped_by_bottom_edge(self) -> None:
        frame = np.full((720, 1280, 3), 245, dtype=np.uint8)
        cardboard = (115, 155, 195)
        polygon = np.asarray(
            [[130, 325], [1100, 325], [1279, 719], [0, 719]],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(frame, polygon, cardboard)
        cv2.line(frame, (0, 665), (720, 719), (35, 45, 55), 5)

        result = detect_box_parallel(frame)

        self.assertTrue(result.ok, result.reason)
        self.assertIsNotNone(result.parallel_error_deg)
        self.assertGreater(float(result.parallel_error_deg), 3.0)


if __name__ == "__main__":
    unittest.main()

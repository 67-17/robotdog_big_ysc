from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from mission_lite3.inspection_runtime.binding import bind_letters_to_meters
from mission_lite3.inspection_runtime import frame_pipeline


class InspectionBindingTests(unittest.TestCase):
    def test_pointer_line_tolerates_perspective_shifted_hub(self) -> None:
        image = Image.new("RGB", (180, 180), "white")
        ImageDraw.Draw(image).line((80, 70, 140, 76), fill="black", width=7)

        result = frame_pipeline.meter_status_recognition.detect_pointer_line(
            np.asarray(image),
            center=(90.0, 90.0),
            radius=60.0,
        )

        self.assertIsNotNone(result)
        self.assertGreater(result["center_offset_ratio"], 0.26)
        self.assertGreaterEqual(result["pointer_support"]["hit_ratio"], 0.90)
        self.assertLess(abs(result["angle"]), 0.20)
        self.assertAlmostEqual(result["sampling_origin"][0], 91.7, delta=3.0)
        self.assertAlmostEqual(result["sampling_origin"][1], 74.2, delta=3.0)

    def test_status_without_geometry_or_ray_color_support_is_unknown(self) -> None:
        recognizer = frame_pipeline.meter_status_recognition
        valid_ring = {
            "red": 100,
            "yellow": 100,
            "green": 100,
            "total": 300,
            "present": 3,
            "ratio": 0.3,
            "max_run": {"red": 20, "yellow": 20, "green": 20},
            "transitions": 3,
        }
        pointer_line = {
            "origin_point": (48.0, 50.0),
            "line_tip_point": (75.0, 50.0),
            "sampling_origin": (50.0, 50.0),
            "angle": 0.0,
            "pointer_support": {
                "hit_ratio": 0.95,
                "longest_run_ratio": 0.90,
            },
        }
        with mock.patch.object(
            recognizer,
            "measure_ring_color_presence",
            return_value=valid_ring,
        ):
            with mock.patch.object(
                recognizer,
                "detect_pointer_line",
                return_value=pointer_line,
            ):
                with mock.patch.object(
                    recognizer,
                    "sample_color_status",
                    return_value=("未知", {"偏高": 0, "偏低": 0, "正常": 0}),
                ):
                    with mock.patch.object(
                        recognizer,
                        "sample_relaxed_pointer_ring_status",
                        return_value=(
                            "未知",
                            {"偏高": 0, "偏低": 0, "正常": 0},
                        ),
                    ):
                        with mock.patch.object(
                            recognizer,
                            "classify_status_from_pointer_geometry",
                            return_value=(
                                "未知",
                                {
                                    "color_angles": {},
                                    "pointer_relative_to_red_deg": None,
                                },
                            ),
                        ):
                            result = recognizer.analyze_meter_rgb_image(
                                Image.new("RGB", (100, 100), "white"),
                                center_hint=(50.0, 50.0),
                                radius_hint=40.0,
                            )

        self.assertEqual(result["status"], "未知")
        self.assertFalse(result["status_evidence"]["status_supported"])

    def test_red_reference_geometry_maps_all_three_dial_states(self) -> None:
        classify = (
            frame_pipeline.meter_status_recognition.classify_status_from_red_reference
        )
        self.assertEqual(classify(np.deg2rad(8.0), 0.0)[0], "偏高")
        self.assertEqual(classify(np.deg2rad(-90.0), 0.0)[0], "正常")
        self.assertEqual(classify(np.deg2rad(135.0), 0.0)[0], "偏低")
        self.assertEqual(classify(np.deg2rad(40.0), 0.0)[0], "未知")

    def test_bind_letters_to_nearest_available_meter(self) -> None:
        letters = [
            {"label": "A", "center_px": [10, 10]},
            {"label": "B", "center_px": [110, 10]},
        ]
        meters = [
            {"meter_id": "m1", "center_px": [12, 10]},
            {"meter_id": "m2", "center_px": [111, 12]},
        ]
        bindings = bind_letters_to_meters(letters, meters, max_distance_px=50)
        self.assertEqual(
            [(item["area"], item["meter_id"], item["binding_status"]) for item in bindings],
            [("A", "m1", "ok"), ("B", "m2", "ok")],
        )

    def test_bind_letters_marks_far_or_reused_meter_ambiguous(self) -> None:
        letters = [
            {"label": "A", "center_px": [0, 0]},
            {"label": "B", "center_px": [4, 0]},
            {"label": "C", "center_px": [200, 0]},
        ]
        meters = [{"meter_id": "m1", "center_px": [1, 0]}]
        bindings = bind_letters_to_meters(letters, meters, max_distance_px=20)
        self.assertEqual(bindings[0]["binding_status"], "ok")
        self.assertEqual(bindings[1]["binding_status"], "ambiguous")
        self.assertEqual(bindings[1]["reason"], "meter_already_bound")
        self.assertEqual(bindings[2]["binding_status"], "ambiguous")
        self.assertEqual(bindings[2]["reason"], "nearest_meter_too_far")

    def test_large_glyph_anchor_recognizes_all_inspection_letters(self) -> None:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            92,
        )
        for expected in "ABCD":
            with self.subTest(letter=expected):
                image = Image.new("RGB", (1280, 720), "white")
                ImageDraw.Draw(image).text(
                    (600, 280),
                    expected,
                    fill="black",
                    font=font,
                )
                anchor = frame_pipeline._locate_letter_anchor(image)  # noqa: SLF001
                self.assertIsNotNone(anchor)
                self.assertEqual(anchor["detection"]["label"], expected)
                self.assertGreaterEqual(anchor["detection"]["confidence"], 0.84)

    def test_clear_d_match_survives_imperfect_hole_structure(self) -> None:
        recognizer = frame_pipeline.letter_recognition
        normalized = mock.Mock()
        templates = {
            letter: (marker,)
            for letter, marker in zip("ABCD", ("A", "B", "C", "D"))
        }
        scores = {"A": 0.44, "B": 0.69, "C": 0.70, "D": 0.815}
        with mock.patch.object(
            recognizer,
            "_normalize_glyph",
            return_value=(normalized, [10.0, 10.0, 50.0, 60.0]),
        ):
            with mock.patch.object(recognizer, "_templates", return_value=templates):
                with mock.patch.object(
                    recognizer,
                    "_similarity",
                    side_effect=lambda _normalized, marker: scores[marker],
                ):
                    with mock.patch.object(
                        recognizer,
                        "_shape_is_plausible",
                        return_value=False,
                    ):
                        result = recognizer.recognize_letter_roi(
                            Image.new("RGB", (80, 80), "white")
                        )

        self.assertEqual(result["label"], "D")
        self.assertEqual(result["confidence"], 0.815)
        self.assertEqual(result["margin"], 0.115)

    def test_letter_anchor_limits_meter_search_below_glyph(self) -> None:
        anchor = {
            "detection": {
                "center_px": [300.0, 200.0],
                "component_bbox_xyxy": [280.0, 175.0, 320.0, 225.0],
            },
            "recognition_bbox_xyxy": [250.0, 160.0, 350.0, 240.0],
            "glyph_height": 50.0,
        }
        rois = frame_pipeline._letter_anchored_rois((1280, 720), anchor)  # noqa: SLF001
        self.assertEqual(
            rois["meter_roi_xyxy"],
            [140.0, 260.0, 460.0, 550.0],
        )
        self.assertEqual(rois["meter_expected_center_px"], [300.0, 395.0])

    def test_meter_circle_constraints_remove_wrong_background_circle(self) -> None:
        recognizer = frame_pipeline.meter_status_recognition
        circles = [((25.0, 30.0), 12.0), ((100.0, 110.0), 45.0)]

        def accept_candidates(_image, candidates, classifier):
            del classifier
            return [
                {
                    "center": center,
                    "radius": radius,
                    "ring_stats": {},
                    "score": radius,
                }
                for center, radius in candidates
            ]

        with mock.patch.object(
            recognizer,
            "_hough_circle_candidates",
            return_value=circles,
        ):
            with mock.patch.object(
                recognizer,
                "_fit_circle_to_color_ring",
                return_value=None,
            ):
                with mock.patch.object(
                    recognizer,
                    "collect_valid_meter_candidates",
                    side_effect=accept_candidates,
                ):
                    located = recognizer.locate_meter_circle(
                        Image.new("RGB", (200, 200), "white"),
                        expected_center=(105.0, 105.0),
                        min_radius=30.0,
                        max_radius=60.0,
                        max_center_distance=25.0,
                    )
        self.assertEqual(located["center"], (100.0, 110.0))
        self.assertEqual(located["radius"], 45.0)

    def test_full_frame_pipeline_uses_letter_anchor_and_matching_meter_roi(self) -> None:
        image = Image.new("RGB", (1280, 720), "white")
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            92,
        )
        ImageDraw.Draw(image).text((600, 280), "C", fill="black", font=font)
        meter_analyzer = mock.Mock(
            return_value={
                "meter_found": True,
                "pointer_found": True,
                "status": "正常",
                "center": (100.0, 120.0),
                "radius": 60.0,
                "tip_point": (100.0, 70.0),
                "pointer_support": {
                    "hit_ratio": 0.7,
                    "longest_run_ratio": 0.3,
                },
            }
        )

        result = frame_pipeline.analyze_inspection_frame(
            image,
            meter_analyzer=meter_analyzer,
        )

        self.assertTrue(result["ok"])
        self.assertEqual(result["geometry_source"], "letter_anchor")
        self.assertEqual(result["letter_detection"]["label"], "C")
        self.assertEqual(result["meter_detection"]["state"], "normal")
        self.assertEqual(result["areas"][0]["area"], "C")
        meter_roi = meter_analyzer.call_args.args[0]
        self.assertGreater(meter_roi.width, 250)
        self.assertGreater(meter_roi.height, 250)
        meter_y0 = result["rois"]["meter_roi_xyxy"][1]
        letter_y1 = result["rois"]["letter_component_bbox_xyxy"][3]
        self.assertGreater(meter_y0, letter_y1)

    def test_meter_refinement_compares_letter_expected_center(self) -> None:
        meter_roi = Image.new("RGB", (288, 261), "white")
        expected_center = (144.0, 121.5)
        rois = {
            "meter_roi_xyxy": [408.0, 447.0, 696.0, 708.0],
            "meter_expected_center_px": [552.0, 568.5],
            "letter_glyph_height_px": 45.0,
        }

        def meter_result(status, center, hit_ratio, run_ratio):
            return {
                "meter_found": True,
                "pointer_found": True,
                "status": status,
                "center": center,
                "radius": 60.0,
                "pointer_support": {
                    "hit_ratio": hit_ratio,
                    "longest_run_ratio": run_ratio,
                },
            }

        def analyze(_image, *, center_hint, radius_hint):
            del radius_hint
            if tuple(center_hint) == expected_center:
                return meter_result("偏低", center_hint, 0.98, 0.85)
            if tuple(center_hint) == (141.0, 141.0):
                return meter_result("偏低", center_hint, 0.30, 0.20)
            return meter_result("正常", center_hint, 0.57, 0.32)

        recognizer = frame_pipeline.meter_status_recognition
        with mock.patch.object(
            recognizer,
            "locate_meter_circle",
            return_value={"center": (141.0, 141.0), "radius": 60.0},
        ):
            with mock.patch.object(
                recognizer,
                "analyze_meter_rgb_image",
                side_effect=analyze,
            ) as analyzer:
                result = frame_pipeline._analyze_letter_anchored_meter(  # noqa: SLF001
                    meter_roi,
                    rois,
                )

        self.assertEqual(result["status"], "偏低")
        self.assertEqual(tuple(result["center"]), expected_center)
        self.assertTrue(
            any(
                tuple(call.kwargs["center_hint"]) == expected_center
                for call in analyzer.call_args_list
            )
        )

    def test_meter_uses_letter_layout_when_circle_detector_misses(self) -> None:
        meter_roi = Image.new("RGB", (288, 261), "white")
        rois = {
            "meter_roi_xyxy": [408.0, 447.0, 696.0, 708.0],
            "meter_expected_center_px": [552.0, 568.5],
            "letter_glyph_height_px": 45.0,
        }
        expected = {
            "meter_found": True,
            "pointer_found": True,
            "status": "正常",
            "center": (144.0, 121.5),
            "radius": 63.0,
            "pointer_support": {
                "hit_ratio": 0.95,
                "longest_run_ratio": 0.90,
            },
        }
        recognizer = frame_pipeline.meter_status_recognition
        with mock.patch.object(recognizer, "locate_meter_circle", return_value=None):
            with mock.patch.object(
                recognizer,
                "analyze_meter_rgb_image",
                return_value=expected,
            ) as analyzer:
                result = frame_pipeline._analyze_letter_anchored_meter(  # noqa: SLF001
                    meter_roi,
                    rois,
                )

        self.assertEqual(result["status"], "正常")
        self.assertEqual(result["circle_source"], "letter_layout_fallback")
        self.assertEqual(analyzer.call_args.kwargs["center_hint"], (144.0, 121.5))
        self.assertAlmostEqual(analyzer.call_args.kwargs["radius_hint"], 63.0)


if __name__ == "__main__":
    unittest.main()

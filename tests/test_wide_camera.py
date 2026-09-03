from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mission_lite3.wide_camera import WideCameraUndistorter, load_wide_calibration


class WideCameraTests(unittest.TestCase):
    def test_project_calibration_loads_and_matches_runtime_size(self) -> None:
        calibration = load_wide_calibration(
            "mission_lite3/config/wide_angle_camera_calibration.json"
        )
        undistorter = WideCameraUndistorter(calibration)
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        output = undistorter.apply(frame)

        self.assertEqual(output.shape, frame.shape)
        self.assertEqual(undistorter.image_size, (1280, 720))

    def test_unvalidated_calibration_is_rejected(self) -> None:
        source = json.loads(
            Path("mission_lite3/config/wide_angle_camera_calibration.json").read_text(
                encoding="utf-8"
            )
        )
        source["validated_for_undistortion"] = False
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not validated"):
                load_wide_calibration(path)

    def test_frame_size_mismatch_is_rejected(self) -> None:
        undistorter = WideCameraUndistorter.from_file(
            "mission_lite3/config/wide_angle_camera_calibration.json"
        )
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            undistorter.apply(np.zeros((480, 640, 3), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()

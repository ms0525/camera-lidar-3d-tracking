# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from track_3d_visualization import Sequence3DProcessor, build_parser, run
from utils.kitti_tracking_labels import KittiTrackingLabel


SYNTHETIC_CALIBRATION = """\
P2: 100 0 50 -20 0 100 40 0 0 0 1 0
R0_rect: 1 0 0 0 1 0 0 0 1
Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0
"""


class FakeModel:
    names = {0: "car"}

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, **kwargs):
        self.calls += 1
        self.last_predict_options = kwargs
        return [SimpleNamespace(boxes=None)]


class FakeTracker:
    def __init__(self) -> None:
        self.calls = 0

    def update_tracks(self, _detections, frame):
        self.calls += 1
        self.last_shape = frame.shape
        return []


class Tracking3DStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        root = Path(self._temporary_directory.name)
        self.image_dir = root / "images"
        self.calib_dir = root / "calib"
        self.image_dir.mkdir()
        self.calib_dir.mkdir()
        self.image_path = self.image_dir / "000000.png"
        self.calibration_file = self.calib_dir / "000000.txt"
        self.missing_pointcloud_dir = root / "missing-pointcloud"
        self.assertTrue(
            cv2.imwrite(
                str(self.image_path),
                np.zeros((48, 64, 3), dtype=np.uint8),
            )
        )
        self.calibration_file.write_text(SYNTHETIC_CALIBRATION, encoding="ascii")

    def test_manual_mode_requires_explicit_sensor_inputs(self) -> None:
        args = build_parser().parse_args(["--headless"])
        with self.assertRaisesRegex(ValueError, r"manual mode requires explicit") as caught:
            run(args)
        message = str(caught.exception)
        self.assertIn("--image-dir", message)
        self.assertIn("--calib-dir or --calib-file", message)
        self.assertIn("--pointcloud-dir", message)

    def test_yolo_prediction_cli_options_are_tristate(self) -> None:
        defaults = build_parser().parse_args([])
        enabled = build_parser().parse_args(["--imgsz", "960", "--yolo-end2end"])
        disabled = build_parser().parse_args(["--no-yolo-end2end"])

        self.assertEqual(defaults.imgsz, 640)
        self.assertIsNone(defaults.yolo_end2end)
        self.assertEqual(enabled.imgsz, 960)
        self.assertIs(enabled.yolo_end2end, True)
        self.assertIs(disabled.yolo_end2end, False)

    def test_missing_lidar_still_advances_2d_tracker(self) -> None:
        model = FakeModel()
        tracker = FakeTracker()
        args = SimpleNamespace(
            calib_file=self.calibration_file,
            confidence=0.25,
            device=None,
            imgsz=960,
            yolo_end2end=False,
            min_lidar_points=3,
            fallback_range_m=0.0,
        )
        processor = Sequence3DProcessor(
            [self.image_path],
            self.calib_dir,
            self.missing_pointcloud_dir,
            model,
            tracker,
            args,
            class_confidence_thresholds={"car": 0.15},
        )

        with (
            patch(
                "track_3d_visualization.load_velodyne_points",
                side_effect=FileNotFoundError("missing scan"),
            ),
            patch(
                "track_3d_visualization.camera_ray_in_velodyne",
                return_value=(np.zeros(3), np.asarray([1.0, 0.0, 0.0])),
            ),
        ):
            result = processor.ensure_processed(0)

        self.assertIsNone(result.error)
        self.assertIsNone(result.pointcloud_path)
        self.assertEqual(model.calls, 1)
        self.assertEqual(tracker.calls, 1)
        self.assertEqual(model.last_predict_options["conf"], 0.15)
        self.assertEqual(model.last_predict_options["imgsz"], 960)
        self.assertIs(model.last_predict_options["end2end"], False)

    def test_ground_truth_box_survives_missing_lidar(self) -> None:
        label = KittiTrackingLabel(
            0, 4, "Car", 0, 0, 0.0,
            10.0, 20.0, 100.0, 120.0,
            1.5, 1.6, 4.0, 1.0, 1.5, 15.0, 0.2,
        )
        model = FakeModel()
        tracker = FakeTracker()
        args = SimpleNamespace(
            calib_file=self.calibration_file,
            confidence=0.25,
            device=None,
            min_lidar_points=3,
            fallback_range_m=0.0,
        )
        processor = Sequence3DProcessor(
            [self.image_path],
            self.calib_dir,
            self.missing_pointcloud_dir,
            model,
            tracker,
            args,
            ground_truth_frames=((label,),),
        )

        with patch(
            "track_3d_visualization.load_velodyne_points",
            side_effect=FileNotFoundError("missing scan"),
        ):
            result = processor.ensure_processed(0)

        self.assertIsNone(result.error)
        self.assertEqual(len(result.ground_truth), 1)
        self.assertEqual(result.ground_truth[0].track_id, 4)
        self.assertEqual(result.ground_truth[0].extent_lwh, (4.0, 1.6, 1.5))
        self.assertTrue(np.isfinite(result.ground_truth[0].center_velodyne).all())


if __name__ == "__main__":
    unittest.main()

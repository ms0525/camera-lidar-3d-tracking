# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from utils.bin_pointcloud_loader import load_velodyne_points
from utils.calib_loader import load_calibration


SYNTHETIC_CALIBRATION = """\
P2: 100 0 50 -20 0 100 40 0 0 0 1 0
R0_rect: 1 0 0 0 1 0 0 0 1
Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0
"""


class SampleDataIntegrityTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.data_root = Path(self._temporary_directory.name) / "data"
        for modality in ("images", "calib", "labels", "pointcloud"):
            (self.data_root / modality).mkdir(parents=True)

        pointcloud = np.asarray(
            [
                [0.20, 0.00, 8.2, 0.10],
                [0.21, 0.01, 8.4, 0.20],
                [0.19, -0.01, 8.6, 0.30],
                [3.00, 0.00, 20.0, 0.05],
            ],
            dtype=np.float32,
        )
        image = np.zeros((8, 12, 3), dtype=np.uint8)
        for index in range(7):
            frame_id = f"{index:06d}"
            self.assertTrue(
                cv2.imwrite(str(self.data_root / "images" / f"{frame_id}.png"), image)
            )
            (self.data_root / "calib" / f"{frame_id}.txt").write_text(
                SYNTHETIC_CALIBRATION,
                encoding="ascii",
            )
            (self.data_root / "labels" / f"{frame_id}.txt").write_text(
                "Car 0 0 0 1 1 5 5 1.5 1.6 4.0 0 0 10 0\n",
                encoding="ascii",
            )
            pointcloud.tofile(self.data_root / "pointcloud" / f"{frame_id}.bin")

    def test_all_sample_modalities_are_paired(self) -> None:
        expected_ids = {f"{index:06d}" for index in range(7)}
        modality_patterns = {
            "images": "*.png",
            "calib": "*.txt",
            "labels": "*.txt",
            "pointcloud": "*.bin",
        }
        for directory, pattern in modality_patterns.items():
            with self.subTest(directory=directory):
                ids = {
                    path.stem
                    for path in (self.data_root / directory).glob(pattern)
                }
                self.assertEqual(ids, expected_ids)

    def test_all_pointclouds_and_calibrations_load(self) -> None:
        for index in range(7):
            frame_id = f"{index:06d}"
            with self.subTest(frame_id=frame_id):
                calibration = load_calibration(
                    self.data_root / "calib" / f"{frame_id}.txt"
                )
                points = load_velodyne_points(
                    self.data_root / "pointcloud" / f"{frame_id}.bin"
                )
                self.assertIn("P2", calibration)
                self.assertEqual(points.shape, (4, 4))
                np.testing.assert_allclose(points[1], [0.21, 0.01, 8.4, 0.20])


if __name__ == "__main__":
    unittest.main()

# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from utils.bin_pointcloud_loader import load_velodyne_points
from utils.calib_loader import load_calibration
from utils.geometry import (
    camera_ray_in_velodyne,
    class_dimensions_lwh,
    estimate_lidar_center_for_box,
    load_kitti_transforms,
    kitti_camera_box_to_velodyne,
    project_velodyne_to_image,
)


SYNTHETIC_CALIBRATION = """\
P2: 100 0 50 -20 0 100 40 0 0 0 1 0
R0_rect: 1 0 0 0 1 0 0 0 1
Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0
"""


class KittiGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary_directory = tempfile.TemporaryDirectory()
        root = Path(cls._temporary_directory.name)
        calibration_path = root / "calibration.txt"
        calibration_path.write_text(SYNTHETIC_CALIBRATION, encoding="ascii")
        pointcloud_path = root / "pointcloud.bin"
        np.asarray(
            [
                [0.20, 0.00, 8.2, 0.10],
                [0.21, 0.01, 8.4, 0.20],
                [0.19, -0.01, 8.6, 0.30],
                [3.00, 0.00, 20.0, 0.05],
            ],
            dtype=np.float32,
        ).tofile(pointcloud_path)

        cls.calibration = load_calibration(calibration_path)
        cls.transforms = load_kitti_transforms(cls.calibration)
        cls.points = load_velodyne_points(pointcloud_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary_directory.cleanup()

    def test_calibration_shapes(self) -> None:
        self.assertEqual(self.transforms.P2.shape, (3, 4))
        self.assertEqual(self.transforms.R0_rect.shape, (3, 3))
        self.assertEqual(self.transforms.Tr_velo_to_cam.shape, (3, 4))
        self.assertEqual(self.transforms.T_velo_to_rect.shape, (4, 4))

    def test_camera_center_is_not_lidar_origin(self) -> None:
        self.assertGreater(np.linalg.norm(self.transforms.camera_center_velodyne), 0.1)

    def test_camera_ray_reprojects_to_original_pixel(self) -> None:
        expected_pixel = np.array([50.0, 40.0])
        origin, direction = camera_ray_in_velodyne(expected_pixel, self.transforms)
        point_on_ray = origin + direction * 20.0
        uv, _, valid = project_velodyne_to_image(point_on_ray[None, :], self.transforms)
        self.assertTrue(valid[0])
        np.testing.assert_allclose(uv[0], expected_pixel, atol=1e-7)

    def test_projection_matches_explicit_kitti_chain(self) -> None:
        sample = self.points[0, :3]
        uv, _, valid = project_velodyne_to_image(sample[None, :], self.transforms)

        point_h = np.append(sample, 1.0)
        projected = self.transforms.P2 @ self.transforms.T_velo_to_rect @ point_h
        manual_uv = projected[:2] / projected[2]
        self.assertEqual(bool(valid[0]), bool(projected[2] > 0))
        if valid[0]:
            np.testing.assert_allclose(uv[0], manual_uv, atol=1e-10)

    def test_lidar_association_recovers_near_synthetic_depth_cluster(self) -> None:
        estimate = estimate_lidar_center_for_box(
            self.points,
            (45.0, 35.0, 55.0, 50.0),
            self.transforms,
            min_points=3,
        )
        self.assertIsNotNone(estimate)
        assert estimate is not None
        self.assertGreaterEqual(estimate.point_count, 3)
        self.assertAlmostEqual(estimate.depth_m, 8.4, delta=0.1)

    def test_car_dimensions_use_lidar_axis_order(self) -> None:
        np.testing.assert_allclose(class_dimensions_lwh("car"), [4.0, 1.8, 1.6])

    def test_kitti_label_box_converts_to_open3d_velodyne_convention(self) -> None:
        velo_to_camera = np.array(
            [
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        )
        transforms = load_kitti_transforms(
            {
                "P2": np.column_stack((np.eye(3), np.zeros(3))).reshape(-1),
                "R0_rect": np.eye(3).reshape(-1),
                "Tr_velo_to_cam": velo_to_camera.reshape(-1),
            }
        )

        box = kitti_camera_box_to_velodyne(
            location_xyz=(1.0, 2.0, 10.0),
            dimensions_hwl=(2.0, 4.0, 6.0),
            rotation_y=-np.pi / 2.0,
            transforms=transforms,
        )

        np.testing.assert_allclose(box.center_velodyne, [10.0, -1.0, -1.0])
        np.testing.assert_allclose(box.extent_lwh, [6.0, 4.0, 2.0])
        np.testing.assert_allclose(box.rotation_velodyne, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(
            box.rotation_velodyne.T @ box.rotation_velodyne,
            np.eye(3),
            atol=1e-12,
        )
        self.assertAlmostEqual(float(np.linalg.det(box.rotation_velodyne)), 1.0)


if __name__ == "__main__":
    unittest.main()

# SPDX-License-Identifier: AGPL-3.0-only
"""Visualize YOLO detections localized with calibrated KITTI LiDAR points."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
from ultralytics import YOLO

from utils.bin_pointcloud_loader import load_velodyne_points, to_open3d_point_cloud
from utils.calib_loader import load_calibration
from utils.geometry import (
    camera_ray_in_velodyne,
    class_dimensions_lwh,
    estimate_lidar_center_for_box,
    load_kitti_transforms,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show a KITTI point cloud with LiDAR-supported YOLO object centers."
    )
    parser.add_argument("--image", required=True, help="Input camera image")
    parser.add_argument("--calib", required=True, help="Matching KITTI calibration file")
    parser.add_argument(
        "--pointcloud",
        required=True,
        help="Matching Velodyne float32 XYZI point cloud",
    )
    parser.add_argument("--model", default="yolo26s.pt")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--min-lidar-points", type=int, default=3)
    parser.add_argument("--ray-length", type=float, default=20.0)
    parser.add_argument("--max-range", type=float, default=80.0)
    parser.add_argument("--min-z", type=float, default=-5.0)
    parser.add_argument("--point-color", choices=("height", "intensity", "none"), default="height")
    return parser.parse_args()


def create_box(center: np.ndarray, class_name: str, color: list[float]):
    """Create an axis-aligned box using LiDAR [length, width, height] axes."""

    box = o3d.geometry.OrientedBoundingBox(
        center=np.asarray(center, dtype=np.float64),
        R=np.eye(3),
        extent=class_dimensions_lwh(class_name),
    )
    box.color = color
    return box


def build_scene(args: argparse.Namespace) -> list:
    image_path = Path(args.image)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    points = load_velodyne_points(args.pointcloud)
    calibration = load_calibration(args.calib)
    transforms = load_kitti_transforms(calibration)

    display_mask = np.linalg.norm(points[:, :2], axis=1) <= args.max_range
    display_mask &= points[:, 2] >= args.min_z
    display_points = points[display_mask]
    color_by = None if args.point_color == "none" else args.point_color
    point_cloud = to_open3d_point_cloud(display_points, color_by=color_by)

    model = YOLO(args.model)
    result = model.predict(image, conf=args.confidence, verbose=False)[0]

    ray_points = [transforms.camera_center_velodyne]
    ray_lines: list[list[int]] = []
    ray_colors: list[list[float]] = []
    boxes = []

    for detection in result.boxes:
        bbox = tuple(map(float, detection.xyxy[0].tolist()))
        class_name = model.names[int(detection.cls[0].item())]
        confidence = float(detection.conf[0].item())
        x1, y1, x2, y2 = bbox
        pixel_center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        ray_origin, ray_direction = camera_ray_in_velodyne(pixel_center, transforms)
        estimate = estimate_lidar_center_for_box(
            points,
            bbox,
            transforms,
            min_points=args.min_lidar_points,
        )

        if estimate is None:
            endpoint = ray_origin + ray_direction * args.ray_length
            status = "ray only; insufficient LiDAR returns"
            ray_color = [1.0, 0.35, 0.0]
        else:
            endpoint = estimate.center_velodyne
            boxes.append(create_box(endpoint, class_name, [0.1, 0.9, 0.2]))
            status = (
                f"depth={estimate.depth_m:.1f}m, "
                f"points={estimate.point_count}"
            )
            ray_color = [1.0, 0.0, 0.0]

        ray_points.append(endpoint)
        ray_lines.append([0, len(ray_points) - 1])
        ray_colors.append(ray_color)
        print(f"{class_name} {confidence:.2f}: {status}")

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.asarray(ray_points)),
        lines=o3d.utility.Vector2iVector(np.asarray(ray_lines, dtype=np.int32).reshape(-1, 2)),
    )
    if ray_colors:
        line_set.colors = o3d.utility.Vector3dVector(np.asarray(ray_colors))

    camera_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.12)
    camera_sphere.paint_uniform_color([0.2, 0.2, 1.0])
    camera_sphere.translate(transforms.camera_center_velodyne)

    print(
        f"Loaded {len(points):,} LiDAR points; displaying {len(display_points):,}; "
        f"localized {len(boxes)}/{len(result.boxes)} detections"
    )
    return [point_cloud, line_set, camera_sphere, *boxes]


def main() -> int:
    args = parse_args()
    geometries = build_scene(args)
    o3d.visualization.draw_geometries(
        geometries,
        window_name="KITTI LiDAR-supported detections",
        width=1280,
        height=720,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-License-Identifier: AGPL-3.0-only
"""Print calibrated camera rays and LiDAR-supported centers for 2D detections."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from utils.bin_pointcloud_loader import load_velodyne_points
from utils.calib_loader import load_calibration
from utils.geometry import (
    camera_ray_in_velodyne,
    estimate_lidar_center_for_box,
    load_kitti_transforms,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Back-project YOLO detections and estimate their centers from KITTI LiDAR."
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = Path(args.image)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    calibration = load_calibration(args.calib)
    transforms = load_kitti_transforms(calibration)
    points = load_velodyne_points(args.pointcloud)

    model = YOLO(args.model)
    result = model.predict(image, conf=args.confidence, verbose=False)[0]
    names = model.names

    if len(result.boxes) == 0:
        print("No detections.")
        return 0

    for index, box in enumerate(result.boxes, start=1):
        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
        confidence = float(box.conf[0].item())
        class_name = names[int(box.cls[0].item())]
        center_pixel = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        ray_origin, ray_direction = camera_ray_in_velodyne(center_pixel, transforms)
        estimate = estimate_lidar_center_for_box(
            points,
            (x1, y1, x2, y2),
            transforms,
            min_points=args.min_lidar_points,
        )

        print(
            f"#{index} {class_name} confidence={confidence:.3f} "
            f"bbox=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f})"
        )
        print(
            "  ray_velodyne: "
            f"origin={ray_origin.round(3)} direction={ray_direction.round(4)}"
        )
        if estimate is None:
            print("  lidar_center: unavailable (too few associated points)")
        else:
            print(
                "  lidar_center: "
                f"xyz={estimate.center_velodyne.round(3)} "
                f"camera_depth={estimate.depth_m:.2f}m "
                f"range={estimate.range_m:.2f}m points={estimate.point_count}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

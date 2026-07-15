# SPDX-License-Identifier: AGPL-3.0-only
"""Load and convert KITTI Velodyne point clouds."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_velodyne_points(bin_path: str | Path) -> np.ndarray:
    """Return an ``N x 4`` float32 array of ``x, y, z, reflectance`` values."""

    path = Path(bin_path)
    if not path.is_file():
        raise FileNotFoundError(f"Velodyne point cloud does not exist: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"Velodyne point cloud is empty: {path}")
    if path.stat().st_size % (4 * np.dtype(np.float32).itemsize) != 0:
        raise ValueError(
            f"{path} has {path.stat().st_size} bytes; KITTI point clouds require "
            "a multiple of 16 bytes"
        )

    points = np.fromfile(path, dtype="<f4").reshape(-1, 4)
    if not np.isfinite(points).all():
        raise ValueError(f"{path} contains NaN or infinity")
    return points


def to_open3d_point_cloud(points: np.ndarray, color_by: str | None = "height"):
    """Convert an array to Open3D, optionally coloring by height or intensity."""

    import open3d as o3d

    values = np.asarray(points)
    if values.ndim != 2 or values.shape[1] < 3:
        raise ValueError(f"Expected an N x 3 or N x 4 point array, got {values.shape}")
    if not np.isfinite(values[:, :3]).all():
        raise ValueError("Point array contains NaN or infinity")

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(values[:, :3].astype(np.float64))

    if color_by is not None and len(values):
        if color_by == "height":
            scalar = values[:, 2].astype(np.float64)
        elif color_by == "intensity":
            if values.shape[1] < 4:
                raise ValueError("Intensity coloring requires an N x 4 point array")
            scalar = values[:, 3].astype(np.float64)
        else:
            raise ValueError("color_by must be 'height', 'intensity', or None")

        low, high = np.percentile(scalar, [2.0, 98.0])
        normalized = np.zeros_like(scalar) if high <= low else np.clip((scalar - low) / (high - low), 0, 1)
        # A small, dependency-free blue -> cyan -> yellow color ramp.
        colors = np.column_stack(
            (
                normalized,
                1.0 - np.abs(2.0 * normalized - 1.0),
                1.0 - normalized,
            )
        )
        point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud


def load_velodyne_bin(bin_path: str | Path, color_by: str | None = "height"):
    """Compatibility wrapper returning an Open3D point cloud."""

    return to_open3d_point_cloud(load_velodyne_points(bin_path), color_by=color_by)

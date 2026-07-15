# SPDX-License-Identifier: AGPL-3.0-only
"""KITTI camera/LiDAR geometry and lightweight point association helpers.

The KITTI object files store Velodyne points in the LiDAR frame and labels in
the rectified camera frame.  A LiDAR point is projected into camera 2 with::

    pixel ~ P2 @ R0_rect @ Tr_velo_to_cam @ point_velodyne

Keeping that complete transform in one module prevents the single-frame and
tracking visualizers from silently using different coordinate conventions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class KittiTransforms:
    """Validated matrices needed for KITTI camera-2/LiDAR conversion."""

    P2: np.ndarray
    R0_rect: np.ndarray
    Tr_velo_to_cam: np.ndarray
    T_velo_to_rect: np.ndarray
    T_rect_to_velo: np.ndarray
    P_velo_to_image: np.ndarray
    camera_center_rect: np.ndarray
    camera_center_velodyne: np.ndarray


@dataclass(frozen=True)
class LidarBoxEstimate:
    """Robust center estimate obtained from LiDAR returns in a 2D box."""

    center_velodyne: np.ndarray
    center_rect: np.ndarray
    depth_m: float
    range_m: float
    point_count: int


@dataclass(frozen=True)
class KittiBoxInVelodyne:
    """An exact KITTI label box expressed for Open3D/Velodyne rendering."""

    center_velodyne: np.ndarray
    rotation_velodyne: np.ndarray
    extent_lwh: np.ndarray


# Approximate physical dimensions in the Velodyne/Open3D axis order
# [length (x), width (y), height (z)].  They are visualization priors, not
# inferred 3D dimensions.
CLASS_DIMENSIONS_LWH: dict[str, np.ndarray] = {
    "car": np.array([4.0, 1.8, 1.6]),
    "truck": np.array([8.0, 2.5, 3.0]),
    "bus": np.array([10.0, 2.6, 3.2]),
    "motorcycle": np.array([2.2, 0.8, 1.5]),
    "bicycle": np.array([1.8, 0.6, 1.5]),
    "person": np.array([0.8, 0.7, 1.75]),
    "skateboard": np.array([0.8, 0.3, 0.15]),
    "traffic light": np.array([0.4, 0.4, 1.0]),
    "potted plant": np.array([0.8, 0.8, 1.0]),
}


def _reshape(calib: Mapping[str, np.ndarray], key: str, shape: tuple[int, ...]) -> np.ndarray:
    if key not in calib:
        raise KeyError(f"Calibration is missing required key {key!r}")
    value = np.asarray(calib[key], dtype=np.float64)
    if value.size != int(np.prod(shape)):
        raise ValueError(
            f"Calibration key {key!r} has {value.size} values; expected {int(np.prod(shape))}"
        )
    matrix = value.reshape(shape)
    if not np.isfinite(matrix).all():
        raise ValueError(f"Calibration key {key!r} contains NaN or infinity")
    return matrix


def _homogeneous_transform(rotation: np.ndarray, translation: np.ndarray | None = None) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    if translation is not None:
        transform[:3, 3] = translation
    return transform


def load_kitti_transforms(calib: Mapping[str, np.ndarray]) -> KittiTransforms:
    """Build the complete Velodyne-to-camera-2 projection chain."""

    P2 = _reshape(calib, "P2", (3, 4))
    R0_rect = _reshape(calib, "R0_rect", (3, 3))
    Tr_velo_to_cam = _reshape(calib, "Tr_velo_to_cam", (3, 4))

    if abs(np.linalg.det(R0_rect)) < 1e-9:
        raise ValueError("R0_rect is singular")

    T_velo_to_cam = _homogeneous_transform(
        Tr_velo_to_cam[:, :3], Tr_velo_to_cam[:, 3]
    )
    T_cam_to_rect = _homogeneous_transform(R0_rect)
    T_velo_to_rect = T_cam_to_rect @ T_velo_to_cam
    T_rect_to_velo = np.linalg.inv(T_velo_to_rect)

    # For P = M [I | -C], C is the optical center in the rectified reference
    # camera frame.  KITTI P2 has a small, non-zero camera baseline term.
    projection_linear = P2[:, :3]
    if abs(np.linalg.det(projection_linear)) < 1e-9:
        raise ValueError("P2 has a singular 3x3 projection block")
    camera_center_rect = -np.linalg.solve(projection_linear, P2[:, 3])
    camera_center_velodyne_h = T_rect_to_velo @ np.append(camera_center_rect, 1.0)

    return KittiTransforms(
        P2=P2,
        R0_rect=R0_rect,
        Tr_velo_to_cam=Tr_velo_to_cam,
        T_velo_to_rect=T_velo_to_rect,
        T_rect_to_velo=T_rect_to_velo,
        P_velo_to_image=P2 @ T_velo_to_rect,
        camera_center_rect=camera_center_rect,
        camera_center_velodyne=camera_center_velodyne_h[:3],
    )


def transform_points(points_xyz: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply a 4x4 rigid transform to an ``N x 3`` point array."""

    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected an N x 3 (or N x 4) point array, got {points.shape}")
    xyz = points[:, :3]
    homogeneous = np.column_stack((xyz, np.ones(len(xyz), dtype=np.float64)))
    return (np.asarray(transform, dtype=np.float64) @ homogeneous.T).T[:, :3]


def project_velodyne_to_image(
    points_xyz: np.ndarray, transforms: KittiTransforms
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project points and return ``(uv, camera_depth, valid_mask)``.

    ``uv`` retains one row per input point. Invalid rows contain ``NaN`` so
    callers can preserve the correspondence with the original point array.
    """

    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"Expected an N x 3 (or N x 4) point array, got {points.shape}")
    xyz = points[:, :3]
    homogeneous = np.column_stack((xyz, np.ones(len(xyz), dtype=np.float64)))
    rectified = (transforms.T_velo_to_rect @ homogeneous.T).T
    projected = (transforms.P2 @ rectified.T).T

    denominator = projected[:, 2]
    depth = rectified[:, 2]
    valid = (
        np.isfinite(xyz).all(axis=1)
        & np.isfinite(projected).all(axis=1)
        & (depth > 0.0)
        & (denominator > 1e-9)
    )
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    uv[valid] = projected[valid, :2] / denominator[valid, None]
    return uv, depth, valid


def camera_ray_in_velodyne(
    pixel: Sequence[float], transforms: KittiTransforms
) -> tuple[np.ndarray, np.ndarray]:
    """Return the camera-2 optical center and a unit ray in Velodyne space."""

    if len(pixel) != 2:
        raise ValueError("pixel must contain exactly (u, v)")
    image_h = np.array([float(pixel[0]), float(pixel[1]), 1.0], dtype=np.float64)
    direction_rect = np.linalg.solve(transforms.P2[:, :3], image_h)
    norm = np.linalg.norm(direction_rect)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError(f"Cannot construct a ray for pixel {tuple(pixel)!r}")
    direction_rect /= norm

    direction_velodyne = transforms.T_rect_to_velo[:3, :3] @ direction_rect
    direction_velodyne /= np.linalg.norm(direction_velodyne)
    return transforms.camera_center_velodyne.copy(), direction_velodyne


def kitti_camera_box_to_velodyne(
    location_xyz: Sequence[float],
    dimensions_hwl: Sequence[float],
    rotation_y: float,
    transforms: KittiTransforms,
) -> KittiBoxInVelodyne:
    """Convert one KITTI label box from rectified camera to Velodyne space.

    KITTI stores ``location_xyz`` at the bottom-face center, dimensions in
    ``(height, width, length)`` order, and yaw about the camera's downward
    y-axis. Open3D expects the geometric center, a right-handed orientation
    matrix, and extents along that matrix's local axes.
    """

    location = np.asarray(location_xyz, dtype=np.float64)
    dimensions = np.asarray(dimensions_hwl, dtype=np.float64)
    if location.shape != (3,) or not np.isfinite(location).all():
        raise ValueError("location_xyz must contain three finite values")
    if dimensions.shape != (3,) or not np.isfinite(dimensions).all():
        raise ValueError("dimensions_hwl must contain three finite values")
    if np.any(dimensions <= 0.0):
        raise ValueError("KITTI box dimensions must be positive")
    if not math.isfinite(float(rotation_y)):
        raise ValueError("rotation_y must be finite")

    height, width, length = dimensions
    center_rect = location.copy()
    center_rect[1] -= height / 2.0
    center_velodyne = transform_points(
        center_rect.reshape(1, 3), transforms.T_rect_to_velo
    )[0]

    cosine = math.cos(float(rotation_y))
    sine = math.sin(float(rotation_y))
    # Columns are the local length, width, and upward axes in the rectified
    # camera frame. Their cross products form a proper right-handed basis.
    box_to_rect = np.array(
        [
            [cosine, sine, 0.0],
            [0.0, 0.0, -1.0],
            [-sine, cosine, 0.0],
        ],
        dtype=np.float64,
    )
    rotation = transforms.T_rect_to_velo[:3, :3] @ box_to_rect
    # Calibration text is rounded, so remove its tiny orthogonality drift
    # before handing the matrix to Open3D.
    u, _, vt = np.linalg.svd(rotation)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt

    return KittiBoxInVelodyne(
        center_velodyne=center_velodyne,
        rotation_velodyne=rotation,
        extent_lwh=np.array([length, width, height], dtype=np.float64),
    )


def estimate_lidar_center_for_box(
    points_xyz: np.ndarray,
    bbox_xyxy: Sequence[float],
    transforms: KittiTransforms,
    *,
    min_points: int = 3,
    depth_bin_size: float = 1.0,
) -> LidarBoxEstimate | None:
    """Estimate an object's center from LiDAR returns projected into a 2D box.

    The central 70% of the box is preferred to reduce boundary/background
    contamination. A one-dimensional depth histogram then selects the nearest
    sufficiently populated surface cluster. This is intentionally lightweight;
    it is a much better depth cue than a constant distance, but it is not a
    substitute for a trained 3D detector or instance point segmentation.
    """

    if min_points < 1:
        raise ValueError("min_points must be at least 1")
    if depth_bin_size <= 0:
        raise ValueError("depth_bin_size must be positive")
    if len(bbox_xyxy) != 4:
        raise ValueError("bbox_xyxy must contain (x1, y1, x2, y2)")

    x1, y1, x2, y2 = map(float, bbox_xyxy)
    if not (x2 > x1 and y2 > y1):
        return None

    points = np.asarray(points_xyz, dtype=np.float64)
    uv, depth, valid = project_velodyne_to_image(points, transforms)
    full_mask = (
        valid
        & (uv[:, 0] >= x1)
        & (uv[:, 0] <= x2)
        & (uv[:, 1] >= y1)
        & (uv[:, 1] <= y2)
    )
    if np.count_nonzero(full_mask) < min_points:
        return None

    width, height = x2 - x1, y2 - y1
    central_mask = (
        full_mask
        & (uv[:, 0] >= x1 + 0.15 * width)
        & (uv[:, 0] <= x2 - 0.15 * width)
        & (uv[:, 1] >= y1 + 0.10 * height)
        & (uv[:, 1] <= y2 - 0.15 * height)
    )
    candidate_mask = central_mask if np.count_nonzero(central_mask) >= min_points else full_mask
    candidate_indices = np.flatnonzero(candidate_mask)
    candidate_depths = depth[candidate_indices]

    # Work with bins relative to the nearest candidate to avoid huge sparse
    # arrays when a scan contains distant outliers.
    min_depth = float(candidate_depths.min())
    bin_ids = np.floor((candidate_depths - min_depth) / depth_bin_size).astype(np.int64)
    unique_bins, counts = np.unique(bin_ids, return_counts=True)
    supported = unique_bins[counts >= min_points]
    if len(supported):
        selected_bin = int(supported.min())
    else:
        selected_bin = int(unique_bins[np.argmax(counts)])

    cluster_indices = candidate_indices[bin_ids == selected_bin]
    cluster_depths = depth[cluster_indices]
    median_depth = float(np.median(cluster_depths))

    # Include neighboring measurements close to the robust cluster median.
    close = np.abs(candidate_depths - median_depth) <= max(0.75, depth_bin_size)
    if np.count_nonzero(close) >= min_points:
        cluster_indices = candidate_indices[close]

    cluster_velodyne = points[cluster_indices, :3]
    cluster_rect = transform_points(cluster_velodyne, transforms.T_velo_to_rect)
    center_velodyne = np.median(cluster_velodyne, axis=0)
    center_rect = np.median(cluster_rect, axis=0)

    return LidarBoxEstimate(
        center_velodyne=center_velodyne,
        center_rect=center_rect,
        depth_m=float(center_rect[2]),
        range_m=float(np.linalg.norm(center_velodyne - transforms.camera_center_velodyne)),
        point_count=int(len(cluster_indices)),
    )


def class_dimensions_lwh(class_name: str) -> np.ndarray:
    """Return a copy of the visualization prior for a COCO class name."""

    return CLASS_DIMENSIONS_LWH.get(class_name.lower(), np.array([1.5, 1.5, 1.5])).copy()


def box_rotation_from_yaw(yaw: float) -> np.ndarray:
    """Return a z-up Open3D rotation matrix for a Velodyne-frame yaw."""

    cosine, sine = np.cos(float(yaw)), np.sin(float(yaw))
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

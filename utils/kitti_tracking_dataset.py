# SPDX-License-Identifier: AGPL-3.0-only
"""Lazy, validated access to one KITTI Tracking sequence.

The official downloads merge into a split layout such as::

    ROOT/training/image_02/0000/000000.png
    ROOT/training/velodyne/0000/000000.bin
    ROOT/training/calib/0000.txt
    ROOT/training/label_02/0000.txt

``ROOT`` may also point directly at the ``training`` or ``testing`` folder.
Images and point clouds remain on disk until :meth:`load_sample` is called.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import overload

import cv2
import numpy as np

from utils.bin_pointcloud_loader import load_velodyne_points
from utils.calib_loader import load_calibration
from utils.kitti_tracking_labels import (
    KittiTrackingLabel,
    KittiTrackingLabels,
    load_kitti_tracking_labels,
)


VALID_SPLITS = {"training", "testing"}


def normalize_sequence_id(sequence: str | int) -> str:
    """Return an official four-digit sequence ID and reject path-like input."""

    text = str(sequence).strip()
    if not text or not text.isascii() or not text.isdigit():
        raise ValueError(f"KITTI sequence must contain one to four digits, got {sequence!r}")
    if len(text) > 4:
        raise ValueError(f"KITTI sequence must contain at most four digits, got {sequence!r}")
    value = int(text)
    if value > 9999:
        raise ValueError(f"KITTI sequence is out of range: {sequence!r}")
    return f"{value:04d}"


def _resolve_split_root(root: Path, split: str) -> Path:
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}, got {split!r}")
    root = root.expanduser().resolve()
    candidates = (root / split, root)
    for candidate in candidates:
        if (candidate / "image_02").is_dir():
            if candidate == root and root.name in VALID_SPLITS and root.name != split:
                raise ValueError(
                    f"Dataset root points to the {root.name!r} split, but "
                    f"split={split!r} was requested"
                )
            return candidate
    checked = ", ".join(str(path / "image_02") for path in candidates)
    raise FileNotFoundError(
        f"Could not find the KITTI {split} image_02 directory. Checked: {checked}"
    )


def _numeric_files(directory: Path, suffix: str, modality: str) -> dict[int, Path]:
    files: dict[int, Path] = {}
    for path in directory.glob(f"*{suffix}"):
        if not path.is_file():
            continue
        if len(path.stem) != 6 or not path.stem.isdigit():
            raise ValueError(
                f"Invalid {modality} filename {path.name!r}; expected six digits followed by {suffix}"
            )
        frame_index = int(path.stem)
        if frame_index in files:
            raise ValueError(f"Duplicate {modality} frame ID {frame_index:06d} in {directory}")
        files[frame_index] = path
    return files


def _format_frame_ids(frame_ids: set[int], limit: int = 12) -> str:
    ordered = sorted(frame_ids)
    shown = ", ".join(f"{value:06d}" for value in ordered[:limit])
    if len(ordered) > limit:
        shown += f", ... (+{len(ordered) - limit} more)"
    return shown


@dataclass(frozen=True)
class KittiTrackingFrame:
    """Paths and annotations for a single synchronized sequence frame."""

    sequence: str
    frame_index: int
    image_path: Path
    pointcloud_path: Path | None
    calibration_path: Path | None
    labels: tuple[KittiTrackingLabel, ...] = ()

    @property
    def frame_id(self) -> str:
        return f"{self.frame_index:06d}"


@dataclass(frozen=True)
class KittiTrackingSample:
    """Materialized sensor data for one frame."""

    frame: KittiTrackingFrame
    image_bgr: np.ndarray
    points_xyzi: np.ndarray | None
    calibration: dict[str, np.ndarray] | None


class KittiTrackingDataset(Sequence[KittiTrackingFrame]):
    """Validate and lazily expose one KITTI Tracking sequence."""

    def __init__(
        self,
        root: str | Path,
        sequence: str | int,
        *,
        split: str = "training",
        require_pointcloud: bool = True,
        require_complete_pointcloud: bool = False,
        require_calibration: bool = True,
        load_labels: bool = True,
        require_labels: bool = False,
        strict: bool = True,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.sequence = normalize_sequence_id(sequence)
        self.split = split
        self.split_root = _resolve_split_root(self.root, split)
        self.image_dir = self.split_root / "image_02" / self.sequence
        self.velodyne_dir = self.split_root / "velodyne" / self.sequence
        self.calibration_path = self.split_root / "calib" / f"{self.sequence}.txt"
        self.label_path = self.split_root / "label_02" / f"{self.sequence}.txt"
        self.require_pointcloud = require_pointcloud
        self.require_complete_pointcloud = require_complete_pointcloud
        self.require_calibration = require_calibration
        self.strict = strict

        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image sequence directory does not exist: {self.image_dir}")
        image_files = _numeric_files(self.image_dir, ".png", "image")
        if not image_files:
            raise ValueError(f"No PNG frames found in {self.image_dir}")
        image_ids = set(image_files)

        if strict:
            expected_ids = set(range(max(image_ids) + 1))
            missing_images = expected_ids - image_ids
            if missing_images:
                raise ValueError(
                    "Image sequence is not contiguous from frame 000000; missing: "
                    f"{_format_frame_ids(missing_images)}"
                )

        pointcloud_files: dict[int, Path] = {}
        if self.velodyne_dir.is_dir():
            pointcloud_files = _numeric_files(self.velodyne_dir, ".bin", "point-cloud")
            missing_clouds = image_ids - set(pointcloud_files)
            extra_clouds = set(pointcloud_files) - image_ids
            if (require_pointcloud or require_complete_pointcloud) and not pointcloud_files:
                raise ValueError(f"No Velodyne point clouds found in {self.velodyne_dir}")
            if require_complete_pointcloud and missing_clouds:
                raise ValueError(
                    "Point clouds are missing for image frames: "
                    f"{_format_frame_ids(missing_clouds)}"
                )
            if strict and extra_clouds:
                raise ValueError(
                    "Point clouds have no matching image frames: "
                    f"{_format_frame_ids(extra_clouds)}"
                )
        elif require_pointcloud or require_complete_pointcloud:
            raise FileNotFoundError(
                f"Velodyne sequence directory does not exist: {self.velodyne_dir}"
            )
        self.missing_pointcloud_frame_ids = tuple(
            sorted(image_ids - set(pointcloud_files))
        )
        self.extra_pointcloud_frame_ids = tuple(
            sorted(set(pointcloud_files) - image_ids)
        )
        self.pointcloud_frame_count = len(image_ids & set(pointcloud_files))

        if require_calibration and not self.calibration_path.is_file():
            raise FileNotFoundError(
                f"Sequence calibration file does not exist: {self.calibration_path}"
            )

        labels: KittiTrackingLabels | None = None
        if require_labels and not self.label_path.is_file():
            raise FileNotFoundError(f"Tracking label file does not exist: {self.label_path}")
        if load_labels and self.label_path.is_file():
            labels = load_kitti_tracking_labels(self.label_path)
            if strict:
                dangling_frames = {label.frame for label in labels} - image_ids
                if dangling_frames:
                    raise ValueError(
                        "Labels reference frames with no image: "
                        f"{_format_frame_ids(dangling_frames)}"
                    )
        self._labels = labels

        self._frames = tuple(
            KittiTrackingFrame(
                sequence=self.sequence,
                frame_index=frame_index,
                image_path=image_files[frame_index],
                pointcloud_path=pointcloud_files.get(frame_index),
                calibration_path=(
                    self.calibration_path if self.calibration_path.is_file() else None
                ),
                labels=(labels.by_frame(frame_index) if labels is not None else ()),
            )
            for frame_index in sorted(image_ids)
        )

    def __len__(self) -> int:
        return len(self._frames)

    @overload
    def __getitem__(self, index: int) -> KittiTrackingFrame: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[KittiTrackingFrame, ...]: ...

    def __getitem__(
        self, index: int | slice
    ) -> KittiTrackingFrame | tuple[KittiTrackingFrame, ...]:
        return self._frames[index]

    def __iter__(self) -> Iterator[KittiTrackingFrame]:
        return iter(self._frames)

    @property
    def image_paths(self) -> tuple[Path, ...]:
        return tuple(frame.image_path for frame in self._frames)

    @property
    def has_pointclouds(self) -> bool:
        """Whether every image frame has a paired point cloud."""

        return all(frame.pointcloud_path is not None for frame in self._frames)

    @property
    def has_calibration(self) -> bool:
        return self.calibration_path.is_file()

    @property
    def has_labels(self) -> bool:
        return self._labels is not None

    @property
    def labels(self) -> KittiTrackingLabels | None:
        return self._labels

    @cached_property
    def calibration(self) -> dict[str, np.ndarray] | None:
        if not self.has_calibration:
            return None
        return load_calibration(self.calibration_path)

    def labels_for_frame(
        self, frame_index: int, *, include_dont_care: bool = True
    ) -> tuple[KittiTrackingLabel, ...]:
        if self._labels is None:
            return ()
        return self._labels.by_frame(frame_index, include_dont_care=include_dont_care)

    def load_sample(self, index: int) -> KittiTrackingSample:
        """Load image/point cloud for one frame; calibration stays cached."""

        frame = self[index]
        image = cv2.imread(str(frame.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode KITTI image: {frame.image_path}")
        points = (
            load_velodyne_points(frame.pointcloud_path)
            if frame.pointcloud_path is not None
            else None
        )
        return KittiTrackingSample(
            frame=frame,
            image_bgr=image,
            points_xyzi=points,
            calibration=self.calibration,
        )

    def summary(self) -> dict[str, object]:
        labels = self._labels
        return {
            "root": str(self.split_root),
            "split": self.split,
            "sequence": self.sequence,
            "frames": len(self),
            "pointcloud_frames": self.pointcloud_frame_count,
            "missing_pointcloud_frames": len(self.missing_pointcloud_frame_ids),
            "complete_pointclouds": self.has_pointclouds,
            "calibration": self.has_calibration,
            "labels": len(labels) if labels is not None else 0,
            "tracks": len(labels.track_ids) if labels is not None else 0,
        }


__all__ = [
    "KittiTrackingDataset",
    "KittiTrackingFrame",
    "KittiTrackingSample",
    "VALID_SPLITS",
    "normalize_sequence_id",
]

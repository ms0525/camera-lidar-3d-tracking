# SPDX-License-Identifier: AGPL-3.0-only
"""Adapter from the existing KITTI/YOLO pipeline to dashboard view objects.

The Streamlit entrypoint imports this module only when local live mode is
explicitly enabled.  Keeping these heavyweight imports out of public preview
mode is what allows the same app to deploy without PyTorch, Open3D, or KITTI.
"""

from __future__ import annotations

import argparse
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.dashboard_core import (
    AnalysisView,
    DashboardFrame,
    GroundTruthView,
    TrackView,
    track_color,
)
from track_3d_visualization import FrameAnalysis, Sequence3DProcessor, make_tracker
from utils.bin_pointcloud_loader import load_velodyne_points
from utils.kitti_tracking_dataset import KittiTrackingDataset, normalize_sequence_id


@dataclass(frozen=True, slots=True)
class LiveDashboardConfig:
    dataset_root: Path
    sequence: str
    yolo11_model: Path
    yolo26_model: Path
    split: str = "training"
    device: str | None = None
    confidence: float = 0.28
    imgsz: int = 640
    min_lidar_points: int = 3
    fallback_range_m: float = 0.0
    max_age: int = 4
    n_init: int = 3
    max_cosine_distance: float = 0.5
    nn_budget: int = 80
    embedder: str = "mobilenet"
    embedder_batch_size: int = 4
    embedder_gpu: bool = False
    yolo26_end2end: bool | None = None

    def normalized(self) -> "LiveDashboardConfig":
        root = self.dataset_root.expanduser().resolve()
        yolo11 = self.yolo11_model.expanduser().resolve()
        yolo26 = self.yolo26_model.expanduser().resolve()
        sequence = normalize_sequence_id(self.sequence)
        if self.split not in {"training", "testing"}:
            raise ValueError("split must be 'training' or 'testing'")
        if not root.is_dir():
            raise FileNotFoundError(f"KITTI Tracking root does not exist: {root}")
        for label, path in (("YOLO11", yolo11), ("YOLO26", yolo26)):
            if not path.is_file():
                raise FileNotFoundError(f"{label} checkpoint does not exist: {path}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between zero and one")
        if self.imgsz < 1:
            raise ValueError("imgsz must be positive")
        if self.min_lidar_points < 1:
            raise ValueError("min_lidar_points must be positive")
        if self.fallback_range_m < 0.0:
            raise ValueError("fallback_range_m cannot be negative")
        return LiveDashboardConfig(
            dataset_root=root,
            sequence=sequence,
            yolo11_model=yolo11,
            yolo26_model=yolo26,
            split=self.split,
            device=self.device or None,
            confidence=float(self.confidence),
            imgsz=int(self.imgsz),
            min_lidar_points=int(self.min_lidar_points),
            fallback_range_m=float(self.fallback_range_m),
            max_age=int(self.max_age),
            n_init=int(self.n_init),
            max_cosine_distance=float(self.max_cosine_distance),
            nn_budget=int(self.nn_budget),
            embedder=self.embedder,
            embedder_batch_size=int(self.embedder_batch_size),
            embedder_gpu=bool(self.embedder_gpu),
            yolo26_end2end=self.yolo26_end2end,
        )


@dataclass(slots=True)
class SharedModelResource:
    """A cached YOLO instance plus a lock for multi-session Streamlit safety."""

    model: Any
    lock: threading.RLock


class _LockedModelProxy:
    def __init__(self, resource: SharedModelResource) -> None:
        self._resource = resource

    @property
    def names(self) -> Any:
        return self._resource.model.names

    def predict(self, **kwargs: Any) -> Any:
        with self._resource.lock:
            return self._resource.model.predict(**kwargs)


def _processor_args(
    config: LiveDashboardConfig,
    calibration_path: Path,
    *,
    yolo_end2end: bool | None,
) -> argparse.Namespace:
    return argparse.Namespace(
        calib_file=calibration_path,
        confidence=config.confidence,
        imgsz=config.imgsz,
        yolo_end2end=yolo_end2end,
        device=config.device,
        min_lidar_points=config.min_lidar_points,
        fallback_range_m=config.fallback_range_m,
        max_age=config.max_age,
        n_init=config.n_init,
        max_cosine_distance=config.max_cosine_distance,
        nn_budget=config.nn_budget,
        embedder=config.embedder,
        embedder_batch_size=config.embedder_batch_size,
        half=False,
        embedder_gpu=config.embedder_gpu,
        export_kitti=None,
    )


def _to_analysis_view(
    analysis: FrameAnalysis,
    *,
    model_key: str,
    labels: tuple[Any, ...],
) -> AnalysisView:
    label_boxes = {
        int(label.track_id): tuple(float(value) for value in label.bbox)
        for label in labels
        if not label.is_dont_care
    }
    annotations = tuple(
        TrackView(
            track_id=item.track_id,
            class_name=item.class_name,
            bbox_xyxy=item.bbox_xyxy,
            center_velodyne=item.center_velodyne,
            dimensions_lwh=item.dimensions_lwh,
            # Track IDs are local to each model. Prefixing the color key avoids
            # accidentally implying cross-model identity correspondence.
            color_rgb=track_color(f"{model_key}:{item.track_id}"),
            source=item.source,
            confidence=item.confidence,
            lidar_point_count=item.lidar_point_count,
        )
        for item in analysis.annotations
    )
    ground_truth = tuple(
        GroundTruthView(
            track_id=item.track_id,
            object_type=item.object_type,
            center_velodyne=item.center_velodyne,
            rotation_velodyne=item.rotation_velodyne,
            extent_lwh=item.extent_lwh,
            bbox_xyxy=label_boxes.get(item.track_id),
        )
        for item in analysis.ground_truth
    )
    return AnalysisView(
        frame_name=analysis.frame_name,
        annotations=annotations,
        ground_truth=ground_truth,
        camera_origin_velodyne=analysis.camera_origin_velodyne,
        error=analysis.error,
    )


class LiveDashboardRuntime:
    """Two independent temporal trackers sharing one synchronized dataset."""

    def __init__(
        self,
        config: LiveDashboardConfig,
        yolo11_resource: SharedModelResource,
        yolo26_resource: SharedModelResource,
    ) -> None:
        self.config = config.normalized()
        self.dataset = KittiTrackingDataset(
            self.config.dataset_root,
            self.config.sequence,
            split=self.config.split,
            require_pointcloud=True,
            require_complete_pointcloud=False,
            require_calibration=True,
            load_labels=True,
            require_labels=False,
            strict=True,
        )
        frames = list(self.dataset.image_paths)
        ground_truth_frames = (
            tuple(frame.labels for frame in self.dataset)
            if self.dataset.has_labels
            else None
        )
        yolo11_args = _processor_args(
            self.config,
            self.dataset.calibration_path,
            yolo_end2end=None,
        )
        yolo26_args = _processor_args(
            self.config,
            self.dataset.calibration_path,
            yolo_end2end=self.config.yolo26_end2end,
        )
        self.processors = {
            "yolo11": Sequence3DProcessor(
                frames,
                self.dataset.calibration_path.parent,
                self.dataset.velodyne_dir,
                _LockedModelProxy(yolo11_resource),
                make_tracker(yolo11_args),
                yolo11_args,
                ground_truth_frames=ground_truth_frames,
            ),
            "yolo26": Sequence3DProcessor(
                frames,
                self.dataset.calibration_path.parent,
                self.dataset.velodyne_dir,
                _LockedModelProxy(yolo26_resource),
                make_tracker(yolo26_args),
                yolo26_args,
                ground_truth_frames=ground_truth_frames,
            ),
        }

    @property
    def frame_count(self) -> int:
        return len(self.dataset)

    @property
    def source_label(self) -> str:
        return f"KITTI {self.config.split} sequence {self.dataset.sequence}"

    @property
    def processed_through(self) -> int:
        """Largest frame cached by both temporal model pipelines."""

        return min(
            (max(processor.results, default=-1) for processor in self.processors.values()),
            default=-1,
        )

    def frame(self, index: int) -> DashboardFrame:
        if index < 0 or index >= self.frame_count:
            raise IndexError(index)
        frame_record = self.dataset[index]
        # Each processor advances its own Deep SORT timeline exactly once. Its
        # cached result then feeds both the camera and LiDAR panel for that model.
        yolo11_result = self.processors["yolo11"].ensure_processed(index)
        yolo26_result = self.processors["yolo26"].ensure_processed(index)
        # Reuse a processor's transient complete scan in the normal case. If a
        # model-specific failure discarded its result path, try the other model
        # and finally load the synchronized sensor frame independently.
        points = self.processors["yolo11"].points_for_display(index)
        if points is None:
            points = self.processors["yolo26"].points_for_display(index)
        if frame_record.pointcloud_path is not None:
            if points is None:
                try:
                    points = load_velodyne_points(frame_record.pointcloud_path)
                except (OSError, ValueError):
                    points = None

        image_bgr = cv2.imread(str(frame_record.image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Could not decode KITTI image: {frame_record.image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        labels = tuple(frame_record.labels)
        return DashboardFrame(
            frame_index=index,
            frame_count=self.frame_count,
            frame_name=frame_record.image_path.name,
            image_rgb=np.asarray(image_rgb),
            points_xyzi=points,
            yolo11=_to_analysis_view(
                yolo11_result,
                model_key="yolo11",
                labels=labels,
            ),
            yolo26=_to_analysis_view(
                yolo26_result,
                model_key="yolo26",
                labels=labels,
            ),
            source_label=self.source_label,
        )


def create_live_runtime(
    config: LiveDashboardConfig,
    yolo11_resource: SharedModelResource,
    yolo26_resource: SharedModelResource,
) -> LiveDashboardRuntime:
    return LiveDashboardRuntime(config, yolo11_resource, yolo26_resource)

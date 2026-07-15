# SPDX-License-Identifier: AGPL-3.0-only
"""Track 2D detections and visualize LiDAR-supported 3D locations.

This is a LiDAR association visualizer, not a learned 3D detector.  A 3D box is
drawn only when projected LiDAR points support the 2D detection, unless the user
explicitly enables the ray-only ``--fallback-range-m`` approximation.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import open3d as o3d
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

from utils.class_aware_deepsort import (
    DEFAULT_EMBEDDER_BATCH_SIZE,
    configure_embedder_for_inference,
    install_class_aware_association,
)
from utils.detection_thresholds import (
    confidence_for_class,
    model_inference_threshold,
    parse_class_confidence_overrides,
    validate_overrides_for_model,
)
from utils.bin_pointcloud_loader import load_velodyne_points
from utils.calib_loader import load_calibration
from utils.geometry import (
    KittiTransforms,
    box_rotation_from_yaw,
    camera_ray_in_velodyne,
    class_dimensions_lwh,
    estimate_lidar_center_for_box,
    kitti_camera_box_to_velodyne,
    load_kitti_transforms,
)
from utils.kitti_tracking_dataset import KittiTrackingDataset
from utils.kitti_tracking_labels import KittiTrackingLabel
from utils.kitti_tracking_results import (
    KittiTrackIdMapper,
    KittiTrackingPrediction,
    kitti_type_for_model_class,
    write_kitti_tracking_results,
)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

@dataclass(frozen=True)
class Track3DAnnotation:
    track_id: str
    class_name: str | None
    bbox_xyxy: tuple[float, float, float, float]
    center_velodyne: tuple[float, float, float] | None
    dimensions_lwh: tuple[float, float, float]
    color_rgb: tuple[float, float, float]
    source: str
    confidence: float = 1.0
    lidar_point_count: int = 0


@dataclass(frozen=True)
class GroundTruth3DAnnotation:
    track_id: int
    object_type: str
    center_velodyne: tuple[float, float, float]
    rotation_velodyne: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    extent_lwh: tuple[float, float, float]


@dataclass(frozen=True)
class FrameAnalysis:
    frame_name: str
    pointcloud_path: Path | None
    annotations: tuple[Track3DAnnotation, ...] = ()
    ground_truth: tuple[GroundTruth3DAnnotation, ...] = ()
    camera_origin_velodyne: tuple[float, float, float] | None = None
    error: str | None = None


def natural_key(path: Path) -> tuple[tuple[int, Any], ...]:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", path.name)
    )


def collect_frames(image_dir: Path) -> list[Path]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    return sorted(
        (
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
        ),
        key=natural_key,
    )


def resolve_class_name(value: Any, names: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        class_id = int(value)
    except (TypeError, ValueError):
        return str(value)
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    try:
        return str(names[class_id])
    except (IndexError, KeyError, TypeError):
        return str(class_id)


def track_class(track: Any, names: Any) -> str | None:
    getter = getattr(track, "get_det_class", None)
    value = getter() if callable(getter) else None
    if value is None:
        value = getattr(track, "det_class", None)
    if value is None:
        value = getattr(track, "cls", None)
    return resolve_class_name(value, names)


def track_detection_confidence(track: Any) -> float:
    getter = getattr(track, "get_det_conf", None)
    value = getter() if callable(getter) else None
    if value is None:
        value = getattr(track, "det_conf", None)
    if value is None:
        return 1.0
    confidence = float(value)
    return confidence if math.isfinite(confidence) else 1.0


def yolo_detections(
    result: Any,
    names: Any,
    threshold: float,
    class_confidence_thresholds: dict[str, float] | None = None,
) -> list[tuple[list[float], float, str]]:
    detections: list[tuple[list[float], float, str]] = []
    if result.boxes is None:
        return detections
    for box, confidence_tensor, class_tensor in zip(
        result.boxes.xyxy, result.boxes.conf, result.boxes.cls
    ):
        confidence = float(confidence_tensor.item())
        class_name = resolve_class_name(int(class_tensor.item()), names) or "object"
        effective_threshold = confidence_for_class(
            threshold, class_confidence_thresholds, class_name
        )
        if confidence < effective_threshold:
            continue
        x1, y1, x2, y2 = (float(value) for value in box.tolist())
        width, height = x2 - x1, y2 - y1
        if width <= 0.0 or height <= 0.0:
            continue
        detections.append(([x1, y1, width, height], confidence, class_name))
    return detections


def color_for_track(track_id: str) -> tuple[float, float, float]:
    """Generate a deterministic bright color without changing NumPy global state."""
    digest = hashlib.sha256(track_id.encode("utf-8")).digest()
    rgb = np.frombuffer(digest[:3], dtype=np.uint8).astype(np.float64) / 255.0
    rgb = 0.3 + 0.7 * rgb
    return tuple(float(value) for value in rgb)


def dimensions_for_class(class_name: str | None) -> tuple[float, float, float]:
    # The shared helper documents and returns Velodyne/Open3D [length, width,
    # height], rather than camera-frame [height, width, length].
    dimensions = class_dimensions_lwh(class_name or "object")
    return tuple(float(value) for value in dimensions)


def make_tracker(args: argparse.Namespace) -> DeepSort:
    tracker = install_class_aware_association(
        DeepSort(
            max_age=args.max_age,
            n_init=args.n_init,
            nms_max_overlap=1.0,
            max_cosine_distance=args.max_cosine_distance,
            nn_budget=args.nn_budget,
            embedder=args.embedder,
            half=args.half,
            embedder_gpu=args.embedder_gpu,
        )
    )
    return configure_embedder_for_inference(
        tracker,
        getattr(args, "embedder_batch_size", DEFAULT_EMBEDDER_BATCH_SIZE),
    )


class Sequence3DProcessor:
    """Process each temporal frame once while caching only its 3D annotations."""

    def __init__(
        self,
        frames: Sequence[Path],
        calib_dir: Path,
        pointcloud_dir: Path,
        model: YOLO,
        tracker: DeepSort,
        args: argparse.Namespace,
        ground_truth_frames: Sequence[Sequence[KittiTrackingLabel]] | None = None,
        class_confidence_thresholds: dict[str, float] | None = None,
    ) -> None:
        self.frames = frames
        self.calib_dir = calib_dir
        self.pointcloud_dir = pointcloud_dir
        self.model = model
        self.tracker = tracker
        self.args = args
        self.class_confidence_thresholds = dict(class_confidence_thresholds or {})
        self.inference_confidence_threshold = model_inference_threshold(
            args.confidence, self.class_confidence_thresholds
        )
        self.imgsz = getattr(args, "imgsz", 640)
        self.yolo_end2end = getattr(args, "yolo_end2end", None)
        self.kitti_classes_only = getattr(args, "export_kitti", None) is not None
        if ground_truth_frames is not None and len(ground_truth_frames) != len(frames):
            raise ValueError("ground_truth_frames must align one-to-one with frames")
        self.ground_truth_frames = ground_truth_frames
        self.names = model.names
        self.id_to_class: dict[str, str] = {}
        self.results: dict[int, FrameAnalysis] = {}
        self._transforms_by_calibration: dict[Path, KittiTransforms] = {}
        self._transient_points_index: int | None = None
        self._transient_points: np.ndarray | None = None

    def ensure_processed(self, index: int) -> FrameAnalysis:
        if index < 0 or index >= len(self.frames):
            raise IndexError(index)
        if index in self.results:
            return self.results[index]

        start = max(self.results, default=-1) + 1
        for pending_index in range(start, index + 1):
            self.results[pending_index] = self._process_one(pending_index)
        return self.results[index]

    def points_for_display(self, index: int) -> np.ndarray | None:
        result = self.ensure_processed(index)
        if result.pointcloud_path is None:
            return None
        if self._transient_points_index == index and self._transient_points is not None:
            return self._transient_points
        try:
            points = load_velodyne_points(result.pointcloud_path)
        except (OSError, ValueError) as exc:
            print(f"WARNING: Display point-cloud load failed: {exc}")
            return None
        self._transient_points_index = index
        self._transient_points = points
        return points

    def _process_one(self, index: int) -> FrameAnalysis:
        image_path = self.frames[index]
        frame_id = image_path.stem
        calib_path = self.args.calib_file or (self.calib_dir / f"{frame_id}.txt")
        expected_pointcloud_path = self.pointcloud_dir / f"{frame_id}.bin"

        frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if frame is None:
            return self._failure(image_path.name, f"Could not read image: {image_path}")

        # Calibration is essential for 3D association. A missing LiDAR frame is
        # not fatal: the official tracking archive has a small known gap, and
        # the 2D tracker must still advance to preserve its temporal state.
        try:
            if not calib_path.is_file():
                raise FileNotFoundError(f"Calibration does not exist: {calib_path}")
            transforms = self._transforms_by_calibration.get(calib_path)
            if transforms is None:
                calibration = load_calibration(str(calib_path))
                transforms = load_kitti_transforms(calibration)
                self._transforms_by_calibration[calib_path] = transforms
        except (OSError, KeyError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
            return self._failure(image_path.name, str(exc))

        ground_truth = self._ground_truth_annotations(index, transforms)

        points: np.ndarray | None = None
        pointcloud_path: Path | None = None
        try:
            points = load_velodyne_points(expected_pointcloud_path)
            pointcloud_path = expected_pointcloud_path
        except (OSError, ValueError) as exc:
            print(
                f"WARNING: {image_path.name}: LiDAR is unavailable; "
                f"continuing 2D tracking without 3D association ({exc})"
            )

        self._transient_points_index = index if points is not None else None
        self._transient_points = points

        predict_options: dict[str, Any] = {
            "source": frame,
            "conf": self.inference_confidence_threshold,
            "imgsz": self.imgsz,
            "verbose": False,
        }
        if self.args.device:
            predict_options["device"] = self.args.device
        if self.yolo_end2end is not None:
            predict_options["end2end"] = self.yolo_end2end

        try:
            result = self.model.predict(**predict_options)[0]
            detections = yolo_detections(
                result,
                self.names,
                self.args.confidence,
                self.class_confidence_thresholds,
            )
            if self.kitti_classes_only:
                detections = [
                    detection
                    for detection in detections
                    if kitti_type_for_model_class(detection[2]) is not None
                ]
            tracks = self.tracker.update_tracks(detections, frame=frame)
        except Exception as exc:
            return self._failure(
                image_path.name,
                f"Detection/tracking failed: {exc}",
                ground_truth=ground_truth,
            )

        try:
            camera_origin, _ = camera_ray_in_velodyne(
                (frame.shape[1] / 2.0, frame.shape[0] / 2.0), transforms
            )
            camera_origin_tuple = tuple(float(value) for value in camera_origin)
        except (ValueError, np.linalg.LinAlgError) as exc:
            return self._failure(
                image_path.name,
                f"Camera-ray transform failed: {exc}",
                ground_truth=ground_truth,
            )

        annotations: list[Track3DAnnotation] = []
        unsupported = 0
        for track in tracks:
            if not track.is_confirmed() or getattr(track, "time_since_update", 0) != 0:
                continue

            track_id = str(track.track_id)
            fresh_class = track_class(track, self.names)
            if fresh_class is not None:
                self.id_to_class[track_id] = fresh_class
            class_name = self.id_to_class.get(track_id)
            bbox = tuple(float(value) for value in track.to_ltrb())

            center: tuple[float, float, float] | None = None
            source = "no-lidar-support" if points is not None else "lidar-unavailable"
            lidar_point_count = 0
            estimate = None
            if points is not None:
                try:
                    estimate = estimate_lidar_center_for_box(
                        points,
                        bbox,
                        transforms,
                        min_points=self.args.min_lidar_points,
                    )
                except (ValueError, np.linalg.LinAlgError) as exc:
                    print(f"WARNING: Track {track_id} LiDAR association failed: {exc}")

            if estimate is not None:
                center = tuple(float(value) for value in estimate.center_velodyne)
                lidar_point_count = int(estimate.point_count)
                source = "lidar"
            elif self.args.fallback_range_m > 0.0:
                x1, y1, x2, y2 = bbox
                origin, direction = camera_ray_in_velodyne(
                    ((x1 + x2) / 2.0, (y1 + y2) / 2.0), transforms
                )
                fallback_center = origin + direction * self.args.fallback_range_m
                center = tuple(float(value) for value in fallback_center)
                source = f"ray-fallback-{self.args.fallback_range_m:g}m"
            else:
                unsupported += 1

            annotations.append(
                Track3DAnnotation(
                    track_id=track_id,
                    class_name=class_name,
                    bbox_xyxy=bbox,
                    center_velodyne=center,
                    dimensions_lwh=dimensions_for_class(class_name),
                    color_rgb=color_for_track(track_id),
                    source=source,
                    confidence=track_detection_confidence(track),
                    lidar_point_count=lidar_point_count,
                )
            )

        located = sum(annotation.center_velodyne is not None for annotation in annotations)
        print(
            f"[{index + 1}/{len(self.frames)}] {image_path.name} | "
            f"detections={len(detections)} fresh_confirmed={len(annotations)} "
            f"lidar/fallback_boxes={located} unsupported={unsupported} "
            f"gt_boxes={len(ground_truth)}"
        )
        for annotation in annotations:
            class_label = annotation.class_name or "object"
            print(
                f"  ID {annotation.track_id}: {class_label}, source={annotation.source}, "
                f"lidar_points={annotation.lidar_point_count}"
            )

        return FrameAnalysis(
            frame_name=image_path.name,
            pointcloud_path=pointcloud_path,
            annotations=tuple(annotations),
            ground_truth=ground_truth,
            camera_origin_velodyne=camera_origin_tuple,
        )

    def _ground_truth_annotations(
        self, index: int, transforms: KittiTransforms
    ) -> tuple[GroundTruth3DAnnotation, ...]:
        if self.ground_truth_frames is None:
            return ()

        annotations: list[GroundTruth3DAnnotation] = []
        for label in self.ground_truth_frames[index]:
            if label.is_dont_care or label.z <= 0.0:
                continue
            try:
                box = kitti_camera_box_to_velodyne(
                    label.location_xyz,
                    label.dimensions_hwl,
                    label.rotation_y,
                    transforms,
                )
            except ValueError as exc:
                print(
                    f"WARNING: GT track {label.track_id} in frame {label.frame:06d} "
                    f"cannot be rendered: {exc}"
                )
                continue
            annotations.append(
                GroundTruth3DAnnotation(
                    track_id=label.track_id,
                    object_type=label.type,
                    center_velodyne=tuple(float(value) for value in box.center_velodyne),
                    rotation_velodyne=tuple(
                        tuple(float(value) for value in row)
                        for row in box.rotation_velodyne
                    ),
                    extent_lwh=tuple(float(value) for value in box.extent_lwh),
                )
            )
        return tuple(annotations)

    @staticmethod
    def _failure(
        frame_name: str,
        message: str,
        *,
        ground_truth: tuple[GroundTruth3DAnnotation, ...] = (),
    ) -> FrameAnalysis:
        print(f"WARNING: {frame_name}: {message}")
        return FrameAnalysis(
            frame_name=frame_name,
            pointcloud_path=None,
            ground_truth=ground_truth,
            error=message,
        )


def point_cloud_geometry(points_xyzi: np.ndarray, point_size: float) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points_xyzi[:, :3].astype(np.float64))

    intensity = points_xyzi[:, 3].astype(np.float64)
    low, high = np.percentile(intensity, [2.0, 98.0])
    if high > low:
        shade = np.clip((intensity - low) / (high - low), 0.0, 1.0)
    else:
        shade = np.full_like(intensity, 0.65)
    colors = np.column_stack((shade, shade, shade))
    cloud.colors = o3d.utility.Vector3dVector(colors)
    # point_size is applied on the visualizer render options; retained in signature
    # to make the rendering dependency explicit.
    _ = point_size
    return cloud


def analysis_geometries(
    result: FrameAnalysis,
    points: np.ndarray | None,
    point_size: float,
) -> list[Any]:
    geometries: list[Any] = []
    if points is not None:
        geometries.append(point_cloud_geometry(points, point_size))
    else:
        geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0))

    if result.camera_origin_velodyne is not None:
        camera = o3d.geometry.TriangleMesh.create_sphere(radius=0.12)
        camera.paint_uniform_color([0.2, 0.4, 1.0])
        camera.translate(np.asarray(result.camera_origin_velodyne, dtype=np.float64))
        geometries.append(camera)

    for annotation in result.ground_truth:
        ground_truth_box = o3d.geometry.OrientedBoundingBox(
            center=np.asarray(annotation.center_velodyne, dtype=np.float64),
            R=np.asarray(annotation.rotation_velodyne, dtype=np.float64),
            extent=np.asarray(annotation.extent_lwh, dtype=np.float64),
        )
        ground_truth_box.color = [1.0, 0.0, 1.0]
        geometries.append(ground_truth_box)

    ray_points: list[tuple[float, float, float]] = []
    ray_lines: list[tuple[int, int]] = []
    ray_colors: list[tuple[float, float, float]] = []
    for annotation in result.annotations:
        if annotation.center_velodyne is None:
            continue
        center = np.asarray(annotation.center_velodyne, dtype=np.float64)
        color = list(annotation.color_rgb)
        rotation = box_rotation_from_yaw(0.0)  # Object yaw is not estimated by this pipeline.
        box = o3d.geometry.OrientedBoundingBox(
            center=center,
            R=rotation,
            extent=np.asarray(annotation.dimensions_lwh, dtype=np.float64),
        )
        box.color = color
        geometries.append(box)

        marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.14)
        marker.paint_uniform_color(color)
        marker.translate(
            center + np.array([0.0, 0.0, annotation.dimensions_lwh[2] / 2.0 + 0.2])
        )
        geometries.append(marker)

        if result.camera_origin_velodyne is not None:
            start = len(ray_points)
            ray_points.extend((result.camera_origin_velodyne, annotation.center_velodyne))
            ray_lines.append((start, start + 1))
            ray_colors.append(annotation.color_rgb)

    if ray_lines:
        rays = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(np.asarray(ray_points, dtype=np.float64)),
            lines=o3d.utility.Vector2iVector(np.asarray(ray_lines, dtype=np.int32)),
        )
        rays.colors = o3d.utility.Vector3dVector(np.asarray(ray_colors, dtype=np.float64))
        geometries.append(rays)
    return geometries


def export_kitti_results(
    destination: Path,
    frames: Sequence[Path],
    results: dict[int, FrameAnalysis],
) -> tuple[int, int]:
    """Export complete 2D tracking rows; approximate 3D fields stay sentinel-valued."""

    if len(results) != len(frames) or set(results) != set(range(len(frames))):
        raise ValueError("KITTI export requires every sequence frame to be processed")
    failures = [result.frame_name for result in results.values() if result.error]
    if failures:
        raise ValueError(
            "KITTI export refused because frame processing failed: "
            + ", ".join(failures[:8])
        )

    mapper = KittiTrackIdMapper()
    predictions: list[KittiTrackingPrediction] = []
    skipped = 0
    for index, frame_path in enumerate(frames):
        if not frame_path.stem.isdigit():
            raise ValueError(f"KITTI export requires numeric frame names: {frame_path.name}")
        frame_index = int(frame_path.stem)
        for annotation in results[index].annotations:
            object_type = kitti_type_for_model_class(annotation.class_name)
            if object_type is None:
                skipped += 1
                continue
            predictions.append(
                KittiTrackingPrediction(
                    frame=frame_index,
                    track_id=mapper.encode(annotation.track_id),
                    object_type=object_type,
                    bbox_xyxy=annotation.bbox_xyxy,
                    score=annotation.confidence,
                )
            )
    return write_kitti_tracking_results(destination, predictions), skipped


class Persistent3DViewer:
    """One Open3D window with cached back/forward frame navigation."""

    def __init__(self, processor: Sequence3DProcessor, args: argparse.Namespace) -> None:
        self.processor = processor
        self.args = args
        self.index = 0
        self.running = True
        self.first_render = True
        self.vis = o3d.visualization.VisualizerWithKeyCallback()

    def run(self) -> int:
        created = self.vis.create_window(
            window_name=self.args.window_name,
            width=self.args.window_width,
            height=self.args.window_height,
        )
        if not created:
            print("Could not create an Open3D window (try --headless).")
            return 1

        render_options = self.vis.get_render_option()
        render_options.point_size = self.args.point_size
        render_options.background_color = np.asarray([0.03, 0.03, 0.03])
        self._register_controls()
        self.show(self.index)

        try:
            while self.running and self.vis.poll_events():
                self.vis.update_renderer()
                time.sleep(0.01)
        finally:
            self.vis.destroy_window()
        return 0

    def _register_controls(self) -> None:
        for key in (ord("N"), ord("n"), 32, 257, 262):
            self.vis.register_key_callback(key, self._next)
        for key in (ord("B"), ord("b"), ord("P"), ord("p"), 263):
            self.vis.register_key_callback(key, self._previous)
        for key in (ord("S"), ord("s")):
            self.vis.register_key_callback(key, self._save)
        for key in (ord("Q"), ord("q"), 256):
            self.vis.register_key_callback(key, self._quit)

    def show(self, index: int) -> None:
        result = self.processor.ensure_processed(index)
        points = self.processor.points_for_display(index)
        geometries = analysis_geometries(result, points, self.args.point_size)

        self.vis.clear_geometries()
        for geometry in geometries:
            self.vis.add_geometry(geometry, reset_bounding_box=self.first_render)
        self.first_render = False
        self.vis.update_renderer()
        print(
            f"Viewing [{index + 1}/{len(self.processor.frames)}] {result.frame_name}. "
            "N/Space/Enter/Right: next, B/P/Left: previous, S: screenshot, Q/Esc: quit"
        )
        if result.ground_truth:
            print(
                f"  Ground truth: {len(result.ground_truth)} magenta boxes; "
                "predictions use track-specific colors"
            )
        if result.error:
            print(f"  Frame error: {result.error}")

    def _next(self, _visualizer: Any) -> bool:
        if self.index < len(self.processor.frames) - 1:
            self.index += 1
            self.show(self.index)
        return False

    def _previous(self, _visualizer: Any) -> bool:
        if self.index > 0:
            self.index -= 1
            self.show(self.index)
        return False

    def _save(self, _visualizer: Any) -> bool:
        result = self.processor.ensure_processed(self.index)
        self.args.output_dir.mkdir(parents=True, exist_ok=True)
        destination = self.args.output_dir / f"track3d_{Path(result.frame_name).stem}.png"
        try:
            self.vis.capture_screen_image(str(destination), do_render=True)
        except RuntimeError as exc:
            print(f"WARNING: Could not save {destination}: {exc}")
        else:
            print(f"Saved: {destination}")
        return False

    def _quit(self, _visualizer: Any) -> bool:
        self.running = False
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="KITTI Tracking root (the folder containing training/testing)",
    )
    parser.add_argument(
        "--sequence",
        help="KITTI Tracking sequence ID, for example 0 or 0000",
    )
    parser.add_argument(
        "--split",
        choices=("training", "testing"),
        default="training",
        help="KITTI Tracking split used with --dataset-root",
    )
    parser.add_argument(
        "--allow-incomplete-dataset",
        action="store_true",
        help="Allow gaps in image IDs and labels outside the downloaded frames",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        help="Manual image folder (cannot be combined with KITTI dataset mode)",
    )
    parser.add_argument("--calib-dir", type=Path)
    parser.add_argument(
        "--calib-file",
        type=Path,
        help="One sequence calibration file, as used by the KITTI tracking dataset",
    )
    parser.add_argument("--pointcloud-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--show-ground-truth",
        action="store_true",
        help="Draw exact KITTI 3D label boxes in magenta (dataset training mode only)",
    )
    parser.add_argument(
        "--export-kitti",
        type=Path,
        help="Write a complete 2D KITTI result TXT file; requires dataset mode and --headless",
    )
    parser.add_argument("--model", default="yolo26s.pt", help="Ultralytics model path or name")
    parser.add_argument("--confidence", type=float, default=0.28)
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Ultralytics inference image size in pixels (default: 640)",
    )
    parser.add_argument(
        "--yolo-end2end",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Explicitly enable or disable the YOLO26 end-to-end prediction head",
    )
    parser.add_argument(
        "--class-confidence",
        action="append",
        default=[],
        metavar="CLASS=VALUE",
        help="Override confidence for a model class, e.g. person=0.45 (repeatable)",
    )
    parser.add_argument("--device", default=None, help="Ultralytics device, for example cpu or 0")
    parser.add_argument("--min-lidar-points", type=int, default=3)
    parser.add_argument(
        "--fallback-range-m",
        type=float,
        default=0.0,
        help="Opt-in camera-ray fallback distance; 0 draws no unsupported 3D boxes",
    )
    parser.add_argument("--max-age", type=int, default=4)
    parser.add_argument("--n-init", type=int, default=3)
    parser.add_argument("--max-cosine-distance", type=float, default=0.5)
    parser.add_argument("--nn-budget", type=int, default=80)
    parser.add_argument("--embedder", default="mobilenet")
    parser.add_argument(
        "--embedder-batch-size",
        type=int,
        default=DEFAULT_EMBEDDER_BATCH_SIZE,
        help=(
            "Maximum Deep SORT embedding batch size; smaller values reduce "
            "CPU/GPU memory pressure (default: 4)"
        ),
    )
    parser.add_argument("--half", action="store_true")
    parser.add_argument("--embedder-gpu", action="store_true")
    parser.add_argument("--window-name", default="LiDAR-supported 3D tracks")
    parser.add_argument("--window-width", type=int, default=1280)
    parser.add_argument("--window-height", type=int, default=720)
    parser.add_argument("--point-size", type=float, default=1.5)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Process all frames and print associations without opening Open3D",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    if not 0.0 <= args.confidence <= 1.0:
        raise ValueError("--confidence must be between 0 and 1")
    imgsz = getattr(args, "imgsz", 640)
    if imgsz < 1:
        raise ValueError("--imgsz must be a positive integer")
    embedder_batch_size = getattr(
        args, "embedder_batch_size", DEFAULT_EMBEDDER_BATCH_SIZE
    )
    if embedder_batch_size < 1:
        raise ValueError("--embedder-batch-size must be a positive integer")
    class_confidence_thresholds = parse_class_confidence_overrides(
        getattr(args, "class_confidence", ())
    )
    if args.min_lidar_points < 1:
        raise ValueError("--min-lidar-points must be at least 1")
    if args.fallback_range_m < 0.0:
        raise ValueError("--fallback-range-m cannot be negative")

    dataset_mode = args.dataset_root is not None or args.sequence is not None
    if args.show_ground_truth and not dataset_mode:
        raise ValueError("--show-ground-truth requires --dataset-root/--sequence")
    if args.export_kitti is not None:
        if not dataset_mode:
            raise ValueError("--export-kitti requires --dataset-root/--sequence")
        if not args.headless:
            raise ValueError("--export-kitti requires --headless for a complete sequence")
        if args.export_kitti.suffix.casefold() != ".txt":
            raise ValueError("--export-kitti must point to a .txt result file")
    dataset: KittiTrackingDataset | None = None
    if dataset_mode:
        if args.dataset_root is None or args.sequence is None:
            raise ValueError("--dataset-root and --sequence must be provided together")
        manual_inputs = {
            "--image-dir": args.image_dir,
            "--calib-dir": args.calib_dir,
            "--calib-file": args.calib_file,
            "--pointcloud-dir": args.pointcloud_dir,
        }
        conflicts = [name for name, value in manual_inputs.items() if value is not None]
        if conflicts:
            raise ValueError(
                f"{', '.join(conflicts)} cannot be combined with "
                "--dataset-root/--sequence"
            )
        dataset = KittiTrackingDataset(
            args.dataset_root,
            args.sequence,
            split=args.split,
            require_pointcloud=True,
            require_complete_pointcloud=False,
            require_calibration=True,
            load_labels=True,
            require_labels=args.show_ground_truth,
            strict=not args.allow_incomplete_dataset,
        )
        frames = list(dataset.image_paths)
        image_dir = dataset.image_dir
        calib_dir = dataset.calibration_path.parent
        pointcloud_dir = dataset.velodyne_dir
        args.calib_file = dataset.calibration_path
        args.output_dir = args.output_dir or Path("data/tracked_3d_frames") / dataset.sequence
    else:
        missing_inputs = [
            name
            for name, missing in (
                ("--image-dir", args.image_dir is None),
                (
                    "--calib-dir or --calib-file",
                    args.calib_dir is None and args.calib_file is None,
                ),
                ("--pointcloud-dir", args.pointcloud_dir is None),
            )
            if missing
        ]
        if missing_inputs:
            raise ValueError(
                "manual mode requires explicit " + ", ".join(missing_inputs)
            )

        assert args.image_dir is not None
        assert args.pointcloud_dir is not None
        image_dir = args.image_dir
        if args.calib_dir is not None:
            calib_dir = args.calib_dir
        else:
            assert args.calib_file is not None
            calib_dir = args.calib_file.parent
        pointcloud_dir = args.pointcloud_dir
        args.output_dir = args.output_dir or Path("data/tracked_3d_frames")

        if args.calib_file is not None:
            if not args.calib_file.is_file():
                raise FileNotFoundError(
                    f"Calibration file does not exist: {args.calib_file}"
                )
        elif not calib_dir.is_dir():
            raise FileNotFoundError(f"Calibration directory does not exist: {calib_dir}")
        if not pointcloud_dir.is_dir():
            raise FileNotFoundError(
                f"Point-cloud directory does not exist: {pointcloud_dir}"
            )
        frames = collect_frames(image_dir)

    if not frames:
        print(f"No supported image files found in: {image_dir}")
        return 1

    try:
        model = YOLO(args.model)
        tracker = make_tracker(args)
    except Exception as exc:
        print(
            "Failed to initialize YOLO/Deep SORT: "
            f"{type(exc).__name__}: {exc!r}"
        )
        return 1
    validate_overrides_for_model(class_confidence_thresholds, model.names)

    ground_truth_frames = (
        tuple(frame.labels for frame in dataset)
        if dataset is not None and args.show_ground_truth
        else None
    )
    processor = Sequence3DProcessor(
        frames,
        calib_dir,
        pointcloud_dir,
        model,
        tracker,
        args,
        ground_truth_frames=ground_truth_frames,
        class_confidence_thresholds=class_confidence_thresholds,
    )
    if class_confidence_thresholds:
        formatted = ", ".join(
            f"{name}={value:g}"
            for name, value in sorted(class_confidence_thresholds.items())
        )
        print(f"Class confidence overrides: {formatted}")
    if dataset is not None:
        details = dataset.summary()
        print(
            f"Loaded KITTI {details['split']} sequence {details['sequence']}: "
            f"{details['frames']} images, {details['pointcloud_frames']} point clouds, "
            f"{details['labels']} labels, {details['tracks']} tracks"
        )
        if dataset.missing_pointcloud_frame_ids:
            missing = ", ".join(
                f"{frame_id:06d}" for frame_id in dataset.missing_pointcloud_frame_ids
            )
            print(f"WARNING: Missing point clouds for frames: {missing}")
    else:
        print(f"Found {len(frames)} frames in '{image_dir}'")

    if args.headless:
        for index in range(len(frames)):
            result = processor.ensure_processed(index)
            if result.error:
                print(
                    "Headless processing stopped at "
                    f"{result.frame_name}: {result.error}"
                )
                return 1
        if args.export_kitti is not None:
            written, skipped = export_kitti_results(
                args.export_kitti, frames, processor.results
            )
            print(
                f"Exported {written} KITTI Car/Pedestrian rows to "
                f"'{args.export_kitti}' (skipped {skipped} other-class tracks)"
            )
        return 0
    return Persistent3DViewer(processor, args).run()


def main() -> int:
    try:
        return run(build_parser().parse_args())
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

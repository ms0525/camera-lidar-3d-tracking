# SPDX-License-Identifier: AGPL-3.0-only
"""Interactive 2D detection and Deep SORT tracking over an image sequence.

Frames are processed by the tracker at most once and only in forward order.  The
viewer caches the small set of annotations for each processed frame, so saving,
pressing an unsupported key, or browsing backwards cannot advance Deep SORT a
second time.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
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
class TrackAnnotation:
    track_id: str
    class_name: str | None
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float = 1.0


@dataclass(frozen=True)
class FrameResult:
    frame_name: str
    annotations: tuple[TrackAnnotation, ...] = ()
    error: str | None = None


def natural_key(path: Path) -> tuple[tuple[int, Any], ...]:
    """Return a stable natural-sort key that also works for mixed filenames."""
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


def resolve_class_name(value: Any, names: Any) -> str | None:
    """Resolve either a Deep SORT class string or a numeric YOLO class id."""
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
    """Cache the current matched detection confidence before Deep SORT mutates it."""

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


class SequenceProcessor:
    """Own the temporal tracker and cache immutable per-frame annotations."""

    def __init__(
        self,
        frames: Sequence[Path],
        model: YOLO,
        tracker: DeepSort,
        confidence_threshold: float,
        device: str | None,
        class_confidence_thresholds: dict[str, float] | None = None,
        imgsz: int = 640,
        yolo_end2end: bool | None = None,
        kitti_classes_only: bool = False,
    ) -> None:
        self.frames = frames
        self.model = model
        self.tracker = tracker
        self.confidence_threshold = confidence_threshold
        self.class_confidence_thresholds = dict(class_confidence_thresholds or {})
        self.inference_confidence_threshold = model_inference_threshold(
            confidence_threshold, self.class_confidence_thresholds
        )
        self.device = device
        self.imgsz = imgsz
        self.yolo_end2end = yolo_end2end
        self.kitti_classes_only = kitti_classes_only
        self.names = model.names
        self.results: dict[int, FrameResult] = {}
        self.id_to_class: dict[str, str] = {}

    def ensure_processed(self, index: int) -> FrameResult:
        """Process all unseen frames through *index* once, preserving time order."""
        if index < 0 or index >= len(self.frames):
            raise IndexError(index)
        if index in self.results:
            return self.results[index]

        start = max(self.results, default=-1) + 1
        for pending_index in range(start, index + 1):
            self.results[pending_index] = self._process_one(pending_index)
        return self.results[index]

    def _process_one(self, index: int) -> FrameResult:
        path = self.frames[index]
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            message = f"Could not read image: {path}"
            print(f"WARNING: {message}")
            return FrameResult(path.name, error=message)

        predict_options: dict[str, Any] = {
            "source": frame,
            "conf": self.inference_confidence_threshold,
            "imgsz": self.imgsz,
            "verbose": False,
        }
        if self.device:
            predict_options["device"] = self.device
        if self.yolo_end2end is not None:
            predict_options["end2end"] = self.yolo_end2end

        try:
            result = self.model.predict(**predict_options)[0]
            detections = yolo_detections(
                result,
                self.names,
                self.confidence_threshold,
                self.class_confidence_thresholds,
            )
            if self.kitti_classes_only:
                detections = [
                    detection
                    for detection in detections
                    if kitti_type_for_model_class(detection[2]) is not None
                ]
            tracks = self.tracker.update_tracks(detections, frame=frame)
        except Exception as exc:  # keep one corrupt frame from killing the viewer
            message = f"Processing failed for {path}: {exc}"
            print(f"WARNING: {message}")
            return FrameResult(path.name, error=message)

        annotations: list[TrackAnnotation] = []
        for track in tracks:
            if not track.is_confirmed() or getattr(track, "time_since_update", 0) != 0:
                continue

            track_id = str(track.track_id)
            fresh_class = track_class(track, self.names)
            if fresh_class is not None:
                self.id_to_class[track_id] = fresh_class
            class_name = self.id_to_class.get(track_id)
            bbox = tuple(float(value) for value in track.to_ltrb())
            annotations.append(
                TrackAnnotation(
                    track_id,
                    class_name,
                    bbox,
                    track_detection_confidence(track),
                )
            )

        print(
            f"[{index + 1}/{len(self.frames)}] {path.name} | "
            f"detections={len(detections)} fresh_confirmed_tracks={len(annotations)}"
        )
        return FrameResult(path.name, tuple(annotations))


def _clipped_box(
    bbox_xyxy: Sequence[float], width: int, height: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox_xyxy
    return (
        max(0, min(width - 1, round(x1))),
        max(0, min(height - 1, round(y1))),
        max(0, min(width - 1, round(x2))),
        max(0, min(height - 1, round(y2))),
    )


def _draw_labeled_box(
    frame: Any,
    bbox_xyxy: Sequence[float],
    label: str,
    color_bgr: tuple[int, int, int],
    *,
    thickness: int,
) -> None:
    height, width = frame.shape[:2]
    left, top, right, bottom = _clipped_box(bbox_xyxy, width, height)
    cv2.rectangle(frame, (left, top), (right, bottom), color_bgr, thickness)
    cv2.putText(
        frame,
        label,
        (left, max(20, top - 7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color_bgr,
        max(1, thickness),
    )


def draw_frame(
    path: Path,
    result: FrameResult,
    index: int,
    total: int,
    ground_truth: Sequence[KittiTrackingLabel] = (),
) -> Any:
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        frame = np.zeros((480, 960, 3), dtype=np.uint8)

    height, _ = frame.shape[:2]
    for annotation in ground_truth:
        if annotation.is_dont_care:
            _draw_labeled_box(
                frame,
                annotation.bbox,
                "GT Ignore",
                (0, 165, 255),
                thickness=1,
            )
        else:
            _draw_labeled_box(
                frame,
                annotation.bbox,
                f"GT ID {annotation.track_id}: {annotation.type}",
                (255, 0, 255),
                thickness=1,
            )

    for annotation in result.annotations:
        label = f"Pred ID {annotation.track_id}"
        if annotation.class_name:
            label += f": {annotation.class_name}"
        _draw_labeled_box(
            frame,
            annotation.bbox_xyxy,
            label,
            (0, 255, 0),
            thickness=2,
        )

    status = result.error or f"fresh confirmed tracks: {len(result.annotations)}"
    if ground_truth:
        real_count = sum(not label.is_dont_care for label in ground_truth)
        ignore_count = len(ground_truth) - real_count
        status += f" | GT={real_count} ignore={ignore_count}"
    cv2.putText(
        frame,
        f"[{index + 1}/{total}] {result.frame_name} | {status}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255) if result.error else (255, 255, 255),
        2,
    )
    if ground_truth:
        cv2.putText(
            frame,
            "Prediction: green | GT: magenta | DontCare: orange",
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
    cv2.putText(
        frame,
        "N/Space/Enter: next  B/P: previous  S: save  Q/Esc: quit",
        (10, max(55, height - 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )
    return frame


def export_kitti_results(
    destination: Path,
    frames: Sequence[Path],
    results: dict[int, FrameResult],
) -> tuple[int, int]:
    """Export a complete cached sequence and return ``(written, skipped)``."""

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


def save_rendered(output_dir: Path, source: Path, image: Any) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"tracked_{source.name}"
    if not cv2.imwrite(str(destination), image):
        print(f"WARNING: Failed to save image: {destination}")
    else:
        print(f"Saved: {destination}")


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
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--show-ground-truth",
        action="store_true",
        help="Overlay KITTI tracking labels (dataset training mode only)",
    )
    parser.add_argument(
        "--export-kitti",
        type=Path,
        help="Write a complete KITTI result TXT file; requires dataset mode and --headless",
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
    parser.add_argument("--window-name", default="Tracked")
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
    parser.add_argument("--half", action="store_true", help="Use FP16 for the Deep SORT embedder")
    parser.add_argument("--embedder-gpu", action="store_true")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Process every frame without opening a window",
    )
    parser.add_argument(
        "--save-all",
        action="store_true",
        help="Save every rendered frame (also useful with --headless)",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    if not 0.0 <= args.confidence <= 1.0:
        raise ValueError("--confidence must be between 0 and 1")
    imgsz = getattr(args, "imgsz", 640)
    if imgsz < 1:
        raise ValueError("--imgsz must be a positive integer")
    yolo_end2end = getattr(args, "yolo_end2end", None)
    embedder_batch_size = getattr(
        args, "embedder_batch_size", DEFAULT_EMBEDDER_BATCH_SIZE
    )
    if embedder_batch_size < 1:
        raise ValueError("--embedder-batch-size must be a positive integer")
    class_confidence_thresholds = parse_class_confidence_overrides(
        getattr(args, "class_confidence", ())
    )

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
        if args.image_dir is not None:
            raise ValueError("--image-dir cannot be combined with --dataset-root/--sequence")
        dataset = KittiTrackingDataset(
            args.dataset_root,
            args.sequence,
            split=args.split,
            require_pointcloud=False,
            require_calibration=False,
            load_labels=True,
            require_labels=args.show_ground_truth,
            strict=not args.allow_incomplete_dataset,
        )
        frames = list(dataset.image_paths)
        image_dir = dataset.image_dir
        output_dir = args.output_dir or Path("data/tracked_frames") / dataset.sequence
    else:
        if args.image_dir is None:
            raise ValueError(
                "manual mode requires --image-dir; alternatively provide "
                "--dataset-root and --sequence"
            )
        image_dir = args.image_dir
        output_dir = args.output_dir or Path("data/tracked_frames")
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

    processor = SequenceProcessor(
        frames,
        model,
        tracker,
        args.confidence,
        args.device,
        class_confidence_thresholds=class_confidence_thresholds,
        imgsz=imgsz,
        yolo_end2end=yolo_end2end,
        kitti_classes_only=args.export_kitti is not None,
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
            f"{details['frames']} frames, {details['labels']} labels, "
            f"{details['tracks']} tracks"
        )
    else:
        print(f"Found {len(frames)} frames in '{image_dir}'")

    if args.headless:
        if args.show_ground_truth and not args.save_all:
            print("NOTE: --show-ground-truth is visible only when --save-all is used headlessly")
        for index, path in enumerate(frames):
            result = processor.ensure_processed(index)
            if result.error:
                print(
                    "Headless processing stopped at "
                    f"{result.frame_name}: {result.error}"
                )
                return 1
            if args.save_all:
                ground_truth = (
                    dataset[index].labels
                    if dataset is not None and args.show_ground_truth
                    else ()
                )
                save_rendered(
                    output_dir,
                    path,
                    draw_frame(path, result, index, len(frames), ground_truth),
                )
        if args.export_kitti is not None:
            written, skipped = export_kitti_results(
                args.export_kitti, frames, processor.results
            )
            print(
                f"Exported {written} KITTI Car/Pedestrian rows to "
                f"'{args.export_kitti}' (skipped {skipped} other-class tracks)"
            )
        return 0

    try:
        cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
    except cv2.error as exc:
        print(f"Could not open an OpenCV window (try --headless): {exc}")
        return 1

    index = 0
    while 0 <= index < len(frames):
        result = processor.ensure_processed(index)
        ground_truth = (
            dataset[index].labels
            if dataset is not None and args.show_ground_truth
            else ()
        )
        rendered = draw_frame(
            frames[index], result, index, len(frames), ground_truth
        )
        cv2.imshow(args.window_name, rendered)
        key = cv2.waitKey(0) & 0xFF

        if key in (27, ord("q"), ord("Q")):
            break
        if key in (ord("s"), ord("S")):
            save_rendered(output_dir, frames[index], rendered)
        elif key in (32, 10, 13, ord("n"), ord("N")):
            if index < len(frames) - 1:
                index += 1
        elif key in (ord("b"), ord("B"), ord("p"), ord("P")):
            index = max(0, index - 1)
        # Saving and unsupported keys intentionally leave the cached frame in place.

    cv2.destroyAllWindows()
    return 0


def main() -> int:
    try:
        return run(build_parser().parse_args())
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

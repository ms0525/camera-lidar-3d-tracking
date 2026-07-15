# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from utils.kitti_tracking_labels import KittiTrackingLabel, load_kitti_tracking_labels
from track_sequence import (
    SequenceProcessor,
    build_parser,
    natural_key,
    resolve_class_name,
    run,
)
from track_sequence import (
    FrameResult,
    TrackAnnotation,
    draw_frame,
    export_kitti_results,
)


class FakeModel:
    names = {0: "car"}

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, **kwargs):
        self.calls += 1
        self.last_predict_options = kwargs
        return [SimpleNamespace(boxes=None)]


class FakeTracker:
    def __init__(self) -> None:
        self.calls = 0

    def update_tracks(self, detections, frame):
        self.calls += 1
        self.last_detections = list(detections)
        self.last_shape = frame.shape
        return []


class TrackingStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        image_dir = Path(self._temporary_directory.name) / "images"
        image_dir.mkdir()
        self.image_paths = tuple(
            image_dir / f"{index:06d}.png" for index in range(2)
        )
        for index, path in enumerate(self.image_paths):
            image = np.full((48, 64, 3), index * 20, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(path), image))

    def test_manual_mode_requires_explicit_image_directory(self) -> None:
        args = build_parser().parse_args(["--headless"])
        with self.assertRaisesRegex(ValueError, r"manual mode requires --image-dir"):
            run(args)

    def test_headless_processing_stops_on_first_frame_error(self) -> None:
        class FailingProcessor:
            def __init__(self) -> None:
                self.calls: list[int] = []

            def ensure_processed(self, index: int) -> FrameResult:
                self.calls.append(index)
                return FrameResult(f"{index:06d}.png", error="synthetic failure")

        failing_processor = FailingProcessor()
        with tempfile.TemporaryDirectory() as directory:
            image_dir = Path(directory)
            (image_dir / "000000.png").touch()
            (image_dir / "000001.png").touch()
            args = build_parser().parse_args(
                ["--image-dir", str(image_dir), "--headless"]
            )
            with (
                patch("track_sequence.YOLO", return_value=FakeModel()),
                patch("track_sequence.make_tracker", return_value=FakeTracker()),
                patch(
                    "track_sequence.SequenceProcessor",
                    return_value=failing_processor,
                ),
            ):
                status = run(args)

        self.assertEqual(status, 1)
        self.assertEqual(failing_processor.calls, [0])

    def test_export_mode_tracks_only_kitti_evaluated_detector_classes(self) -> None:
        class MultiClassModel(FakeModel):
            names = {
                0: "car",
                1: "bus",
                2: "person",
                3: "bicycle",
                4: "pedestrian",
            }

            def predict(self, **kwargs):
                self.calls += 1
                boxes = SimpleNamespace(
                    xyxy=np.asarray(
                        [
                            [1.0, 1.0, 11.0, 11.0],
                            [2.0, 2.0, 12.0, 12.0],
                            [3.0, 3.0, 13.0, 13.0],
                            [4.0, 4.0, 14.0, 14.0],
                            [5.0, 5.0, 15.0, 15.0],
                        ]
                    ),
                    conf=np.asarray([0.9, 0.9, 0.9, 0.9, 0.9]),
                    cls=np.asarray([0, 1, 2, 3, 4]),
                )
                return [SimpleNamespace(boxes=boxes)]

        tracker = FakeTracker()
        processor = SequenceProcessor(
            [self.image_paths[0]],
            MultiClassModel(),
            tracker,
            0.25,
            None,
            kitti_classes_only=True,
        )

        processor.ensure_processed(0)

        self.assertEqual(
            [detection[2] for detection in tracker.last_detections],
            ["car", "person", "pedestrian"],
        )

    def test_yolo_prediction_cli_options_are_tristate(self) -> None:
        defaults = build_parser().parse_args([])
        enabled = build_parser().parse_args(["--imgsz", "1280", "--yolo-end2end"])
        disabled = build_parser().parse_args(["--no-yolo-end2end"])

        self.assertEqual(defaults.imgsz, 640)
        self.assertIsNone(defaults.yolo_end2end)
        self.assertEqual(enabled.imgsz, 1280)
        self.assertIs(enabled.yolo_end2end, True)
        self.assertIs(disabled.yolo_end2end, False)

    def test_processed_frame_is_never_sent_to_tracker_twice(self) -> None:
        frames = list(self.image_paths)
        model = FakeModel()
        tracker = FakeTracker()
        processor = SequenceProcessor(frames, model, tracker, 0.25, None)

        first_result = processor.ensure_processed(0)
        self.assertIs(processor.ensure_processed(0), first_result)
        self.assertEqual((model.calls, tracker.calls), (1, 1))

        processor.ensure_processed(1)
        processor.ensure_processed(0)  # simulates backward browsing
        self.assertEqual((model.calls, tracker.calls), (2, 2))

    def test_deepsort_string_class_is_preserved(self) -> None:
        self.assertEqual(resolve_class_name("car", {0: "person"}), "car")

    def test_class_override_lowers_upstream_inference_threshold(self) -> None:
        model = FakeModel()
        tracker = FakeTracker()
        processor = SequenceProcessor(
            [self.image_paths[0]],
            model,
            tracker,
            0.28,
            None,
            {"car": 0.15},
        )
        processor.ensure_processed(0)
        self.assertEqual(model.last_predict_options["conf"], 0.15)
        self.assertEqual(model.last_predict_options["imgsz"], 640)
        self.assertNotIn("end2end", model.last_predict_options)

    def test_prediction_options_include_explicit_imgsz_and_end2end(self) -> None:
        model = FakeModel()
        tracker = FakeTracker()
        processor = SequenceProcessor(
            [self.image_paths[0]],
            model,
            tracker,
            0.28,
            "cpu",
            imgsz=1280,
            yolo_end2end=True,
        )
        processor.ensure_processed(0)

        self.assertEqual(model.last_predict_options["imgsz"], 1280)
        self.assertIs(model.last_predict_options["end2end"], True)
        self.assertEqual(model.last_predict_options["device"], "cpu")

    def test_natural_sort_handles_mixed_names(self) -> None:
        names = [Path("frame_10.png"), Path("alpha.png"), Path("frame_2.png")]
        ordered = sorted(names, key=natural_key)
        self.assertEqual([path.name for path in ordered], ["alpha.png", "frame_2.png", "frame_10.png"])

    def test_export_preserves_frame_ids_scores_and_class_mapping(self) -> None:
        frames = [Path("000002.png"), Path("000004.png")]
        results = {
            0: FrameResult(
                "000002.png",
                (
                    TrackAnnotation("7", "car", (1.0, 2.0, 10.0, 20.0), 0.8),
                    TrackAnnotation("8", "truck", (2.0, 3.0, 11.0, 21.0), 0.7),
                ),
            ),
            1: FrameResult(
                "000004.png",
                (TrackAnnotation("7", "person", (3.0, 4.0, 12.0, 22.0), 0.6),),
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "0000.txt"
            written, skipped = export_kitti_results(path, frames, results)
            labels = load_kitti_tracking_labels(path)

        self.assertEqual((written, skipped), (2, 1))
        self.assertEqual([label.frame for label in labels], [2, 4])
        self.assertEqual([label.track_id for label in labels], [7, 7])
        self.assertEqual([label.type for label in labels], ["Car", "Pedestrian"])
        self.assertAlmostEqual(labels[0].score or 0.0, 0.8)

    def test_draw_frame_distinguishes_prediction_gt_and_dontcare(self) -> None:
        real = KittiTrackingLabel(
            0, 3, "Car", 0, 0, 0.0,
            5.0, 5.0, 15.0, 15.0,
            1.5, 1.6, 4.0, 0.0, 1.5, 10.0, 0.0,
        )
        dont_care = KittiTrackingLabel(
            0, -1, "DontCare", -1, -1, -10.0,
            40.0, 40.0, 50.0, 50.0,
            -1.0, -1.0, -1.0, -1000.0, -1000.0, -1000.0, -10.0,
        )
        result = FrameResult(
            "000000.png",
            (TrackAnnotation("1", "car", (20.0, 20.0, 30.0, 30.0), 0.9),),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "000000.png"
            self.assertTrue(cv2.imwrite(str(path), np.zeros((80, 80, 3), dtype=np.uint8)))
            rendered = draw_frame(path, result, 0, 1, (real, dont_care))

        np.testing.assert_array_equal(rendered[5, 5], [255, 0, 255])
        np.testing.assert_array_equal(rendered[20, 20], [0, 255, 0])
        np.testing.assert_array_equal(rendered[40, 40], [0, 165, 255])


if __name__ == "__main__":
    unittest.main()

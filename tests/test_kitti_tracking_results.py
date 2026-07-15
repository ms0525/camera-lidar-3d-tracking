# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from utils.kitti_tracking_labels import load_kitti_tracking_labels
from utils.kitti_tracking_results import (
    KittiTrackIdMapper,
    KittiTrackingPrediction,
    kitti_type_for_model_class,
    write_kitti_tracking_results,
)


class KittiTrackingResultTests(unittest.TestCase):
    def test_writes_official_18_field_row_and_round_trips(self) -> None:
        prediction = KittiTrackingPrediction(
            frame=4,
            track_id=12,
            object_type="Car",
            bbox_xyxy=(10.25, 20.5, 100.75, 200.0),
            score=0.875,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "0000.txt"
            count = write_kitti_tracking_results(path, [prediction])
            tokens = path.read_text(encoding="ascii").split()
            labels = load_kitti_tracking_labels(path)

        self.assertEqual(count, 1)
        self.assertEqual(len(tokens), 18)
        self.assertEqual(tokens[3:6], ["-1", "-1", "-10"])
        self.assertEqual(tokens[10:17], ["-1", "-1", "-1", "-1000", "-1000", "-1000", "-10"])
        self.assertEqual(labels[0].frame, 4)
        self.assertEqual(labels[0].track_id, 12)
        self.assertAlmostEqual(labels[0].score or 0.0, 0.875)

    def test_writer_sorts_frames_and_rejects_duplicate_ids(self) -> None:
        first = KittiTrackingPrediction(2, 1, "Car", (0.0, 0.0, 2.0, 2.0), 0.5)
        second = KittiTrackingPrediction(0, 1, "Car", (1.0, 1.0, 3.0, 3.0), 0.6)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.txt"
            write_kitti_tracking_results(path, [first, second])
            lines = path.read_text(encoding="ascii").splitlines()
            with self.assertRaisesRegex(ValueError, "duplicate prediction"):
                write_kitti_tracking_results(path, [first, first])

        self.assertTrue(lines[0].startswith("0 1 Car "))
        self.assertTrue(lines[1].startswith("2 1 Car "))

    def test_empty_predictions_create_an_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.txt"
            self.assertEqual(write_kitti_tracking_results(path, []), 0)
            self.assertEqual(path.read_bytes(), b"")

    def test_track_id_mapper_is_stable_and_collision_free(self) -> None:
        mapper = KittiTrackIdMapper()
        self.assertEqual(mapper.encode("7"), 7)
        self.assertEqual(mapper.encode("7"), 7)
        dated = mapper.encode("2026-07-14_7")
        self.assertNotEqual(dated, 7)
        self.assertEqual(mapper.encode("2026-07-14_7"), dated)

    def test_maps_only_evaluated_classes(self) -> None:
        self.assertEqual(kitti_type_for_model_class("car"), "Car")
        self.assertEqual(kitti_type_for_model_class("PERSON"), "Pedestrian")
        self.assertIsNone(kitti_type_for_model_class("truck"))
        self.assertIsNone(kitti_type_for_model_class("bicycle"))

    def test_rejects_invalid_numeric_values_and_boxes(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive width"):
            KittiTrackingPrediction(0, 1, "Car", (2.0, 1.0, 2.0, 3.0), 0.5)
        with self.assertRaisesRegex(ValueError, "score must be finite"):
            KittiTrackingPrediction(0, 1, "Car", (1.0, 1.0, 2.0, 3.0), float("nan"))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(TypeError, "KittiTrackingPrediction"):
                write_kitti_tracking_results(
                    Path(directory) / "bad.txt",
                    [object()],  # type: ignore[list-item]
                )


if __name__ == "__main__":
    unittest.main()

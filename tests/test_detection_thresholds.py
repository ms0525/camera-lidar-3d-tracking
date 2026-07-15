# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from track_3d_visualization import yolo_detections as yolo_3d_detections
from track_sequence import yolo_detections as yolo_2d_detections
from utils.detection_thresholds import (
    confidence_for_class,
    model_inference_threshold,
    parse_class_confidence_overrides,
    validate_overrides_for_model,
)


class DetectionThresholdTests(unittest.TestCase):
    def test_parses_normalizes_and_applies_overrides(self) -> None:
        overrides = parse_class_confidence_overrides(
            [" Person = 0.18", "car=0.35"]
        )
        self.assertEqual(overrides, {"person": 0.18, "car": 0.35})
        self.assertEqual(model_inference_threshold(0.28, overrides), 0.18)
        self.assertEqual(confidence_for_class(0.28, overrides, "Person"), 0.18)
        self.assertEqual(confidence_for_class(0.28, overrides, "bus"), 0.28)

    def test_rejects_malformed_duplicate_and_invalid_values(self) -> None:
        invalid_cases = (
            ["person"],
            ["=0.2"],
            ["person=nope"],
            ["person=nan"],
            ["person=inf"],
            ["person=-0.1"],
            ["person=1.1"],
            ["person=0.2", "Person=0.3"],
        )
        for assignments in invalid_cases:
            with self.subTest(assignments=assignments):
                with self.assertRaises(ValueError):
                    parse_class_confidence_overrides(assignments)

    def test_rejects_class_names_absent_from_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "exact model class names"):
            validate_overrides_for_model(
                {"pedestrian": 0.4}, {0: "person", 1: "car"}
            )

    def test_2d_and_3d_post_filters_use_the_class_threshold(self) -> None:
        names = {0: "person", 1: "car", 2: "bus", 3: "bicycle"}
        boxes = SimpleNamespace(
            xyxy=np.asarray(
                [
                    [0.0, 0.0, 10.0, 10.0],
                    [10.0, 0.0, 20.0, 10.0],
                    [20.0, 0.0, 30.0, 10.0],
                    [30.0, 0.0, 40.0, 10.0],
                ]
            ),
            conf=np.asarray([0.20, 0.30, 0.27, 0.29]),
            cls=np.asarray([0.0, 1.0, 2.0, 3.0]),
        )
        result = SimpleNamespace(boxes=boxes)
        overrides = {"person": 0.18, "car": 0.35}

        for detector in (yolo_2d_detections, yolo_3d_detections):
            with self.subTest(detector=detector.__module__):
                detections = detector(result, names, 0.28, overrides)
                self.assertEqual(
                    [detection[2] for detection in detections],
                    ["person", "bicycle"],
                )


if __name__ == "__main__":
    unittest.main()

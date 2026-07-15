# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from utils.kitti_tracking_labels import (
    KittiTrackingLabel,
    KittiTrackingLabelError,
    KittiTrackingLabels,
    load_kitti_tracking_labels,
    parse_kitti_tracking_labels,
)


CAR_FRAME_0 = (
    "0 7 Car 0 1 -0.25 100.0 50.0 220.0 160.0 "
    "1.50 1.60 4.20 1.0 1.5 20.0 0.30"
)
CAR_FRAME_1 = (
    "1 7 Car 0.00 0 -0.20 105.0 51.0 225.0 161.0 "
    "1.52 1.62 4.18 1.2 1.5 19.5 0.32"
)
PEDESTRIAN_FRAME_1_WITH_SCORE = (
    "1 12 Pedestrian 0.00 2 0.10 300.0 60.0 330.0 155.0 "
    "1.72 0.55 0.75 -2.0 1.7 15.0 -0.20 0.875"
)
DONT_CARE_FRAME_1 = (
    "1 -1 DontCare -1 -1 -10 400.0 70.0 510.0 180.0 "
    "-1 -1 -1 -1000 -1000 -1000 -10"
)


class KittiTrackingLabelParserTests(unittest.TestCase):
    def test_parses_all_ground_truth_fields(self) -> None:
        labels = parse_kitti_tracking_labels([CAR_FRAME_0])

        self.assertIsInstance(labels, KittiTrackingLabels)
        self.assertEqual(len(labels), 1)
        label = labels[0]
        self.assertIsInstance(label, KittiTrackingLabel)
        self.assertEqual(label.frame, 0)
        self.assertEqual(label.track_id, 7)
        self.assertEqual(label.type, "Car")
        self.assertEqual(label.object_type, "Car")
        self.assertEqual(label.truncated, 0)
        self.assertEqual(label.occluded, 1)
        self.assertAlmostEqual(label.alpha, -0.25)
        self.assertEqual(label.bbox, (100.0, 50.0, 220.0, 160.0))
        self.assertEqual(label.dimensions_hwl, (1.5, 1.6, 4.2))
        self.assertEqual(label.location_xyz, (1.0, 1.5, 20.0))
        self.assertAlmostEqual(label.rotation_y, 0.30)
        self.assertIsNone(label.score)

    def test_parses_optional_prediction_score(self) -> None:
        label = parse_kitti_tracking_labels(
            [PEDESTRIAN_FRAME_1_WITH_SCORE]
        )[0]

        self.assertAlmostEqual(label.score or 0.0, 0.875)

    def test_loads_path_and_ignores_blank_and_comment_lines(self) -> None:
        text = (
            "# KITTI tracking fixture\n\n"
            f"{CAR_FRAME_0}  # first observation\n"
            f"{CAR_FRAME_1}\n"
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "0000.txt"
            path.write_text(text, encoding="utf-8")

            labels = load_kitti_tracking_labels(path)

        self.assertEqual(len(labels), 2)
        self.assertEqual(labels.frames, (0, 1))

    def test_indexes_by_frame_and_track(self) -> None:
        labels = parse_kitti_tracking_labels(
            [CAR_FRAME_0, CAR_FRAME_1, PEDESTRIAN_FRAME_1_WITH_SCORE]
        )

        self.assertEqual(labels.frames, (0, 1))
        self.assertEqual(labels.track_ids, (7, 12))
        self.assertEqual(labels.by_frame(0), (labels[0],))
        self.assertEqual(labels.by_frame(1), (labels[1], labels[2]))
        self.assertEqual(labels.by_frame(999), ())
        self.assertEqual(labels.by_track(7), (labels[0], labels[1]))
        self.assertEqual(labels.by_track(999), ())
        self.assertEqual(tuple(labels.group_by_frame()), ((0, 1)))
        self.assertEqual(tuple(labels.group_by_track()), (7, 12))

    def test_accepts_and_filters_kitti_dontcare_sentinels(self) -> None:
        labels = parse_kitti_tracking_labels(
            [CAR_FRAME_1, DONT_CARE_FRAME_1]
        )

        dont_care = labels[1]
        self.assertTrue(dont_care.is_dont_care)
        self.assertEqual(dont_care.dimensions_hwl, (-1.0, -1.0, -1.0))
        self.assertEqual(dont_care.location_xyz, (-1000.0, -1000.0, -1000.0))
        self.assertEqual(labels.track_ids, (7,))
        self.assertEqual(labels.by_track(-1), (dont_care,))
        self.assertEqual(labels.object_labels, (labels[0],))
        self.assertEqual(labels.dont_care_labels, (dont_care,))
        self.assertEqual(tuple(labels.without_dont_care()), (labels[0],))
        self.assertEqual(tuple(labels.only_dont_care()), (dont_care,))
        self.assertEqual(labels.by_frame(1, include_dont_care=False), (labels[0],))

    def test_reports_source_line_for_wrong_field_count(self) -> None:
        with self.assertRaisesRegex(
            KittiTrackingLabelError,
            r"fixture\.txt:2: expected 17 fields .* got 3",
        ):
            parse_kitti_tracking_labels(
                ["# heading", "0 1 Car"], source="fixture.txt"
            )

    def test_reports_field_name_for_non_numeric_value(self) -> None:
        broken = CAR_FRAME_0.replace("1.50 1.60", "bad 1.60")

        with self.assertRaisesRegex(
            KittiTrackingLabelError,
            r"fixture\.txt:4: height must be numeric; got 'bad'",
        ):
            parse_kitti_tracking_labels(
                ["", "# comment", "", broken], source="fixture.txt"
            )

    def test_rejects_invalid_real_object_box(self) -> None:
        broken = CAR_FRAME_0.replace("100.0 50.0 220.0", "220.0 50.0 100.0")

        with self.assertRaisesRegex(
            KittiTrackingLabelError,
            r"labels:1: bbox_right must be greater than or equal to bbox_left",
        ):
            parse_kitti_tracking_labels([broken], source="labels")

    def test_accepts_zero_area_box_used_by_kitti_ignored_objects(self) -> None:
        collapsed = CAR_FRAME_0.replace(
            "100.0 50.0 220.0 160.0", "100.0 50.0 100.0 50.0"
        )

        label = parse_kitti_tracking_labels([collapsed])[0]

        self.assertEqual(label.bbox, (100.0, 50.0, 100.0, 50.0))

    def test_accepts_tracking_truncation_level_two_and_decimal_integer(self) -> None:
        label = parse_kitti_tracking_labels(
            [CAR_FRAME_0.replace("Car 0 1", "Car 2.00 1")]
        )[0]
        self.assertEqual(label.truncated, 2)

    def test_rejects_fractional_tracking_truncation(self) -> None:
        broken = CAR_FRAME_0.replace("Car 0 1", "Car 0.5 1")
        with self.assertRaisesRegex(KittiTrackingLabelError, "truncated must be integral"):
            parse_kitti_tracking_labels([broken])

    def test_rejects_invalid_real_object_dimensions(self) -> None:
        broken = CAR_FRAME_0.replace("1.50 1.60 4.20", "-1 -1 -1")

        with self.assertRaisesRegex(
            KittiTrackingLabelError,
            r"height, width, and length must be positive",
        ):
            parse_kitti_tracking_labels([broken])

    def test_dontcare_still_requires_a_valid_2d_region(self) -> None:
        broken = DONT_CARE_FRAME_1.replace(
            "400.0 70.0 510.0", "510.0 70.0 400.0"
        )

        with self.assertRaisesRegex(
            KittiTrackingLabelError,
            r"bbox_right must be greater than or equal to bbox_left",
        ):
            parse_kitti_tracking_labels([broken])

    def test_missing_path_has_useful_error(self) -> None:
        path = Path("definitely_missing_tracking_labels.txt")
        with self.assertRaisesRegex(
            FileNotFoundError, r"KITTI tracking label file does not exist"
        ):
            load_kitti_tracking_labels(path)


if __name__ == "__main__":
    unittest.main()

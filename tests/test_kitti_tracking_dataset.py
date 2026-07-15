# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from utils.calib_loader import load_calibration
from utils.kitti_tracking_dataset import KittiTrackingDataset, normalize_sequence_id


TRACKING_CALIBRATION = """\
P2: 1 0 0 0 0 1 0 0 0 0 1 0
R_rect 1 0 0 0 1 0 0 0 1
Tr_velo_cam 1 0 0 0 0 1 0 0 0 0 1 0
"""

TRACKING_LABELS = """\
0 7 Car 0 0 0.0 10 20 30 40 1.5 1.6 4.0 1 2 20 0.1
0 -1 DontCare -1 -1 -10 40 20 50 40 -1 -1 -1 -1000 -1000 -1000 -10
2 7 Car 1 1 0.1 11 20 31 40 1.5 1.6 4.0 1.2 2 20 0.1
2 9 Pedestrian 2 2 -0.2 60 10 70 40 1.7 0.6 0.8 2 2 15 -0.1
"""


def _create_sequence(
    root: Path,
    *,
    split: str = "training",
    sequence: str = "0003",
    frame_ids: tuple[int, ...] = (0, 1, 2),
    pointcloud_ids: tuple[int, ...] | None = None,
    labels: str | None = TRACKING_LABELS,
    calibration: str | None = TRACKING_CALIBRATION,
) -> Path:
    """Create a minimal on-disk sequence with real PNG and binary payloads."""

    split_root = root / split
    image_dir = split_root / "image_02" / sequence
    image_dir.mkdir(parents=True)
    for frame_id in frame_ids:
        image = np.full((4, 6, 3), (frame_id, 20, 200), dtype=np.uint8)
        written = cv2.imwrite(str(image_dir / f"{frame_id:06d}.png"), image)
        if not written:
            raise RuntimeError("OpenCV could not create a synthetic test image")

    if pointcloud_ids is None:
        pointcloud_ids = frame_ids
    if pointcloud_ids:
        pointcloud_dir = split_root / "velodyne" / sequence
        pointcloud_dir.mkdir(parents=True)
        for frame_id in pointcloud_ids:
            points = np.asarray(
                [[frame_id + 0.25, 1.0, 2.0, 0.5]], dtype="<f4"
            )
            points.tofile(pointcloud_dir / f"{frame_id:06d}.bin")

    if calibration is not None:
        calibration_dir = split_root / "calib"
        calibration_dir.mkdir(parents=True, exist_ok=True)
        (calibration_dir / f"{sequence}.txt").write_text(
            calibration, encoding="ascii"
        )

    if labels is not None:
        label_dir = split_root / "label_02"
        label_dir.mkdir(parents=True, exist_ok=True)
        (label_dir / f"{sequence}.txt").write_text(labels, encoding="utf-8")

    return split_root


class KittiTrackingDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary_directory.name)

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    def test_sequence_ids_are_normalized_and_path_like_values_are_rejected(self) -> None:
        self.assertEqual(normalize_sequence_id(0), "0000")
        self.assertEqual(normalize_sequence_id(" 17 "), "0017")
        self.assertEqual(normalize_sequence_id("9999"), "9999")

        for invalid in ("", "-1", "../1", "1/2", "00000", "12a", "\uff11"):
            with self.subTest(sequence=invalid):
                with self.assertRaises(ValueError):
                    normalize_sequence_id(invalid)

    def test_dataset_root_and_direct_split_root_resolve_to_same_sequence(self) -> None:
        split_root = _create_sequence(self.root)

        from_dataset_root = KittiTrackingDataset(self.root, 3)
        from_split_root = KittiTrackingDataset(split_root, "0003")

        self.assertEqual(from_dataset_root.split_root, split_root.resolve())
        self.assertEqual(from_split_root.split_root, split_root.resolve())
        self.assertEqual(from_dataset_root.image_paths, from_split_root.image_paths)
        self.assertEqual([frame.frame_id for frame in from_dataset_root], [
            "000000",
            "000001",
            "000002",
        ])

    def test_requested_split_is_used_and_invalid_split_is_rejected(self) -> None:
        _create_sequence(
            self.root,
            split="testing",
            sequence="0001",
            frame_ids=(0,),
            labels=None,
        )
        dataset = KittiTrackingDataset(
            self.root, 1, split="testing", load_labels=False
        )
        self.assertEqual(dataset.split, "testing")
        self.assertEqual(dataset.split_root, (self.root / "testing").resolve())

        with self.assertRaisesRegex(ValueError, "split"):
            KittiTrackingDataset(self.root, 1, split="validation")

        with self.assertRaisesRegex(ValueError, "points to the 'testing' split"):
            KittiTrackingDataset(
                self.root / "testing", 1, split="training", load_labels=False
            )

    def test_labels_are_indexed_by_frame_and_track(self) -> None:
        _create_sequence(self.root)
        dataset = KittiTrackingDataset(self.root, 3)

        self.assertTrue(dataset.has_labels)
        self.assertIsNotNone(dataset.labels)
        assert dataset.labels is not None
        self.assertEqual(dataset.labels.track_ids, (7, 9))
        self.assertEqual([label.track_id for label in dataset[0].labels], [7, -1])
        self.assertEqual(dataset[1].labels, ())
        self.assertEqual(
            [label.track_id for label in dataset.labels_for_frame(0)], [7, -1]
        )
        self.assertEqual(
            [
                label.track_id
                for label in dataset.labels_for_frame(
                    0, include_dont_care=False
                )
            ],
            [7],
        )
        self.assertEqual([label.frame for label in dataset.labels.by_track(7)], [0, 2])
        self.assertEqual(dataset.summary()["labels"], 4)
        self.assertEqual(dataset.summary()["tracks"], 2)

    def test_load_sample_materializes_image_pointcloud_and_cached_calibration(self) -> None:
        _create_sequence(self.root)
        dataset = KittiTrackingDataset(self.root, 3)

        sample = dataset.load_sample(2)

        self.assertEqual(sample.frame.frame_id, "000002")
        self.assertEqual(sample.image_bgr.shape, (4, 6, 3))
        np.testing.assert_array_equal(sample.image_bgr[0, 0], [2, 20, 200])
        self.assertIsNotNone(sample.points_xyzi)
        np.testing.assert_allclose(sample.points_xyzi, [[2.25, 1.0, 2.0, 0.5]])
        self.assertIs(sample.calibration, dataset.calibration)
        assert sample.calibration is not None
        self.assertIn("R0_rect", sample.calibration)
        self.assertIn("Tr_velo_to_cam", sample.calibration)

    def test_tracking_calibration_accepts_unseparated_alias_keys(self) -> None:
        split_root = _create_sequence(self.root, frame_ids=(0,))

        calibration = load_calibration(split_root / "calib" / "0003.txt")

        np.testing.assert_array_equal(
            calibration["R0_rect"], calibration["R_rect"]
        )
        np.testing.assert_array_equal(
            calibration["Tr_velo_to_cam"], calibration["Tr_velo_cam"]
        )

    def test_missing_pointcloud_frames_are_tolerated_and_reported_by_default(self) -> None:
        _create_sequence(self.root, pointcloud_ids=(0, 2))

        dataset = KittiTrackingDataset(self.root, 3, require_pointcloud=True)

        self.assertFalse(dataset.has_pointclouds)
        self.assertEqual(dataset.pointcloud_frame_count, 2)
        self.assertEqual(dataset.missing_pointcloud_frame_ids, (1,))
        self.assertIsNone(dataset[1].pointcloud_path)
        self.assertIsNone(dataset.load_sample(1).points_xyzi)

    def test_complete_pointcloud_mode_rejects_missing_frames(self) -> None:
        _create_sequence(self.root, pointcloud_ids=(0, 2))

        with self.assertRaisesRegex(ValueError, "missing"):
            KittiTrackingDataset(
                self.root, 3, require_complete_pointcloud=True
            )

    def test_pointcloud_modality_can_be_optional_but_required_mode_needs_directory(self) -> None:
        _create_sequence(self.root, pointcloud_ids=())

        optional = KittiTrackingDataset(
            self.root, 3, require_pointcloud=False
        )
        self.assertEqual(optional.pointcloud_frame_count, 0)
        self.assertEqual(optional.missing_pointcloud_frame_ids, (0, 1, 2))

        with self.assertRaises(FileNotFoundError):
            KittiTrackingDataset(self.root, 3, require_pointcloud=True)

    def test_dangling_label_frames_are_rejected_only_in_strict_mode(self) -> None:
        dangling_label = (
            "9 4 Car 0 0 0 1 1 2 2 1.5 1.6 4.0 0 1 10 0\n"
        )
        _create_sequence(self.root, frame_ids=(0,), labels=dangling_label)

        with self.assertRaisesRegex(ValueError, "no image"):
            KittiTrackingDataset(self.root, 3, strict=True)

        permissive = KittiTrackingDataset(self.root, 3, strict=False)
        self.assertTrue(permissive.has_labels)
        self.assertEqual(permissive[0].labels, ())
        assert permissive.labels is not None
        self.assertEqual(permissive.labels.frames, (9,))

    def test_labels_and_calibration_can_be_optional_or_required(self) -> None:
        _create_sequence(
            self.root,
            frame_ids=(0,),
            labels=None,
            calibration=None,
        )

        optional = KittiTrackingDataset(
            self.root,
            3,
            require_calibration=False,
            require_labels=False,
        )
        self.assertFalse(optional.has_calibration)
        self.assertFalse(optional.has_labels)
        self.assertIsNone(optional.calibration)
        self.assertEqual(optional.labels_for_frame(0), ())

        with self.assertRaises(FileNotFoundError):
            KittiTrackingDataset(self.root, 3, require_calibration=True)
        with self.assertRaises(FileNotFoundError):
            KittiTrackingDataset(
                self.root,
                3,
                require_calibration=False,
                require_labels=True,
            )

        # Requiring labels checks their presence independently of whether the
        # caller wants to parse them into memory.
        label_dir = self.root / "training" / "label_02"
        label_dir.mkdir(parents=True)
        (label_dir / "0003.txt").write_text(
            TRACKING_LABELS.splitlines()[0] + "\n", encoding="utf-8"
        )
        presence_only = KittiTrackingDataset(
            self.root,
            3,
            require_calibration=False,
            load_labels=False,
            require_labels=True,
        )
        self.assertFalse(presence_only.has_labels)

    def test_noncontiguous_images_are_rejected_only_in_strict_mode(self) -> None:
        _create_sequence(
            self.root,
            frame_ids=(0, 2),
            pointcloud_ids=(0, 2),
            labels=None,
        )

        with self.assertRaisesRegex(ValueError, "not contiguous"):
            KittiTrackingDataset(self.root, 3, strict=True)

        permissive = KittiTrackingDataset(self.root, 3, strict=False)
        self.assertEqual([frame.frame_index for frame in permissive], [0, 2])


if __name__ == "__main__":
    unittest.main()

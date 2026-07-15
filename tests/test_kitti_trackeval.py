# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from utils.kitti_trackeval import (
    SUPPORTED_TRACKEVAL_COMMIT,
    clear_previous_evaluation_outputs,
    compact_class_metrics,
    discover_prediction_files,
    load_trackeval,
    resolve_trackeval_root,
    run_trackeval,
    stage_trackeval_workspace,
    trackeval_revision,
    validate_evaluation_sequences,
    validate_tracker_name,
)
from utils.kitti_tracking_labels import load_kitti_tracking_labels


PROJECT_ROOT = Path(__file__).resolve().parents[1]


GROUND_TRUTH_ROWS = """\
0 1 Car 0 0 0 10 10 60 60 1.5 1.6 4.0 0 0 10 0
0 2 Pedestrian 0 0 0 70 10 100 80 1.7 0.6 0.8 0 0 10 0
1 1 Car 0 0 0 12 10 62 60 1.5 1.6 4.0 0 0 10 0
1 2 Pedestrian 0 0 0 72 10 102 80 1.7 0.6 0.8 0 0 10 0
"""

PERFECT_PREDICTION_ROWS = """\
0 1 Car -1 -1 -10 10 10 60 60 -1 -1 -1 -1000 -1000 -1000 -10 0.9
0 2 Pedestrian -1 -1 -10 70 10 100 80 -1 -1 -1 -1000 -1000 -1000 -10 0.9
1 1 Car -1 -1 -10 12 10 62 60 -1 -1 -1 -1000 -1000 -1000 -10 0.9
1 2 Pedestrian -1 -1 -10 72 10 102 80 -1 -1 -1 -1000 -1000 -1000 -10 0.9
"""


def _create_dataset(root: Path, *, sequence: str = "0003") -> Path:
    image_dir = root / "training" / "image_02" / sequence
    label_dir = root / "training" / "label_02"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    for frame in range(2):
        (image_dir / f"{frame:06d}.png").write_bytes(b"test")
    (label_dir / f"{sequence}.txt").write_text(
        GROUND_TRUTH_ROWS, encoding="utf-8"
    )
    return root


class KittiTrackEvalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.dataset_root = _create_dataset(self.root / "KITTI")
        self.prediction_dir = self.root / "predictions"
        self.prediction_dir.mkdir()
        self.prediction_path = self.prediction_dir / "0003.txt"
        self.prediction_path.write_text(PERFECT_PREDICTION_ROWS, encoding="ascii")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_discovers_single_file_and_requested_directory_subset(self) -> None:
        single = discover_prediction_files(self.prediction_path)
        subset = discover_prediction_files(self.prediction_dir, [3])
        self.assertEqual(single, subset)
        self.assertEqual(tuple(single), ("0003",))

    def test_validates_rows_and_generates_frame_count_seqmap(self) -> None:
        files = discover_prediction_files(self.prediction_path)
        sequences = validate_evaluation_sequences(
            self.dataset_root, files, ("car", "pedestrian")
        )
        self.assertEqual(sequences[0].frame_count, 2)
        self.assertEqual(sequences[0].prediction_rows, 4)
        self.assertEqual(sequences[0].selected_class_rows, 4)

        workspace = stage_trackeval_workspace(
            self.root / "workspace", sequences, "test_tracker"
        )
        self.assertEqual(
            workspace.sequence_map_path.read_text(encoding="utf-8"),
            "0003 empty 000000 000002\n",
        )
        self.assertTrue(
            (workspace.trackers_root / "test_tracker" / "data" / "0003.txt").is_file()
        )

    def test_rejects_duplicate_track_id_in_one_frame(self) -> None:
        self.prediction_path.write_text(
            PERFECT_PREDICTION_ROWS
            + "0 1 Car -1 -1 -10 20 20 40 50 -1 -1 -1 "
            "-1000 -1000 -1000 -10 0.8\n",
            encoding="ascii",
        )
        with self.assertRaisesRegex(ValueError, "occurs more than once"):
            validate_evaluation_sequences(
                self.dataset_root,
                discover_prediction_files(self.prediction_path),
                ("car",),
            )

    def test_rejects_out_of_range_frame_and_missing_score(self) -> None:
        cases = {
            "outside the sequence": (
                "2 1 Car -1 -1 -10 10 10 60 60 -1 -1 -1 "
                "-1000 -1000 -1000 -10 0.9\n"
            ),
            "eighteenth": (
                "0 1 Car 0 0 0 10 10 60 60 1.5 1.6 4.0 0 0 10 0\n"
            ),
        }
        for expected, row in cases.items():
            with self.subTest(expected=expected):
                self.prediction_path.write_text(row, encoding="ascii")
                with self.assertRaisesRegex(ValueError, expected):
                    validate_evaluation_sequences(
                        self.dataset_root,
                        discover_prediction_files(self.prediction_path),
                        ("car",),
                    )

    def test_rejects_non_evaluatable_prediction_type(self) -> None:
        self.prediction_path.write_text(
            "0 1 Cyclist -1 -1 -10 10 10 60 60 -1 -1 -1 "
            "-1000 -1000 -1000 -10 0.9\n",
            encoding="ascii",
        )
        with self.assertRaisesRegex(ValueError, "only Car and Pedestrian"):
            validate_evaluation_sequences(
                self.dataset_root,
                discover_prediction_files(self.prediction_path),
                ("car", "pedestrian"),
            )

    def test_empty_prediction_file_is_valid(self) -> None:
        self.prediction_path.write_text("", encoding="ascii")
        sequences = validate_evaluation_sequences(
            self.dataset_root,
            discover_prediction_files(self.prediction_path),
            ("car",),
        )
        self.assertEqual(sequences[0].prediction_rows, 0)
        self.assertEqual(sequences[0].selected_class_rows, 0)

    def test_rejects_unsafe_tracker_name(self) -> None:
        for value in (
            "../escape",
            "has space",
            "",
            "C:\\absolute",
            "NUL",
            "con.txt",
            "COM1",
            "tracker.",
        ):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_tracker_name(value)

    def test_staging_densifies_very_large_track_ids(self) -> None:
        large_id = 10**18
        self.prediction_path.write_text(
            f"0 {large_id} Car -1 -1 -10 10 10 60 60 -1 -1 -1 "
            "-1000 -1000 -1000 -10 0.9\n",
            encoding="ascii",
        )
        sequences = validate_evaluation_sequences(
            self.dataset_root,
            discover_prediction_files(self.prediction_path),
            ("car",),
        )
        workspace = stage_trackeval_workspace(
            self.root / "large-id-workspace", sequences, "safe"
        )
        staged = load_kitti_tracking_labels(
            workspace.trackers_root / "safe" / "data" / "0003.txt"
        )
        self.assertEqual(staged.track_ids, (0,))

    def test_clears_only_known_generated_evaluation_files(self) -> None:
        output = self.root / "output"
        output.mkdir()
        stale = output / "pedestrian_summary.txt"
        unrelated = output / "notes.txt"
        stale.write_text("old", encoding="utf-8")
        unrelated.write_text("keep", encoding="utf-8")
        clear_previous_evaluation_outputs(output)
        self.assertFalse(stale.exists())
        self.assertTrue(unrelated.exists())

    def test_perfect_sequence_scores_one_hundred_with_official_trackeval(self) -> None:
        try:
            trackeval_root = resolve_trackeval_root(None, project_root=PROJECT_ROOT)
        except FileNotFoundError as exc:
            self.skipTest(str(exc))

        commented_predictions = (
            "# comments and blank lines are accepted by the project parser\n\n"
            + PERFECT_PREDICTION_ROWS.replace(
                "-10 0.9\n", "-10 0.9 # inline comment\n", 1
            )
        )
        self.prediction_path.write_text(commented_predictions, encoding="ascii")
        sequences = validate_evaluation_sequences(
            self.dataset_root,
            discover_prediction_files(self.prediction_path),
            ("car", "pedestrian"),
        )
        workspace = stage_trackeval_workspace(
            self.root / "workspace", sequences, "perfect"
        )
        module = load_trackeval(trackeval_root)
        results = run_trackeval(
            module,
            workspace,
            self.root / "output with spaces",
            "perfect",
            ("car", "pedestrian"),
        )
        for class_name in ("car", "pedestrian"):
            metrics = compact_class_metrics(
                results["COMBINED_SEQ"][class_name]
            )
            with self.subTest(class_name=class_name):
                self.assertAlmostEqual(metrics["HOTA"], 100.0)
                self.assertAlmostEqual(metrics["MOTA"], 100.0)
                self.assertAlmostEqual(metrics["IDF1"], 100.0)
                self.assertEqual(metrics["FP"], 0)
                self.assertEqual(metrics["FN"], 0)
                self.assertEqual(metrics["IDSW"], 0)

        revision = trackeval_revision(trackeval_root)
        if revision is not None:
            self.assertEqual(revision, SUPPORTED_TRACKEVAL_COMMIT)


if __name__ == "__main__":
    unittest.main()

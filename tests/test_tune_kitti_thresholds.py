# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from tune_kitti_thresholds import build_parser, run


def _arguments(root: Path, *extra: str):
    return build_parser().parse_args(
        [
            "--dataset-root",
            str(root / "dataset"),
            "--predictions",
            str(root / "predictions"),
            "--source-class-confidence",
            "0.28",
            "--output-dir",
            str(root / "output"),
            *extra,
        ]
    )


class TuneKittiThresholdOrchestrationTests(unittest.TestCase):
    def test_default_run_never_discovers_filters_or_evaluates_holdout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prediction_directory = root / "predictions"
            prediction_directory.mkdir()
            tune_source = prediction_directory / "0000.txt"
            tune_source.write_text("tune-only source\n", encoding="ascii")
            args = _arguments(
                root,
                "--split-preset",
                "smoke",
                "--thresholds",
                "0.4",
            )
            counts = {
                "all_rows_before": 10,
                "all_rows_after": 8,
                "target_rows_before": 5,
                "target_rows_retained": 3,
                "target_rows_removed": 2,
            }
            baseline_counts = dict(counts, target_rows_retained=5, target_rows_removed=0)
            metrics = {
                "HOTA": 50.0,
                "DetA": 49.0,
                "AssA": 51.0,
                "MOTA": 40.0,
                "IDF1": 45.0,
                "FP": 1,
                "FN": 2,
                "IDSW": 0,
                "GroundTruthDetections": 10,
            }
            discovered_sequence_sets: list[tuple[str, ...]] = []
            filtered_sequence_sets: list[tuple[str, ...]] = []
            evaluated_sequence_sets: list[tuple[str, ...]] = []

            def discover(_directory: Path, sequences: tuple[str, ...]):
                discovered_sequence_sets.append(tuple(sequences))
                return {"0000": tune_source}

            def filter_set(
                _files: dict[str, Path],
                sequences: tuple[str, ...],
                _destination: Path,
                _target_class: str,
                threshold: float,
            ) -> dict[str, int]:
                filtered_sequence_sets.append(tuple(sequences))
                return baseline_counts if threshold == 0.28 else counts

            def evaluate(**kwargs: object):
                evaluated_sequence_sets.append(tuple(kwargs["sequences"]))  # type: ignore[arg-type]
                return metrics, ()

            stdout = io.StringIO()
            with (
                patch("tune_kitti_thresholds.discover_prediction_files", side_effect=discover),
                patch("tune_kitti_thresholds.validate_evaluation_sequences"),
                patch("tune_kitti_thresholds.resolve_trackeval_root", return_value=root),
                patch("tune_kitti_thresholds.trackeval_revision", return_value="test-revision"),
                patch("tune_kitti_thresholds.load_trackeval", return_value=object()),
                patch("tune_kitti_thresholds._filter_sequence_set", side_effect=filter_set),
                patch("tune_kitti_thresholds._evaluate_files", side_effect=evaluate),
                redirect_stdout(stdout),
            ):
                self.assertEqual(run(args), 0)

            report = json.loads(
                (root / "output" / "pedestrian" / "sweep.json").read_text(
                    encoding="utf-8"
                )
            )

        self.assertEqual(discovered_sequence_sets, [("0000",)])
        self.assertTrue(filtered_sequence_sets)
        self.assertTrue(evaluated_sequence_sets)
        self.assertTrue(all(value == ("0000",) for value in filtered_sequence_sets))
        self.assertTrue(all(value == ("0000",) for value in evaluated_sequence_sets))
        self.assertEqual(report["schema_version"], 2)
        self.assertEqual(set(report["source_files"]), {"0000"})
        self.assertFalse(report["source_experiment_manifest"]["present"])
        self.assertFalse(report["selection_protocol"]["holdout_predictions_read"])
        self.assertFalse(report["selection_protocol"]["holdout_ground_truth_read"])
        self.assertEqual(report["locked_holdout_evaluation"]["status"], "not_run")
        self.assertNotIn("locked_holdout_metrics", report)
        self.assertNotIn("0014.txt", stdout.getvalue())
        self.assertIn("Locked holdout was not read", stdout.getvalue())

    def test_requires_exactly_one_sequence_selection_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "provide --split-preset"):
                run(_arguments(root))

            conflicting = _arguments(
                root,
                "--split-preset",
                "smoke",
                "--tune-sequences",
                "0000",
                "--holdout-sequences",
                "0014",
            )
            with self.assertRaisesRegex(ValueError, "cannot be combined"):
                run(conflicting)

    def test_rejects_source_output_overlap_before_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = _arguments(root, "--split-preset", "smoke")
            args.predictions = args.output_dir / "pedestrian" / "selected_predictions"
            with self.assertRaisesRegex(ValueError, "must not overlap"):
                run(args)

    def test_rejects_empty_model_class(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = _arguments(Path(directory), "--split-preset", "smoke")
            args.model_class = "   "
            with self.assertRaisesRegex(ValueError, "must not be empty"):
                run(args)

    def test_rejects_semantically_wrong_model_class(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = _arguments(
                Path(directory),
                "--split-preset",
                "smoke",
                "--class",
                "car",
                "--model-class",
                "person",
            )
            with self.assertRaisesRegex(ValueError, "cannot produce"):
                run(args)

    def test_nonempty_output_requires_explicit_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            class_output = root / "output" / "pedestrian"
            class_output.mkdir(parents=True)
            (class_output / "sweep.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "--overwrite"):
                run(_arguments(root, "--split-preset", "smoke"))


if __name__ == "__main__":
    unittest.main()

# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from utils.kitti_threshold_sweep import (
    choose_best_threshold,
    filter_prediction_file,
    load_sequence_split_preset,
    normalize_model_class,
    normalize_thresholds,
    paths_overlap,
    sha256_file,
    validate_export_manifest_if_present,
    validate_sequence_split,
    write_sweep_csv,
)
from utils.kitti_tracking_labels import load_kitti_tracking_labels


PREDICTIONS = """\
# source comments are accepted
0 7 Pedestrian -1 -1 -10 10 10 40 80 -1 -1 -1 -1000 -1000 -1000 -10 0.50
0 8 Car -1 -1 -10 50 10 100 70 -1 -1 -1 -1000 -1000 -1000 -10 0.20
1 7 Pedestrian -1 -1 -10 12 10 42 80 -1 -1 -1 -1000 -1000 -1000 -10 0.49
"""


def _candidate(threshold: float, hota: float, idf1: float, mota: float) -> dict:
    return {
        "threshold": threshold,
        "row_counts": {"target_rows_before": 2, "target_rows_retained": 1},
        "metrics": {
            "HOTA": hota,
            "IDF1": idf1,
            "MOTA": mota,
            "DetA": hota,
            "AssA": hota,
            "FP": 0,
            "FN": 0,
            "IDSW": 0,
            "GroundTruthDetections": 2,
        },
    }


def _write_export_manifest(
    prediction_directory: Path,
    *,
    confidence: float = 0.4,
    overrides: dict[str, float] | None = None,
    schema_version: int = 1,
) -> Path:
    source = prediction_directory / "0000.txt"
    document = {
        "schema_version": schema_version,
        "status": "running",
        "configuration": {
            "dataset_root": "H:/Datasets/KITTI/tracking",
            "output_dir": str(prediction_directory),
            "sequences": ["0000", "0014"],
            "model": "detector.pt",
            "confidence": confidence,
            "class_confidence": overrides or {},
            "imgsz": 640,
            "yolo_end2end": True,
            "device": "cpu",
            "max_age": 4,
            "n_init": 3,
            "max_cosine_distance": 0.5,
            "nn_budget": 80,
            "embedder": "mobilenet",
            "half": False,
            "embedder_gpu": False,
        },
        "model": {"argument": "detector.pt", "local_file": False},
        "sequence_results": {
            "0000": {
                "status": "completed",
                "output_sha256": sha256_file(source),
            },
            "0014": {"status": "pending", "output_sha256": None},
        },
    }
    if schema_version >= 2:
        document["configuration"][
            "association_policy"
        ] = "deepsort-class-exact-match-v1"
    if schema_version >= 3:
        document["configuration"].update(
            {
                "tracked_class_policy": "kitti-evaluated-classes-v1",
                "embedder_batch_size": 4,
            }
        )
    path = prediction_directory / "experiment_manifest.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


class KittiThresholdSweepTests(unittest.TestCase):
    def test_schema_three_authenticates_memory_safe_tracking_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            predictions = Path(directory)
            source = predictions / "0000.txt"
            source.write_text(PREDICTIONS, encoding="ascii")
            _write_export_manifest(predictions, schema_version=3)

            provenance = validate_export_manifest_if_present(
                predictions,
                {"0000": source},
                ("0000",),
                target_class="pedestrian",
                model_class="person",
                declared_source_confidence=0.4,
            )

            self.assertEqual(provenance["schema_version"], 3)
            self.assertEqual(
                provenance["configuration"]["tracked_class_policy"],
                "kitti-evaluated-classes-v1",
            )
            self.assertEqual(
                provenance["configuration"]["embedder_batch_size"], 4
            )
            self.assertEqual(provenance["warnings"], [])

    def test_model_class_must_be_semantically_compatible(self) -> None:
        self.assertEqual(normalize_model_class("Car", None), "car")
        self.assertEqual(normalize_model_class("Pedestrian", None), "person")
        self.assertEqual(
            normalize_model_class("pedestrian", " PEDESTRIAN "), "pedestrian"
        )
        for target, model_class in (
            ("car", "person"),
            ("car", "pedestrian"),
            ("pedestrian", "car"),
        ):
            with self.subTest(target=target, model_class=model_class):
                with self.assertRaisesRegex(ValueError, "cannot produce"):
                    normalize_model_class(target, model_class)

    def test_export_manifest_separates_inference_floor_from_target_floor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            predictions = Path(directory)
            source = predictions / "0000.txt"
            source.write_text(PREDICTIONS, encoding="ascii")
            _write_export_manifest(
                predictions, confidence=0.4, overrides={"car": 0.2}
            )

            provenance = validate_export_manifest_if_present(
                predictions,
                {"0000": source},
                ("0000",),
                target_class="pedestrian",
                model_class="person",
                declared_source_confidence=0.4,
            )

        self.assertTrue(provenance["present"])
        self.assertIn("non-final", provenance["warnings"][0])
        self.assertEqual(provenance["thresholds"]["global_confidence"], 0.4)
        self.assertEqual(
            provenance["thresholds"]["upstream_yolo_inference_floor"], 0.2
        )
        self.assertEqual(
            provenance["thresholds"]["target_class_effective_confidence"], 0.4
        )
        self.assertEqual(provenance["configuration"]["imgsz"], 640)
        self.assertIs(provenance["configuration"]["yolo_end2end"], True)

    def test_export_manifest_requires_target_effective_source_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            predictions = Path(directory)
            source = predictions / "0000.txt"
            source.write_text(PREDICTIONS, encoding="ascii")
            _write_export_manifest(
                predictions, confidence=0.28, overrides={"person": 0.45}
            )

            with self.assertRaisesRegex(ValueError, "effective confidence 0.45"):
                validate_export_manifest_if_present(
                    predictions,
                    {"0000": source},
                    ("0000",),
                    target_class="pedestrian",
                    model_class="person",
                    declared_source_confidence=0.28,
                )

    def test_export_manifest_rejects_model_class_override_conflict_and_hash_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            predictions = Path(directory)
            source = predictions / "0000.txt"
            source.write_text(PREDICTIONS, encoding="ascii")
            _write_export_manifest(
                predictions, confidence=0.28, overrides={"pedestrian": 0.45}
            )
            with self.assertRaisesRegex(ValueError, "conflicts with export manifest"):
                validate_export_manifest_if_present(
                    predictions,
                    {"0000": source},
                    ("0000",),
                    target_class="pedestrian",
                    model_class="person",
                    declared_source_confidence=0.28,
                )

            _write_export_manifest(predictions, confidence=0.28)
            source.write_text(PREDICTIONS + "\n", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "hash does not match"):
                validate_export_manifest_if_present(
                    predictions,
                    {"0000": source},
                    ("0000",),
                    target_class="pedestrian",
                    model_class="person",
                    declared_source_confidence=0.28,
                )

    def test_plain_prediction_directory_returns_explicit_provenance_warning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            provenance = validate_export_manifest_if_present(
                root,
                {},
                ("0000",),
                target_class="pedestrian",
                model_class="person",
                declared_source_confidence=0.28,
            )
        self.assertFalse(provenance["present"])
        self.assertIn("could not be authenticated", provenance["warning"])

    def test_repository_split_presets_are_disjoint_and_complete(self) -> None:
        config = (
            Path(__file__).resolve().parents[1]
            / "config"
            / "kitti_tracking_splits.json"
        )
        smoke_tune, smoke_holdout = load_sequence_split_preset(config, "smoke")
        self.assertEqual(smoke_tune, ("0000",))
        self.assertEqual(smoke_holdout, ("0014",))

        tune, holdout = load_sequence_split_preset(config, "trackeval")
        self.assertEqual(
            tune,
            (
                "0000", "0001", "0003", "0004", "0005", "0009",
                "0011", "0012", "0015", "0017", "0019", "0020",
            ),
        )
        self.assertEqual(
            holdout,
            (
                "0002", "0006", "0007", "0008", "0010",
                "0013", "0014", "0016", "0018",
            ),
        )
        self.assertFalse(set(tune) & set(holdout))
        self.assertEqual(set(tune) | set(holdout), {f"{i:04d}" for i in range(21)})

    def test_normalizes_thresholds_and_always_includes_baseline(self) -> None:
        self.assertEqual(
            normalize_thresholds((0.5, 0.3, 0.5), source_threshold=0.28),
            (0.28, 0.3, 0.5),
        )
        for values in (
            (-0.1,),
            (0.27,),
            (1.1,),
            (float("nan"),),
            (float("inf"),),
        ):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    normalize_thresholds(values, source_threshold=0.28)
        with self.assertRaisesRegex(ValueError, "above the source"):
            normalize_thresholds((0.28,), source_threshold=0.28)

    def test_detects_source_output_path_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertTrue(paths_overlap(root, root / "selected"))
            self.assertTrue(paths_overlap(root / "selected", root))
            self.assertFalse(paths_overlap(root / "source", root / "output"))

    def test_sequence_split_is_disjoint_and_normalized(self) -> None:
        tune, holdout = validate_sequence_split((0, "12"), (14,))
        self.assertEqual(tune, ("0000", "0012"))
        self.assertEqual(holdout, ("0014",))
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_sequence_split((0, 1), (1, 2))

        with self.assertRaisesRegex(ValueError, "unknown split preset"):
            load_sequence_split_preset(Path("unused.json"), "unknown")

    def test_filter_is_inclusive_class_isolated_and_does_not_mutate_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "0000.txt"
            destination = root / "filtered" / "0000.txt"
            source.write_text(PREDICTIONS, encoding="ascii")
            digest_before = sha256_file(source)
            counts = filter_prediction_file(
                source, destination, "pedestrian", 0.50
            )
            filtered = load_kitti_tracking_labels(destination)

            self.assertEqual(sha256_file(source), digest_before)
            self.assertEqual(counts["target_rows_before"], 2)
            self.assertEqual(counts["target_rows_retained"], 1)
            self.assertEqual(
                [(label.frame, label.track_id, label.type) for label in filtered],
                [(0, 7, "Pedestrian"), (0, 8, "Car")],
            )
            retained_lines = destination.read_text(encoding="utf-8").splitlines()
            source_data_lines = [
                line
                for line in PREDICTIONS.splitlines()
                if line and not line.startswith("#")
            ]
            self.assertEqual(retained_lines, source_data_lines[:2])

    def test_selection_uses_predeclared_tie_breakers(self) -> None:
        results = (
            _candidate(0.3, 40.0, 35.0, 10.0),
            _candidate(0.5, 40.0, 38.0, 5.0),
            _candidate(0.7, 39.0, 90.0, 90.0),
        )
        self.assertEqual(choose_best_threshold(results)["threshold"], 0.5)

        exact_tie = (
            _candidate(0.5, 40.0, 38.0, 5.0),
            _candidate(0.4, 40.0, 38.0, 5.0),
        )
        self.assertEqual(choose_best_threshold(exact_tie)["threshold"], 0.4)

    def test_selection_rejects_degenerate_tuning_class(self) -> None:
        result = _candidate(0.3, 0.0, 0.0, 0.0)
        with self.assertRaisesRegex(ValueError, "cannot be tuned"):
            choose_best_threshold((result,))

    def test_csv_contains_candidate_metrics(self) -> None:
        results = (_candidate(0.3, 40.0, 35.0, 10.0),)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "sweep.csv"
            write_sweep_csv(destination, results)
            content = destination.read_text(encoding="utf-8")
        self.assertIn("threshold,target_rows_before", content)
        self.assertIn("0.3,2,1,40.0", content)


if __name__ == "__main__":
    unittest.main()

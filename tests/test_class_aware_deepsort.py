# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch
from deep_sort_realtime.deep_sort.detection import Detection
from deep_sort_realtime.deep_sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort_realtime.deepsort_tracker import DeepSort

from track_3d_visualization import make_tracker as make_3d_tracker
from track_sequence import make_tracker as make_2d_tracker
from utils.class_aware_deepsort import (
    ClassAwareTracker,
    canonical_detection_class,
    configure_embedder_for_inference,
    detection_classes_compatible,
    install_class_aware_association,
)


def detection(class_name: str) -> Detection:
    return Detection(
        np.asarray([20.0, 30.0, 40.0, 50.0]),
        0.9,
        np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        class_name=class_name,
    )


def tracker() -> ClassAwareTracker:
    metric = NearestNeighborDistanceMetric("cosine", 0.2, None)
    return ClassAwareTracker(
        metric,
        max_iou_distance=0.7,
        max_age=5,
        n_init=2,
    )


def confirmed_tracker() -> ClassAwareTracker:
    instance = tracker()
    instance.update([detection("Car")])
    instance.predict()
    instance.update([detection("car")])
    if not instance.tracks[0].is_confirmed():
        raise AssertionError("synthetic track was not confirmed")
    instance.predict()
    return instance


def tracker_args() -> SimpleNamespace:
    return SimpleNamespace(
        max_age=4,
        n_init=3,
        max_cosine_distance=0.5,
        nn_budget=80,
        embedder=None,
        half=False,
        embedder_gpu=False,
        embedder_batch_size=4,
    )


class ClassAwareDeepSortTests(unittest.TestCase):
    def test_embedder_is_chunked_and_runs_in_inference_mode(self) -> None:
        class FakeEmbedder:
            max_batch_size = 16

            def __init__(self) -> None:
                self.calls: list[tuple[int, bool, bool]] = []

            def predict(self, images):
                self.calls.append(
                    (
                        len(images),
                        torch.is_grad_enabled(),
                        torch.is_inference_mode_enabled(),
                    )
                )
                return [f"feature-{value}" for value in images]

        embedder = FakeEmbedder()
        wrapper = SimpleNamespace(embedder=embedder)

        configured = configure_embedder_for_inference(wrapper, 4)
        features = configured.embedder.predict(list(range(10)))

        self.assertIs(configured, wrapper)
        self.assertEqual(embedder.max_batch_size, 4)
        self.assertEqual([call[0] for call in embedder.calls], [4, 4, 2])
        self.assertTrue(all(not call[1] and call[2] for call in embedder.calls))
        self.assertEqual(features, [f"feature-{value}" for value in range(10)])

    def test_canonicalization_is_case_insensitive_but_not_alias_based(self) -> None:
        self.assertEqual(canonical_detection_class("  CaR  "), "car")
        self.assertNotEqual(
            canonical_detection_class("person"),
            canonical_detection_class("person_sitting"),
        )
        self.assertTrue(detection_classes_compatible(None, None))
        self.assertFalse(detection_classes_compatible("car", None))
        self.assertFalse(detection_classes_compatible(None, "car"))

    def test_confirmed_appearance_associates_same_class(self) -> None:
        instance = confirmed_tracker()

        matches, unmatched_tracks, unmatched_detections = instance._match(
            [detection(" CAR ")]
        )

        self.assertEqual(matches, [(0, 0)])
        self.assertEqual(unmatched_tracks, [])
        self.assertEqual(unmatched_detections, [])

    def test_confirmed_appearance_rejects_cross_class_even_with_same_feature(self) -> None:
        instance = confirmed_tracker()
        candidate = detection("person")

        appearance_cost = instance._appearance_cost(
            instance.tracks, [candidate], [0], [0]
        )
        matches, unmatched_tracks, unmatched_detections = instance._match([candidate])

        self.assertEqual(appearance_cost[0, 0], 100000.0)
        self.assertEqual(matches, [])
        self.assertEqual(unmatched_tracks, [0])
        self.assertEqual(unmatched_detections, [0])
        self.assertEqual(instance.tracks[0].det_class, "car")

    def test_unconfirmed_iou_associates_same_class(self) -> None:
        instance = tracker()
        instance._initiate_track(detection("Car"))
        instance.predict()

        matches, unmatched_tracks, unmatched_detections = instance._match(
            [detection("car")]
        )

        self.assertEqual(matches, [(0, 0)])
        self.assertEqual(unmatched_tracks, [])
        self.assertEqual(unmatched_detections, [])

    def test_unconfirmed_iou_rejects_overlapping_cross_class(self) -> None:
        instance = tracker()
        instance._initiate_track(detection("car"))
        instance.predict()
        candidate = detection("person")

        iou_cost = instance._iou_cost(instance.tracks, [candidate], [0], [0])
        matches, unmatched_tracks, unmatched_detections = instance._match([candidate])

        self.assertEqual(iou_cost[0, 0], 100000.0)
        self.assertEqual(matches, [])
        self.assertEqual(unmatched_tracks, [0])
        self.assertEqual(unmatched_detections, [0])

    def test_install_preserves_wrapper_and_internal_state(self) -> None:
        wrapper = DeepSort(
            embedder=None,
            max_age=9,
            n_init=4,
            max_iou_distance=0.6,
            max_cosine_distance=0.3,
        )
        metric = wrapper.tracker.metric
        kalman_filter = wrapper.tracker.kf

        installed = install_class_aware_association(wrapper)

        self.assertIs(installed, wrapper)
        self.assertIsInstance(installed.tracker, ClassAwareTracker)
        self.assertIs(installed.tracker.metric, metric)
        self.assertIs(installed.tracker.kf, kalman_filter)
        self.assertEqual(installed.tracker.max_age, 9)
        self.assertEqual(installed.tracker.n_init, 4)
        self.assertEqual(installed.tracker.max_iou_distance, 0.6)
        self.assertIsNone(installed.embedder)

    def test_both_application_factories_install_class_aware_tracker(self) -> None:
        tracker_2d = make_2d_tracker(tracker_args())
        tracker_3d = make_3d_tracker(tracker_args())

        self.assertIsInstance(tracker_2d, DeepSort)
        self.assertIsInstance(tracker_2d.tracker, ClassAwareTracker)
        self.assertIsInstance(tracker_3d, DeepSort)
        self.assertIsInstance(tracker_3d.tracker, ClassAwareTracker)
        self.assertIsNone(tracker_2d.embedder)
        self.assertIsNone(tracker_3d.embedder)


if __name__ == "__main__":
    unittest.main()

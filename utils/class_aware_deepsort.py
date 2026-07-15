# SPDX-License-Identifier: AGPL-3.0-only
"""Class-aware association for :mod:`deep_sort_realtime`.

``deep-sort-realtime`` 1.3.2 does not consider a detection's class during
association.  Consequently, a track can change from (for example) ``car`` to
``person`` when appearance or overlap costs happen to be favourable.  This
module keeps the package's public :class:`DeepSort` wrapper and embedder, but
replaces its internal association engine with a subclass that rejects those
cross-class matches in both matching stages.
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Sequence

import numpy as np
from deep_sort_realtime.deep_sort import iou_matching, linear_assignment
from deep_sort_realtime.deep_sort.tracker import Tracker
from deep_sort_realtime.deepsort_tracker import DeepSort


DEFAULT_EMBEDDER_BATCH_SIZE = 4


def canonical_detection_class(value: Any) -> str | None:
    """Return the class key used for association.

    Detector class names are semantic identifiers, so only surrounding
    whitespace and character case are normalised.  Deliberately avoiding
    aliases keeps distinct labels such as ``person`` and ``person_sitting``
    from sharing a track.
    """

    if value is None:
        return None
    key = str(value).strip().casefold()
    return key or None


def detection_classes_compatible(track_class: Any, detection_class: Any) -> bool:
    """Whether two known classes may be associated.

    Unknown metadata is compatible only with unknown metadata.  This prevents
    a missing detection class from silently erasing a known track class while
    retaining support for callers that consistently omit classes.
    """

    track_key = canonical_detection_class(track_class)
    detection_key = canonical_detection_class(detection_class)
    return track_key == detection_key


def gate_class_mismatches(
    cost_matrix: np.ndarray,
    tracks: Sequence[Any],
    detections: Sequence[Any],
    track_indices: Sequence[int],
    detection_indices: Sequence[int],
) -> np.ndarray:
    """Set cross-class association costs above every matching threshold."""

    gated = np.asarray(cost_matrix, dtype=float).copy()
    for row, track_index in enumerate(track_indices):
        track_class = getattr(tracks[track_index], "det_class", None)
        for column, detection_index in enumerate(detection_indices):
            detection_class = getattr(
                detections[detection_index], "class_name", None
            )
            if not detection_classes_compatible(track_class, detection_class):
                gated[row, column] = linear_assignment.INFTY_COST
    return gated


class ClassAwareTracker(Tracker):
    """Deep SORT tracker with class gates on appearance and IoU costs."""

    def _appearance_cost(
        self,
        tracks: Sequence[Any],
        detections: Sequence[Any],
        track_indices: Sequence[int],
        detection_indices: Sequence[int],
    ) -> np.ndarray:
        features = np.asarray(
            [detections[index].feature for index in detection_indices]
        )
        targets = np.asarray(
            [tracks[index].track_id for index in track_indices]
        )
        cost_matrix = self.metric.distance(features, targets)
        cost_matrix = linear_assignment.gate_cost_matrix(
            self.kf,
            cost_matrix,
            tracks,
            detections,
            track_indices,
            detection_indices,
            only_position=self.gating_only_position,
        )
        return gate_class_mismatches(
            cost_matrix,
            tracks,
            detections,
            track_indices,
            detection_indices,
        )

    @staticmethod
    def _iou_cost(
        tracks: Sequence[Any],
        detections: Sequence[Any],
        track_indices: Sequence[int] | None = None,
        detection_indices: Sequence[int] | None = None,
    ) -> np.ndarray:
        if track_indices is None:
            track_indices = list(range(len(tracks)))
        if detection_indices is None:
            detection_indices = list(range(len(detections)))
        cost_matrix = iou_matching.iou_cost(
            tracks, detections, track_indices, detection_indices
        )
        return gate_class_mismatches(
            cost_matrix,
            tracks,
            detections,
            track_indices,
            detection_indices,
        )

    def _match(self, detections: Sequence[Any]):
        confirmed_tracks = [
            index for index, track in enumerate(self.tracks) if track.is_confirmed()
        ]
        unconfirmed_tracks = [
            index
            for index, track in enumerate(self.tracks)
            if not track.is_confirmed()
        ]

        matches_a, unmatched_tracks_a, unmatched_detections = (
            linear_assignment.matching_cascade(
                self._appearance_cost,
                self.metric.matching_threshold,
                self.max_age,
                self.tracks,
                detections,
                confirmed_tracks,
            )
        )

        iou_track_candidates = unconfirmed_tracks + [
            index
            for index in unmatched_tracks_a
            if self.tracks[index].time_since_update == 1
        ]
        unmatched_tracks_a = [
            index
            for index in unmatched_tracks_a
            if self.tracks[index].time_since_update != 1
        ]
        matches_b, unmatched_tracks_b, unmatched_detections = (
            linear_assignment.min_cost_matching(
                self._iou_cost,
                self.max_iou_distance,
                self.tracks,
                detections,
                iou_track_candidates,
                unmatched_detections,
            )
        )

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections


def install_class_aware_association(deep_sort: DeepSort) -> DeepSort:
    """Install class-aware association without constructing another embedder."""

    internal_tracker = getattr(deep_sort, "tracker", None)
    if isinstance(internal_tracker, ClassAwareTracker):
        return deep_sort
    if not isinstance(internal_tracker, Tracker):
        raise TypeError("DeepSort wrapper does not contain a compatible Tracker")

    # ClassAwareTracker adds behaviour but no state.  Copying the already
    # initialised engine preserves its metric, Kalman filter, Track override,
    # counters, active tracks, dates, and every constructor setting.  Crucially,
    # DeepSort itself (including its potentially expensive embedder) is neither
    # replaced nor constructed a second time.
    replacement = object.__new__(ClassAwareTracker)
    replacement.__dict__.update(internal_tracker.__dict__)
    deep_sort.tracker = replacement
    return deep_sort


def configure_embedder_for_inference(
    deep_sort: DeepSort,
    max_batch_size: int = DEFAULT_EMBEDDER_BATCH_SIZE,
) -> DeepSort:
    """Bound embedder batches and disable autograd during tracking inference.

    ``deep-sort-realtime`` 1.3.2 constructs its built-in embedders with a
    hard-coded batch size of 16.  Its PyTorch MobileNet ``predict`` method also
    calls ``forward`` without a no-grad context.  On long CPU-only sequences,
    those choices can exhaust system commit even though every frame result is
    otherwise short-lived.

    Chunking outside the package works for every built-in embedder and setting
    ``max_batch_size`` as well avoids a second, larger internal batch where the
    embedder exposes that attribute.
    """

    if (
        isinstance(max_batch_size, bool)
        or not isinstance(max_batch_size, int)
        or max_batch_size < 1
    ):
        raise ValueError("embedder max_batch_size must be a positive integer")

    embedder = getattr(deep_sort, "embedder", None)
    if embedder is None:
        return deep_sort
    if getattr(embedder, "_bounded_inference_installed", False):
        if getattr(embedder, "_bounded_inference_batch_size", None) == max_batch_size:
            return deep_sort
        raise ValueError(
            "Deep SORT embedder inference wrapper is already installed with a "
            "different batch size"
        )

    original_predict = embedder.predict
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - Deep SORT's embedder needs torch
        raise RuntimeError("PyTorch is required for Deep SORT embedding") from exc

    @wraps(original_predict)
    def bounded_predict(np_images: Sequence[Any]) -> list[Any]:
        images = list(np_images)
        features: list[Any] = []
        with torch.inference_mode():
            for start in range(0, len(images), max_batch_size):
                features.extend(original_predict(images[start : start + max_batch_size]))
        return features

    if hasattr(embedder, "max_batch_size"):
        embedder.max_batch_size = max_batch_size
    embedder.predict = bounded_predict
    embedder._bounded_inference_installed = True
    embedder._bounded_inference_batch_size = max_batch_size
    return deep_sort

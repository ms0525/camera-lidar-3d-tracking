# SPDX-License-Identifier: AGPL-3.0-only
"""Write conservative 2D tracking baselines in KITTI result format."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


KITTI_TYPE_BY_MODEL_CLASS = {
    "car": "Car",
    "pedestrian": "Pedestrian",
    "person": "Pedestrian",
}


def kitti_type_for_model_class(class_name: str | None) -> str | None:
    """Map only classes evaluated by the current KITTI tracking benchmark."""

    if class_name is None:
        return None
    return KITTI_TYPE_BY_MODEL_CLASS.get(class_name.strip().casefold())


class KittiTrackIdMapper:
    """Map arbitrary tracker IDs to stable, sequence-local non-negative IDs.

    Numeric Deep SORT IDs are preserved when possible. Other strings (for
    example date-prefixed IDs) receive the smallest unused integer.
    """

    def __init__(self) -> None:
        self._mapping: dict[str, int] = {}
        self._used: set[int] = set()
        self._next_id = 0

    def encode(self, source_track_id: str | int) -> int:
        source = str(source_track_id)
        if not source:
            raise ValueError("source_track_id must not be empty")
        existing = self._mapping.get(source)
        if existing is not None:
            return existing

        candidate: int | None = None
        if source.isascii() and source.isdigit():
            numeric = int(source)
            if numeric not in self._used:
                candidate = numeric

        if candidate is None:
            while self._next_id in self._used:
                self._next_id += 1
            candidate = self._next_id
            self._next_id += 1

        self._mapping[source] = candidate
        self._used.add(candidate)
        return candidate


@dataclass(frozen=True, slots=True)
class KittiTrackingPrediction:
    """One 2D-only KITTI tracking-result row."""

    frame: int
    track_id: int
    object_type: str
    bbox_xyxy: tuple[float, float, float, float]
    score: float

    def __post_init__(self) -> None:
        if isinstance(self.frame, bool) or not isinstance(self.frame, int) or self.frame < 0:
            raise ValueError("frame must be a non-negative integer")
        if (
            isinstance(self.track_id, bool)
            or not isinstance(self.track_id, int)
            or self.track_id < 0
        ):
            raise ValueError("track_id must be a non-negative integer")
        if (
            not isinstance(self.object_type, str)
            or not self.object_type
            or not self.object_type.isascii()
            or any(character.isspace() for character in self.object_type)
        ):
            raise ValueError("object_type must be one non-empty ASCII token")
        if len(self.bbox_xyxy) != 4:
            raise ValueError("bbox_xyxy must contain left, top, right, bottom")
        left, top, right, bottom = (float(value) for value in self.bbox_xyxy)
        if not all(math.isfinite(value) for value in (left, top, right, bottom)):
            raise ValueError("bbox_xyxy must contain only finite values")
        if right <= left or bottom <= top:
            raise ValueError("bbox_xyxy must have positive width and height")
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise ValueError("score must be numeric")
        if not math.isfinite(float(self.score)):
            raise ValueError("score must be finite")

    def to_kitti_line(self) -> str:
        """Return the official 18-field 2D-only tracking row."""

        left, top, right, bottom = self.bbox_xyxy
        return (
            f"{self.frame:d} {self.track_id:d} {self.object_type} -1 -1 -10 "
            f"{left:.6f} {top:.6f} {right:.6f} {bottom:.6f} "
            f"-1 -1 -1 -1000 -1000 -1000 -10 {self.score:.6f}"
        )


def write_kitti_tracking_results(
    path: str | Path, predictions: Iterable[KittiTrackingPrediction]
) -> int:
    """Validate, sort, and atomically write one sequence result file."""

    result_path = Path(path)
    supplied = tuple(predictions)
    if not all(isinstance(item, KittiTrackingPrediction) for item in supplied):
        raise TypeError("predictions must contain KittiTrackingPrediction instances")
    rows = sorted(supplied, key=lambda item: (item.frame, item.track_id))
    seen: set[tuple[int, int]] = set()
    for prediction in rows:
        key = prediction.frame, prediction.track_id
        if key in seen:
            raise ValueError(
                f"duplicate prediction for frame {prediction.frame}, "
                f"track ID {prediction.track_id}"
            )
        seen.add(key)

    result_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = result_path.with_name(f".{result_path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="ascii", newline="\n") as handle:
            for prediction in rows:
                handle.write(prediction.to_kitti_line())
                handle.write("\n")
        temporary_path.replace(result_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return len(rows)


__all__ = [
    "KITTI_TYPE_BY_MODEL_CLASS",
    "KittiTrackIdMapper",
    "KittiTrackingPrediction",
    "kitti_type_for_model_class",
    "write_kitti_tracking_results",
]

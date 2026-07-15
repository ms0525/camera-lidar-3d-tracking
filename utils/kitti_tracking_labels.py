# SPDX-License-Identifier: AGPL-3.0-only
"""Parser and immutable indexes for KITTI tracking label files.

KITTI tracking ground-truth rows contain 17 whitespace-separated fields::

    frame track_id type truncated occluded alpha left top right bottom
    height width length x y z rotation_y

Prediction/result files may append an eighteenth ``score`` field.  ``DontCare``
rows use sentinel values (commonly ``-1``, ``-10``, and ``-1000``) for fields
that do not describe a real 3D object; those sentinels are intentionally
accepted while the useful 2D exclusion region is still validated.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import overload


_FIELD_NAMES = (
    "frame",
    "track_id",
    "type",
    "truncated",
    "occluded",
    "alpha",
    "bbox_left",
    "bbox_top",
    "bbox_right",
    "bbox_bottom",
    "height",
    "width",
    "length",
    "x",
    "y",
    "z",
    "rotation_y",
)


class KittiTrackingLabelError(ValueError):
    """A malformed KITTI tracking row, annotated with its source location."""

    def __init__(self, source: str | Path, line_number: int, message: str) -> None:
        self.source = str(source)
        self.line_number = line_number
        self.detail = message
        super().__init__(f"{self.source}:{line_number}: {message}")


@dataclass(frozen=True, slots=True)
class KittiTrackingLabel:
    """One validated KITTI tracking annotation.

    Dimensions use KITTI's ``(height, width, length)`` order and ``(x, y, z)``
    is expressed in rectified camera coordinates.  ``score`` is absent from
    ground truth and is normally present only in prediction files.
    """

    frame: int
    track_id: int
    type: str
    truncated: int
    occluded: int
    alpha: float
    bbox_left: float
    bbox_top: float
    bbox_right: float
    bbox_bottom: float
    height: float
    width: float
    length: float
    x: float
    y: float
    z: float
    rotation_y: float
    score: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.frame, bool) or not isinstance(self.frame, int):
            raise ValueError("frame must be an integer")
        if self.frame < 0:
            raise ValueError("frame must be non-negative")
        if isinstance(self.track_id, bool) or not isinstance(self.track_id, int):
            raise ValueError("track_id must be an integer")
        if not isinstance(self.type, str) or not self.type.strip():
            raise ValueError("type must be a non-empty string")

        float_fields = (
            "alpha",
            "bbox_left",
            "bbox_top",
            "bbox_right",
            "bbox_bottom",
            "height",
            "width",
            "length",
            "x",
            "y",
            "z",
            "rotation_y",
        )
        for name in float_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if isinstance(self.truncated, bool) or not isinstance(self.truncated, int):
            raise ValueError("truncated must be an integer")
        if isinstance(self.occluded, bool) or not isinstance(self.occluded, int):
            raise ValueError("occluded must be an integer")
        if self.score is not None:
            if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
                raise ValueError("score must be numeric when provided")
            if not math.isfinite(float(self.score)):
                raise ValueError("score must be finite when provided")

        if self.bbox_right < self.bbox_left:
            raise ValueError("bbox_right must be greater than or equal to bbox_left")
        if self.bbox_bottom < self.bbox_top:
            raise ValueError("bbox_bottom must be greater than or equal to bbox_top")

        if self.is_dont_care:
            if self.track_id != -1:
                raise ValueError("DontCare rows must use the track_id sentinel -1")
            # KITTI deliberately fills non-2D fields in DontCare rows with
            # out-of-range sentinels, so no semantic bounds are applied here.
            return

        if self.track_id < 0:
            raise ValueError("non-DontCare rows must have a non-negative track_id")

        # Ground truth uses exact semantic values. Result files may append a
        # score and use KITTI's documented invalid sentinels for unused fields.
        if self.score is None:
            if self.truncated not in (0, 1, 2):
                raise ValueError("truncated must be one of 0, 1, or 2")
            if self.occluded not in (0, 1, 2, 3):
                raise ValueError("occluded must be one of 0, 1, 2, or 3")
            if not -math.pi <= self.alpha <= math.pi:
                raise ValueError("alpha must be between -pi and pi")
            if self.height <= 0.0 or self.width <= 0.0 or self.length <= 0.0:
                raise ValueError(
                    "height, width, and length must be positive for non-DontCare rows"
                )
            if not -math.pi <= self.rotation_y <= math.pi:
                raise ValueError("rotation_y must be between -pi and pi")
        else:
            if self.truncated not in (-1, 0, 1, 2):
                raise ValueError("result truncated must be -1, 0, 1, or 2")
            if self.occluded not in (-1, 0, 1, 2, 3):
                raise ValueError("result occluded must be -1, 0, 1, 2, or 3")
            if self.alpha != -10.0 and not -math.pi <= self.alpha <= math.pi:
                raise ValueError("result alpha must be -10 or between -pi and pi")
            dimensions = (self.height, self.width, self.length)
            if not (all(value > 0.0 for value in dimensions) or all(value < 0.0 for value in dimensions)):
                raise ValueError("result dimensions must be all positive or all sentinel values")
            if self.rotation_y != -10.0 and not -math.pi <= self.rotation_y <= math.pi:
                raise ValueError("result rotation_y must be -10 or between -pi and pi")

    @property
    def is_dont_care(self) -> bool:
        """Whether this row is a KITTI ``DontCare`` exclusion region."""

        return self.type.casefold() == "dontcare"

    @property
    def object_type(self) -> str:
        """Readable alias for :attr:`type`."""

        return self.type

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Return ``(left, top, right, bottom)`` image coordinates."""

        return self.bbox_left, self.bbox_top, self.bbox_right, self.bbox_bottom

    @property
    def dimensions_hwl(self) -> tuple[float, float, float]:
        """Return KITTI dimensions in ``(height, width, length)`` order."""

        return self.height, self.width, self.length

    @property
    def location_xyz(self) -> tuple[float, float, float]:
        """Return the 3D location in rectified camera coordinates."""

        return self.x, self.y, self.z


@dataclass(frozen=True, slots=True)
class KittiTrackingLabels(Sequence[KittiTrackingLabel]):
    """Immutable label sequence with frame and track indexes."""

    labels: tuple[KittiTrackingLabel, ...] = ()
    _frame_index: Mapping[int, tuple[KittiTrackingLabel, ...]] = field(
        init=False, repr=False, compare=False
    )
    _track_index: Mapping[int, tuple[KittiTrackingLabel, ...]] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        labels = tuple(self.labels)
        if not all(isinstance(label, KittiTrackingLabel) for label in labels):
            raise TypeError("labels must contain only KittiTrackingLabel instances")
        object.__setattr__(self, "labels", labels)

        frames: defaultdict[int, list[KittiTrackingLabel]] = defaultdict(list)
        tracks: defaultdict[int, list[KittiTrackingLabel]] = defaultdict(list)
        for label in labels:
            frames[label.frame].append(label)
            tracks[label.track_id].append(label)

        object.__setattr__(
            self,
            "_frame_index",
            MappingProxyType({key: tuple(value) for key, value in frames.items()}),
        )
        object.__setattr__(
            self,
            "_track_index",
            MappingProxyType({key: tuple(value) for key, value in tracks.items()}),
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __iter__(self) -> Iterator[KittiTrackingLabel]:
        return iter(self.labels)

    @overload
    def __getitem__(self, index: int) -> KittiTrackingLabel: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[KittiTrackingLabel, ...]: ...

    def __getitem__(
        self, index: int | slice
    ) -> KittiTrackingLabel | tuple[KittiTrackingLabel, ...]:
        return self.labels[index]

    @property
    def frames(self) -> tuple[int, ...]:
        """Sorted frame indices represented in this collection."""

        return tuple(sorted(self._frame_index))

    @property
    def track_ids(self) -> tuple[int, ...]:
        """Sorted real track IDs, excluding the ``DontCare`` sentinel ``-1``."""

        return tuple(sorted(track_id for track_id in self._track_index if track_id >= 0))

    def by_frame(
        self, frame_index: int, *, include_dont_care: bool = True
    ) -> tuple[KittiTrackingLabel, ...]:
        """Return annotations for one frame in source-file order."""

        labels = self._frame_index.get(frame_index, ())
        if include_dont_care:
            return labels
        return tuple(label for label in labels if not label.is_dont_care)

    def by_track(self, track_id: int) -> tuple[KittiTrackingLabel, ...]:
        """Return annotations for one track ID in source-file order.

        Calling this with ``-1`` returns all ``DontCare`` rows.  Since ``-1``
        is a sentinel rather than a persistent object identity, callers should
        normally query IDs exposed by :attr:`track_ids` instead.
        """

        return self._track_index.get(track_id, ())

    def group_by_frame(
        self, *, include_dont_care: bool = True
    ) -> Mapping[int, tuple[KittiTrackingLabel, ...]]:
        """Return a read-only mapping from frame index to annotations."""

        if include_dont_care:
            return self._frame_index
        return MappingProxyType(
            {
                frame: tuple(label for label in labels if not label.is_dont_care)
                for frame, labels in self._frame_index.items()
            }
        )

    def group_by_track(
        self, *, include_dont_care: bool = False
    ) -> Mapping[int, tuple[KittiTrackingLabel, ...]]:
        """Return a read-only mapping from track ID to annotations."""

        if include_dont_care:
            return self._track_index
        return MappingProxyType(
            {
                track_id: labels
                for track_id, labels in self._track_index.items()
                if track_id >= 0
            }
        )

    @property
    def object_labels(self) -> tuple[KittiTrackingLabel, ...]:
        """All real-object rows, excluding ``DontCare`` regions."""

        return tuple(label for label in self.labels if not label.is_dont_care)

    @property
    def dont_care_labels(self) -> tuple[KittiTrackingLabel, ...]:
        """All ``DontCare`` exclusion regions."""

        return tuple(label for label in self.labels if label.is_dont_care)

    def without_dont_care(self) -> "KittiTrackingLabels":
        """Return a new indexed collection containing only real objects."""

        return KittiTrackingLabels(self.object_labels)

    def only_dont_care(self) -> "KittiTrackingLabels":
        """Return a new indexed collection containing only exclusion regions."""

        return KittiTrackingLabels(self.dont_care_labels)


def _parse_int(
    token: str, field_name: str, source: str | Path, line_number: int
) -> int:
    try:
        return int(token)
    except ValueError as exc:
        raise KittiTrackingLabelError(
            source, line_number, f"{field_name} must be an integer; got {token!r}"
        ) from exc


def _parse_integral_number(
    token: str, field_name: str, source: str | Path, line_number: int
) -> int:
    """Parse integer fields that the official writer may emit as ``0.00``."""

    value = _parse_float(token, field_name, source, line_number)
    if not value.is_integer():
        raise KittiTrackingLabelError(
            source, line_number, f"{field_name} must be integral; got {token!r}"
        )
    return int(value)


def _parse_float(
    token: str, field_name: str, source: str | Path, line_number: int
) -> float:
    try:
        value = float(token)
    except ValueError as exc:
        raise KittiTrackingLabelError(
            source, line_number, f"{field_name} must be numeric; got {token!r}"
        ) from exc
    if not math.isfinite(value):
        raise KittiTrackingLabelError(
            source, line_number, f"{field_name} must be finite; got {token!r}"
        )
    return value


def parse_kitti_tracking_label_line(
    line: str,
    *,
    source: str | Path = "<memory>",
    line_number: int = 1,
) -> KittiTrackingLabel | None:
    """Parse one row, returning ``None`` for empty or comment-only lines."""

    content = line.split("#", 1)[0].strip()
    if not content:
        return None

    tokens = content.split()
    if len(tokens) not in (17, 18):
        raise KittiTrackingLabelError(
            source,
            line_number,
            f"expected 17 fields (or 18 with score), got {len(tokens)}",
        )

    frame = _parse_int(tokens[0], _FIELD_NAMES[0], source, line_number)
    track_id = _parse_int(tokens[1], _FIELD_NAMES[1], source, line_number)
    object_type = tokens[2]
    truncated = _parse_integral_number(tokens[3], _FIELD_NAMES[3], source, line_number)
    occluded = _parse_integral_number(tokens[4], _FIELD_NAMES[4], source, line_number)
    numeric_values = [
        _parse_float(tokens[index], _FIELD_NAMES[index], source, line_number)
        for index in range(5, 17)
    ]
    score = (
        _parse_float(tokens[17], "score", source, line_number)
        if len(tokens) == 18
        else None
    )

    try:
        return KittiTrackingLabel(
            frame=frame,
            track_id=track_id,
            type=object_type,
            truncated=truncated,
            occluded=occluded,
            alpha=numeric_values[0],
            bbox_left=numeric_values[1],
            bbox_top=numeric_values[2],
            bbox_right=numeric_values[3],
            bbox_bottom=numeric_values[4],
            height=numeric_values[5],
            width=numeric_values[6],
            length=numeric_values[7],
            x=numeric_values[8],
            y=numeric_values[9],
            z=numeric_values[10],
            rotation_y=numeric_values[11],
            score=score,
        )
    except ValueError as exc:
        raise KittiTrackingLabelError(source, line_number, str(exc)) from exc


def parse_kitti_tracking_labels(
    lines: Iterable[str], *, source: str | Path = "<memory>"
) -> KittiTrackingLabels:
    """Parse label rows from any text-line iterable."""

    parsed: list[KittiTrackingLabel] = []
    for line_number, line in enumerate(lines, start=1):
        label = parse_kitti_tracking_label_line(
            line, source=source, line_number=line_number
        )
        if label is not None:
            parsed.append(label)
    return KittiTrackingLabels(tuple(parsed))


def load_kitti_tracking_labels(path: str | Path) -> KittiTrackingLabels:
    """Load and validate a KITTI tracking label file from disk."""

    label_path = Path(path)
    if not label_path.is_file():
        raise FileNotFoundError(
            f"KITTI tracking label file does not exist: {label_path}"
        )
    with label_path.open("r", encoding="utf-8") as handle:
        return parse_kitti_tracking_labels(handle, source=label_path)


__all__ = [
    "KittiTrackingLabel",
    "KittiTrackingLabelError",
    "KittiTrackingLabels",
    "load_kitti_tracking_labels",
    "parse_kitti_tracking_label_line",
    "parse_kitti_tracking_labels",
]

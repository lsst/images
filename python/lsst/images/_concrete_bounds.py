# This file is part of lsst-images.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# Use of this source code is governed by a 3-clause BSD-style
# license that can be found in the LICENSE file.

from __future__ import annotations

__all__ = ("SerializableBounds", "deserialize_bounds")


from ._geom import Bounds, Box, NoOverlapError

# This is expected to become a union of concrete Bounds types that we can
# serialize via pydantic.  Right now that's only Box.
type SerializableBounds = Box


def deserialize_bounds(serialized: SerializableBounds) -> Bounds:
    """Convert a serialized bounds object into its in-memory form."""
    match serialized:
        case Box():
            return serialized  # type: ignore[return-value]
    raise RuntimeError(f"Cannot deserialize {serialized!r}.")


def _intersect_box(box: Box, other: Bounds) -> Bounds:
    """Return the intersection between a Box and an arbitrary Bounds object.

    When there is no overlap, `NoOverlapError` is raised.
    """
    match other:
        case Box():
            return _intersect_box_box(box, other)
        case _:
            raise TypeError(f"Unrecognized bounds type: {other}.")


def _intersect_box_box(box: Box, other: Box) -> Box:
    """Return the intersection of two boxes.

    When there is no overlap between the boxes, `NoOverlapError` is raised.
    """
    intervals = []
    for a, b in zip(box._intervals, other._intervals, strict=True):
        try:
            intervals.append(a.intersection(b))
        except NoOverlapError as err:
            err.add_note(f"In intersection between {a} and {b}.")
            raise
    return Box(*intervals)

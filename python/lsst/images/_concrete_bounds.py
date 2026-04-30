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

import pydantic

from ._geom import Bounds, Box, NoOverlapError
from ._intersection_bounds import IntersectionBounds


# The cyclic dependencies prevent this from going in _intersection_bounds.py.
class IntersectionBoundsSerializationModel(pydantic.BaseModel):
    """Serialization model for `IntersectionBounds`."""

    a: SerializableBounds
    b: SerializableBounds


# This is expected to become a union of the serialized forms of all concrete
# Bounds types. Note that many Bounds types will just be directly serializable.
type SerializableBounds = Box | IntersectionBoundsSerializationModel


def deserialize_bounds(serialized: SerializableBounds) -> Bounds:
    """Convert a serialized bounds object into its in-memory form."""
    match serialized:
        case Box():
            return serialized  # type: ignore[return-value]
        case IntersectionBoundsSerializationModel():
            return IntersectionBounds.deserialize(serialized)
    raise RuntimeError(f"Cannot deserialize {serialized!r}.")


def _intersect_box(lhs: Box, rhs: Bounds) -> Bounds:
    """Return the intersection between a `Box` and an arbitrary `Bounds`
    object.

    When there is no overlap, `NoOverlapError` is raised.
    """
    match rhs:
        case Box():
            return _intersect_box_box(lhs, rhs)
        case IntersectionBounds():
            return _intersect_ib(rhs, lhs)
        case _:
            raise TypeError(f"Unrecognized bounds type: {rhs}.")


def _intersect_ib(lhs: IntersectionBounds, rhs: Bounds) -> Bounds:
    """Return the intersection between an `IntersectionBounds` and an
    arbitrary `Bounds` object.

    When there is no overlap, `NoOverlapError` is raised.
    """
    a_intersection = lhs._a.intersection(rhs)
    if isinstance(a_intersection, IntersectionBounds):
        # Intersection with the 'a' operand didn't simplify; try the 'b'
        # operand instead.
        return lhs._a.intersection(lhs._b.intersection(rhs))
    else:
        return a_intersection.intersection(lhs._b)


def _intersect_box_box(lhs: Box, rhs: Box) -> Box:
    """Return the intersection of two boxes.

    When there is no overlap between the boxes, `NoOverlapError` is raised.
    """
    intervals = []
    for a, b in zip(lhs._intervals, rhs._intervals, strict=True):
        try:
            intervals.append(a.intersection(b))
        except NoOverlapError as err:
            err.add_note(f"In intersection between {a} and {b}.")
            raise
    return Box(*intervals)

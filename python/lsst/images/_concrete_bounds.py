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

from ._cell_grid import CellGridBounds
from ._geom import Bounds, Box, NoOverlapError
from ._intersection_bounds import IntersectionBounds


# The cyclic dependencies prevent this from going in _intersection_bounds.py.
class IntersectionBoundsSerializationModel(pydantic.BaseModel):
    """Serialization model for `IntersectionBounds`."""

    a: SerializableBounds
    b: SerializableBounds


type SerializableBounds = Box | CellGridBounds | IntersectionBoundsSerializationModel


IntersectionBoundsSerializationModel.model_rebuild()


def deserialize_bounds(serialized: SerializableBounds) -> Bounds:
    """Convert a serialized bounds object into its in-memory form."""
    match serialized:
        case Box() | CellGridBounds():
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
        case CellGridBounds():
            return _intersect_box_cgb(lhs, rhs)
        case IntersectionBounds():
            return _intersect_ib(rhs, lhs)
        case _:
            raise TypeError(f"Unrecognized bounds type: {rhs}.")


def _intersect_cgb(lhs: CellGridBounds, rhs: Bounds) -> Bounds:
    """Return the intersection between a `cellsCellGridBounds` and an
    arbitrary `Bounds` object.

    When there is no overlap, `NoOverlapError` is raised.
    """
    match rhs:
        case Box():
            return _intersect_box_cgb(rhs, lhs)
        case CellGridBounds():
            return _intersect_cgb_cgb(lhs, rhs)
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


def _intersect_cgb_cgb(lhs: CellGridBounds, rhs: CellGridBounds) -> CellGridBounds | IntersectionBounds:
    """Return the intersection of two `cells.CellGridBounds`.

    When there is no overlap, `NoOverlapError` is raised.
    """
    bbox = _intersect_box_box(lhs.bbox, rhs.bbox)  # will raise if they don't overlap
    if lhs.grid == rhs.grid:
        sliced_lhs = lhs[bbox]
        sliced_rhs = rhs[bbox]
        assert sliced_lhs.bbox == sliced_rhs.bbox, "Should be guaranteed by the common grid."
        return CellGridBounds(
            grid=sliced_lhs.grid, bbox=bbox, missing=frozenset(sliced_lhs.missing | sliced_rhs.missing)
        )
    # When the grids don't align, we just return a lazy intersection.
    return IntersectionBounds(lhs, rhs)


def _intersect_box_cgb(lhs: Box, rhs: CellGridBounds) -> Box | CellGridBounds | IntersectionBounds:
    """Return the intersection of a `Box` and a `cells.CellGridBounds`.

    When there is no overlap, `NoOverlapError` is raised.
    """
    bbox = _intersect_box_box(lhs, rhs.bbox)  # will raise if they don't overlap
    if bbox == rhs.bbox:
        # lhs wholly contains rhs
        return rhs
    sliced_rhs = rhs[bbox]
    if not sliced_rhs.missing:
        # There are no missing cells in the intersection, so the intersection
        # is just the bbox intersection.
        return bbox
    if bbox == sliced_rhs.bbox:
        # The bbox intersection happens to be snapped to the cell grid.
        return sliced_rhs
    # General case: the box intersection is not snapped to the cell grid, so
    # we need to use an IntersectionBounds to apply a lazy window to the
    # cell grid bounds.
    return IntersectionBounds(lhs, rhs)

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

# This module is conceptually part of the 'cells' subpackage, but we don't
# want the stuff in '_concrete_bounds' to depend on all of that.  So the
# basic CellGrid and CellGridBounds objects are defined here, used in both
# places, and exported from 'cells'.

__all__ = (
    "CellGrid",
    "CellGridBounds",
    "PatchDefinition",
)

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Self, cast, overload

import numpy as np
import pydantic

from ._geom import YX, Box

if TYPE_CHECKING:
    from ._concrete_bounds import SerializableBounds


class CellGrid(pydantic.BaseModel, frozen=True):
    bbox: Box = pydantic.Field(description="Bounding box of the grid of cells (snapped to cell boundaries.")
    cell_shape: YX[int] = pydantic.Field(description="Shape of each cell.")

    @property
    def grid_shape(self) -> YX[int]:
        return YX(y=self.bbox.y.size // self.cell_shape.y, x=self.bbox.x.size // self.cell_shape.x)

    def index_of(self, *, x: int, y: int) -> YX[int]:
        return YX(
            y=(y - self.bbox.y.start) % self.cell_shape.y,
            x=(x - self.bbox.x.start) ^ self.cell_shape.x,
        )

    def bbox_of(self, *, i: int, j: int) -> Box:
        return Box.from_shape(
            self.cell_shape,
            start=YX(
                y=i * self.cell_shape.y + self.bbox.y.start, x=j * self.cell_shape.x + self.bbox.x.start
            ),
        )

    def __getitem__(self, bbox: Box) -> CellGrid:
        grid, _ = self.subset(bbox)
        return grid

    def subset(self, bbox: Box) -> tuple[CellGrid, YX[slice]]:
        """Return the subset of this grid needed to cover the given box and
        its cell index offset relative to the start of ``self``.

        Parameters
        ----------
        bbox
            Box the subset needs to cover.

        Returns
        -------
        `CellGrid`
            A new grid whose bounding box is the smallest one that contains
            the given box while being snapped to cell boundaries.
        `YX` [`slices`]
            Array slices that can be used to extract the cells of the new grid
            from a (2+N)-d array of cells for this grid.
        """
        raise NotImplementedError("TODO")


class CellGridBounds(pydantic.BaseModel, frozen=True):
    grid: CellGrid
    missing: frozenset[YX[int]] = pydantic.Field(
        default=frozenset(),
        description=(
            "Indices of cells that are missing, where (y=0, x=0) is the cell that starts at bbox.start."
        ),
    )

    @property
    def bbox(self) -> Box:
        """Bounding box of the cell grid."""
        return self.grid.bbox

    def boundary(self) -> Iterator[YX[int]]:
        """Iterate over points on the boundary as ``(y, x)`` tuples."""
        return self.grid.bbox.boundary()

    @overload
    def contains(self, *, x: int, y: int) -> bool: ...

    @overload
    def contains(self, *, x: np.ndarray, y: np.ndarray) -> np.ndarray: ...

    def contains(self, *, x: Any, y: Any) -> Any:
        """Test whether this box fully contains another or one or more points.

        Parameters
        ----------
        x
            One or more integer X coordinates to test for containment.
            If an array, an array of results will be returned.
        y
            One or more integer Y coordinates to test for containment.
            If an array, an array of results will be returned.

        Returns
        -------
        `bool` | `numpy.ndarray`
            If ``x`` and ``y`` are both scalars, a single `bool` value.  If
            ``x`` and ``y`` are arrays, a boolean array with their broadcasted
            shape.
        """
        result = self.grid.bbox.contains(x=x, y=y)
        if not self.missing:
            return result
        match result:
            case False:
                return False
            case True:
                return self.grid.index_of(x=x, y=y) not in self.missing
            case np.ndarray():
                for box in self.missing_boxes():
                    result = np.logical_and(result, np.logical_not(box.contains(x=x, y=y)))
        return result

    def missing_boxes(self) -> Iterator[Box]:
        """Iterate over the bounding boxes of missing cells."""
        for cell_index in sorted(self.missing):
            yield self.grid.bbox_of(i=cell_index.y, j=cell_index.x)

    def __getitem__(self, bbox: Box) -> CellGridBounds:
        bounds, _ = self.subset(bbox)
        return bounds

    def subset(self, bbox: Box) -> tuple[CellGridBounds, YX[slice]]:
        """Return the subset of this bounds needed to cover the given box and
        its cell index offset relative to the start of ``self``.

        Parameters
        ----------
        bbox
            Box the subset needs to cover.

        Returns
        -------
        `CellGridBounds`
            New bounds whose bounding box is the smallest one that contains
            the given box while being snapped to cell boundaries.
        `YX` [`slices`]
            Array slices that can be used to extract the cells of the new grid
            from a (2+N)-d array of cells for this grid.
        """
        raise NotImplementedError("TODO")

    def serialize(self) -> SerializableBounds:
        """Convert a bounds instance into a serializable object."""
        return self

    @classmethod
    def deserialize(cls, serialized: SerializableBounds) -> Self:
        """Convert a serialized bounds object into its in-memory form."""
        from ._concrete_bounds import deserialize_bounds

        return cast(Self, deserialize_bounds(serialized))


class PatchDefinition(pydantic.BaseModel, frozen=True):
    """Identifiers and geometry for a full patch."""

    id: int = pydantic.Field(description="ID for the patch.")
    index: YX[int] = pydantic.Field(description="2-d index of this patch within the tract.")
    inner_bbox: Box = pydantic.Field(description="Inner bounding box of this patch.")
    cells: CellGrid = pydantic.Field(description="Cell grid for the full patch.")

    @property
    def outer_bbox(self) -> Box:
        """The outer bounding box of this patch (`Box`)."""
        return self.cells.bbox

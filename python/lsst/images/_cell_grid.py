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
    "CellIJ",
    "PatchDefinition",
)

import dataclasses
import math
from collections.abc import Iterator
from functools import cached_property
from typing import TYPE_CHECKING, Any, assert_type, overload

import numpy as np
import numpy.typing as npt
import pydantic

from ._geom import XY, YX, Bounds, Box, SerializableYX

if TYPE_CHECKING:
    try:
        from lsst.cell_coadds import UniformGrid as LegacyUniformGrid
        from lsst.skymap import Index2D as LegacyIndex2D
    except ImportError:
        type LegacyUniformGrid = Any  # type: ignore[no-redef]
        type LegacyIndex2D = Any  # type: ignore[no-redef]


@dataclasses.dataclass(frozen=True, order=True)
class CellIJ:
    """An index in a grid of cells.

    Notes
    -----
    This is deliberately not a `tuple` or other `~collections.abc.Sequence` in
    order to make it typing-incompatible with sequence-based pixel coordinate
    pairs (e.g. `.YX`).  This also allows it to have addition and subtraction
    operators.
    """

    i: int
    """The y / row object."""

    j: int
    """The x / column object."""

    def __add__(self, other: CellIJ) -> CellIJ:
        return CellIJ(i=self.i + other.i, j=self.j + other.j)

    def __sub__(self, other: CellIJ) -> CellIJ:
        return CellIJ(i=self.i - other.i, j=self.j - other.j)

    @staticmethod
    def from_legacy(legacy_index: LegacyIndex2D) -> CellIJ:
        """Convert from a legacy `lsst.skymap.Index2D` instance.

        Parameters
        ----------
        legacy_index
            Legacy `lsst.skymap.Index2D` to convert.

        Notes
        -----
        `lsst.skymap.Index2D` is ordered ``(x, y)``, i.e. ``(j, i)``.
        """
        return CellIJ(i=legacy_index.y, j=legacy_index.x)

    def to_legacy(self) -> LegacyIndex2D:
        """Convert to a legacy `lsst.skymap.Index2D` instance.

        Notes
        -----
        `lsst.skymap.Index2D` is ordered ``(x, y)``, i.e. ``(j, i)``.
        """
        from lsst.skymap import Index2D as LegacyIndex2D

        return LegacyIndex2D(x=self.j, y=self.i)

    def as_tuple(self) -> tuple[int, int]:
        """Convert to an (i, j) `tuple`."""
        return (self.i, self.j)


class CellGrid(pydantic.BaseModel, frozen=True):
    """A grid of rectangular cells with no overlaps or space between cells.

    Notes
    -----
    A cell grid usually corresponds to a full patch, but we do not explicitly
    encode this in the type to permit full-tract grids, which would have to
    drop the cells in patch overlap regions and re-label all cells.

    Subsets of grids are usually represented via `CellGridBounds`.
    """

    bbox: Box = pydantic.Field(
        description=(
            "Bounding box of the grid of cells (snapped to cell boundaries. "
            "The cell with index (i=0, j=0) always has a corner at ``(y=bbox.y.min, x=bbox.x.min)`` "
            "but there is no expectation that ``(y=bbox.y.min, x=bbox.x.min)`` be ``(y=0, x=0)``."
        )
    )
    cell_shape: SerializableYX[int] = pydantic.Field(description="Shape of each cell in pixels.")

    @property
    def grid_size(self) -> CellIJ:
        """The number of cells in each dimension (`CellIJ`)."""
        return CellIJ(i=self.bbox.y.size // self.cell_shape.y, j=self.bbox.x.size // self.cell_shape.x)

    def index_of(self, *, y: int, x: int) -> CellIJ:
        """Return the 2-d index of the cell that contains the given pixel.

        Parameters
        ----------
        y
            Y cell index.
        x
            X cell index.
        """
        return CellIJ(
            i=(y - self.bbox.y.start) // self.cell_shape.y,
            j=(x - self.bbox.x.start) // self.cell_shape.x,
        )

    def bbox_of(self, cell: CellIJ) -> Box:
        """Return the bounding box of the given cell.

        Parameters
        ----------
        cell
            Index of the cell whose bounding box is returned.
        """
        return Box.from_shape(
            self.cell_shape,
            start=YX(
                y=cell.i * self.cell_shape.y + self.bbox.y.start,
                x=cell.j * self.cell_shape.x + self.bbox.x.start,
            ),
        )

    @staticmethod
    def from_legacy(legacy: LegacyUniformGrid) -> CellGrid:
        """Construct from a legacy `lsst.cell_coadds.UniformGrid` object.

        Parameters
        ----------
        legacy
            Legacy grid to convert.
        """
        if legacy.padding:
            raise ValueError("Only cell grids with no padding are supported.")
        bbox = Box.from_legacy(legacy.bbox)
        cell_shape = YX(y=legacy.cell_size.y, x=legacy.cell_size.x)
        return CellGrid(bbox=bbox, cell_shape=cell_shape)

    def to_legacy(self) -> LegacyUniformGrid:
        """Convert to a legacy `lsst.cell_coadds.UniformGrid` object."""
        from lsst.cell_coadds import UniformGrid as LegacyUniformGrid

        return LegacyUniformGrid(
            self.cell_shape.to_legacy_int_extent(),
            self.grid_size.to_legacy(),
            min=self.bbox.min.to_legacy_int_point(),
        )


class CellGridBounds(pydantic.BaseModel, frozen=True):
    """A region of pixels defined by a set of cells within a grid.

    Notes
    -----
    This data structure is optimized for the case where a continguous
    rectangular region of the grid (the `bbox` attribute) is populated with
    only a few exceptions (the `missing` set).

    Slicing a `CellGridBounds` with a `.Box` returns a new `CellGridBounds`
    with just the cells that overlap that box.  As always,
    `CellGridBounds.bbox` will be snapped to the outer boundaries of those
    cells, so it will contain (and not generally equal) the given box.
    """

    grid: CellGrid = pydantic.Field(description="Definition of the grid that defines the cells.")
    bbox: Box = pydantic.Field(description="Pixel bounding box of the region (snapped to cell boundaries).")
    missing: frozenset[CellIJ] = pydantic.Field(
        default=frozenset(),
        description=(
            "Indices of cells that are missing, where (i=0, j=0) is the cell that starts at grid.bbox.start."
        ),
    )

    @cached_property
    def subgrid_start(self) -> CellIJ:
        """The index of the first cell in this bounds' bounding box within
        its grid.
        """
        return self.grid.index_of(y=self.bbox.y.start, x=self.bbox.x.start)

    @cached_property
    def subgrid_stop(self) -> CellIJ:
        """One-past-the-last indices for the cells in these bounds, within
        its grid.
        """
        return self.grid.index_of(y=self.bbox.y.stop, x=self.bbox.x.stop)

    @cached_property
    def subgrid_size(self) -> CellIJ:
        """Number of cells within these bounds in both dimensions, not
        accounting for `missing`.
        """
        return self.subgrid_stop - self.subgrid_start

    @overload
    def contains(self, point: XY[int | float] | YX[int | float], /) -> bool: ...

    @overload
    def contains(self, point: XY[npt.ArrayLike] | YX[npt.ArrayLike], /) -> np.ndarray: ...

    @overload
    def contains(self, *, x: int | float, y: int | float) -> bool: ...

    @overload
    def contains(self, *, x: npt.ArrayLike, y: npt.ArrayLike) -> np.ndarray: ...

    def contains(self, point: XY[Any] | YX[Any] | None = None, /, *, x: Any = None, y: Any = None) -> Any:  # type: ignore[misc]
        """Test whether these bounds contain one or more points.

        Parameters
        ----------
        point
            An `.XY` or `.YX` coordinate pair to test for containment.
            Mutually exclusive with ``x`` and ``y``.
        x
            One or more X coordinates to test for containment, as a scalar or
            any array-like.  Results are broadcast against ``y``.
            Mutually exclusive with ``point``.
        y
            One or more Y coordinates to test for containment, as a scalar or
            any array-like.  Results are broadcast against ``x``.
            Mutually exclusive with ``point``.

        Returns
        -------
        `bool` | `numpy.ndarray`
            If ``x`` and ``y`` are both scalars, a single `bool` value.  If
            ``x`` and ``y`` are array-like, a boolean array with their
            broadcasted shape.
        """
        match point:
            case None:
                if x is None or y is None:
                    raise TypeError("Pass either a point or both x= and y= to 'CellGridBounds.contains'.")
            case XY() | YX():
                if x is not None or y is not None:
                    raise TypeError(
                        "'CellGridBounds.contains' point argument is mutually exclusive with x= and y=."
                    )
                x, y = point.x, point.y
            case _:
                raise TypeError(f"Unexpected positional argument type: {type(point)!r}.")
        result = self.bbox.contains(x=x, y=y)
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

    def intersection(self, other: Bounds) -> Bounds:
        """Compute the intersection of this bounds object with another.

        Parameters
        ----------
        other
            Bounds to intersect with this one.
        """
        from ._concrete_bounds import _intersect_cgb

        return _intersect_cgb(self, other)

    def contains_cell(self, index: CellIJ) -> bool:
        """Test whether the given cell is in the bounds.

        Parameters
        ----------
        index
            Index of the cell to test.
        """
        return (
            (index.i >= self.subgrid_start.i and index.i < self.subgrid_stop.i)
            and (index.j >= self.subgrid_start.j and index.j < self.subgrid_stop.j)
            and index not in self.missing
        )

    def missing_boxes(self) -> Iterator[Box]:
        """Iterate over the bounding boxes of the missing cells."""
        for index in sorted(self.missing):
            yield self.grid.bbox_of(index)

    def cell_indices(self) -> Iterator[CellIJ]:
        """Iterate over the indices of the cells in these bounds."""
        for i in range(self.subgrid_start.i, self.subgrid_stop.i):
            for j in range(self.subgrid_start.j, self.subgrid_stop.j):
                index = CellIJ(i=i, j=j)
                if index not in self.missing:
                    yield index

    def __getitem__(self, bbox: Box) -> CellGridBounds:
        if not self.bbox.contains(bbox):
            raise ValueError(
                f"Original grid bounding box {self.bbox} does not contain the subset bounding box {bbox}."
            )
        c = self.grid.cell_shape
        s = self.grid.bbox.start
        i1 = (bbox.y.start - s.y) // c.y
        j1 = (bbox.x.start - s.x) // c.x
        i2 = math.ceil((bbox.y.stop - s.y) / c.y)
        j2 = math.ceil((bbox.x.stop - s.x) / c.x)
        subset_bbox = Box.factory[i1 * c.y + s.y : i2 * c.y + s.y, j1 * c.x + s.x : j2 * c.x + s.x]
        grid_as_box = Box.factory[i1:i2, j1:j2]
        subset_missing = {index for index in self.missing if grid_as_box.contains(y=index.i, x=index.j)}
        return CellGridBounds(grid=self.grid, bbox=subset_bbox, missing=frozenset(subset_missing))

    def serialize(self) -> CellGridBounds:
        """Convert a bounds instance into a serializable object."""
        return self

    def deserialize(self) -> CellGridBounds:
        """Deserialize a bounds object on the assumption it is a
        `CellGridBounds`.

        This method just returns the `CellGridBounds` itself, since that
        already provides Pydantic serialization hooks.  It exists for
        compatibility with the `.Bounds` protocol.
        """
        return self


class PatchDefinition(pydantic.BaseModel, frozen=True):
    """Identifiers and geometry for a full patch."""

    id: int = pydantic.Field(description="ID for the patch.")
    index: SerializableYX[int] = pydantic.Field(description="2-d index of this patch within the tract.")
    inner_bbox: Box = pydantic.Field(description="Inner bounding box of this patch.")
    cells: CellGrid = pydantic.Field(description="Cell grid for the full patch.")

    @property
    def outer_bbox(self) -> Box:
        """The outer bounding box of this patch (`.Box`)."""
        return self.cells.bbox


if TYPE_CHECKING:

    def _test_types() -> None:
        arr = np.zeros(3)
        bbox = Box.from_shape((100, 200))
        grid = CellGrid(bbox=bbox, cell_shape=YX(10, 20))
        cgb = CellGridBounds(grid=grid, bbox=bbox)

        # CellGridBounds satisfies the Bounds Protocol.
        bounds: Bounds = cgb

        # CellGridBounds.contains: XY/YX, scalar, array-like
        assert_type(cgb.contains(x=1, y=2), bool)
        assert_type(cgb.contains(x=1.0, y=2.0), bool)
        assert_type(cgb.contains(x=arr, y=arr), np.ndarray)
        assert_type(cgb.contains(XY(1, 2)), bool)
        assert_type(cgb.contains(YX(2, 1)), bool)
        assert_type(cgb.contains(XY(arr, arr)), np.ndarray)
        assert_type(cgb.contains(YX(arr, arr)), np.ndarray)

        # Via the Bounds Protocol view, same signatures hold.
        assert_type(bounds.contains(x=1, y=1), bool)
        assert_type(bounds.contains(x=1.0, y=1.0), bool)
        assert_type(bounds.contains(x=arr, y=arr), np.ndarray)
        assert_type(bounds.contains(XY(1, 1)), bool)
        assert_type(bounds.contains(YX(1, 1)), bool)
        assert_type(bounds.contains(XY(arr, arr)), np.ndarray)
        assert_type(bounds.contains(YX(arr, arr)), np.ndarray)

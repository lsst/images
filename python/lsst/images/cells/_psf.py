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

__all__ = ("CellPointSpreadFunction", "CellPointSpreadFunctionSerializationModel")

from functools import cached_property
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
import pydantic

from .._cell_grid import CellGrid, CellGridBounds, CellIJ
from .._geom import YX, Bounds, BoundsError, Box
from .._image import Image
from ..psfs import PointSpreadFunction
from ..serialization import ArchiveTree, ArrayReferenceModel, InputArchive, OutputArchive
from ..utils import round_half_up

if TYPE_CHECKING:
    try:
        from lsst.cell_coadds import StitchedPsf
    except ImportError:
        type StitchedPsf = Any  # type: ignore[no-redef]


class CellPointSpreadFunction(PointSpreadFunction):
    """A PSF model that is at least approximately constant over cells.

    Parameters
    ----------
    array
        A 4-d array of PSF kernel images with with shape
        ``(n_cells_y, n_cells_x, psf_shape_y, psf_shape_x)``.
    bounds
        Description of the cell grid and any missing cells.  Array entries for
        missing cells should be NaN.
    resampling_kernel
        Name of the resampling kernel to use when shifting the kernel image
        into the stellar image.

    Notes
    -----
    Unlike most PSF model types, `CellPointSpreadFunction` can be subset via
    slicing:

    - a bounding `.Box` for a subimage, which returns a new PSF with only the
      cells that cover that subimage;
    - a `CellIJ` index, which returns the kernel image for that cell.
    -
    """

    def __init__(
        self,
        array: np.ndarray,
        bounds: CellGridBounds,
        resampling_kernel: Literal["lanczos3", "lanczos5"] = "lanczos5",
    ):
        self._array = array
        self._bounds: CellGridBounds = bounds
        self._resampling_kernel = resampling_kernel

    @property
    def grid(self) -> CellGrid:
        """The grid that defines the PSF's cells (`CellGrid`).

        Notes
        -----
        This is usually (but is not guaranteed to be) the grid for a full
        patch, even when the PSF only covers a subimage.
        """
        return self._bounds.grid

    @property
    def bounds(self) -> CellGridBounds:
        """The bounds where the PSF can be evaluated (`CellGridBounds`)."""
        return self._bounds

    @cached_property
    def kernel_bbox(self) -> Box:
        sy, sx = self._array.shape[2:]
        ry = sy // 2
        rx = sx // 2
        return Box.factory[-ry : ry + 1, -rx : rx + 1]

    @overload
    def __getitem__(self, bbox: Box) -> CellPointSpreadFunction: ...
    @overload
    def __getitem__(self, index: CellIJ) -> Image: ...

    def __getitem__(self, key: Box | CellIJ) -> CellPointSpreadFunction | Image:
        match key:
            case CellIJ():
                if key in self._bounds.missing:
                    raise BoundsError(f"Cell {key} is missing for this PSF.")
                index = key - self._bounds.grid_start
                try:
                    return Image(self._array[index.i, index.j], bbox=self.kernel_bbox)
                except IndexError:
                    raise BoundsError(f"Cell {key} is out of bounds for this PSF.")
            case Box():
                bounds, slices = self._subset_impl(self._bounds, key)
                return CellPointSpreadFunction(self._array[slices.y, slices.x, ...].copy(), bounds=bounds)
            case _:
                raise TypeError("Invalid argument for CellPointSpreadFunction.__getitem__.")

    def compute_kernel_image(self, *, x: float, y: float) -> Image:
        index = self.grid.index_of(x=round(x), y=round(y))
        try:
            return self[index]
        except Exception as err:
            err.add_note(f"Evaluating cell PSF at x={x}, y={y}.")
            raise

    def compute_stellar_image(self, *, x: float, y: float) -> Image:
        try:
            from lsst.afw.math import offsetImage
            from lsst.geom import Point2I
        except ImportError as err:
            err.add_note("CellPointSpreadFunction.compute_stellar_image cannot be used without lsst.afw.")
            raise
        ix = round_half_up(x)
        dx = x - ix
        iy = round_half_up(y)
        dy = y - iy
        kernel_image = self.compute_kernel_image(x=x, y=y)
        if dx != 0 or dy != 0:
            legacy_result = offsetImage(kernel_image.to_legacy(), dx, dy, self._resampling_kernel, 5)
        else:
            # This branch is equal to the other up to round-off error, but it's
            # convenient nonetheless because it maintains exact compatibility
            # with the legacy implementation, where the caching mechanism
            # causes the offsetImage call to be skipped.
            legacy_result = kernel_image.to_legacy()
        legacy_result.setXY0(Point2I(legacy_result.getX0() + ix, legacy_result.getY0() + iy))
        return Image.from_legacy(legacy_result)

    def compute_stellar_bbox(self, *, x: float, y: float) -> Box:
        # This is obviously inefficient, but it's what afw does, and hence the
        # only easy way we've got to replicate what afw does.
        return self.compute_stellar_image(x=x, y=y).bbox

    def serialize(self, archive: OutputArchive[Any]) -> CellPointSpreadFunctionSerializationModel:
        array_model = archive.add_array(self._array)
        return CellPointSpreadFunctionSerializationModel(array=array_model, bounds=self.bounds)

    @staticmethod
    def deserialize(
        model: CellPointSpreadFunctionSerializationModel,
        archive: InputArchive[Any],
        *,
        bbox: Box | None = None,
    ) -> CellPointSpreadFunction:
        bounds = model.bounds
        if bbox is not None:
            bounds, slices = CellPointSpreadFunction._subset_impl(bounds, bbox)
            array = archive.get_array(model.array, slices=slices)
        else:
            array = archive.get_array(model.array)
        return CellPointSpreadFunction(array, bounds)

    @classmethod
    def from_legacy(cls, legacy_psf: Any, bounds: Bounds | None = None) -> CellPointSpreadFunction:
        # 'bounds' is accepted as an argument only for base-class
        # compatibility; we always generate our own bounds.
        from lsst.geom import Box2I

        grid = CellGrid.from_legacy(legacy_psf.grid)
        # Start with bounds that cover the entire grid.
        bounds = CellGridBounds(grid=grid, bbox=grid.bbox)
        # Shrink bounds to just the bbox where we have data.
        legacy_bbox = Box2I()
        for legacy_index in legacy_psf.images.keys():
            legacy_bbox.include(legacy_psf.grid.bbox_of(legacy_index))
        bounds = bounds[Box.from_legacy(legacy_bbox)]
        # Allocate and populate the array.
        psf_image_size_y, psf_image_size_x = legacy_psf.images.arbitrary.array.shape
        array = np.zeros(
            (
                bounds.bbox.y.size // grid.cell_shape.y,
                bounds.bbox.x.size // grid.cell_shape.x,
                psf_image_size_y,
                psf_image_size_x,
            ),
            dtype=np.float64,
        )
        missing: set[CellIJ] = set()
        for cell_index in bounds.cell_indices():
            legacy_index = cell_index.to_legacy()
            array_index = cell_index - bounds.grid_start
            if legacy_index in legacy_psf.images:
                array[array_index.i, array_index.j] = legacy_psf.images[legacy_index].array
            else:
                array[array_index.i, array_index.j] = np.nan
                missing.add(cell_index)
        # Modify the bounds one last time to account for missing cells.
        bounds = CellGridBounds(grid=grid, bbox=bounds.bbox, missing=frozenset(missing))
        return cls(array, bounds=bounds)

    @staticmethod
    def _subset_impl(bounds: CellGridBounds, bbox: Box) -> tuple[CellGridBounds, YX[slice]]:
        subset_bounds = bounds[bbox]
        start = subset_bounds.grid_start - bounds.grid_start
        stop = subset_bounds.grid_stop - bounds.grid_start
        return subset_bounds, YX(y=slice(start.i, stop.i), x=slice(start.j, stop.j))


class CellPointSpreadFunctionSerializationModel(ArchiveTree):
    """Model used to serialize CellPointSpreadFunction objects."""

    array: ArrayReferenceModel = pydantic.Field(
        description=(
            "A 4-d array of PSF kernel images with with shape "
            "(n_cells_y, n_cells_x, psf_shape_y, psf_shape_x)."
        )
    )
    bounds: CellGridBounds = pydantic.Field(
        description=(
            "Description of the cell grid and any missing cells.  Array entries for "
            "missing cells should be NaN."
        )
    )

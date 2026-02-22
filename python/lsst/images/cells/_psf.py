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

import math
from functools import cached_property
from typing import Any

import numpy as np
import pydantic

from .._cell_grid import CellGrid, CellGridBounds
from .._geom import BoundsError, Box
from .._image import Image
from ..psfs import PointSpreadFunction
from ..serialization import ArchiveTree, ArrayReferenceModel, InputArchive, OutputArchive


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
    """

    def __init__(self, array: np.ndarray, bounds: CellGridBounds):
        self._array = array
        self._bounds: CellGridBounds = bounds

    @property
    def grid(self) -> CellGrid:
        return self._bounds.grid

    @property
    def bounds(self) -> CellGridBounds:
        return self._bounds

    @cached_property
    def kernel_bbox(self) -> Box:
        sy, sx = self._array.shape[2:]
        ry = sy // 2
        rx = sx // 2
        return Box.factory[-ry : ry + 1, -rx : rx + 1]

    def __getitem__(self, bbox: Box) -> CellPointSpreadFunction:
        bounds, slices = self._bounds.subset(bbox)
        return CellPointSpreadFunction(self._array[slices + (...,)].copy(), bounds=bounds)

    def compute_kernel_image(self, *, x: float, y: float) -> Image:
        index = self.grid.index_of(x=round(x), y=round(y))
        if index in self._bounds.missing:
            raise BoundsError(f"Cell {index} for point (x={x}, y={y}) is missing for this PSF.")
        try:
            return Image(self._array[index.y, index.x], bbox=self.kernel_bbox)
        except IndexError:
            raise BoundsError(f"Cell {index} for point (x={x}, y={y}) is out of bounds for this PSF.")

    def compute_stellar_image(self, *, x: float, y: float) -> Image:
        try:
            from lsst.afw.math import offsetImage
        except ImportError as err:
            err.add_note("CellPointSpreadFunction.compute_stellar_image cannot be used without afw.")
            raise
        ix, dx = math.modf(x)
        iy, dy = math.modf(y)
        kernel_image = self.compute_kernel_image(x=x, y=y)
        legacy_result = offsetImage(kernel_image.to_legacy(), dx, dy, "lanczos5", 5)
        legacy_result.setX0(legacy_result.getX0() + ix)
        legacy_result.setY0(legacy_result.getY0() + iy)
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
            bounds, slices = bounds.subset(bbox)
            array = archive.get_array(model.array, slices=slices)
        else:
            array = archive.get_array(model.array)
        return CellPointSpreadFunction(array, bounds)


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

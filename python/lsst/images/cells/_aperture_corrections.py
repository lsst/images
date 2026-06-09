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

__all__ = ("CellApertureCorrectionMapSerializationModel", "CellField")

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar, final

import astropy.table
import astropy.units
import numpy as np
import pydantic

from .._cell_grid import CellGridBounds, CellIJ
from .._geom import BoundsError, Box
from .._image import Image
from ..fields import BaseField
from ..serialization import (
    ArchiveReadError,
    ArchiveTree,
    InputArchive,
    InvalidParameterError,
    OutputArchive,
    TableModel,
)

if TYPE_CHECKING:
    try:
        from lsst.afw.image import ApCorrMap as LegacyApCorrMap
        from lsst.cell_coadds import StitchedApertureCorrection as LegacyStichedApertureCorrection
    except ImportError:
        type LegacyApCorrMap = Any  # type: ignore[no-redef]
        type LegacyStichedApertureCorrection = Any  # type: ignore[no-redef]


@final
class CellField(BaseField):
    """A piecewise 2-d function on a cell-coadd grid.

    Parameters
    ----------
    array
        A 2-d array of cell values with shape
        ``bounds.subgrid_size.as_tuple()``.
    bounds
        Description of the cell grid and any missing cells.  Array entries for
        missing cells should be NaN.

    Notes
    -----
    `CellField` is not directly serializable and is not included in the
    ``Field`` union type alias as a result.  A `~collections.abc.Mapping` of
    `CellField` is instead serializable via
    `CellApertureCorrectionMapSerializationModel`.
    """

    def __init__(
        self, bounds: CellGridBounds, array: np.ndarray, unit: astropy.units.UnitBase | None = None
    ) -> None:
        self._array = array
        self._bounds = bounds
        self._unit = unit
        if self._array.shape != self._bounds.subgrid_size.as_tuple():
            raise ValueError(
                f"Array shape ({self._array.shape}) differs from subgrid size ({self._bounds.subgrid_size})."
            )

    __hash__ = None  # type: ignore[assignment]

    @property
    def bounds(self) -> CellGridBounds:
        return self._bounds

    @property
    def unit(self) -> astropy.units.UnitBase | None:
        return self._unit

    @property
    def is_constant(self) -> bool:
        indices = iter(self._bounds.cell_indices())
        try:
            first = self.value_in_cell(next(indices))
        except StopIteration:
            return True
        for other_index in indices:
            if self.value_in_cell(other_index) != first:
                return False
        return True

    def value_in_cell(self, key: CellIJ) -> float:
        """Return the value of the field in the cell with the given index."""
        if key in self._bounds.missing:
            raise BoundsError(f"Cell {key} is missing for this field.")
        index = key - self._bounds.subgrid_start
        try:
            return self._array[index.i, index.j]
        except IndexError:
            raise BoundsError(f"Cell {key} is out of bounds for this field.") from None

    def quantity_in_cell(self, key: CellIJ) -> astropy.units.Quantity:
        """Return the quantity (value with units) of the field in the cell
        with the given index.
        """
        return astropy.units.Quantity(self.value_in_cell(key), self._unit)

    def evaluate(
        self, *, x: np.ndarray, y: np.ndarray, quantity: bool
    ) -> np.ndarray | astropy.units.Quantity:
        # This implementation is optimized for the case where there are many
        # more evaluation points than cells. We could switch to an
        # implementation that zip-broadcast-iterates over x and y when that is
        # not the case, but that feels like a premature optimization right now.
        result = np.full(np.broadcast_shapes(y.shape, x.shape), np.nan, dtype=np.float64)
        for cell_index in self._bounds.cell_indices():
            cell_bbox = self._bounds.grid.bbox_of(cell_index)
            result[cell_bbox.contains(x=x, y=y)] = self.value_in_cell(cell_index)
        if quantity:
            return astropy.units.Quantity(result, self._unit)
        return result

    def render(self, bbox: Box | None = None, *, dtype: np.typing.DTypeLike | None = None) -> Image:
        if bbox is None:
            bbox = self._bounds.bbox
            bounds = self._bounds
        else:
            bounds = self._bounds[bbox]
        result = Image(np.nan, bbox=bbox, dtype=dtype, unit=self._unit)
        for cell_index in bounds.cell_indices():
            cell_bbox = self._bounds.grid.bbox_of(cell_index).intersection(bbox)
            result[cell_bbox].array = self.value_in_cell(cell_index)
        return result

    def multiply_constant(self, factor: float | astropy.units.Quantity | astropy.units.UnitBase) -> CellField:
        factor, unit = self._handle_factor_units(factor)
        return CellField(self._bounds, self._array * factor, unit=unit)

    @staticmethod
    def from_legacy_aperture_correction(
        legacy: LegacyStichedApertureCorrection, bounds: CellGridBounds
    ) -> CellField:
        """Convert from a legacy `lsst.cell_coadds.StitchedApertureCorrection`.

        Parameters
        ----------
        legacy
            Legacy field to convert.
        bounds
            The grid and bounds of the returned field.
        """
        array = np.full(bounds.subgrid_size.as_tuple(), np.nan, dtype=np.float64)
        for cell_index in bounds.cell_indices():
            array_index = cell_index - bounds.subgrid_start
            array[array_index.i, array_index.j] = legacy.gc[cell_index.to_legacy()]
        return CellField(bounds, array)

    def to_legacy_aperture_correction(self) -> LegacyStichedApertureCorrection:
        """Convert to a legacy
        `lsst.cell_coadds.StitchedApertureCorrection`.
        """
        from lsst.cell_coadds import GridContainer, StitchedApertureCorrection

        grid = self.bounds.grid.to_legacy()
        gc = GridContainer[float](grid.shape)
        for cell_index in self.bounds.cell_indices():
            gc[cell_index.to_legacy()] = self.value_in_cell(cell_index)
        return StitchedApertureCorrection(grid, gc)


class CellApertureCorrectionMapSerializationModel(ArchiveTree):
    """A serialization model for a `~collections.abc.Mapping` of `CellField`,
    which is used to represent aperture corrections for cell-based coadds.
    """

    SCHEMA_NAME: ClassVar[str] = "cell_aperture_correction_map"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"
    MIN_READ_VERSION: ClassVar[int] = 1
    PUBLIC_TYPE: ClassVar[type] = dict

    table: TableModel = pydantic.Field(
        description="Table with one row for each cell and different photometry algorithms in columns."
    )
    bounds: CellGridBounds = pydantic.Field(
        description=(
            "Description of the cell grid and any missing cells.  Array entries for "
            "missing cells should be NaN."
        ),
    )

    @staticmethod
    def serialize(
        aperture_correction_map: Mapping[str, CellField], archive: OutputArchive[Any]
    ) -> CellApertureCorrectionMapSerializationModel | None:
        if not aperture_correction_map:
            return None
        bounds = next(iter(aperture_correction_map.values())).bounds
        if not all(field.bounds == bounds for field in aperture_correction_map.values()):
            raise ValueError("Cell aperture corrections do not have consistent bounds.")
        if any(field.unit is not None for field in aperture_correction_map.values()):
            raise ValueError("Aperture corrections should be dimensionless.")
        table = astropy.table.Table(
            rows=[cell_index.as_tuple() for cell_index in bounds.cell_indices()], names=["cell_i", "cell_j"]
        )
        good_cell_mask = np.ones(bounds.subgrid_size.as_tuple(), dtype=bool)
        for cell_index in bounds.missing:
            array_index = cell_index - bounds.subgrid_start
            good_cell_mask[array_index.i, array_index.j] = False
        for name, field in aperture_correction_map.items():
            table.add_column(field._array[good_cell_mask], name=name, copy=False)
        return CellApertureCorrectionMapSerializationModel(
            table=archive.add_table(table, name="table"), bounds=bounds
        )

    def deserialize(self, archive: InputArchive[Any], **kwargs: Any) -> dict[str, CellField]:
        if kwargs:
            raise InvalidParameterError(
                f"Unrecognized parameters for cell aperture correction map: {set(kwargs.keys())}."
            )
        good_cell_mask = np.zeros(self.bounds.subgrid_size.as_tuple(), dtype=bool)
        table = archive.get_table(self.table)
        for tbl_ij, cell_index in zip(
            table["cell_i", "cell_j"].iterrows(), self.bounds.cell_indices(), strict=True
        ):
            if cell_index.as_tuple() != tbl_ij:
                raise ArchiveReadError(
                    "Inconsistency between serialized aperture correction bounds and table."
                )
            array_index = cell_index - self.bounds.subgrid_start
            good_cell_mask[array_index.i, array_index.j] = True
        result: dict[str, CellField] = {}
        for name, column in table.columns.items():
            if name in ("cell_i", "cell_j"):
                continue
            array = np.full(self.bounds.subgrid_size.as_tuple(), np.nan, dtype=np.float64)
            array[good_cell_mask] = column
            result[name] = CellField(self.bounds, array)
        return result

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

__all__ = (
    "CellApertureCorrectionMapSerializationModel",
    "CellField",
    "approximate_aperture_corrections_from_legacy",
)

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar, final

import astropy.table
import astropy.units
import numpy as np
import pydantic

from .._cell_grid import CellGridBounds, CellIJ
from .._geom import BoundsError, Box, Interval
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
        from lsst.afw.math import BoundedField as LegacyBoundedField
        from lsst.cell_coadds import StitchedApertureCorrection as LegacyStitchedApertureCorrection
    except ImportError:
        type LegacyBoundedField = Any  # type: ignore[no-redef]
        type LegacyStitchedApertureCorrection = Any  # type: ignore[no-redef]


@final
class CellField(BaseField):
    """A piecewise 2-d function on a cell-coadd grid.

    Parameters
    ----------
    bounds
        Description of the cell grid and any missing cells.  Array entries for
        missing cells should be NaN.
    array
        A 2-d array of cell values with shape
        ``bounds.subgrid_size.as_tuple()``.
    unit
        Units of the field values, or `None` if dimensionless.

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
        first = self.value_in_cell(next(indices))
        for other_index in indices:
            if self.value_in_cell(other_index) != first:
                return False
        return True

    def value_in_cell(self, key: CellIJ) -> float:
        """Return the value of the field in the cell with the given index.

        Parameters
        ----------
        key
            Index of the cell to evaluate.
        """
        if key in self._bounds.missing:
            raise BoundsError(f"Cell {key} is missing for this field.")
        if not self._bounds.contains_cell(key):
            raise BoundsError(f"Cell {key} is out of bounds for this field.")
        index = key - self._bounds.subgrid_start
        try:
            return self._array[index.i, index.j]
        except IndexError:
            raise BoundsError(f"Cell {key} is out of bounds for this field.") from None

    def quantity_in_cell(self, key: CellIJ) -> astropy.units.Quantity:
        """Return the quantity (value with units) of the field in the cell
        with the given index.

        Parameters
        ----------
        key
            Index of the cell to evaluate.
        """
        return astropy.units.Quantity(self.value_in_cell(key), self._unit)

    def _evaluate(
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

    def _multiply_constant(
        self, factor: float | astropy.units.Quantity | astropy.units.UnitBase
    ) -> CellField:
        factor, unit = self._handle_factor_units(factor)
        return CellField(self._bounds, self._array * factor, unit=unit)

    @staticmethod
    def from_legacy_aperture_correction(
        legacy: LegacyStitchedApertureCorrection, bounds: CellGridBounds
    ) -> CellField | None:
        """Convert from a legacy `lsst.cell_coadds.StitchedApertureCorrection`.

        Parameters
        ----------
        legacy
            Legacy field to convert.
        bounds
            The grid and bounds of the returned field.

        Returns
        -------
        field : `CellField` or `None`
            The converted field, or `None` if no cell has a usable value.
        """
        values: dict[CellIJ, float] = {}
        new_missing: set[CellIJ] = set()
        for cell_index in bounds.cell_indices():
            try:
                value = legacy.gc[cell_index.to_legacy()]
            except KeyError:
                value = np.nan
            if np.isfinite(value):
                values[cell_index] = value
            else:
                new_missing.add(cell_index)
        if not values:
            return None
        if new_missing:
            bounds = CellGridBounds(
                grid=bounds.grid, missing=frozenset(bounds.missing | new_missing), bbox=bounds.bbox
            )
        array = np.full(bounds.subgrid_size.as_tuple(), np.nan, dtype=np.float64)
        for cell_index, value in values.items():
            array[cell_index.within_subgrid_array(bounds)] = value
        return CellField(bounds, array)

    def to_legacy_aperture_correction(self) -> LegacyStitchedApertureCorrection:
        """Convert to a legacy
        `lsst.cell_coadds.StitchedApertureCorrection`.
        """
        from lsst.cell_coadds import GridContainer, StitchedApertureCorrection

        grid = self.bounds.grid.to_legacy()
        gc = GridContainer[float](grid.shape)
        for cell_index in self.bounds.cell_indices():
            gc[cell_index.to_legacy()] = self.value_in_cell(cell_index)
        return StitchedApertureCorrection(grid, gc)

    @staticmethod
    def approximate_from_legacy(
        legacy: LegacyBoundedField, bounds: CellGridBounds, default_is_missing: bool = True
    ) -> CellField | None:
        """Convert from a legacy bounded field by evaluating it at cell
        centers and assuming variation within a cell is approximately
        constant.

        When the center sample is unusable (e.g. the center falls outside
        the field's bounding box), the field is instead sampled at the
        cell's four corners and the usable samples are averaged.  Cells
        with no usable sample at all are marked missing.

        Parameters
        ----------
        legacy
            Legacy bounded field to approximate.
        bounds
            Cell grid and known-missing cells.
        default_is_missing
            If `True` and ``legacy.getThrowOnMissing()`` returns `False`,
            treat values that are identical to ``legacy.getDefault()`` as
            missing.  This covers some confusing
            `lsst.meas.algorithms.CoaddBoundedField` behavior for backwards
            compatibility.

        Returns
        -------
        field : `CellField` or `None`
            The converted field, or `None` if no cell has a usable value.
        """
        from lsst.pex.exceptions import DomainError

        get_throw_on_missing = getattr(legacy, "getThrowOnMissing", None)
        get_default = getattr(legacy, "getDefault", None)
        default = (
            get_default()
            if default_is_missing
            and get_throw_on_missing is not None
            and get_default is not None
            and not get_throw_on_missing()
            else None
        )

        def usable(value: float) -> bool:
            return np.isfinite(value) and (default is None or value != default)

        values: dict[CellIJ, float] = {}
        missing: set[CellIJ] = set()
        for cell_index in bounds.cell_indices():
            cell_bbox = bounds.grid.bbox_of(cell_index)
            try:
                value = legacy.evaluate(cell_bbox.center.to_legacy_float_point())
            except DomainError:
                value = np.nan
            if not usable(value):
                corner_values: list[float] = []
                for corner in cell_bbox.boundary():
                    try:
                        corner_value = legacy.evaluate(corner.to_legacy_float_point())
                    except DomainError:
                        continue
                    if not usable(corner_value):
                        continue
                    corner_values.append(corner_value)
                if not corner_values:
                    missing.add(cell_index)
                    continue
                else:
                    value = np.mean(corner_values)
            values[cell_index] = value
        if not values:
            return None
        if missing:
            bounds = CellGridBounds(
                grid=bounds.grid, missing=frozenset(bounds.missing | missing), bbox=bounds.bbox
            )
        array = np.full(bounds.subgrid_size.as_tuple(), np.nan, dtype=np.float64)
        for cell_index, value in values.items():
            array[cell_index.within_subgrid_array(bounds)] = value
        return CellField(bounds, array)


def approximate_aperture_corrections_from_legacy(
    legacy_map: Mapping[str, LegacyBoundedField],
    bounds: CellGridBounds,
    flux_suffix: str = "_instFlux",
    err_suffix: str = "_instFluxErr",
) -> dict[str, CellField | None]:
    """Convert a mapping of legacy aperture-correction bounded fields to
    `CellField` instances by approximating them on a cell grid.

    Parameters
    ----------
    legacy_map
        Mapping of aperture-correction names to legacy bounded fields, as
        returned by a legacy `lsst.afw.image.Exposure`'s ``getApCorrMap()``.
    bounds
        Cell grid and known-missing cells to convert onto.
    flux_suffix
        Suffix of the flux member of an aperture-correction name pair.
    err_suffix
        Suffix of the error member of an aperture-correction name pair.

    Returns
    -------
    `dict` [`str`, `CellField` | `None`]
        Mapping with the same keys, holding the converted fields.  A name whose
        legacy field has no usable value in any cell maps to `None`.

    Notes
    -----
    This routine relies on the fact that aperture correction fields come in
    ``*_instFlux`` + ``*_instFluxErr`` pairs to resolve the fact that zero is
    a possible valid (albeit suspicious) value for the latter *and* the
    sentinel used for missing data.  We assume a zero ``*_instFlux``
    evaluation represents bad data, but a zero ``*_instFluxErr`` is bad data
    only if the corresponding ``*_instFlux`` evaluation is also bad data.
    """
    converted: dict[str, CellField | None] = {}
    for name, legacy_field in legacy_map.items():
        if name.endswith(err_suffix):
            continue
        converted[name] = CellField.approximate_from_legacy(legacy_field, bounds)
    for name, legacy_field in legacy_map.items():
        if not name.endswith(err_suffix):
            continue
        partner = converted.get(name[: -len(err_suffix)] + flux_suffix)
        err_bounds = partner.bounds if partner is not None else bounds
        converted[name] = CellField.approximate_from_legacy(
            legacy_field, err_bounds, default_is_missing=False
        )
    return converted


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
        aperture_correction_map: Mapping[str, CellField | None], archive: OutputArchive[Any]
    ) -> CellApertureCorrectionMapSerializationModel | None:
        """Serialize an aperture-correction map, one table column per name.

        Names that map to `None` are written as columns that are entirely NaN.
        A map whose every name is `None` is written as no aperture
        corrections (top-level `None`).

        Parameters
        ----------
        aperture_correction_map
            Aperture corrections to write, keyed by name.  A name with no
            usable value in any cell maps to `None`.
        archive
            Archive to add the cell-value table to.

        Returns
        -------
        model : `CellApertureCorrectionMapSerializationModel` or `None`
            The serialized model, or `None` if no name has a usable value.
        """
        usable = {name: field for name, field in aperture_correction_map.items() if field is not None}
        if not usable:
            return None
        grid = next(iter(usable.values())).bounds.grid
        for field in usable.values():
            if field.bounds.grid != grid:
                raise ValueError("Cell aperture corrections do not have a consistent grid.")
        if any(field.unit is not None for field in usable.values()):
            raise ValueError("Aperture corrections should be dimensionless.")
        # Fields differ only in which cells are missing, so their bounds differ
        # if one has missing cells along a full row or column on the boundary.
        # The table therefore covers the cells any field uses.  This avoids a
        # schema change and makes reading older files easier.
        pixel_bbox = Box(
            y=Interval.hull(*(field.bounds.bbox.y for field in usable.values())),
            x=Interval.hull(*(field.bounds.bbox.x for field in usable.values())),
        )
        present = {index for field in usable.values() for index in field.bounds.cell_indices()}
        all_cells = set(CellGridBounds(grid=grid, bbox=pixel_bbox).cell_indices())
        bounds = CellGridBounds(grid=grid, bbox=pixel_bbox, missing=frozenset(all_cells - present))
        rows = {cell_index: row for row, cell_index in enumerate(bounds.cell_indices())}
        table = astropy.table.Table(
            rows=[cell_index.as_tuple() for cell_index in bounds.cell_indices()], names=["cell_i", "cell_j"]
        )
        for name, ap_corr in aperture_correction_map.items():
            column = np.full(len(table), np.nan)
            if ap_corr is not None:
                for cell_index in ap_corr.bounds.cell_indices():
                    column[rows[cell_index]] = ap_corr._array[cell_index.within_subgrid_array(ap_corr.bounds)]
            table.add_column(column, name=name, copy=False)
        return CellApertureCorrectionMapSerializationModel(
            table=archive.add_table(table, name="table"), bounds=bounds
        )

    def deserialize(self, archive: InputArchive[Any], **kwargs: Any) -> dict[str, CellField | None]:
        """Read an aperture-correction map from a table with one column per
        name.

        A column that is entirely NaN denotes a name with no usable value in
        any cell, which becomes `None`.  Cells that are missing for just one
        name (recorded as NaN) become missing cells of that field's bounds.

        Parameters
        ----------
        archive
            Archive to read the cell-value table from.
        **kwargs
            Must be empty.

        Returns
        -------
        aperture_corrections : `dict` [`str`, `CellField` | `None`]
            Aperture corrections, keyed by name.
        """
        if kwargs:
            raise InvalidParameterError(
                f"Unrecognized parameters for cell aperture correction map: {set(kwargs.keys())}."
            )
        table = archive.get_table(self.table)
        rows: dict[CellIJ, int] = {}
        for row, (tbl_ij, cell_index) in enumerate(
            zip(table["cell_i", "cell_j"].iterrows(), self.bounds.cell_indices(), strict=True)
        ):
            if cell_index.as_tuple() != tbl_ij:
                raise ArchiveReadError(
                    "Inconsistency between serialized aperture correction bounds and table."
                )
            rows[cell_index] = row
        result: dict[str, CellField | None] = {}
        for name, column in table.columns.items():
            if name in ("cell_i", "cell_j"):
                continue
            extra_missing_mask = np.isnan(column)
            if extra_missing_mask.all():
                result[name] = None
                continue
            # If there are any NaN entries in this column, make a custom
            # CellGridBounds with additional missing cells for this field.
            bounds = self.bounds
            if extra_missing_mask.any():
                extra_missing = frozenset(
                    {CellIJ(i=i, j=j) for i, j in table["cell_i", "cell_j"][extra_missing_mask].iterrows()}
                )
                bounds = CellGridBounds(
                    grid=self.bounds.grid, missing=self.bounds.missing | extra_missing, bbox=self.bounds.bbox
                )
            array = np.full(bounds.subgrid_size.as_tuple(), np.nan, dtype=np.float64)
            for cell_index in bounds.cell_indices():
                array[cell_index.within_subgrid_array(bounds)] = column[rows[cell_index]]
            result[name] = CellField(bounds, array)
        return result

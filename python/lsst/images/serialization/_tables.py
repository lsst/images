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
    "ColumnDefinitionModel",
    "TableCellReferenceModel",
    "TableModel",
    "UnsupportedTableError",
)

import operator
from typing import TYPE_CHECKING, ClassVar, Literal

import astropy.units
import numpy as np
import numpy.typing as npt
import pydantic

from ._asdf_utils import Unit
from ._common import ArchiveTree
from ._dtypes import NumberType

if TYPE_CHECKING:
    import astropy.table


class UnsupportedTableError(NotImplementedError):
    """Exception raised if a table object has column types or structure that
    are not supported by this library.
    """


class ColumnDefinitionModel(ArchiveTree):
    """A model that describes a column in a table."""

    name: str
    """Name of the column."""

    datatype: NumberType
    """Type of the column."""

    unit: Unit | None = pydantic.Field(default=None, exclude_if=operator.not_)
    """Units of the column."""

    description: str = pydantic.Field(default="", exclude_if=operator.not_)
    """Extended description of the column."""

    shape: tuple[int, ...] = pydantic.Field(default=(), exclude_if=operator.not_)
    """Shape of a single cell in of this column.

    An empty `tuple` is used to represent a scalar column.
    """

    is_variable_length: bool = pydantic.Field(default=False, exclude_if=operator.not_)
    """Whether this column is a variable-length array."""

    @classmethod
    def from_record_dtype(cls, dtype: npt.DTypeLike) -> list[ColumnDefinitionModel]:
        """Extract a list of column definitions from a structured numpy dtype.

        Parameters
        ----------
        dtype
            Object convertible to `numpy.dtype`.
        """
        dtype = np.dtype(dtype)
        result: list[ColumnDefinitionModel] = []
        if dtype.fields is None:
            raise TypeError(f"{dtype} is not a structured dtype.")
        for name, (field_dtype, *_) in dtype.fields.items():
            # TODO: support string and variable-length array columns here.
            try:
                datatype, shape = NumberType.from_numpy_with_shape(field_dtype)
            except TypeError:
                raise UnsupportedTableError(f"Column type {field_dtype} is not supported.") from None
            result.append(ColumnDefinitionModel.model_construct(name=name, datatype=datatype, shape=shape))
        return result

    def update_from_table(self, table: astropy.table.Table) -> None:
        """Update the unit and description of this column from an astropy
        table.
        """
        astropy_column: astropy.table.Column = table.columns[self.name]
        self.unit = astropy.units.Unit(astropy_column.unit) if astropy_column.unit is not None else None
        self.description = astropy_column.description

    def update_table(self, table: astropy.table.Table) -> None:
        """Update the unit and description of an astropy column from this
        object.
        """
        astropy_column: astropy.table.Column = table.columns[self.name]
        astropy_column.unit = self.unit
        astropy_column.description = self.description


class TableModel(ArchiveTree):
    """Placeholder for an ASDF-like model for referencing binary tabular
    data.
    """

    source: str | int
    """Reference to the table data.

    This is analogous to the ASDF ``ndarray`` field of the same name, i.e
    for a FITS binary table, use "fits:EXTNAME[,EXTVER]" or "fits:INDEX"
    (zero-indexed) to identify the HDU.
    """

    columns: list[ColumnDefinitionModel] = pydantic.Field(default_factory=list)
    """Definitions of all columns."""

    source_is_table: ClassVar[Literal[True]] = True


class TableCellReferenceModel(ArchiveTree):
    """A model that acts as a pointer to data in a table cell."""

    model_config = pydantic.ConfigDict(frozen=True)

    source: str | int
    """Identifier for the table as a whole.

    This is analogous to the ASDF ``ndarray`` field of the same name, i.e
    for a FITS binary table, use "fits:EXTNAME[,EXTVER]" or "fits:INDEX"
    (zero-indexed) to identify the HDU.
    """

    column: str
    """Name of the column."""

    row: int
    """Row of the cell (zero-indexed)."""

    source_is_table: ClassVar[Literal[True]] = True

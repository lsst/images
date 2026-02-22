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
    "TableColumnModel",
    "TableModel",
    "UnsupportedTableError",
)

import operator
from typing import TYPE_CHECKING

import astropy.units
import numpy as np
import numpy.typing as npt
import pydantic

from ._asdf_utils import ArrayReferenceModel, InlineArrayModel, Unit
from ._common import ArchiveReadError
from ._dtypes import NumberType

if TYPE_CHECKING:
    import astropy.table


class UnsupportedTableError(NotImplementedError):
    """Exception raised if a table object has column types or structure that
    are not supported by this library.
    """


class TableColumnModel(pydantic.BaseModel, ser_json_inf_nan="constants"):
    """Model for a subset of the ASDF table/column schema."""

    data: InlineArrayModel | ArrayReferenceModel
    """Column data."""

    name: str
    """Name of the column."""

    description: str = pydantic.Field(default="", exclude_if=operator.not_)
    """Extended description of the column."""

    unit: Unit | None = pydantic.Field(default=None, exclude_if=operator.not_)
    """Units of the column."""

    meta: dict[str, int | float | str | bool | None] = pydantic.Field(
        default_factory=dict, exclude_if=operator.not_, description="Free-form metadata for the column."
    )

    model_config = pydantic.ConfigDict(
        json_schema_extra={
            "$schema": "http://stsci.edu/schemas/yaml-schema/draft-01",
            "id": "http://stsci.edu/schemas/asdf/table/column-1.1.0",
            "tag": "!table/column-1.1.0",
        }
    )

    @classmethod
    def from_record_dtype(cls, dtype: npt.DTypeLike) -> list[TableColumnModel]:
        """Extract a list of column definitions from a structured numpy dtype.

        Parameters
        ----------
        dtype
            Object convertible to `numpy.dtype`.

        Notes
        -----
        This sets the `data` field to an `ArrayReferenceModel` with ``source``
        set to an empty string.  This will need to be modified later.
        """
        dtype = np.dtype(dtype)
        result: list[TableColumnModel] = []
        if dtype.fields is None:
            raise TypeError(f"{dtype} is not a structured dtype.")
        for name, (field_dtype, *_) in dtype.fields.items():
            # TODO: support string and variable-length array columns here.
            try:
                datatype, shape = NumberType.from_numpy_with_shape(field_dtype)
            except TypeError:
                raise UnsupportedTableError(f"Column type {field_dtype} is not supported.") from None
            result.append(
                TableColumnModel(
                    data=ArrayReferenceModel(source="", datatype=datatype, shape=list(shape)),
                    name=name,
                )
            )
        return result

    @classmethod
    def from_record_array(cls, array: np.ndarray, inline: bool = False) -> list[TableColumnModel]:
        """Extract a list of column definitions from a structured numpy array.

        Parameters
        ----------
        array
            A table-like array.
        inline
            Whether to store the array data directly in the columns.

        Notes
        -----
        When ``inline=False``, this sets the `data` field to an
        `ArrayReferenceModel` with ``source`` set to an empty string.  This
        will need to be modified later.
        """
        if not inline:
            return cls.from_record_dtype(array.dtype)
        result: list[TableColumnModel] = []
        if array.dtype.fields is None:
            raise TypeError(f"{array.dtype} is not a structured dtype.")
        for name, (field_dtype, *_) in array.dtype.fields.items():
            # TODO: support string and variable-length array columns here.
            try:
                datatype, shape = NumberType.from_numpy_with_shape(field_dtype)
            except TypeError:
                raise UnsupportedTableError(f"Column type {field_dtype} is not supported.") from None
            result.append(
                TableColumnModel(
                    data=InlineArrayModel(data=array[name].tolist(), datatype=datatype),
                    name=name,
                )
            )
        return result

    @classmethod
    def from_table(cls, table: astropy.table.Table, inline: bool = False) -> list[TableColumnModel]:
        """Extract column definitions and (optionally) data from an Astropy
        table.
        """
        return [cls.from_column(c, inline=inline) for c in table.columns.values()]

    @classmethod
    def from_column(cls, column: astropy.table.Column, inline: bool = False) -> TableColumnModel:
        """Extract a column definition and (optionally) data from an Astropy
        column.

        Notes
        -----
        When ``inline=False`, this sets the `data` field to an
        `ArrayReferenceModel` with ``source`` set to an empty string.  This
        will need to be modified later.
        """
        # TODO: support string and variable-length array columns here.
        try:
            datatype = NumberType.from_numpy(column.dtype)
        except TypeError:
            raise UnsupportedTableError(f"Column type {column.dtype} is not supported.") from None

        data = (
            InlineArrayModel(data=column.tolist(), datatype=datatype)
            if inline
            else ArrayReferenceModel(
                source="",
                datatype=datatype,
                shape=column.shape[1:],
            )
        )
        return TableColumnModel(
            data=data,
            name=column.name,
            unit=astropy.units.Unit(column.unit) if column.unit is not None else None,
            meta=column.meta,
            description=column.description or "",
        )

    def update_table(self, table: astropy.table.Table) -> None:
        """Update the unit and description of an astropy column from this
        object.
        """
        astropy_column: astropy.table.Column = table.columns[self.name]
        astropy_column.unit = self.unit
        astropy_column.description = self.description
        if (datatype := NumberType.from_numpy(astropy_column.dtype)) != self.data.datatype:
            raise ArchiveReadError(
                f"Table column {self.name} has type {datatype}; expected {self.data.datatype}."
            )
        if (shape := astropy_column.shape[1:]) != tuple(self.data.shape):
            raise ArchiveReadError(
                f"Table column {self.name} has shape {shape}; expected {tuple(self.data.shape)}."
            )


class TableModel(pydantic.BaseModel):
    """Placeholder for an ASDF-like model for referencing or holding binary
    tabular data.
    """

    columns: list[TableColumnModel] = pydantic.Field(
        default_factory=list, description="Definitions of all columns."
    )
    meta: dict[str, int | float | str | bool | None] = pydantic.Field(
        default_factory=dict, exclude_if=operator.not_, description="Free-form metadata for the table."
    )

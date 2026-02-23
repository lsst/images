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

__all__ = ("TableRowModel", "is_none")

import operator
import sys
from collections.abc import Iterable
from typing import Self

import astropy.table
import astropy.units
import numpy as np
import pydantic

if sys.version_info >= (3, 14, 0):
    is_none = operator.is_none  # type: ignore[attr-defined]
else:

    def is_none(x: object) -> bool:
        """Test whether an object is None."""
        return x is None


class TableRowModel(pydantic.BaseModel):
    """An intermediate base class for Pydantic models that represent the rows
    of an Astropy table.

    Notes
    -----
    The fields of a `TableRowModel` should be simple, array-friendly types
    and may not be nested Pydantic models.  Annotations can be used to control
    the column array data type and/or add units::

        from typing import Annotation
        import numpy as np
        import astropy.units as u


        class ExampleRow(TableRowModel):
            a: Annotated[int, np.dtype(np.uint16)]
            b: Annotated[float, u.s]


        rows = [ExampleRow(a=1, b=3.0), ExampleRow(a=2, b=2.5)]

        tbl = Example.model_make_table(rows)

    Annotations must be `numpy.dtype` instances (not just types convertible to
    `numpy.dtype`) or `astropy.unit.Unit` instances.
    """

    @classmethod
    def model_column_dtypes(cls) -> dict[str, np.dtype]:
        """Return a mapping from column name to array data type."""
        result: dict[str, np.dtype] = {}
        for name, field_info in cls.model_fields.items():
            for annotation in field_info.metadata:
                if isinstance(annotation, np.dtype):
                    result[name] = annotation
                    break
            else:
                result[name] = np.dtype(field_info.annotation)
        return result

    @classmethod
    def model_column_units(cls) -> dict[str, astropy.unit.Unit | None]:
        """Return a mapping from column name to its units."""
        result: dict[str, astropy.unit.Unit | None] = {}
        for name, field_info in cls.model_fields.items():
            for annotation in field_info.metadata:
                if isinstance(annotation, astropy.unit.Unit):
                    result[name] = annotation
                    break
            else:
                result[name] = None
        return result

    @classmethod
    def model_make_table(cls, rows: Iterable[Self]) -> astropy.table.Table:
        """Transform a row of model instances into an `astropy.table.Table`."""
        return astropy.table.Table(
            row=[r.model_dump() for r in rows],
            names=list(cls.model_fields.keys()),
            dtype=cls.model_column_dtypes(),
            units=cls.model_column_units(),
            descriptions={name: field_info.description for name, field_info in cls.model_fields.items()},
        )

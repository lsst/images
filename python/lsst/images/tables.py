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
    "TableCellReferenceModel",
    "TableModel",
)

from typing import ClassVar, Literal

import pydantic


class TableModel(pydantic.BaseModel):
    """Placeholder for an ASDF-like model for referencing binary tabular
    data.
    """


class TableCellReferenceModel(pydantic.BaseModel):
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

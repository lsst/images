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

"""Utility code for working with JSON or raw JSON-derived dictionaries."""

from __future__ import annotations

__all__ = ("JsonValue", "PointerModel")

from typing import Annotated

import pydantic

type JsonValue = int | str | float | None | list["JsonValue"] | dict[str, "JsonValue"]


class PointerModel(pydantic.BaseModel):
    """Pydantic model for a JSON Pointer (IETF RFC 6901)."""

    # Using Annotated here instead of ' = Field(...)' keeps type checkers from
    # generating the wrong __init__ signature.
    ref: Annotated[str, pydantic.Field(alias="$ref")]

    model_config = pydantic.ConfigDict(serialize_by_alias=True, validate_by_alias=True)

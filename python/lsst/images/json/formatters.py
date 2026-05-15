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

"""Deprecated re-export of the unified ``lsst.images.formatters`` module.

`lsst.images.json.formatters.GenericFormatter` exists so that deployed
butler configs that point Transform and Projection storage classes at
this path keep working. The shim overrides ``default_extension`` to
``.json`` so writes default to JSON output when no ``format`` write
parameter is supplied.
"""

from __future__ import annotations

__all__ = ("GenericFormatter",)

import warnings
from typing import Any, ClassVar

from .. import formatters as _unified


def _warn(name: str) -> None:
    warnings.warn(
        f"lsst.images.json.formatters.{name} is deprecated; "
        f"use lsst.images.formatters.{name} with format='json' "
        f"instead. The json-only formatter forwards to the unified "
        f"one and will be removed in a future release.",
        DeprecationWarning,
        stacklevel=3,
    )


class GenericFormatter(_unified.GenericFormatter):
    """Deprecated alias for `lsst.images.formatters.GenericFormatter`.

    Defaults to ``.json`` output so existing butler configs that point
    Transform/Projection storage classes here keep producing JSON
    without specifying a ``format`` write parameter.
    """

    default_extension: ClassVar[str] = ".json"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _warn("GenericFormatter")
        super().__init__(*args, **kwargs)

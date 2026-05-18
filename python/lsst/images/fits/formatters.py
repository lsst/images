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

"""Deprecated re-exports of the unified ``lsst.images.formatters`` module.

These names are kept so that deployed butler configs in
``daf_butler/configs/datastores/formatters.yaml`` continue to work.
Each class is a one-line subclass of the corresponding unified
formatter that emits a `DeprecationWarning` on first instantiation.
"""

from __future__ import annotations

__all__ = (
    "CellCoaddFormatter",
    "GenericFormatter",
    "ImageFormatter",
    "MaskedImageFormatter",
    "VisitImageFormatter",
)

import warnings
from typing import Any

from .. import formatters as _unified


def _warn(name: str) -> None:
    warnings.warn(
        f"lsst.images.fits.formatters.{name} is deprecated; "
        f"use lsst.images.formatters.{name} instead. The fits-only "
        f"formatter forwards to the unified one and will be removed "
        f"in a future release.",
        DeprecationWarning,
        stacklevel=3,
    )


class GenericFormatter(_unified.GenericFormatter):
    """Deprecated alias for `lsst.images.formatters.GenericFormatter`."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _warn("GenericFormatter")
        super().__init__(*args, **kwargs)


class ImageFormatter(_unified.GenericFormatter):
    """Deprecated alias for `lsst.images.formatters.GenericFormatter`."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _warn("ImageFormatter")
        super().__init__(*args, **kwargs)


class MaskedImageFormatter(_unified.GenericFormatter):
    """Deprecated alias for `lsst.images.formatters.GenericFormatter`."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _warn("MaskedImageFormatter")
        super().__init__(*args, **kwargs)


class VisitImageFormatter(_unified.GenericFormatter):
    """Deprecated alias for `lsst.images.formatters.GenericFormatter`."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _warn("VisitImageFormatter")
        super().__init__(*args, **kwargs)


class CellCoaddFormatter(_unified.GenericFormatter):
    """Deprecated alias for `lsst.images.formatters.GenericFormatter`."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _warn("CellCoaddFormatter")
        super().__init__(*args, **kwargs)

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

"""Unified butler formatter for lsst.images.

This formatter dispatches on a write-time ``format`` parameter and on the
file extension at read time, replacing the three per-format
(`lsst.images.fits.formatters`, `lsst.images.json.formatters`,
`lsst.images.ndf.formatters`) hierarchies that previously duplicated almost
all of their logic.
"""

from __future__ import annotations

__all__ = ()  # populated in later tasks

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from . import fits as _fits
from . import json as _json
from .fits._common import PointerModel as _FitsPointerModel
from .fits._input_archive import FitsInputArchive as _FitsInputArchive

try:
    from . import ndf as _ndf
    from .ndf._common import NdfPointerModel as _NdfPointerModel
    from .ndf._input_archive import NdfInputArchive as _NdfInputArchive

    _HAVE_NDF = True
except ImportError:  # h5py is optional; see ndf/__init__.py
    _ndf = None  # type: ignore[assignment]
    _NdfPointerModel = None  # type: ignore[assignment]
    _NdfInputArchive = None  # type: ignore[assignment]
    _HAVE_NDF = False


@dataclass(frozen=True)
class _Backend:
    """One row of the extension-to-backend lookup table."""

    read: Callable[..., Any]
    write: Callable[..., Any]
    input_archive: type | None
    pointer_model: type | None


_BACKENDS: dict[str, _Backend] = {
    ".fits": _Backend(
        read=_fits.read,
        write=_fits.write,
        input_archive=_FitsInputArchive,
        pointer_model=_FitsPointerModel,
    ),
    ".json": _Backend(
        read=_json.read,
        write=_json.write,
        input_archive=None,
        pointer_model=None,
    ),
}
if _HAVE_NDF:
    _BACKENDS[".sdf"] = _Backend(
        read=_ndf.read,
        write=_ndf.write,
        input_archive=_NdfInputArchive,
        pointer_model=_NdfPointerModel,
    )

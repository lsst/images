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

__all__ = ("Backend", "backend_for_path")

import dataclasses
from collections.abc import Callable
from typing import TYPE_CHECKING

from lsst.resources import ResourcePathExpression

if TYPE_CHECKING:
    from ._input_archive import InputArchive


@dataclasses.dataclass(frozen=True)
class Backend:
    """A file-format backend resolved from a path suffix.

    Bundles the backend's free ``read``/``write`` functions and its
    `InputArchive` subclass (whose `~InputArchive.get_basic_info` reads
    file metadata).
    """

    name: str
    read: Callable[..., object]
    write: Callable[..., object]
    input_archive: type[InputArchive]


def backend_for_path(path: ResourcePathExpression) -> Backend:
    """Return the `Backend` for ``path`` based on its file extension.

    Supported extensions: ``.fits`` / ``.fits.gz`` (FITS), ``.sdf`` /
    ``.ndf`` (NDF), and ``.json`` (JSON).  The NDF and FITS backends are
    imported lazily so optional dependencies (e.g. ``h5py``) are only
    required when actually used.

    Raises
    ------
    ValueError
        If the extension is not recognised.
    """
    s = str(path)
    if s.endswith(".fits") or s.endswith(".fits.gz"):
        from ..fits import FitsInputArchive
        from ..fits import read as fits_read
        from ..fits import write as fits_write

        return Backend("fits", fits_read, fits_write, FitsInputArchive)
    if s.endswith(".sdf") or s.endswith(".ndf"):
        from ..ndf import NdfInputArchive
        from ..ndf import read as ndf_read
        from ..ndf import write as ndf_write

        return Backend("ndf", ndf_read, ndf_write, NdfInputArchive)
    if s.endswith(".json"):
        from ..json import JsonInputArchive
        from ..json import read as json_read
        from ..json import write as json_write

        return Backend("json", json_read, json_write, JsonInputArchive)
    raise ValueError(
        f"Unrecognised file extension: {path!r}; "
        "expected one of .fits, .fits.gz, .sdf, .ndf, .json."
    )

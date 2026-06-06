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

    Bundles the backend's free ``write`` function and its `InputArchive`
    subclass.  Reading goes through the generic ``open`` / ``read`` in
    `lsst.images.serialization`, which use the `InputArchive`'s
    ``get_basic_info`` and ``open_tree``.
    """

    name: str
    write: Callable[..., object]
    input_archive: type[InputArchive]


def backend_for_path(path: ResourcePathExpression) -> Backend:
    """Return the `Backend` for ``path`` based on its file extension.

    Supported extensions: ``.fits`` / ``.fits.gz`` (FITS), ``.sdf`` /
    ``.ndf`` (NDF), ``.json`` (JSON), and ``.zarr`` / ``.zarr.zip`` (zarr).
    The NDF, FITS, and zarr backends are imported lazily so optional
    dependencies (e.g. ``h5py``, ``zarr``) are only required when actually
    used.

    Raises
    ------
    ValueError
        If the extension is not recognised.
    """
    s = str(path)
    if s.endswith(".fits") or s.endswith(".fits.gz"):
        from ..fits import FitsInputArchive
        from ..fits import write as fits_write

        return Backend("fits", fits_write, FitsInputArchive)
    if s.endswith(".sdf") or s.endswith(".ndf"):
        from ..ndf import NdfInputArchive
        from ..ndf import write as ndf_write

        return Backend("ndf", ndf_write, NdfInputArchive)
    if s.endswith(".json"):
        from ..json import JsonInputArchive
        from ..json import write as json_write

        return Backend("json", json_write, JsonInputArchive)
    if s.endswith(".zarr") or s.endswith(".zarr.zip"):
        from ..zarr import ZarrInputArchive
        from ..zarr import write as zarr_write

        return Backend("zarr", zarr_write, ZarrInputArchive)
    raise ValueError(
        f"Unrecognised file extension: {path!r}; "
        "expected one of .fits, .fits.gz, .sdf, .ndf, .json, .zarr, .zarr.zip."
    )

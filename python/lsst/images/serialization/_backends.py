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

from lsst.resources import ResourcePath, ResourcePathExpression

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

    Supported extensions: ``.fits`` / ``.fits.gz`` (FITS), ``.h5`` /
    ``.sdf`` (NDF), ``.json`` (JSON), and ``.zarr`` / ``.zarr.zip`` (zarr).
    The NDF, FITS, and zarr backends are imported lazily so optional
    dependencies (e.g. ``h5py``, ``zarr``) are only required when actually
    used.

    Parameters
    ----------
    path
        Path whose file extension selects the backend.

    Raises
    ------
    ValueError
        If the extension is not recognised.
    """
    uri = ResourcePath(path)
    match uri.getExtension():
        case ".fits" | ".fits.gz":
            from ..fits import FitsInputArchive
            from ..fits import write as fits_write

            return Backend("fits", fits_write, FitsInputArchive)
        case ".h5" | ".sdf":
            from ..ndf import NdfInputArchive
            from ..ndf import write as ndf_write

            return Backend("ndf", ndf_write, NdfInputArchive)
        case ".json":
            from ..json import JsonInputArchive
            from ..json import write as json_write

            return Backend("json", json_write, JsonInputArchive)
        # A zip zarr store is a ``.zarr.zip`` path, but ``.zip`` is not a
        # compression modifier that ``getExtension`` folds into a compound
        # extension (the way it does for ``.fits.gz``), so a ``.zarr.zip``
        # path reports its extension as ``.zip``.  Accept both here; the
        # zarr store layer re-derives directory-vs-zip from the full path.
        case ".zarr" | ".zarr.zip" | ".zip":
            from ..zarr import ZarrInputArchive
            from ..zarr import write as zarr_write

            return Backend("zarr", zarr_write, ZarrInputArchive)
        case ext:
            raise ValueError(
                f"Unrecognised file extension: {ext!r} from {uri!r}; "
                "expected one of .fits, .fits.gz, .h5, .sdf, .json, .zarr, .zarr.zip."
            )

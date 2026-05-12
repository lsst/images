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

__all__ = ("NdfPointerModel", "archive_path_to_hdf5_path")

import pydantic


class NdfPointerModel(pydantic.BaseModel, serialize_by_alias=True):
    """Reference to an NDF-archive sub-tree by HDF5 path.

    Used by `NdfOutputArchive`/`NdfInputArchive` to point to
    sub-trees that have been hoisted out of the main JSON tree into separate
    HDS components.
    """

    ref: str = pydantic.Field(alias="$ref")
    """HDF5 absolute path (e.g. ``/MORE/LSST/PSF``)."""

    model_config = pydantic.ConfigDict(populate_by_name=True)


def archive_path_to_hdf5_path(archive_path: str) -> str:
    """Translate a serialization archive path to an NDF HDF5 path.

    The empty path maps to the main JSON tree at ``/MORE/LSST/JSON``.
    Any non-empty path is uppercased and its separators are replaced with
    ``_`` to form a single component name under ``/MORE/LSST/``. Mirrors
    the FITS archive's ``EXTNAME`` convention.
    """
    if not archive_path:
        return "/MORE/LSST/JSON"
    flattened = archive_path.lstrip("/").upper().replace("/", "_")
    return f"/MORE/LSST/{flattened}"

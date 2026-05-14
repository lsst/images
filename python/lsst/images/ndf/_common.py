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

__all__ = ("NdfPointerModel", "archive_path_to_hdf5_path", "archive_path_to_hdf5_path_components")

import pydantic


class NdfPointerModel(pydantic.BaseModel):
    """Reference to an NDF-archive sub-tree by HDF5 path.

    Used by `NdfOutputArchive`/`NdfInputArchive` to point to
    sub-trees that have been hoisted out of the main JSON tree into separate
    HDS components.
    """

    path: str
    """HDF5 absolute path (e.g. ``/MORE/LSST/PSF``)."""


def archive_path_to_hdf5_path(archive_path: str) -> str:
    """Translate a serialization archive path to an NDF HDF5 path.

    The empty path maps to the main JSON tree at ``/MORE/LSST/JSON``.
    Any non-empty path is uppercased and kept hierarchical under
    ``/MORE/LSST/``. This mirrors the serialization path while keeping HDS
    component names within their 16-character limit.
    """
    if not archive_path:
        return "/MORE/LSST/JSON"
    components = archive_path_to_hdf5_path_components(archive_path)
    return "/MORE/LSST/" + "/".join(components)


def archive_path_to_hdf5_path_components(archive_path: str) -> list[str]:
    """Return HDS-compatible path components for an archive path."""
    components = [component.upper() for component in archive_path.strip("/").split("/") if component]
    for component in components:
        if len(component) > 16:
            raise ValueError(
                f"NDF/HDS component {component!r} from archive path {archive_path!r} "
                "is longer than the 16-character HDS limit."
            )
    return components

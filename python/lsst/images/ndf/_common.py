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
    "NdfPointerModel",
    "archive_path_to_hdf5_path",
    "archive_path_to_hdf5_path_components",
    "shrink_versioned_component",
)

import hashlib

import pydantic

from ._hds import DAT__SZNAM


class NdfPointerModel(pydantic.BaseModel):
    """Reference to an NDF-archive sub-tree by HDF5 path.

    Used by `NdfOutputArchive`/`NdfInputArchive` to point to
    sub-trees that have been hoisted out of the main JSON tree into separate
    HDS components.
    """

    path: str
    """HDF5 absolute path (e.g. ``/MORE/LSST/PSF``)."""


def _shrink_hds_name(name: str, max_length: int = DAT__SZNAM, hash_size: int = 4) -> str:
    """Shrink an HDS component name to fit the HDS length limit.

    The name is uppercased.
    Names at or under ``max_length`` are returned unchanged.
    Longer names are replaced by a readable prefix and an
    underscore-separated `blake2b` digest of the full uppercased name, so the
    result is exactly ``max_length`` characters and distinct inputs almost
    never collide.
    ``hash_size`` is the digest length in bytes; it occupies
    ``hash_size * 2 + 1`` characters of the result.
    """
    name = name.upper()
    if len(name) <= max_length:
        return name
    digest = hashlib.blake2b(name.encode("ascii"), digest_size=hash_size).hexdigest().upper()
    trunc = max_length - 2 * hash_size - 1
    shrunk = f"{name[:trunc]}_{digest}"
    assert len(shrunk) == max_length
    return shrunk


def shrink_versioned_component(
    base: str, version: int, max_length: int = DAT__SZNAM, hash_size: int = 4
) -> str:
    """Shrink a component while preserving a visible version suffix.

    ``version`` is the 0-based occurrence count of the name.
    Occurrences after the first reserve a 1-based ``_{version + 1}`` suffix at
    the tail (the second occurrence of ``DATA`` becomes ``DATA_2``) and the
    ``base`` is shrunk into the remaining characters, so the version number
    stays readable in Starlink tools.
    Version zero (the first occurrence) is shrunk exactly like an unversioned
    component.
    """
    suffix = f"_{version + 1}" if version > 0 else ""
    return _shrink_hds_name(base, max_length - len(suffix), hash_size) + suffix


def archive_path_to_hdf5_path(archive_path: str) -> str:
    """Translate a serialization archive path to an NDF HDF5 path.

    The empty path maps to the main JSON tree at ``/MORE/LSST/JSON``.
    Any non-empty path is uppercased and kept hierarchical under
    ``/MORE/LSST/``. This mirrors the serialization path while keeping HDS
    component names within the HDS object-name limit (``DAT__SZNAM``).
    """
    if not archive_path:
        return "/MORE/LSST/JSON"
    components = archive_path_to_hdf5_path_components(archive_path)
    return "/MORE/LSST/" + "/".join(components)


def archive_path_to_hdf5_path_components(archive_path: str) -> list[str]:
    """Return HDS-compatible path components for an archive path.

    Each component is uppercased; components longer than the HDS object-name
    limit (``DAT__SZNAM``) are deterministically shrunk by `_shrink_hds_name`.
    """
    return [_shrink_hds_name(component) for component in archive_path.strip("/").split("/") if component]

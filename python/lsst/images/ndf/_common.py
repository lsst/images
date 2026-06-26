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
    "HdsNameShrinker",
    "NdfPointerModel",
    "archive_path_to_hdf5_path",
    "archive_path_to_hdf5_path_components",
)

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


class HdsNameShrinker:
    """Shrink HDS component names to fit the HDS object-name length limit.

    Names are uppercased; names at or under the limit pass through unchanged.
    Each distinct over-long name is assigned a readable prefix followed by an
    underscore-separated hexadecimal counter that increments per assignment,
    so distinct names written through the same shrinker never collide.
    Assignments are remembered, so repeated requests for the same name return
    the same result.

    A shrinker is scoped to a single output file: shrunk names are not stable
    across files, so readers must use the paths recorded in the file's JSON
    tree rather than recomputing them.

    Parameters
    ----------
    max_length
        Maximum component length, by default the HDS limit (``DAT__SZNAM``).
    """

    def __init__(self, max_length: int = DAT__SZNAM) -> None:
        self._max_length = max_length
        self._assigned: dict[tuple[str, int], str] = {}
        self._counter = 0

    def shrink(self, name: str, reserve: int = 0) -> str:
        """Shrink a component name to fit in ``max_length - reserve``.

        Parameters
        ----------
        name
            The component name to shrink.
        reserve
            Number of characters to leave available for a suffix the caller
            will append (e.g. a version suffix).

        Returns
        -------
        `str`
            The uppercased name, unchanged if it fits, otherwise truncated
            and suffixed with ``_`` and an uppercase hexadecimal counter
            (at least three digits) so the result is exactly
            ``max_length - reserve`` characters.
        """
        max_length = self._max_length - reserve
        name = name.upper()
        if len(name) <= max_length:
            return name
        key = (name, max_length)
        if (shrunk := self._assigned.get(key)) is None:
            self._counter += 1
            token = f"_{self._counter:03X}"
            shrunk = f"{name[: max_length - len(token)]}{token}"
            self._assigned[key] = shrunk
        return shrunk

    def shrink_versioned(self, base: str, version: int) -> str:
        """Shrink a component while preserving a visible version suffix.

        When ``version`` is greater than one a ``_{version}`` suffix is
        reserved at the tail and the ``base`` is shrunk into the remaining
        characters, so the version number stays readable in Starlink tools.
        Version one (the first occurrence) is shrunk exactly like an
        unversioned component.

        Parameters
        ----------
        base
            Component name to shrink.
        version
            Version number whose suffix is preserved when greater than one.
        """
        suffix = f"_{version}" if version > 1 else ""
        return self.shrink(base, reserve=len(suffix)) + suffix


def archive_path_to_hdf5_path(archive_path: str, shrinker: HdsNameShrinker) -> str:
    """Translate a serialization archive path to an NDF HDF5 path.

    The empty path maps to the main JSON tree at ``/MORE/LSST/JSON``.
    Any non-empty path is uppercased and kept hierarchical under
    ``/MORE/LSST/``. This mirrors the serialization path while keeping HDS
    component names within the HDS object-name limit (``DAT__SZNAM``).

    Parameters
    ----------
    archive_path
        Serialization archive path to translate.
    shrinker
        Name shrinker used to keep components within the HDS name limit.
    """
    if not archive_path:
        return "/MORE/LSST/JSON"
    components = archive_path_to_hdf5_path_components(archive_path, shrinker)
    return "/MORE/LSST/" + "/".join(components)


def archive_path_to_hdf5_path_components(archive_path: str, shrinker: HdsNameShrinker) -> list[str]:
    """Return HDS-compatible path components for an archive path.

    Each component is uppercased; components longer than the HDS object-name
    limit (``DAT__SZNAM``) are shrunk by ``shrinker``.

    Parameters
    ----------
    archive_path
        Serialization archive path to split into components.
    shrinker
        Name shrinker used to keep components within the HDS name limit.
    """
    return [shrinker.shrink(component) for component in archive_path.strip("/").split("/") if component]

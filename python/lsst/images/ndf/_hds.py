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

"""HDS-on-HDF5 read/write helpers.

These follow the conventions used by the canonical Starlink ``hds-v5``
library (see ``reference/hds-v5/dat1.h`` and ``dat1New.c``):

* HDS structures are HDF5 groups with a ``CLASS`` attribute holding the
  HDS type string (``"NDF"``, ``"WCS"``, ``"EXT"``, ``"ARRAY"``, ...).
* Arrays of structures additionally carry an ``HDS_STRUCTURE_DIMS``
  attribute (deferred from v1; we only handle scalar structures).
* The file root group, when it represents a top-level HDS structure,
  carries an ``HDS_ROOT_NAME`` attribute giving the HDS object name.
* HDS primitives are bare HDF5 datasets with no HDS-specific attributes;
  the HDS type is inferred from the HDF5 dtype.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import h5py
import numpy as np

__all__ = (
    "ATTR_CLASS",
    "ATTR_ROOT_NAME",
    "ATTR_STRUCTURE_DIMS",
    "HDS_TO_NUMPY",
    "NUMPY_TO_HDS",
    "create_structure",
    "hds_type_for_dtype",
    "iter_children",
    "open_structure",
    "read_array",
    "read_char_array",
    "set_root_name",
    "write_array",
    "write_char_array",
)


# Canonical attribute names used by hds-v5 (see reference/hds-v5/dat1.h).
ATTR_CLASS = "CLASS"
ATTR_STRUCTURE_DIMS = "HDS_STRUCTURE_DIMS"
ATTR_ROOT_NAME = "HDS_ROOT_NAME"


HDS_TO_NUMPY: dict[str, np.dtype] = {
    "_REAL": np.dtype(np.float32),
    "_DOUBLE": np.dtype(np.float64),
    "_UBYTE": np.dtype(np.uint8),
    "_WORD": np.dtype(np.int16),
    "_INTEGER": np.dtype(np.int32),
    "_INT64": np.dtype(np.int64),
}

NUMPY_TO_HDS: dict[np.dtype, str] = {
    np.dtype(np.float32): "_REAL",
    np.dtype(np.float64): "_DOUBLE",
    np.dtype(np.uint8): "_UBYTE",
    np.dtype(np.int16): "_WORD",
    np.dtype(np.int32): "_INTEGER",
    np.dtype(np.int64): "_INT64",
}


def hds_type_for_dtype(dtype: np.dtype) -> str:
    """Return the HDS type string for a numpy dtype.

    Fixed-width byte strings ``|S<N>`` map to ``"_CHAR*<N>"``. Numeric
    dtypes are looked up in `NUMPY_TO_HDS`. Anything else raises
    ``NotImplementedError``.
    """
    if dtype.kind == "S":
        return f"_CHAR*{dtype.itemsize}"
    try:
        return NUMPY_TO_HDS[dtype]
    except KeyError:
        raise NotImplementedError(f"No HDS type mapping for dtype {dtype!r}.") from None


def write_array(
    parent: h5py.Group,
    name: str,
    data: np.ndarray,
    *,
    compression: str | None = None,
) -> h5py.Dataset:
    """Write a numpy C-order array as an HDS primitive.

    The HDF5 dataset carries no HDS-specific attributes; the HDS type
    is inferred on read from the HDF5 dtype. Refuses dtypes that don't
    map to a supported HDS primitive type.

    The HDF5 dataset has the array's natural shape (C-order). Combined
    with HDF5's native byte ordering, this matches the Fortran-on-disk
    layout required by HDS for an NDF whose Fortran-order shape is the
    reverse of ``data.shape``.
    """
    # Validate the dtype is supported up front so callers get a clear error.
    hds_type_for_dtype(data.dtype)
    kwargs: dict = {}
    if compression is not None:
        kwargs["compression"] = compression
    return parent.create_dataset(name, data=data, **kwargs)


def read_array(dataset: h5py.Dataset) -> np.ndarray:
    """Read an HDS primitive into a C-order numpy array.

    The HDS type is inferred from the HDF5 dtype. Raises
    ``NotImplementedError`` if the dtype is not a supported numeric HDS
    primitive type. Use `read_char_array` for ``_CHAR*N`` datasets.
    """
    if dataset.dtype.kind == "S":
        raise ValueError(f"Dataset {dataset.name!r} is _CHAR*N; use read_char_array instead.")
    if dataset.dtype not in NUMPY_TO_HDS:
        raise NotImplementedError(
            f"Dataset {dataset.name!r} has dtype {dataset.dtype} which does not "
            f"map to a supported HDS primitive type."
        )
    return dataset[()]


def write_char_array(
    parent: h5py.Group,
    name: str,
    lines: Sequence[str],
    *,
    width: int = 80,
) -> h5py.Dataset:
    """Write a sequence of strings as a 1D HDS ``_CHAR*N`` primitive.

    Each string is padded to ``width`` with trailing spaces (HDS
    convention) and truncated if longer. The HDF5 dataset has dtype
    ``|S<width>``; no HDS-specific attributes are written.
    """
    encoded = np.array(
        [line.encode("ascii", errors="replace").ljust(width)[:width] for line in lines],
        dtype=f"|S{width}",
    )
    return parent.create_dataset(name, data=encoded)


def read_char_array(dataset: h5py.Dataset) -> list[str]:
    """Read an HDS ``_CHAR*N`` 1D primitive as a list of stripped strings.

    Validates the dataset has a fixed-width byte-string dtype (``|S<N>``).
    """
    if dataset.dtype.kind != "S":
        raise ValueError(f"Dataset {dataset.name!r} is not _CHAR*N (dtype {dataset.dtype}).")
    raw = dataset[()]
    return [item.decode("ascii").rstrip(" ") for item in raw]


def create_structure(parent: h5py.Group, name: str, hds_type: str) -> h5py.Group:
    """Create a named HDS structure (h5py group with ``CLASS`` attribute).

    Parameters
    ----------
    parent
        Group to create the new structure under.
    name
        Component name (HDS rules apply: uppercase letters/digits/underscores,
        max 15 characters; not enforced here).
    hds_type
        HDS type string for the new structure (e.g. ``"NDF"``, ``"WCS"``,
        ``"ARRAY"``, ``"EXT"``).
    """
    group = parent.create_group(name)
    group.attrs[ATTR_CLASS] = hds_type
    return group


def set_root_name(file: h5py.File, hds_name: str, hds_type: str) -> None:
    """Mark a file's root group as a top-level HDS structure.

    Sets ``HDS_ROOT_NAME`` (the HDS object name) and ``CLASS`` (the HDS
    type) on the root group, matching what ``hds-v5`` writes for a root
    structure created via :c:func:`dat1New`.
    """
    file["/"].attrs[ATTR_ROOT_NAME] = hds_name
    file["/"].attrs[ATTR_CLASS] = hds_type


def open_structure(parent: h5py.Group, name: str) -> tuple[h5py.Group, str]:
    """Open a child structure by name. Returns ``(group, hds_type)``.

    Raises ``ValueError`` if the child is not a group, or has no
    ``CLASS`` attribute. Accepts the legacy ``HDSTYPE`` attribute name
    as a fallback so files written by older HDS variants can still be
    inspected.
    """
    obj = parent[name]
    if not isinstance(obj, h5py.Group):
        raise ValueError(f"{parent.name}/{name} is a dataset, not a structure.")
    hds_type = obj.attrs.get(ATTR_CLASS)
    if hds_type is None:
        # Legacy fallback for older HDS-on-HDF5 variants.
        hds_type = obj.attrs.get("HDSTYPE")
    if isinstance(hds_type, bytes):
        hds_type = hds_type.decode("ascii")
    if not isinstance(hds_type, str):
        raise ValueError(f"Group {obj.name!r} has no {ATTR_CLASS!r} (or legacy HDSTYPE) attribute.")
    return obj, hds_type


def iter_children(group: h5py.Group) -> Iterator[tuple[str, h5py.Group | h5py.Dataset]]:
    """Iterate over a structure's direct children.

    Yields ``(name, child)`` pairs where ``child`` is a group or dataset.
    """
    yield from group.items()

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
from typing import Any

import h5py
import numpy as np

__all__ = (
    "ATTR_CLASS",
    "ATTR_ROOT_NAME",
    "ATTR_STRUCTURE_DIMS",
    "HDS_TO_NUMPY",
    "NUMPY_TO_HDS",
    "create_structure",
    "decode_ndf_ast_data",
    "encode_ndf_ast_data",
    "hds_type_for_dtype",
    "iter_children",
    "open_structure",
    "read_array",
    "read_char_array",
    "set_ascii_attr",
    "set_root_name",
    "write_array",
    "write_char_array",
)


# Canonical attribute names used by hds-v5 (see reference/hds-v5/dat1.h).
ATTR_CLASS = "CLASS"
ATTR_STRUCTURE_DIMS = "HDS_STRUCTURE_DIMS"
ATTR_ROOT_NAME = "HDS_ROOT_NAME"


HDS_TO_NUMPY: dict[str, np.dtype] = {
    "_LOGICAL": np.dtype(np.bool_),
    "_REAL": np.dtype(np.float32),
    "_DOUBLE": np.dtype(np.float64),
    "_UBYTE": np.dtype(np.uint8),
    "_WORD": np.dtype(np.int16),
    "_INTEGER": np.dtype(np.int32),
    "_INT64": np.dtype(np.int64),
}

NUMPY_TO_HDS: dict[np.dtype, str] = {
    np.dtype(np.bool_): "_LOGICAL",
    np.dtype(np.float32): "_REAL",
    np.dtype(np.float64): "_DOUBLE",
    np.dtype(np.uint8): "_UBYTE",
    np.dtype(np.int16): "_WORD",
    np.dtype(np.int32): "_INTEGER",
    np.dtype(np.int64): "_INT64",
}


NDF_AST_DATA_WIDTH = 32
NDF_AST_DATA_MIN_WIDTH = 16

# HDS object-name length limit, from the Starlink DAT_PAR include
# (``dat_par.h``): ``DAT__SZNAM 15`` ("Size of object name").
# The sibling limits ``DAT__SZGRP`` (group name) and ``DAT__SZTYP`` (type
# string) are also 15 today; only the object-name limit is enforced here,
# because HDS type tags are derived from already-shrunk component names.
DAT__SZNAM = 15


def hds_type_for_dtype(dtype: np.dtype) -> str:
    """Return the HDS type string for a numpy dtype.

    Parameters
    ----------
    dtype
        Numpy dtype to map to an HDS primitive type.

    Returns
    -------
    str
        HDS primitive type string.

    Raises
    ------
    NotImplementedError
        Raised if ``dtype`` does not map to a supported HDS primitive type.

    Notes
    -----
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
    compression_opts: int | None = None,
) -> h5py.Dataset:
    """Write a numpy C-order array as an HDS primitive.

    Parameters
    ----------
    parent
        HDF5 group to receive the dataset.
    name
        Name of the primitive dataset to create.
    data
        Array to write.
    compression
        Optional compression algorithm accepted by `h5py.Group.create_dataset`.
    compression_opts
        Optional compression options accepted by `h5py.Group.create_dataset`.

    Returns
    -------
    h5py.Dataset
        Newly-created dataset.

    Notes
    -----
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
    if data.dtype == np.dtype(np.bool_):
        return _write_logical_array(parent, name, data, compression=compression)
    # h5py rejects compression options if no compression algorithm is set.
    kwargs: dict[str, Any] = {}
    if compression is not None:
        kwargs["compression"] = compression
    if compression_opts is not None:
        kwargs["compression_opts"] = compression_opts
    return parent.create_dataset(name, data=data, **kwargs)


def _write_logical_array(
    parent: h5py.Group,
    name: str,
    data: np.ndarray,
    *,
    compression: str | None = None,
) -> h5py.Dataset:
    """Write an HDS ``_LOGICAL`` primitive using the HDF5 bitfield type.

    Parameters
    ----------
    parent
        HDF5 group to receive the dataset.
    name
        Name of the primitive dataset to create.
    data
        Boolean array to write.
    compression
        Compression is not supported for this low-level bitfield writer.

    Returns
    -------
    h5py.Dataset
        Newly-created dataset.

    Notes
    -----
    High-level h5py writes numpy bool data as an HDF5 enum, but hds-v5
    identifies ``_LOGICAL`` primitives by the HDF5 bitfield class.
    """
    if compression is not None:
        raise NotImplementedError("Compression is not implemented for HDS _LOGICAL arrays.")
    logical_data = np.asarray(data, dtype=np.uint8)
    if logical_data.shape:
        space = h5py.h5s.create_simple(logical_data.shape)
    else:
        space = h5py.h5s.create(h5py.h5s.SCALAR)
    dataset_id = h5py.h5d.create(
        parent.id,
        name.encode("ascii"),
        h5py.h5t.STD_B8LE,
        space,
    )
    dataset_id.write(
        h5py.h5s.ALL,
        h5py.h5s.ALL,
        logical_data,
        mtype=h5py.h5t.NATIVE_B8,
    )
    dataset_id.close()
    return parent[name]


def read_array(dataset: h5py.Dataset) -> np.ndarray:
    """Read an HDS primitive into a C-order numpy array.

    Parameters
    ----------
    dataset
        HDF5 dataset to read.

    Returns
    -------
    numpy.ndarray
        Data read from ``dataset``.

    Notes
    -----
    The HDS type is inferred from the HDF5 dtype. Raises
    ``NotImplementedError`` if the dtype is not a supported numeric HDS
    primitive type. Use `read_char_array` for ``_CHAR*N`` datasets.
    """
    if dataset.dtype.kind == "S":
        raise ValueError(f"Dataset {dataset.name!r} is _CHAR*N; use read_char_array instead.")
    dataset_type = dataset.id.get_type()
    if dataset_type.get_class() == h5py.h5t.BITFIELD:
        if dataset_type.get_size() not in {1, 4}:
            raise NotImplementedError(
                f"Dataset {dataset.name!r} has bitfield size {dataset_type.get_size()} "
                "which does not map to HDS _LOGICAL."
            )
        data = dataset[()] != 0
        if isinstance(data, np.ndarray):
            return data.astype(np.bool_)
        return np.atleast_1d(np.bool_(data))
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

    Parameters
    ----------
    parent
        HDF5 group to receive the dataset.
    name
        Name of the primitive dataset to create.
    lines
        Strings to write. Each string must be ASCII and no longer than
        ``width`` characters.
    width
        Fixed width of each HDS character-array element.

    Returns
    -------
    h5py.Dataset
        Newly-created dataset.

    Raises
    ------
    ValueError
        Raised if any string is not ASCII or is longer than ``width``.

    Notes
    -----
    Each string is padded to ``width`` with trailing spaces (HDS
    convention). The HDF5 dataset has dtype ``|S<width>``; no HDS-specific
    attributes are written.
    """
    encoded_lines: list[bytes] = []
    for n, line in enumerate(lines):
        try:
            encoded_line = line.encode("ascii")
        except UnicodeEncodeError as err:
            raise ValueError(f"Line {n} for {name!r} is not ASCII.") from err
        if len(encoded_line) > width:
            raise ValueError(
                f"Line {n} for {name!r} is {len(encoded_line)} bytes, longer than width={width}."
            )
        encoded_lines.append(encoded_line.ljust(width))
    encoded = np.array(
        encoded_lines,
        dtype=f"|S{width}",
    )
    return parent.create_dataset(name, data=encoded)


def encode_ndf_ast_data(text: str, *, width: int = NDF_AST_DATA_WIDTH) -> list[str]:
    """Encode AST Channel text for an NDF ``WCS.DATA`` component.

    Parameters
    ----------
    text
        Text emitted by an AST Channel.
    width
        Fixed width of the target NDF ``WCS.DATA`` records.

    Returns
    -------
    list [str]
        HDS character-array records.

    Notes
    -----
    Starlink NDF stores each AST text line in one or more fixed-width
    ``_CHAR*32`` records. The first character of each record is a flag:
    a space starts a new AST line and ``+`` continues the previous one.
    The payload is the AST text line with leading indentation removed.
    """
    if width < NDF_AST_DATA_MIN_WIDTH:
        raise ValueError(
            f"NDF AST DATA record width {width} is too short; minimum is {NDF_AST_DATA_MIN_WIDTH}."
        )

    records: list[str] = []
    payload_width = width - 1
    for raw_line in text.splitlines():
        line = raw_line.lstrip(" ").rstrip(" ")
        if not line:
            continue
        for start in range(0, len(line), payload_width):
            flag = " " if start == 0 else "+"
            records.append(f"{flag}{line[start : start + payload_width]}")
    return records


def decode_ndf_ast_data(records: Sequence[str]) -> str:
    """Decode an NDF ``WCS.DATA`` component into AST Channel text.

    Parameters
    ----------
    records
        HDS character-array records from ``WCS.DATA``.

    Returns
    -------
    str
        AST Channel text.

    Notes
    -----
    This reverses `encode_ndf_ast_data`. If the input does not look like
    NDF AST records, it is treated as plain AST Channel text for backward
    compatibility with earlier non-canonical files.
    """
    if not records:
        return ""
    if any(record and record[0] not in {" ", "+"} for record in records):
        return "\n".join(records) + "\n"

    lines: list[str] = []
    current: list[str] = []
    for record in records:
        if not record:
            continue
        flag = record[0]
        payload = record[1:]
        if flag == "+":
            if current:
                current.append(payload)
            else:
                current = [payload]
        else:
            if current:
                lines.append("".join(current).rstrip(" "))
            current = [payload]
    if current:
        lines.append("".join(current).rstrip(" "))
    return "\n".join(lines) + ("\n" if lines else "")


def read_char_array(dataset: h5py.Dataset) -> list[str]:
    """Read an HDS ``_CHAR*N`` 1D primitive as a list of stripped strings.

    Parameters
    ----------
    dataset
        HDF5 dataset to read.

    Returns
    -------
    list [str]
        Dataset elements decoded from ASCII with trailing spaces stripped.

    Notes
    -----
    Validates the dataset has a fixed-width byte-string dtype (``|S<N>``).
    """
    if dataset.dtype.kind != "S":
        raise ValueError(f"Dataset {dataset.name!r} is not _CHAR*N (dtype {dataset.dtype}).")
    if dataset.ndim == 0:
        raise ValueError(f"Dataset {dataset.name!r} is a scalar _CHAR*N; only 1-D arrays are supported.")
    raw = dataset[()]
    return [item.decode("ascii").rstrip(" ") for item in raw]


def set_ascii_attr(target: h5py.Group | h5py.Dataset, name: str, value: str) -> None:
    """Write a fixed-length ASCII byte attribute.

    Parameters
    ----------
    target
        HDF5 object whose attribute should be set.
    name
        Attribute name.
    value
        ASCII value to write.

    Notes
    -----
    Canonical ``hds-v5`` stores ``CLASS`` and ``HDS_ROOT_NAME`` as
    fixed-length ASCII byte strings (e.g. ``|S5`` for ``"ARRAY"``).
    h5py's default for Python ``str`` is variable-length UTF-8, which
    Starlink tools (KAPPA, ``hdstrace``) can't decode — they show
    garbage in the type-tag column. Writing as fixed-length bytes
    matches the canonical layout.
    """
    encoded = value.encode("ascii")
    if name in target.attrs:
        del target.attrs[name]
    target.attrs.create(name, encoded, dtype=f"|S{len(encoded)}")


def create_structure(parent: h5py.Group, name: str, hds_type: str) -> h5py.Group:
    """Create a named HDS structure (h5py group with ``CLASS`` attribute).

    Parameters
    ----------
    parent
        Group to create the new structure under.
    name
        Component name (HDS rules apply: uppercase letters/digits/underscores,
        max ``DAT__SZNAM`` characters; not enforced here).
    hds_type
        HDS type string for the new structure (e.g. ``"NDF"``, ``"WCS"``,
        ``"ARRAY"``, ``"EXT"``).
    """
    group = parent.create_group(name)
    set_ascii_attr(group, ATTR_CLASS, hds_type)
    return group


def set_root_name(file: h5py.File, hds_name: str, hds_type: str) -> None:
    """Mark a file's root group as a top-level HDS structure.

    Parameters
    ----------
    file
        HDF5 file whose root group represents the HDS root object.
    hds_name
        HDS root object name.
    hds_type
        HDS type string for the root object.

    Notes
    -----
    Sets ``HDS_ROOT_NAME`` (the HDS object name) and ``CLASS`` (the HDS
    type) on the root group, matching what ``hds-v5`` writes for a root
    structure created via :c:func:`dat1New`.
    """
    set_ascii_attr(file["/"], ATTR_ROOT_NAME, hds_name)
    set_ascii_attr(file["/"], ATTR_CLASS, hds_type)


def open_structure(parent: h5py.Group, name: str) -> tuple[h5py.Group, str]:
    """Open a child structure by name. Returns ``(group, hds_type)``.

    Parameters
    ----------
    parent
        HDF5 group containing the structure.
    name
        Name of the child structure.

    Returns
    -------
    group : h5py.Group
        Opened HDF5 group.
    hds_type : str
        HDS type string from the structure's ``CLASS`` attribute.

    Notes
    -----
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

    Parameters
    ----------
    group
        HDF5 group to inspect.

    Yields
    ------
    name
        Name of the structure.
    child
        Child is a group or dataset.
    """
    yield from group.items()

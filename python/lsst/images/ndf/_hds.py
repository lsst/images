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

"""HDS-on-HDF5 read/write helpers."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING

import h5py
import numpy as np

__all__ = (
    "create_structure",
    "open_structure",
    "iter_children",
    "write_array",
    "read_array",
    "write_char_array",
    "read_char_array",
    "HDS_TO_NUMPY",
    "NUMPY_TO_HDS",
)


HDS_TO_NUMPY: dict[str, np.dtype] = {
    "_REAL": np.dtype(np.float32),
    "_DOUBLE": np.dtype(np.float64),
    "_UBYTE": np.dtype(np.uint8),
    "_INTEGER": np.dtype(np.int32),
    "_WORD": np.dtype(np.int16),
}

NUMPY_TO_HDS: dict[np.dtype, str] = {
    np.dtype(np.float32): "_REAL",
    np.dtype(np.float64): "_DOUBLE",
    np.dtype(np.uint8): "_UBYTE",
    np.dtype(np.int32): "_INTEGER",
}


def write_array(
    parent: h5py.Group,
    name: str,
    data: np.ndarray,
    *,
    hdstype: str | None = None,
    compression: str | None = None,
) -> h5py.Dataset:
    """Write a numpy C-order array as an HDS primitive.

    The HDF5 dataset has the array's natural shape (C-order). Combined with
    HDF5's native byte ordering, this matches the Fortran-on-disk layout
    required by HDS for an NDF whose Fortran-order shape is the reverse of
    ``data.shape``.
    """
    if hdstype is None:
        try:
            hdstype = NUMPY_TO_HDS[data.dtype]
        except KeyError:
            raise NotImplementedError(f"No HDS write support for dtype {data.dtype!r}.") from None
    kwargs: dict = {}
    if compression is not None:
        kwargs["compression"] = compression
    ds = parent.create_dataset(name, data=data, **kwargs)
    ds.attrs["HDSTYPE"] = hdstype
    ds.attrs["HDSNDIMS"] = data.ndim
    ds.attrs["HDS_DATASET_IS_DEFINED"] = True
    return ds


def read_array(dataset: h5py.Dataset) -> np.ndarray:
    """Read an HDS primitive into a C-order numpy array.

    Validates ``HDSTYPE`` is in the supported set and that ``HDSNDIMS``
    matches the dataset's HDF5 ndim.
    """
    hdstype = dataset.attrs.get("HDSTYPE")
    if not isinstance(hdstype, (bytes, str)):
        raise ValueError(f"Dataset {dataset.name!r} has no HDSTYPE attribute.")
    if isinstance(hdstype, bytes):
        hdstype = hdstype.decode("ascii")
    if hdstype.startswith("_CHAR"):
        raise ValueError(f"Use read_char_array for _CHAR primitives at {dataset.name!r}.")
    if hdstype not in HDS_TO_NUMPY:
        raise NotImplementedError(f"HDS type {hdstype!r} not supported for read.")
    expected_dtype = HDS_TO_NUMPY[hdstype]
    if dataset.dtype != expected_dtype:
        raise ValueError(
            f"Dataset {dataset.name!r} has HDF5 dtype {dataset.dtype} "
            f"but HDSTYPE {hdstype!r} expects {expected_dtype}."
        )
    return dataset[()]

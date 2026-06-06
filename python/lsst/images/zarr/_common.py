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
    "DEFAULT_CHUNK_AXIS_LIMIT",
    "DEFAULT_TARGET_SHARD_BYTES",
    "LSST_NS",
    "LSST_VERSION",
    "OME_NS",
    "OME_VERSION",
    "ZarrCompressionOptions",
    "ZarrPointerModel",
    "archive_path_to_zarr_path",
    "mask_dtype_for_plane_count",
)

import os
from dataclasses import dataclass
from typing import ClassVar, Self

import numpy as np
import pydantic

LSST_NS = "lsst"
"""Top-level zarr-attributes namespace key for LSST extensions."""

OME_NS = "ome"
"""Top-level zarr-attributes namespace key for OME-NGFF metadata."""

OME_VERSION = "0.5"
"""OME-Zarr / NGFF version this backend writes."""

LSST_VERSION = 1
"""Container (file-format) version this backend writes, emitted as the
``lsst.version`` root-group attribute.

Bumps when the zarr group and attribute layout changes, independent of any
data-model ``SCHEMA_VERSION``. Readers refuse a newer on-disk container
version than they understand (see
:func:`lsst.images.serialization._common._check_format_version`), and treat
its absence as ``1``. See :ref:`lsst.images-schema-versioning`.
"""

DEFAULT_CHUNK_AXIS_LIMIT = 256
"""Per-axis cap on the auto-derived chunk shape for plain image arrays.

Used by `lsst.images.zarr._layout.chunks_for` when the caller does not
supply an explicit override and the archive class does not have a
class-specific chunk rule. Chunks of ~256 elements per spatial axis
trade some compression ratio for cutout-friendly partial reads.
"""


def _read_target_shard_bytes() -> int:
    """Read ``LSST_IMAGES_ZARR_TARGET_SHARD_BYTES`` or return the default.

    Parsed as a base-10 integer. A non-integer value raises ``ValueError``
    at import time — silent typos are worse than loud failure.
    """
    raw = os.environ.get("LSST_IMAGES_ZARR_TARGET_SHARD_BYTES")
    if raw is None:
        return 16 * 1024 * 1024
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"LSST_IMAGES_ZARR_TARGET_SHARD_BYTES={raw!r} is not a base-10 integer.") from exc


DEFAULT_TARGET_SHARD_BYTES: int = _read_target_shard_bytes()
"""Target uncompressed byte size for an auto-derived shard.

Read from ``LSST_IMAGES_ZARR_TARGET_SHARD_BYTES`` once at import time;
defaults to 16 MiB. Used by `lsst.images.zarr._layout.default_shards` to
decide how many chunks to combine into a shard.
"""


class ZarrPointerModel(pydantic.BaseModel):
    """Reference to a zarr archive sub-tree by absolute zarr path.

    Used by `ZarrOutputArchive` / `ZarrInputArchive` to point to
    sub-trees that have been hoisted out of the main JSON tree into
    separate zarr arrays. The path is interpreted relative to the
    archive root, e.g. ``"/lsst/psf/lsst_json"``.
    """

    path: str
    """Absolute zarr path (e.g. ``/lsst/psf/lsst_json``)."""


@dataclass(frozen=True)
class ZarrCompressionOptions:
    """Per-array zarr v3 codec configuration.

    The default codec stack is ``bytes -> blosc(zstd, clevel=5)`` with
    byte-shuffle for floats and bit-shuffle for integers (and masks).
    All defaults are overridable per-array via the ``compression``
    keyword to ``write()``.
    """

    codec: str = "blosc"
    cname: str = "zstd"
    clevel: int = 5
    shuffle: str = "shuffle"  # 'shuffle' (byte) or 'bitshuffle' or 'noshuffle'

    DEFAULT_FLOAT: ClassVar[Self]
    DEFAULT_INT: ClassVar[Self]

    @classmethod
    def default_for_dtype(cls, dtype: str | np.dtype) -> Self:
        """Return the default codec stack for a numpy dtype."""
        kind = np.dtype(dtype).kind
        # 'u' (unsigned int), 'i' (signed int), 'b' (bool) -> bit-shuffle.
        if kind in ("u", "i", "b"):
            return cls.DEFAULT_INT
        return cls.DEFAULT_FLOAT


ZarrCompressionOptions.DEFAULT_FLOAT = ZarrCompressionOptions(shuffle="shuffle")
ZarrCompressionOptions.DEFAULT_INT = ZarrCompressionOptions(shuffle="bitshuffle")


def archive_path_to_zarr_path(archive_path: str) -> str:
    """Translate a serialization archive path to its zarr path.

    The empty archive path maps to the root-level JSON tree at
    ``/lsst_json``. Non-empty archive paths are kept verbatim (with a
    leading slash). The v1 design's JSON-pointer mapping table is
    intentionally absent: arrays land where their archive name says
    they do.
    """
    if not archive_path:
        return "/lsst_json"
    stripped = archive_path.strip("/")
    return f"/{stripped}"


def mask_dtype_for_plane_count(n_planes: int) -> np.dtype:
    """Pick the smallest unsigned-integer dtype that holds ``n_planes`` bits.

    Returns ``uint8`` for <=8 planes, ``uint16`` for <=16, ``uint32``
    for <=32, ``uint64`` for <=64. Raises `ValueError` for >64 planes;
    a 3-D fallback for that case is tracked as a follow-up.
    """
    if n_planes <= 0:
        raise ValueError(f"n_planes must be positive, got {n_planes}.")
    if n_planes <= 8:
        return np.dtype("uint8")
    if n_planes <= 16:
        return np.dtype("uint16")
    if n_planes <= 32:
        return np.dtype("uint32")
    if n_planes <= 64:
        return np.dtype("uint64")
    raise ValueError(f"Mask has {n_planes} planes; v1 supports up to 64. 3-D fallback is a follow-up.")

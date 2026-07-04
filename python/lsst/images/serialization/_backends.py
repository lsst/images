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

__all__ = ("Backend", "backend_for_name", "backend_for_path", "backend_for_stream")

import dataclasses
import gzip
import io
import os
from collections.abc import Callable
from typing import IO, TYPE_CHECKING

from lsst.resources import ResourcePath, ResourcePathExpression

if TYPE_CHECKING:
    from typing_extensions import TypeIs

    from ._input_archive import InputArchive

_GZIP_MAGIC = b"\x1f\x8b"
"""Leading bytes of a gzip member."""

_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"
"""Leading bytes of a zstd frame."""

_COMPRESSION_SUFFIXES = (".gz", ".zst")
"""File name suffixes implying whole-file compression."""

_FITS_MAGIC = b"SIMPLE  ="
"""Leading bytes of a standard FITS primary header."""

_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"
"""Leading bytes of an HDF5 file."""


def _is_binary_stream(obj: object) -> TypeIs[IO[bytes]]:
    """Return whether ``obj`` is a readable, seekable binary stream.

    ``seek`` is the discriminator against `~lsst.resources.ResourcePath`,
    which also has a ``read`` method.
    """
    return hasattr(obj, "read") and hasattr(obj, "seek")


def _path_is_compressed(path: ResourcePathExpression) -> bool:
    """Return whether ``path``'s file name ends in a compression suffix.

    Parameters
    ----------
    path
        Path to inspect; convertible to `lsst.resources.ResourcePath`.
    """
    return ResourcePath(path).basename().endswith(_COMPRESSION_SUFFIXES)


def _decompress_zstd(data: bytes) -> bytes:
    """Decompress zstd ``data``, preferring the stdlib decompressor.

    The imports are function-scoped optional-dependency guards:
    ``compression.zstd`` exists only on Python >= 3.14 and ``zstandard``
    is a third-party fallback that is not a required dependency.

    Parameters
    ----------
    data
        Complete zstd-compressed payload.

    Raises
    ------
    ValueError
        If no zstd decompressor is available.
    """
    try:
        from compression import zstd
    except ImportError:
        try:
            import zstandard
        except ImportError:
            raise ValueError(
                "Data is zstd-compressed, but no zstd decompressor is "
                "available; install the 'zstandard' package or use "
                "Python >= 3.14."
            ) from None
        # decompressobj() handles frames that do not record their
        # decompressed size, unlike ZstdDecompressor.decompress().
        return zstandard.ZstdDecompressor().decompressobj().decompress(data)
    return zstd.decompress(data)


def _maybe_decompress_stream(stream: IO[bytes]) -> IO[bytes]:
    """Return ``stream``, or an in-memory decompressed copy of it.

    gzip and zstd content is detected from its magic number and
    decompressed in full: the backends need random access, which
    compressed streams cannot provide.  Anything else is returned
    unchanged, positioned where it was passed in.

    Parameters
    ----------
    stream
        Seekable binary stream to inspect.
    """
    start = stream.tell()
    magic = stream.read(4)
    stream.seek(start)
    if magic.startswith(_GZIP_MAGIC):
        return io.BytesIO(gzip.decompress(stream.read()))
    if magic == _ZSTD_MAGIC:
        return io.BytesIO(_decompress_zstd(stream.read()))
    return stream


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


def _backend_for_format(name: str) -> Backend | None:
    """Return the `Backend` registered under ``name``, or `None`.

    Backends are imported lazily so optional dependencies (e.g. ``h5py``)
    are only required when actually used.
    """
    match name:
        case "fits":
            from ..fits import FitsInputArchive
            from ..fits import write as fits_write

            return Backend("fits", fits_write, FitsInputArchive)
        case "ndf":
            from ..ndf import NdfInputArchive
            from ..ndf import write as ndf_write

            return Backend("ndf", ndf_write, NdfInputArchive)
        case "json":
            from ..json import JsonInputArchive
            from ..json import write as json_write

            return Backend("json", json_write, JsonInputArchive)
    return None


def backend_for_name(name: str) -> Backend:
    """Return the `Backend` with the given format name.

    Parameters
    ----------
    name
        Backend format name: ``"fits"``, ``"ndf"``, or ``"json"``.

    Raises
    ------
    ValueError
        If ``name`` is not a recognized backend name.
    """
    backend = _backend_for_format(name)
    if backend is None:
        raise ValueError(f"Unrecognized format name: {name!r}; expected one of 'fits', 'ndf', 'json'.")
    return backend


def backend_for_path(path: ResourcePathExpression) -> Backend:
    """Return the `Backend` for ``path`` based on its file extension.

    Supported extensions: ``.fits`` (FITS), ``.h5`` / ``.sdf`` (NDF), and
    ``.json`` (JSON), each optionally followed by a ``.gz`` or ``.zst``
    compression suffix.  The NDF and FITS backends are imported lazily so
    optional dependencies (e.g. ``h5py``) are only required when actually
    used.

    Parameters
    ----------
    path
        Path whose file extension selects the backend.

    Raises
    ------
    ValueError
        If the extension is not recognized.
    """
    uri = ResourcePath(path)
    name = uri.basename()
    for suffix in _COMPRESSION_SUFFIXES:
        if name.endswith(suffix):
            name = name.removesuffix(suffix)
            break
    match os.path.splitext(name)[1]:
        case ".fits":
            return backend_for_name("fits")
        case ".h5" | ".sdf":
            return backend_for_name("ndf")
        case ".json":
            return backend_for_name("json")
        case ext:
            raise ValueError(
                f"Unrecognized file extension: {ext!r} from {uri!r}; "
                "expected one of .fits, .h5, .sdf, .json, optionally with "
                "a .gz or .zst compression suffix."
            )


def backend_for_stream(stream: IO[bytes]) -> Backend:
    """Return the `Backend` for ``stream`` based on its leading bytes.

    The stream is restored to the position it was passed in with.
    Compressed content is not recognized here; decompress first (see
    `_maybe_decompress_stream`).

    Parameters
    ----------
    stream
        Seekable binary stream positioned at the start of the data.

    Raises
    ------
    ValueError
        If the leading bytes match no supported format.
    """
    start = stream.tell()
    head = stream.read(512)
    stream.seek(start)
    if head.startswith(_FITS_MAGIC):
        return backend_for_name("fits")
    if head.startswith(_HDF5_MAGIC):
        return backend_for_name("ndf")
    if head.lstrip().startswith(b"{"):
        return backend_for_name("json")
    raise ValueError(
        f"Could not identify a supported format from the leading bytes "
        f"{head[:16]!r}; expected FITS, HDF5/NDF, or JSON content.  "
        "Specify the format explicitly if it is known."
    )

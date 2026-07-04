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
import gzip
import io
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


def backend_for_path(path: ResourcePathExpression) -> Backend:
    """Return the `Backend` for ``path`` based on its file extension.

    Supported extensions: ``.fits`` / ``.fits.gz`` (FITS), ``.h5`` /
    ``.sdf`` (NDF), and ``.json`` (JSON).  The NDF and FITS backends are
    imported lazily so optional dependencies (e.g. ``h5py``) are only
    required when actually used.

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
        case ext:
            raise ValueError(
                f"Unrecognised file extension: {ext!r} from {uri!r}; "
                "expected one of .fits, .fits.gz, .h5, .sdf, .json."
            )

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
    "DEFAULT_PAGE_SIZE",
    "READ_CACHE_MAX_BYTES",
    "FitsInputArchive",
    "FitsOpaqueMetadata",
)

import io
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import cached_property
from types import EllipsisType
from typing import IO, Any, Self

import astropy.io.fits
import astropy.table
import fsspec
import numpy as np

from lsst.resources import ResourcePath, ResourcePathExpression

from .._transforms import FrameSet
from ..serialization import (
    ArchiveInfo,
    ArchiveReadError,
    ArchiveTree,
    ArrayReferenceModel,
    InlineArrayModel,
    InputArchive,
    TableModel,
    no_header_updates,
    parameterize_tree,
    tree_class_for_info,
)
from ..serialization._common import _check_format_version
from ._common import (
    JSON_COLUMN,
    JSON_EXTNAME,
    ExtensionHDU,
    ExtensionKey,
    FitsOpaqueMetadata,
    InvalidFitsArchiveError,
    PointerModel,
)

_FITS_FORMAT_VERSION = 1
"""Container layout version this release of `FitsInputArchive` understands."""

DEFAULT_PAGE_SIZE = 2880 * 800
"""Default fsspec read-block size for partial (remote) reads, in bytes.

This is the single place to tune the block size for remote-store performance.
On a buffered remote filesystem (e.g. GCS) each cache miss is one range
request, so the block size trades round trips against over-fetch: a component
read that touches scattered compressed tiles pulls one block per cluster of
nearby tiles, rounded up to this size.

The optimum depends on the access pattern.  Larger blocks favor reads that
touch most of the file (full planes, large cutouts); smaller blocks reduce
wasted bytes for small scattered cutouts.  ``2880 * 800`` (~2.3 MB, and a
multiple of the 2880-byte FITS block) is a robust middle: across cutout sizes
and full reads it stays within ~1.5x of the per-pattern optimum, whereas the
historical 144 KB default was several times slower for all but the tiniest
cutout.  Raise it (e.g. ``2880 * 1600``) when whole-file or large-cutout reads
dominate; lower it when many tiny cutouts across many files dominate.

Local filesystems ignore this (their opener does no buffering), so it only
affects remote stores.
"""

_READ_CACHE_TYPE = "blockcache"
"""fsspec cache strategy for partial reads.

``blockcache`` keeps a bounded set of fixed-size blocks (so memory stays
capped) and reuses them across the multiple components of one file -- image,
mask, variance and so on often share blocks -- unlike the default unbounded
single-block ``readahead``.
"""

READ_CACHE_MAX_BYTES = 64 * 1024 * 1024
"""Approximate memory budget for the partial-read block cache, per open file.

The fsspec block cache evicts least-recently-used blocks once it holds more
than ``maxblocks``; we derive ``maxblocks`` from this budget and the block
size (`DEFAULT_PAGE_SIZE`) so the memory cap is expressed in bytes and stays
fixed even when the block size is retuned.  Measured benefit saturates at two
cached blocks for the access patterns we care about, so this budget is purely
headroom plus a guard against unbounded growth; it is far below fsspec's
implicit default of ``32 * block_size``.
"""


class FitsInputArchive(InputArchive[PointerModel]):
    """An implementation of the `.serialization.InputArchive` interface that
    reads from FITS files.

    Instances of this class should only be constructed via the `open`
    context manager.
    """

    @classmethod
    def get_basic_info(cls, path: ResourcePathExpression) -> ArchiveInfo:
        """Read ``DATAMODL`` (schema URL) and ``FMTVER`` (container version)
        from the primary header.

        Every FITS file written by this package records the schema URL in
        the ``DATAMODL`` card, so the schema can be identified without
        reading the (potentially large) JSON tree HDU.
        """
        with ResourcePath(path).open("rb") as stream:
            primary = astropy.io.fits.PrimaryHDU.readfrom(stream)
            header = primary.header
            format_version = int(header.get("FMTVER", 1))
            schema_url = header.get("DATAMODL")
        if not schema_url:
            raise ArchiveReadError(f"{path!r} is not an lsst.images FITS archive (no DATAMODL card).")
        return ArchiveInfo.from_schema_url(schema_url, format_version=format_version)

    @classmethod
    @contextmanager
    def open_tree(
        cls,
        path: ResourcePathExpression,
        *,
        partial: bool = True,
        **backend_kwargs: Any,
    ) -> Iterator[tuple[Self, ArchiveTree, ArchiveInfo]]:
        """Open the FITS file and yield ``(archive, tree, info)``.

        Parameters
        ----------
        path
            The file resource to open.
        partial
            If `True` the file is opened without reading it all into memory.
        **backend_kwargs
            Optional parameters for this backend. Currently supports
            ``page_size`` which can be used to override the default
            page size (which can be overridden globally by modifying
            `DEFAULT_PAGE_SIZE`).
        """
        page_size = backend_kwargs.pop("page_size", DEFAULT_PAGE_SIZE)
        with cls.open(path, page_size=page_size, partial=partial) as archive:
            info = archive.info
            tree_cls = tree_class_for_info(info, path)
            parameterized = parameterize_tree(tree_cls, PointerModel)
            tree = archive.get_tree(parameterized)
            yield archive, tree, info

    def __init__(self, stream: IO[bytes]):
        self._primary_hdu = astropy.io.fits.PrimaryHDU.readfrom(stream)
        on_disk_fmtver: int = self._primary_hdu.header.pop("FMTVER", 1)
        # DATAMODL is informational only on read; the JSON tree's
        # schema_version / min_read_version drive data-model checks.  We
        # capture it here as ArchiveInfo so callers (e.g. open_tree) can
        # identify the schema from this open rather than reopening the file.
        # A schema-less file can still be opened directly; only callers that
        # need the schema (via the `info` property) require DATAMODL.
        schema_url = self._primary_hdu.header.pop("DATAMODL", None)
        self._info = (
            ArchiveInfo.from_schema_url(schema_url, format_version=on_disk_fmtver) if schema_url else None
        )
        _check_format_version("fits", on_disk_fmtver, _FITS_FORMAT_VERSION)
        # TODO: do some basic checks that the file format conforms to our
        # expectations (e.g. primary HDU should have no data).
        #
        # Read and strip the addresses and sizes from the headers.  We don't
        # actually need the index address because we always want to read the
        # JSON HDU, too, and the index HDU is always the next one (but this
        # could change in the future).
        json_address: int = self._primary_hdu.header.pop("JSONADDR")
        json_size: int = self._primary_hdu.header.pop("JSONSIZE")
        del self._primary_hdu.header["INDXADDR"]
        index_size: int = self._primary_hdu.header.pop("INDXSIZE")
        # Save the remaining primary header keys so we can propagate them on
        # rewrite.
        self._opaque_metadata = FitsOpaqueMetadata()
        self._opaque_metadata.add_header(self._primary_hdu.header.copy(strip=True), name="", ver=1)
        # Read the JSON and index HDUs from the end.
        stream.seek(json_address)
        tail_data = stream.read(json_size + index_size)
        index_hdu = astropy.io.fits.BinTableHDU.fromstring(tail_data[json_size:])
        # Initialize lazy readers for all of the regular HDUs and the JSON HDU.
        self._readers = {
            ExtensionKey.from_index_row(row): _ExtensionReader.from_index_row(row, stream)
            for row in index_hdu.data
        }
        self._readers[ExtensionKey(JSON_COLUMN)] = _ExtensionReader.from_bytes(
            astropy.io.fits.BinTableHDU, tail_data[:json_size]
        )
        # Make any empty dictionary to cache deserialized objects.  Keys are
        # the zero-indexed row in the JSON table.
        self._deserialized_pointer_cache: dict[int, Any] = {}

    @classmethod
    @contextmanager
    def open(
        cls,
        path: ResourcePathExpression,
        *,
        page_size: int = DEFAULT_PAGE_SIZE,
        partial: bool = False,
    ) -> Iterator[Self]:
        """Create an output archive that writes to the given file.

        Parameters
        ----------
        path
            File to read; convertible to `lsst.resources.ResourcePath`.
        page_size
            Size of the fsspec read block for partial (remote) reads, in
            bytes; a multiple of the FITS block size (2880) is recommended.
            Defaults to `DEFAULT_PAGE_SIZE`; see it for the tuning tradeoff.
        partial
            Whether we will be reading only some of the archive, or if memory
            pressure forces us to read it only a little at a time.  If `False`
            (default), the entire raw file may be read into memory up front.

        Returns
        -------
        `contextlib.AbstractContextManager` [`FitsInputArchive`]
            A context manager that returns a `FitsInputArchive` when entered.
        """
        path = ResourcePath(path)
        stream: IO[bytes]
        if not partial:
            stream = io.BytesIO(path.read())
            yield cls(stream)
        else:
            fs: fsspec.AbstractFileSystem
            fs, fp = path.to_fsspec()
            # Cap cached blocks from the byte budget so memory stays bounded as
            # the block size is retuned; keep at least two so the shared
            # header/index block survives between a file's components.
            maxblocks = max(2, READ_CACHE_MAX_BYTES // page_size)
            with fs.open(
                fp,
                block_size=page_size,
                cache_type=_READ_CACHE_TYPE,
                cache_options={"maxblocks": maxblocks},
            ) as stream:
                yield cls(stream)

    @property
    def info(self) -> ArchiveInfo:
        """Schema/format info read from the primary header on open.
        (`.serialization.ArchiveInfo`)
        """
        if self._info is None:
            raise ArchiveReadError("This is not an lsst.images FITS archive (no DATAMODL card).")
        return self._info

    def get_tree[T: ArchiveTree](self, model_type: type[T]) -> T:
        """Read the JSON tree from the archive.

        Parameters
        ----------
        model_type
            A Pydantic model type to use to validate the JSON.

        Returns
        -------
        T
            The validated Pydantic model.
        """
        json_bytes = self._readers[ExtensionKey(JSON_EXTNAME)].data[0][JSON_COLUMN].tobytes()
        return model_type.model_validate_json(json_bytes)

    def deserialize_pointer[U: ArchiveTree, V](
        self,
        pointer: PointerModel,
        model_type: type[U],
        deserializer: Callable[[U, InputArchive[PointerModel]], V],
    ) -> V:
        # Docstring inherited.
        if (cached := self._deserialized_pointer_cache.get(pointer.row)) is not None:
            return cached
        if not isinstance(pointer.column.data, ArrayReferenceModel):
            raise ArchiveReadError(f"Invalid pointer with inline array:\n{pointer.model_dump_json(indent=2)}")
        _, reader = self._get_source_reader(pointer.column.data.source, is_table=True)
        try:
            json_bytes = reader.data[pointer.row][JSON_COLUMN].tobytes()
        except Exception as err:
            raise InvalidFitsArchiveError(
                f"Failed to access the table cell referenced by {pointer.model_dump_json()}."
            ) from err
        result = deserializer(model_type.model_validate_json(json_bytes), self)
        self._deserialized_pointer_cache[pointer.row] = result
        return result

    def get_frame_set(self, ref: PointerModel) -> FrameSet:
        try:
            result = self._deserialized_pointer_cache[ref.row]
        except KeyError:
            raise AssertionError(
                f"Frame set at {ref.model_dump_json(indent=2)} must be deserialized "
                "before any dependent transform can be."
            ) from None
        if not isinstance(result, FrameSet):
            raise InvalidFitsArchiveError(f"Expected a FrameSet instance at {ref.model_dump_json(indent=2)}.")
        return result

    def get_array(
        self,
        model: ArrayReferenceModel | InlineArrayModel,
        *,
        slices: tuple[slice, ...] | EllipsisType = ...,
        strip_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> np.ndarray:
        if not isinstance(model, ArrayReferenceModel):
            raise ArchiveReadError("Inline array found where a reference array was expected.")
        key, reader = self._get_source_reader(model.source, is_table=False)
        if slices is not ...:
            array = reader.section[slices]
        else:
            array = reader.data
        if key not in self._opaque_metadata.headers:
            opaque_header = reader.header.copy(strip=True)
            strip_header(opaque_header)
            self._opaque_metadata.add_header(opaque_header, key=key)
        return array

    def get_table(
        self,
        model: TableModel,
        strip_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> astropy.table.Table:
        # Docstring inherited.
        array = self.get_structured_array(model, strip_header)
        table = astropy.table.Table(array)
        for c in model.columns:
            c.update_table(table)
        return table

    def get_structured_array(
        self,
        model: TableModel,
        strip_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> np.ndarray:
        # Docstring inherited.
        if not isinstance(model.columns[0].data, ArrayReferenceModel):
            raise ArchiveReadError("Inline array found where a reference array was expected.")
        # All columns should have the same data.source; just use the first.
        key, reader = self._get_source_reader(model.columns[0].data.source, is_table=True)
        if key not in self._opaque_metadata.headers:
            opaque_header = reader.header.copy(strip=True)
            strip_header(opaque_header)
            self._opaque_metadata.add_header(opaque_header, key=key)
        return reader.hdu.data

    def get_opaque_metadata(self) -> FitsOpaqueMetadata:
        # Docstring inherited.
        return self._opaque_metadata

    def _get_source_reader(self, source: str | int, is_table: bool) -> tuple[ExtensionKey, _ExtensionReader]:
        """Get a reader for the extension referenced by a serialiation model's
        ``source`` field.

        Parameters
        ----------
        source
            A ``source`` field of the form ``fits:${hdu}`` or
            ``fits:${hdu}[${col}]``.
        is_table
            Whether the source should be for a table HDU.

        Returns
        -------
        key
            Identifier pair for the HDU (EXTNAME, EXTVER).
        reader
            A reader object for the extension.
        """
        if not isinstance(source, str):
            raise InvalidFitsArchiveError(f"Reference with source={source!r} is not a string.")
        if not source.startswith("fits:"):
            raise InvalidFitsArchiveError(f"Reference with source={source!r} does not start with 'fits:'.")
        key = ExtensionKey.from_str(source)
        try:
            reader = self._readers[key]
        except KeyError:
            raise InvalidFitsArchiveError(f"Unrecognized source value {key}.") from None
        if is_table and not reader.is_table:
            raise InvalidFitsArchiveError(
                f"Extension with source={key} was expected to be be a binary table, not an image."
            )
        elif not is_table and reader.is_table:
            raise InvalidFitsArchiveError(
                f"Extension with source={key} was expected to be be an image, not a binary table."
            )
        return key, reader


class _ExtensionReader:
    """A lazy-load reader for a single extension HDU.

    Parameters
    ----------
    hdu_cls
        The type of the astropy HDU instance to construct.
    stream
        The file-like object to read from.
    """

    def __init__(self, hdu_cls: type[ExtensionHDU], stream: IO[bytes]):
        self._hdu_cls = hdu_cls
        self._stream = stream

    @classmethod
    def from_index_row(cls, index_row: np.void, stream: IO[bytes]) -> _ExtensionReader:
        """Construct from a row of the binary table index HDU.

        Parameters
        ----------
        index_row
            A record array row from the index HDU.
        stream
            The file-like object being used to read the full FITS file.

        Returns
        -------
        reader
            A reader object for the extension.
        """
        match index_row["XTENSION"].strip():
            case "IMAGE":
                hdu_cls = astropy.io.fits.ImageHDU
            case "BINTABLE":
                if index_row["ZIMAGE"]:
                    hdu_cls = astropy.io.fits.CompImageHDU
                else:
                    hdu_cls = astropy.io.fits.BinTableHDU
            case other:
                raise AssertionError(f"Unsupported HDU type {other!r}.")
        return _ExtensionReader(
            hdu_cls,
            _RangeStreamProxy(
                stream,
                start=int(index_row["HDRADDR"]),
            ),
        )

    @classmethod
    def from_bytes(cls, hdu_cls: type[ExtensionHDU], data: bytes) -> _ExtensionReader:
        """Construct from already-read `bytes`

        Parameters
        ----------
        hdu_cls
            The HDU type to instantiate.
        data
            Raw data for the HDU.

        Returns
        -------
        reader
            A reader object for extension.
        """
        return _ExtensionReader(hdu_cls, io.BytesIO(data))

    @property
    def is_table(self) -> bool:
        """Whether this is logically a table HDU.

        This is `False` for compressed image HDUs, even though they are
        represented in FITS as a binary table.
        """
        return issubclass(self._hdu_cls, astropy.io.fits.BinTableHDU) and not issubclass(
            self._hdu_cls, astropy.io.fits.CompImageHDU
        )

    @cached_property
    def hdu(self) -> ExtensionHDU:
        """The Astropy HDU object."""
        self._stream.seek(0)
        if self._hdu_cls is astropy.io.fits.CompImageHDU:
            # CompImageHDU.readfrom doesn't work; we need to make a minimal
            # example and report it upstream.  Happily this workaround does
            # work.
            bintable_hdu = astropy.io.fits.BinTableHDU.readfrom(self._stream, memmap=False, cache=False)
            return self._hdu_cls(bintable=bintable_hdu)
        else:
            return self._hdu_cls.readfrom(self._stream, memmap=False, cache=False, uint=True)

    @property
    def header(self) -> astropy.io.fits.Header:
        """The header of the HDU."""
        return self.hdu.header

    @property
    def data(self) -> np.ndarray:
        """The data for the HDU."""
        return self.hdu.data

    @property
    def section(self) -> astropy.io.fits.Section | astropy.io.fits.CompImageSection:
        """An Astropy expression object that reads a subset of the data when
        sliced.
        """
        return self.hdu.section


class _RangeStreamProxy(IO[bytes]):
    """A readable IO proxy object that makes the beginning of the file appear
    at a custom position.

    Parameters
    ----------
    base
        Underlying readable, seekable buffer to proxy.
    start
        Offset into the base stream that will be considered the start of the
        proxy stream.

    Notes
    -----
    This class exists because Astropy doesn't seem to provide a way to read a
    single HDU that starts at the current seek position of a file-like object.
    It does provide ``readfrom`` methods on its HDU objects that take a
    file-like object, but these assume (possibly unintentionally; it only
    happens when Astropy is trying to see whether the file was opened for
    appending) that ``seek(0)`` will set the file-like object to the start of
    the HDU.
    """

    def __init__(self, base: IO[bytes], start: int):
        self._base = base
        self._start = start

    @property
    def mode(self) -> str:
        return "rb"

    def __enter__(self) -> Self:
        raise AssertionError("This proxy should not be used as a context manager.")

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        raise AssertionError("This proxy should not be used as a context manager.")

    def __iter__(self) -> Iterator[bytes]:
        return self._base.__iter__()

    def __next__(self) -> bytes:
        return self._base.__next__()

    def close(self) -> None:
        raise AssertionError("This proxy should not ever be closed.")

    @property
    def closed(self) -> bool:
        return False

    def fileno(self) -> int:
        raise OSError()

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False

    def read(self, n: int = -1, /) -> bytes:
        result = self._base.read(n)
        return result

    def readable(self) -> bool:
        return True

    def readline(self, limit: int = -1, /) -> bytes:
        return self._base.readline(limit)

    def readlines(self, hint: int = -1, /) -> list[bytes]:
        return self._base.readlines(hint)

    def seek(self, offset: int, whence: int = 0) -> int:
        match whence:
            case os.SEEK_SET:
                return self._base.seek(offset + self._start, os.SEEK_SET) - self._start
            case os.SEEK_CUR:
                return self._base.seek(offset, os.SEEK_CUR) - self._start
            case os.SEEK_END:
                return self._base.seek(offset, os.SEEK_END) - self._start
        raise TypeError(f"Invalid value for 'whence': {whence}.")

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._base.tell() - self._start

    def truncate(self, size: int | None = None, /) -> int:
        raise OSError()

    def writable(self) -> bool:
        return False

    def write(self, arg: Any, /) -> int:
        raise OSError()

    def writelines(self, arg: Any, /) -> None:
        raise OSError()

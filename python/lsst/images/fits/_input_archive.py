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
    "FitsInputArchive",
    "FitsOpaqueMetadata",
)

import io
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import cached_property
from typing import IO, Any, Self

import astropy.io.fits
import astropy.table
import fsspec
import numpy as np
import pydantic

from lsst.resources import ResourcePath, ResourcePathExpression

from .._geom import Box
from .._image import Image
from .._mask import Mask, MaskSchema
from ..serialization import (
    ArrayReferenceModel,
    ImageModel,
    InputArchive,
    MaskModel,
    TableCellReferenceModel,
    TableModel,
    no_header_updates,
)
from ._common import (
    ExtensionHDU,
    FitsOpaqueMetadata,
    InvalidFitsArchiveError,
    strip_wcs_cards,
)


class FitsInputArchive(InputArchive[TableCellReferenceModel]):
    """An implementation of the `.serialization.InputArchive` interface that
    reads from FITS files.

    Instances of this class should only be constructed via the `open`
    context manager.
    """

    def __init__(self, stream: IO[bytes]):
        self._primary_hdu = astropy.io.fits.PrimaryHDU.readfrom(stream)
        # TODO: read and strip subformat declaration and version, once we start
        # writing those.
        #
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
        self._opaque_metadata.headers[""] = self._primary_hdu.header.copy(strip=True)
        # Read the JSON and index HDUs from the end.
        stream.seek(json_address)
        tail_data = stream.read(json_size + index_size)
        index_hdu = astropy.io.fits.BinTableHDU.fromstring(tail_data[json_size:])
        # Initialize lazy readers for all of the regular HDUs and the JSON HDU.
        self._readers = {
            str(row["EXTNAME"]): _ExtensionReader.from_index_row(row, stream) for row in index_hdu.data
        }
        self._readers["JSON"] = _ExtensionReader.from_bytes(
            astropy.io.fits.BinTableHDU, tail_data[:json_size]
        )
        # Make any empty dictionary to cache deserialized objects.
        self._deserialized_pointer_cache: dict[TableCellReferenceModel, Any] = {}

    @classmethod
    @contextmanager
    def open(
        cls,
        path: ResourcePathExpression,
        *,
        page_size: int = 2880 * 50,
        partial: bool = False,
    ) -> Iterator[Self]:
        """Create an output archive that writes to the given file.

        Parameters
        ----------
        path
            File to read; convertible to `lsst.resources.ResourcePath`.
        page_size
            Minimum number of bytes to read at at once.  Making this a multiple
            of the FITS block size (2880) is recommended.
        partial
            Whether we will be reading only some of the archive, or if memory
            pressure forces us to read it only a little at a time..  If `False`
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
            with fs.open(fp, block_size=page_size) as stream:
                yield cls(stream)

    def get_tree[T: pydantic.BaseModel](self, model_type: type[T]) -> T:
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
        json_bytes = self._readers["JSON"].data[0]["JSON"].tobytes()
        return model_type.model_validate_json(json_bytes)

    def deserialize_pointer[U: pydantic.BaseModel, V](
        self,
        pointer: TableCellReferenceModel,
        model_type: type[U],
        deserializer: Callable[[U, InputArchive[TableCellReferenceModel]], V],
    ) -> V:
        # Docstring inherited.
        if (cached := self._deserialized_pointer_cache.get(pointer)) is not None:
            return cached
        _, reader = self._get_source_reader(pointer)
        try:
            json_bytes = reader.data[pointer.row][pointer.column].tobytes()
        except Exception as err:
            raise InvalidFitsArchiveError(
                f"Failed to access the table cell referenced by {pointer.model_dump_json()}."
            ) from err
        result = deserializer(model_type.model_validate_json(json_bytes), self)
        self._deserialized_pointer_cache[pointer] = result
        return result

    def get_image(
        self,
        ref: ImageModel,
        *,
        bbox: Box | None = None,
        strip_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> Image:
        # Docstring inherited.
        array_model, unit = ref.unpack()
        name, reader = self._get_source_reader(array_model)
        if bbox is not None:
            array = reader.section[bbox.slice_within(ref.bbox)]
        else:
            array = reader.data
            bbox = ref.bbox
        if name not in self._opaque_metadata.headers:
            opaque_header = reader.header.copy(strip=True)
            if unit is not None:
                opaque_header.pop("BUINT", None)
            strip_wcs_cards(opaque_header)
            strip_header(opaque_header)
            self._opaque_metadata.headers[name] = opaque_header
        return Image(array, bbox=bbox, unit=unit)

    def get_mask(
        self,
        ref: MaskModel,
        *,
        bbox: Box | None = None,
        strip_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> Mask:
        # Docstring inherited.
        name, reader = self._get_source_reader(ref.data)
        if bbox is not None:
            array = reader.section[bbox.slice_within(ref.bbox) + (slice(None),)]
        else:
            array = reader.data
            bbox = ref.bbox
        schema = MaskSchema(ref.planes, dtype=array.dtype)
        if name not in self._opaque_metadata.headers:
            opaque_header = reader.header.copy(strip=True)
            # TODO: strip mask plane information from headers.
            strip_wcs_cards(opaque_header)
            strip_header(opaque_header)
            self._opaque_metadata.headers[name] = opaque_header
        return Mask(array, schema=schema, bbox=bbox)

    def get_table(
        self,
        ref: TableModel,
        strip_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> astropy.table.Table:
        # Docstring inherited.
        array = self.get_structured_array(ref, strip_header)
        table = astropy.table.Table(array)
        for c in ref.columns:
            c.update_table(table)
        return table

    def get_structured_array(
        self,
        ref: TableModel,
        strip_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> np.ndarray:
        # Docstring inherited.
        name, reader = self._get_source_reader(ref)
        if name not in self._opaque_metadata.headers:
            opaque_header = reader.header.copy(strip=True)
            strip_header(opaque_header)
            self._opaque_metadata.headers[name] = opaque_header
        return reader.hdu.data

    def get_opaque_metadata(self) -> FitsOpaqueMetadata:
        # Docstring inherited.
        return self._opaque_metadata

    def _get_source_reader(
        self, ref: ArrayReferenceModel | TableCellReferenceModel | TableModel
    ) -> tuple[str, _ExtensionReader]:
        """Get a reader for the extension referenced by a model.

        Parameters
        ----------
        ref
            A Pydantic model with a 'source' field that references a FITS HDU.

        Returns
        -------
        name
            EXTNAME of the HDU.
        reader
            A reader object for the extension.
        """
        if not isinstance(ref.source, str):
            raise InvalidFitsArchiveError(f"Reference with source={ref.source!r} is not a string.")
        if not ref.source.startswith("fits:"):
            raise InvalidFitsArchiveError(
                f"Reference with source={ref.source!r} does not start with 'fits:'."
            )
        name = ref.source.removeprefix("fits:")
        try:
            reader = self._readers[name]
        except KeyError:
            if name.isnumeric():
                msg = f"Extension index {name!r} in source={ref.source!r} is not supported."
            else:
                msg = f"Unrecognized EXTNAME {name!r} in source={ref.source!r}"
            raise InvalidFitsArchiveError(msg) from None
        if ref.source_is_table and not reader.is_table:
            raise InvalidFitsArchiveError(
                f"Extension with EXTNAME={name!r} was expected to be be a binary table, not an image."
            )
        elif not ref.source_is_table and reader.is_table:
            raise InvalidFitsArchiveError(
                f"Extension with EXTNAME={name!r} was expected to be be an image, not a binary table."
            )
        return name, reader


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
            return self._hdu_cls.readfrom(self._stream, memmap=False, cache=False)

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

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
    "FitsCompressionOptions",
    "FitsInputArchive",
    "FitsOpaqueMetadata",
    "FitsOutputArchive",
    "InvalidFitsArchiveError",
)

import dataclasses
import io
from collections.abc import Callable, Hashable, Iterable
from functools import cached_property
from typing import IO, Any

import astropy.io.fits
import astropy.table
import numpy as np
import pydantic

from ._coordinate_transform import CoordinateTransform
from ._dtypes import NumberType
from ._geom import Box
from ._image import Image, ImageModel
from ._mask import Mask, MaskModel, MaskSchema
from .archive import (
    InputArchive,
    NestedOutputArchive,
    OutputArchive,
    TableCellReferenceModel,
    TableModel,
    no_header_updates,
)
from .asdf_utils import ArrayReferenceModel

type ExtensionHDU = astropy.io.fits.ImageHDU | astropy.io.fits.CompImageHDU | astropy.io.fits.BinTableHDU


class InvalidFitsArchiveError(RuntimeError):
    """The error type raised when the content of a FITS file presumed to have
    been written by FitsOutputArchive is not self-consistent.
    """


@dataclasses.dataclass
class FitsCompressionOptions:
    """Configuration options for FITS compression."""


@dataclasses.dataclass
class FitsOpaqueMetadata:
    """Opaque metadata that may be carried around by a serializable type to
    propagate serialization options and opaque information without that type
    knowing how it was serialized.
    """

    headers: dict[str, astropy.io.fits.Header] = dataclasses.field(default_factory=dict)
    """FITS headers found (but not interpreted and stripped) when reading, to
    be propagated on write.

    Keys are EXTNAME values, or "" for the primary header.
    """

    compression: dict[str, FitsCompressionOptions] = dataclasses.field(default_factory=dict)
    """FITS compression options found on when reading, to be used again (by
    default) on write.
    """


class FitsOutputArchive(OutputArchive[TableCellReferenceModel]):
    """An implementation of the `OutputArchive` interface that writes to FITS
    files.
    """

    def __init__(
        self,
        compression_options: FitsCompressionOptions | None = None,
        opaque_metadata: FitsOpaqueMetadata | None = None,
    ):
        self._primary_hdu = astropy.io.fits.PrimaryHDU()
        # TODO: add subformat description and version to primary HDU.
        self._hdu_list = astropy.io.fits.HDUList([self._primary_hdu])  # actual FITS HDUs to write.
        self._primary_hdu.header.set("INDXADDR", 0, "Offset in bytes to the HDU index.")
        self._primary_hdu.header.set("INDXSIZE", 0, "Size of the HDU index.")
        self._primary_hdu.header.set("JSONADDR", 0, "Offset in bytes to the JSON tree HDU.")
        self._primary_hdu.header.set("JSONSIZE", 0, "Size of the JSON tree HDU.")
        self._compression_options = (
            compression_options if compression_options is not None else FitsCompressionOptions()
        )
        self._opaque_metadata = opaque_metadata if opaque_metadata is not None else FitsOpaqueMetadata()
        if (opaque_primary_header := self._opaque_metadata.headers.get("")) is not None:
            self._primary_hdu.header.extend(opaque_primary_header)
        # JSON blobs for objects we've saved as pointers:
        self._pointer_targets: list[bytes] = []
        # Mapping from user provided key (e.g. id(some object)) to a table
        # pointer to where we actually saved it:
        self._pointers_by_key: dict[Hashable, TableCellReferenceModel] = {}

    def add_coordinate_transform(
        self, transform: CoordinateTransform, from_frame: str, to_frame: str = "sky"
    ) -> TableCellReferenceModel:
        raise NotImplementedError("TODO")

    def serialize_direct[T: pydantic.BaseModel](
        self, name: str, serializer: Callable[[OutputArchive], T]
    ) -> T:
        nested = NestedOutputArchive[TableCellReferenceModel](f"/{name}", self)
        return serializer(nested)

    def serialize_pointer[T: pydantic.BaseModel](
        self, name: str, serializer: Callable[[OutputArchive], T], key: Hashable
    ) -> TableCellReferenceModel:
        if (pointer := self._pointers_by_key.get(key)) is not None:
            return pointer
        pointer = TableCellReferenceModel(
            source="fits:JSON", column="JSON", row=len(self._pointer_targets) + 1
        )
        model = self.serialize_direct("/", serializer)
        self._pointer_targets.append(model.model_dump_json().encode())
        self._pointers_by_key[key] = pointer
        return pointer

    def add_image(
        self,
        name: str,
        image: Image,
        *,
        pixel_frame: str | None = None,
        wcs_frames: Iterable[str] = (),
        update_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> ImageModel:
        # TODO: look for compression options in both the main options and the
        # opaque metadata, make a CompImageHDU instead if requested.
        hdu = astropy.io.fits.ImageHDU(image.array, name=name)
        if image.unit:
            hdu.header["BUNIT"] = image.unit.to_string(format="fits")
        # TODO: add a default WCS from pixel_frame to 'sky', if pixel_frame is
        # provided and 'sky' is known to the archive; use the 'A'-suffix WCS to
        # transform from 1-indexed FITS to the image's bounding box (or the
        # default WCS otherwise); use B-Z for mappings from pixel_frame to each
        # entry in wcs_frames.
        update_header(hdu.header)
        if (opaque_headers := self._opaque_metadata.headers.get(name)) is not None:
            hdu.header.extend(opaque_headers)
        array_model = ArrayReferenceModel(
            source=f"fits:{name}",
            shape=list(image.array.shape),
            datatype=NumberType.from_numpy(image.array.dtype),
        )
        self._hdu_list.append(hdu)
        return ImageModel.pack(array_model, start=[i.start for i in image.bbox], unit=image.unit)

    def add_mask(
        self,
        name: str,
        mask: Mask,
        *,
        pixel_frame: str | None = None,
        wcs_frames: Iterable[str] = (),
        update_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> MaskModel:
        # TODO: look for compression options in both the main options and the
        # opaque metadata, make a CompImageHDU instead if requested.
        hdu = astropy.io.fits.ImageHDU(mask.array, name=name)
        # TODO: add a default WCS from pixel_frame to 'sky', if pixel_frame is
        # provided and 'sky' is known to the archive; use the 'A'-suffix WCS to
        # transform from 1-indexed FITS to the image's bounding box (or the
        # default WCS otherwise); use B-Z for mappings from pixel_frame to each
        # entry in wcs_frames.
        # TODO: write mask schema to FITS header.
        update_header(hdu.header)
        if (opaque_headers := self._opaque_metadata.headers.get(name)) is not None:
            hdu.header.extend(opaque_headers)
        array_model = ArrayReferenceModel(
            source=f"fits:{name}",
            shape=list(mask.array.shape),
            datatype=NumberType.from_numpy(mask.array.dtype),
        )
        self._hdu_list.append(hdu)
        return MaskModel(
            data=array_model,
            start=[i.start for i in mask.bbox],
            planes=list(mask.schema),
        )

    def add_table(
        self,
        name: str,
        table: astropy.table.Table,
        *,
        update_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> TableModel:
        raise NotImplementedError("TODO")

    def write(self, filename: str, tree: pydantic.BaseModel) -> None:
        json_hdu = astropy.io.fits.BinTableHDU.from_columns(
            [astropy.io.fits.Column("JSON", "PA")],
            nrows=len(self._pointer_targets) + 1,
            name="JSON",
        )
        json_hdu.data["JSON"][0] = tree.model_dump_json().encode()
        for n, json_target_data in enumerate(self._pointer_targets):
            json_hdu.data["JSON"][n + 1] = json_target_data
        # We write the file in three phases, opening it again each time. There
        # might be ways to avoid that, but this avoids relying on any astropy
        # behavior that isn't crystal-clear in the docs. The first write
        # includes a preliminary version of the primary HDU (with zero
        # placeholders for the addresses), all of the regular image and binary
        # table extensions, and the JSON binary table extension.
        self._hdu_list.writeto(filename)
        json_file_info = json_hdu.fileinfo()
        # The next write appends a binary table HDU an index into all of the
        # other HDUs.
        with astropy.io.fits.open(filename, mode="append") as reopened_hdu_list:
            index_hdu = self._make_index_table(self._hdu_list)
            reopened_hdu_list.append(index_hdu)
            reopened_hdu_list.flush()
        index_file_info = index_hdu.fileinfo()
        # Update the primary HDU with the address and size of the index and
        # JSON HDUs, and rewrite just that.  We do this write manually, since
        # astropy's docs on its 'update' mode are scarce and it's not obvious
        # what it's possible to ask it to just rewrite the primary header and
        # definitely not resize the rest of the file.
        self._primary_hdu.header["INDXADDR"] = index_file_info["hdrLoc"]
        self._primary_hdu.header["INDXSIZE"] = self._size_from_file_info(index_file_info)
        self._primary_hdu.header["JSONADDR"] = json_file_info["hdrLoc"]
        self._primary_hdu.header["JSONSIZE"] = self._size_from_file_info(json_file_info)
        with open(filename, "r+b") as stream:
            stream.write(self._primary_hdu.header.tostring().encode())

    @staticmethod
    def _make_index_table(hdu_list: astropy.io.fits.HDUList) -> astropy.io.fits.BinTableHDU:
        index_hdu = astropy.io.fits.BinTableHDU.from_columns(
            [
                astropy.io.fits.Column("EXTNAME", "PA"),
                astropy.io.fits.Column("XTENSION", "A8"),
                astropy.io.fits.Column("ZIMAGE", "L"),
                astropy.io.fits.Column("HDRADDR", "K"),
                astropy.io.fits.Column("DATADDR", "K"),
                astropy.io.fits.Column("DATSIZE", "K"),
            ],
            nrows=len(hdu_list),
            name="INDEX",
        )
        hdu: ExtensionHDU | astropy.io.fits.PrimaryHDU
        for n, hdu in enumerate(hdu_list):
            file_info = hdu.fileinfo()
            index_hdu.data[n]["EXTNAME"] = hdu.header.get("EXTNAME", "")
            index_hdu.data[n]["XTENSION"] = hdu.header.get("XTENSION", "IMAGE")
            index_hdu.data[n]["ZIMAGE"] = isinstance(hdu, astropy.io.fits.CompImageHDU)
            index_hdu.data[n]["HDRADDR"] = file_info["hdrLoc"]
            index_hdu.data[n]["DATADDR"] = file_info["datLoc"]
            index_hdu.data[n]["DATSIZE"] = file_info["dataSpan"]
        return index_hdu

    @staticmethod
    def _size_from_file_info(file_info: dict[str, int]) -> int:
        return file_info["datSpan"] + file_info["datLoc"] - file_info["hdrLoc"]


class FitsInputArchive(InputArchive):
    """An implementation of the `InputArchive` interface that reads from FITS
    files.

    Parameters
    ----------
    stream
        File-like object to read from.  Assuming to point to same point that
        ``stream.seek(0)`` would bring it to, which must be the start of the
        FITS file.
    page_size, optional
        Minimum number of bytes to read at at once.  Making this a multiple of
        the FITS block size (2880) is recommended.
    partial, optional
        Whether we will be reading only some of the archive, or if memory
        pressure forces us to read it only a little at a time..  If `False`
        (default), the entire raw file may be read into memory up front.
    """

    def __init__(
        self,
        stream: IO[bytes],
        *,
        page_size: int = 2880 * 50,
        partial: bool = False,
    ):
        # Note that we're not asking astropy to use fsspec because fsspec can't
        # know about the addresses we've saved and hence it has no choice but
        # to skip along 2880 bytes at a time looking for HDU boundaries, which
        # we really don't want on network storage.
        buffered_stream: IO[bytes]
        if not partial:
            buffered_stream = io.BytesIO(stream.read())
        else:
            # We don't know how big the primary header is, but we want to read
            # at least page_size at a time to avoid per-read overheads in
            # network storage.  So we use a buffered file-like object with that
            # page_size.
            #
            # For some reason typing.IO[bytes] isn't substitutable for
            # io.RawIOBase.
            buffered_stream = io.BufferedRandom(stream, page_size)  # type:ignore[arg-type]
        self._primary_hdu = astropy.io.fits.PrimaryHDU.readfrom(buffered_stream)
        # TODO: read and strip subformat declaration and version, once we start
        # writing those.
        #
        # TODO: do some basic checks that the file format conforms to our
        # expectations (e.g. primary HDU should have no data).
        #
        # Read and strip the addresses and sizes from the headers.  We don't
        # actually need the index address because we always want to read the
        # JSON HDU, too (but that could change in the future).
        json_address: int = self._primary_hdu.header.pop("JSONADDR")
        json_size: int = self._primary_hdu.header.pop("JSONSIZE")
        del self._primary_hdu.header["INDXADDR"]
        index_size: int = self._primary_hdu.header.pop("INDXSIZE")
        # Save the remaining primary header keys so we can propagate them on
        # rewrite.
        self._opaque_metadata = FitsOpaqueMetadata()
        self._opaque_metadata.headers[""] = self._primary_hdu.header.copy(strip=True)
        # Read the JSON and index HDUs from the buffered stream.  If
        # partial=True, the seek here will probably drop some bytes of earlier
        # HDUs that were in the buffer, so there might be a small opportunity
        # for future optimization.
        buffered_stream.seek(json_address)
        tail_data = buffered_stream.read(json_size + index_size)
        index_hdu = astropy.io.fits.BinTableHDU.fromstring(tail_data[json_size:])
        # Initialize lazy readers for all of the regular HDUs and the JSON HDU.
        self._readers = {
            row["EXTNAME"]: ExtensionReader.from_index_row(row, buffered_stream) for row in index_hdu.data
        }
        self._readers["JSON"] = ExtensionReader.from_bytes(astropy.io.fits.BinTableHDU, tail_data[:json_size])
        # Make any empty dictionary to cache deserialized objects.
        self._deserialized_pointer_cache: dict[TableCellReferenceModel, Any] = {}

    def get_tree[T: pydantic.BaseModel](self, model_type: type[T]) -> T:
        json_bytes = self._readers["JSON"].data["JSON"][0]
        return model_type.model_validate_json(json_bytes)

    def get_coordinate_transform(self, from_frame: str, to_frame: str = "sky") -> CoordinateTransform:
        raise NotImplementedError("TODO")

    def deserialize_pointer[U: pydantic.BaseModel, V](
        self,
        pointer: TableCellReferenceModel,
        model_type: type[U],
        deserializer: Callable[[U, InputArchive], V],
    ) -> V:
        if (cached := self._deserialized_pointer_cache.get(pointer)) is not None:
            return cached
        _, reader = self._get_source_reader(pointer)
        try:
            json_bytes = reader.data[pointer.column][pointer.row]
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
        array_model, unit = ref.unpack()
        name, reader = self._get_source_reader(array_model)
        if bbox is not None:
            array = reader.section[ref.bbox.slice_within(bbox)]
        else:
            array = reader.data
            bbox = ref.bbox
        if name not in self._opaque_metadata.headers:
            opaque_header = reader.header.copy(strip=True)
            if unit is not None:
                opaque_header.pop("BUINT", None)
            # TODO: strip any WCS keys we might have added.
            strip_header(opaque_header)
            self._opaque_metadata.headers[name] = opaque_header
        # TODO: extract compression information into opaque_metadata
        return Image(array, bbox=bbox, unit=unit)

    def get_mask(
        self,
        ref: MaskModel,
        *,
        bbox: Box | None = None,
        strip_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> Mask:
        name, hdu = self._get_source_reader(ref.data)
        if bbox is not None:
            array = hdu.section[ref.bbox.slice_within(bbox) + (slice(None),)]
        else:
            array = hdu.data
            bbox = ref.bbox
        schema = MaskSchema(ref.planes, dtype=array.dtype)
        if name not in self._opaque_metadata.headers:
            opaque_header = hdu.header.copy(strip=True)
            # TODO: strip mask plane information from headers.
            # TODO: strip any WCS keys we might have added.
            strip_header(opaque_header)
            self._opaque_metadata.headers[name] = opaque_header
        # TODO: extract compression information into opaque_metadata
        return Mask(array, schema=schema, bbox=bbox)

    def get_table(
        self,
        ref: TableModel,
        strip_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> astropy.table.Table:
        raise NotImplementedError("TODO")

    def get_opaque_metadata(self) -> FitsOpaqueMetadata:
        return self._opaque_metadata

    def _get_source_reader(
        self, ref: ArrayReferenceModel | TableCellReferenceModel
    ) -> tuple[str, ExtensionReader]:
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


class ExtensionReader:
    def __init__(self, hdu_cls: type[ExtensionHDU], header_address: int, stream: IO[bytes]):
        self._hdu_cls = hdu_cls
        self._header_address = header_address
        self._stream = stream

    @classmethod
    def from_index_row(cls, index_row: np.void, stream: IO[bytes]) -> ExtensionReader:
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
        return ExtensionReader(hdu_cls, index_row["HDRADDR"], stream)

    @classmethod
    def from_bytes(cls, hdu_cls: type[ExtensionHDU], data: bytes) -> ExtensionReader:
        return ExtensionReader(hdu_cls, 0, io.BytesIO(data))

    @property
    def is_table(self) -> bool:
        return issubclass(self._hdu_cls, astropy.io.fits.BinTableHDU) and not issubclass(
            self._hdu_cls, astropy.io.fits.CompImageHDU
        )

    @cached_property
    def hdu(self) -> ExtensionHDU:
        self._stream.seek(self._header_address)
        return self._hdu_cls.readfrom(self._stream, memmap=False, cache=False)

    @property
    def header(self) -> astropy.io.fits.Header:
        return self.hdu.header

    @property
    def data(self) -> np.ndarray:
        return self.hdu.data

    @property
    def section(self) -> astropy.io.fits.Section | astropy.io.fits.CompImageSection:
        return self.hdu.section

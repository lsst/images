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

__all__ = ("FitsOutputArchive",)

import dataclasses
from collections.abc import Callable, Hashable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any, Self

import astropy.io.fits
import astropy.table
import numpy as np
import pydantic

from .._coordinate_transform import CoordinateTransform
from .._dtypes import NumberType
from .._image import Image, ImageModel
from .._mask import Mask, MaskModel
from ..archive import (
    NestedOutputArchive,
    OutputArchive,
    TableCellReferenceModel,
    TableModel,
    no_header_updates,
)
from ..asdf_utils import ArrayReferenceModel
from ._common import ExtensionHDU, FitsCompressionOptions, FitsOpaqueMetadata


class FitsOutputArchive(OutputArchive[TableCellReferenceModel]):
    """An implementation of the `OutputArchive` interface that writes to FITS
    files.

    Instances of this class should only be constructed via the `open`
    context manager.
    """

    def __init__(
        self,
        hdu_list: astropy.io.fits.HDUList,
        compression_options: Mapping[str, FitsCompressionOptions | None] | None = None,
        opaque_metadata: Any = None,
    ):
        # JSON blobs for objects we've saved as pointers:
        self._pointer_targets: list[bytes] = []
        # Mapping from user provided key (e.g. id(some object)) to a table
        # pointer to where we actually saved it:
        self._pointers_by_key: dict[Hashable, TableCellReferenceModel] = {}
        self._hdu_list = hdu_list
        self._primary_hdu = astropy.io.fits.PrimaryHDU()
        # TODO: add subformat description and version to primary HDU.
        self._primary_hdu.header.set("INDXADDR", 0, "Offset in bytes to the HDU index.")
        self._primary_hdu.header.set("INDXSIZE", 0, "Size of the HDU index.")
        self._primary_hdu.header.set("JSONADDR", 0, "Offset in bytes to the JSON tree HDU.")
        self._primary_hdu.header.set("JSONSIZE", 0, "Size of the JSON tree HDU.")
        self._compression_options = dict(compression_options) if compression_options is not None else {}
        self._opaque_metadata = (
            opaque_metadata if isinstance(opaque_metadata, FitsOpaqueMetadata) else FitsOpaqueMetadata()
        )
        if (opaque_primary_header := self._opaque_metadata.headers.get("")) is not None:
            self._primary_hdu.header.extend(opaque_primary_header)
        self._hdu_list.append(self._primary_hdu)
        self._json_hdu_added: bool = False

    @classmethod
    @contextmanager
    def open(
        cls,
        filename: str,
        compression_options: Mapping[str, FitsCompressionOptions | None] | None = None,
        opaque_metadata: Any = None,
    ) -> Iterator[Self]:
        """Create an output archive that writes to the given file.

        Parameters
        ----------
        filename
            Name of the file to write to.  Must not already exist.
        compression_options, optional
            Options for how to compress the FITS file, keyed by the name of
            the attribute (with JSON pointer ``/`` separators for nested
            attributes).
        opaque_metadata, optional
            Metadata read from an input archive along with the object being
            written now.  Ignored if the metadata is not from a FITS archive.

        Returns
        -------
        context
            A context manager that returns a `FitsOutputArchive` when entered.
        """
        with astropy.io.fits.open(filename, mode="append") as hdu_list:
            if hdu_list:
                raise OSError(f"File {filename!r} already exists.")
            archive = cls(hdu_list, compression_options, opaque_metadata)
            yield archive
            if not archive._json_hdu_added:
                raise RuntimeError("Write context exited without 'add_tree' being called.")
            hdu_list.flush()
        # This multi-open dance is necessary to get Astropy to tell us the
        # byte addresses of the HDUs.  Hopefully we can get an upstream change
        # make this unnecessary at some point.
        with astropy.io.fits.open(filename, mode="readonly", disable_image_compression=True) as hdu_list:
            index_hdu = cls._make_index_table(hdu_list)
        with astropy.io.fits.open(filename, mode="append") as hdu_list:
            hdu_list.append(index_hdu)
        json_bytes = _HDUBytes.from_index_row(index_hdu.data[-1])
        index_bytes = _HDUBytes.from_write_hdu(index_hdu)
        # Update the primary HDU with the address and size of the index and
        # JSON HDUs, and rewrite just that.  We do this write manually, since
        # astropy's docs on its 'update' mode are scarce and it's not obvious
        # whether we can guarantee it won't rewrite the whole file if we edit
        # the primary header.
        archive._primary_hdu.header["INDXADDR"] = index_bytes.header_address
        archive._primary_hdu.header["INDXSIZE"] = index_bytes.size
        archive._primary_hdu.header["JSONADDR"] = json_bytes.header_address
        archive._primary_hdu.header["JSONSIZE"] = json_bytes.size
        with open(filename, "r+b") as stream:
            stream.write(archive._primary_hdu.header.tostring().encode())

    def add_coordinate_transform(
        self, transform: CoordinateTransform, from_frame: str, to_frame: str = "sky"
    ) -> TableCellReferenceModel:
        raise NotImplementedError("TODO")

    def serialize_direct[T: pydantic.BaseModel](
        self, name: str, serializer: Callable[[OutputArchive[TableCellReferenceModel]], T]
    ) -> T:
        nested = NestedOutputArchive[TableCellReferenceModel](f"/{name}", self)
        return serializer(nested)

    def serialize_pointer[T: pydantic.BaseModel](
        self, name: str, serializer: Callable[[OutputArchive[TableCellReferenceModel]], T], key: Hashable
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
        extname = name.upper()
        hdu = self._opaque_metadata.maybe_use_precompressed(extname)
        if hdu is None:
            if (compression_options := self._get_compression_options(name)) is not None:
                hdu = compression_options.make_hdu(image.array, name=extname)
            else:
                hdu = astropy.io.fits.ImageHDU(image.array, name=extname)
        if image.unit:
            hdu.header["BUNIT"] = image.unit.to_string(format="fits")
        # TODO: add a default WCS from pixel_frame to 'sky', if pixel_frame
        # is provided and 'sky' is known to the archive; use the 'A'-suffix
        # WCS to transform from 1-indexed FITS to the image's bounding box
        # (or the default WCS otherwise); use B-Z for mappings from
        # pixel_frame to each entry in wcs_frames.
        update_header(hdu.header)
        if (opaque_headers := self._opaque_metadata.headers.get(extname)) is not None:
            hdu.header.extend(opaque_headers)
        array_model = ArrayReferenceModel(
            source=f"fits:{name.upper()}",
            shape=list(image.array.shape),
            datatype=NumberType.from_numpy(image.array.dtype),
        )
        self._hdu_list.append(hdu)
        return ImageModel.pack(array_model, start=list(image.bbox.start), unit=image.unit)

    def add_mask(
        self,
        name: str,
        mask: Mask,
        *,
        pixel_frame: str | None = None,
        wcs_frames: Iterable[str] = (),
        update_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> MaskModel:
        extname = name.upper()
        hdu = self._opaque_metadata.maybe_use_precompressed(extname)
        if hdu is None:
            if (compression_options := self._get_compression_options(name)) is not None:
                hdu = compression_options.make_hdu(mask.array, name=extname)
            else:
                hdu = astropy.io.fits.ImageHDU(mask.array, name=extname)
        # TODO: add a default WCS from pixel_frame to 'sky', if pixel_frame
        # is provided and 'sky' is known to the archive; use the 'A'-suffix
        # WCS to transform from 1-indexed FITS to the image's bounding box
        # (or the default WCS otherwise); use B-Z for mappings from
        # pixel_frame to each entry in wcs_frames.
        # TODO: write mask schema to FITS header.
        update_header(hdu.header)
        if (opaque_headers := self._opaque_metadata.headers.get(extname)) is not None:
            hdu.header.extend(opaque_headers)
        array_model = ArrayReferenceModel(
            source=f"fits:{extname}",
            shape=list(mask.array.shape),
            datatype=NumberType.from_numpy(mask.array.dtype),
        )
        self._hdu_list.append(hdu)
        return MaskModel(
            data=array_model,
            start=list(mask.bbox.start),
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

    def add_tree(self, tree: pydantic.BaseModel) -> None:
        """Write the JSON tree to the archive.

        This method must be called exactly once, just before the `open` context
        is exited.

        Parameters
        ----------
        tree
            Pydantic model that represents the tree.
        """
        json_hdu = astropy.io.fits.BinTableHDU.from_columns(
            [astropy.io.fits.Column("JSON", "PB")],
            nrows=len(self._pointer_targets) + 1,
            name="JSON",
        )
        json_hdu.data[0]["JSON"] = np.frombuffer(tree.model_dump_json().encode(), dtype=np.byte)
        for n, json_target_data in enumerate(self._pointer_targets):
            json_hdu.data[n + 1]["JSON"] = np.frombuffer(json_target_data, dtype=np.byte)
        self._hdu_list.append(json_hdu)
        self._json_hdu_added = True

    def _get_compression_options(self, name: str) -> FitsCompressionOptions | None:
        return self._compression_options.get(name, FitsCompressionOptions.DEFAULT)

    @staticmethod
    def _make_index_table(hdu_list: astropy.io.fits.HDUList) -> astropy.io.fits.BinTableHDU:
        # We use a fixed-length string for the EXTNAME column; it might be
        # better to use a variable-length array, but I have not been able to
        # figure out how to get astropy to accept a string for the the
        # character (TFORM='A') variant of that.  And that's only better if the
        # EXTNAMEs get super long, which is not likely (but maybe something to
        # guard against).
        max_name_size = max(len(hdu.header.get("EXTNAME", "")) for hdu in hdu_list)
        index_hdu = astropy.io.fits.BinTableHDU.from_columns(
            [
                astropy.io.fits.Column("EXTNAME", f"A{max_name_size}"),
                astropy.io.fits.Column("XTENSION", "A8"),
                astropy.io.fits.Column("ZIMAGE", "L"),
            ]
            + _HDUBytes.get_index_hdu_columns(),
            nrows=len(hdu_list),
            name="INDEX",
        )
        hdu: ExtensionHDU | astropy.io.fits.PrimaryHDU
        for n, hdu in enumerate(hdu_list):
            index_hdu.data[n]["EXTNAME"] = hdu.header.get("EXTNAME", "")
            index_hdu.data[n]["XTENSION"] = hdu.header.get("XTENSION", "IMAGE")
            index_hdu.data[n]["ZIMAGE"] = hdu.header.get("ZIMAGE", False)
            bytes = _HDUBytes.from_read_hdu(hdu)
            bytes.update_index_row(index_hdu.data[n])
        return index_hdu


@dataclasses.dataclass
class _HDUBytes:
    """A struct that records the byte offsets into a FITS HDU."""

    @classmethod
    def from_write_hdu(cls, hdu: astropy.io.fits.PrimaryHDU | ExtensionHDU) -> Self:
        """Construct from an Astropy HDU instance that has just been written.

        Parameters
        ----------
        hdu
            An Astropy HDU object.

        Returns
        -------
        hdu_bytes
            Struct with byte offsets.

        Notes
        -----
        This method relies on internal Astropy attributes and does not work on
        CompImageHDU objects.
        """
        # This is implemented by accessing private Astropy attributes because
        # it turns out that's much more reliable than the public fileinfo()
        # method, which seems to always return a dict with `None` entries or
        # raise; it looks buggy, but docs are scarce enough that it's not clear
        # what the right behavior is supposed to be.
        if (header_address := getattr(hdu, "_header_offset", None)) is None:
            raise RuntimeError("Failed to get Astropy's _header_offset.")
        if (data_address := getattr(hdu, "_data_offset", None)) is None:
            raise RuntimeError("Failed to get Astropy's _data_offset.")
        if (data_size := getattr(hdu, "_data_size", None)) is None:
            raise RuntimeError("Failed to get Astropy's _data_size.")
        return cls(header_address, data_address, data_size)

    @classmethod
    def from_read_hdu(cls, hdu: astropy.io.fits.PrimaryHDU | ExtensionHDU) -> Self:
        """Construct from an Astropy HDU instance that has just been read.

        Parameters
        ----------
        hdu
            An Astropy HDU object.

        Returns
        -------
        hdu_bytes
            Struct with byte offsets.
        """
        info = hdu.fileinfo()
        header_address = info["hdrLoc"]
        data_address = info["datLoc"]
        data_size = info["datSpan"]
        return cls(header_address, data_address, data_size)

    @classmethod
    def from_index_row(cls, row: np.void) -> Self:
        """Construct from a row of the index HDU.

        Parameters
        ----------
        row
            A Numpy struct-like scalar.

        Returns
        -------
        hdu_bytes
            Struct with byte offsets.
        """
        return cls(
            header_address=int(row["HDRADDR"]),
            data_address=int(row["DATADDR"]),
            data_size=int(row["DATSIZE"]),
        )

    @staticmethod
    def get_index_hdu_columns() -> list[astropy.io.fits.Column]:
        """Return the definitions of the columns this class gets and sets
        from the index HDU.

        Returns
        -------
        columns
            A `list` of `astropy.io.fits.Column` objects that represent the
            header address, data address, and data size.
        """
        return [
            astropy.io.fits.Column("HDRADDR", "K"),
            astropy.io.fits.Column("DATADDR", "K"),
            astropy.io.fits.Column("DATSIZE", "K"),
        ]

    header_address: int
    """Offset from the beginning of the start of the file to the header of this
    HDU, in bytes.
    """

    data_address: int
    """Offset from the beginning of the start of the file to the data section
    of this HDU, in bytes.
    """

    data_size: int
    """Size of the data section in bytes."""

    @property
    def header_size(self) -> int:
        """Size of the header in bytes."""
        return self.data_address - self.header_address

    @property
    def end_address(self) -> int:
        """Offset in bytes from the start of the file to the end of the HDU."""
        return self.data_address + self.data_size

    @property
    def size(self) -> int:
        """Total size of this HDU in bytes."""
        return self.data_size + self.data_address - self.header_address

    def update_index_row(self, row: np.void) -> None:
        """Set the values of a row of the index HDU from this strut.

        Parameters
        ----------
        row
            A Numpy struct-like scalar to modify in place.
        """
        row["HDRADDR"] = self.header_address
        row["DATADDR"] = self.data_address
        row["DATSIZE"] = self.data_size

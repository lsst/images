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

"""Archive implementations for saving to FITS.

The FITS archive meta-format assumes the following layout:

- a header-only primary HDU, with some special keys (prefixed with 'ADR', for
  'address') providing information about the rest of the file;

- any number of [compressed] image and binary table extension HDUs;

- a special final binary table HDU with a single row and a single
  variable-length string column, holding a JSON block.

The JSON block's contents corresponds to the Pydantic `FitsArchiveModel` class,
with the `FitsArchiveModel.tree` attribute defined by the type being saved to
the file and the other attributes for internal use by the archive
implementations.
"""

from __future__ import annotations

__all__ = (
    "FitsArchiveModel",
    "FitsCompressionOptions",
    "FitsInputArchive",
    "FitsOpaqueMetadata",
    "FitsOutputArchive",
    "InvalidFitsArchiveError",
)

import dataclasses
import io
from collections.abc import Callable, Hashable, Iterable
from typing import IO, Any, Literal

import astropy.io.fits
import astropy.table
import pydantic

from ._coordinate_transform import CoordinateTransform
from ._dtypes import NumberType
from ._geom import Box
from ._image import Image, ImageModel
from ._mask import Mask, MaskModel, MaskSchema
from .archive import InputArchive, NestedOutputArchive, OutputArchive, TableModel, no_header_updates
from .asdf_utils import ArrayReferenceModel
from .json_utils import JsonValue, PointerModel

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


class FitsExtensionModel(pydantic.BaseModel):
    """Model the describes a FITS extension.

    Addresses are typically initialized with zeros and written to the primary
    when an HDU is added to an output archive, in order to reserve space for
    the real values added later.
    """

    index: int
    """Index of this HDU (where zero is the primary HDU)."""

    name: str
    """Value of the EXTNAME keyword."""

    type: Literal["IMAGE", "BINTABLE", "COMPRESSED_IMAGE"]
    """FITS extension type."""

    header_address: int = 0
    """Offset (in bytes) from the beginning of the file to the start of the
    header.
    """

    data_address: int = 0
    """Offset (in bytes) from the beginning of the file to the start of the
    binary data.
    """

    data_size: int = 0
    """Size of the binary data (not including the header) for this extension.

    This is not stored in the header because it can be computed from the
    address of the following header or (for the last HDU) the tree address.
    """

    @property
    def end_address(self) -> int:
        return self.data_address + self.data_size

    @property
    def size(self) -> int:
        return self.end_address - self.header_address

    @pydantic.computed_field
    def target(self) -> PointerModel:
        """JSON Pointer to the JSON element that fully describes the content of
        this extension.
        """
        return PointerModel(ref=f"/tree/{self.name}")

    def set_header_addresses(self, header: astropy.io.fits.Header) -> None:
        """Write addresses to a FITS header."""
        if self.index >= 999:
            # Too much for the primary HDU.  We'll still write the addresses
            # into the JSON block at the end of the file, though.
            return
        header.set(f"ADRNM{self.index:03d}", self.name, f"EXTNAME for HDU {self.index}")
        header.set(f"ADRTP{self.index:03d}", self.type, f"Type of HDU {self.index}")
        header.set(
            f"ADRHD{self.index:03d}",
            self.header_address,
            f"Offset in bytes to the header of HDU {self.index}.",
        )
        header.set(
            f"ADRDT{self.index:03d}", self.data_address, f"Offset in bytes to the data of HDU {self.index}."
        )

    @classmethod
    def read_header_addresses(
        cls, header: astropy.io.fits.Header
    ) -> tuple[int, int, dict[str, FitsExtensionModel]]:
        extensions: dict[str, FitsExtensionModel] = {}
        last_extension: FitsExtensionModel | None = None
        for index in range(1, 1000):
            if (name := header.pop(f"ADRNM{index:03d}", None)) is None:
                break
            extension_type: str = header.pop(f"ADRTP{index:03d}")
            header_address: int = header.pop(f"ADRHD{index:03d}")
            data_address: int = header.pop(f"ADRDT{index:03d}")
            if last_extension:
                last_extension.data_size = header_address - last_extension.data_address
            last_extension = FitsExtensionModel(
                index=index,
                name=name,
                type=extension_type,
                header_address=header_address,
                data_address=data_address,
            )
            extensions[name] = last_extension
        tree_address: int = header.pop("ADRTREE")
        file_size: int = header.pop("ADRSIZE")
        if last_extension:
            last_extension.data_size = tree_address - last_extension.data_address
        return tree_address, file_size, extensions

    def read_hdu(self, stream: IO[bytes], seek: bool) -> ExtensionHDU:
        if seek:
            stream.seek(self.header_address)
        match self.type:
            case "IMAGE":
                hdu_type = astropy.io.fits.ImageHDU
            case "BINTABLE":
                hdu_type = astropy.io.fits.BinTableHDU
            case "COMPRESSED_IMAGE":
                hdu_type = astropy.io.fits.CompImageHDU
            case other:
                raise AssertionError(f"Unexpected extension type {other!r}")
        return hdu_type.readfrom(stream)


class FitsArchiveModel(pydantic.BaseModel):
    """Model that describes the JSON block of the FITS files written by this
    module.
    """

    # TODO: just using Any here (instead of generics) simplifies the archive
    # implementations, but it might make it harder to use Pydantic to extract
    # JSON Schema information to document any concrete FITS format with a
    # particular tree type.  But we could also just ask the tree model for JSON
    # schema and manually compose it into the schema for this outer model.
    tree: Any
    """Information specific to the type stored in this file.
    """

    pointers: list[JsonValue] = pydantic.Field(default_factory=list)
    """JSON-compatible raw data (nested dictionaries, etc.) for objects
    serialized as pointers to the archive.

    At present this drops any JSON schema information we might have had for
    objects serialized as pointers (it's just not something we
    """

    extensions: list[FitsExtensionModel] = pydantic.Field(default_factory=list)


class FitsOutputArchive(OutputArchive):
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
        self._pointer_targets: list[JsonValue] = []
        self._extension_models: list[FitsExtensionModel] = []
        self._primary_hdu.header.set("ADRTREE", 0, "Offset in bytes to the JSON tree HDU.")
        self._primary_hdu.header.set("ADRSIZE", 0, "Total size of the file in bytes.")
        self._compression_options = (
            compression_options if compression_options is not None else FitsCompressionOptions()
        )
        self._opaque_metadata = opaque_metadata if opaque_metadata is not None else FitsOpaqueMetadata()
        # Mapping from user provided key (e.g. id(some object)) to a JSON
        # Pointer to where we actually saved it:
        self._pointers_by_key: dict[Hashable, PointerModel] = {}

    def add_coordinate_transform(
        self, mapping: CoordinateTransform, from_frame: str, to_frame: str = "sky"
    ) -> PointerModel:
        raise NotImplementedError("TODO")

    def serialize_direct[T: pydantic.BaseModel](
        self, name: str, serializer: Callable[[OutputArchive], T]
    ) -> T:
        nested = NestedOutputArchive(f"/tree/{name}", self)
        return serializer(nested)

    def serialize_pointer[T: pydantic.BaseModel](
        self, name: str, serializer: Callable[[OutputArchive], T], key: Hashable
    ) -> T | PointerModel:
        if (pointer := self._pointers_by_key.get(key)) is not None:
            return pointer
        ref = f"/pointers/{len(self._pointer_targets)}"
        pointer = PointerModel(ref=ref)
        model = self.serialize_direct(ref, serializer)
        self._pointer_targets.append(model.model_dump())
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
        extension_model = FitsExtensionModel(index=len(self._hdu_list), name=name, type="IMAGE")
        extension_model.set_header_addresses(self._primary_hdu.header)
        self._extension_models.append(extension_model)
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
        extension_model = FitsExtensionModel(index=len(self._hdu_list), name=name, type="IMAGE")
        extension_model.set_header_addresses(self._primary_hdu.header)
        self._extension_models.append(extension_model)
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

    def write(self, filename: str, tree: Any) -> None:
        if (opaque_primary_header := self._opaque_metadata.headers.get("")) is not None:
            self._primary_hdu.header.extend(opaque_primary_header)
        # We write the file in three phases, opening it again each time. There
        # might be ways to avoid that, but this avoids relying on any astropy
        # behavior that isn't crystal-clear in the docs.
        # The first write includes a preliminary version of the primary HDU
        # (with zero placeholders for all addresses) and all of the regular
        # image and binary table extensions.
        self._hdu_list.writeto(filename)
        # Now that we've written most of the file, we can determine what all
        # of the addressses were.
        for extension in self._extension_models:
            file_info = self._hdu_list.fileinfo(extension.index)
            extension.header_address = file_info["hdrLoc"]
            extension.data_address = file_info["datLoc"]
            extension.data_size = file_info["dataSpan"]
        # The next write appends a binary table HDU with a single
        # variable-length string column and a single row, holding the JSON
        # block.  Docs are scarce on how astropy's open-for-append mode works,
        # but this is such a simple case that it's hard to imagine it not
        # working.  It might be nice to compress this, but
        archive_model = FitsArchiveModel(
            tree=tree, pointers=self._pointer_targets, extensions=self._extension_models
        )
        json_data = archive_model.model_dump_json().encode()
        json_hdu = astropy.io.fits.BinTableHDU.from_columns(
            [astropy.io.fits.Column("json", f"PA({len(json_data)})")], nrows=1
        )
        json_hdu.data[0]["json"] = json_data
        with astropy.io.fits.open(filename, mode="append") as reopened_hdu_list:
            reopened_hdu_list.append(json_hdu)
            reopened_hdu_list.flush()
        file_info = json_hdu.fileinfo()
        tree_address = file_info["hdrLoc"]
        file_size = file_info["datLoc"] + file_info["datSpan"]
        # Update the primary HDU with all of the addresses, and rewrite just
        # that.  We do this write manually, since astropy's docs on its
        # 'update' mode are scarce and it's not obvious what it's possible to
        # ask it to just rewrite the primary header and definitely not resize
        # the rest of the file.
        self._primary_hdu.header["ADRTREE"] = tree_address
        self._primary_hdu.header["ADRSIZE"] = file_size
        for extension in self._extension_models:
            extension.set_header_addresses(self._primary_hdu.header)
        with open(filename, "r+b") as stream:
            stream.write(self._primary_hdu.header.tostring().encode())


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
        # TODO: everything below is wholly untested, and it relies on guesses
        # about what astropy's 'readfrom' methods do (these look like the only
        # way to read an individual HDU from a file-like object).  Note that
        # we're no asking astropy to use fsspec because fsspec can't know about
        # the addresses we've saved and hence it has no choice but to skip
        # along 2880 bytes at a time looking for HDU boundaries, which we
        # really don't want on network storage.
        buffered_stream: IO[bytes]
        if not partial:
            buffered_stream = io.BytesIO(stream.read())
        else:
            # We don't know how big the primary header is, but we want to read
            # at least page_size at a time to avoid per-read overheads in
            # network storage.  So we use a buffered file-like object with that
            # page_size.
            buffered_stream = io.BufferedRandom(stream, page_size)
        self._primary_hdu = astropy.io.fits.PrimaryHDU.readfrom(buffered_stream)
        # TODO: read and strip subformat declaration and version, once we start
        # writing those.
        # TODO: do some basic checks that the file format conforms to our
        # expectations (e.g. primary HDU should have no data).
        self._tree_address, self._file_size, self._extension_models = (
            FitsExtensionModel.read_header_addresses(self._primary_hdu.header)
        )
        self._opaque_metadata = FitsOpaqueMetadata()
        self._opaque_metadata.headers[""] = self._primary_hdu.header.copy(strip=True)
        # We've probably now read more of the file than just the primary HDU,
        # due to either partial=False or page_size buffering.  If that includes
        # any HDUs in their entirety just initialize them now.
        # TODO: we could optimize this further by also handling the case where
        # we've read enough for the header but not the data section of an HDU.
        # It seems like that would require a custom file-like object or a new
        # lower-level astropy interface.
        buffer_end_address = stream.tell()
        self._extension_hdus: dict[str, ExtensionHDU] = {}
        for extension in self._extension_models.values():
            if extension.end_address < buffer_end_address:
                hdu_stream = io.BytesIO(buffered_stream.read(extension.size))
                self._extension_hdus[extension.name] = extension.read_hdu(hdu_stream, seek=False)
            else:
                break
        # Load the JSON tree on first use only, unless we've already read those
        # bytes, too.  It's not clear this is actually useful, if you need to
        # (e.g.) load the tree just to get an ImageModel with which to call
        # `get_image` anyway, but for prototyping we want to at least see if a
        # hyper-optimized logic path works with the interfaces, in case we need
        # it later.
        self._archive_model: FitsArchiveModel | None = None
        # We only set self._stream if we didn't already read the whole file, to
        # force an AttributeError if we try to reread something unnecessarily.
        self._stream: IO[bytes]
        if self._file_size == stream.tell():
            # If we've already read the entire file, load the JSON tree, too.
            tree_hdu = astropy.io.fits.BinTableHDU.fromstring(buffered_stream.read())
            self._archive_model = FitsArchiveModel.model_validate_json(tree_hdu.data["json"][0])
        else:
            self._stream = stream
        self._deserialized_pointer_cache: dict[int, Any] = {}

    def get_tree[T: pydantic.BaseModel](self, model_type: type[T]) -> T:
        archive_model = self._get_archive_model()
        archive_model.tree = model_type.model_validate(archive_model.tree)
        return archive_model.tree

    def get_coordinate_transform(self, from_frame: str, to_frame: str = "sky") -> CoordinateTransform:
        raise NotImplementedError("TODO")

    def deserialize_pointer[U: pydantic.BaseModel, V](
        self, pointer: PointerModel, model_type: type[U], deserializer: Callable[[U, InputArchive], V]
    ) -> V:
        if not pointer.ref.startswith("/pointers/"):
            raise InvalidFitsArchiveError(
                f"JSON pointer to serialized object {pointer.ref!r} does not start "
                "with the expected prefix ('/pointers') for a FITs archive."
            )
        try:
            index = int(pointer.ref.removeprefix("/pointers/"))
        except ValueError as err:
            raise InvalidFitsArchiveError(
                f"JSON pointer to serialized object {pointer.ref!r} does not reference "
                "an element in the /pointers/ array."
            ) from err
        if (cached := self._deserialized_pointer_cache.get(index)) is not None:
            return cached
        archive_model = self._get_archive_model()
        result = deserializer(model_type.model_validate(archive_model.pointers[index]), self)
        self._deserialized_pointer_cache[index] = result
        return result

    def get_image(
        self,
        ref: ImageModel,
        *,
        bbox: Box | None = None,
        strip_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> Image:
        array_model, unit = ref.unpack()
        name, hdu = self._get_array_hdu(array_model)
        if bbox is not None:
            array = hdu.section[ref.bbox.slice_within(bbox)]
        else:
            array = hdu.data
            bbox = ref.bbox
        if name not in self._opaque_metadata.headers:
            opaque_header = hdu.header.copy(strip=True)
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
        name, hdu = self._get_array_hdu(ref.data)
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

    def _get_archive_model(self) -> FitsArchiveModel:
        if self._archive_model is None:
            self._stream.seek(self._tree_address)
            tree_hdu = astropy.io.fits.BinTableHDU.fromstring(self._stream.read())
            self._archive_model = FitsArchiveModel.model_validate_json(tree_hdu.data["json"][0])
            if len(self._archive_model.extensions) > len(self._extension_models):
                # There were so many HDUs we didn't write all of them to the
                # primary header.  Add the remaining ones now.
                for extension_model in self._archive_model.extensions:
                    self._extension_models.setdefault(extension_model.name, extension_model)
        return self._archive_model

    def _get_hdu(self, name: str) -> ExtensionHDU:
        if (hdu := self._extension_hdus.get(name)) is None:
            if (extension_model := self._extension_models.get(name)) is None:
                # Maybe we have more than 999 extensions, so we might not know
                # about them all from reading the primary header.only.  Force a
                # read of the archive model to get a complete list.
                self._get_archive_model()
                try:
                    extension_model = self._extension_models[name]
                except KeyError:
                    raise InvalidFitsArchiveError(f"HDU with name {name!r} does not exist.") from None
            # TODO: we want to enforce the page_size passed at construction
            # here, too; probably requires a custom file-like object.
            hdu = extension_model.read_hdu(self._stream, seek=True)
            self._extension_hdus[name] = hdu
        return hdu

    def _get_array_hdu(self, ref: ArrayReferenceModel) -> tuple[str, ExtensionHDU]:
        if not ref.source.startswith("fits:"):
            raise InvalidFitsArchiveError(f"Array reference {ref.source!r} does not start with 'fits:'.")
        name = ref.source.removeprefix("fits:")
        return name, self._get_hdu(name)

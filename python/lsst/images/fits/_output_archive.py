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

__all__ = ("FitsOutputArchive", "write")

import dataclasses
from collections import Counter
from collections.abc import Callable, Hashable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any, Self

import astropy.io.fits
import astropy.table
import numpy as np
import pydantic

from .._transforms import FrameSet
from ..serialization import (
    ArchiveTree,
    ArrayReferenceModel,
    ButlerInfo,
    MetadataValue,
    NestedOutputArchive,
    NumberType,
    OutputArchive,
    TableColumnModel,
    TableModel,
    no_header_updates,
)
from ._common import (
    JSON_COLUMN,
    JSON_EXTNAME,
    ExtensionHDU,
    ExtensionKey,
    FitsCompressionOptions,
    FitsOpaqueMetadata,
    PointerModel,
)


def write(
    obj: Any,
    filename: str,
    compression_options: Mapping[str, FitsCompressionOptions | None] | None = None,
    update_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    compression_seed: int | None = None,
    metadata: dict[str, MetadataValue] | None = None,
    butler_info: ButlerInfo | None = None,
) -> Any:
    """Write an object with a ``serialize`` method to a FITS file.

    Parameters
    ----------
    filename
        Name of the file to write to.  Must not already exist.
    compression_options
        Options for how to compress the FITS file, keyed by the name of
        the attribute (with JSON pointer ``/`` separators for nested
        attributes).
    update_header
        A callback that will be given the primary HDU FITS header and an
        opportunity to modify it.
    compression_seed
        A FITS tile compression seed to use whenever the configured
        compression seed is `None` or (for backwards compatibility) ``0``.
        This value is then incremented every time it is used.
    metadata
        Additional metadata to save with the object.  This will override any
        flexible metadata carried by the object itself with the same keys.
    butler_info
        Butler information to store in the file.

    Returns
    -------
    `.serialization.ArchiveTree`
        The serialized representation of the object.
    """
    opaque_metadata = getattr(obj, "_opaque_metadata", None)
    name = getattr(obj, "_archive_default_name", None)
    with FitsOutputArchive.open(
        filename,
        compression_options=compression_options,
        opaque_metadata=opaque_metadata,
        update_header=update_header,
        compression_seed=compression_seed,
    ) as archive:
        tree = archive.serialize_direct(name, obj.serialize) if name is not None else obj.serialize(archive)
        if metadata is not None:
            tree.metadata.update(metadata)
        if butler_info is not None:
            tree.butler_info = butler_info
        archive.add_tree(tree)
    return tree


class FitsOutputArchive(OutputArchive[PointerModel]):
    """An implementation of the `.serialization.OutputArchive` interface that
    writes to FITS files.

    Instances of this class should only be constructed via the `open`
    context manager.
    """

    def __init__(
        self,
        hdu_list: astropy.io.fits.HDUList,
        compression_options: Mapping[str, FitsCompressionOptions | None] | None = None,
        opaque_metadata: Any = None,
        compression_seed: int | None = None,
    ):
        # JSON blobs for objects we've saved as pointers:
        self._pointer_targets: list[bytes] = []
        # Mapping from user provided key (e.g. id(some object)) to a table
        # pointer to where we actually saved it:
        self._pointers_by_key: dict[Hashable, PointerModel] = {}
        self._hdu_list = hdu_list
        self._primary_hdu = astropy.io.fits.PrimaryHDU()
        # TODO: add subformat description and version to primary HDU.
        self._primary_hdu.header.set("INDXADDR", 0, "Offset in bytes to the HDU index.")
        self._primary_hdu.header.set("INDXSIZE", 0, "Size of the HDU index.")
        self._primary_hdu.header.set("JSONADDR", 0, "Offset in bytes to the JSON tree HDU.")
        self._primary_hdu.header.set("JSONSIZE", 0, "Size of the JSON tree HDU.")
        self._hdus_by_name = Counter[str]()
        self._compression_options = dict(compression_options) if compression_options is not None else {}
        self._compression_seed = compression_seed
        self._opaque_metadata = (
            opaque_metadata if isinstance(opaque_metadata, FitsOpaqueMetadata) else FitsOpaqueMetadata()
        )
        if (opaque_primary_header := self._opaque_metadata.headers.get(ExtensionKey())) is not None:
            self._primary_hdu.header.extend(opaque_primary_header)
        self._hdu_list.append(self._primary_hdu)
        self._json_hdu_added: bool = False
        self._frame_sets: list[tuple[FrameSet, PointerModel]] = []

    @classmethod
    @contextmanager
    def open(
        cls,
        filename: str,
        compression_options: Mapping[str, FitsCompressionOptions | None] | None = None,
        opaque_metadata: Any = None,
        update_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
        compression_seed: int | None = None,
    ) -> Iterator[Self]:
        """Create an output archive that writes to the given file.

        Parameters
        ----------
        filename
            Name of the file to write to.  Must not already exist.
        compression_options
            Options for how to compress the FITS file, keyed by the name of
            the attribute (with JSON pointer ``/`` separators for nested
            attributes).
        opaque_metadata
            Metadata read from an input archive along with the object being
            written now.  Ignored if the metadata is not from a FITS archive.
        update_header
            A callback that will be given the primary HDU FITS header and an
            opportunity to modify it.
        compression_seed
            A FITS tile compression seed to use whenever the configured
            compression seed is `None` or (for backwards compatibility) ``0``.
            This value is then incremented every time it is used.

        Returns
        -------
        `contextlib.AbstractContextManager` [`FitsOutputArchive`]
            A context manager that returns a `FitsOutputArchive` when entered.
        """
        with astropy.io.fits.open(filename, mode="append") as hdu_list:
            if hdu_list:
                raise OSError(f"File {filename!r} already exists.")
            archive = cls(hdu_list, compression_options, opaque_metadata, compression_seed=compression_seed)
            update_header(hdu_list[0].header)
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

    def serialize_direct[T: pydantic.BaseModel](
        self, name: str, serializer: Callable[[OutputArchive[PointerModel]], T]
    ) -> T:
        nested = NestedOutputArchive[PointerModel](name, self)
        return serializer(nested)

    def serialize_pointer[T: ArchiveTree](
        self, name: str, serializer: Callable[[OutputArchive[PointerModel]], T], key: Hashable
    ) -> PointerModel:
        if (pointer := self._pointers_by_key.get(key)) is not None:
            return pointer
        pointer = PointerModel(
            column=TableColumnModel(
                name=JSON_COLUMN,
                data=ArrayReferenceModel(source=f"fits:{JSON_EXTNAME}[1]", datatype=NumberType.uint8),
            ),
            row=len(self._pointer_targets) + 1,
        )
        model = self.serialize_direct("", serializer)
        self._pointer_targets.append(model.model_dump_json().encode())
        self._pointers_by_key[key] = pointer
        return pointer

    def serialize_frame_set[T: ArchiveTree](
        self, name: str, frame_set: FrameSet, serializer: Callable[[OutputArchive], T], key: Hashable
    ) -> PointerModel:
        # Docstring inherited.
        pointer = self.serialize_pointer(name, serializer, key)
        self._frame_sets.append((frame_set, pointer))
        return pointer

    def iter_frame_sets(self) -> Iterator[tuple[FrameSet, PointerModel]]:
        return iter(self._frame_sets)

    def add_array(
        self,
        array: np.ndarray,
        *,
        name: str | None = None,
        update_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> ArrayReferenceModel:
        if name is None:
            raise RuntimeError("Cannot save array with name=None unless it is nested.")
        extname = name.upper()
        hdu = self._opaque_metadata.maybe_use_precompressed(extname)
        if hdu is None:
            if (compression_options := self._get_compression_options(name)) is not None:
                hdu = compression_options.make_hdu(array, name=extname)
            else:
                hdu = astropy.io.fits.ImageHDU(array, name=extname)
        key = self._add_hdu(hdu, update_header)
        return ArrayReferenceModel(
            source=str(key),
            shape=list(array.shape),
            datatype=NumberType.from_numpy(array.dtype),
        )

    def add_table(
        self,
        table: astropy.table.Table,
        *,
        name: str | None = None,
        update_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> TableModel:
        if name is None:
            raise RuntimeError("Cannot save table with name=None unless it is nested.")
        extname = name.upper()
        hdu: astropy.io.fits.BinTableHDU = astropy.io.fits.table_to_hdu(table, name=extname)
        # Extract column information directly from the input array, not the
        # data in the binary table HDU, because we want to assume as little as
        # possible about where Astropy does uint -> TZERO stuff.
        columns = TableColumnModel.from_table(table)
        key = self._add_hdu(hdu, update_header)
        for n, c in enumerate(columns, start=1):
            assert isinstance(c.data, ArrayReferenceModel)
            c.data.source = f"{key}[{n}]"
        return TableModel(columns=columns, meta=table.meta)

    def add_structured_array(
        self,
        array: np.ndarray,
        *,
        name: str | None = None,
        units: Mapping[str, astropy.units.Unit] | None = None,
        descriptions: Mapping[str, str] | None = None,
        update_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> TableModel:
        if name is None:
            raise RuntimeError("Cannot save structured array with name=None unless it is nested.")
        extname = name.upper()
        # Extract column information directly from the input array, not the
        # data in the binary table HDU, because we want to assume as little as
        # possible about where Astropy does uint -> TZERO stuff.
        columns = TableColumnModel.from_record_dtype(array.dtype)
        hdu = astropy.io.fits.BinTableHDU(array, name=extname)
        if units is not None:
            for c in columns:
                c.unit = units.get(c.name)
        if descriptions is not None:
            for c in columns:
                c.description = descriptions.get(c.name, "")
        key = self._add_hdu(hdu, update_header)
        for n, c in enumerate(columns, start=1):
            assert isinstance(c.data, ArrayReferenceModel)
            c.data.source = f"{key}[{n}]"
        return TableModel(columns=columns)

    def _add_hdu(
        self,
        hdu: astropy.io.fits.ImageHDU | astropy.io.fits.CompImageHDU | astropy.io.fits.BinTableHDU,
        update_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> ExtensionKey:
        n_hdus = self._hdus_by_name.get(hdu.name, 0)
        key = ExtensionKey(hdu.name, n_hdus + 1)
        key.check()
        if n_hdus:
            hdu.header["EXTVER"] = key.ver
        self._hdus_by_name[hdu.name] += 1
        update_header(hdu.header)
        if (opaque_headers := self._opaque_metadata.headers.get(key)) is not None:
            hdu.header.extend(opaque_headers)
        self._hdu_list.append(hdu)
        return key

    def add_tree(self, tree: ArchiveTree) -> None:
        """Write the JSON tree to the archive.

        This method must be called exactly once, just before the `open` context
        is exited.

        Parameters
        ----------
        tree
            Pydantic model that represents the tree.
        """
        json_hdu = astropy.io.fits.BinTableHDU.from_columns(
            [astropy.io.fits.Column(JSON_COLUMN, "PB")],
            nrows=len(self._pointer_targets) + 1,
            name=JSON_EXTNAME,
        )
        json_hdu.data[0][JSON_COLUMN] = np.frombuffer(tree.model_dump_json().encode(), dtype=np.byte)
        for n, json_target_data in enumerate(self._pointer_targets):
            json_hdu.data[n + 1][JSON_COLUMN] = np.frombuffer(json_target_data, dtype=np.byte)
        self._hdu_list.append(json_hdu)
        self._json_hdu_added = True

    def _get_compression_options(self, name: str) -> FitsCompressionOptions | None:
        result = self._compression_options.get(name, FitsCompressionOptions.DEFAULT)
        if result is None or result.quantization is None:
            return result
        if self._compression_seed is not None and not result.quantization.seed:
            result = result.model_copy(
                update={
                    "quantization": result.quantization.model_copy(update={"seed": self._compression_seed})
                }
            )
            self._compression_seed += 1
            if self._compression_seed > 10000:
                self._compression_seed = 1
        # MyPy can tell that result.quantization is not None in the 'if', but
        # forgets that by this 'else':
        elif result.quantization.seed is None:  # type: ignore[union-attr]
            raise RuntimeError("No quantization seed provided.")
        return result

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
                astropy.io.fits.Column("EXTVER", "J"),
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
            index_hdu.data[n]["EXTVER"] = hdu.header.get("EXTVER", 1)
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

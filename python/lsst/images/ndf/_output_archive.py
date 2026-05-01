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
    "NdfOutputArchive",
    "write",
)

from collections.abc import Callable, Hashable, Iterator, Mapping
from typing import Any

import astropy.io.fits
import astropy.table
import astropy.units
import h5py
import numpy as np
import pydantic

from .._transforms import FrameSet
from ..fits._common import ExtensionKey, FitsOpaqueMetadata
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
from . import _hds
from ._common import NdfPointerModel, json_pointer_to_hdf5_path


def write(
    obj: Any,
    filename: str | None = None,
    *,
    metadata: dict[str, MetadataValue] | None = None,
    butler_info: ButlerInfo | None = None,
    update_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    compression_options: Mapping[str, Any] | None = None,
) -> ArchiveTree:
    """Write a serializable object to an NDF (HDS-on-HDF5) file.

    Parameters
    ----------
    obj
        Object with a ``serialize`` method. May carry an
        ``_opaque_metadata`` attribute (a :class:`FitsOpaqueMetadata`)
        whose primary-HDU header gets written to ``/MORE/FITS``.
    filename
        Path to write to. If `None`, an in-memory HDF5 file is used and
        the on-disk artefact is discarded; the returned tree still
        reflects all the writes the archive made (useful for tests).
    metadata, butler_info
        Optional caller-supplied entries that are written into the
        returned :class:`ArchiveTree`.
    update_header
        Callback that mutates the opaque primary FITS header, used by
        the butler formatter to inject provenance.
    compression_options
        Optional dict forwarded to the archive constructor for h5py
        dataset compression.

    Returns
    -------
    `~lsst.images.serialization.ArchiveTree`
        The Pydantic tree the object's ``serialize`` produced (with
        ``metadata``/``butler_info`` applied).
    """
    if filename is None:
        h5_file = h5py.File("inmem.sdf", "w", driver="core", backing_store=False)
    else:
        h5_file = h5py.File(filename, "w")
    try:
        opaque_metadata = getattr(obj, "_opaque_metadata", None)
        if not isinstance(opaque_metadata, FitsOpaqueMetadata):
            opaque_metadata = FitsOpaqueMetadata()
        primary_header = opaque_metadata.headers.get(ExtensionKey()) or astropy.io.fits.Header()
        update_header(primary_header)
        if len(primary_header):
            opaque_metadata.headers[ExtensionKey()] = primary_header

        archive = NdfOutputArchive(
            h5_file,
            compression_options=compression_options,
            opaque_metadata=opaque_metadata,
        )

        archive_default_name = getattr(obj, "_archive_default_name", None)
        if archive_default_name is not None:
            tree = archive.serialize_direct(archive_default_name, obj.serialize)
        else:
            tree = obj.serialize(archive)

        if metadata is not None:
            tree.metadata.update(metadata)
        if butler_info is not None:
            tree.butler_info = butler_info

        # Write the canonical NDF /WCS component (HDS structure containing
        # an AST channel text dump) when the object has a non-None
        # projection. This is what KAPPA / hdstrace expect for an NDF's
        # WCS; it lives alongside the projection's Pydantic model in
        # /MORE/LSST/JSON, which is the source of truth for our own
        # symmetric round-trip. The same WCS is also written into the
        # mask sub-NDF at /MORE/LSST/MASK/WCS (if a sub-NDF was created
        # for an incompatible mask) so Starlink tools displaying the
        # mask as an image use the same projection as the parent.
        # Auto-detect read of /WCS/DATA is a deferred follow-up: building
        # a typed Projection from a bare FrameSet requires new
        # infrastructure in _transforms/.
        projection = getattr(obj, "projection", None)
        if projection is not None:
            ast_frame_set = projection._pixel_to_sky._get_ast_frame_set()
            text = ast_frame_set.show(False)
            lines = text.splitlines()
            width = max(80, max((len(line) for line in lines), default=0) + 1)
            wcs_parents: list[h5py.Group] = [h5_file["/"]]
            if "/MORE/LSST/MASK" in h5_file:
                wcs_parents.append(h5_file["/MORE/LSST/MASK"])
            for parent in wcs_parents:
                if "WCS" in parent:
                    del parent["WCS"]
                wcs_group = _hds.create_structure(parent, "WCS", "WCS")
                _hds.write_char_array(wcs_group, "DATA", lines, width=width)

        # Main JSON tree.
        json_text = tree.model_dump_json()
        more_lsst = archive._ensure_path("/MORE/LSST")
        if "JSON" in more_lsst:
            del more_lsst["JSON"]
        _hds.write_char_array(more_lsst, "JSON", [json_text], width=max(80, len(json_text)))

        # Opaque FITS cards in /MORE/FITS.
        primary = opaque_metadata.headers.get(ExtensionKey())
        if primary is not None and len(primary):
            cards = [card.image for card in primary.cards]
            more = archive._ensure_path("/MORE")
            if "FITS" in more:
                del more["FITS"]
            _hds.write_char_array(more, "FITS", cards, width=80)

        # Backfill bbox-derived ORIGIN for DATA_ARRAY and VARIANCE.
        bbox = getattr(obj, "bbox", None)
        if bbox is not None:
            origin = _origin_from_bbox(bbox)
            for struct_path in (
                "/DATA_ARRAY",
                "/VARIANCE",
                "/MORE/LSST/MASK/DATA_ARRAY",
            ):
                if struct_path in h5_file:
                    archive.set_array_origin(struct_path, origin)

        # Mark the root group with HDS_ROOT_NAME (CLASS=NDF was set by the
        # archive constructor in Task 7) using fixed-length ASCII bytes
        # so KAPPA / hdstrace can decode the attribute.
        root_name = archive_default_name or type(obj).__name__
        _hds.set_root_name(h5_file, root_name, "NDF")

        return tree
    finally:
        h5_file.close()


def _origin_from_bbox(bbox: Any) -> tuple[int, ...]:
    """Extract NDF/Fortran-order origin tuple from an lsst.images Box.

    The exact attribute names on Box depend on the version. Inspect the
    object and pick whatever exposes the integer lower bounds. For a 2D
    image with bbox lower bound (x_min, y_min) the returned tuple is
    ``(x_min, y_min)``.
    """
    # Box exposes .x and .y properties returning Interval objects with a
    # .start attribute (the lower bound).
    if hasattr(bbox, "x") and hasattr(bbox, "y"):
        x = bbox.x
        y = bbox.y
        if hasattr(x, "start") and hasattr(y, "start"):
            return (int(x.start), int(y.start))
    raise AttributeError(
        f"Don't know how to extract origin from bbox of type {type(bbox).__name__!r}; "
        "_origin_from_bbox needs updating."
    )


class NdfOutputArchive(OutputArchive[NdfPointerModel]):
    """An :class:`~lsst.images.serialization.OutputArchive` implementation
    that writes HDS-on-HDF5 files compatible with the Starlink NDF data
    model.

    Parameters
    ----------
    file
        An open ``h5py.File`` opened in a writable mode. The archive does
        not close the file; the caller is responsible for that.
    compression_options
        Optional dict passed through to ``h5py.create_dataset`` for image
        arrays (e.g. ``{"compression": "gzip", "compression_opts": 4}``).
        Reserved for use by `add_array` (Task 8).
    opaque_metadata
        Optional :class:`FitsOpaqueMetadata`; if its primary-HDU header is
        non-empty its cards will be written to ``/MORE/FITS`` by the
        top-level write() function (Task 11).
    """

    def __init__(
        self,
        file: h5py.File,
        compression_options: Mapping[str, Any] | None = None,
        opaque_metadata: FitsOpaqueMetadata | None = None,
    ) -> None:
        self._file = file
        self._compression_options = dict(compression_options) if compression_options else {}
        self._opaque_metadata = opaque_metadata if opaque_metadata is not None else FitsOpaqueMetadata()
        self._frame_sets: list[tuple[FrameSet, NdfPointerModel]] = []
        self._pointers: dict[Hashable, NdfPointerModel] = {}
        # Mark the root group as a top-level NDF if not already marked.
        if _hds.ATTR_CLASS not in self._file["/"].attrs:
            _hds.set_ascii_attr(self._file["/"], _hds.ATTR_CLASS, "NDF")

    def serialize_direct[T: pydantic.BaseModel](
        self, name: str, serializer: Callable[[OutputArchive[NdfPointerModel]], T]
    ) -> T:
        nested = NestedOutputArchive[NdfPointerModel](name, self)
        return serializer(nested)

    def serialize_pointer[T: ArchiveTree](
        self, name: str, serializer: Callable[[OutputArchive[NdfPointerModel]], T], key: Hashable
    ) -> NdfPointerModel:
        if (pointer := self._pointers.get(key)) is not None:
            return pointer
        json_pointer = name if name.startswith("/") else f"/{name}"
        path = json_pointer_to_hdf5_path(json_pointer)
        # Run the serializer first so any nested add_array / serialize_pointer
        # calls write into the file before we dump this sub-tree to JSON.
        model = self.serialize_direct(name, serializer)
        json_text = model.model_dump_json()
        parent_path, leaf = path.rsplit("/", 1)
        parent = self._ensure_path(parent_path)
        if leaf in parent:
            del parent[leaf]
        _hds.write_char_array(parent, leaf, [json_text], width=max(80, len(json_text)))
        pointer = NdfPointerModel(ref=path)
        self._pointers[key] = pointer
        return pointer

    def serialize_frame_set[T: ArchiveTree](
        self, name: str, frame_set: FrameSet, serializer: Callable[[OutputArchive], T], key: Hashable
    ) -> NdfPointerModel:
        pointer = self.serialize_pointer(name, serializer, key)
        self._frame_sets.append((frame_set, pointer))
        return pointer

    def iter_frame_sets(self) -> Iterator[tuple[FrameSet, NdfPointerModel]]:
        return iter(self._frame_sets)

    _COMPATIBLE_MASK_DTYPES = (np.dtype(np.uint8),)
    _prefer_native_mask_arrays = True

    def add_array(
        self,
        array: np.ndarray,
        *,
        name: str | None = None,
        update_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> ArrayReferenceModel:
        # Recognised top-level names go to standard NDF locations.
        # Anything else hoists under /MORE/LSST.
        if name == "image":
            self._ensure_array_structure("/DATA_ARRAY")
            path = "/DATA_ARRAY/DATA"
            self._write_origin_for_array("/DATA_ARRAY", array)
        elif name == "variance":
            self._ensure_array_structure("/VARIANCE")
            path = "/VARIANCE/DATA"
            self._write_origin_for_array("/VARIANCE", array)
        elif name == "mask":
            if array.ndim == 2 and array.dtype in self._COMPATIBLE_MASK_DTYPES:
                self._ensure_quality_structure()
                path = "/QUALITY/QUALITY"
            else:
                # Native Mask serialization writes the 3-D uint8 mask here
                # with HDF5 axes reversed from the HDS axes so Starlink sees
                # dimensions (x, y, mask-byte).
                self._ensure_struct("/MORE/LSST/MASK", "NDF")
                self._ensure_array_structure("/MORE/LSST/MASK/DATA_ARRAY")
                path = "/MORE/LSST/MASK/DATA_ARRAY/DATA"
                self._write_origin_for_array("/MORE/LSST/MASK/DATA_ARRAY", array)
        else:
            if name is None:
                raise ValueError("Anonymous arrays are not supported in the NDF archive.")
            json_pointer = name if name.startswith("/") else f"/{name}"
            # Hoisted numeric arrays are wrapped as sub-NDFs under
            # /MORE/LSST/<UPPER_PATH> so Starlink tools (KAPPA `display`,
            # `hdstrace`, etc.) can inspect them just like the main
            # image. The sub-NDF has the canonical layout: top-level
            # group with CLASS="NDF" containing a DATA_ARRAY structure
            # (CLASS="ARRAY") with DATA + ORIGIN primitives. Hoisted
            # JSON sub-trees from serialize_pointer stay as bare
            # _CHAR*N datasets at /MORE/LSST/<NAME> (no NDF wrapper) —
            # they're JSON documents, not numeric arrays.
            sub_ndf_path = json_pointer_to_hdf5_path(json_pointer)
            self._ensure_struct(sub_ndf_path, "NDF")
            self._ensure_array_structure(f"{sub_ndf_path}/DATA_ARRAY")
            path = f"{sub_ndf_path}/DATA_ARRAY/DATA"
            self._write_origin_for_array(f"{sub_ndf_path}/DATA_ARRAY", array)
        parent_path, leaf = path.rsplit("/", 1)
        parent = self._ensure_path(parent_path)
        _hds.write_array(
            parent,
            leaf,
            array,
            compression=self._compression_options.get("compression"),
        )
        # Shape is stored in the JSON tree (matching the FITS archive) because
        # MaskSerializationModel.bbox needs it before any arrays are read.
        # Future work: resolve shape from the HDF5 dataset on read instead.
        return ArrayReferenceModel(
            source=f"ndf:{path}",
            shape=list(array.shape),
            datatype=NumberType.from_numpy(array.dtype),
        )

    def _ensure_path(self, path: str) -> h5py.Group:
        """Walk/create groups for an HDF5 absolute path.

        Intermediate groups created on demand are tagged with
        ``CLASS="EXT"``, the HDS type for general-purpose extension
        containers (matches what hds-v5 writes for ``MORE``).
        """
        if path in ("", "/"):
            return self._file["/"]
        parts = path.lstrip("/").split("/")
        cursor: h5py.Group = self._file["/"]
        for part in parts:
            if part not in cursor:
                cursor = _hds.create_structure(cursor, part, "EXT")
            else:
                cursor = cursor[part]
        return cursor

    def _ensure_struct(self, path: str, hds_type: str) -> h5py.Group:
        """Ensure a structure exists at ``path`` with the given HDS type."""
        if path in self._file:
            return self._file[path]
        parent_path, leaf = path.rsplit("/", 1)
        parent = self._ensure_path(parent_path or "/")
        return _hds.create_structure(parent, leaf, hds_type)

    def _ensure_array_structure(self, path: str) -> h5py.Group:
        """Ensure an HDS ``ARRAY`` structure exists at ``path``."""
        return self._ensure_struct(path, "ARRAY")

    def _ensure_quality_structure(self) -> h5py.Group:
        """Ensure ``/QUALITY`` exists with ``CLASS="QUALITY"`` and BADBITS.

        BADBITS defaults to 0xFF (treat all defined plane bits as
        "bad"); a future enhancement may make this configurable.
        """
        if "/QUALITY" in self._file:
            return self._file["/QUALITY"]
        group = _hds.create_structure(self._file, "QUALITY", "QUALITY")
        _hds.write_array(group, "BADBITS", np.array(0xFF, dtype=np.uint8))
        return group

    def _write_origin_for_array(self, struct_path: str, array: np.ndarray) -> None:
        """Write a placeholder ORIGIN of zeros (int64).

        The top-level ``write()`` function (Task 11) overwrites this
        with bbox-derived values via :meth:`set_array_origin` once the
        bbox is known.
        """
        struct = self._file[struct_path]
        if "ORIGIN" not in struct:
            _hds.write_array(struct, "ORIGIN", np.zeros(array.ndim, dtype=np.int64))

    def set_array_origin(self, struct_path: str, origin: tuple[int, ...]) -> None:
        """Overwrite the ORIGIN of an ARRAY structure.

        Parameters
        ----------
        struct_path
            HDF5 path to the ARRAY structure (e.g. ``"/DATA_ARRAY"``).
        origin
            Origin in NDF/Fortran axis order (e.g. ``(x_min, y_min)``
            for a 2D image with bbox lower bound ``(x_min, y_min)``).
        """
        struct = self._file[struct_path]
        if "ORIGIN" in struct:
            del struct["ORIGIN"]
        origin_array = np.asarray(origin, dtype=np.int64)
        if (
            "DATA" in struct
            and isinstance(struct["DATA"], h5py.Dataset)
            and origin_array.size < struct["DATA"].ndim
        ):
            origin_array = np.pad(origin_array, (0, struct["DATA"].ndim - origin_array.size))
        _hds.write_array(struct, "ORIGIN", origin_array)

    def add_table(
        self,
        table: astropy.table.Table,
        *,
        name: str | None = None,
        update_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> TableModel:
        columns = TableColumnModel.from_table(table, inline=True)
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
        columns = TableColumnModel.from_record_array(array, inline=True)
        for c in columns:
            if units and (unit := units.get(c.name)):
                c.unit = unit
            if descriptions and (description := descriptions.get(c.name)):
                c.description = description
        return TableModel(columns=columns)

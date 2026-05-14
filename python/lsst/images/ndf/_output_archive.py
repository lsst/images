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

import os
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any, Self

import astropy.io.fits
import astropy.table
import astropy.units
import h5py
import numpy as np
import pydantic

from .._color_image import ColorImage
from .._transforms import FrameSet
from .._transforms._ast import Channel, CmpFrame, CmpMap, ShiftMap, StringStream, UnitMap
from .._transforms._ast import Frame as AstFrame
from .._transforms._ast import FrameSet as AstFrameSet
from .._transforms._transform import _prepend_ast_shift
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
from ._common import NdfPointerModel, archive_path_to_hdf5_path, archive_path_to_hdf5_path_components
from ._model import HdsPrimitive, HdsStructure, Ndf, NdfArray, NdfContainer, NdfDocument, NdfQuality, NdfWcs


def write(
    obj: Any,
    filename: str | None = None,
    *,
    metadata: dict[str, MetadataValue] | None = None,
    butler_info: ButlerInfo | None = None,
    compression_options: Mapping[str, Any] | None = None,
) -> ArchiveTree:
    """Write a serializable object to an NDF (HDS-on-HDF5) file.

    Parameters
    ----------
    obj
        Object with a ``serialize`` method. May carry an
        ``_opaque_metadata`` attribute (a
        `~lsst.images.fits.FitsOpaqueMetadata`)
        whose primary-HDU header gets written to ``/MORE/FITS``. This
        preserves FITS cards on objects that originated from a FITS read;
        butler provenance is conveyed through ``butler_info`` instead.
    filename
        Path to write to.  Must not already exist.  If `None`, an
        in-memory HDF5 file is used and the on-disk artefact is
        discarded; the returned tree still reflects all the writes the
        archive made (useful for tests).
    metadata, butler_info
        Optional caller-supplied entries that are written into the
        returned `~lsst.images.serialization.ArchiveTree`.
    compression_options
        Optional dict forwarded to the archive constructor for h5py
        dataset compression.

    Returns
    -------
    `~lsst.images.serialization.ArchiveTree`
        The Pydantic tree the object's ``serialize`` produced (with
        ``metadata``/``butler_info`` applied).
    """
    opaque_metadata = getattr(obj, "_opaque_metadata", None)
    if not isinstance(opaque_metadata, FitsOpaqueMetadata):
        opaque_metadata = FitsOpaqueMetadata()
    archive_default_name = getattr(obj, "_archive_default_name", None)
    with NdfOutputArchive.open(
        filename,
        compression_options=compression_options,
        opaque_metadata=opaque_metadata,
        **_get_archive_layout(obj),
    ) as archive:
        if archive_default_name is not None:
            tree = archive.serialize_direct(archive_default_name, obj.serialize)
        else:
            tree = obj.serialize(archive)
        if metadata is not None:
            tree.metadata.update(metadata)
        if butler_info is not None:
            tree.butler_info = butler_info
        archive.add_tree(
            tree,
            projection=getattr(obj, "projection", None),
            bbox=getattr(obj, "bbox", None),
            unit=getattr(obj, "unit", None),
            root_name=archive_default_name or type(obj).__name__,
        )
    return tree


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


def _unit_to_ndf_string(unit: astropy.units.UnitBase) -> str:
    """Return an ASCII unit string for the NDF UNITS component."""
    try:
        return unit.to_string(format="fits")
    except ValueError:
        return unit.to_string()


def _fits_header_records(header: astropy.io.fits.Header) -> list[str]:
    """Return fixed-width FITS records for an opaque NDF FITS extension.

    NDF ``.MORE.FITS`` is a ``_CHAR*80`` vector.  Use Astropy's FITS
    header serializer to preserve multi-record logical cards such as
    CONTINUE strings, but omit the FITS END card and 2880-byte padding
    because this is an NDF extension component, not a complete FITS
    header block.
    """
    block = header.tostring(sep="", endcard=False, padding=False)
    encoded = block.encode("ascii")
    if len(encoded) % 80:
        raise ValueError(
            f"FITS header block is {len(encoded)} bytes, not a multiple of the 80-byte FITS record size."
        )
    return [block[n : n + 80] for n in range(0, len(block), 80)]


def _get_archive_layout(obj: Any) -> dict[str, Any]:
    """Return NDF document layout options for a top-level object."""
    if isinstance(obj, ColorImage):
        return {
            "root": NdfContainer(),
            "lsst_path": "/LSST",
            "direct_ndf_array_paths": {
                "red": "/RED",
                "green": "/GREEN",
                "blue": "/BLUE",
            },
            "wcs_ndf_paths": ("/RED", "/GREEN", "/BLUE"),
        }
    return {}


def _show_ast_for_ndf(ast_frame_set: Any, bbox: Any | None) -> str:
    """Return AST Channel text matching Starlink NDF WCS serialization.

    Tags the original base frame with ``Domain="PIXEL"`` and prepends a
    new ``Domain="GRID"`` base frame related to it by a `ShiftMap` whose
    shift converts ``bbox``-origin pixel coordinates into 1-based grid
    coordinates. The result is written via an abstraction-layer
    ``Channel`` configured with the same options the Starlink C writer
    uses (``Full=-1,Comment=0``; see ``ndf1Wwrt.c``) plus ``Indent=0`` so
    each line is just the bare AST item with the single-space prefix
    that ``ndf1Rdast`` strips back off on read.
    """
    if bbox is None:
        x_shift = 1.0
        y_shift = 1.0
    else:
        x_shift = 1.0 - float(bbox.x.start)
        y_shift = 1.0 - float(bbox.y.start)

    saved_current = ast_frame_set.current
    ast_frame_set.current = ast_frame_set.base
    ast_frame_set.domain = "PIXEL"
    ast_frame_set.current = saved_current
    _prepend_ast_shift(ast_frame_set, x=x_shift, y=y_shift, ast_domain="GRID")

    stream = StringStream()
    channel = Channel(stream, options="Full=-1,Comment=0,Indent=0")
    channel.write(ast_frame_set)
    return stream.getSinkData()


def _show_mask_ast_for_ndf(
    parent_ast_frame_set: Any,
    origin: Sequence[int],
    *,
    labels: Sequence[str] = (),
) -> str:
    """Return an NDF WCS for the 3D native mask sub-NDF.

    The first two axes reuse the parent image's pixel-to-sky mapping.  The
    third axis is a generic mask-byte coordinate that passes through unchanged.
    """
    n_axes = len(origin)
    ast_frame_set = AstFrameSet(AstFrame(n_axes, "Domain=GRID"))
    pixel_frame = AstFrame(n_axes, "Domain=PIXEL")
    for axis, label in enumerate(labels[:n_axes], start=1):
        pixel_frame.setLabel(axis, label)
    shifts = [1.0 - float(axis_origin) for axis_origin in origin]
    ast_frame_set.addFrame(AstFrameSet.BASE, ShiftMap(shifts), pixel_frame)

    parent_pixel_to_sky = parent_ast_frame_set.getMapping(
        parent_ast_frame_set.base,
        parent_ast_frame_set.current,
    )
    parent_sky_frame = parent_ast_frame_set.getFrame(parent_ast_frame_set.current)
    mask_axis_frame = AstFrame(1, "Domain=MASK")
    mask_axis_frame.setLabel(1, labels[2] if len(labels) > 2 else "mask-byte")
    ast_frame_set.addFrame(
        ast_frame_set.current,
        CmpMap(parent_pixel_to_sky, UnitMap(1), False),
        CmpFrame(parent_sky_frame, mask_axis_frame),
    )

    stream = StringStream()
    channel = Channel(stream, options="Full=-1,Comment=0,Indent=0")
    channel.write(ast_frame_set)
    return stream.getSinkData()


class NdfOutputArchive(OutputArchive[NdfPointerModel]):
    """An `~lsst.images.serialization.OutputArchive` implementation
    that writes HDS-on-HDF5 files compatible with the Starlink NDF data
    model.

    Parameters
    ----------
    file
        An open `h5py.File` opened in a writable mode. The archive does
        not close the file; the caller is responsible for that.
    compression_options
        Optional dict passed through to `h5py.Group.create_dataset` for image
        arrays (e.g. ``{"compression": "gzip", "compression_opts": 4}``).
    opaque_metadata
        Optional `~lsst.images.fits.FitsOpaqueMetadata`; if its primary-HDU
        header is non-empty its cards will be written to ``/MORE/FITS`` by the
        top-level `write` function.
    """

    def __init__(
        self,
        file: h5py.File,
        compression_options: Mapping[str, Any] | None = None,
        opaque_metadata: FitsOpaqueMetadata | None = None,
        root: Ndf | NdfContainer | None = None,
        lsst_path: str = "/MORE/LSST",
        direct_ndf_array_paths: Mapping[str, str] | None = None,
        wcs_ndf_paths: Sequence[str] = ("/",),
    ) -> None:
        self._file = file
        self._document = NdfDocument(root=root if root is not None else Ndf())
        self._lsst_path = lsst_path.rstrip("/") or "/LSST"
        self._direct_ndf_array_paths = dict(direct_ndf_array_paths) if direct_ndf_array_paths else {}
        self._wcs_ndf_paths = tuple(wcs_ndf_paths)
        self._bbox_array_struct_paths: set[str] = set()
        self._compression_options = dict(compression_options) if compression_options else {}
        self._opaque_metadata = opaque_metadata if opaque_metadata is not None else FitsOpaqueMetadata()
        self._frame_sets: list[tuple[FrameSet, NdfPointerModel]] = []
        self._pointers: dict[Hashable, NdfPointerModel] = {}
        # Keep the open file in sync so existing direct-archive tests can
        # inspect it immediately, while all mutations go through the IR.
        self._flush()

    @classmethod
    @contextmanager
    def open(
        cls,
        filename: str | None,
        *,
        compression_options: Mapping[str, Any] | None = None,
        opaque_metadata: FitsOpaqueMetadata | None = None,
        root: Ndf | NdfContainer | None = None,
        lsst_path: str = "/MORE/LSST",
        direct_ndf_array_paths: Mapping[str, str] | None = None,
        wcs_ndf_paths: Sequence[str] = ("/",),
    ) -> Iterator[Self]:
        """Open an NDF file for writing and yield an `NdfOutputArchive`.

        ``filename=None`` uses an in-memory HDF5 file; the on-disk
        artefact is discarded but the archive's writes still produce a
        usable returned tree (handy for tests).
        """
        if filename is None:
            h5_file = h5py.File("inmem.sdf", "w", driver="core", backing_store=False)
        else:
            if os.path.exists(filename) and os.path.getsize(filename) > 0:
                raise OSError(f"File {filename!r} already exists.")
            h5_file = h5py.File(filename, "w")
        try:
            yield cls(
                h5_file,
                compression_options=compression_options,
                opaque_metadata=opaque_metadata,
                root=root,
                lsst_path=lsst_path,
                direct_ndf_array_paths=direct_ndf_array_paths,
                wcs_ndf_paths=wcs_ndf_paths,
            )
        finally:
            h5_file.close()

    def add_tree(
        self,
        tree: ArchiveTree,
        *,
        projection: Any = None,
        bbox: Any = None,
        unit: astropy.units.UnitBase | None = None,
        root_name: str | None = None,
    ) -> None:
        """Finalize the file: write WCS, units, JSON tree, and ORIGIN.

        Writes the canonical NDF ``/WCS`` HDS structure (an AST channel
        text dump that KAPPA / hdstrace expect) when ``projection`` is
        provided. A native mask sub-NDF at ``/MORE/LSST/MASK`` gets a 3D
        WCS whose first two axes reuse the parent's sky projection and
        whose third axis is the mask-byte coordinate. The JSON tree at
        ``<lsst_path>/JSON`` remains the source of truth for symmetric
        round-trips; ``/WCS`` is for Starlink tools. Auto-detect read of
        ``/WCS/DATA`` into a typed ``Projection`` is a follow-up.

        Parameters
        ----------
        tree
            Pydantic tree returned by the object's ``serialize`` method,
            with ``metadata``/``butler_info`` already applied.
        projection, bbox, unit
            Top-level object attributes that drive NDF-canonical writes.
        root_name
            Value to assign to the root group's ``HDS_ROOT_NAME``
            attribute (fixed-length ASCII so KAPPA / hdstrace decode it).
        """
        if projection is not None:
            self._write_wcs(projection, bbox)
        if unit is not None and isinstance(self._document.root, Ndf):
            self._document.ensure_ndf("/").set_units(_unit_to_ndf_string(unit))
        json_text = tree.model_dump_json()
        lsst = self._ensure_model_structure(self._lsst_path, "EXT")
        lsst.children["JSON"] = HdsPrimitive.char_array([json_text], width=max(80, len(json_text)))
        primary = self._opaque_metadata.headers.get(ExtensionKey())
        if primary is not None and len(primary):
            cards = _fits_header_records(primary)
            more = self._ensure_model_structure("/MORE", "EXT")
            more.children["FITS"] = HdsPrimitive.char_array(cards, width=80)
        if bbox is not None:
            origin = _origin_from_bbox(bbox)
            for struct_path in self._bbox_array_struct_paths:
                if self._has_model_path(struct_path):
                    self.set_array_origin(struct_path, origin)
        if root_name is not None:
            self._document.root_name = root_name
        self._flush()

    def _write_wcs(self, projection: Any, bbox: Any) -> None:
        ast_frame_set = projection._pixel_to_sky._get_ast_frame_set()
        text = _show_ast_for_ndf(ast_frame_set, bbox)
        lines = _hds.encode_ndf_ast_data(text)
        for ndf_path in self._wcs_ndf_paths:
            if self._has_model_path(ndf_path):
                self._document.ensure_ndf(ndf_path).set_wcs(NdfWcs(lines))
        if self._has_model_path("/MORE/LSST/MASK"):
            mask_origin = _origin_from_bbox(bbox) if bbox is not None else (0, 0)
            mask_ndim = self._model_array_ndim("/MORE/LSST/MASK/DATA_ARRAY")
            mask_origin = (*mask_origin, *((0,) * max(0, mask_ndim - len(mask_origin))))
            mask_ast_frame_set = projection._pixel_to_sky._get_ast_frame_set()
            mask_text = _show_mask_ast_for_ndf(
                mask_ast_frame_set,
                mask_origin,
                labels=("x", "y", "mask-byte"),
            )
            self._document.ensure_ndf("/MORE/LSST/MASK").set_wcs(NdfWcs(_hds.encode_ndf_ast_data(mask_text)))

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
        archive_path = name if name.startswith("/") else f"/{name}"
        path = self._archive_path_to_hdf5_path(archive_path)
        # Run the serializer first so any nested add_array / serialize_pointer
        # calls write into the file before we dump this sub-tree to JSON.
        model = self.serialize_direct(name, serializer)
        json_text = model.model_dump_json()
        parent_path, leaf = path.rsplit("/", 1)
        parent = self._ensure_model_structure(parent_path, "EXT")
        parent.children[leaf] = HdsPrimitive.char_array([json_text], width=max(80, len(json_text)))
        self._flush()
        pointer = NdfPointerModel(path=path)
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
            root = self._document.ensure_ndf("/")
            root.set_array_component(
                "DATA_ARRAY",
                array,
                origin=np.zeros(array.ndim, dtype=np.int64),
                compression_options=self._compression_options,
            )
            path = "/DATA_ARRAY/DATA"
            self._bbox_array_struct_paths.add("/DATA_ARRAY")
        elif name == "variance":
            root = self._document.ensure_ndf("/")
            root.set_array_component(
                "VARIANCE",
                array,
                origin=np.zeros(array.ndim, dtype=np.int64),
                compression_options=self._compression_options,
            )
            path = "/VARIANCE/DATA"
            self._bbox_array_struct_paths.add("/VARIANCE")
        elif name == "mask":
            if array.ndim == 2 and array.dtype in self._COMPATIBLE_MASK_DTYPES:
                self._set_quality_array(array)
                path = "/QUALITY/QUALITY/DATA"
                self._bbox_array_struct_paths.add("/QUALITY/QUALITY")
            else:
                # Native Mask serialization passes HDF5 shape
                # (mask-byte, y, x). HDS reports the reverse dimension order,
                # so Starlink tools see (x, y, mask-byte).
                if array.ndim == 3 and array.dtype in self._COMPATIBLE_MASK_DTYPES:
                    self._set_quality_array(self._collapse_mask_to_quality(array))
                    self._bbox_array_struct_paths.add("/QUALITY/QUALITY")
                mask_ndf = self._document.ensure_ndf("/MORE/LSST/MASK")
                mask_ndf.set_array_component(
                    "DATA_ARRAY",
                    array,
                    origin=np.zeros(array.ndim, dtype=np.int64),
                    bad_pixel=False,
                    compression_options=self._compression_options,
                )
                path = "/MORE/LSST/MASK/DATA_ARRAY/DATA"
                self._bbox_array_struct_paths.add("/MORE/LSST/MASK/DATA_ARRAY")
        elif name in self._direct_ndf_array_paths:
            sub_ndf_path = self._direct_ndf_array_paths[name]
            sub_ndf = self._document.ensure_ndf(sub_ndf_path)
            sub_ndf.set_array_component(
                "DATA_ARRAY",
                array,
                origin=np.zeros(array.ndim, dtype=np.int64),
                compression_options=self._compression_options,
            )
            path = f"{sub_ndf_path}/DATA_ARRAY/DATA"
            self._bbox_array_struct_paths.add(f"{sub_ndf_path}/DATA_ARRAY")
        else:
            if name is None:
                raise ValueError("Anonymous arrays are not supported in the NDF archive.")
            archive_path = name if name.startswith("/") else f"/{name}"
            # Hoisted numeric arrays are wrapped as sub-NDFs under
            # /MORE/LSST/<UPPER_PATH> so Starlink tools (KAPPA `display`,
            # `hdstrace`, etc.) can inspect them just like the main
            # image. The sub-NDF has the canonical layout: top-level
            # group with CLASS="NDF" containing a DATA_ARRAY structure
            # (CLASS="ARRAY") with DATA + ORIGIN primitives. Hoisted
            # JSON sub-trees from serialize_pointer stay as bare
            # _CHAR*N datasets at /MORE/LSST/<NAME> (no NDF wrapper) —
            # they're JSON documents, not numeric arrays.
            sub_ndf_path = self._archive_path_to_hdf5_path(archive_path)
            sub_ndf = self._document.ensure_ndf(sub_ndf_path)
            sub_ndf.set_array_component(
                "DATA_ARRAY",
                array,
                origin=np.zeros(array.ndim, dtype=np.int64),
                compression_options=self._compression_options,
            )
            path = f"{sub_ndf_path}/DATA_ARRAY/DATA"
        self._flush()
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
        self._ensure_model_structure(path, "EXT")
        self._flush()
        return self._file["/" if path in ("", "/") else path]

    def _ensure_struct(self, path: str, hds_type: str) -> h5py.Group:
        """Ensure a structure exists at ``path`` with the given HDS type."""
        self._ensure_model_structure(path, hds_type)
        self._flush()
        return self._file["/" if path in ("", "/") else path]

    def _ensure_array_structure(self, path: str) -> h5py.Group:
        """Ensure an HDS ``ARRAY`` structure exists at ``path``."""
        return self._ensure_struct(path, "ARRAY")

    def _ensure_quality_structure(self) -> h5py.Group:
        """Ensure ``/QUALITY`` exists with ``CLASS="QUALITY"`` and BADBITS.

        BADBITS is set to 255 so every bit of the collapsed single-byte
        QUALITY plane is treated as bad by NDF applications.
        """
        root = self._document.ensure_ndf("/")
        if "QUALITY" not in root.children:
            root.children["QUALITY"] = HdsStructure("QUALITY")
        quality = root.get_structure("QUALITY")
        quality.hds_type = "QUALITY"
        quality.children["BADBITS"] = HdsPrimitive.array(np.array(255, dtype=np.uint8))
        self._flush()
        return self._file["/QUALITY"]

    def _ensure_quality_array_structure(self) -> h5py.Group:
        """Ensure the nested ``/QUALITY/QUALITY`` ARRAY structure exists."""
        root = self._document.ensure_ndf("/")
        if "QUALITY" not in root.children:
            root.children["QUALITY"] = HdsStructure("QUALITY")
        quality = root.get_structure("QUALITY")
        quality.hds_type = "QUALITY"
        quality.children["BADBITS"] = HdsPrimitive.array(np.array(255, dtype=np.uint8))
        if "QUALITY" not in quality.children or not isinstance(quality.children["QUALITY"], HdsStructure):
            quality.children["QUALITY"] = HdsStructure("ARRAY")
        quality_array = quality.get_structure("QUALITY")
        quality_array.hds_type = "ARRAY"
        quality_array.children.setdefault("ORIGIN", HdsPrimitive.array(np.zeros(2, dtype=np.int32)))
        quality_array.children.setdefault("BAD_PIXEL", HdsPrimitive.array(np.array(False, dtype=np.bool_)))
        self._flush()
        return self._file["/QUALITY/QUALITY"]

    def _write_quality_array(self, quality: np.ndarray) -> None:
        """Write or replace the NDF QUALITY array."""
        self._set_quality_array(quality)
        self._flush()

    def _set_quality_array(self, quality: np.ndarray) -> None:
        """Set or replace the NDF QUALITY array in the model."""
        root = self._document.ensure_ndf("/")
        root.set_quality(
            NdfQuality(
                NdfArray(
                    quality,
                    origin=np.zeros(2, dtype=np.int32),
                    bad_pixel=False,
                    compression_options=self._compression_options,
                )
            )
        )

    def _collapse_mask_to_quality(self, array: np.ndarray) -> np.ndarray:
        """Compress an NDF-native 3-D mask array into 2-D QUALITY.

        The input array is in HDF5 storage order ``(mask-byte, y, x)``.
        Single-byte masks copy directly to preserve bit values.  Wider masks
        collapse to 1 where any byte is non-zero and 0 otherwise.
        """
        if array.shape[0] == 1:
            return array[0, :, :]
        return np.any(array != 0, axis=0).astype(np.uint8)

    def _write_origin_for_array(self, struct_path: str, array: np.ndarray) -> None:
        """Write a placeholder ORIGIN of zeros (int64).

        The top-level `write` function overwrites this
        with bbox-derived values via :meth:`set_array_origin` once the
        bbox is known.
        """
        struct = self._document.root.get_structure(struct_path)
        if "ORIGIN" not in struct.children:
            struct.children["ORIGIN"] = HdsPrimitive.array(np.zeros(array.ndim, dtype=np.int64))

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
        struct = self._document.root.get_structure(struct_path)
        origin_dtype = np.int32 if struct_path == "/QUALITY/QUALITY" else np.int64
        origin_array = np.asarray(origin, dtype=origin_dtype)
        data_node = struct.children.get("DATA")
        if isinstance(data_node, HdsPrimitive):
            data_ndim = data_node.read_array().ndim
            if origin_array.size < data_ndim:
                origin_array = np.pad(origin_array, (0, data_ndim - origin_array.size))
        struct.children["ORIGIN"] = HdsPrimitive.array(origin_array)
        self._flush()

    def _ensure_model_structure(self, path: str, hds_type: str) -> HdsStructure:
        """Return or create a structure in the NDF document model."""
        if path in ("", "/"):
            return self._document.root
        return self._document.root.ensure_structure(path, hds_type)

    def _archive_path_to_hdf5_path(self, archive_path: str) -> str:
        """Translate an archive path to this layout's HDF5 path."""
        if self._lsst_path == "/MORE/LSST":
            return archive_path_to_hdf5_path(archive_path)
        if not archive_path:
            return f"{self._lsst_path}/JSON"
        components = archive_path_to_hdf5_path_components(archive_path)
        return f"{self._lsst_path}/{'/'.join(components)}"

    def _has_model_path(self, path: str) -> bool:
        """Return `True` if a path exists in the NDF document model."""
        try:
            self._document.get(path)
        except KeyError:
            return False
        return True

    def _model_array_ndim(self, struct_path: str) -> int:
        """Return the dimensionality of an ARRAY structure's DATA primitive."""
        struct = self._document.root.get_structure(struct_path)
        data_node = struct.children["DATA"]
        if not isinstance(data_node, HdsPrimitive):
            raise TypeError(f"{struct_path}/DATA is not an HDS primitive.")
        return data_node.read_array().ndim

    def _flush(self) -> None:
        """Synchronize the Python NDF document model to the open HDF5 file."""
        self._document.write_to_hdf5(self._file)

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
        if name is None:
            columns = TableColumnModel.from_record_array(array, inline=True)
            for c in columns:
                if units and (unit := units.get(c.name)):
                    c.unit = unit
                if descriptions and (description := descriptions.get(c.name)):
                    c.description = description
            return TableModel(columns=columns)
        columns = TableColumnModel.from_record_dtype(array.dtype)
        for c in columns:
            column_path = name if len(columns) == 1 else f"{name}/{c.name}"
            archive_path = column_path if column_path.startswith("/") else f"/{column_path}"
            sub_ndf_path = self._archive_path_to_hdf5_path(archive_path)
            column_array = np.asarray(array[c.name])
            sub_ndf = self._document.ensure_ndf(sub_ndf_path)
            sub_ndf.set_array_component(
                "DATA_ARRAY",
                column_array,
                origin=np.zeros(column_array.ndim, dtype=np.int64),
                compression_options=self._compression_options,
            )
            assert isinstance(c.data, ArrayReferenceModel)
            c.data.source = f"ndf:{sub_ndf_path}/DATA_ARRAY/DATA"
        for c in columns:
            if units and (unit := units.get(c.name)):
                c.unit = unit
            if descriptions and (description := descriptions.get(c.name)):
                c.description = description
        self._flush()
        return TableModel(columns=columns)

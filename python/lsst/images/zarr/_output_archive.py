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

__all__ = ("ZarrOutputArchive", "write")

from collections.abc import Callable, Hashable, Iterator, Mapping
from typing import Any, ClassVar, cast

import astropy.io.fits
import astropy.table
import astropy.units
import numpy as np
import pydantic
import zarr

from .._mask import Mask, MaskSchema
from .._transforms import FrameSet
from .._transforms._ast import Channel, StringStream
from ..fits._common import FitsOpaqueMetadata
from ..serialization import (
    ArchiveTree,
    ArrayReferenceModel,
    NestedOutputArchive,
    NumberType,
    OutputArchive,
    TableColumnModel,
    TableModel,
    no_header_updates,
)
from ._common import (
    DEFAULT_TARGET_SHARD_BYTES,
    ZarrCompressionOptions,
    ZarrPointerModel,
    archive_path_to_zarr_path,
    mask_dtype_for_plane_count,
)
from ._layout import (
    affine_check,
    axes_for_archive_class,
    chunks_aligned_to,
    chunks_for,
    decorate_sub_archives,
    default_shards,
    serialize_fits_opaque_metadata,
)
from ._model import (
    CfFlagAttributes,
    MaskPlaneEntry,
    OmeMultiscale,
    ZarrArray,
    ZarrDocument,
    ZarrGroup,
    build_image_array_attrs,
)
from ._store import open_store_for_write


class ZarrOutputArchive(OutputArchive[ZarrPointerModel]):
    """Output archive that populates a ``ZarrDocument`` IR.

    Bytes are not written until the IR is materialized via
    ``ZarrDocument.to_zarr``, which the public `write` helper performs
    on context-manager exit.

    Parameters
    ----------
    chunks
        Per-array chunk overrides keyed by the array's archive path
        (e.g. ``"image"``). ``None`` for a key means "use the layout
        default".
    shards, compression
        Same shape as ``chunks``.
    archive_class
        Top-level archive class name (``"VisitImage"``, ``"CellCoadd"``,
        …). Used by the layout layer to pick chunk defaults; set by
        ``write()`` before ``obj.serialize`` runs so ``add_array``
        sees the right value.
    archive_metadata
        Class-specific layout hints (``cell_shape`` for ``CellCoadd``,
        ``mask_schema`` for the mask packer).
    """

    _prefer_native_mask_arrays: ClassVar[bool] = True
    """Tell ``Mask.serialize`` to hand us the 3-D ``(y, x, mask_size)``
    array in one ``add_array`` call. ``add_array`` packs it into a 2-D
    wide-integer array on disk with CF ``flag_masks`` / ``flag_meanings``
    attributes.
    """

    def __init__(
        self,
        *,
        chunks: Mapping[str, tuple[int, ...] | None] | None = None,
        shards: Mapping[str, tuple[int, ...] | None] | None = None,
        compression: Mapping[str, ZarrCompressionOptions | None] | None = None,
        archive_class: str = "Image",
        archive_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.document = ZarrDocument(root=ZarrGroup())
        self._chunks = dict(chunks) if chunks else {}
        self._shards = dict(shards) if shards else {}
        self._compression = dict(compression) if compression else {}
        self._archive_class = archive_class
        self._archive_metadata = dict(archive_metadata) if archive_metadata else {}
        self._pointers: dict[Hashable, ZarrPointerModel] = {}
        self._frame_sets: list[tuple[FrameSet, ZarrPointerModel]] = []
        self._image_chunks: tuple[int, ...] | None = None

    def serialize_direct[T: pydantic.BaseModel](
        self,
        name: str,
        serializer: Callable[[OutputArchive[ZarrPointerModel]], T],
    ) -> T:
        nested = NestedOutputArchive[ZarrPointerModel](name, self)
        return serializer(nested)

    def serialize_pointer[T: ArchiveTree](
        self,
        name: str,
        serializer: Callable[[OutputArchive[ZarrPointerModel]], T],
        key: Hashable,
    ) -> ZarrPointerModel:
        if (cached := self._pointers.get(key)) is not None:
            return cached
        archive_path = name if name.startswith("/") else f"/{name}"
        sub_zarr_path = archive_path_to_zarr_path(archive_path)
        # Run the serializer first so any nested add_array calls land
        # inside the IR before we dump this sub-tree to JSON.
        model = self.serialize_direct(name, serializer)
        json_bytes = model.model_dump_json().encode("utf-8")
        parent = self.document.root.ensure_group(sub_zarr_path)
        # Single-chunk storage: the JSON tree is always read whole.
        tree_data = np.frombuffer(json_bytes, dtype=np.uint8)
        parent.arrays["lsst_json"] = ZarrArray(data=tree_data, chunks=tree_data.shape)
        pointer = ZarrPointerModel(path=f"{sub_zarr_path}/lsst_json")
        self._pointers[key] = pointer
        return pointer

    def serialize_frame_set[T: ArchiveTree](
        self,
        name: str,
        frame_set: FrameSet,
        serializer: Callable[[OutputArchive], T],
        key: Hashable,
    ) -> ZarrPointerModel:
        pointer = self.serialize_pointer(name, serializer, key)
        self._frame_sets.append((frame_set, pointer))
        return pointer

    def iter_frame_sets(self) -> Iterator[tuple[FrameSet, ZarrPointerModel]]:
        return iter(self._frame_sets)

    def add_array(
        self,
        array: np.ndarray,
        *,
        name: str | None = None,
        update_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> ArrayReferenceModel:
        if name is None:
            raise ValueError("Anonymous arrays are not supported in ZarrOutputArchive.")
        archive_path = name if name.startswith("/") else f"/{name}"
        zarr_path = archive_path_to_zarr_path(archive_path)
        leaf = zarr_path.rsplit("/", 1)[-1]
        parent_path = zarr_path[: -(len(leaf) + 1)] or "/"
        parent = self.document.root.ensure_group(parent_path)

        # Mask: pack 3-D (mask_size, y, x) -> 2-D (y, x) wide-int packed.
        # Mask.serialize emits the byte axis first when the archive opts in
        # via _prefer_native_mask_arrays (matching the HDF5/NDF convention);
        # we undo that so the on-disk array is the natural xarray layout.
        if leaf == "mask" and array.ndim == 3:
            array = np.moveaxis(array, 0, -1)
            packed, flag_attrs = self._pack_mask(array)
            chunks = self._chunks.get(name) or self._chunks.get(leaf)
            if chunks is None and self._image_chunks is not None:
                chunks = chunks_aligned_to(image_chunks=self._image_chunks, shape=packed.shape)
            shards = self._shards.get(name) or self._shards.get(leaf)
            if shards is None and chunks is not None:
                shards = default_shards(
                    chunks=tuple(chunks),
                    shape=tuple(packed.shape),
                    dtype=packed.dtype,
                    target_bytes=DEFAULT_TARGET_SHARD_BYTES,
                )
            extra: dict[str, Any] = {"_ARRAY_DIMENSIONS": ["y", "x"]}
            extra.update(flag_attrs.dump())
            ir_array = ZarrArray(
                data=packed,
                chunks=chunks,
                shards=shards,
                compression=self._compression.get(name),
            )
            ir_array.attributes.extra = extra
            parent.arrays[leaf] = ir_array
            # The model reports the schema's element dtype (uint8 /
            # uint16 / ...) so the input archive can recover the
            # original ``(y, x, mask_size)`` array; the on-disk array
            # itself is the wide packed integer.
            return ArrayReferenceModel(
                source=f"zarr:{zarr_path}",
                shape=list(packed.shape),
                datatype=NumberType.from_numpy(array.dtype),
            )

        chunks = self._chunks.get(name) or self._chunks.get(leaf)
        # variance / other top-level siblings: align to image's chunks.
        if (
            chunks is None
            and self._image_chunks is not None
            and parent_path == "/"
            and leaf == "variance"
            and array.ndim == len(self._image_chunks)
        ):
            chunks = chunks_aligned_to(image_chunks=self._image_chunks, shape=array.shape)

        # Default chunks for the top-level image: from layout rules.
        if chunks is None and parent_path == "/" and leaf == "image":
            chunks = chunks_for(
                self._archive_class,
                array.shape,
                None,
                archive_metadata=self._archive_metadata,
            )

        # Default chunks for a CellCoadd-style 4-D PSF: one cell per chunk.
        if chunks is None and leaf == "psf" and array.ndim == 4 and parent_path == "/":
            chunks = (1, 1, array.shape[2], array.shape[3])

        shards = self._shards.get(name) or self._shards.get(leaf)
        if shards is None and chunks is not None:
            shards = default_shards(
                chunks=tuple(chunks),
                shape=tuple(array.shape),
                dtype=array.dtype,
                target_bytes=DEFAULT_TARGET_SHARD_BYTES,
            )
        ir_array = ZarrArray(
            data=np.ascontiguousarray(array),
            chunks=chunks,
            shards=shards,
            compression=self._compression.get(name),
        )
        if parent_path == "/" and leaf in ("image", "variance"):
            ir_array.attributes.extra = build_image_array_attrs(
                axes=("y", "x"),
                long_name="science image" if leaf == "image" else "image variance",
            )
        parent.arrays[leaf] = ir_array

        # Remember the image's chunks so siblings can align.
        if parent_path == "/" and leaf == "image" and chunks is not None:
            self._image_chunks = tuple(chunks)

        return ArrayReferenceModel(
            source=f"zarr:{zarr_path}",
            shape=list(array.shape),
            datatype=NumberType.from_numpy(array.dtype),
        )

    def _pack_mask(self, array: np.ndarray) -> tuple[np.ndarray, CfFlagAttributes]:
        """Pack a 3-D ``(y, x, mask_size)`` mask into a 2-D wide-int array.

        The schema is taken from ``self._archive_metadata["mask_schema"]``.
        Returns the packed array and the CF flag attributes.
        """
        schema = self._archive_metadata.get("mask_schema")
        if not isinstance(schema, MaskSchema):
            raise ValueError(
                "Writing a 3-D mask requires archive_metadata['mask_schema'] "
                "to be set; the output archive cannot infer the plane "
                "definitions otherwise."
            )
        n_planes = len(schema)
        target_dtype = mask_dtype_for_plane_count(n_planes)
        # Pack: each (y, x) pixel's mask_size schema-dtype elements
        # become one wide integer. Element 0 occupies bits
        # [0, stride), element 1 occupies [stride, 2*stride), etc.,
        # where stride = 8 * schema.dtype.itemsize. Plane N therefore
        # lives at packed bit position N, matching the CF flag_masks
        # attribute (1 << N).
        stride = 8 * array.dtype.itemsize
        packed = np.zeros(array.shape[:2], dtype=target_dtype)
        for i in range(array.shape[2]):
            packed |= array[..., i].astype(target_dtype) << (stride * i)
        # ``MaskSchema`` may carry ``None`` placeholders for retired plane
        # bits; drop them in the CF flag list.
        planes = [
            MaskPlaneEntry(name=p.name, bit=i, description=p.description)
            for i, p in enumerate(schema)
            if p is not None
        ]
        return packed, CfFlagAttributes(planes=planes)

    def add_table(
        self,
        table: astropy.table.Table,
        *,
        name: str | None = None,
        update_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> TableModel:
        if name is None:
            raise ValueError("Anonymous tables are not supported in ZarrOutputArchive.")
        columns = TableColumnModel.from_table(table)
        archive_path = name if name.startswith("/") else f"/{name}"
        table_zarr_path = f"/lsst/tables{archive_path}"
        parent = self.document.root.ensure_group(table_zarr_path)
        for c in columns:
            assert isinstance(c.data, ArrayReferenceModel)
            column_array = np.ascontiguousarray(np.asarray(table[c.name]))
            parent.arrays[c.name] = ZarrArray(data=column_array)
            c.data.source = f"zarr:{table_zarr_path}/{c.name}"
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
            raise ValueError("Anonymous structured arrays are not supported.")
        columns = TableColumnModel.from_record_dtype(array.dtype)
        archive_path = name if name.startswith("/") else f"/{name}"
        table_zarr_path = f"/lsst/tables{archive_path}"
        parent = self.document.root.ensure_group(table_zarr_path)
        for c in columns:
            assert isinstance(c.data, ArrayReferenceModel)
            column_array = np.ascontiguousarray(array[c.name])
            parent.arrays[c.name] = ZarrArray(data=column_array)
            c.data.source = f"zarr:{table_zarr_path}/{c.name}"
            if units and (unit := units.get(c.name)):
                c.unit = unit
            if descriptions and (description := descriptions.get(c.name)):
                c.description = description
        return TableModel(columns=columns)

    def add_tree(self, tree: ArchiveTree) -> None:
        """Finalize the IR: write JSON tree, WCS, and root attributes.

        Called once after the user's serializer has populated arrays
        / sub-trees. Sets the ``lsst.*`` and ``ome.*`` blocks on the
        root group, stages ``/lsst_json`` as 1-D ``uint8`` UTF-8 JSON,
        and runs the affine residual validator if the archive carries
        a frame set.
        """
        # Stage the JSON tree at /lsst_json (single chunk — read whole).
        # Name mirrors NDF's /MORE/LSST/JSON and FITS's "JSON" HDU.
        json_bytes = tree.model_dump_json().encode("utf-8")
        tree_data = np.frombuffer(json_bytes, dtype=np.uint8)
        self.document.root.arrays["lsst_json"] = ZarrArray(data=tree_data, chunks=tree_data.shape)

        # Stage the AST WCS string at /wcs_ast when a frame set is registered.
        wcs_ast_path: str | None = None
        if self._frame_sets:
            wcs_ast_path = self._stage_wcs_ast(self._frame_sets[0][0])

        # Root LSST attrs.
        lsst = self.document.root.attributes.lsst
        lsst["archive_class"] = self._archive_class
        lsst["json"] = "lsst_json"
        if wcs_ast_path is not None:
            lsst["wcs_ast"] = wcs_ast_path
        if "cell_grid" in self._archive_metadata:
            lsst["cell_grid"] = self._archive_metadata["cell_grid"]

        # Canonical data-model schema URL, mirroring the FITS DATAMODL keyword
        # and the NDF .MORE.LSST.DATA_MODEL component. Informational on read:
        # the JSON tree's own schema_version / min_read_version drive
        # data-model compatibility. The container (file-format) version travels
        # separately as the lsst.version attribute (see ZarrAttributes.dump).
        lsst["data_model"] = tree.schema_url

        # OME multiscale block, gated by axes_for_archive_class.
        axes = axes_for_archive_class(self._archive_class)
        if axes and "image" in self.document.root.arrays:
            image_array = self.document.root.arrays["image"]
            ct: list[dict[str, Any]] | None = None
            if self._frame_sets:
                fs = self._frame_sets[0][0]
                if len(image_array.shape) != 2:
                    raise ValueError(
                        f"Top-level image must be 2-D for the OME affine "
                        f"check; got shape {image_array.shape}."
                    )
                image_shape: tuple[int, int] = (image_array.shape[0], image_array.shape[1])
                check = affine_check(
                    frame_set=fs,
                    image_shape=image_shape,
                    max_residual_pixels=1.0,
                )
                lsst["wcs_simplified_dropped"] = check.dropped
                if not check.dropped:
                    ct = check.coordinate_transformations
            multiscale = OmeMultiscale(
                name=self._archive_class.lower(),
                axes=axes,
                dataset_path="image",
                coordinate_transformations=ct,
            )
            self.document.root.attributes.ome["multiscales"] = [multiscale.dump()]

        # Walk sub-groups and decorate each one that holds an ``image``
        # array (e.g. ``ColorImage`` channels) as its own valid Image
        # sub-archive with OME multiscales.
        decorate_sub_archives(self.document)

    def _stage_wcs_ast(self, frame_set: FrameSet) -> str:
        """Encode an AST FrameSet as UTF-8 text and stage at /wcs_ast.

        Currently dead — left for future use; see ``add_tree``'s frame-set
        hook.
        """
        from .._transforms._ast import Object as _AstObject

        stream = StringStream()
        # FrameSet inherits from Object in our AST bridge; cast for the
        # ``Channel.write`` signature which is typed against the base class.
        Channel(stream, options="Full=-1,Comment=0,Indent=0").write(cast(_AstObject, frame_set))
        text = stream.getSinkData()
        wcs_data = np.frombuffer(text.encode("utf-8"), dtype=np.uint8)
        # Single chunk: WCS is always read whole.
        self.document.root.arrays["wcs_ast"] = ZarrArray(data=wcs_data, chunks=wcs_data.shape)
        return "wcs_ast"


def build_archive_metadata(obj: Any) -> dict[str, Any]:
    """Resolve layout-affecting metadata from an in-memory archive object.

    The output archive's chunk and metadata rules consult
    ``cell_shape`` (used by `~lsst.images.zarr._layout.chunks_for` to
    align chunks to a `CellCoadd`'s cells) and ``mask_schema`` (used
    by `_pack_mask` to produce CF flag attributes). Different archive
    classes expose this information under different attribute names:

    - ``Image``: nothing (no cell grid, no mask schema).
    - ``MaskedImage``: ``mask.schema``.
    - ``Mask``: ``schema`` directly on the object.
    - ``CellCoadd``: ``mask.schema`` and ``grid.cell_shape``.

    Returns a flat ``dict`` ready to pass as
    ``ZarrOutputArchive(archive_metadata=...)``. Keys are present
    only when a value was found.
    """
    metadata: dict[str, Any] = {}
    cell_shape = _resolve_cell_shape(obj)
    if cell_shape is not None:
        metadata["cell_shape"] = cell_shape
    mask_schema = _resolve_mask_schema(obj)
    if mask_schema is not None:
        metadata["mask_schema"] = mask_schema
    return metadata


def _resolve_cell_shape(obj: Any) -> tuple[int, ...] | None:
    """Return the cell shape as a ``(y, x)`` tuple, or ``None``.

    Tries ``obj.cell_shape`` first, then ``obj.grid.cell_shape``
    (used by `CellCoadd`), then ``obj.cell_grid.cell_shape``.
    """
    direct = getattr(obj, "cell_shape", None)
    if direct is not None:
        return tuple(direct)
    grid = getattr(obj, "grid", None)
    if grid is None:
        grid = getattr(obj, "cell_grid", None)
    if grid is not None:
        nested = getattr(grid, "cell_shape", None)
        if nested is not None:
            return tuple(nested)
    return None


def _resolve_mask_schema(obj: Any) -> MaskSchema | None:
    """Return the mask schema, or ``None`` if the object has no mask."""
    direct = getattr(obj, "mask_schema", None)
    if direct is not None:
        return direct
    mask = getattr(obj, "mask", None)
    if mask is not None:
        nested = getattr(mask, "schema", None)
        if nested is not None:
            return nested
    if isinstance(obj, Mask):
        # Top-level Mask: schema is on the object itself.
        return obj.schema
    return None


def write(
    obj: Any,
    path: Any,
    *,
    chunks: Mapping[str, tuple[int, ...] | None] | None = None,
    shards: Mapping[str, tuple[int, ...] | None] | None = None,
    compression: Mapping[str, ZarrCompressionOptions | None] | None = None,
    metadata: Mapping[str, Any] | None = None,
    butler_info: Any | None = None,
) -> ArchiveTree:
    """Write ``obj`` to a zarr archive at ``path``.

    Parameters mirror the FITS / NDF write helpers. The store
    implementation (LocalStore / ZipStore / FsspecStore) is selected
    from the URI shape by ``_store.open_store_for_write``.
    """
    archive_class = type(obj).__name__
    archive_default_name = getattr(obj, "_archive_default_name", None)
    archive_metadata = build_archive_metadata(obj)

    archive = ZarrOutputArchive(
        chunks=chunks,
        shards=shards,
        compression=compression,
        archive_class=archive_class,
        archive_metadata=archive_metadata,
    )
    if archive_default_name is not None:
        tree = archive.serialize_direct(archive_default_name, obj.serialize)
    else:
        tree = obj.serialize(archive)
    if metadata is not None:
        tree.metadata.update(metadata)
    if butler_info is not None:
        tree.butler_info = butler_info
    archive.add_tree(tree)
    # Stage opaque metadata after add_tree so the namespace attribute
    # writes happen in the right order.
    opaque = getattr(obj, "_opaque_metadata", None)
    if isinstance(opaque, FitsOpaqueMetadata):
        serialize_fits_opaque_metadata(archive.document, opaque)
    with open_store_for_write(path) as store:
        archive.document.to_zarr(store)
        # Consolidate metadata so a single read fetches the whole
        # hierarchy's zarr.json contents — significant perf win on
        # remote stores. Skip for ZipStore: a zip is read end-to-end
        # locally, so consolidation buys nothing, and rewriting the
        # root zarr.json into an append-only zip emits a spurious
        # "Duplicate name" warning.
        if not isinstance(store, zarr.storage.ZipStore):
            try:
                zarr.consolidate_metadata(store)
            except TypeError:
                pass
    return tree

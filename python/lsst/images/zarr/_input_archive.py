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

__all__ = ("ZarrInputArchive",)

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import EllipsisType
from typing import Any, Self

import astropy.io.fits
import astropy.table
import numpy as np

from lsst.resources import ResourcePathExpression

from .._transforms import FrameSet
from ..fits._common import FitsOpaqueMetadata
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
from ._common import LSST_VERSION, ZarrPointerModel
from ._layout import deserialize_fits_opaque_metadata
from ._model import ZarrArray, ZarrDocument
from ._store import open_store_for_read


class ZarrInputArchive(InputArchive[ZarrPointerModel]):
    """Reads zarr archives written by `ZarrOutputArchive`.

    Parameters
    ----------
    document
        In-memory IR for the open archive, built by
        ``ZarrDocument.from_zarr``.
    """

    def __init__(self, document: ZarrDocument) -> None:
        self._document = document
        self._validate_root_attributes()
        self._deserialized_pointer_cache: dict[str, Any] = {}
        self._frame_set_cache: dict[str, FrameSet] = {}
        self._opaque_metadata = deserialize_fits_opaque_metadata(document)

    def get_opaque_metadata(self) -> FitsOpaqueMetadata | None:
        """Return any FITS opaque metadata recovered from the archive."""
        return self._opaque_metadata

    @classmethod
    @contextmanager
    def open(cls, path: ResourcePathExpression) -> Iterator[Self]:
        """Open a zarr archive for reading.

        Parameters
        ----------
        path
            URI or path of the zarr archive to open.
        """
        with open_store_for_read(path) as store:
            doc = ZarrDocument.from_zarr(store)
            yield cls(doc)

    @classmethod
    def get_basic_info(cls, path: ResourcePathExpression) -> ArchiveInfo:
        """Return the schema URL and container version from the root group.

        Reads only the root group's ``lsst`` attributes (``data_model``
        and ``version``); building the lazy IR does not fetch pixel data.

        Parameters
        ----------
        path
            URI or path of the zarr archive to inspect.
        """
        with open_store_for_read(path) as store:
            doc = ZarrDocument.from_zarr(store)
        attrs = doc.root.attributes.lsst
        schema_url = attrs.get("data_model")
        if not schema_url:
            raise ArchiveReadError(
                f"{path!r} is not an lsst.images zarr archive (no lsst.data_model attribute)."
            )
        format_version = int(attrs.get("__version_remembered_at_load__", 1))
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
        """Open the zarr archive and yield ``(archive, tree, info)``.

        The schema is read from the open document's root attributes rather
        than a separate `get_basic_info` open.  Zarr reads are always lazy,
        so ``partial`` is accepted for interface compatibility but has no
        effect.

        Parameters
        ----------
        path
            URI or path of the zarr archive to open.
        partial
            Accepted for interface compatibility; ignored because zarr
            reads are always lazy.
        **backend_kwargs
            Accepted for interface compatibility; ignored.
        """
        with cls.open(path) as archive:
            info = archive.info
            tree_cls = tree_class_for_info(info, path)
            parameterized = parameterize_tree(tree_cls, ZarrPointerModel)
            tree = archive.get_tree(parameterized)
            yield archive, tree, info

    @property
    def info(self) -> ArchiveInfo:
        """Schema/format info read from the open document's root attributes."""
        attrs = self._document.root.attributes.lsst
        schema_url = attrs.get("data_model")
        if not schema_url:
            raise ArchiveReadError("This is not an lsst.images zarr archive (no lsst.data_model attribute).")
        format_version = int(attrs.get("__version_remembered_at_load__", 1))
        return ArchiveInfo.from_schema_url(schema_url, format_version=format_version)

    @property
    def document(self) -> ZarrDocument:
        return self._document

    def get_tree[T: ArchiveTree](self, model_type: type[T]) -> T:
        """Read and validate the main Pydantic tree at ``/lsst_json``.

        Parameters
        ----------
        model_type
            Pydantic tree class to validate the ``/lsst_json`` bytes
            against.
        """
        try:
            node = self._document.root.get("/lsst_json")
        except KeyError:
            raise ArchiveReadError(
                "File has no /lsst_json array; this is not an LSST zarr archive."
            ) from None
        if not isinstance(node, ZarrArray):
            raise ArchiveReadError("/lsst_json must be a zarr array, not a group.")
        json_bytes = bytes(node.read())
        return model_type.model_validate_json(json_bytes.decode("utf-8"))

    def _validate_root_attributes(self) -> None:
        attrs = self._document.root.attributes.lsst
        if "archive_class" not in attrs:
            raise ArchiveReadError("File is not an LSST zarr archive (missing lsst.archive_class).")
        # lsst.version is the container (file-format) layout version; absence
        # is treated as 1. lsst.data_model is informational only — the JSON
        # tree's schema_version / min_read_version drive data-model checks.
        on_disk = int(attrs.get("__version_remembered_at_load__", 1))
        _check_format_version("zarr", on_disk, LSST_VERSION)

    def deserialize_pointer[U: ArchiveTree, V](
        self,
        pointer: ZarrPointerModel,
        model_type: type[U],
        deserializer: Callable[[U, InputArchive[ZarrPointerModel]], V],
    ) -> V:
        if (cached := self._deserialized_pointer_cache.get(pointer.path)) is not None:
            return cached
        try:
            node = self._document.root.get(pointer.path)
        except KeyError:
            raise ArchiveReadError(f"Pointer reference {pointer.path!r} not in store.") from None
        if not isinstance(node, ZarrArray):
            raise ArchiveReadError(f"Pointer target {pointer.path!r} is not an array.")
        json_text = bytes(node.read()).decode("utf-8")
        model = model_type.model_validate_json(json_text)
        result = deserializer(model, self)
        self._deserialized_pointer_cache[pointer.path] = result
        if isinstance(result, FrameSet):
            self._frame_set_cache[pointer.path] = result
        return result

    def get_frame_set(self, pointer: ZarrPointerModel) -> FrameSet:
        try:
            return self._frame_set_cache[pointer.path]
        except KeyError:
            raise AssertionError(
                f"Frame set at {pointer.path!r} must be deserialised via "
                f"deserialize_pointer before any dependent transform can be."
            ) from None

    def get_array(
        self,
        model: ArrayReferenceModel | InlineArrayModel,
        *,
        slices: tuple[slice, ...] | EllipsisType = ...,
        strip_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> np.ndarray:
        if isinstance(model, InlineArrayModel):
            data: np.ndarray = np.array(model.data, dtype=model.datatype.to_numpy())
            return data if slices is ... else data[slices]
        if not isinstance(model.source, str) or not model.source.startswith("zarr:"):
            raise ArchiveReadError(
                f"ZarrInputArchive cannot resolve array source {model.source!r}; "
                f"expected a 'zarr:<path>' reference."
            )
        zarr_path = model.source[len("zarr:") :]
        try:
            node = self._document.root.get(zarr_path)
        except KeyError:
            raise ArchiveReadError(f"Array reference {zarr_path!r} not in store.") from None
        if not isinstance(node, ZarrArray):
            raise ArchiveReadError(f"{zarr_path!r} is not an array.")

        # Mask unpack: model claims 3-D (mask_size, y, x); on-disk is 2-D
        # (y, x) packed wide-int with flag_masks attribute.
        claimed_shape = tuple(model.shape) if model.shape is not None else None
        if (
            claimed_shape is not None
            and len(claimed_shape) == 3
            and len(node.shape) == 2
            and "flag_masks" in node.attributes.extra
        ):
            return self._read_packed_mask(node, claimed_shape, np.dtype(model.datatype.to_numpy()), slices)

        # Standard path: forward slices straight to the lazy handle.
        return node.read(slices=slices)

    def _read_packed_mask(
        self,
        node: ZarrArray,
        claimed_shape: tuple[int, ...],
        element_dtype: np.dtype,
        slices: tuple[slice, ...] | EllipsisType,
    ) -> np.ndarray:
        """Unpack a 2-D wide-int mask back to 3-D ``(mask_size, y, x)``.

        Mask deserialization expects the storage layout that
        ``Mask.serialize`` streamed — ``(mask_size, y, x)`` — with one
        ``element_dtype`` element per slice along the leading axis,
        matching the schema's element packing. Each element's bits
        live at packed positions ``[stride*i, stride*(i+1))`` where
        ``stride = 8 * element_dtype.itemsize``. Rank-3 ``slices``
        from the deserializer are ``(element_axis, y_slice,
        x_slice)``; the leading slice is stripped before forwarding
        the spatial slice to the lazy handle and re-applied to the
        unpacked output.
        """
        mask_size = claimed_shape[-1]
        # Forward slice to the lazy handle so only intersecting chunks
        # are fetched even on remote stores.
        if slices is ...:
            spatial_slices: tuple[slice, ...] | EllipsisType = ...
            element_slice: slice | EllipsisType = ...
        elif len(slices) == 3:
            element_slice = slices[0]
            spatial_slices = slices[1:]
        else:
            spatial_slices = slices
            element_slice = ...
        packed = node.read(slices=spatial_slices)
        stride = 8 * element_dtype.itemsize
        element_mask = (np.uint64(1) << np.uint64(stride)) - np.uint64(1)
        out = np.empty((mask_size,) + packed.shape, dtype=element_dtype)
        for i in range(mask_size):
            out[i] = ((packed >> np.uint64(stride * i)) & element_mask).astype(element_dtype)
        if element_slice is ...:
            return out
        return out[element_slice]

    def get_table(
        self,
        model: TableModel,
        strip_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> astropy.table.Table:
        result = astropy.table.Table(meta=model.meta)
        for column_model in model.columns:
            if isinstance(column_model.data, InlineArrayModel):
                data: Any = column_model.data.data
            else:
                data = self.get_array(column_model.data, strip_header=strip_header)
            result[column_model.name] = astropy.table.Column(
                data,
                name=column_model.name,
                dtype=column_model.data.datatype.to_numpy(),
                unit=column_model.unit,
                description=column_model.description,
                meta=column_model.meta,
            )
        return result

    def get_structured_array(
        self,
        model: TableModel,
        strip_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> np.ndarray:
        return self.get_table(model, strip_header).as_array()

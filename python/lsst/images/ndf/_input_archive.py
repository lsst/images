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

__all__ = ("NdfInputArchive",)

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from types import EllipsisType
from typing import TYPE_CHECKING, Any, Self

import astropy.table
import h5py
import numpy as np

from lsst.resources import ResourcePath, ResourcePathExpression

from .._transforms import FrameSet
from ..fits._common import FitsOpaqueMetadata
from ..serialization import (
    ArchiveReadError,
    ArchiveTree,
    ArrayReferenceModel,
    InlineArrayModel,
    InputArchive,
    TableModel,
    no_header_updates,
)
from . import _hds
from ._common import NdfPointerModel

if TYPE_CHECKING:
    import astropy.io.fits


_LOG = logging.getLogger(__name__)


class NdfInputArchive(InputArchive[NdfPointerModel]):
    """Reads HDS-on-HDF5 NDF files written by `NdfOutputArchive`.

    Instances should only be constructed via the :meth:`open` context
    manager.

    Parameters
    ----------
    file
        Open ``h5py.File`` handle. Owned by the caller of :meth:`open`;
        the archive does not close it.
    """

    def __init__(self, file: h5py.File) -> None:
        self._file = file
        self._opaque_metadata = FitsOpaqueMetadata()
        # Hooks for Tasks 13–14. The opaque-FITS reader and the array
        # / pointer / frame-set caches will be populated then.
        self._deserialized_pointer_cache: dict[str, Any] = {}
        self._frame_set_cache: dict[str, FrameSet] = {}

    @classmethod
    @contextmanager
    def open(cls, path: ResourcePathExpression) -> Iterator[Self]:
        """Open an NDF file for reading and yield an `NdfInputArchive`.

        Remote ResourcePaths are materialised locally first; fsspec-direct
        h5py reads are a deferred follow-up.
        """
        rp = ResourcePath(path)
        with rp.as_local() as local:
            with h5py.File(local.ospath, "r") as f:
                yield cls(f)

    def get_tree[T: ArchiveTree](self, model_type: type[T]) -> T:
        """Read and validate the main Pydantic tree at ``/MORE/LSST/JSON``."""
        if "/MORE/LSST/JSON" not in self._file:
            raise ArchiveReadError(
                "File has no /MORE/LSST/JSON tree; this is either a "
                "Starlink-only NDF (use ndf.read() with auto-detect) or "
                "the file was written by an unrelated tool."
            )
        lines = _hds.read_char_array(self._file["/MORE/LSST/JSON"])
        return model_type.model_validate_json("".join(lines))

    def deserialize_pointer[U: ArchiveTree, V](
        self,
        pointer: NdfPointerModel,
        model_type: type[U],
        deserializer: Callable[[U, InputArchive[NdfPointerModel]], V],
    ) -> V:
        # Cache by pointer.ref so repeated dereferences reuse the same
        # deserialised result and don't re-run the deserializer.
        if (cached := self._deserialized_pointer_cache.get(pointer.ref)) is not None:
            return cached
        if pointer.ref not in self._file:
            raise ArchiveReadError(f"Pointer reference {pointer.ref!r} not found in NDF file.")
        dataset = self._file[pointer.ref]
        if not isinstance(dataset, h5py.Dataset):
            raise ArchiveReadError(f"Pointer reference {pointer.ref!r} is not a primitive dataset.")
        lines = _hds.read_char_array(dataset)
        json_text = "".join(lines)
        model = model_type.model_validate_json(json_text)
        result = deserializer(model, self)
        self._deserialized_pointer_cache[pointer.ref] = result
        return result

    def get_frame_set(self, ref: NdfPointerModel) -> FrameSet:
        try:
            return self._frame_set_cache[ref.ref]
        except KeyError:
            raise AssertionError(
                f"Frame set at {ref.ref!r} must be deserialised via "
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
        if not isinstance(model.source, str) or not model.source.startswith("ndf:"):
            raise ArchiveReadError(
                f"NdfInputArchive cannot resolve array source {model.source!r}; "
                f"expected an 'ndf:<HDF5-path>' reference."
            )
        path = model.source[len("ndf:") :]
        if path not in self._file:
            raise ArchiveReadError(f"Array reference {path!r} not in file.")
        dataset = self._file[path]
        if not isinstance(dataset, h5py.Dataset):
            raise ArchiveReadError(f"Array reference {path!r} is not a primitive dataset.")
        # h5py supports lazy slicing via dataset[slices].
        return dataset[()] if slices is ... else dataset[slices]

    def get_table(
        self,
        model: TableModel,
        strip_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> astropy.table.Table:
        # Inline-only for v1, paralleling JsonInputArchive (Task 13 may
        # promote this if any inline-vs-reference distinction matters for
        # NDF). For now: same logic as JsonInputArchive.get_table.
        result = astropy.table.Table(meta=model.meta)
        for column_model in model.columns:
            if not isinstance(column_model.data, InlineArrayModel):
                raise ArchiveReadError("Only inline tables are supported in NDF archives in v1.")
            result[column_model.name] = astropy.table.Column(
                column_model.data.data,
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

    def get_opaque_metadata(self) -> FitsOpaqueMetadata:
        # The opaque-FITS reader is wired up in Task 14; for v1 this just
        # returns whatever has been accumulated in __init__ (currently empty).
        return self._opaque_metadata

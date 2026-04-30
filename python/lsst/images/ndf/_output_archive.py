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

__all__ = ("NdfOutputArchive",)

from collections.abc import Callable, Hashable, Iterator, Mapping
from typing import TYPE_CHECKING, Any

import astropy.table
import astropy.units
import h5py
import numpy as np
import pydantic

from .._transforms import FrameSet
from ..fits._common import FitsOpaqueMetadata
from ..serialization import (
    ArchiveTree,
    ArrayReferenceModel,
    InlineArrayModel,
    NestedOutputArchive,
    OutputArchive,
    TableModel,
    no_header_updates,
)
from . import _hds
from ._common import NdfPointerModel

if TYPE_CHECKING:
    import astropy.io.fits


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
            self._file["/"].attrs[_hds.ATTR_CLASS] = "NDF"

    def serialize_direct[T: pydantic.BaseModel](
        self, name: str, serializer: Callable[[OutputArchive[NdfPointerModel]], T]
    ) -> T:
        nested = NestedOutputArchive[NdfPointerModel](name, self)
        return serializer(nested)

    def serialize_pointer[T: ArchiveTree](
        self, name: str, serializer: Callable[[OutputArchive[NdfPointerModel]], T], key: Hashable
    ) -> NdfPointerModel:
        # Implemented in Task 9.
        raise NotImplementedError

    def serialize_frame_set[T: ArchiveTree](
        self, name: str, frame_set: FrameSet, serializer: Callable[[OutputArchive], T], key: Hashable
    ) -> NdfPointerModel:
        # Implemented in Task 9.
        raise NotImplementedError

    def iter_frame_sets(self) -> Iterator[tuple[FrameSet, NdfPointerModel]]:
        return iter(self._frame_sets)

    def add_array(
        self,
        array: np.ndarray,
        *,
        name: str | None = None,
        update_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> ArrayReferenceModel | InlineArrayModel:
        # Implemented in Task 8.
        raise NotImplementedError

    def add_table(
        self,
        table: astropy.table.Table,
        *,
        name: str | None = None,
        update_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> TableModel:
        # Implemented in Task 10.
        raise NotImplementedError

    def add_structured_array(
        self,
        array: np.ndarray,
        *,
        name: str | None = None,
        units: Mapping[str, astropy.units.Unit] | None = None,
        descriptions: Mapping[str, str] | None = None,
        update_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> TableModel:
        # Implemented in Task 10.
        raise NotImplementedError

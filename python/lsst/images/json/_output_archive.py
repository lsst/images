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

__all__ = ("JsonOutputArchive", "write")

from collections.abc import Callable, Hashable, Iterator, Mapping
from typing import TYPE_CHECKING, Any

import astropy.table
import numpy as np
import pydantic

from lsst.resources import ResourcePath, ResourcePathExpression

from .._transforms import FrameSet
from ..serialization import (
    ArchiveTree,
    ButlerInfo,
    InlineArrayModel,
    JsonRef,
    MetadataValue,
    NestedOutputArchive,
    NumberType,
    OutputArchive,
    TableColumnModel,
    TableModel,
    no_header_updates,
)

if TYPE_CHECKING:
    import astropy.io.fits


def write(
    obj: Any,
    path: ResourcePathExpression | None = None,
    metadata: dict[str, MetadataValue] | None = None,
    butler_info: ButlerInfo | None = None,
) -> ArchiveTree:
    """Write an object with a ``serialize`` method to a JSON file.

    Parameters
    ----------
    path
        Name of the file to write to (may be a URI).  If not provided, a
        serializable model is returned but not written to disk.
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
    archive = JsonOutputArchive()
    name = getattr(obj, "_archive_default_name", None)
    tree = archive.serialize_direct(name, obj.serialize) if name is not None else obj.serialize(archive)
    if metadata is not None:
        tree.metadata.update(metadata)
    if butler_info is not None:
        tree.butler_info = butler_info
    archive.finish(tree)
    if path is not None:
        ResourcePath(path).write(tree.model_dump_json().encode())
    return tree


class JsonOutputArchive(OutputArchive[JsonRef]):
    """An implementation of the `.serialization.OutputArchive` interface that
    writes to JSON files.

    This archive type is designed for pure-JSON objects and cases where any
    images or tables are tiny.  It will be *extremely* inefficient for large
    images or tables, if it works at all.
    """

    def __init__(self) -> None:
        self._pointers: dict[Hashable, JsonRef] = {}
        self._indirect: list[Any] = []
        self._frame_sets: list[tuple[FrameSet, JsonRef]] = []

    def serialize_direct[T: pydantic.BaseModel](
        self, name: str, serializer: Callable[[OutputArchive[JsonRef]], T]
    ) -> T:
        nested = NestedOutputArchive[JsonRef](name, self)
        return serializer(nested)

    def serialize_pointer[T: ArchiveTree](
        self, name: str, serializer: Callable[[OutputArchive[JsonRef]], T], key: Hashable
    ) -> JsonRef:
        if (pointer := self._pointers.get(key)) is not None:
            return pointer
        pointer = JsonRef.model_construct(ref=f"#/indirect/{len(self._indirect)}")
        self._indirect.append(self.serialize_direct(name, serializer).model_dump())
        self._pointers[key] = pointer
        return pointer

    def serialize_frame_set[T: ArchiveTree](
        self, name: str, frame_set: FrameSet, serializer: Callable[[OutputArchive], T], key: Hashable
    ) -> JsonRef:
        pointer = self.serialize_pointer(name, serializer, key)
        self._frame_sets.append((frame_set, pointer))
        return pointer

    def iter_frame_sets(self) -> Iterator[tuple[FrameSet, JsonRef]]:
        return iter(self._frame_sets)

    def add_array(
        self,
        array: np.ndarray,
        *,
        name: str | None = None,
        update_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> InlineArrayModel:
        return InlineArrayModel(
            data=array.tolist(),
            datatype=NumberType.from_numpy(array.dtype),
        )

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

    def finish[T: ArchiveTree](self, tree: T) -> T:
        """Finish serialization.

        Parameters
        ----------
        tree
            Serialized archive tree to write, which is modified in place
            (the ``indirect`` attribute is overwritten) and then returned.
        """
        tree.indirect = self._indirect
        return tree

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

__all__ = ("JsonInputArchive", "read")

from collections.abc import Callable
from types import EllipsisType
from typing import TYPE_CHECKING, Any

import astropy.table
import numpy as np

from lsst.resources import ResourcePath, ResourcePathExpression

from .._transforms import FrameSet
from ..serialization import (
    ArchiveReadError,
    ArchiveTree,
    ArrayReferenceModel,
    InlineArrayModel,
    InputArchive,
    JsonRef,
    ReadResult,
    TableModel,
    no_header_updates,
)

if TYPE_CHECKING:
    import astropy.io.fits


def read[T: Any](cls: type[T], target: ResourcePathExpression | ArchiveTree) -> ReadResult[T]:
    """Read an object from a FITS file.

    Parameters
    ----------
    target
        File to read (convertible to `lsst.resources.ResourcePath`) or an
        `.serialization.ArchiveTree` to finish deserializing.  If the latter,
        its ``indirect`` `list` will be interpreted and then cleared.

    Returns
    -------
    ReadResult
        A named tuple containing the deserialized object and any additional
        metadata or butler information saved alongside it.

    Notes
    -----
    Supported types must implement ``deserialize`` and
    ``_get_archive_tree_type`` (see `.Image` for an example).
    """
    tree_type: type[ArchiveTree] = cls._get_archive_tree_type(JsonRef)
    if not isinstance(target, ArchiveTree):
        target = tree_type.model_validate_json(ResourcePath(target).read())
    archive = JsonInputArchive(target.indirect)
    obj = cls.deserialize(target, archive)
    target.indirect = []
    return ReadResult(obj, target.metadata, target.butler_info)


class JsonInputArchive(InputArchive[JsonRef]):
    """An implementation of the `.serialization.InputArchive` interface that
    reads from JSON files.

    Parameters
    ----------
    indirect
        The `.serialization.ArchiveTree.indirect` attribute of the root
        serialization model.
    """

    def __init__(self, indirect: list[Any] | None = None):
        self._indirect = indirect if indirect is not None else []
        self._deserialized_pointer_cache: dict[int, Any] = {}

    def deserialize_pointer[U: ArchiveTree, V](
        self,
        pointer: JsonRef,
        model_type: type[U],
        deserializer: Callable[[U, InputArchive[JsonRef]], V],
    ) -> V:
        index = int(pointer.ref.removeprefix("#/indirect/"))
        if (existing := self._deserialized_pointer_cache.get(index)) is not None:
            return existing
        model = model_type.model_validate(self._indirect[index])
        result = deserializer(model, self)
        self._deserialized_pointer_cache[index] = result
        return result

    def get_frame_set(self, ref: JsonRef) -> FrameSet:
        index = int(ref.ref.removeprefix("#/indirect/"))
        try:
            result = self._deserialized_pointer_cache[index]
        except KeyError:
            raise AssertionError(
                f"Frame set at {ref.model_dump_json(indent=2)} must be deserialized "
                "before any dependent transform can be."
            ) from None
        if not isinstance(result, FrameSet):
            raise ArchiveReadError(f"Expected a FrameSet instance at {ref.model_dump_json(indent=2)}.")
        return result

    def get_array(
        self,
        model: ArrayReferenceModel | InlineArrayModel,
        *,
        slices: tuple[slice, ...] | EllipsisType = ...,
        strip_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> np.ndarray:
        if not isinstance(model, InlineArrayModel):
            raise ArchiveReadError("Only inline arrays are supported in JSON archives.")
        return np.array(model.data, dtype=model.datatype.to_numpy())[slices]

    def get_table(
        self,
        model: TableModel,
        strip_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> astropy.table.Table:
        result = astropy.table.Table(meta=model.meta)
        for column_model in model.columns:
            if not isinstance(column_model.data, InlineArrayModel):
                raise ArchiveReadError("Only inline arrays are supported in JSON archives.")
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
        table = self.get_table(model)
        return table.as_array()

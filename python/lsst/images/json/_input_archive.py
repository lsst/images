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

__all__ = ("JsonInputArchive", "read", "read_tree")

from collections.abc import Callable
from types import EllipsisType
from typing import TYPE_CHECKING, Any

import astropy.table
import numpy as np
from pydantic_core import from_json

from lsst.resources import ResourcePath, ResourcePathExpression

from .._transforms import FrameSet
from ..serialization import (
    ArchiveInfo,
    ArchiveReadError,
    ArchiveTree,
    ArrayReferenceModel,
    InlineArrayModel,
    InputArchive,
    JsonRef,
    ReadResult,
    TableModel,
    no_header_updates,
    parameterize_tree,
)

if TYPE_CHECKING:
    import astropy.io.fits


def read[T: Any](
    cls: type[T],
    target: ResourcePathExpression | ArchiveTree,
    **kwargs: Any,
) -> ReadResult[T]:
    """Read an object from a JSON file.

    Parameters
    ----------
    target
        File to read (convertible to `lsst.resources.ResourcePath`) or an
        `.serialization.ArchiveTree` to finish deserializing.  If the latter,
        its ``indirect`` `list` will be interpreted and then cleared.
    **kwargs
        Extra keyword arguments passed to ``cls.deserialize`` (e.g. ``bbox``
        for image subset reads), matching the FITS and NDF backends.

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
    return read_tree(cls._get_archive_tree_type(JsonRef), target, **kwargs)


def read_tree(
    tree_cls: type[ArchiveTree],
    target: ResourcePathExpression | ArchiveTree,
    **kwargs: Any,
) -> ReadResult[Any]:
    """Read an object using a known `.serialization.ArchiveTree` subclass
    instead of an in-memory type.

    Parameters
    ----------
    tree_cls
        The `.serialization.ArchiveTree` subclass that describes the
        file's top-level tree.  This function parameterises it with the
        JSON pointer model.
    target
        File to read (convertible to `lsst.resources.ResourcePath`) or
        an already-validated `.serialization.ArchiveTree` whose
        ``indirect`` list will be interpreted and then cleared.
    **kwargs
        Extra keyword arguments passed to ``tree.deserialize``.

    Returns
    -------
    ReadResult
        Named tuple of the deserialised object, its metadata, and any
        butler info.
    """
    parameterized = parameterize_tree(tree_cls, JsonRef)
    if not isinstance(target, ArchiveTree):
        target = parameterized.model_validate_json(ResourcePath(target).read())
    archive = JsonInputArchive(target.indirect)
    obj = target.deserialize(archive, **kwargs)
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

    @classmethod
    def get_basic_info(cls, path: ResourcePathExpression) -> ArchiveInfo:
        """Read the top-level tree's ``schema_url``; JSON has no container
        format version.

        This parses the whole document.  Unlike the FITS and NDF backends
        there is no cheap header to read: ``schema_url`` is a computed field
        serialized after the (potentially large) ``indirect`` payload, and
        nested trees carry their own ``schema_url``, so a bounded prefix
        cannot identify the top-level tree reliably.  JSON is not intended
        for large pixel archives, where FITS or NDF should be used instead.
        """
        raw = from_json(ResourcePath(path).read())
        if not isinstance(raw, dict) or not raw.get("schema_url"):
            raise ArchiveReadError(f"{path!r} has no schema_url in its top-level JSON tree.")
        return ArchiveInfo.from_schema_url(raw["schema_url"], format_version=None)

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

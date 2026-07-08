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
"""User-facing ``open_archive`` for incremental, component-wise reads."""

from __future__ import annotations

__all__ = ("Reader", "open_archive")

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import IO, Any, TypeVar, overload

from lsst.resources import ResourcePathExpression

from ._backends import (
    _decompress_path_to_temp_file,
    _is_binary_stream,
    _path_is_compressed,
    backend_for_name,
    backend_for_path,
    backend_for_stream,
)
from ._common import ArchiveTree, ButlerInfo, MetadataValue
from ._input_archive import ArchiveInfo, InputArchive
from ._io import public_type_for_schema

# This pre-python-3.12 declaration is needed so Sphinx (the
# autodoc-typehints plugin) can resolve the ``T`` forward reference in the
# stringized annotations; the PEP 695 ``[T]`` parameters below are scoped to
# their class/function and are not visible in the module globals.
T = TypeVar("T")


class Reader[T]:
    """A handle to an open ``lsst.images`` file.

    Returned by `open_archive`.
    Lets the caller pull individual components, or the whole object, out of a
    file that is opened once; the underlying archive caches dereferenced
    pointers so repeated reads share work.
    Valid only inside the ``with`` block that produced it.

    Parameters
    ----------
    archive
        Input archive backing the open file.
    tree
        Validated on-disk deserialization tree for the file.
    info
        Schema name, version, URL, and format version for the file.
    expected_cls
        Expected type of the deserialized object, or `None` to accept any
        type.
    """

    def __init__(
        self,
        archive: InputArchive[Any],
        tree: ArchiveTree,
        info: ArchiveInfo,
        expected_cls: type[T] | None,
    ) -> None:
        self._archive = archive
        self._tree = tree
        self._info = info
        self._expected_cls = expected_cls
        self._closed = False

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("Reader is closed; use it only inside its 'with' block.")

    @property
    def info(self) -> ArchiveInfo:
        """Schema name/version/url and format version for this file."""
        return self._info

    @property
    def metadata(self) -> dict[str, MetadataValue]:
        """Flexible metadata stored with the object."""
        return self._tree.metadata

    @property
    def butler_info(self) -> ButlerInfo | None:
        """Butler dataset info stored with the object, or `None`."""
        return self._tree.butler_info

    def get_tree(self) -> ArchiveTree:
        """Return the validated on-disk tree for advanced, low-level access.

        Most callers want `read` or `get_component` instead; the tree is the
        raw deserialization model that those methods build on.
        """
        self._check_open()
        return self._tree

    def get_component(self, name: str, **kwargs: Any) -> Any:
        """Deserialize and return a single named component.

        Raises `~lsst.images.serialization.InvalidComponentError` for an
        unknown component name.

        Parameters
        ----------
        name
            Name of the component to deserialize.
        **kwargs
            Additional keyword arguments forwarded to the component
            deserializer.
        """
        self._check_open()
        return self._tree.deserialize_component(name, self._archive, **kwargs)

    def read(self, **kwargs: Any) -> T:
        """Deserialize and return the whole object.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments forwarded to the deserializer.
        """
        self._check_open()
        obj = self._tree.deserialize(self._archive, **kwargs)
        if hasattr(obj, "_opaque_metadata"):
            obj._opaque_metadata = self._archive.get_opaque_metadata()
        if self._expected_cls is not None and not isinstance(obj, self._expected_cls):
            raise TypeError(
                f"{self._info.schema_name!r} deserialized to {type(obj).__name__}, "
                f"not the requested {self._expected_cls.__name__}."
            )
        return obj


@overload
def open_archive[T](
    path: ResourcePathExpression | IO[bytes],
    cls: type[T],
    *,
    format: str | None = ...,
    partial: bool = ...,
    **backend_kwargs: Any,
) -> AbstractContextManager[Reader[T]]: ...
@overload
def open_archive(
    path: ResourcePathExpression | IO[bytes],
    cls: None = ...,
    *,
    format: str | None = ...,
    partial: bool = ...,
    **backend_kwargs: Any,
) -> AbstractContextManager[Reader[Any]]: ...
@contextmanager
def open_archive(
    path: ResourcePathExpression | IO[bytes],
    cls: type[Any] | None = None,
    *,
    format: str | None = None,
    partial: bool = True,
    **backend_kwargs: Any,
) -> Iterator[Reader]:
    """Open an ``lsst.images`` file for incremental, component-wise reads.

    Dispatches to the appropriate backend by file extension (or, for stream
    input, by the leading bytes), resolves the registered in-memory type
    from the file's schema, and returns a `Reader` context manager.

    A path with a ``.gz``/``.zst`` compression suffix is stream-decompressed
    into an anonymous temporary file first (in bounded-memory chunks, since
    compressed data has no random access; ``partial`` is ignored for it).
    zstd requires Python >= 3.14 or the ``zstandard`` package.
    Stream input must already be decompressed: whoever produced the stream
    knows how it was compressed.

    Parameters
    ----------
    path
        File to read; convertible to `lsst.resources.ResourcePath`, or a
        seekable binary stream containing the file's content.
    cls
        Optional expected in-memory type.
        When given, `open_archive` validates that the file's schema
        resolves to a subclass of ``cls`` (raising `TypeError` otherwise)
        and the returned `Reader` is typed accordingly, so `Reader.read`
        needs no cast.
    format
        Optional backend name (``"fits"``, ``"ndf"``, or ``"json"``)
        forcing the backend, instead of dispatching on the path's extension
        or the stream's leading bytes.
    partial
        Forwarded to the backend ``open_tree``; defaults to `True` (a reader
        is for incremental access).
        A no-op for the JSON and NDF backends and for stream input.
    **backend_kwargs
        Backend-specific open options (e.g. ``page_size`` for FITS).

    Raises
    ------
    ValueError
        If the file extension or the stream's leading bytes are not
        recognized, or ``format`` is not a known backend name.
    ArchiveReadError
        If the file's schema is not registered.
    TypeError
        If ``cls`` is given and the file's schema resolves to an
        incompatible type.
    """
    source: ResourcePathExpression | IO[bytes]
    temp_file: IO[bytes] | None = None
    if _is_binary_stream(path):
        source = path
        backend = backend_for_name(format) if format is not None else backend_for_stream(path)
    else:
        backend = backend_for_name(format) if format is not None else backend_for_path(path)
        if _path_is_compressed(path):
            temp_file = _decompress_path_to_temp_file(path)
            source = temp_file
        else:
            source = path
    try:
        with backend.input_archive.open_tree(source, partial=partial, **backend_kwargs) as (
            archive,
            tree,
            info,
        ):
            if cls is not None:
                resolved = public_type_for_schema(info.schema_name)
                if resolved is not None and not issubclass(resolved, cls):
                    raise TypeError(
                        f"{path!r} has schema {info.schema_name!r} (type {resolved.__name__}), "
                        f"which is not a {cls.__name__}."
                    )
            reader: Reader[Any] = Reader(archive, tree, info, cls)
            try:
                yield reader
            finally:
                reader._closed = True
    finally:
        if temp_file is not None:
            temp_file.close()

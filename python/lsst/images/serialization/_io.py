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
"""Generic ``read`` / ``write`` dispatchers and the schema-name registry."""

from __future__ import annotations

__all__ = ("class_for_schema", "parameterize_tree", "read", "register_schema_class", "write")

import typing
from typing import Any, cast

from lsst.resources import ResourcePathExpression

from ._backends import backend_for_path
from ._common import ArchiveReadError, ArchiveTree, ReadResult

_REGISTRY: dict[str, type[ArchiveTree]] = {}
"""Map of ``SCHEMA_NAME`` to the registered ``ArchiveTree`` subclass.

The registry is keyed by name only.  Schema-version compatibility is
enforced when the selected tree's ``model_validate*`` runs, via
``min_read_version``.
"""


def class_for_schema(schema_name: str) -> type[ArchiveTree] | None:
    """Return the registered ``ArchiveTree`` subclass for ``schema_name``,
    or ``None`` if nothing is registered for that name.

    Parameters
    ----------
    schema_name
        Schema name (e.g. ``"visit_image"``).
    """
    return _REGISTRY.get(schema_name)


def register_schema_class(cls: type[ArchiveTree]) -> None:
    """Register ``cls`` under ``cls.SCHEMA_NAME``.

    No-op when the same class is registered again (re-import during
    tests).  Raises `RuntimeError` when a *different* class is
    registered under an existing name.

    Intended to be called from ``ArchiveTree.__pydantic_init_subclass__``;
    not part of the public API.
    """
    key = cls.SCHEMA_NAME
    existing = _REGISTRY.get(key)
    if existing is cls:
        return
    if existing is not None:
        raise RuntimeError(
            f"Schema {cls.SCHEMA_NAME!r} is already registered to "
            f"{existing.__qualname__}; refusing to replace it with "
            f"{cls.__qualname__}."
        )
    _REGISTRY[key] = cls


def parameterize_tree(
    tree_cls: type[ArchiveTree],
    pointer_type: type[Any],
) -> type[ArchiveTree]:
    """Parameterise ``tree_cls`` over ``pointer_type`` if it is generic.

    Some `ArchiveTree` subclasses (e.g. ``SumFieldSerializationModel``)
    take no type parameters; their ``_get_archive_tree_type`` returns
    the class itself.  Match that behaviour here so per-backend
    ``read_tree`` implementations can call this uniformly.
    """
    if not getattr(tree_cls, "__parameters__", ()):
        return tree_cls
    return tree_cls[pointer_type]  # type: ignore[index]


_PUBLIC_TYPE_ATTR = "_lsst_images_public_type"
"""Attribute name used to cache the resolved public type on each
``ArchiveTree`` subclass."""

_UNRESOLVED = object()
"""Sentinel cached when the return annotation is ``Any`` or could not be
resolved.  Distinguishes "we tried and failed" from "we have not tried"."""


def _public_type(tree_cls: type[ArchiveTree]) -> type | None:
    """Return the in-memory class produced by ``tree_cls.deserialize``.

    Derived from the return annotation of ``deserialize`` and cached on
    the class.  Returns `None` when the annotation is `Any` or cannot be
    resolved (e.g. it references a name that is not importable from the
    class's module globals).
    """
    cached = tree_cls.__dict__.get(_PUBLIC_TYPE_ATTR, None)
    if cached is _UNRESOLVED:
        return None
    if cached is not None:
        return cached  # type: ignore[no-any-return]
    try:
        hints = typing.get_type_hints(tree_cls.deserialize)
    except Exception:
        setattr(tree_cls, _PUBLIC_TYPE_ATTR, _UNRESOLVED)
        return None
    annotation = hints.get("return", Any)
    if annotation is Any:
        setattr(tree_cls, _PUBLIC_TYPE_ATTR, _UNRESOLVED)
        return None
    # PEP 695 ``type X = ...`` aliases reach us as TypeAliasType; unwrap to
    # the underlying type so e.g. ``type ApertureCorrectionMap = dict[str,
    # Field]`` resolves to ``dict``.
    if isinstance(annotation, typing.TypeAliasType):
        annotation = annotation.__value__
    resolved = typing.get_origin(annotation) or annotation
    if not isinstance(resolved, type):
        setattr(tree_cls, _PUBLIC_TYPE_ATTR, _UNRESOLVED)
        return None
    setattr(tree_cls, _PUBLIC_TYPE_ATTR, resolved)
    return resolved


def read(
    path: ResourcePathExpression,
    **kwargs: Any,
) -> ReadResult[Any]:
    """Read an archive whose in-memory type is inferred from its schema.

    Dispatches to the FITS / NDF / JSON backend based on ``path``'s
    extension, looks up the registered ``ArchiveTree`` subclass for the
    file's ``schema_name``, and forwards the call to the per-backend
    ``read_tree`` along with ``**kwargs``.  Schema-version compatibility
    is enforced when the model validates the on-disk tree, via
    ``min_read_version``.

    Parameters
    ----------
    path
        File to read; convertible to `lsst.resources.ResourcePath`.
    **kwargs
        Backend- and type-specific keyword arguments.  Forwarded
        verbatim; mis-targeted arguments surface as ``TypeError`` from
        the underlying ``deserialize``.

    Returns
    -------
    ReadResult
        Named tuple of the deserialized object, its metadata, and any
        butler info, matching the per-backend ``read`` signature.

    Raises
    ------
    ValueError
        Raised by `backend_for_path` if the file extension is not
        recognised.
    ArchiveReadError
        Raised when the file's ``schema_name`` is not registered, or
        propagated from the model's ``min_read_version`` check on
        ``model_validate*``.
    """
    backend = backend_for_path(path)
    info = backend.input_archive.get_basic_info(path)
    tree_cls = class_for_schema(info.schema_name)
    if tree_cls is None:
        raise ArchiveReadError(
            f"No registered schema {info.schema_name!r}; cannot determine in-memory type for {path!r}."
        )
    return cast(ReadResult[Any], backend.read_tree(tree_cls, path, **kwargs))


def write(obj: Any, path: str, **kwargs: Any) -> Any:
    """Write ``obj`` to ``path``, dispatching by file extension.

    Forwards ``**kwargs`` to the per-backend ``write`` (e.g.
    ``compression_options`` for FITS).  No registry lookup is performed:
    the per-backend ``write`` already accepts any object with a
    ``serialize`` method.

    Parameters
    ----------
    obj
        Object to write; must implement ``serialize`` like the per-backend
        write functions expect.
    path
        Destination path.  The extension selects the backend.
    **kwargs
        Forwarded verbatim to the backend's ``write``.

    Returns
    -------
    Any
        Whatever the per-backend ``write`` returns (the serialised
        archive tree).
    """
    return backend_for_path(path).write(obj, path, **kwargs)

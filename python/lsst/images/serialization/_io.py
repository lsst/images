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

__all__ = ("class_for_schema", "register_schema_class")

import typing
from typing import Any

from ._common import ArchiveTree

_REGISTRY: dict[tuple[str, str], type[ArchiveTree]] = {}
"""Map of ``(SCHEMA_NAME, SCHEMA_VERSION)`` to the registered
``ArchiveTree`` subclass."""


def class_for_schema(schema_name: str, schema_version: str) -> type[ArchiveTree] | None:
    """Return the registered ``ArchiveTree`` subclass, or ``None``.

    Parameters
    ----------
    schema_name
        Schema name (e.g. ``"visit_image"``).
    schema_version
        Schema version (e.g. ``"1.0.0"``).
    """
    return _REGISTRY.get((schema_name, schema_version))


def register_schema_class(cls: type[ArchiveTree]) -> None:
    """Register ``cls`` under ``(cls.SCHEMA_NAME, cls.SCHEMA_VERSION)``.

    No-op when the same class is registered for the same key (re-import
    during tests).  Raises `RuntimeError` when a *different* class is
    registered under an existing key.

    Intended to be called from ``ArchiveTree.__pydantic_init_subclass__``;
    not part of the public API.
    """
    key = (cls.SCHEMA_NAME, cls.SCHEMA_VERSION)
    existing = _REGISTRY.get(key)
    if existing is cls:
        return
    if existing is not None:
        raise RuntimeError(
            f"Schema {cls.SCHEMA_NAME!r} version {cls.SCHEMA_VERSION!r} "
            f"is already registered to {existing.__qualname__}; refusing to "
            f"replace it with {cls.__qualname__}."
        )
    _REGISTRY[key] = cls


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
    resolved = typing.get_origin(annotation) or annotation
    if not isinstance(resolved, type):
        setattr(tree_cls, _PUBLIC_TYPE_ATTR, _UNRESOLVED)
        return None
    setattr(tree_cls, _PUBLIC_TYPE_ATTR, resolved)
    return resolved

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

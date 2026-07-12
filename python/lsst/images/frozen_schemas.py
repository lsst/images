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
"""Frozen JSON schema files for the serialization data models.

Every `~lsst.images.serialization.ArchiveTree` subclass has a canonical JSON
Schema derived from its pydantic model.  These are written to git-committed
``schemas/`` files so the published schema at
``https://images.lsst.io/schemas/{name}-{version}`` is a stable artifact
rather than whatever the code currently produces, and so superseded versions
remain available after the models move on.
"""

from __future__ import annotations

__all__ = (
    "available_schema_classes",
    "check_frozen_schemas",
    "dump_schema",
    "frozen_schema_filename",
    "frozen_schema_path",
    "write_frozen_schemas",
)

import json
from pathlib import Path
from typing import Any

from .serialization._asdf_utils import ArrayReferenceModel
from .serialization._common import ArchiveTree
from .serialization._io import (
    _BUILTIN_SCHEMA_PROVIDERS,
    _REGISTRY,
    class_for_schema,
    parameterize_tree,
)


def available_schema_classes() -> list[type[ArchiveTree]]:
    """Return every `~lsst.images.serialization.ArchiveTree` subclass owned
    by this package, sorted by schema name.

    Schemas registered from outside the ``lsst.images`` package — via the
    ``lsst.images.schemas`` entry point group or by direct class creation
    (e.g. test doubles) — are deliberately excluded: this package only
    freezes and publishes its own schemas.
    """
    # Local import to avoid a circular import: this module is part of
    # lsst.images, and importing the package here (to register the
    # unconditionally-imported models) at module scope would recurse.
    import lsst.images  # noqa: F401

    classes: list[type[ArchiveTree]] = []
    for name in sorted(set(_REGISTRY) | set(_BUILTIN_SCHEMA_PROVIDERS)):
        cls = class_for_schema(name)
        if cls is None:
            raise RuntimeError(f"Schema {name!r} is registered but its class could not be loaded.")
        if not cls.__module__.startswith("lsst.images."):
            continue
        classes.append(cls)
    return classes


def dump_schema(tree_cls: type[ArchiveTree]) -> dict[str, Any]:
    """Return the JSON Schema for ``tree_cls``.

    Parameters
    ----------
    tree_cls
        Serialization model class to dump.

    Notes
    -----
    Generic trees are parameterized over
    `~lsst.images.serialization.ArrayReferenceModel`, matching the convention
    used by ``lsst-images-admin diagram``.
    """
    schema = parameterize_tree(tree_cls, ArrayReferenceModel).model_json_schema()
    # A recursive model (e.g. sum_field) produces a root that is just a $ref
    # into $defs, with the class's json_schema_extra landing on the $def.
    # Hoist the canonical identity to the document root so every frozen
    # document self-identifies; $ref siblings are valid in draft 2020-12.
    schema.setdefault("$id", f"{tree_cls.SCHEMA_URL_BASE}/{tree_cls.SCHEMA_NAME}-{tree_cls.SCHEMA_VERSION}")
    schema.setdefault("title", tree_cls.SCHEMA_NAME)
    # Nested ArchiveTree definitions inherit their class's $id, but $id
    # starts a new resolution scope in draft 2020-12, which would break the
    # root-relative "#/$defs/..." references pydantic generates inside them.
    # Record the canonical URL under a non-reserved key instead, which
    # validators ignore and documentation tooling can still use to identify
    # published sub-schemas.
    for definition in schema.get("$defs", {}).values():
        if isinstance(definition, dict) and "$id" in definition:
            definition["x-lsst-schema-url"] = definition.pop("$id")
    return schema


def frozen_schema_filename(tree_cls: type[ArchiveTree]) -> str:
    """Return the frozen-schema filename for ``tree_cls``.

    Parameters
    ----------
    tree_cls
        Serialization model class to name the file for.
    """
    return f"{tree_cls.SCHEMA_NAME}-{tree_cls.SCHEMA_VERSION}.json"


def frozen_schema_path(directory: Path, tree_cls: type[ArchiveTree]) -> Path:
    """Return the frozen-schema file path for ``tree_cls`` under
    ``directory``.

    Parameters
    ----------
    directory
        Directory holding the frozen schema files.
    tree_cls
        Serialization model class to locate the file for.

    Notes
    -----
    Files are laid out as ``{name}/{name}-{version}.json``: one
    subdirectory per schema so the directory stays navigable as versions
    accumulate, with the full name-version filename kept so a file is
    self-identifying when copied elsewhere.
    """
    return directory / tree_cls.SCHEMA_NAME / frozen_schema_filename(tree_cls)


def _canonical_text(schema: dict[str, Any]) -> str:
    """Return the canonical file serialization of ``schema``."""
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


def write_frozen_schemas(directory: Path) -> list[Path]:
    """Write the frozen schema file for every current schema.

    Parameters
    ----------
    directory
        Directory to write the ``{name}-{version}.json`` files into; created
        if necessary.

    Returns
    -------
    changed
        Paths that were created or rewritten.

    Notes
    -----
    Files for the *same* name and version are overwritten when their content
    is stale (schemas evolve in place at 1.0.0 until the first data release).
    Files for superseded versions are never touched, so old schema URLs keep
    resolving.
    """
    changed: list[Path] = []
    for cls in available_schema_classes():
        path = frozen_schema_path(directory, cls)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = _canonical_text(dump_schema(cls))
        if not path.exists() or path.read_text() != text:
            path.write_text(text)
            changed.append(path)
    return changed


def check_frozen_schemas(directory: Path) -> list[str]:
    """Check the frozen schema files against the current models.

    Parameters
    ----------
    directory
        Directory holding the frozen ``{name}-{version}.json`` files.

    Returns
    -------
    problems
        One problem description per current schema whose frozen file is
        missing or does not match the current model; empty when the frozen
        files are up to date.
    """
    problems: list[str] = []
    for cls in available_schema_classes():
        path = frozen_schema_path(directory, cls)
        if not path.exists():
            problems.append(f"{path.relative_to(directory)}: missing")
        elif path.read_text() != _canonical_text(dump_schema(cls)):
            problems.append(f"{path.relative_to(directory)}: differs from the current model")
    return problems

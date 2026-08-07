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
"""Registry of per-schema migrations from one data-model major to the next.

A migration is needed when a backward-incompatible version bump means the
current model cannot validate an older tree directly: a renamed or retyped
field, a split or merged field, a restructured sub-tree.  Additive changes
need no migration, because defaulting the new fields on input is enough.

Registering one step per adjacent major pair means only adjacent transforms
are ever written, and the reader chains them to cross a larger gap.  See
:ref:`lsst.images-schema-versioning` for how this pairs with
``min_read_version``.
"""

from __future__ import annotations

__all__ = ("migration",)

from collections.abc import Callable
from typing import Any, Protocol


class Migration(Protocol):
    """A function that advances an on-disk tree one data-model major."""

    __qualname__: str
    """Name used to identify this step in a duplicate-registration error."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        """Rewrite ``data`` into the next major's shape.

        Parameters
        ----------
        data
            Raw on-disk tree, at the major this migration was registered for.
            May be mutated in place and returned; the caller has already
            isolated it from any other union candidate.

        Returns
        -------
        `dict`
            The tree in the ``from_major + 1`` shape.  Must not set
            ``schema_version``; the caller does that as it chains steps.
        """
        ...


_MIGRATIONS: dict[tuple[str, int], Migration] = {}
"""Registered migrations, keyed by ``(schema name, from_major)``.

Each entry advances a tree exactly one major.
"""

_MIGRATABLE_NAMES: set[str] = set()
"""Schema names with at least one registered migration.

The read path checks this set before doing any other work, so a schema with
no migrations -- every schema in this package today -- pays only an empty-set
truthiness test per validation.
"""


def migration(schema_name: str, from_major: int) -> Callable[[Migration], Migration]:
    """Register a function that advances a tree one data-model major.

    Parameters
    ----------
    schema_name
        ``SCHEMA_NAME`` of the schema this migration applies to.
    from_major
        Major version the incoming tree is at.  The function must return a
        tree in the ``from_major + 1`` shape.

    Returns
    -------
    `~collections.abc.Callable`
        Decorator that registers and returns the function unchanged.

    Raises
    ------
    RuntimeError
        If a migration is already registered for this schema and major.

    Notes
    -----
    The function receives the raw on-disk tree as a `dict` and may mutate and
    return it.  It must not set ``schema_version``; the caller does that as it
    chains steps.
    """

    def register(func: Migration) -> Migration:
        key = (schema_name, from_major)
        if (existing := _MIGRATIONS.get(key)) is not None and existing is not func:
            raise RuntimeError(
                f"A migration for {schema_name!r} major {from_major} is already registered to "
                f"{existing.__qualname__}; refusing to replace it with {func.__qualname__}."
            )
        _MIGRATIONS[key] = func
        _MIGRATABLE_NAMES.add(schema_name)
        return func

    return register

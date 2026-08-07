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
"""Purpose-built schemas that exist only to exercise the fixture machinery.

These are defined in the test tree, so ``available_schema_classes`` — which
filters by defining module — never sees them: they are never frozen, never
published, and never part of shipped data.  Their fixtures live under
``tests/data/fixture_doubles`` in the same layout as the real fixture tree, so
the same ladder runs over both.

This is a plain module rather than a test module so importing it registers the
schemas deterministically, independent of pytest's collection order.
"""

from __future__ import annotations

__all__ = (
    "ChainTestModel",
    "GapTestModel",
    "MigrationIsolationTestModel",
    "MigrationTestModel",
    "ProjectionTestModel",
    "RetireTestModel",
    "UnmigratableTestModel",
)

from typing import Any, ClassVar

import pydantic

from lsst.images.serialization import ArchiveTree, InputArchive, migration


class ProjectionTestModel(ArchiveTree):
    """Schema whose 1.1.0 adds an optional field over 1.0.0.

    The additive common case: the model absorbs a 1.0.0 tree unaided, so the
    projection oracle must pass on this pair with no escape hatch.  This is the
    oracle's positive control.
    """

    SCHEMA_NAME: ClassVar[str] = "projection_test"
    SCHEMA_VERSION: ClassVar[str] = "1.1.0"
    MIN_READ_VERSION: ClassVar[int] = 1
    PUBLIC_TYPE: ClassVar[type] = dict

    kept: str = pydantic.Field(default="", description="Present at both versions.")
    added: int = pydantic.Field(default=0, description="Added at 1.1.0.")

    def deserialize(self, archive: InputArchive[Any], **kwargs: Any) -> Any:
        return {"kept": self.kept, "added": self.added}


class RetireTestModel(ArchiveTree):
    """Schema whose 2.0.0 shape a 1.0.0 tree cannot satisfy.

    ``required_at_v2`` has no default and no migration is registered, so the
    1.0.0 fixture can no longer be read.  That is what retirement means, and
    the fixture under ``retired/`` asserts it.
    """

    SCHEMA_NAME: ClassVar[str] = "retire_test"
    SCHEMA_VERSION: ClassVar[str] = "2.0.0"
    MIN_READ_VERSION: ClassVar[int] = 2
    PUBLIC_TYPE: ClassVar[type] = dict

    required_at_v2: str = pydantic.Field(description="Required from 2.0.0 on.")

    def deserialize(self, archive: InputArchive[Any], **kwargs: Any) -> Any:
        return {"required_at_v2": self.required_at_v2}


class MigrationTestModel(ArchiveTree):
    """Purpose-built schema whose 2.0.0 shape renames a 1.0.0 field.

    Pydantic alone cannot validate a 1.0.0 tree, because ``renamed`` is
    required and ``original`` is not a field, so a successful read proves the
    migration chain did the work.
    """

    SCHEMA_NAME: ClassVar[str] = "migration_test"
    SCHEMA_VERSION: ClassVar[str] = "2.0.0"
    MIN_READ_VERSION: ClassVar[int] = 2
    PUBLIC_TYPE: ClassVar[type] = dict

    renamed: str = pydantic.Field(description="Renamed from 'original' in 2.0.0.")

    def deserialize(self, archive: InputArchive[Any], **kwargs: Any) -> Any:
        return {"renamed": self.renamed}


class ChainTestModel(ArchiveTree):
    """Purpose-built schema at 3.0.0, to prove steps chain across majors."""

    SCHEMA_NAME: ClassVar[str] = "chain_test"
    SCHEMA_VERSION: ClassVar[str] = "3.0.0"
    MIN_READ_VERSION: ClassVar[int] = 1
    PUBLIC_TYPE: ClassVar[type] = dict

    steps: list[str] = pydantic.Field(default_factory=list, description="Migrations applied.")

    def deserialize(self, archive: InputArchive[Any], **kwargs: Any) -> Any:
        return {"steps": self.steps}


class UnmigratableTestModel(ArchiveTree):
    """Purpose-built schema at 2.0.0 with no migration registered."""

    SCHEMA_NAME: ClassVar[str] = "unmigratable_test"
    SCHEMA_VERSION: ClassVar[str] = "2.0.0"
    MIN_READ_VERSION: ClassVar[int] = 1
    PUBLIC_TYPE: ClassVar[type] = dict

    required_at_v2: str = pydantic.Field(description="Required from 2.0.0 on.")

    def deserialize(self, archive: InputArchive[Any], **kwargs: Any) -> Any:
        return {"required_at_v2": self.required_at_v2}


class GapTestModel(ArchiveTree):
    """Purpose-built schema at 3.0.0 that registers only a non-adjacent step.

    Only the 2->3 step is registered below, so a tree stamped at major 1 hits
    a genuine migration gap: nothing is registered for 1->2.  This is
    distinct from a schema with no migrations at all (`UnmigratableTestModel`),
    which falls through to ordinary validation instead of raising a gap
    error.
    """

    SCHEMA_NAME: ClassVar[str] = "gap_test"
    SCHEMA_VERSION: ClassVar[str] = "3.0.0"
    MIN_READ_VERSION: ClassVar[int] = 1
    PUBLIC_TYPE: ClassVar[type] = dict

    def deserialize(self, archive: InputArchive[Any], **kwargs: Any) -> Any:
        return {}


class MigrationIsolationTestModel(ArchiveTree):
    """Schema whose migration mutates the tree and then fails validation.

    ``required`` has no default and the migration below appends to a nested
    list, so a tree that omits ``required`` is mutated and *then* rejected --
    the shape a union needs to isolate, since the next variant must see the
    unmutated input.
    """

    SCHEMA_NAME: ClassVar[str] = "migration_isolation_test"
    SCHEMA_VERSION: ClassVar[str] = "2.0.0"
    MIN_READ_VERSION: ClassVar[int] = 1
    PUBLIC_TYPE: ClassVar[type] = dict

    steps: list[str] = pydantic.Field(description="Migrations applied.")
    required: str = pydantic.Field(description="Required, so validation fails without it.")

    def deserialize(self, archive: InputArchive[Any], **kwargs: Any) -> Any:
        return {"steps": self.steps, "required": self.required}


@migration("migration_test", 1)
def _migration_test_1_to_2(data: dict[str, Any]) -> dict[str, Any]:
    """Rename 'original' to 'renamed' for the 2.0.0 shape.

    Parameters
    ----------
    data
        On-disk tree at major 1.
    """
    data["renamed"] = data.pop("original")
    return data


@migration("chain_test", 1)
def _chain_test_1_to_2(data: dict[str, Any]) -> dict[str, Any]:
    """Record that the 1 -> 2 step ran.

    Parameters
    ----------
    data
        On-disk tree at major 1.
    """
    data.setdefault("steps", []).append("1->2")
    return data


@migration("chain_test", 2)
def _chain_test_2_to_3(data: dict[str, Any]) -> dict[str, Any]:
    """Record that the 2 -> 3 step ran.

    Parameters
    ----------
    data
        On-disk tree at major 2.
    """
    data.setdefault("steps", []).append("2->3")
    return data


@migration("migration_isolation_test", 1)
def _migration_isolation_test_1_to_2(data: dict[str, Any]) -> dict[str, Any]:
    """Mutate a nested list, so an unisolated candidate is visibly dirty.

    Parameters
    ----------
    data
        On-disk tree at major 1.
    """
    data.setdefault("steps", []).append("migration ran")
    return data


@migration("gap_test", 2)
def _gap_test_2_to_3(data: dict[str, Any]) -> dict[str, Any]:
    """Advance a major-2 tree to major 3; unreachable from major 1.

    Deliberately not registered for major 1, so a tree stamped at major 1
    exercises a genuine migration gap rather than the adjacent case.

    Parameters
    ----------
    data
        On-disk tree at major 2.
    """
    return data

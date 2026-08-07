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
"""The schema migration chain.

Migration is proven by purpose-built schemas from ``schema_doubles``, which is
never frozen, published, or part of shipped data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pydantic
import pytest
from schema_doubles import (
    ChainTestModel,
    GapTestModel,
    MigrationIsolationTestModel,
    MigrationTestModel,
    ProjectionTestModel,
    UnmigratableTestModel,
)

from lsst.images.serialization import ArchiveReadError, ArchiveTree, InputArchive, migration
from lsst.images.serialization._migrations import _MIGRATIONS
from lsst.images.tests import iter_schema_fixtures, read_fixture_tree

DOUBLE_DIR = Path(__file__).parent / "data" / "fixture_doubles"


def test_registration() -> None:
    """Verify the decorator registers one entry per (name, from_major)."""
    assert ("migration_test", 1) in _MIGRATIONS
    assert ("migration_isolation_test", 1) in _MIGRATIONS
    assert ("chain_test", 1) in _MIGRATIONS
    assert ("chain_test", 2) in _MIGRATIONS
    assert ("gap_test", 2) in _MIGRATIONS


def test_registering_twice_raises() -> None:
    """Verify a duplicate registration is refused rather than silently won."""
    with pytest.raises(RuntimeError, match="already registered"):

        @migration("migration_test", 1)
        def _duplicate(data: dict[str, Any]) -> dict[str, Any]:
            return data


def test_migration_morphs_an_old_tree() -> None:
    """Verify a 1.0.0 tree reads through the chain into the 2.0.0 shape."""
    tree = MigrationTestModel.model_validate(
        {"schema_version": "1.0.0", "min_read_version": 1, "original": "hello"}
    )
    assert tree.renamed == "hello"
    assert tree.schema_version == "2.0.0"
    assert tree.min_read_version == 2


def test_migration_leaves_a_current_tree_alone() -> None:
    """Verify a current-version tree is not passed through any step."""
    tree = MigrationTestModel.model_validate(
        {"schema_version": "2.0.0", "min_read_version": 2, "renamed": "hello"}
    )
    assert tree.renamed == "hello"


def test_migration_chains_across_two_majors() -> None:
    """Verify a 1.0.0 tree runs both registered steps, in order."""
    tree = ChainTestModel.model_validate({"schema_version": "1.0.0", "min_read_version": 1})
    assert tree.steps == ["1->2", "2->3"]


def test_failed_union_migration_does_not_mutate_the_fallback_input() -> None:
    """Verify a failed migration candidate is isolated from later variants.

    ``MigrationIsolationTestModel`` and its migration live in
    ``schema_doubles`` rather than here: registration is process-global and
    refuses to be replaced, so registering from inside a test body would make
    the test fail on any second run in the same process.
    """

    class _Fallback(pydantic.BaseModel):
        schema_version: str
        min_read_version: int
        steps: list[str]

    class _Wrapper(pydantic.BaseModel):
        member: MigrationIsolationTestModel | _Fallback = pydantic.Field(union_mode="left_to_right")

    payload = {"member": {"schema_version": "1.0.0", "min_read_version": 1, "steps": ["original"]}}
    wrapper = _Wrapper.model_validate(payload)
    assert isinstance(wrapper.member, _Fallback)
    assert wrapper.member.steps == ["original"]
    assert payload["member"]["steps"] == ["original"]


def test_missing_step_raises_a_clean_error() -> None:
    """Verify a genuine migration gap names the missing step, not a stray
    error.

    GapTestModel registers only its 2->3 step, so a tree stamped at major 1
    has nothing registered for 1->2 -- a real gap, unlike a schema that has
    registered no migration at all (see the fallthrough tests below).

    The gap is raised from a ``mode="before"`` validator as an exception that
    is deliberately both `ArchiveReadError` and `ValueError` (see
    ``_MigrationGapError``), so a ``left_to_right`` union can treat it as "try
    the next variant" (`test_migration_gap_falls_through_a_left_to_right_union`
    below).  That same dual nature means Pydantic -- which wraps any
    validator-raised `ValueError` into its own `pydantic.ValidationError`,
    indistinguishably from a nested union trial or a standalone top-level
    call -- reports this one as a `pydantic.ValidationError` here rather than
    a bare `ArchiveReadError`, exactly as it already does for other schema
    incompatibilities (see ``test_retire_double_fixture_is_rejected``); the
    message naming the missing step survives that wrapping unchanged.
    """
    with pytest.raises((ArchiveReadError, pydantic.ValidationError), match="no migration from major 1 to 2"):
        GapTestModel.model_validate({"schema_version": "1.0.0", "min_read_version": 1})


def test_schemas_without_migrations_are_untouched() -> None:
    """Verify the fast path leaves a migration-free schema alone.

    UnmigratableTestModel has no registered migration despite being at major
    2, and reading a tree stamped at an older version must not raise a
    missing-step error: only a schema that has itself registered a migration
    should ever reach the chaining loop, regardless of what some unrelated
    schema (e.g. GapTestModel, registered above) has registered.  Before the
    gate was fixed to check only this schema's own registrations, this exact
    read raised the gap error as soon as any other schema in the process had
    registered a migration.
    """
    assert UnmigratableTestModel.SCHEMA_NAME not in {name for name, _ in _MIGRATIONS}
    tree = UnmigratableTestModel.model_validate(
        {"schema_version": "1.0.0", "min_read_version": 1, "required_at_v2": "present"}
    )
    assert tree.required_at_v2 == "present"  # type: ignore[attr-defined]
    assert tree.schema_version == "2.0.0"


def test_no_migration_schema_falls_through_to_a_validation_error() -> None:
    """Verify a migration-free schema falls through to ordinary Pydantic
    validation rather than a migration-gap error when the older tree
    genuinely cannot satisfy the current shape.

    UnmigratableTestModel's ``required_at_v2`` has no default, so a tree that
    omits it fails Pydantic's own validation -- the other half of what
    :ref:`lsst.images-schema-versioning` promises for a schema with no
    migration: either in-model backfill handles the older tree (the test
    above), or Pydantic reports a validation error, but never the clean
    migration-gap error, which is reserved for a schema that has actually
    opted into migrations.
    """
    with pytest.raises(pydantic.ValidationError):
        UnmigratableTestModel.model_validate({"schema_version": "1.0.0", "min_read_version": 1})


def test_migration_gap_falls_through_a_left_to_right_union() -> None:
    """Verify a migration gap in one union member does not abort the union.

    ``ArchiveReadError`` derives from `RuntimeError`, which Pydantic does not
    treat as a failed union candidate; the gap raises a narrower exception
    that is also a `ValueError`, so Pydantic moves on to the next variant
    instead of the whole union attempt failing.  This is the shape of
    `~lsst.images._visit_image.VisitImageSerializationModel.psf`: a
    ``left_to_right`` union where an earlier variant may no longer accept an
    older tree while a later one still matches it directly.
    """

    class _Wrapper(pydantic.BaseModel):
        member: GapTestModel | ProjectionTestModel = pydantic.Field(union_mode="left_to_right")

    wrapper = _Wrapper.model_validate({"member": {"schema_version": "1.0.0", "min_read_version": 1}})
    assert isinstance(wrapper.member, ProjectionTestModel)


@pytest.mark.parametrize("stamp", ["not.a.version", "", 2, None])
def test_malformed_stamp_falls_through_a_left_to_right_union(stamp: object) -> None:
    """Verify an unparsable stamp fails one variant, not the whole union.

    The on-disk major is parsed in the same ``mode="before"`` validator as the
    migration chain, ahead of any field validation, so it runs for every
    migratable variant a union tries -- including variants the payload does
    not belong to.  A stamp the reader cannot parse must therefore fail that
    one candidate the way a migration gap does, rather than abort the union.
    """

    class _Fallback(pydantic.BaseModel):
        schema_version: Any
        min_read_version: int

    class _Wrapper(pydantic.BaseModel):
        member: MigrationTestModel | _Fallback = pydantic.Field(union_mode="left_to_right")

    wrapper = _Wrapper.model_validate({"member": {"schema_version": stamp, "min_read_version": 1}})
    assert isinstance(wrapper.member, _Fallback)


def test_malformed_stamp_names_the_unparsable_version() -> None:
    """Verify a standalone read reports which stamp could not be parsed.

    Wrapped by Pydantic for the reason
    ``test_missing_step_raises_a_clean_error`` describes, with the message
    preserved.
    """
    with pytest.raises((ArchiveReadError, pydantic.ValidationError), match="has non-integer major"):
        MigrationTestModel.model_validate({"schema_version": "x.0.0", "min_read_version": 1})


def test_nested_subtree_migrates() -> None:
    """Verify a nested old sub-tree is morphed by its own registered step."""

    class _Parent(ArchiveTree):
        SCHEMA_NAME: ClassVar[str] = "migration_parent_test"
        SCHEMA_VERSION: ClassVar[str] = "1.0.0"
        MIN_READ_VERSION: ClassVar[int] = 1
        PUBLIC_TYPE: ClassVar[type] = dict

        child: MigrationTestModel = pydantic.Field(description="An embedded sub-tree.")

        def deserialize(self, archive: InputArchive[Any], **kwargs: Any) -> Any:
            return {"child": self.child.deserialize(archive)}

    parent = _Parent.model_validate(
        {
            "schema_version": "1.0.0",
            "min_read_version": 1,
            "child": {"schema_version": "1.0.0", "min_read_version": 1, "original": "nested"},
        }
    )
    assert parent.child.renamed == "nested"


def test_migration_test_fixture_reads_from_disk() -> None:
    """Verify the declared projection divergence transforms correctly."""
    fixtures = {f.version: f for f in iter_schema_fixtures(DOUBLE_DIR) if f.name == "migration_test"}
    tree = read_fixture_tree(fixtures["1.0.0"])
    assert tree.renamed == "exemplar"  # type: ignore[attr-defined]
    current = read_fixture_tree(fixtures["2.0.0"])
    assert current.renamed == "exemplar"  # type: ignore[attr-defined]

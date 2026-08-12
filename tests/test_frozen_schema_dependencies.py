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
"""Tests for the freeze-time check that a finalized schema depends only on
other finalized schemas.

The doubles live in their own module so the ``package`` filter used by
``write_frozen_schemas`` selects exactly them, leaving the doubles in
``test_frozen_schemas`` unaffected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pydantic
import pytest

from lsst.images.serialization import (
    ArchiveTree,
    FrozenSchemaError,
    InputArchive,
    available_schema_classes,
    dump_schema,
    frozen_schema_path,
    is_development_version,
    schema_dependencies,
    write_frozen_schemas,
)
from lsst.images.serialization._frozen_schemas import _canonical_text

PACKAGE = __name__


class _DevLeaf(ArchiveTree):
    """Development schema that others depend on."""

    SCHEMA_NAME: ClassVar[str] = "frozen_schema_deps_test_dev_leaf"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0.dev0"
    MIN_READ_VERSION: ClassVar[int] = 1
    PUBLIC_TYPE: ClassVar[type] = object

    note: str = pydantic.Field(default="", description="A field.")

    def deserialize(
        self, archive: InputArchive[Any], **kwargs: Any
    ) -> Any:  # pragma: no cover - never invoked
        raise NotImplementedError()


class _FinalLeaf(ArchiveTree):
    """Finalized schema with no dependencies of its own."""

    SCHEMA_NAME: ClassVar[str] = "frozen_schema_deps_test_final_leaf"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"
    MIN_READ_VERSION: ClassVar[int] = 1
    PUBLIC_TYPE: ClassVar[type] = object

    note: str = pydantic.Field(default="", description="A field.")

    def deserialize(
        self, archive: InputArchive[Any], **kwargs: Any
    ) -> Any:  # pragma: no cover - never invoked
        raise NotImplementedError()


class _FinalInheritsDev(_DevLeaf):
    """Finalized schema that inherits from a development schema.

    The ``cell_coadd`` / ``masked_image`` shape.
    """

    SCHEMA_NAME: ClassVar[str] = "frozen_schema_deps_test_inherits_dev"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"


class _FinalEmbedsDev(ArchiveTree):
    """Finalized schema with a field referencing a development schema."""

    SCHEMA_NAME: ClassVar[str] = "frozen_schema_deps_test_embeds_dev"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"
    MIN_READ_VERSION: ClassVar[int] = 1
    PUBLIC_TYPE: ClassVar[type] = object

    child: _DevLeaf = pydantic.Field(description="An embedded development schema.")

    def deserialize(
        self, archive: InputArchive[Any], **kwargs: Any
    ) -> Any:  # pragma: no cover - never invoked
        raise NotImplementedError()


class _FinalEmbedsDevIndirectly(ArchiveTree):
    """Finalized schema whose dependency on a development schema is indirect.

    Reaches ``_DevLeaf`` only through a container and an optional union, and
    only via another finalized schema, so a non-transitive check that looked
    at bare annotations would miss it.
    """

    SCHEMA_NAME: ClassVar[str] = "frozen_schema_deps_test_embeds_dev_indirect"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"
    MIN_READ_VERSION: ClassVar[int] = 1
    PUBLIC_TYPE: ClassVar[type] = object

    children: dict[str, _FinalEmbedsDev | None] = pydantic.Field(
        default_factory=dict, description="Indirectly reaches the development schema."
    )

    def deserialize(
        self, archive: InputArchive[Any], **kwargs: Any
    ) -> Any:  # pragma: no cover - never invoked
        raise NotImplementedError()


class _FinalClean(ArchiveTree):
    """Finalized schema depending only on another finalized schema."""

    SCHEMA_NAME: ClassVar[str] = "frozen_schema_deps_test_clean"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"
    MIN_READ_VERSION: ClassVar[int] = 1
    PUBLIC_TYPE: ClassVar[type] = object

    children: list[_FinalLeaf] = pydantic.Field(
        default_factory=list, description="Embedded finalized schemas."
    )

    def deserialize(
        self, archive: InputArchive[Any], **kwargs: Any
    ) -> Any:  # pragma: no cover - never invoked
        raise NotImplementedError()


class _SelfRecursive(ArchiveTree):
    """Finalized schema that references itself."""

    SCHEMA_NAME: ClassVar[str] = "frozen_schema_deps_test_recursive"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"
    MIN_READ_VERSION: ClassVar[int] = 1
    PUBLIC_TYPE: ClassVar[type] = object

    child: _SelfRecursive | None = pydantic.Field(default=None, description="Itself.")

    def deserialize(
        self, archive: InputArchive[Any], **kwargs: Any
    ) -> Any:  # pragma: no cover - never invoked
        raise NotImplementedError()


def test_dependencies_follow_inheritance() -> None:
    """Verify a schema-declaring base class counts as a dependency."""
    assert set(schema_dependencies(_FinalInheritsDev)) == {_DevLeaf.SCHEMA_NAME}


def test_dependencies_follow_fields() -> None:
    """Verify an embedded model counts as a dependency."""
    assert set(schema_dependencies(_FinalEmbedsDev)) == {_DevLeaf.SCHEMA_NAME}


def test_dependencies_are_transitive() -> None:
    """Verify dependencies are followed through containers and unions."""
    assert set(schema_dependencies(_FinalEmbedsDevIndirectly)) == {
        _FinalEmbedsDev.SCHEMA_NAME,
        _DevLeaf.SCHEMA_NAME,
    }


def test_dependencies_exclude_self() -> None:
    """Verify a self-referential schema is not its own dependency."""
    assert schema_dependencies(_SelfRecursive) == {}


def test_dependencies_of_clean_schema() -> None:
    """Verify a clean finalized schema reports only finalized dependencies."""
    assert set(schema_dependencies(_FinalClean)) == {_FinalLeaf.SCHEMA_NAME}


def test_freeze_refuses_schema_inheriting_development(tmp_path: Path) -> None:
    """Verify freezing fails when a finalized schema depends on a dev one."""
    with pytest.raises(FrozenSchemaError, match=_DevLeaf.SCHEMA_NAME):
        write_frozen_schemas(tmp_path, package=PACKAGE)


def test_freeze_error_names_the_dependent_schema(tmp_path: Path) -> None:
    """Verify the error identifies which schema may not be frozen."""
    with pytest.raises(FrozenSchemaError) as caught:
        write_frozen_schemas(tmp_path, package=PACKAGE)
    message = str(caught.value)
    assert _DevLeaf.SCHEMA_VERSION in message
    assert any(
        name in message
        for name in (
            _FinalInheritsDev.SCHEMA_NAME,
            _FinalEmbedsDev.SCHEMA_NAME,
            _FinalEmbedsDevIndirectly.SCHEMA_NAME,
        )
    )


def test_shipped_schemas_have_no_development_dependencies() -> None:
    """Verify no finalized schema in the package depends on a development one.

    This is the invariant the freeze-time check exists to protect, asserted
    over the real schema set rather than the doubles above.
    """
    offenders = {
        cls.SCHEMA_NAME: sorted(
            f"{name}-{dependency.SCHEMA_VERSION}"
            for name, dependency in schema_dependencies(cls).items()
            if is_development_version(dependency.SCHEMA_VERSION)
        )
        for cls in available_schema_classes()
        if not is_development_version(cls.SCHEMA_VERSION)
    }
    assert not {name: dev for name, dev in offenders.items() if dev}


def test_already_frozen_file_is_not_rechecked(tmp_path: Path) -> None:
    """Verify the check applies only at freeze time.

    An already-committed frozen file is left alone even when its dependencies
    are still in development, so a pre-existing violation does not become a
    permanent failure.
    """
    for cls in (_FinalInheritsDev, _FinalEmbedsDev, _FinalEmbedsDevIndirectly):
        path = frozen_schema_path(tmp_path, cls)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_canonical_text(dump_schema(cls)))
    write_frozen_schemas(tmp_path, package=PACKAGE)
    assert frozen_schema_path(tmp_path, _FinalClean).exists()
    assert not frozen_schema_path(tmp_path, _DevLeaf).exists()

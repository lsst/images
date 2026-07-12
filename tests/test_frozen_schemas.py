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

import importlib.metadata
import json
from pathlib import Path
from typing import Any, ClassVar

import pytest

from lsst.images.frozen_schemas import (
    available_schema_classes,
    check_frozen_schemas,
    dump_schema,
    frozen_schema_filename,
    frozen_schema_path,
    write_frozen_schemas,
)
from lsst.images.serialization import ArchiveTree, InputArchive

REPO_SCHEMA_DIR = Path(__file__).parent.parent / "schemas"


class _ForeignArchiveTree(ArchiveTree):
    """Concrete ArchiveTree defined outside lsst.images; registers itself on
    class creation, like the test doubles in other test modules.
    """

    SCHEMA_NAME: ClassVar[str] = "frozen_schemas_test_foreign"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"
    MIN_READ_VERSION: ClassVar[int] = 1
    PUBLIC_TYPE: ClassVar[type] = object

    def deserialize(
        self, archive: InputArchive[Any], **kwargs: Any
    ) -> Any:  # pragma: no cover - never invoked
        raise NotImplementedError()


class _CustomBaseArchiveTree(ArchiveTree):
    """Concrete ArchiveTree with a third-party schema URL base."""

    SCHEMA_URL_BASE: ClassVar[str] = "https://example.org/schemas"
    SCHEMA_NAME: ClassVar[str] = "frozen_schemas_test_custom_base"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"
    MIN_READ_VERSION: ClassVar[int] = 1
    PUBLIC_TYPE: ClassVar[type] = object

    def deserialize(
        self, archive: InputArchive[Any], **kwargs: Any
    ) -> Any:  # pragma: no cover - never invoked
        raise NotImplementedError()


def test_custom_schema_url_base() -> None:
    """Verify a subclass can override SCHEMA_URL_BASE so its schema URL and
    $id are minted under its own documentation site.
    """
    expected = "https://example.org/schemas/frozen_schemas_test_custom_base-1.0.0"
    assert _CustomBaseArchiveTree().schema_url == expected
    schema = dump_schema(_CustomBaseArchiveTree)
    assert schema["$id"] == expected
    assert schema["title"] == "frozen_schemas_test_custom_base"


def test_available_schema_classes() -> None:
    """Verify enumeration finds the package's schemas, sorted by name."""
    classes = available_schema_classes()
    names = [cls.SCHEMA_NAME for cls in classes]
    assert names == sorted(names)
    assert "image" in names
    assert "cell_coadd" in names  # lazily-loaded built-in provider


def test_available_schema_classes_excludes_foreign_modules() -> None:
    """Verify classes registered from outside lsst.images (e.g. test
    doubles) are not treated as package-owned schemas.
    """
    names = [cls.SCHEMA_NAME for cls in available_schema_classes()]
    assert "frozen_schemas_test_foreign" not in names


def test_available_schema_classes_package_filter() -> None:
    """Verify the package argument selects schemas by defining module, so an
    external package can freeze its own schemas.
    """
    classes = available_schema_classes(package=_ForeignArchiveTree.__module__)
    names = [cls.SCHEMA_NAME for cls in classes]
    assert "frozen_schemas_test_foreign" in names
    assert "image" not in names


def test_available_schema_classes_discovers_entry_points(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify schemas provided only through the ``lsst.images.schemas``
    entry point group are discovered without prior registration.
    """
    state: dict[str, type[ArchiveTree]] = {}

    def _load() -> type[ArchiveTree]:
        if "cls" not in state:

            class _EntryPointArchiveTree(ArchiveTree):
                SCHEMA_NAME: ClassVar[str] = "frozen_schemas_test_entry_point"
                SCHEMA_VERSION: ClassVar[str] = "1.0.0"
                MIN_READ_VERSION: ClassVar[int] = 1
                PUBLIC_TYPE: ClassVar[type] = object

                def deserialize(
                    self, archive: InputArchive[Any], **kwargs: Any
                ) -> Any:  # pragma: no cover - never invoked
                    raise NotImplementedError()

            state["cls"] = _EntryPointArchiveTree
        return state["cls"]

    class _FakeEntryPoint:
        name = "frozen_schemas_test_entry_point"
        value = "fake.module:FakeClass"

        def load(self) -> type[ArchiveTree]:
            return _load()

    real_entry_points = importlib.metadata.entry_points

    def _fake_entry_points(**kwargs: Any) -> Any:
        if kwargs.get("group") == "lsst.images.schemas":
            eps = [_FakeEntryPoint()]
            if (name := kwargs.get("name")) is not None:
                eps = [ep for ep in eps if ep.name == name]
            return eps
        return real_entry_points(**kwargs)

    monkeypatch.setattr(importlib.metadata, "entry_points", _fake_entry_points)
    classes = available_schema_classes(package=_ForeignArchiveTree.__module__)
    assert "frozen_schemas_test_entry_point" in [cls.SCHEMA_NAME for cls in classes]


def test_dump_schema_has_id_and_title() -> None:
    """Verify the dumped schema carries the canonical $id and title."""
    (cls,) = [c for c in available_schema_classes() if c.SCHEMA_NAME == "image"]
    schema = dump_schema(cls)
    assert schema["$id"] == f"https://images.lsst.io/schemas/image-{cls.SCHEMA_VERSION}"
    assert schema["title"] == "image"
    assert frozen_schema_filename(cls) == f"image-{cls.SCHEMA_VERSION}.json"


def test_dumped_schema_ids_match_declaring_class() -> None:
    """Verify every dumped schema's $id and title come from the declaring
    class, not an inherited concrete schema (e.g. visit_image, which
    subclasses masked_image).
    """
    for cls in available_schema_classes():
        schema = dump_schema(cls)
        expected = f"https://images.lsst.io/schemas/{cls.SCHEMA_NAME}-{cls.SCHEMA_VERSION}"
        assert schema["$id"] == expected, cls.SCHEMA_NAME
        assert schema["title"] == cls.SCHEMA_NAME


def test_write_and_check_round_trip(tmp_path: Path) -> None:
    """Verify write produces files that check accepts, and rewrite is a
    no-op.
    """
    changed = write_frozen_schemas(tmp_path)
    assert changed
    assert check_frozen_schemas(tmp_path) == []
    assert write_frozen_schemas(tmp_path) == []


def test_frozen_layout_per_schema_subdirectory(tmp_path: Path) -> None:
    """Verify frozen files are laid out as ``{name}/{name}-{version}.json``
    so the directory scales as versions accumulate.
    """
    write_frozen_schemas(tmp_path)
    (image_cls,) = [c for c in available_schema_classes() if c.SCHEMA_NAME == "image"]
    path = frozen_schema_path(tmp_path, image_cls)
    assert path == tmp_path / "image" / frozen_schema_filename(image_cls)
    assert path.exists()
    assert not list(tmp_path.glob("*.json"))


def test_check_reports_missing_and_stale(tmp_path: Path) -> None:
    """Verify check flags a missing file and a stale file."""
    write_frozen_schemas(tmp_path)
    (cls, *_) = available_schema_classes()
    path = frozen_schema_path(tmp_path, cls)
    stale = json.loads(path.read_text())
    stale["description"] = "something else"
    path.write_text(json.dumps(stale, indent=2, sort_keys=True) + "\n")
    problems = check_frozen_schemas(tmp_path)
    assert any("differs" in p for p in problems)
    path.unlink()
    problems = check_frozen_schemas(tmp_path)
    assert any("missing" in p for p in problems)


def test_write_preserves_superseded_versions(tmp_path: Path) -> None:
    """Verify write never deletes a frozen file for an old schema version."""
    old = tmp_path / "image" / "image-0.9.9.json"
    old.parent.mkdir()
    old.write_text("{}\n")
    write_frozen_schemas(tmp_path)
    assert old.exists()


def test_fixtures_validate_against_frozen_schemas() -> None:
    """Verify representative archive fixtures validate against the frozen
    schemas with a strict draft 2020-12 validator, which also proves every
    reference inside the published documents is resolvable.
    """
    jsonschema = pytest.importorskip("jsonschema")
    # piff_psf deliberately keeps a legacy dict-shaped image_pos that readers
    # accept but the current writer (and therefore the schema) does not emit.
    legacy_shaped = {"piff_psf"}
    checked = 0
    for fixture_path in sorted((Path(__file__).parent / "data" / "schema_v1").glob("*.json")):
        if fixture_path.stem in legacy_shaped:
            continue
        instance = json.loads(fixture_path.read_text())
        name, _, version = instance["schema_url"].rsplit("/", 1)[-1].rpartition("-")
        schema_file = REPO_SCHEMA_DIR / name / f"{name}-{version}.json"
        schema = json.loads(schema_file.read_text())
        jsonschema.Draft202012Validator(schema).validate(instance)
        checked += 1
    assert checked


def test_committed_frozen_schemas_are_current() -> None:
    """Verify the git-committed frozen schema files match the models.

    A failure here means a serialization model changed; run
    ``lsst-images-admin schemas write`` and commit the result.
    """
    problems = check_frozen_schemas(REPO_SCHEMA_DIR)
    assert not problems, (
        "Frozen schema files are stale; run 'lsst-images-admin schemas write' "
        "and commit the result: " + ", ".join(problems)
    )

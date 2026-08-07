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

import json
from pathlib import Path
from typing import Any, ClassVar

import pydantic
import pytest
from packaging.version import Version

# Importing this registers the purpose-built projection_test and retire_test
# schemas whose fixtures live under fixture_doubles/.  It is a plain module
# rather than a test module, so registration does not depend on pytest's
# collection order.
from schema_doubles import ProjectionTestModel, RetireTestModel  # noqa: F401

import lsst.images.tests._schema_fixtures as schema_fixture_module
from lsst.images.serialization import ArchiveReadError, ArchiveTree, InputArchive
from lsst.images.tests import (
    SchemaFixtureError,
    canonical_fixture_text,
    check_schema_fixtures,
    compare_fixture_versions,
    current_fixture_path,
    fixture_version,
    freeze_schema_fixtures,
    iter_schema_fixtures,
    read_fixture_tree,
    refresh_schema_fixtures,
)
from lsst.images.tests._schema_fixtures import (
    _AS_SHIPPED_VARIANT,
    _CANONICAL_VARIANT,
    _fixture_dir_path,
    _fixture_filename,
)


class _FrozenDouble(ArchiveTree):
    """Finalized-version double for the fixture machinery tests."""

    SCHEMA_NAME: ClassVar[str] = "fixture_machinery_frozen"
    SCHEMA_VERSION: ClassVar[str] = "1.2.3"
    MIN_READ_VERSION: ClassVar[int] = 1
    PUBLIC_TYPE: ClassVar[type] = dict

    label: str = pydantic.Field(default="", description="A value to round-trip.")

    def deserialize(self, archive: InputArchive[Any], **kwargs: Any) -> Any:
        return {"label": self.label}


class _DevDouble(ArchiveTree):
    """Development-version double for the fixture machinery tests."""

    SCHEMA_NAME: ClassVar[str] = "fixture_machinery_dev"
    SCHEMA_VERSION: ClassVar[str] = "2.0.0.dev3"
    MIN_READ_VERSION: ClassVar[int] = 1
    PUBLIC_TYPE: ClassVar[type] = dict

    label: str = pydantic.Field(default="", description="A value to round-trip.")

    def deserialize(self, archive: InputArchive[Any], **kwargs: Any) -> Any:
        return {"label": self.label}


class _StrictDouble(ArchiveTree):
    """Double with a required field, used to force a genuine pydantic
    validation failure distinct from a schema-compatibility rejection.
    """

    SCHEMA_NAME: ClassVar[str] = "fixture_machinery_strict"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"
    MIN_READ_VERSION: ClassVar[int] = 1
    PUBLIC_TYPE: ClassVar[type] = dict

    required: str = pydantic.Field(description="A value with no default.")

    def deserialize(self, archive: InputArchive[Any], **kwargs: Any) -> Any:
        return {"required": self.required}


# All three test doubles above share this module, so `available_schema_classes`
# filtering by `package=<double>.__module__` always sees all three regardless
# of which double a test is nominally about; tests that assert an exact
# problem list rather than membership must exempt the siblings they are not
# exercising.
_SIBLING_SCHEMAS = ("fixture_machinery_dev", "fixture_machinery_strict")


def test_fixture_version_strips_the_dev_counter() -> None:
    """Verify a filename version keeps the release part and a bare .dev."""
    assert fixture_version("1.2.3") == "1.2.3"
    assert fixture_version("2.0.0.dev0") == "2.0.0.dev"
    assert fixture_version("2.0.0.dev3") == "2.0.0.dev"


def test_fixture_filename_and_path() -> None:
    """Verify names and paths follow {name}/{name}-{version}[-{variant}]."""
    assert _fixture_filename("image", "1.0.0") == "image-1.0.0.json"
    assert _fixture_filename("image", "1.0.0", "dp1") == "image-1.0.0-dp1.json"
    root = Path("/fixtures")
    assert _fixture_dir_path(root, "image", "1.0.0") == root / "image" / "image-1.0.0.json"
    assert _fixture_dir_path(root, "image", "1.0.0", "dp1") == root / "image" / "image-1.0.0-dp1.json"


def test_current_fixture_path_resolves_by_schema_name() -> None:
    """Verify a consumer can find a fixture without naming a version.

    Covers both a finalized schema and a development one, since the live
    version is what the filename is built from.
    """
    root = Path("/fixtures")
    assert current_fixture_path(root, _FrozenDouble.SCHEMA_NAME) == (
        root / "fixture_machinery_frozen" / "fixture_machinery_frozen-1.2.3.json"
    )
    assert current_fixture_path(root, _DevDouble.SCHEMA_NAME, variant="dp1") == (
        root / "fixture_machinery_dev" / "fixture_machinery_dev-2.0.0.dev-dp1.json"
    )
    with pytest.raises(LookupError, match="no_such_schema"):
        current_fixture_path(root, "no_such_schema")


def test_canonical_fixture_text_is_indented_and_newline_terminated() -> None:
    """Verify the canonical form is indent=2 with a trailing newline."""
    text = canonical_fixture_text(_FrozenDouble(label="x"))
    assert text.startswith('{\n  "schema_version"')
    assert text.endswith("}\n")
    assert json.loads(text)["label"] == "x"


def test_canonical_fixture_text_is_idempotent() -> None:
    """Verify re-reading canonical text reproduces it byte for byte."""
    first = canonical_fixture_text(_FrozenDouble(label="x"))
    tree = _FrozenDouble.model_validate_json(first)
    assert canonical_fixture_text(tree) == first


def test_iter_schema_fixtures_parses_the_layout(tmp_path: Path) -> None:
    """Verify the scan reports version, variant, retired flag and class."""
    directory = tmp_path / "fixture_machinery_frozen"
    (directory / "retired").mkdir(parents=True)
    (directory / "fixture_machinery_frozen-1.2.3.json").write_text(
        canonical_fixture_text(_FrozenDouble(label="current"))
    )
    (directory / "fixture_machinery_frozen-1.2.3-as_shipped.json").write_text(
        canonical_fixture_text(_FrozenDouble(label="shipped"))
    )
    (directory / "fixture_machinery_frozen-1.2.3-canonical.json").write_text(
        canonical_fixture_text(_FrozenDouble(label="shipped"))
    )
    (directory / "retired" / "fixture_machinery_frozen-1.0.0.json").write_text("{}\n")
    found = {(f.version, f.variant, f.retired) for f in iter_schema_fixtures(tmp_path)}
    assert found == {
        ("1.2.3", None, False),
        ("1.2.3", _AS_SHIPPED_VARIANT, False),
        ("1.2.3", _CANONICAL_VARIANT, False),
        ("1.0.0", None, True),
    }
    by_variant = {f.variant: f for f in iter_schema_fixtures(tmp_path)}
    assert by_variant[_AS_SHIPPED_VARIANT].is_as_shipped
    assert not by_variant[_AS_SHIPPED_VARIANT].is_canonical_twin
    assert by_variant[_CANONICAL_VARIANT].is_canonical_twin is True
    assert by_variant[_CANONICAL_VARIANT].is_as_shipped is False
    assert by_variant[None].tree_cls is _FrozenDouble


def test_iter_schema_fixtures_reports_an_unknown_schema(tmp_path: Path) -> None:
    """Verify a directory with no matching schema yields tree_cls None."""
    directory = tmp_path / "not_a_schema"
    directory.mkdir()
    (directory / "not_a_schema-1.0.0.json").write_text("{}\n")
    (fixture,) = list(iter_schema_fixtures(tmp_path))
    assert fixture.tree_cls is None
    assert fixture.name == "not_a_schema"


def test_iter_schema_fixtures_skips_non_fixture_files(tmp_path: Path) -> None:
    """Verify a README and a mis-named file are not reported as fixtures."""
    directory = tmp_path / "fixture_machinery_frozen"
    directory.mkdir()
    (tmp_path / "README.md").write_text("notes\n")
    (directory / "fixture_machinery_frozen-1.2.3.json").write_text(canonical_fixture_text(_FrozenDouble()))
    (directory / "wrong-name.json").write_text("{}\n")
    assert [f.variant for f in iter_schema_fixtures(tmp_path)] == [None]


def test_read_fixture_tree(tmp_path: Path) -> None:
    """Verify a fixture is validated through its live model."""
    directory = tmp_path / "fixture_machinery_frozen"
    directory.mkdir()
    path = directory / "fixture_machinery_frozen-1.2.3.json"
    path.write_text(canonical_fixture_text(_FrozenDouble(label="read me")))
    (fixture,) = list(iter_schema_fixtures(tmp_path))
    tree = read_fixture_tree(fixture)
    assert isinstance(tree, ArchiveTree)
    assert tree.label == "read me"  # type: ignore[attr-defined]


def _write(directory: Path, name: str, text: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(text)
    return path


def test_check_accepts_a_sound_tree(tmp_path: Path) -> None:
    """Verify a canonical fixture at the live version reports no problems."""
    _write(
        tmp_path / "fixture_machinery_frozen",
        "fixture_machinery_frozen-1.2.3.json",
        canonical_fixture_text(_FrozenDouble(label="ok")),
    )
    problems = check_schema_fixtures(tmp_path, package=_FrozenDouble.__module__, exempt=_SIBLING_SCHEMAS)
    assert problems == []


def test_check_reports_drift(tmp_path: Path) -> None:
    """Verify a fixture that does not round-trip is reported."""
    text = canonical_fixture_text(_FrozenDouble(label="ok"))
    stale = json.loads(text)
    del stale["label"]
    _write(
        tmp_path / "fixture_machinery_frozen",
        "fixture_machinery_frozen-1.2.3.json",
        json.dumps(stale, indent=2) + "\n",
    )
    problems = check_schema_fixtures(tmp_path, package=_FrozenDouble.__module__)
    assert any("not canonical" in p for p in problems)


def test_check_wording_distinguishes_frozen_from_development(tmp_path: Path) -> None:
    """Verify a frozen fixture says to bump, a dev fixture says to refresh."""
    frozen = json.loads(canonical_fixture_text(_FrozenDouble(label="x")))
    del frozen["label"]
    _write(
        tmp_path / "fixture_machinery_frozen",
        "fixture_machinery_frozen-1.2.3.json",
        json.dumps(frozen, indent=2) + "\n",
    )
    dev = json.loads(canonical_fixture_text(_DevDouble(label="x")))
    del dev["label"]
    _write(
        tmp_path / "fixture_machinery_dev",
        "fixture_machinery_dev-2.0.0.dev.json",
        json.dumps(dev, indent=2) + "\n",
    )
    problems = "\n".join(check_schema_fixtures(tmp_path, package=_FrozenDouble.__module__))
    assert "bump SCHEMA_VERSION" in problems
    assert "fixtures refresh" in problems


def test_check_reports_a_stamp_that_disagrees_with_the_filename(tmp_path: Path) -> None:
    """Verify a filename version that contradicts the stamp is reported."""
    tree = json.loads(canonical_fixture_text(_FrozenDouble()))
    _write(
        tmp_path / "fixture_machinery_frozen",
        "fixture_machinery_frozen-9.9.9.json",
        json.dumps(tree, indent=2) + "\n",
    )
    problems = check_schema_fixtures(tmp_path, package=_FrozenDouble.__module__)
    assert any("filename version" in p for p in problems)


def test_check_reports_a_malformed_schema_version_without_raising(tmp_path: Path) -> None:
    """Verify a malformed-but-integer-major stamp is reported, not raised.

    ``fixture_version`` parses the stamp with ``packaging.version.Version``,
    which raises ``InvalidVersion`` for a plausible hand-edit like "1.2.O"
    (letter O for zero); ``check_schema_fixtures`` promises never to raise,
    so that must become a reported problem like any other, not an escaped
    exception.
    """
    tree = json.loads(canonical_fixture_text(_FrozenDouble()))
    tree["schema_version"] = "1.2.O"
    _write(
        tmp_path / "fixture_machinery_frozen",
        "fixture_machinery_frozen-1.2.3.json",
        json.dumps(tree, indent=2) + "\n",
    )
    problems = check_schema_fixtures(tmp_path, package=_FrozenDouble.__module__)
    assert any("1.2.O" in p and "not a valid version" in p for p in problems)


def test_check_reports_a_missing_development_fixture(tmp_path: Path) -> None:
    """Verify a development schema with no fixture at all is reported."""
    (tmp_path / "fixture_machinery_dev").mkdir()
    problems = check_schema_fixtures(tmp_path, package=_DevDouble.__module__)
    assert any("fixture_machinery_dev-2.0.0.dev.json" in p and "missing" in p for p in problems)


def test_check_reports_an_unknown_schema_directory(tmp_path: Path) -> None:
    """Verify a directory that maps to no registered schema is reported."""
    _write(tmp_path / "not_a_schema", "not_a_schema-1.0.0.json", "{}\n")
    problems = check_schema_fixtures(tmp_path, package=_FrozenDouble.__module__)
    assert any("not_a_schema" in p and "not registered" in p for p in problems)


def test_check_requires_a_retired_fixture_to_be_rejected(tmp_path: Path) -> None:
    """Verify a retired fixture that still validates is reported."""
    _write(
        tmp_path / "fixture_machinery_frozen",
        "fixture_machinery_frozen-1.2.3.json",
        canonical_fixture_text(_FrozenDouble()),
    )
    _write(
        tmp_path / "fixture_machinery_frozen" / "retired",
        "fixture_machinery_frozen-1.0.0.json",
        canonical_fixture_text(_FrozenDouble()),
    )
    problems = check_schema_fixtures(tmp_path, package=_FrozenDouble.__module__)
    assert any("retired" in p and "still validates" in p for p in problems)


def test_check_applies_identity_checks_to_a_retired_fixture(tmp_path: Path) -> None:
    """Verify expected rejection does not hide a retired fixture's identity."""
    directory = tmp_path / "fixture_machinery_frozen"
    _write(
        directory,
        "fixture_machinery_frozen-1.2.3.json",
        canonical_fixture_text(_FrozenDouble()),
    )
    retired = json.loads(canonical_fixture_text(_FrozenDouble()))
    retired["schema_version"] = "1.0.0"
    retired["schema_url"] = "https://example.org/wrong"
    retired["min_read_version"] = 999
    _write(
        directory / "retired",
        "fixture_machinery_frozen-1.0.0.json",
        json.dumps(retired, indent=2) + "\n",
    )
    problems = check_schema_fixtures(tmp_path, package=_FrozenDouble.__module__, exempt=_SIBLING_SCHEMAS)
    assert any("schema_url" in problem and "expected" in problem for problem in problems)


def test_check_reports_malformed_retired_fixture_json(tmp_path: Path) -> None:
    """Verify invalid JSON is a defect, not successful retirement."""
    directory = tmp_path / "fixture_machinery_frozen"
    _write(
        directory,
        "fixture_machinery_frozen-1.2.3.json",
        canonical_fixture_text(_FrozenDouble()),
    )
    _write(directory / "retired", "fixture_machinery_frozen-1.0.0.json", "{\n")
    problems = check_schema_fixtures(tmp_path, package=_FrozenDouble.__module__, exempt=_SIBLING_SCHEMAS)
    assert any("is not valid JSON" in problem for problem in problems)


def test_check_reports_an_unexpected_retired_read_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify a programming failure cannot masquerade as retirement."""
    directory = tmp_path / "fixture_machinery_frozen"
    _write(
        directory,
        "fixture_machinery_frozen-1.2.3.json",
        canonical_fixture_text(_FrozenDouble()),
    )
    retired = json.loads(canonical_fixture_text(_FrozenDouble()))
    retired["schema_version"] = "1.0.0"
    retired["schema_url"] = "https://images.lsst.io/schemas/fixture_machinery_frozen-1.0.0"
    retired["min_read_version"] = 999
    _write(
        directory / "retired",
        "fixture_machinery_frozen-1.0.0.json",
        json.dumps(retired, indent=2) + "\n",
    )
    original = schema_fixture_module.read_fixture_tree

    def fail_retired(fixture: schema_fixture_module.SchemaFixture) -> ArchiveTree:
        if fixture.retired:
            raise RuntimeError("reader bug")
        return original(fixture)

    monkeypatch.setattr(schema_fixture_module, "read_fixture_tree", fail_retired)
    problems = check_schema_fixtures(tmp_path, package=_FrozenDouble.__module__, exempt=_SIBLING_SCHEMAS)
    assert any("unexpected RuntimeError: reader bug" in problem for problem in problems)


def test_check_binds_the_as_shipped_pair(tmp_path: Path) -> None:
    """Verify an as_shipped fixture needs a twin matching its canonical
    read.
    """
    directory = tmp_path / "fixture_machinery_frozen"
    _write(
        directory,
        "fixture_machinery_frozen-1.2.3.json",
        canonical_fixture_text(_FrozenDouble()),
    )
    shipped = json.loads(canonical_fixture_text(_FrozenDouble(label="shipped")))
    _write(
        directory,
        "fixture_machinery_frozen-1.2.3-as_shipped.json",
        json.dumps(shipped, indent=2) + "\n",
    )
    exempt = _SIBLING_SCHEMAS
    problems = check_schema_fixtures(tmp_path, package=_FrozenDouble.__module__, exempt=exempt)
    assert any("canonical" in p and "missing" in p for p in problems)

    _write(
        directory,
        "fixture_machinery_frozen-1.2.3-canonical.json",
        canonical_fixture_text(_FrozenDouble(label="something else")),
    )
    problems = check_schema_fixtures(tmp_path, package=_FrozenDouble.__module__, exempt=exempt)
    assert any("does not match" in p for p in problems)

    _write(
        directory,
        "fixture_machinery_frozen-1.2.3-canonical.json",
        canonical_fixture_text(_FrozenDouble(label="shipped")),
    )
    assert check_schema_fixtures(tmp_path, package=_FrozenDouble.__module__, exempt=exempt) == []


def test_check_pairs_fixtures_with_frozen_documents(tmp_path: Path) -> None:
    """Verify both pairing directions against a frozen schema directory."""
    fixtures = tmp_path / "fixtures"
    schemas = tmp_path / "schemas"
    schemas.mkdir()
    _write(
        fixtures / "fixture_machinery_frozen",
        "fixture_machinery_frozen-1.2.3.json",
        canonical_fixture_text(_FrozenDouble()),
    )
    # No frozen document at all: the finalized schema is unpaired.
    exempt = _SIBLING_SCHEMAS
    problems = check_schema_fixtures(
        fixtures, schema_directory=schemas, package=_FrozenDouble.__module__, exempt=exempt
    )
    assert any("has no frozen document" in p for p in problems)

    _write(schemas / "fixture_machinery_frozen", "fixture_machinery_frozen-1.2.3.json", "{}\n")
    assert (
        check_schema_fixtures(
            fixtures, schema_directory=schemas, package=_FrozenDouble.__module__, exempt=exempt
        )
        == []
    )


def test_check_honors_exemptions(tmp_path: Path) -> None:
    """Verify an exempt schema is not reported as missing a fixture."""
    problems = check_schema_fixtures(
        tmp_path,
        package=_DevDouble.__module__,
        exempt=["fixture_machinery_dev", "fixture_machinery_frozen", "fixture_machinery_strict"],
    )
    assert problems == []


def test_check_does_not_flag_a_retired_fixture_as_unfrozen(tmp_path: Path) -> None:
    """Verify a retired fixture with no frozen document is not reported.

    A retired fixture is checked only for being rejected; it must not also
    be run through the fixture/frozen-document pairing check, which would
    otherwise flag it as an interrupted freeze.
    """
    fixtures = tmp_path / "fixtures"
    schemas = tmp_path / "schemas"
    _write(
        fixtures / "fixture_machinery_frozen",
        "fixture_machinery_frozen-1.2.3.json",
        canonical_fixture_text(_FrozenDouble()),
    )
    _write(schemas / "fixture_machinery_frozen", "fixture_machinery_frozen-1.2.3.json", "{}\n")
    retired = json.loads(canonical_fixture_text(_FrozenDouble()))
    retired["schema_version"] = "1.0.0"
    retired["schema_url"] = "https://images.lsst.io/schemas/fixture_machinery_frozen-1.0.0"
    retired["min_read_version"] = 999
    _write(
        fixtures / "fixture_machinery_frozen" / "retired",
        "fixture_machinery_frozen-1.0.0.json",
        json.dumps(retired, indent=2) + "\n",
    )
    problems = check_schema_fixtures(
        fixtures, schema_directory=schemas, package=_FrozenDouble.__module__, exempt=_SIBLING_SCHEMAS
    )
    assert problems == []


def test_check_does_not_canonicalize_a_retired_as_shipped_fixture(tmp_path: Path) -> None:
    """Verify a retired as_shipped fixture is not run through
    canonicalization.

    A retired fixture no longer validates by design, so passing it to
    `_check_as_shipped_pairs` would always fail to canonicalize; that
    pairing check must skip retired fixtures just as the single-fixture
    check does.
    """
    directory = tmp_path / "fixture_machinery_frozen"
    _write(
        directory,
        "fixture_machinery_frozen-1.2.3.json",
        canonical_fixture_text(_FrozenDouble()),
    )
    retired = json.loads(canonical_fixture_text(_FrozenDouble()))
    retired["schema_version"] = "1.0.0"
    retired["schema_url"] = "https://images.lsst.io/schemas/fixture_machinery_frozen-1.0.0"
    retired["min_read_version"] = 999
    # Both halves of the retired pair, so the bug being guarded against
    # (reaching read_fixture_tree(shipped) instead of short-circuiting on a
    # missing twin) is exercised precisely.
    _write(
        directory / "retired",
        "fixture_machinery_frozen-1.0.0-as_shipped.json",
        json.dumps(retired, indent=2) + "\n",
    )
    _write(
        directory / "retired",
        "fixture_machinery_frozen-1.0.0-canonical.json",
        json.dumps(retired, indent=2) + "\n",
    )
    problems = check_schema_fixtures(tmp_path, package=_FrozenDouble.__module__, exempt=_SIBLING_SCHEMAS)
    assert problems == []


def test_check_reports_a_non_retired_as_shipped_fixture_that_fails_to_validate(
    tmp_path: Path,
) -> None:
    """Verify a non-retired as_shipped fixture that fails to validate is
    reported rather than raising.

    `_check_as_shipped_pairs` reads each as_shipped fixture through its live
    model to canonicalize it for comparison against its twin; unlike a
    retired fixture, this one is expected to validate, so a failure here is
    a genuine defect that must be reported, not an exception that escapes
    `check_schema_fixtures`.
    """
    directory = tmp_path / "fixture_machinery_strict"
    tree = json.loads(canonical_fixture_text(_StrictDouble(required="x")))
    del tree["required"]
    _write(
        directory,
        "fixture_machinery_strict-1.0.0-as_shipped.json",
        json.dumps(tree, indent=2) + "\n",
    )
    _write(
        directory,
        "fixture_machinery_strict-1.0.0-canonical.json",
        canonical_fixture_text(_StrictDouble(required="x")),
    )
    problems = check_schema_fixtures(tmp_path, package=_StrictDouble.__module__)
    assert any("cannot be canonicalized" in p for p in problems)


def test_check_reports_a_validation_failure_without_raising(tmp_path: Path) -> None:
    """Verify a fixture that fails pydantic validation is reported as a
    problem instead of propagating the exception; check_schema_fixtures
    must never raise.
    """
    tree = json.loads(canonical_fixture_text(_StrictDouble(required="x")))
    del tree["required"]
    _write(
        tmp_path / "fixture_machinery_strict",
        "fixture_machinery_strict-1.0.0.json",
        json.dumps(tree, indent=2) + "\n",
    )
    problems = check_schema_fixtures(tmp_path, package=_StrictDouble.__module__)
    assert any("does not validate" in p for p in problems)


def test_check_reports_a_read_incompatibility_without_raising(tmp_path: Path) -> None:
    """Verify a fixture whose min_read_version exceeds the reader major is
    reported as a problem instead of propagating the exception.
    """
    tree = json.loads(canonical_fixture_text(_FrozenDouble()))
    tree["min_read_version"] = 999
    _write(
        tmp_path / "fixture_machinery_frozen",
        "fixture_machinery_frozen-1.2.3.json",
        json.dumps(tree, indent=2) + "\n",
    )
    problems = check_schema_fixtures(tmp_path, package=_FrozenDouble.__module__)
    assert any("does not validate" in p for p in problems)


def test_check_reports_a_superseded_frozen_document_with_no_fixture(tmp_path: Path) -> None:
    """Verify every frozen document, not just the live one, needs a
    same-version fixture, and that a retired fixture satisfies it.
    """
    fixtures = tmp_path / "fixtures"
    schemas = tmp_path / "schemas"
    _write(
        fixtures / "fixture_machinery_frozen",
        "fixture_machinery_frozen-1.2.3.json",
        canonical_fixture_text(_FrozenDouble()),
    )
    _write(schemas / "fixture_machinery_frozen", "fixture_machinery_frozen-1.2.3.json", "{}\n")
    # A superseded version, frozen but with no fixture anywhere.
    _write(schemas / "fixture_machinery_frozen", "fixture_machinery_frozen-1.0.0.json", "{}\n")
    problems = check_schema_fixtures(
        fixtures, schema_directory=schemas, package=_FrozenDouble.__module__, exempt=_SIBLING_SCHEMAS
    )
    assert any("fixture_machinery_frozen-1.0.0.json" in p and "has no fixture" in p for p in problems)

    # A retired fixture at that version counts as present.
    retired = json.loads(canonical_fixture_text(_FrozenDouble()))
    retired["schema_version"] = "1.0.0"
    retired["schema_url"] = "https://images.lsst.io/schemas/fixture_machinery_frozen-1.0.0"
    retired["min_read_version"] = 999
    _write(
        fixtures / "fixture_machinery_frozen" / "retired",
        "fixture_machinery_frozen-1.0.0.json",
        json.dumps(retired, indent=2) + "\n",
    )
    problems = check_schema_fixtures(
        fixtures, schema_directory=schemas, package=_FrozenDouble.__module__, exempt=_SIBLING_SCHEMAS
    )
    assert problems == []


def test_check_reports_a_schema_url_that_disagrees_with_the_stamp(tmp_path: Path) -> None:
    """Verify an on-disk schema_url that does not match its own
    schema_version stamp is reported.
    """
    tree = json.loads(canonical_fixture_text(_FrozenDouble()))
    tree["schema_url"] = "https://example.org/wrong"
    _write(
        tmp_path / "fixture_machinery_frozen",
        "fixture_machinery_frozen-1.2.3.json",
        json.dumps(tree, indent=2) + "\n",
    )
    problems = check_schema_fixtures(tmp_path, package=_FrozenDouble.__module__)
    assert any("schema_url" in p and "expected" in p for p in problems)


def test_refresh_rewrites_a_stale_development_fixture(tmp_path: Path) -> None:
    """Verify refresh canonicalizes a drifted .dev fixture in place."""
    stale = json.loads(canonical_fixture_text(_DevDouble(label="x")))
    del stale["label"]
    path = _write(
        tmp_path / "fixture_machinery_dev",
        "fixture_machinery_dev-2.0.0.dev.json",
        json.dumps(stale, indent=2) + "\n",
    )
    assert refresh_schema_fixtures(tmp_path, package=_DevDouble.__module__) == [path]
    assert json.loads(path.read_text())["label"] == ""
    assert refresh_schema_fixtures(tmp_path, package=_DevDouble.__module__) == []


def test_refresh_refuses_to_rewrite_a_frozen_fixture(tmp_path: Path) -> None:
    """Verify a drifted frozen fixture raises rather than being rewritten."""
    stale = json.loads(canonical_fixture_text(_FrozenDouble(label="x")))
    del stale["label"]
    path = _write(
        tmp_path / "fixture_machinery_frozen",
        "fixture_machinery_frozen-1.2.3.json",
        json.dumps(stale, indent=2) + "\n",
    )
    before = path.read_text()
    with pytest.raises(SchemaFixtureError, match="bump SCHEMA_VERSION"):
        refresh_schema_fixtures(tmp_path, package=_FrozenDouble.__module__)
    assert path.read_text() == before


def test_refresh_seeds_a_missing_development_fixture(tmp_path: Path) -> None:
    """Verify a new .dev fixture is seeded from the newest existing version."""
    directory = tmp_path / "fixture_machinery_dev"
    _write(
        directory,
        "fixture_machinery_dev-1.0.0.json",
        canonical_fixture_text(_DevDouble(label="carried forward")),
    )
    _write(
        directory,
        "fixture_machinery_dev-1.0.0-dp1.json",
        canonical_fixture_text(_DevDouble(label="variant carried forward")),
    )
    written = refresh_schema_fixtures(tmp_path, package=_DevDouble.__module__)
    seeded = directory / "fixture_machinery_dev-2.0.0.dev.json"
    seeded_variant = directory / "fixture_machinery_dev-2.0.0.dev-dp1.json"
    assert set(written) == {seeded, seeded_variant}
    assert json.loads(seeded.read_text())["label"] == "carried forward"
    assert json.loads(seeded_variant.read_text())["label"] == "variant carried forward"
    assert json.loads(seeded.read_text())["schema_version"] == "2.0.0.dev3"


def test_refresh_seeds_from_a_frozen_release_over_a_stale_dev_residual(tmp_path: Path) -> None:
    """Verify a frozen release outranks a same-release .dev residual.

    A stale ``.dev`` fixture left behind by an interrupted freeze and its
    properly frozen same-release counterpart must not tie when picking a
    newest source to seed from: the frozen one is strictly newer and must
    win regardless of which one a directory listing happens to yield
    first.
    """
    directory = tmp_path / "fixture_machinery_dev"
    _write(
        directory,
        "fixture_machinery_dev-1.0.0.dev.json",
        canonical_fixture_text(_DevDouble(label="stale dev residual")),
    )
    _write(
        directory,
        "fixture_machinery_dev-1.0.0.json",
        canonical_fixture_text(_DevDouble(label="frozen release")),
    )
    written = refresh_schema_fixtures(tmp_path, package=_DevDouble.__module__)
    seeded = directory / "fixture_machinery_dev-2.0.0.dev.json"
    assert written == [seeded]
    assert json.loads(seeded.read_text())["label"] == "frozen release"


def test_refresh_refuses_to_change_a_canonical_twin_at_a_frozen_version(tmp_path: Path) -> None:
    """Verify refresh cannot erase a shipped-file compatibility failure."""
    directory = tmp_path / "fixture_machinery_frozen"
    _write(
        directory,
        "fixture_machinery_frozen-1.2.3.json",
        canonical_fixture_text(_FrozenDouble()),
    )
    twin = _write(
        directory,
        "fixture_machinery_frozen-1.2.3-canonical.json",
        canonical_fixture_text(_FrozenDouble(label="old normalization")),
    )
    _write(
        directory,
        "fixture_machinery_frozen-1.2.3-as_shipped.json",
        canonical_fixture_text(_FrozenDouble(label="new normalization")),
    )
    before = twin.read_text()
    with pytest.raises(SchemaFixtureError, match="compatibility contract"):
        refresh_schema_fixtures(tmp_path, package=_FrozenDouble.__module__)
    assert twin.read_text() == before


def test_refresh_regenerates_a_development_canonical_twin(tmp_path: Path) -> None:
    """Verify a development twin remains derived refreshable data."""
    directory = tmp_path / "fixture_machinery_dev"
    _write(
        directory,
        "fixture_machinery_dev-2.0.0.dev-as_shipped.json",
        canonical_fixture_text(_DevDouble(label="shipped")),
    )
    twin = directory / "fixture_machinery_dev-2.0.0.dev-canonical.json"
    assert refresh_schema_fixtures(tmp_path, package=_DevDouble.__module__) == [twin]
    assert json.loads(twin.read_text())["label"] == "shipped"


def test_refresh_never_touches_retired_fixtures(tmp_path: Path) -> None:
    """Verify a retired fixture is left exactly as it is."""
    directory = tmp_path / "fixture_machinery_dev"
    _write(
        directory,
        "fixture_machinery_dev-2.0.0.dev.json",
        canonical_fixture_text(_DevDouble()),
    )
    retired = _write(directory / "retired", "fixture_machinery_dev-1.0.0.json", "{}\n")
    refresh_schema_fixtures(tmp_path, package=_DevDouble.__module__)
    assert retired.read_text() == "{}\n"


def test_refresh_never_touches_a_retired_as_shipped_fixture(tmp_path: Path) -> None:
    """Verify a retired as_shipped fixture is not read while regenerating
    canonical twins.

    A retired fixture no longer validates by design (that is what makes it
    retired), so a twin-regeneration loop that does not exclude retired
    fixtures would pass it to `read_fixture_tree` and raise instead of
    leaving it alone, exactly the defect `check_schema_fixtures` was fixed
    for with its own retired fixtures.
    """
    directory = tmp_path / "fixture_machinery_frozen"
    _write(
        directory,
        "fixture_machinery_frozen-1.2.3.json",
        canonical_fixture_text(_FrozenDouble()),
    )
    retired = json.loads(canonical_fixture_text(_FrozenDouble()))
    retired["min_read_version"] = 999
    retired_path = _write(
        directory / "retired",
        "fixture_machinery_frozen-1.0.0-as_shipped.json",
        json.dumps(retired, indent=2) + "\n",
    )
    before = retired_path.read_text()
    assert refresh_schema_fixtures(tmp_path, package=_FrozenDouble.__module__) == []
    assert retired_path.read_text() == before
    assert not (directory / "retired" / "fixture_machinery_frozen-1.0.0-canonical.json").exists()


def test_freeze_writes_the_final_version_and_deletes_the_dev_file(tmp_path: Path) -> None:
    """Verify freeze writes {name}-{X.Y.Z}.json and removes the .dev file.

    The double's live version is finalized, so its .dev fixture is the
    residue of a freeze that has not been completed yet.
    """
    directory = tmp_path / "fixture_machinery_frozen"
    dev_tree = json.loads(canonical_fixture_text(_FrozenDouble(label="frozen soon")))
    dev_tree["schema_version"] = "1.2.3.dev0"
    dev_path = _write(
        directory, "fixture_machinery_frozen-1.2.3.dev.json", json.dumps(dev_tree, indent=2) + "\n"
    )
    frozen_path = directory / "fixture_machinery_frozen-1.2.3.json"
    assert freeze_schema_fixtures(tmp_path, package=_FrozenDouble.__module__) == [(frozen_path, dev_path)]
    assert not dev_path.exists()
    assert json.loads(frozen_path.read_text())["schema_version"] == "1.2.3"
    assert json.loads(frozen_path.read_text())["label"] == "frozen soon"
    assert freeze_schema_fixtures(tmp_path, package=_FrozenDouble.__module__) == []


def test_freeze_carries_variants_and_refuses_to_clobber(tmp_path: Path) -> None:
    """Verify each variant freezes too, and an existing target is refused."""
    directory = tmp_path / "fixture_machinery_frozen"
    _write(
        directory,
        "fixture_machinery_frozen-1.2.3.dev-dp1.json",
        canonical_fixture_text(_FrozenDouble(label="variant")),
    )
    ((written, _),) = freeze_schema_fixtures(tmp_path, package=_FrozenDouble.__module__)
    assert written.name == "fixture_machinery_frozen-1.2.3-dp1.json"

    _write(
        directory,
        "fixture_machinery_frozen-1.2.3.dev-dp1.json",
        canonical_fixture_text(_FrozenDouble(label="again")),
    )
    with pytest.raises(SchemaFixtureError, match="already exists"):
        freeze_schema_fixtures(tmp_path, package=_FrozenDouble.__module__)


def test_freeze_refuses_an_as_shipped_development_fixture(tmp_path: Path) -> None:
    """Verify freeze neither rewrites shipped bytes nor mislabels their
    version.
    """
    directory = tmp_path / "fixture_machinery_frozen"
    shipped_text = (
        '{"label": "shipped", "min_read_version": 1, '
        '"schema_version": "1.2.3.dev0", '
        '"schema_url": "https://images.lsst.io/schemas/fixture_machinery_frozen-1.2.3.dev0"}'
    )
    shipped_path = _write(directory, "fixture_machinery_frozen-1.2.3.dev-as_shipped.json", shipped_text)
    twin_path = _write(
        directory,
        "fixture_machinery_frozen-1.2.3.dev-canonical.json",
        canonical_fixture_text(_FrozenDouble(label="shipped")),
    )
    with pytest.raises(SchemaFixtureError, match="cannot be frozen"):
        freeze_schema_fixtures(tmp_path, package=_FrozenDouble.__module__)
    assert shipped_path.read_text() == shipped_text
    assert twin_path.exists()
    assert not (directory / "fixture_machinery_frozen-1.2.3-as_shipped.json").exists()


def test_freeze_preflights_all_target_conflicts_before_writing(tmp_path: Path) -> None:
    """Verify a later variant conflict leaves earlier sources untouched."""
    directory = tmp_path / "fixture_machinery_frozen"
    base_source = _write(
        directory,
        "fixture_machinery_frozen-1.2.3.dev.json",
        canonical_fixture_text(_FrozenDouble(label="base")),
    )
    variant_source = _write(
        directory,
        "fixture_machinery_frozen-1.2.3.dev-dp1.json",
        canonical_fixture_text(_FrozenDouble(label="variant")),
    )
    _write(
        directory,
        "fixture_machinery_frozen-1.2.3-dp1.json",
        canonical_fixture_text(_FrozenDouble(label="existing")),
    )
    with pytest.raises(SchemaFixtureError, match="already exists"):
        freeze_schema_fixtures(tmp_path, package=_FrozenDouble.__module__)
    assert base_source.exists()
    assert variant_source.exists()
    assert not (directory / "fixture_machinery_frozen-1.2.3.json").exists()


DOUBLE_DIR = Path(__file__).parent / "data" / "fixture_doubles"


def test_compare_fixture_versions_reports_a_shared_path_that_differs() -> None:
    """Verify a value the old file expressed differently is reported."""
    problems = compare_fixture_versions({"kept": "a"}, {"kept": "b", "added": 1})
    assert len(problems) == 1
    assert ".kept" in problems[0]


def test_compare_fixture_versions_reports_a_path_lost_from_the_current_dump() -> None:
    """Verify a path in the old dump but not the current one is reported."""
    problems = compare_fixture_versions({"kept": "a", "dropped": 1}, {"kept": "a"})
    assert any("dropped" in p for p in problems)


def test_compare_fixture_versions_is_strict_about_numeric_type() -> None:
    """Verify an integer where the current dump has a float is reported.

    This is the drift that put integers into piff_psf's float fields.
    """
    assert compare_fixture_versions({"x": 0}, {"x": 0.0})


def test_compare_fixture_versions_treats_nan_as_equal() -> None:
    """Verify NaN compares equal to NaN, since many fields default to it."""
    nan = float("nan")
    assert compare_fixture_versions({"x": nan}, {"x": nan}) == []


def test_compare_fixture_versions_requires_equal_list_lengths() -> None:
    """Verify a list length change is reported rather than partly compared."""
    problems = compare_fixture_versions({"xs": [1, 2]}, {"xs": [1, 2, 3]})
    assert any("length" in p for p in problems)


def test_compare_fixture_versions_recurses() -> None:
    """Verify nested dicts and lists are compared path by path."""
    old = {"a": {"b": [{"c": 1}]}}
    current = {"a": {"b": [{"c": 2}]}}
    problems = compare_fixture_versions(old, current)
    assert ".a.b[0].c" in problems[0]


def test_compare_fixture_versions_ignores_a_later_born_field_with_a_real_value() -> None:
    """Verify a later-born field is ignored using the on-disk expressed set,
    not ``old``'s own materialized-default value.

    ``old`` already carries ``added`` at its post-read default (``0``),
    exactly as reading a fixture that never mentioned it through the live
    model would produce; only ``on_disk`` (the raw file) lacks it. Guards
    against the regression this design corrects: deciding "later-born" from
    ``old`` itself would treat this exact input as already agreeing only by
    the coincidence that the real value chosen for ``added`` in ``current``
    happens to equal its default -- it would report ``.added: 0 != 7`` for
    any other value, which is exactly what the pre-fix algorithm does.
    """
    old = {"kept": "exemplar", "added": 0}
    on_disk = {"kept": "exemplar"}
    current = {"kept": "exemplar", "added": 7}
    assert compare_fixture_versions(old, current, on_disk=on_disk) == []


def test_compare_fixture_versions_expressed_paths_include_ancestors() -> None:
    """Verify a path counts as expressed when a container ancestor is on
    disk with a different container kind, even with no exact path match.

    A field whose shape changed between versions -- here, a pair collapsed
    from a two-element list into named components -- has no path on disk
    that matches its new leaves exactly, but its container does, as a list
    where the dump now has a dict; that reshape is what must be enough to
    bring the leaves into comparison.
    """
    on_disk = {"image_pos": [1.0, 2.0]}
    old = {"image_pos": {"x": 1.0, "y": 2.0}}
    current = {"image_pos": {"x": 1.0, "y": 9.0}}
    problems = compare_fixture_versions(old, current, on_disk=on_disk)
    assert any(".image_pos.y" in p for p in problems)


def test_compare_fixture_versions_ignores_a_later_born_field_under_an_empty_container() -> None:
    """Verify a later-born field nested under an on-disk *empty* container
    is ignored, not compared via the ancestor rule.

    The container itself (``.container``) is on disk as an empty dict, and
    stays a dict in the old dump too -- no reshape -- so its new leaf must
    not inherit "expressed" status from it.
    """
    on_disk = {"kept": "exemplar", "container": {}}
    old = {"kept": "exemplar", "container": {"newfield": 0}}
    current = {"kept": "exemplar", "container": {"newfield": 42}}
    assert compare_fixture_versions(old, current, on_disk=on_disk) == []


def test_compare_fixture_versions_ignores_a_later_born_field_under_a_populated_container() -> None:
    """Verify a later-born field nested under an on-disk *populated*
    container is ignored, not compared via the ancestor rule.

    Same as the empty-container case, except the container already has a
    sibling field on disk, which is the more realistic shape these schemas
    will actually take when a field is added to an existing group.
    """
    on_disk = {"kept": "exemplar", "container": {"already": 1}}
    old = {"kept": "exemplar", "container": {"already": 1, "newfield": 0}}
    current = {"kept": "exemplar", "container": {"already": 1, "newfield": 42}}
    assert compare_fixture_versions(old, current, on_disk=on_disk) == []


def test_compare_fixture_versions_exact_path_match_still_compares() -> None:
    """Verify a path present on disk exactly still gets compared, with
    ``on_disk`` passed explicitly rather than left to default to ``old``.
    """
    on_disk = {"kept": "a"}
    old = {"kept": "a"}
    current = {"kept": "b"}
    assert compare_fixture_versions(old, current, on_disk=on_disk) == [".kept: 'a' != 'b'"]


def test_fixture_double_projection_contract() -> None:
    """Verify the additive fixture double projects with no escape hatch.

    ``projection_test`` is the oracle's positive control on real committed
    files. Its 1.1.0 fixture's ``added`` field carries a real, non-default
    value (``7``): a later-born field is only meaningfully tested by a value
    that is not its own type's default.  The other cross-version double,
    ``migration_test``, renames a path and so cannot serve as a positive
    control; it is declared in ``test_schema_fixtures.EXPECTED_DIVERGENCE``
    and ``test_migration_test_fixture_reads_from_disk`` asserts its
    transformed value.
    """
    fixtures = {f.version: f for f in iter_schema_fixtures(DOUBLE_DIR) if f.name == "projection_test"}
    old_version = min(fixtures, key=Version)
    current_version = max(fixtures, key=Version)
    on_disk = json.loads(fixtures[old_version].path.read_text())
    old = json.loads(canonical_fixture_text(read_fixture_tree(fixtures[old_version])))
    current = json.loads(canonical_fixture_text(read_fixture_tree(fixtures[current_version])))
    assert compare_fixture_versions(old, current, on_disk=on_disk) == []


def test_projection_double_negative_control() -> None:
    """Verify the oracle fails when the old fixture disagrees.

    Without this, a projection check that could never fail would still pass
    every run.
    """
    fixtures = {f.version: f for f in iter_schema_fixtures(DOUBLE_DIR) if f.name == "projection_test"}
    on_disk = json.loads(fixtures["1.0.0"].path.read_text())
    old = json.loads(canonical_fixture_text(read_fixture_tree(fixtures["1.0.0"])))
    current = json.loads(canonical_fixture_text(read_fixture_tree(fixtures["1.1.0"])))
    old["kept"] = "tampered"
    assert compare_fixture_versions(old, current, on_disk=on_disk)


def test_projection_double_stamps_normalize_to_the_live_version() -> None:
    """Verify the older fixture's stamps compare equal after reading."""
    fixtures = {f.version: f for f in iter_schema_fixtures(DOUBLE_DIR) if f.name == "projection_test"}
    old = json.loads(canonical_fixture_text(read_fixture_tree(fixtures["1.0.0"])))
    assert old["schema_version"] == "1.1.0"


def test_retire_double_fixture_is_rejected() -> None:
    """Verify a fixture under retired/ raises rather than reading.

    A 1.0.0 tree cannot satisfy the 2.0.0 shape and no migration covers the
    gap, so Pydantic rejects it. ``read_fixture_tree`` translates that
    validation failure to the archive-domain error required by the retirement
    contract.
    """
    fixtures = {
        (f.version, f.retired): f for f in iter_schema_fixtures(DOUBLE_DIR) if f.name == "retire_test"
    }
    retired = fixtures[("1.0.0", True)]
    assert retired.retired
    with pytest.raises(ArchiveReadError):
        read_fixture_tree(retired)


def test_current_fixture_of_a_retiring_schema_still_reads() -> None:
    """Verify retirement scopes to the old version, not the whole schema."""
    fixtures = {
        (f.version, f.retired): f for f in iter_schema_fixtures(DOUBLE_DIR) if f.name == "retire_test"
    }
    tree = read_fixture_tree(fixtures[("2.0.0", False)])
    assert tree.required_at_v2 == "present"  # type: ignore[attr-defined]


def test_check_accepts_a_retired_fixture_that_is_rejected(tmp_path: Path) -> None:
    """Verify a properly retired fixture is not reported as a problem."""
    directory = tmp_path / "retire_test"
    _write(
        directory,
        "retire_test-2.0.0.json",
        canonical_fixture_text(RetireTestModel(required_at_v2="present")),
    )
    _write(
        directory / "retired",
        "retire_test-1.0.0.json",
        '{\n  "schema_version": "1.0.0",\n  "min_read_version": 1,\n'
        '  "schema_url": "https://images.lsst.io/schemas/retire_test-1.0.0"\n}\n',
    )
    # Every schema in schema_doubles shares this __module__, so the package
    # filter pulls all of them in; exempt the ones this test is not about,
    # or their absent fixtures are reported as missing.
    assert (
        check_schema_fixtures(
            tmp_path,
            package=RetireTestModel.__module__,
            exempt=[
                "chain_test",
                "gap_test",
                "migration_isolation_test",
                "migration_test",
                "projection_test",
                "unmigratable_test",
            ],
        )
        == []
    )


def test_check_reports_a_retired_fixture_at_the_live_version(tmp_path: Path) -> None:
    """Verify a stray file under retired/ is reported, not quietly accepted.

    Retirement is how a *superseded* version's fixture is kept, so a retired
    fixture at the live version is a file in the wrong place.  Every other
    check passes it: its stamps are self-consistent, and the model rejects it,
    which is exactly what a retired fixture is required to do.
    """
    directory = tmp_path / "retire_test"
    _write(
        directory,
        "retire_test-2.0.0.json",
        canonical_fixture_text(RetireTestModel(required_at_v2="present")),
    )
    _write(
        directory / "retired",
        "retire_test-2.0.0.json",
        '{\n  "schema_version": "2.0.0",\n  "min_read_version": 2,\n'
        '  "schema_url": "https://images.lsst.io/schemas/retire_test-2.0.0"\n}\n',
    )
    problems = check_schema_fixtures(
        tmp_path,
        package=RetireTestModel.__module__,
        exempt=[
            "chain_test",
            "gap_test",
            "migration_isolation_test",
            "migration_test",
            "projection_test",
            "unmigratable_test",
        ],
    )
    assert [p for p in problems if "retired" in p], problems

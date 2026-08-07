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
"""The committed schema fixtures, checked against the live models.

Every fixture is read through its model on every run, so a model change that
alters what a file looks like fails here rather than passing silently.  See
:ref:`lsst.images-schema-versioning` for the lifecycle these checks enforce.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# Importing this registers the purpose-built schemas whose fixtures live under
# fixture_doubles/.  It is a plain module rather than a test module so the
# registration cannot depend on pytest's collection order.
import schema_doubles  # noqa: F401
from packaging.version import Version

from lsst.images.serialization import (
    ArchiveReadError,
    JsonRef,
    dump_schema,
    is_development_version,
    parameterize_tree,
    read_archive,
)
from lsst.images.tests import (
    SchemaFixture,
    canonical_fixture_text,
    check_schema_fixtures,
    compare_fixture_versions,
    fixture_version,
    iter_schema_fixtures,
    read_fixture_tree,
)

FIXTURE_DIR = Path(__file__).parent / "data" / "schemas"
SCHEMA_DIR = Path(__file__).parent.parent / "schemas"

NO_FIXTURE = {
    "psfex_psf": "needs PSFEx data this package cannot construct",
}
"""Schemas allowed to have no fixture, with the reason each.

Kept beside ``_DEVELOPMENT_SCHEMAS`` in style: an explicit list, so adding a
schema without a fixture is a deliberate, reviewable act.
"""

DOUBLE_DIR = Path(__file__).parent / "data" / "fixture_doubles"

# The ladder runs over the package's own fixtures and over the purpose-built
# doubles, so every rung -- including the cross-version projection, which no
# lsst.images schema can exercise yet -- has real cases from day one.
_FIXTURES = sorted(
    [*iter_schema_fixtures(FIXTURE_DIR), *iter_schema_fixtures(DOUBLE_DIR)],
    key=lambda f: f.path,
)

_READABLE = [f for f in _FIXTURES if not f.retired]
_RETIRED = [f for f in _FIXTURES if f.retired]


def _id(fixture: SchemaFixture) -> str:
    return fixture.path.name


def test_fixtures_present() -> None:
    """Verify the fixture tree is populated."""
    assert _READABLE, f"no fixtures found in {FIXTURE_DIR}"


def test_check_schema_fixtures_is_clean() -> None:
    """Verify the committed fixture tree has no reported problems.

    A failure here names the remedy: 'fixtures refresh' for a development
    schema, or a SCHEMA_VERSION bump for a finalized one.
    """
    problems = check_schema_fixtures(FIXTURE_DIR, schema_directory=SCHEMA_DIR, exempt=NO_FIXTURE)
    assert not problems, "\n".join(problems)


def test_every_schema_without_a_fixture_has_a_recorded_reason() -> None:
    """Verify the exemption list has not grown stale.

    An exemption for a schema that now has a fixture should be deleted.
    """
    have = {f.name for f in _READABLE}
    assert not (set(NO_FIXTURE) & have), "exempt schemas that now have fixtures"


@pytest.mark.parametrize("fixture", _READABLE, ids=_id)
def test_fixture_reads(fixture: SchemaFixture) -> None:
    """Verify every retained fixture validates through its live model."""
    assert fixture.tree_cls is not None, f"{fixture.name} is not registered"
    tree = read_fixture_tree(fixture)
    assert tree.schema_version == fixture.tree_cls.SCHEMA_VERSION


def _older_release(schema_version: str) -> str | None:
    """Return a release strictly older than ``schema_version``'s release.

    Operates on the release components only (major, minor, patch), ignoring
    any development-release suffix, so the result is genuinely older even
    when ``schema_version`` is itself a development release.  The rightmost
    nonzero component is decremented, which is always the release
    immediately below in version ordering.

    Parameters
    ----------
    schema_version
        Schema version string, e.g. ``1.0.0`` or ``1.0.0.dev0``.

    Returns
    -------
    `str` or `None`
        The older release string, or `None` if the release is already
        ``0.0.0`` and so has no older release to name.
    """
    version = Version(schema_version)
    major, minor, patch = version.major, version.minor, version.micro
    if patch > 0:
        return f"{major}.{minor}.{patch - 1}"
    if minor > 0:
        return f"{major}.{minor - 1}.0"
    if major > 0:
        return f"{major - 1}.0.0"
    return None


@pytest.mark.parametrize("fixture", _READABLE, ids=_id)
def test_fixture_upgrades_on_write(fixture: SchemaFixture) -> None:
    """Verify a tree stamped at an older version re-stamps at the live one.

    One model class serves every version of its schema, so reading an older,
    compatible tree and writing it back should emit the current shape and
    stamps.  Re-reading the fixture's own output would only restate what
    `~lsst.images.serialization.ArchiveTree` normalization already produced,
    so this instead fabricates an older ``schema_version`` / ``schema_url`` /
    ``min_read_version`` on the fixture's payload -- in memory only, the
    committed file is never touched -- and validates that mutated payload
    directly, exercising the normalization rather than restating its result.
    """
    tree_cls = fixture.tree_cls
    assert tree_cls is not None
    older = _older_release(tree_cls.SCHEMA_VERSION)
    if older is None:
        pytest.skip(f"{tree_cls.SCHEMA_NAME} has no older release to stamp the payload with")
    if (
        Version(older).major != Version(tree_cls.SCHEMA_VERSION).major
        and Version(tree_cls.SCHEMA_VERSION).major > 1
    ):
        # A relabeled-only payload keeps every current-shape field, so it
        # only stands in for a genuinely older tree within the same major:
        # nothing was renamed, split, or removed there, so the current model
        # accepts it unchanged.  Once a schema has moved past major 1, an
        # older major may need a registered migration to bridge a real shape
        # difference, and this relabel-only payload cannot exercise that; the
        # migration chain tests and the projection oracle cover cross-major
        # compatibility instead.  No schema in this package has moved past
        # major 1 yet, so this never skips real coverage today.
        pytest.skip(f"{tree_cls.SCHEMA_NAME} has moved past major 1; not a relabel-only case")
    payload = json.loads(fixture.path.read_text())
    payload["schema_version"] = older
    payload["min_read_version"] = 1
    payload["schema_url"] = f"{tree_cls.SCHEMA_URL_BASE}/{tree_cls.SCHEMA_NAME}-{older}"
    parameterized = parameterize_tree(tree_cls, JsonRef)
    tree = parameterized.model_validate_json(json.dumps(payload))
    dumped = json.loads(canonical_fixture_text(tree))
    assert dumped["schema_version"] == tree_cls.SCHEMA_VERSION
    assert dumped["min_read_version"] == tree_cls.MIN_READ_VERSION
    assert dumped["schema_url"] == (
        f"{tree_cls.SCHEMA_URL_BASE}/{tree_cls.SCHEMA_NAME}-{tree_cls.SCHEMA_VERSION}"
    )


@pytest.mark.parametrize("fixture", _READABLE, ids=_id)
def test_fixture_is_canonical(fixture: SchemaFixture) -> None:
    """Verify a same-version fixture round-trips byte for byte.

    An as_shipped fixture is exempt: re-serializing would rewrite the shipped
    spelling it exists to preserve, and its canonical twin carries the
    pairwise check instead.
    """
    assert fixture.tree_cls is not None, f"{fixture.name} is not registered"
    if fixture.is_as_shipped:
        pytest.skip("as_shipped fixtures preserve real bytes and are not canonicalized")
    if fixture.version != fixture_version(fixture.tree_cls.SCHEMA_VERSION):
        pytest.skip("older retained version; covered by the projection check")
    assert fixture.path.read_text() == canonical_fixture_text(read_fixture_tree(fixture))


@pytest.mark.parametrize("fixture", _READABLE, ids=_id)
def test_fixture_conforms_to_its_schema_document(fixture: SchemaFixture) -> None:
    """Verify every fixture validates against a draft 2020-12 schema.

    A frozen version validates against its committed document, which also
    proves every reference inside the published document resolves.  A
    development version validates against the live generated schema, so a
    development fixture is no longer skipped as it was before this ticket.

    A fixture whose schema looks finalized (no ``.dev`` suffix) but has no
    frozen document under ``schemas/`` -- true of every purpose-built double
    in ``schema_doubles``, which is never frozen or published -- falls back
    to the live generated schema as well, provided the fixture is at that
    schema's live version; an older fixture of such a schema has no document
    of any kind to validate against and is still skipped.
    """
    jsonschema = pytest.importorskip("jsonschema")
    tree_cls = fixture.tree_cls
    assert tree_cls is not None
    live_version = fixture_version(tree_cls.SCHEMA_VERSION)
    if is_development_version(tree_cls.SCHEMA_VERSION):
        schema = dump_schema(tree_cls)
    else:
        document = SCHEMA_DIR / fixture.name / f"{fixture.name}-{fixture.version}.json"
        if document.exists():
            schema = json.loads(document.read_text())
        elif fixture.version == live_version:
            schema = dump_schema(tree_cls)
        else:
            pytest.skip(f"{document} is not committed")
    jsonschema.Draft202012Validator(schema).validate(json.loads(fixture.path.read_text()))


@pytest.mark.parametrize("fixture", _READABLE, ids=_id)
def test_fixture_deserializes(fixture: SchemaFixture) -> None:
    """Verify the tree yields its in-memory object where deps allow.

    A fixture whose deserialization needs an optional dependency (Piff,
    PSFEx, lsst.afw) raises ArchiveReadError at the point of use rather than
    at read time, so that outcome is accepted here.
    """
    try:
        obj = read_archive(fixture.path)
    except ArchiveReadError as exc:
        pytest.skip(f"deserialization needs an unavailable dependency: {exc}")
    assert obj is not None


@pytest.mark.parametrize("fixture", _RETIRED, ids=_id)
def test_retired_fixture_is_rejected(fixture: SchemaFixture) -> None:
    """Verify a retired fixture raises rather than reading.

    Rejection is a contract worth testing: retiring a fixture is how a read
    contract ends, and this asserts the new behavior rather than merely
    stopping to test the old one.
    """
    with pytest.raises(ArchiveReadError):
        read_fixture_tree(fixture)


EXPECTED_DIVERGENCE: dict[tuple[str, str], str] = {}
"""Fixture pairs whose projection is expected to fail, with the reason each.

A path-based oracle cannot compare a field whose spelling changed, so such a
pair is registered here with a reason and a dedicated test asserts the
result directly.  This is the sole registry of those declarations, and it is
empty until a schema has a second version.
"""


def _older_fixtures() -> list[SchemaFixture]:
    """Return every retained fixture at less than its schema's live version."""
    return [
        f
        for f in _READABLE
        if f.tree_cls is not None and f.version != fixture_version(f.tree_cls.SCHEMA_VERSION)
    ]


@pytest.mark.parametrize("fixture", _older_fixtures(), ids=_id)
def test_older_fixture_projects_onto_the_current_one(fixture: SchemaFixture) -> None:
    """Verify an older fixture reads to the same exemplar as the current one.

    Reading proves acceptance; this proves the result is right.  Paths the
    older file could not express are ignored, so additive evolution needs no
    hand-written expectations.
    """
    assert fixture.tree_cls is not None, f"{fixture.name} is not registered"
    if (reason := EXPECTED_DIVERGENCE.get((fixture.name, fixture.version))) is not None:
        pytest.skip(f"expected divergence: {reason}")
    current = next(
        f
        for f in _READABLE
        if f.name == fixture.name
        and f.variant == fixture.variant
        and f.version == fixture_version(fixture.tree_cls.SCHEMA_VERSION)
    )
    problems = compare_fixture_versions(
        json.loads(canonical_fixture_text(read_fixture_tree(fixture))),
        json.loads(canonical_fixture_text(read_fixture_tree(current))),
        on_disk=json.loads(fixture.path.read_text()),
    )
    assert not problems, "\n".join(problems)


def test_every_expected_divergence_names_a_real_fixture_pair() -> None:
    """Verify the escape hatch has not gone stale.

    An entry naming a pair the ladder no longer runs skips nothing, so it no
    longer records a reviewed decision and should be deleted.  Without this,
    a divergence declared for a fixture that was retired or renamed would sit
    there reading as coverage.
    """
    declared = set(EXPECTED_DIVERGENCE)
    older = {(f.name, f.version) for f in _older_fixtures()}
    assert not (declared - older), "expected-divergence entries with no matching older fixture"

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
"""Committed reference fixtures for the serialization data models.

Every retained ``{name}-{version}`` fixture under ``tests/data/schemas`` is
the instance-level twin of the frozen schema document of the same version
under ``schemas``: it is read through the live model on every test run, so a
model change that would alter what a file looks like shows up as fixture
drift rather than as silence.

The layout, the lifecycle and the checks are described in
:ref:`lsst.images-schema-versioning`.
"""

from __future__ import annotations

__all__ = (
    "SchemaFixture",
    "SchemaFixtureError",
    "canonical_fixture_text",
    "check_schema_fixtures",
    "compare_fixture_versions",
    "current_fixture_path",
    "fixture_version",
    "freeze_schema_fixtures",
    "iter_schema_fixtures",
    "read_fixture_tree",
    "refresh_schema_fixtures",
)

import dataclasses
import json
import math
import re
from collections.abc import Collection, Iterator
from pathlib import Path

import pydantic
from packaging.version import InvalidVersion, Version

from ..serialization import (
    ArchiveReadError,
    ArchiveTree,
    JsonRef,
    available_schema_classes,
    class_for_schema,
    frozen_schema_path,
    is_development_version,
    parameterize_tree,
)

_AS_SHIPPED_VARIANT = "as_shipped"
"""Variant name reserved for a fixture whose bytes are preserved exactly as a
real shipped file produced them.  Never canonicalized or rewritten."""

_CANONICAL_VARIANT = "canonical"
"""Variant name reserved for the canonicalized twin of an ``as_shipped``
fixture: the same tree as the current code reads and would write it."""

_FIXTURE_RE = re.compile(r"^(?P<version>\d+\.\d+\.\d+(?:\.dev)?)(?:-(?P<variant>[a-z0-9_]+))?$")
"""Filename pattern, applied to the stem with the ``{name}-`` prefix removed.

The version token is matched greedily first, so a variant name can never be
mistaken for part of the version.
"""

_RETIRED_DIR = "retired"


def fixture_version(schema_version: str) -> str:
    """Return the fixture-filename version for a schema version.

    Parameters
    ----------
    schema_version
        Schema version string, e.g. ``1.0.0`` or ``1.0.0.dev0``.

    Returns
    -------
    `str`
        The release part, with a bare ``.dev`` suffix for a development
        release.  The exact development counter lives only in the
        ``schema_version`` stamp inside the file, so a development schema has
        exactly one fixture path whatever its counter.
    """
    version = Version(schema_version)
    release = f"{version.major}.{version.minor}.{version.micro}"
    return f"{release}.dev" if is_development_version(schema_version) else release


def _fixture_filename(name: str, version: str, variant: str | None = None) -> str:
    """Return the fixture filename for a schema name, version and variant.

    Parameters
    ----------
    name
        Schema name.
    version
        Fixture-filename version, as returned by `fixture_version`.
    variant
        Variant name, or `None` for the base fixture.
    """
    stem = f"{name}-{version}" if variant is None else f"{name}-{version}-{variant}"
    return f"{stem}.json"


def _fixture_dir_path(directory: Path, name: str, version: str, variant: str | None = None) -> Path:
    """Return the path of a fixture within a fixture tree.

    Parameters
    ----------
    directory
        Directory holding the fixture tree.
    name
        Schema name.
    version
        Fixture-filename version, as returned by `fixture_version`.
    variant
        Variant name, or `None` for the base fixture.

    Notes
    -----
    Files are laid out as ``{name}/{name}-{version}.json``, mirroring the
    frozen schema documents so the instance-level and schema-level trees are
    navigable the same way.
    """
    return directory / name / _fixture_filename(name, version, variant)


def current_fixture_path(directory: Path, name: str, *, variant: str | None = None) -> Path:
    """Return the fixture path for a schema's live version, by name.

    Parameters
    ----------
    directory
        Directory holding the fixture tree.
    name
        Schema name, e.g. ``visit_image``.
    variant
        Variant name, or `None` for the base fixture.

    Returns
    -------
    `pathlib.Path`
        Path to the fixture for whatever version the code is currently at.

    Raises
    ------
    LookupError
        If no schema is registered under ``name``.

    Notes
    -----
    This is the entry point for tests that just want representative data.
    Resolving through the live class means such a test does not need editing
    when a schema is frozen or its version is bumped.
    """
    tree_cls = class_for_schema(name)
    if tree_cls is None:
        raise LookupError(f"No schema is registered under {name!r}.")
    return _fixture_dir_path(directory, name, fixture_version(tree_cls.SCHEMA_VERSION), variant)


def canonical_fixture_text(tree: ArchiveTree) -> str:
    """Return the canonical file serialization of a tree.

    Parameters
    ----------
    tree
        Serialization model instance to write.

    Notes
    -----
    Model field order is kept rather than sorting keys, so the version stamps
    stay at the top of the file where a human reading it looks first.  The form
    is idempotent: re-reading the result and re-serializing reproduces it byte
    for byte.
    """
    return tree.model_dump_json(indent=2) + "\n"


@dataclasses.dataclass(frozen=True)
class SchemaFixture:
    """One committed fixture file, located and classified."""

    path: Path
    """Path to the fixture file."""

    name: str
    """Schema name, taken from the containing directory."""

    version: str
    """Fixture-filename version, e.g. ``1.0.0`` or ``1.0.0.dev``."""

    variant: str | None
    """Variant name, or `None` for the base fixture."""

    retired: bool
    """Whether the fixture sits in the schema's ``retired`` subdirectory, and
    so is expected to be rejected rather than read."""

    tree_cls: type[ArchiveTree] | None
    """The registered model class, or `None` if no schema of this name is
    registered."""

    @property
    def is_as_shipped(self) -> bool:
        """Whether this fixture preserves real shipped bytes."""
        return self.variant == _AS_SHIPPED_VARIANT

    @property
    def is_canonical_twin(self) -> bool:
        """Whether this fixture is the canonicalized twin of an ``as_shipped``
        sibling.
        """
        return self.variant == _CANONICAL_VARIANT

    def problem(self, message: str) -> str:
        """Return ``message`` prefixed with this fixture's filename.

        Parameters
        ----------
        message
            Problem description, phrased to read after the filename.
        """
        return f"{self.path.name}: {message}"


def iter_schema_fixtures(directory: Path) -> Iterator[SchemaFixture]:
    """Yield every fixture found under a fixture tree.

    Parameters
    ----------
    directory
        Directory holding the fixture tree.

    Yields
    ------
    `SchemaFixture`
        One entry per file whose name matches the fixture grammar, sorted by
        path.  Files that do not match are skipped, so a README or other
        supporting file in the tree is not reported as a fixture.
    """
    if not directory.is_dir():
        return
    for schema_dir in sorted(p for p in directory.iterdir() if p.is_dir()):
        name = schema_dir.name
        tree_cls = class_for_schema(name)
        for retired, parent in ((False, schema_dir), (True, schema_dir / _RETIRED_DIR)):
            if not parent.is_dir():
                continue
            for path in sorted(parent.glob("*.json")):
                if (match := _FIXTURE_RE.fullmatch(path.stem.removeprefix(f"{name}-"))) is None:
                    continue
                yield SchemaFixture(
                    path=path,
                    name=name,
                    version=match.group("version"),
                    variant=match.group("variant"),
                    retired=retired,
                    tree_cls=tree_cls,
                )


def read_fixture_tree(fixture: SchemaFixture) -> ArchiveTree:
    """Validate a fixture through its live model and return the tree.

    Parameters
    ----------
    fixture
        The fixture to read.

    Returns
    -------
    `~lsst.images.serialization.ArchiveTree`
        The validated tree, parameterized over
        `~lsst.images.serialization.JsonRef` as the JSON backend does.

    Raises
    ------
    ArchiveReadError
        Raised when a retired fixture is rejected by Pydantic validation.
    RuntimeError
        If the fixture's schema is not registered.
    """
    if fixture.tree_cls is None:
        raise RuntimeError(f"No schema is registered under {fixture.name!r}.")
    parameterized = parameterize_tree(fixture.tree_cls, JsonRef)
    try:
        return parameterized.model_validate_json(fixture.path.read_text())
    except pydantic.ValidationError as exc:
        if fixture.retired:
            raise ArchiveReadError(
                f"Retired fixture {fixture.path.name!r} is rejected by its current schema: {exc}"
            ) from exc
        raise


def _check_one(fixture: SchemaFixture) -> list[str]:
    """Return the problems found in a single fixture."""
    if fixture.tree_cls is None:
        return [fixture.problem(f"schema {fixture.name!r} is not registered")]
    problems: list[str] = []
    try:
        on_disk_json = fixture.path.read_text()
        on_disk = json.loads(on_disk_json)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [fixture.problem(f"is not valid JSON: {type(exc).__name__}: {exc}")]
    if not isinstance(on_disk, dict):
        return [fixture.problem(f"top-level JSON value is {type(on_disk).__name__}, expected object")]

    # Identity applies even to retired fixtures.  Check it before validation,
    # because rejection is expected for a retired tree and must not hide a bad
    # stamp, URL, or filename.
    on_disk_version = on_disk.get("schema_version")
    if not isinstance(on_disk_version, str):
        problems.append(fixture.problem(f"schema_version is {on_disk_version!r}, expected a version string"))
    else:
        expected_url = f"{fixture.tree_cls.SCHEMA_URL_BASE}/{fixture.name}-{on_disk_version}"
        on_disk_url = on_disk.get("schema_url")
        if on_disk_url != expected_url:
            problems.append(fixture.problem(f"schema_url is {on_disk_url!r}, expected {expected_url!r}"))
        try:
            on_disk_fixture_version = fixture_version(on_disk_version)
        except InvalidVersion as exc:
            # fixture_version parses the stamp with packaging.version.Version,
            # which raises for a malformed-but-string value (e.g. a
            # hand-edited "1.0.O", letter O for zero); this function's
            # contract is that it never raises, so report it like any other
            # fixture problem.
            problems.append(
                fixture.problem(f"schema_version {on_disk_version!r} is not a valid version: {exc}")
            )
        else:
            if fixture.version != on_disk_fixture_version:
                problems.append(
                    fixture.problem(
                        f"filename version {fixture.version!r} disagrees with stamp {on_disk_version!r}",
                    )
                )

    if fixture.retired:
        # Retirement keeps a *superseded* version's fixture, so a retired file
        # at or above the live version is misplaced rather than retired.  No
        # other check catches that: its stamps can be self-consistent, being
        # rejected is what a retired fixture is required to do, and the
        # never-frozen check below deliberately skips retired fixtures.
        retired_live_version = fixture_version(fixture.tree_cls.SCHEMA_VERSION)
        if Version(fixture.version) >= Version(retired_live_version):
            problems.append(
                fixture.problem(
                    f"is retired at {fixture.version}, which is not older than the live version "
                    f"{retired_live_version}; retirement is for a superseded version"
                )
            )

    try:
        tree = read_fixture_tree(fixture)
    except Exception as exc:
        if fixture.retired:
            if not isinstance(exc, ArchiveReadError):
                problems.append(
                    fixture.problem(
                        f"rejection raised unexpected {type(exc).__name__}: {exc}",
                    )
                )
            return problems
        problems.append(fixture.problem(f"does not validate: {type(exc).__name__}: {exc}"))
        return problems
    if fixture.retired:
        problems.append(fixture.problem("is retired but still validates; move it back or update the model"))
        return problems
    live_version = fixture_version(fixture.tree_cls.SCHEMA_VERSION)
    if fixture.version == live_version and not fixture.is_as_shipped:
        if on_disk_json != canonical_fixture_text(tree):
            if is_development_version(fixture.tree_cls.SCHEMA_VERSION):
                remedy = "run 'lsst-images-admin fixtures refresh'"
            else:
                remedy = "bump SCHEMA_VERSION rather than rewriting a frozen fixture"
            problems.append(fixture.problem(f"is not canonical; {remedy}"))
    return problems


def _check_as_shipped_pairs(fixtures: list[SchemaFixture]) -> list[str]:
    """Return the problems found in as_shipped / canonical fixture pairs.

    A retired fixture is excluded from both directions: it is checked only
    for being rejected, and (being retired) it no longer validates, so
    running it through `read_fixture_tree` here would always fail.
    """
    problems: list[str] = []
    twins = {(f.name, f.version): f for f in fixtures if f.is_canonical_twin and not f.retired}
    for shipped in fixtures:
        if shipped.retired or not shipped.is_as_shipped or shipped.tree_cls is None:
            continue
        key = (shipped.name, shipped.version)
        twin = twins.get(key)
        expected = _fixture_filename(shipped.name, shipped.version, _CANONICAL_VARIANT)
        if twin is None:
            problems.append(shipped.problem(f"canonical twin {expected} is missing"))
            continue
        try:
            text = canonical_fixture_text(read_fixture_tree(shipped))
        except Exception as exc:
            problems.append(shipped.problem(f"cannot be canonicalized: {exc}"))
            continue
        if twin.path.read_text() != text:
            problems.append(
                twin.problem(
                    "does not match the canonical read of its as_shipped sibling; "
                    "how a shipped file is read has changed, which is a compatibility "
                    "change and not a fixture to refresh",
                )
            )
    for twin in fixtures:
        if twin.retired or not twin.is_canonical_twin:
            continue
        sibling = twin.path.with_name(_fixture_filename(twin.name, twin.version, _AS_SHIPPED_VARIANT))
        if not sibling.exists():
            problems.append(twin.problem(f"has no as_shipped sibling {sibling.name}"))
    return problems


def check_schema_fixtures(
    directory: Path,
    *,
    schema_directory: Path | None = None,
    package: str = "lsst.images",
    exempt: Collection[str] = (),
) -> list[str]:
    """Check the committed fixtures against the current models.

    Parameters
    ----------
    directory
        Directory holding the fixture tree.
    schema_directory
        Directory holding the frozen schema documents.  When given, the
        fixture and schema trees are also checked for pairing in both
        directions.
    package
        Package whose schemas to check; see
        `~lsst.images.serialization.available_schema_classes`.
    exempt
        Schema names that are allowed to have no fixture, for schemas whose
        data this package cannot construct.

    Returns
    -------
    `list` [ `str` ]
        One problem description per defect found; empty when the fixture tree
        is sound.  This never raises, so a caller can report every problem at
        once.

    Notes
    -----
    A ``retired`` fixture is checked only for being rejected; it cannot be
    validated, so the canonical and pairing checks do not apply to it.  An
    ``as_shipped`` fixture is exempt from the canonical check, and its
    ``canonical`` twin carries the stronger pairwise check in its place.
    """
    exempt = frozenset(exempt)
    fixtures = list(iter_schema_fixtures(directory))
    problems: list[str] = []
    for fixture in fixtures:
        problems.extend(_check_one(fixture))
    problems.extend(_check_as_shipped_pairs(fixtures))
    live: set[tuple[str, str]] = set()
    scope: set[str] = set()
    for tree_cls in available_schema_classes(package):
        name = tree_cls.SCHEMA_NAME
        scope.add(name)
        version = fixture_version(tree_cls.SCHEMA_VERSION)
        live.add((name, version))
        if name in exempt:
            continue
        if not any(f.name == name and f.version == version for f in fixtures if not f.retired):
            problems.append(f"{_fixture_filename(name, version)}: missing")
        if schema_directory is not None and not is_development_version(tree_cls.SCHEMA_VERSION):
            if not frozen_schema_path(schema_directory, tree_cls).exists():
                problems.append(
                    f"{_fixture_filename(name, version)}: has no frozen document; "
                    "run 'lsst-images-admin schemas write'"
                )
    if schema_directory is not None:
        # Direction 1: a fixture at a version that is neither the live one
        # nor frozen anywhere -- an interrupted freeze left it behind.  A
        # retired fixture is excluded: it is checked only for being
        # rejected, and it is expected to have no frozen document once its
        # schema has moved on.
        for fixture in fixtures:
            if fixture.tree_cls is None or fixture.retired or (fixture.name, fixture.version) in live:
                continue
            document = schema_directory / fixture.name / f"{fixture.name}-{fixture.version}.json"
            if not document.exists():
                problems.append(
                    fixture.problem(
                        "is at a version that was never frozen and is not the live version; "
                        "an interrupted freeze leaves this behind",
                    )
                )
        # Direction 2: every frozen document -- not just the live one --
        # needs a same-version fixture, so a superseded version's fixture is
        # never silently dropped.  A fixture under retired/ still counts as
        # present: retirement is how a superseded version's fixture is kept.
        if schema_directory.is_dir():
            for schema_dir in sorted(p for p in schema_directory.iterdir() if p.is_dir()):
                name = schema_dir.name
                if name not in scope or name in exempt:
                    continue
                for path in sorted(schema_dir.glob("*.json")):
                    match = _FIXTURE_RE.fullmatch(path.stem.removeprefix(f"{name}-"))
                    if match is None:
                        continue
                    version = match.group("version")
                    if not any(f.name == name and f.version == version for f in fixtures):
                        problems.append(
                            f"{path.name}: has no fixture; run 'lsst-images-admin fixtures refresh'"
                        )
    return problems


class SchemaFixtureError(RuntimeError):
    """A fixture that must not be rewritten would change."""


def _newest_source(fixtures: list[SchemaFixture], name: str, variant: str | None) -> SchemaFixture | None:
    """Return the highest-version non-retired fixture to seed from."""
    candidates = [
        f
        for f in fixtures
        if f.name == name and f.variant == variant and not f.retired and f.tree_cls is not None
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda f: Version(f.version))


def refresh_schema_fixtures(directory: Path, *, package: str = "lsst.images") -> list[Path]:
    """Rewrite every development fixture in canonical form.

    Parameters
    ----------
    directory
        Directory holding the fixture tree.
    package
        Package whose schemas to refresh; see
        `~lsst.images.serialization.available_schema_classes`.

    Returns
    -------
    `list` [ `pathlib.Path` ]
        Paths that were created or rewritten.

    Raises
    ------
    SchemaFixtureError
        If a fixture at a finalized version would change.  Bump
        ``SCHEMA_VERSION`` instead of rewriting it.

    Notes
    -----
    Only fixtures of a schema whose live version is a development release are
    rewritten.  A development schema with no fixture yet is seeded from its
    newest non-retired fixture, per variant, so the exemplar is carried forward
    rather than reinvented.  ``retired`` fixtures are never touched.

    Canonical twins are regenerated from their ``as_shipped`` siblings only
    while their schema version is in development.  At a finalized version the
    twin pins how shipped bytes normalize on read, so changing or creating it
    is a reviewed, manual operation rather than something ``refresh`` may do.
    A retired ``as_shipped`` fixture is excluded, since it no longer validates
    and has no twin to regenerate.

    This writes and creates files; it never invokes version control.
    """
    fixtures = list(iter_schema_fixtures(directory))
    changed: list[Path] = []
    for tree_cls in available_schema_classes(package):
        name = tree_cls.SCHEMA_NAME
        live_version = fixture_version(tree_cls.SCHEMA_VERSION)
        if is_development_version(tree_cls.SCHEMA_VERSION):
            variants = {
                f.variant for f in fixtures if f.name == name and not f.retired and not f.is_canonical_twin
            } or {None}
            for variant in sorted(variants, key=lambda v: (v is not None, v or "")):
                if variant == _AS_SHIPPED_VARIANT:
                    continue
                target = _fixture_dir_path(directory, name, live_version, variant)
                source = (
                    SchemaFixture(
                        path=target,
                        name=name,
                        version=live_version,
                        variant=variant,
                        retired=False,
                        tree_cls=tree_cls,
                    )
                    if target.exists()
                    else _newest_source(fixtures, name, variant)
                )
                if source is None:
                    continue
                text = canonical_fixture_text(read_fixture_tree(source))
                if not target.exists() or target.read_text() != text:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(text)
                    changed.append(target)
        else:
            for fixture in fixtures:
                if fixture.name != name or fixture.retired or fixture.is_canonical_twin:
                    continue
                if fixture.is_as_shipped or fixture.version != live_version:
                    continue
                text = canonical_fixture_text(read_fixture_tree(fixture))
                if fixture.path.read_text() != text:
                    raise SchemaFixtureError(
                        f"{fixture.path.name} is at a finalized version and would change; "
                        "bump SCHEMA_VERSION rather than rewriting a frozen fixture."
                    )
        for shipped in fixtures:
            if shipped.name != name or shipped.retired or not shipped.is_as_shipped:
                continue
            twin = shipped.path.with_name(_fixture_filename(name, shipped.version, _CANONICAL_VARIANT))
            text = canonical_fixture_text(read_fixture_tree(shipped))
            if twin.exists() and twin.read_text() == text:
                continue
            if not shipped.version.endswith(".dev"):
                raise SchemaFixtureError(
                    f"{twin.name} is at a finalized version and would change; "
                    "how shipped bytes normalize is a compatibility contract, so bump "
                    "SCHEMA_VERSION rather than refreshing this twin."
                )
            twin.write_text(text)
            changed.append(twin)
    return changed


def freeze_schema_fixtures(directory: Path, *, package: str = "lsst.images") -> list[tuple[Path, Path]]:
    """Move ordinary development fixtures to their finalized versions.

    Parameters
    ----------
    directory
        Directory holding the fixture tree.
    package
        Package whose schemas to freeze; see
        `~lsst.images.serialization.available_schema_classes`.

    Returns
    -------
    `list` [ `tuple` [ `pathlib.Path`, `pathlib.Path` ] ]
        One ``(written, removed)`` pair per fixture that was frozen.

    Raises
    ------
    SchemaFixtureError
        If a target path already exists or an ``as_shipped`` development
        fixture cannot be frozen without rewriting its preserved bytes.

    Notes
    -----
    Writes the final-version fixture from the ``.dev`` fixture's content in
    canonical form, which normalizes the stamp from ``X.Y.Z.devN`` to
    ``X.Y.Z``, then deletes the ``.dev`` file.  Every ordinary variant is
    carried over.

    An ``as_shipped`` fixture cannot be frozen: changing its embedded
    development stamp would violate its byte-preservation contract, while
    copying it unchanged under a final-version filename would make the stamp
    and filename disagree.  Such a fixture must be replaced with bytes from a
    genuinely final-version shipped artifact before freezing.

    All targets, source reads, and conflicts are checked before any file is
    written or deleted, so a predictable validation or target conflict cannot
    leave a partially frozen fixture tree.

    This writes and deletes files; it never invokes version control.  Staging
    the addition and the deletion is left to the caller.
    """
    pending: list[tuple[Path, Path, str]] = []
    fixtures = list(iter_schema_fixtures(directory))
    for tree_cls in available_schema_classes(package):
        if is_development_version(tree_cls.SCHEMA_VERSION):
            continue
        name = tree_cls.SCHEMA_NAME
        live_version = fixture_version(tree_cls.SCHEMA_VERSION)
        for fixture in fixtures:
            if fixture.name != name or fixture.retired or not fixture.version.endswith(".dev"):
                continue
            if fixture.version.removesuffix(".dev") != live_version:
                continue
            if fixture.is_canonical_twin:
                # Its as_shipped sibling below produces a clean error for the
                # pair.  An orphan twin is left for `check_schema_fixtures` to
                # diagnose without mutating it.
                continue
            if fixture.is_as_shipped:
                raise SchemaFixtureError(
                    f"{fixture.path.name} preserves development-version shipped bytes and cannot "
                    "be frozen without changing them; replace it with bytes from a final-version "
                    "shipped artifact."
                )
            target = fixture.path.with_name(_fixture_filename(name, live_version, fixture.variant))
            if target.exists():
                raise SchemaFixtureError(
                    f"{target.name} already exists; refusing to overwrite it while freezing "
                    f"{fixture.path.name}."
                )
            pending.append((target, fixture.path, canonical_fixture_text(read_fixture_tree(fixture))))

    # Write all destinations before removing any sources.  Preflight above
    # handles expected failures; this ordering also avoids data loss if an
    # unexpected filesystem error occurs during a write.
    for target, _, text in pending:
        target.write_text(text)
    for _, source, _ in pending:
        source.unlink()
    return [(target, source) for target, source, _ in pending]


def _values_equal(old: object, current: object) -> bool:
    """Return whether two leaf values are equal, treating NaN as equal.

    Types must match as well as values, so an integer in a field the current
    model writes as a float is a disagreement rather than a coincidence.
    """
    if isinstance(old, float) and isinstance(current, float):
        if math.isnan(old) and math.isnan(current):
            return True
    return type(old) is type(current) and old == current


def _paths_and_values(value: object, path: str = "") -> dict[str, object]:
    """Map every path a raw JSON-like value expresses to its value there.

    A path is recorded for every dict key and list index encountered, not
    only for leaves, so a query can find that a key exists (and inspect its
    value) even when the read side gives it a different shape; see
    `_is_expressed`.  The root is excluded, matching the path syntax
    `compare_fixture_versions` uses for problem messages.
    """
    values: dict[str, object] = {path: value} if path else {}
    if isinstance(value, dict):
        for key, sub in value.items():
            values.update(_paths_and_values(sub, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            values.update(_paths_and_values(item, f"{path}[{index}]"))
    return values


def _container_kind(value: object) -> str | None:
    """Return ``"dict"``, ``"list"``, or `None` for a non-container value."""
    if isinstance(value, dict):
        return "dict"
    if isinstance(value, list):
        return "list"
    return None


def _parent_path(path: str) -> str:
    """Return the path one level up from ``path``, or ``""`` at the root."""
    return path[: path.rindex("[")] if path.endswith("]") else path.rpartition(".")[0]


def _is_expressed(path: str, on_disk_values: dict[str, object], dump_values: dict[str, object]) -> bool:
    """Return whether the old file expresses ``path``.

    An exact path match on disk always counts.  Otherwise this finds the
    nearest proper ancestor (excluding the root) present on disk, and counts
    ``path`` as expressed only if that ancestor holds different kinds of
    container on disk and in the old dump -- one a list, the other a dict.

    That recovery is deliberately narrow, covering only a field whose spelling
    changed because its container was reshaped, so that no exact path below it
    can match (e.g. a pair collapsed from a list into named components).
    Dropping the kind check would extend it to any path under any container
    found on disk, which would wrongly compare a later-born field nested under
    an unreshaped container -- the most likely way these schemas evolve.
    """
    if path in on_disk_values:
        return True
    ancestor = _parent_path(path)
    while ancestor:
        if ancestor in on_disk_values:
            return _container_kind(on_disk_values[ancestor]) != _container_kind(dump_values.get(ancestor))
        ancestor = _parent_path(ancestor)
    return False


def _compare_expressed(
    old: object,
    current: object,
    on_disk_values: dict[str, object],
    dump_values: dict[str, object],
    path: str,
) -> list[str]:
    """Recursive worker for `compare_fixture_versions`.

    Identical to the public function's comparison logic, except that it
    takes the already-computed on-disk and old-dump value maps instead of
    deriving them, so each recursive call need not recompute them from an
    ever-shrinking ``old`` subtree.
    """
    problems: list[str] = []
    if isinstance(old, dict) and isinstance(current, dict):
        for key in sorted(old):
            child = f"{path}.{key}"
            if not _is_expressed(child, on_disk_values, dump_values):
                continue
            if key not in current:
                problems.append(f"{child}: in the older fixture but not the current one")
            else:
                problems.extend(
                    _compare_expressed(old[key], current[key], on_disk_values, dump_values, child)
                )
    elif isinstance(old, list) and isinstance(current, list):
        if len(old) != len(current):
            problems.append(f"{path}: list length {len(old)} != {len(current)}")
        else:
            for index, (o, c) in enumerate(zip(old, current, strict=True)):
                problems.extend(_compare_expressed(o, c, on_disk_values, dump_values, f"{path}[{index}]"))
    elif not _values_equal(old, current):
        problems.append(f"{path}: {old!r} != {current!r}")
    return problems


_NOT_GIVEN = object()
"""Sentinel marking an omitted ``on_disk`` argument, distinct from `None`."""


def compare_fixture_versions(
    old: object, current: object, *, path: str = "", on_disk: object = _NOT_GIVEN
) -> list[str]:
    """Compare an older fixture's read against the current-version fixture.

    Parameters
    ----------
    old
        Canonical dump of the older fixture, read under current code.
    current
        Canonical dump of the current-version fixture.
    path
        Path prefix used in problem messages; callers leave this empty.
    on_disk
        The older fixture's raw content as stored on disk, before model
        validation.  Comparing a real fixture means passing the parsed file
        content here: this decides which paths the older file expressed, and
        reading it through a model materializes every field at its default,
        which would make later-born fields look present.  The default of
        ``old`` itself is correct only for a caller whose ``old`` never went
        through validation, such as a test comparing literals.

    Returns
    -------
    `list` [ `str` ]
        One problem description per disagreement; empty when the older
        fixture projects cleanly onto the current one.

    Notes
    -----
    Both fixtures encode the same logical exemplar, which is allowed to grow
    as versions add fields.  Whether a path counts as one the older file
    expresses is decided from ``on_disk``, not from ``old``: reading a
    fixture through a pydantic model materializes every field, including
    ones the file never mentioned, at its declared default, so deciding
    "later-born" from ``old`` itself would treat every additive field as
    already present whenever its default happened to match, and would then
    reject any real, meaningful value chosen for it in the current exemplar.

    A path present in both ``old`` and ``current`` and expressed by
    ``on_disk`` must agree; a path in ``current`` that is not expressed is a
    later-born field and is ignored; an expressed path missing from
    ``current`` is a failure, because the older file said something the
    current exemplar does not.

    Whether an unmatched path is expressed is narrow by design; see
    `_is_expressed`.  It recovers only a reshaped container whose exact path
    changed on disk, not any later-born field nested under an existing
    container, so a migration that renames or restructures a field beyond
    that one recovery is registered as expected divergence by its caller,
    and its migration test asserts the morphed result directly instead.
    """
    reference = old if on_disk is _NOT_GIVEN else on_disk
    on_disk_values = _paths_and_values(reference)
    dump_values = _paths_and_values(old)
    return _compare_expressed(old, current, on_disk_values, dump_values, path)

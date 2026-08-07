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
"""What the committed schema fixtures do and do not exercise.

Coverage is attributed per schema, not per fixture file.  Every
`~lsst.images.serialization.ArchiveTree` serializes its own
``schema_version`` and ``schema_url``, root or embedded, so fixture JSON is
self-describing at every depth: a walk can switch attribution at each nested
stamp and credit what it finds to the schema that owns it.  A sub-model
reached from several containers is therefore credited by all of them, and no
report can claim a container "fails to cover" a schema that another container
exercises.

Two things are reported.  Property coverage is the set of paths a schema
declares that no fixture expresses.  Sub-schema positions are the places a
schema can hold another stamped schema, with the candidates each position
admits and the ones fixtures actually put there; a composite model is under
no obligation to reach every candidate, but knowing which it misses is what
tells you whether a fixture set matches what you assumed it covered.

See :ref:`lsst.images-schema-fixtures` for the fixture tree this reads.
"""

from __future__ import annotations

__all__ = (
    "CoverageReport",
    "SchemaCoverage",
    "SubSchemaPosition",
    "format_coverage_report",
    "schema_coverage",
)

import dataclasses
import json
from collections import defaultdict
from pathlib import Path

from ..serialization import ArchiveTree, available_schema_classes, dump_schema
from ._schema_fixtures import fixture_version, iter_schema_fixtures

_SCHEMA_URL_KEY = "x-lsst-schema-url"
"""Key `~lsst.images.serialization.dump_schema` writes the canonical URL of a
nested published schema under.

Draft 2020-12 makes ``$id`` start a new resolution scope, which would break
the root-relative references pydantic generates, so nested definitions carry
their identity under this non-reserved key instead.
"""


def _schema_name_from_url(url: str) -> str:
    """Return the schema name encoded in a canonical schema URL.

    Parameters
    ----------
    url
        Canonical schema URL, as ``{base}/{name}-{version}``.
    """
    return url.rsplit("/", 1)[-1].rsplit("-", 1)[0]


def _stamp_key(value: object) -> tuple[str, str] | None:
    """Return the ``(name, version)`` a JSON value stamps itself with.

    Parameters
    ----------
    value
        Raw JSON-like value to inspect.

    Returns
    -------
    `tuple` [ `str`, `str` ] or `None`
        The schema name and fixture-filename version, or `None` if the value
        is not a stamped tree.  A malformed stamp yields `None` rather than
        raising; `~lsst.images.tests.check_schema_fixtures` is what reports
        such a fixture as broken.
    """
    if not isinstance(value, dict):
        return None
    url = value.get("schema_url")
    version = value.get("schema_version")
    if not isinstance(url, str) or not isinstance(version, str):
        return None
    try:
        return _schema_name_from_url(url), fixture_version(version)
    except Exception:
        return None


def _credit(
    value: object,
    key: tuple[str, str],
    path: str,
    expressed: dict[tuple[str, str], set[str]],
    embeddings: dict[tuple[str, str], dict[str, set[str]]],
) -> None:
    """Credit the paths a raw fixture value expresses to their owning schema.

    Parameters
    ----------
    value
        Raw JSON-like value to walk.
    key
        Schema name and version of the nearest enclosing stamped tree.
    path
        Path of ``value`` within that tree, empty at the tree's own root.
    expressed
        Accumulator mapping each schema to the paths it expresses.
    embeddings
        Accumulator mapping each schema to the sub-schema names found at each
        of its paths.

    Notes
    -----
    List indices collapse to ``[]``, because an index is data rather than a
    position in the schema.  Attribution switches at every nested stamp, and
    the nested path is recorded in the container as well: the container did
    put something there, even though the contents belong to the sub-schema.
    """
    if path:
        expressed[key].add(path)
    if isinstance(value, dict):
        for name, sub in value.items():
            child_path = f"{path}.{name}"
            if (child_key := _stamp_key(sub)) is not None:
                expressed[key].add(child_path)
                embeddings[key].setdefault(child_path, set()).add(child_key[0])
                _credit(sub, child_key, "", expressed, embeddings)
            else:
                _credit(sub, key, child_path, expressed, embeddings)
    elif isinstance(value, list):
        for item in value:
            item_path = f"{path}[]"
            if (item_key := _stamp_key(item)) is not None:
                expressed[key].add(item_path)
                embeddings[key].setdefault(item_path, set()).add(item_key[0])
                _credit(item, item_key, "", expressed, embeddings)
            else:
                _credit(item, key, item_path, expressed, embeddings)


def _resolve(node: object, defs: dict[str, object], seen: frozenset[str]) -> tuple[object, frozenset[str]]:
    """Follow a ``$ref`` chain to the definition it names.

    Parameters
    ----------
    node
        Schema node, which may be a ``$ref``.
    defs
        The document's ``$defs`` mapping.
    seen
        Definition names already followed on this branch, which stops a
        recursive model (a sum of fields that may themselves be sums) from
        looping.

    Returns
    -------
    resolved : `object`
        The resolved node, or `None` if the chain revisits a definition.
    seen : `frozenset` [ `str` ]
        ``seen`` extended with the names followed here.
    """
    while isinstance(node, dict) and "$ref" in node:
        name = str(node["$ref"]).rsplit("/", 1)[-1]
        if name in seen:
            return None, seen
        seen = seen | {name}
        node = defs.get(name, {})
    return node, seen


def _candidates(node: object, defs: dict[str, object], seen: frozenset[str]) -> frozenset[str]:
    """Return the stamped schema names a schema node may hold.

    Parameters
    ----------
    node
        Schema node describing a position.
    defs
        The document's ``$defs`` mapping.
    seen
        Definition names already followed on this branch.

    Notes
    -----
    A node carrying the nested-schema URL key is itself a candidate.
    Otherwise its ``anyOf`` and ``oneOf`` branches are unioned, which is what
    turns a plain union of PSF models, or a discriminated union of field
    models, into the set of schemas that position admits.  A branch that is
    ``null`` or unconstrained contributes nothing.
    """
    resolved, seen = _resolve(node, defs, seen)
    if not isinstance(resolved, dict):
        return frozenset()
    if isinstance(url := resolved.get(_SCHEMA_URL_KEY), str):
        return frozenset({_schema_name_from_url(url)})
    found: set[str] = set()
    for branch in (*resolved.get("anyOf", ()), *resolved.get("oneOf", ())):
        found |= _candidates(branch, defs, seen)
    return frozenset(found)


def _declare(
    node: object,
    defs: dict[str, object],
    prefix: str,
    seen: frozenset[str],
    paths: set[str],
    positions: dict[str, frozenset[str]],
    mappings: set[str],
) -> None:
    """Collect the paths and sub-schema positions a schema node declares.

    Parameters
    ----------
    node
        Schema node to walk.
    defs
        The document's ``$defs`` mapping.
    prefix
        Path of ``node`` within the document, empty at the root.
    seen
        Definition names already followed on this branch.
    paths
        Accumulator of declared paths.
    positions
        Accumulator mapping a path to the sub-schema names it admits.
    mappings
        Accumulator of paths whose children are data-keyed rather than
        declared, so a credited path below one can be truncated back to it.

    Notes
    -----
    A position that admits a stamped sub-schema is a boundary: it is recorded
    and not descended into, because everything below it belongs to that
    sub-schema and is credited there.  This is the same boundary `_credit`
    switches attribution at, which is what makes the two sides comparable.
    """
    resolved, seen = _resolve(node, defs, seen)
    if not isinstance(resolved, dict):
        return
    for branch in (*resolved.get("anyOf", ()), *resolved.get("oneOf", ())):
        _declare(branch, defs, prefix, seen, paths, positions, mappings)
    for name, sub in resolved.get("properties", {}).items():
        path = f"{prefix}.{name}"
        paths.add(path)
        if candidates := _candidates(sub, defs, seen):
            positions[path] = candidates
            continue
        _declare(sub, defs, path, seen, paths, positions, mappings)
    for keyword in ("items", "contains"):
        if (items := resolved.get(keyword)) is not None:
            path = f"{prefix}[]"
            if candidates := _candidates(items, defs, seen):
                positions[path] = candidates
            else:
                _declare(items, defs, path, seen, paths, positions, mappings)
    for item in resolved.get("prefixItems", ()):
        _declare(item, defs, f"{prefix}[]", seen, paths, positions, mappings)
    if isinstance(values := resolved.get("additionalProperties"), dict):
        mappings.add(prefix)
        if candidates := _candidates(values, defs, seen):
            positions[prefix] = candidates


def _truncate(path: str, mappings: frozenset[str]) -> str:
    """Return a credited path cut back to its nearest data-keyed ancestor.

    Parameters
    ----------
    path
        Credited path, which may descend through a mapping's data keys.
    mappings
        Paths whose children are data keys rather than declared properties.

    Notes
    -----
    A mapping's keys are values chosen by whoever wrote the data, so they are
    not schema positions and cannot be compared against declared paths.  The
    longest matching ancestor wins, so a mapping nested inside another is cut
    at the inner one.
    """
    best = ""
    for mapping in mappings:
        if path.startswith(f"{mapping}.") and len(mapping) > len(best):
            best = mapping
    return best or path


@dataclasses.dataclass(frozen=True)
class SubSchemaPosition:
    """One place a schema can hold another stamped schema."""

    path: str
    """Path of the position within its containing schema."""

    candidates: frozenset[str]
    """Schema names the position admits."""

    reached: frozenset[str]
    """Schema names some committed fixture actually put there."""

    @property
    def missing(self) -> frozenset[str]:
        """Candidates no fixture ever put at this position."""
        return self.candidates - self.reached


@dataclasses.dataclass(frozen=True)
class SchemaCoverage:
    """What the fixture tree exercises of one schema version."""

    name: str
    """Schema name."""

    version: str
    """Fixture-filename version, e.g. ``1.0.0`` or ``1.0.0.dev``."""

    expressed: frozenset[str]
    """Declared paths that at least one fixture expresses."""

    absent: frozenset[str]
    """Declared paths no fixture expresses."""

    positions: tuple[SubSchemaPosition, ...]
    """Sub-schema positions, ordered by path."""

    sources: frozenset[str]
    """Names of the fixture files that credit this schema, whether as their
    own top-level tree or by embedding it."""

    @property
    def absent_roots(self) -> frozenset[str]:
        """The shallowest absent path of each absent subtree.

        A field left unset takes its whole declared subtree with it, so
        ``butler_info`` being `None` everywhere would otherwise report every
        path beneath it as a separate gap.  Only the outermost is actionable:
        populate that field and its children become reachable.
        """
        return frozenset(
            path
            for path in self.absent
            if not any(path.startswith(f"{other}.") or path.startswith(f"{other}[") for other in self.absent)
        )


@dataclasses.dataclass(frozen=True)
class CoverageReport:
    """Coverage of every schema in scope, keyed by name and version."""

    schemas: dict[tuple[str, str], SchemaCoverage]
    """Coverage per ``(name, version)``."""

    @property
    def positions_missing_candidates(self) -> tuple[tuple[str, SubSchemaPosition], ...]:
        """Positions with an unreached candidate, as ``(schema, position)``."""
        return tuple(
            (f"{cov.name} {cov.version}", position)
            for cov in self.schemas.values()
            for position in cov.positions
            if position.missing
        )


def _coverage_for(
    tree_cls: type[ArchiveTree],
    expressed: dict[tuple[str, str], set[str]],
    embeddings: dict[tuple[str, str], dict[str, set[str]]],
    sources: dict[tuple[str, str], set[str]],
) -> SchemaCoverage:
    """Build one schema's coverage from the credited walk results.

    Parameters
    ----------
    tree_cls
        Serialization model class to report on.
    expressed
        Paths credited to each schema.
    embeddings
        Sub-schema names credited at each path of each schema.
    sources
        Fixture filenames crediting each schema.
    """
    key = (tree_cls.SCHEMA_NAME, fixture_version(tree_cls.SCHEMA_VERSION))
    document = dump_schema(tree_cls)
    defs = document.get("$defs", {})
    paths: set[str] = set()
    positions: dict[str, frozenset[str]] = {}
    mappings: set[str] = set()
    _declare(document, defs, "", frozenset(), paths, positions, mappings)
    frozen_mappings = frozenset(mappings)
    seen = {_truncate(path, frozen_mappings) for path in expressed.get(key, ())}
    reached: dict[str, set[str]] = defaultdict(set)
    for path, names in embeddings.get(key, {}).items():
        reached[_truncate(path, frozen_mappings)] |= names
    declared = paths | set(positions)
    return SchemaCoverage(
        name=key[0],
        version=key[1],
        expressed=frozenset(declared & seen),
        absent=frozenset(declared - seen),
        positions=tuple(
            SubSchemaPosition(
                path=path,
                candidates=candidates,
                reached=frozenset(reached.get(path, ())),
            )
            for path, candidates in sorted(positions.items())
        ),
        sources=frozenset(sources.get(key, ())),
    )


def schema_coverage(directory: Path, *, package: str = "lsst.images") -> CoverageReport:
    """Report what the committed fixtures exercise of each schema.

    Parameters
    ----------
    directory
        Directory holding the fixture tree.
    package
        Package whose schemas to report on; see
        `~lsst.images.serialization.available_schema_classes`.

    Returns
    -------
    `CoverageReport`
        Coverage keyed by schema name and fixture-filename version.

    Notes
    -----
    Every non-retired fixture in the tree contributes, whatever schema it is
    a fixture *of*: coverage is credited to the schema that owns each stamped
    subtree, so a sub-model embedded by several containers is credited by all
    of them.  Retired fixtures are excluded because they do not validate.

    This reports reach, not judgement.  A path counted as expressed only
    proves some fixture put a value there, not that the value is interesting;
    what pins payload data is described in
    :ref:`lsst.images-schema-fixtures`.
    """
    expressed: dict[tuple[str, str], set[str]] = defaultdict(set)
    embeddings: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(dict)
    sources: dict[tuple[str, str], set[str]] = defaultdict(set)
    for fixture in iter_schema_fixtures(directory):
        if fixture.retired:
            continue
        try:
            content = json.loads(fixture.path.read_text())
        except (OSError, UnicodeError, json.JSONDecodeError):
            # A fixture that will not parse is check_schema_fixtures's problem
            # to report; skipping it here keeps this function a pure reporter.
            continue
        if (key := _stamp_key(content)) is None:
            continue
        # Walk into per-fixture accumulators first, so which schemas this one
        # file credits is known exactly, then merge.  A schema embedded by a
        # container is credited by it even when it owns no fixture itself.
        seen_paths: dict[tuple[str, str], set[str]] = defaultdict(set)
        seen_embeddings: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(dict)
        _credit(content, key, "", seen_paths, seen_embeddings)
        for credited, paths in seen_paths.items():
            expressed[credited] |= paths
        for credited, found in seen_embeddings.items():
            for path, names in found.items():
                embeddings[credited].setdefault(path, set()).update(names)
        for credited in set(seen_paths) | set(seen_embeddings):
            sources[credited].add(fixture.path.name)
    return CoverageReport(
        schemas={
            (tree_cls.SCHEMA_NAME, fixture_version(tree_cls.SCHEMA_VERSION)): _coverage_for(
                tree_cls, expressed, embeddings, sources
            )
            for tree_cls in available_schema_classes(package)
        }
    )


def format_coverage_report(report: CoverageReport, *, schema: str | None = None) -> str:
    """Render a coverage report as text.

    Parameters
    ----------
    report
        The report to render.
    schema
        Restrict the output to this schema name, or `None` for all of them.

    Returns
    -------
    `str`
        The rendered report, one block per schema.

    Notes
    -----
    Absent paths are collapsed to the root of each absent subtree, and every
    sub-schema position is listed whether or not it has a gap: a position that
    reaches all its candidates is as much of an answer to "what does this
    fixture set cover?" as one that does not.  A gap is marked ``gap`` rather
    than reported as an error, because a composite model is under no
    obligation to hold every candidate its schema admits.
    """
    lines: list[str] = []
    for key in sorted(report.schemas):
        coverage = report.schemas[key]
        if schema is not None and coverage.name != schema:
            continue
        declared = len(coverage.expressed) + len(coverage.absent)
        lines.append(
            f"{coverage.name} {coverage.version}: "
            f"{len(coverage.expressed)}/{declared} paths, {len(coverage.sources)} fixture(s)"
        )
        if not coverage.sources:
            lines.append("    no fixture reaches this schema")
        for path in sorted(coverage.absent_roots):
            lines.append(f"    absent  {path}")
        for position in coverage.positions:
            marker = "gap    " if position.missing else "holds  "
            reached = ", ".join(sorted(position.reached)) or "nothing"
            if position.candidates == position.reached:
                lines.append(f"    {marker}{position.path} [{reached}]")
            else:
                lines.append(
                    f"    {marker}{position.path} [{reached}] of {{{', '.join(sorted(position.candidates))}}}"
                )
    return "\n".join(lines)

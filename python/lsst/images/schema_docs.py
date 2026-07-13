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
"""Generate Sphinx pages for the frozen JSON schemas.

Called from ``doc/conf.py`` at the start of every documentation build.  Each
frozen ``schemas/{name}/{name}-{version}.json`` file becomes a page at
``doc/schemas/{name}-{version}/index.rst`` so that the canonical schema URL
``https://images.lsst.io/schemas/{name}-{version}`` resolves (LSST the Docs
redirects extensionless directory paths to their ``index.html``), and the raw
JSON is staged for publication alongside it via ``html_extra_path``.
"""

from __future__ import annotations

__all__ = ("generate_schema_docs",)

import json
import re
import shutil
from pathlib import Path
from typing import Any

from .diagram import build_graph, make_policy, render
from .serialization import (
    SCHEMA_URL_BASE,
    ArrayReferenceModel,
    class_for_schema,
    parameterize_tree,
)


def _link_map(schema: dict[str, Any], available_stems: set[str]) -> dict[str, str]:
    """Map ``$defs`` keys to rst cross-references for definitions that are
    themselves published schemas.

    Nested `~lsst.images.serialization.ArchiveTree` models carry their
    canonical URL in the ``x-lsst-schema-url`` key of their ``$defs`` entry
    (``$id`` would start a new resolution scope), so a composite schema's
    page can hyperlink directly to its sub-schemas (e.g. ``visit_image`` →
    ``mask``): a relative page reference when the sub-schema is hosted on
    this site, or its absolute URL when it is published elsewhere.
    """
    links: dict[str, str] = {}
    for key, definition in schema.get("$defs", {}).items():
        if not isinstance(definition, dict):
            continue
        schema_id = definition.get("x-lsst-schema-url", "")
        if not schema_id:
            continue
        stem = schema_id.rstrip("/").rsplit("/", 1)[-1]
        title = definition.get("title", stem)
        if stem in available_stems:
            links[key] = f":doc:`{title} <../{stem}/index>`"
        else:
            links[key] = f"`{title} <{schema_id}>`__"
    return links


def _type_summary(prop: dict[str, Any], links: dict[str, str]) -> str:
    """Return a short human-readable type description for a property,
    hyperlinking types that have their own schema page.
    """
    if "$ref" in prop:
        key = prop["$ref"].rsplit("/", 1)[-1]
        return links.get(key, key)
    for key in ("anyOf", "oneOf", "allOf"):
        if key in prop:
            return " | ".join(_type_summary(sub, links) for sub in prop[key])
    if prop.get("type") == "array":
        items = prop.get("items")
        if isinstance(items, dict):
            return f"array of {_type_summary(items, links)}"
        return "array"
    if "type" in prop:
        return str(prop["type"])
    if "const" in prop:
        return repr(prop["const"])
    return "any"


_DEFAULT_ROLE_RE = re.compile(r"(?<!`)`([^`]+)`(?!`)")
"""Single-backtick (default-role) references in docstring-derived text."""


def _reference_to_literal(match: re.Match[str]) -> str:
    """Rewrite a default-role Python reference as an inline literal.

    Model docstrings use `X`, `~pkg.mod.X`, and `.X` references that cannot
    resolve as py:obj targets on the generated schema pages; render them as
    code, applying the ``~`` convention of showing only the last component.
    """
    target = match.group(1)
    if target.startswith("~"):
        target = target[1:].rsplit(".", 1)[-1]
    return f"``{target.lstrip('.')}``"


def _one_line(text: str) -> str:
    """Collapse ``text`` to a single line safe for a list-table cell."""
    return " ".join(text.split())


def _description_text(text: str) -> str:
    """Return docstring-derived description text as a single rst line with
    Python references neutralized.
    """
    return _DEFAULT_ROLE_RE.sub(_reference_to_literal, _one_line(text))


def _root_body(schema: dict[str, Any]) -> dict[str, Any]:
    """Return the mapping holding the schema's description and properties.

    A recursive model's document root is just a ``$ref`` into ``$defs``, so
    the referenced definition holds the interesting content.
    """
    if "$ref" in schema and "properties" not in schema:
        key = schema["$ref"].rsplit("/", 1)[-1]
        target = schema.get("$defs", {}).get(key)
        if isinstance(target, dict):
            return target
    return schema


def _field_table(body: dict[str, Any], links: dict[str, str]) -> list[str]:
    """Return rst list-table lines for the schema's top-level properties."""
    properties: dict[str, Any] = body.get("properties", {})
    if not properties:
        return ["This schema has no top-level properties."]
    required = set(body.get("required", []))
    lines = [
        ".. list-table::",
        "   :header-rows: 1",
        "   :widths: 20 25 10 45",
        "",
        "   * - Field",
        "     - Type",
        "     - Required",
        "     - Description",
    ]
    for name, prop in properties.items():
        lines.append(f"   * - ``{name}``")
        lines.append(f"     - {_one_line(_type_summary(prop, links))}")
        lines.append(f"     - {'yes' if name in required else 'no'}")
        lines.append(f"     - {_description_text(prop.get('description', ''))}")
    return lines


def _mermaid_lines(name: str, version: str) -> list[str]:
    """Return a mermaid composition-diagram block for the schema, or a
    superseded-version note when the live model no longer matches.
    """
    cls = class_for_schema(name)
    if cls is None or cls.SCHEMA_VERSION != version:
        note = "This schema version has been superseded."
        if cls is not None:
            note = f"This schema version has been superseded; the current version is {cls.SCHEMA_VERSION}."
        return [
            ".. note::",
            "",
            f"   {note}",
        ]
    graph = build_graph(parameterize_tree(cls, ArrayReferenceModel), policy=make_policy())
    lines = ["Composition", "===========", "", ".. mermaid::", ""]
    lines.extend(f"   {line}" for line in render(graph, "mermaid").splitlines())
    return lines


def _version_key(version: str) -> tuple[int, ...]:
    """Return a numeric sort key for a ``major.minor.patch`` version string,
    so that e.g. ``1.0.10`` sorts after ``1.0.2``.
    """
    return tuple(int(part) for part in version.split("."))


def _family_page(
    name: str, versions: list[str], latest_schema: dict[str, Any], available_stems: set[str]
) -> str:
    """Return the rst source for a schema family page.

    The family page lists every published version of one schema (newest
    first, with the current version marked), renders the latest version's
    content inline, and owns the nav toctree for the version pages, so the
    top-level schema index only grows when a new schema is added, not on
    every version bump.  It is published at the versionless URL
    ``{SCHEMA_URL_BASE}/{name}``.

    Parameters
    ----------
    name
        Schema name.
    versions
        Published versions of the schema, newest first.
    latest_schema
        Parsed frozen schema of the newest version, rendered inline.
    available_stems
        ``{name}-{version}`` stems of every published schema, for
        cross-linking sub-schemas that have their own pages.
    """
    cls = class_for_schema(name)
    current = cls.SCHEMA_VERSION if cls is not None else None
    lines = [
        "#" * len(name),
        name,
        "#" * len(name),
        "",
        _description_text(_root_body(latest_schema).get("description", "")),
        "",
        "Versions",
        "========",
        "",
        ".. list-table::",
        "   :header-rows: 1",
        "",
        "   * - Version",
        "     - Status",
    ]
    for version in versions:
        lines.append(f"   * - :doc:`{version} <../{name}-{version}/index>`")
        lines.append(f"     - {'current' if version == current else 'superseded'}")
    # Render the latest version's content inline so the landing page shows the
    # schema itself, not just a list of links to click through.
    lines.append("")
    lines.extend(_content_lines(name, versions[0], latest_schema, available_stems))
    lines.extend(["", ".. toctree::", "   :hidden:", ""])
    lines.extend(f"   {version} <../{name}-{version}/index>" for version in versions)
    lines.append("")
    return "\n".join(lines)


def _content_lines(name: str, version: str, schema: dict[str, Any], available_stems: set[str]) -> list[str]:
    """Return the rst body shared by the version page and the family page:
    canonical URL, raw-JSON link, composition diagram, and field table.
    """
    links = _link_map(schema, available_stems)
    body = _root_body(schema)
    canonical_url = schema.get("$id", f"{SCHEMA_URL_BASE}/{name}-{version}")
    lines = [
        f"- Canonical URL: ``{canonical_url}``",
        f"- `Raw JSON schema <../{name}-{version}.json>`__",
        "",
    ]
    lines.extend(_mermaid_lines(name, version))
    lines.extend(["", "Fields", "======", ""])
    lines.extend(_field_table(body, links))
    return lines


def _schema_page(name: str, version: str, schema: dict[str, Any], available_stems: set[str]) -> str:
    """Return the rst source for one schema version page."""
    title = f"{name} {version}"
    lines = [
        "#" * len(title),
        title,
        "#" * len(title),
        "",
        _description_text(_root_body(schema).get("description", "")),
        "",
    ]
    lines.extend(_content_lines(name, version, schema, available_stems))
    lines.append("")
    return "\n".join(lines)


def generate_schema_docs(schema_dir: Path, page_dir: Path, extra_dir: Path) -> None:
    """Generate one Sphinx page per frozen schema file.

    Parameters
    ----------
    schema_dir
        Directory of frozen ``{name}/{name}-{version}.json`` files.
    page_dir
        Output directory for generated rst; wiped and fully regenerated.
    extra_dir
        Staging directory for ``html_extra_path``; receives a ``schemas/``
        subdirectory with copies of the raw JSON files.
    """
    if page_dir.exists():
        shutil.rmtree(page_dir)
    page_dir.mkdir(parents=True)
    extra_schemas = extra_dir / "schemas"
    if extra_schemas.exists():
        shutil.rmtree(extra_schemas)
    extra_schemas.mkdir(parents=True)
    paths = sorted(schema_dir.glob("*/*.json"))
    available_stems = {path.stem for path in paths}
    families: dict[str, list[str]] = {}
    schemas_by_stem: dict[str, dict[str, Any]] = {}
    for path in paths:
        name, _, version = path.stem.rpartition("-")
        schema = json.loads(path.read_text())
        entry_dir = page_dir / path.stem
        entry_dir.mkdir()
        (entry_dir / "index.rst").write_text(_schema_page(name, version, schema, available_stems))
        shutil.copyfile(path, extra_schemas / path.name)
        families.setdefault(name, []).append(version)
        schemas_by_stem[path.stem] = schema
    for name, versions in families.items():
        versions.sort(key=_version_key, reverse=True)
        family_dir = page_dir / name
        family_dir.mkdir()
        (family_dir / "index.rst").write_text(
            _family_page(name, versions, schemas_by_stem[f"{name}-{versions[0]}"], available_stems)
        )
    index_lines = [
        "#######",
        "Schemas",
        "#######",
        "",
        "JSON schemas for the ``lsst.images`` serialization data models.",
        "Each schema's page lists its published versions; every version page links the raw JSON schema.",
        "See :ref:`lsst.images-schema-versioning` for the versioning rules.",
        "",
        ".. toctree::",
        "   :maxdepth: 1",
        "",
        *(f"   {name}/index" for name in sorted(families)),
        "",
    ]
    (page_dir / "index.rst").write_text("\n".join(index_lines))

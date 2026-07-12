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
import shutil
from pathlib import Path
from typing import Any

from .diagram import build_graph, make_policy, render
from .serialization._asdf_utils import ArrayReferenceModel
from .serialization._common import SCHEMA_URL_BASE
from .serialization._io import class_for_schema, parameterize_tree


def _link_map(schema: dict[str, Any], available_stems: set[str]) -> dict[str, str]:
    """Map ``$defs`` keys to rst cross-references for definitions that are
    themselves published schemas with a generated page.

    Nested `~lsst.images.serialization.ArchiveTree` models carry their
    canonical URL in the ``x-lsst-schema-url`` key of their ``$defs`` entry
    (``$id`` would start a new resolution scope), so a composite schema's
    page can hyperlink directly to the pages of its sub-schemas (e.g.
    ``visit_image`` → ``mask``).
    """
    links: dict[str, str] = {}
    prefix = f"{SCHEMA_URL_BASE}/"
    for key, definition in schema.get("$defs", {}).items():
        if not isinstance(definition, dict):
            continue
        schema_id = definition.get("x-lsst-schema-url", "")
        if schema_id.startswith(prefix):
            stem = schema_id.removeprefix(prefix)
            if stem in available_stems:
                title = definition.get("title", stem)
                links[key] = f":doc:`{title} <../{stem}/index>`"
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


def _one_line(text: str) -> str:
    """Collapse ``text`` to a single line safe for a list-table cell."""
    return " ".join(text.split())


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
        lines.append(f"     - {_one_line(prop.get('description', ''))}")
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


def _schema_page(name: str, version: str, schema: dict[str, Any], available_stems: set[str]) -> str:
    """Return the rst source for one schema page."""
    links = _link_map(schema, available_stems)
    body = _root_body(schema)
    title = f"{name} {version}"
    lines = [
        "#" * len(title),
        title,
        "#" * len(title),
        "",
        _one_line(body.get("description", "")),
        "",
        f"- Canonical URL: ``{SCHEMA_URL_BASE}/{name}-{version}``",
        f"- `Raw JSON schema <../{name}-{version}.json>`__",
        "",
    ]
    lines.extend(_mermaid_lines(name, version))
    lines.extend(["", "Fields", "======", ""])
    lines.extend(_field_table(body, links))
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
    entries: list[str] = []
    for path in paths:
        name, _, version = path.stem.rpartition("-")
        schema = json.loads(path.read_text())
        entry_dir = page_dir / path.stem
        entry_dir.mkdir()
        (entry_dir / "index.rst").write_text(_schema_page(name, version, schema, available_stems))
        shutil.copyfile(path, extra_schemas / path.name)
        entries.append(f"   {path.stem}/index")
    index_lines = [
        "#######",
        "Schemas",
        "#######",
        "",
        "JSON schemas for the ``lsst.images`` serialization data models.",
        "Each page documents one schema version; the raw JSON schema is linked from each page.",
        "See :ref:`lsst.images-schema-versioning` for the versioning rules.",
        "",
        ".. toctree::",
        "   :maxdepth: 1",
        "",
        *entries,
        "",
    ]
    (page_dir / "index.rst").write_text("\n".join(index_lines))

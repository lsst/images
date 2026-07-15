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
import re
from pathlib import Path

import pytest

from lsst.images.schema_docs import generate_schema_docs
from lsst.images.serialization import write_frozen_schemas


@pytest.fixture(scope="module")
def generated(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """Generate pages from freshly-written frozen schemas."""
    root = tmp_path_factory.mktemp("schema_docs")
    schema_dir = root / "schemas"
    page_dir = root / "pages"
    extra_dir = root / "extra"
    write_frozen_schemas(schema_dir)
    generate_schema_docs(schema_dir, page_dir, extra_dir)
    return page_dir, extra_dir


def test_page_per_schema(generated: tuple[Path, Path]) -> None:
    """Verify each frozen schema gets a directory-style page, reachable from
    the top index through its family page rather than listed directly.
    """
    page_dir, _ = generated
    (image_dir,) = [p for p in page_dir.glob("image-*") if p.is_dir()]
    assert (image_dir / "index.rst").exists()
    index = (page_dir / "index.rst").read_text()
    assert "image/index" in index
    assert f"{image_dir.name}/index" not in index
    family = (page_dir / "image" / "index.rst").read_text()
    version = image_dir.name.removeprefix("image-")
    assert f":doc:`{version} <../{image_dir.name}/index>`" in family


def test_family_page_versions(tmp_path: Path) -> None:
    """Verify a family page lists versions newest-first with the current
    version marked, and hosts the nav toctree for its version pages.
    """
    schema_dir = tmp_path / "schemas"
    write_frozen_schemas(schema_dir)
    old = {
        "$id": "https://images.lsst.io/schemas/image-0.9.0",
        "title": "image",
        "description": "Old image schema.",
        "type": "object",
        "properties": {},
    }
    (schema_dir / "image" / "image-0.9.0.json").write_text(json.dumps(old) + "\n")
    page_dir = tmp_path / "pages"
    generate_schema_docs(schema_dir, page_dir, tmp_path / "extra")
    family = (page_dir / "image" / "index.rst").read_text()
    i_new = family.index(":doc:`1.0.0 <../image-1.0.0/index>`")
    i_old = family.index(":doc:`0.9.0 <../image-0.9.0/index>`")
    assert i_new < i_old
    assert "current" in family
    assert "superseded" in family
    assert ".. toctree::" in family


def test_family_version_sort_numeric(tmp_path: Path) -> None:
    """Verify versions sort numerically, so 1.0.10 outranks 1.0.2."""
    schema_dir = tmp_path / "schemas"
    schema_dir.mkdir()
    for version in ("1.0.2", "1.0.10"):
        gadget = {
            "$id": f"https://example.org/schemas/gadget-{version}",
            "title": "gadget",
            "description": "A gadget.",
            "type": "object",
            "properties": {},
        }
        (schema_dir / "gadget").mkdir(exist_ok=True)
        (schema_dir / "gadget" / f"gadget-{version}.json").write_text(json.dumps(gadget) + "\n")
    page_dir = tmp_path / "pages"
    generate_schema_docs(schema_dir, page_dir, tmp_path / "extra")
    family = (page_dir / "gadget" / "index.rst").read_text()
    i_ten = family.index(":doc:`1.0.10 <../gadget-1.0.10/index>`")
    i_two = family.index(":doc:`1.0.2 <../gadget-1.0.2/index>`")
    assert i_ten < i_two


def test_family_page_shows_latest_content(generated: tuple[Path, Path]) -> None:
    """Verify the family (versionless) page renders the latest version's
    composition diagram and field table inline, not just a list of versions.
    """
    page_dir, _ = generated
    family = (page_dir / "image" / "index.rst").read_text()
    assert ".. mermaid::" in family
    assert "\nFields\n======\n" in family
    # Both the versions table and the inline field table are list-tables.
    assert family.count(".. list-table::") >= 2


def test_page_content(generated: tuple[Path, Path]) -> None:
    """Verify the image page has URL, raw-JSON link, mermaid, and fields."""
    page_dir, _ = generated
    (image_dir,) = [p for p in page_dir.glob("image-*") if p.is_dir()]
    text = (image_dir / "index.rst").read_text()
    assert f"https://images.lsst.io/schemas/{image_dir.name}" in text
    assert f"<../{image_dir.name}.json>" in text
    assert ".. mermaid::" in text
    assert ".. list-table::" in text


def test_sub_schema_links(generated: tuple[Path, Path]) -> None:
    """Verify composite schema pages hyperlink their sub-schema pages.

    The visit_image field table must link the mask and sky_projection types
    to those schemas' own generated pages.
    """
    page_dir, _ = generated
    (vi_dir,) = [p for p in page_dir.glob("visit_image-*") if p.is_dir()]
    text = (vi_dir / "index.rst").read_text()
    assert ":doc:`mask <../mask-" in text
    assert ":doc:`sky_projection <../sky_projection-" in text


def test_external_sub_schema_links(tmp_path: Path) -> None:
    """Verify a sub-schema not hosted on this site is linked by its absolute
    $id URL, and the page's canonical URL comes from the schema's own $id,
    so an external package can generate pages under its own URL base.
    """
    schema_dir = tmp_path / "schemas"
    schema_dir.mkdir()
    widget = {
        "$id": "https://example.org/schemas/widget-1.0.0",
        "title": "widget",
        "description": "A widget.",
        "type": "object",
        "properties": {"mask": {"$ref": "#/$defs/MaskModel", "description": "The mask."}},
        "$defs": {
            "MaskModel": {
                "x-lsst-schema-url": "https://images.lsst.io/schemas/mask-1.0.0",
                "title": "mask",
                "type": "object",
            }
        },
    }
    (schema_dir / "widget").mkdir()
    (schema_dir / "widget" / "widget-1.0.0.json").write_text(json.dumps(widget) + "\n")
    page_dir = tmp_path / "pages"
    generate_schema_docs(schema_dir, page_dir, tmp_path / "extra")
    text = (page_dir / "widget-1.0.0" / "index.rst").read_text()
    assert "`mask <https://images.lsst.io/schemas/mask-1.0.0>`__" in text
    assert "https://example.org/schemas/widget-1.0.0" in text


def test_recursive_schema_page_has_fields(generated: tuple[Path, Path]) -> None:
    """Verify a recursive schema, whose document root is a $ref into $defs,
    still gets its description and field table from the referenced
    definition.
    """
    page_dir, _ = generated
    (pf_dir,) = [p for p in page_dir.glob("product_field-*") if p.is_dir()]
    text = (pf_dir / "index.rst").read_text()
    assert ".. list-table::" in text
    assert "``operands``" in text


def test_python_references_become_literals(generated: tuple[Path, Path]) -> None:
    """Verify single-backtick Python references from model docstrings are
    rendered as inline literals, since they cannot resolve as py:obj targets
    on the generated pages.
    """
    page_dir, _ = generated
    (cfs_dir,) = [p for p in page_dir.glob("camera_frame_set-*") if p.is_dir()]
    text = (cfs_dir / "index.rst").read_text()
    assert "``CameraFrameSet``" in text
    assert re.search(r"(?<!`)`CameraFrameSet`(?!`)", text) is None


def test_raw_json_staged(generated: tuple[Path, Path]) -> None:
    """Verify the raw JSON is staged for html_extra_path publication."""
    _, extra_dir = generated
    assert list((extra_dir / "schemas").glob("image-*.json"))


def test_superseded_version_notes_current(tmp_path: Path) -> None:
    """Verify a frozen file for an old version still gets a page, with a
    pointer at the current version and no diagram.
    """
    schema_dir = tmp_path / "schemas"
    (schema_dir / "image").mkdir(parents=True)
    (schema_dir / "image" / "image-0.0.1.json").write_text(
        '{"$id": "https://images.lsst.io/schemas/image-0.0.1", "title": "image",'
        ' "description": "Old.", "type": "object", "properties": {}}\n'
    )
    page_dir = tmp_path / "pages"
    generate_schema_docs(schema_dir, page_dir, tmp_path / "extra")
    text = (page_dir / "image-0.0.1" / "index.rst").read_text()
    assert "superseded" in text
    assert ".. mermaid::" not in text


def test_page_dir_is_regenerated(tmp_path: Path) -> None:
    """Verify stale generated pages are removed on regeneration."""
    schema_dir = tmp_path / "schemas"
    write_frozen_schemas(schema_dir)
    page_dir = tmp_path / "pages"
    generate_schema_docs(schema_dir, page_dir, tmp_path / "extra")
    stale = page_dir / "gone-1.0.0" / "index.rst"
    stale.parent.mkdir()
    stale.write_text("stale\n")
    generate_schema_docs(schema_dir, page_dir, tmp_path / "extra")
    assert not stale.exists()


def test_extra_dir_is_regenerated(tmp_path: Path) -> None:
    """Verify raw JSON staged for a removed frozen schema is not republished
    by later builds.
    """
    schema_dir = tmp_path / "schemas"
    write_frozen_schemas(schema_dir)
    extra_dir = tmp_path / "extra"
    generate_schema_docs(schema_dir, tmp_path / "pages", extra_dir)
    stale = extra_dir / "schemas" / "gone-1.0.0.json"
    stale.write_text("{}\n")
    generate_schema_docs(schema_dir, tmp_path / "pages", extra_dir)
    assert not stale.exists()

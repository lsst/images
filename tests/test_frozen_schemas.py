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

from lsst.images.frozen_schemas import (
    available_schema_classes,
    check_frozen_schemas,
    dump_schema,
    frozen_schema_filename,
    frozen_schema_path,
    write_frozen_schemas,
)


def test_available_schema_classes() -> None:
    """Verify enumeration finds the package's schemas, sorted by name."""
    classes = available_schema_classes()
    names = [cls.SCHEMA_NAME for cls in classes]
    assert names == sorted(names)
    assert "image" in names
    assert "cell_coadd" in names  # lazily-loaded built-in provider


def test_dump_schema_has_id_and_title() -> None:
    """Verify the dumped schema carries the canonical $id and title."""
    (cls,) = [c for c in available_schema_classes() if c.SCHEMA_NAME == "image"]
    schema = dump_schema(cls)
    assert schema["$id"] == f"https://images.lsst.io/schemas/image-{cls.SCHEMA_VERSION}"
    assert schema["title"] == "image"
    assert frozen_schema_filename(cls) == f"image-{cls.SCHEMA_VERSION}.json"


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

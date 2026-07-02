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

import pytest

SCHEMA_DIR = Path(__file__).parent / "data" / "schema_v1"

# Every committed top-level fixture; the ``legacy/`` subdirectory (derived
# from real files) is exercised separately by test_schema_v1_legacy_fixtures.
_FIXTURES = sorted(SCHEMA_DIR.glob("*.json"))


@pytest.mark.parametrize("path", _FIXTURES, ids=lambda p: p.stem)
def test_fixture_has_top_level_stamps(path: Path) -> None:
    """Verify every fixture has schema_url, schema_version,
    min_read_version.
    """
    tree = json.loads(path.read_text())
    assert "schema_url" in tree
    assert "schema_version" in tree
    assert "min_read_version" in tree


def test_fixtures_present() -> None:
    """Verify the fixture directory is populated."""
    assert _FIXTURES, f"no fixtures found in {SCHEMA_DIR}"


@pytest.mark.parametrize("path", _FIXTURES, ids=lambda p: p.stem)
def test_fixture_url_matches_name_and_version(path: Path) -> None:
    """Verify schema_url matches the ``<name>-<version>`` pattern.

    The name is the fixture file's stem.
    """
    tree = json.loads(path.read_text())
    expected = f"https://images.lsst.io/schemas/{path.stem}-{tree['schema_version']}"
    assert tree["schema_url"] == expected

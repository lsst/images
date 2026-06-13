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
import unittest
from pathlib import Path

SCHEMA_DIR = Path(__file__).parent / "data" / "schema_v1"

# Every committed top-level fixture; the ``legacy/`` subdirectory (derived
# from real files) is exercised separately by test_schema_v1_legacy_fixtures.
_FIXTURES = sorted(SCHEMA_DIR.glob("*.json"))


class SchemaV1FixturesTestCase(unittest.TestCase):
    """Tests over the bundled v1 reference JSON fixtures."""

    def test_fixtures_present(self) -> None:
        """The fixture directory is populated."""
        self.assertTrue(_FIXTURES, f"no fixtures found in {SCHEMA_DIR}")

    def test_fixture_has_top_level_stamps(self) -> None:
        """Every fixture has schema_url, schema_version, min_read_version."""
        for path in _FIXTURES:
            with self.subTest(name=path.stem):
                tree = json.loads(path.read_text())
                self.assertIn("schema_url", tree)
                self.assertIn("schema_version", tree)
                self.assertIn("min_read_version", tree)

    def test_fixture_url_matches_name_and_version(self) -> None:
        """schema_url matches the ``<name>-<version>`` pattern, where the name
        is the fixture file's stem.
        """
        for path in _FIXTURES:
            with self.subTest(name=path.stem):
                tree = json.loads(path.read_text())
                expected = f"https://images.lsst.io/schemas/{path.stem}-{tree['schema_version']}"
                self.assertEqual(tree["schema_url"], expected)


if __name__ == "__main__":
    unittest.main()

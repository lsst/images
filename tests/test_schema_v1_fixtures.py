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

from lsst.images.tests._make_schema_fixtures import BUILDERS, FIXTURE_DIR


def _supported_fixture_names() -> list[str]:
    """Return fixture names whose builders don't raise NotImplementedError."""
    names: list[str] = []
    for name, builder in BUILDERS.items():
        try:
            builder()
        except NotImplementedError:
            continue
        names.append(name)
    return names


_SUPPORTED = _supported_fixture_names()


class SchemaV1FixturesTestCase(unittest.TestCase):
    """Tests over the bundled v1 reference JSON fixtures."""

    def test_every_supported_builder_has_a_fixture(self):
        """Every supported builder produces a fixture file on disk."""
        for name in _SUPPORTED:
            with self.subTest(name=name):
                path = FIXTURE_DIR / f"{name}.json"
                self.assertTrue(
                    path.exists(),
                    f"{path} missing — run _make_schema_fixtures",
                )

    def test_fixture_has_top_level_stamps(self):
        """Every fixture has schema_url, schema_version, min_read_version."""
        for name in _SUPPORTED:
            with self.subTest(name=name):
                path = FIXTURE_DIR / f"{name}.json"
                tree = json.loads(path.read_text())
                self.assertIn("schema_url", tree)
                self.assertIn("schema_version", tree)
                self.assertIn("min_read_version", tree)

    def test_fixture_url_matches_name_and_version(self):
        """schema_url matches the ``<name>-<version>`` pattern."""
        for name in _SUPPORTED:
            with self.subTest(name=name):
                path = FIXTURE_DIR / f"{name}.json"
                tree = json.loads(path.read_text())
                expected = f"https://images.lsst.io/schemas/{name}-{tree['schema_version']}"
                self.assertEqual(tree["schema_url"], expected)


if __name__ == "__main__":
    unittest.main()

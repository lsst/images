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

import os
import tempfile
import unittest

import lsst.images  # registers schema classes
import lsst.images.cells  # noqa: F401  -- side-effect import for cell schemas
from lsst.images.serialization import (
    ArchiveReadError,
    ReadResult,
    backend_for_path,
    class_for_schema,
    read,
)
from lsst.images.serialization._io import _public_type

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "schema_v1")


class GenericReadTestCase(unittest.TestCase):
    """read(path) dispatches by extension and produces the registered type."""

    def test_visit_image_json(self) -> None:
        from lsst.images import VisitImage

        path = os.path.join(DATA_DIR, "visit_image.json")
        result = read(path)
        self.assertIsInstance(result, ReadResult)
        self.assertIsInstance(result.deserialized, VisitImage)

    def test_image_json(self) -> None:
        from lsst.images import Image

        path = os.path.join(DATA_DIR, "image.json")
        result = read(path)
        self.assertIsInstance(result.deserialized, Image)


class GenericReadErrorsTestCase(unittest.TestCase):
    """Unknown schemas and bad extensions raise clean errors."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name

    def test_unsupported_extension(self) -> None:
        path = os.path.join(self.tmp, "bogus.txt")
        with open(path, "w") as f:
            f.write("nope")
        # backend_for_path raises ValueError; read() must let it through.
        with self.assertRaises(ValueError):
            read(path)

    def test_unregistered_schema(self) -> None:
        # Write a JSON file with a fabricated schema_url so its
        # (name, version) is not in the registry.
        path = os.path.join(self.tmp, "fake.json")
        with open(path, "w") as f:
            f.write(
                '{"schema_url": "https://images.lsst.io/schemas/no-such-schema-99.0.0",'
                ' "schema_version": "99.0.0", "min_read_version": 1, "indirect": []}'
            )
        with self.assertRaises(ArchiveReadError) as ctx:
            read(path)
        self.assertIn("no-such-schema", str(ctx.exception))
        self.assertIn("99.0.0", str(ctx.exception))


class FixtureSweepTestCase(unittest.TestCase):
    """Every schema_v1 JSON fixture reads through the generic API and
    produces the in-memory type registered for its schema.
    """

    def test_sweep(self) -> None:
        # piff and PSFEx are optional dependencies; the corresponding
        # serialization models import them lazily when deserialising.
        # Skip those fixtures here when the dependency isn't installed.
        try:
            import piff  # noqa: F401
        except ImportError:
            piff_available = False
        else:
            piff_available = True
        skip_if_missing = {
            "piff_psf.json": piff_available,
        }
        roots = [DATA_DIR, os.path.join(DATA_DIR, "legacy")]
        for root in roots:
            if not os.path.isdir(root):
                continue
            for entry in sorted(os.listdir(root)):
                if not entry.endswith(".json"):
                    continue
                if entry in skip_if_missing and not skip_if_missing[entry]:
                    continue
                path = os.path.join(root, entry)
                with self.subTest(path=path):
                    result = read(path)
                    info = backend_for_path(path).input_archive.get_basic_info(path)
                    cls = class_for_schema(info.schema_name, info.schema_version)
                    self.assertIsNotNone(cls)
                    expected_type = _public_type(cls)  # type: ignore[arg-type]
                    self.assertIsNotNone(expected_type)
                    self.assertIsInstance(result.deserialized, expected_type)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()

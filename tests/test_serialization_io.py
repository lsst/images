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

import numpy as np

import lsst.images  # registers schema classes
import lsst.images.cells  # noqa: F401  -- side-effect import for cell schemas
from lsst.images import Box, Image, VisitImage
from lsst.images.serialization import (
    ArchiveReadError,
    ReadResult,
    backend_for_path,
    class_for_schema,
    read,
    write,
)
from lsst.images.serialization._io import _public_type

try:
    import h5py  # noqa: F401  -- detect availability for NDF round-trip skip
except ImportError:
    H5PY_AVAILABLE = False
else:
    H5PY_AVAILABLE = True

try:
    import piff  # noqa: F401  -- detect availability for piff_psf fixture skip
except ImportError:
    PIFF_AVAILABLE = False
else:
    PIFF_AVAILABLE = True

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "schema_v1")


class GenericReadTestCase(unittest.TestCase):
    """read(path) dispatches by extension and produces the registered type."""

    def test_visit_image_json(self) -> None:
        path = os.path.join(DATA_DIR, "visit_image.json")
        result = read(path)
        self.assertIsInstance(result, ReadResult)
        self.assertIsInstance(result.deserialized, VisitImage)

    def test_image_json(self) -> None:
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
        # piff is an optional dependency that PiffSerializationModel
        # imports lazily on deserialise; skip its fixture when missing.
        skip = set() if PIFF_AVAILABLE else {"piff_psf.json"}
        roots = [DATA_DIR, os.path.join(DATA_DIR, "legacy")]
        for root in roots:
            if not os.path.isdir(root):
                continue
            for entry in sorted(os.listdir(root)):
                if not entry.endswith(".json"):
                    continue
                if entry in skip:
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


class GenericWriteRoundTripTestCase(unittest.TestCase):
    """write(obj, path) dispatches by extension and round-trips."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.image = Image(np.arange(16, dtype=np.float32).reshape(4, 4), bbox=Box.factory[0:4, 0:4])

    def test_round_trip_fits(self) -> None:
        path = os.path.join(self.tmp, "x.fits")
        write(self.image, path)
        result = read(path)
        self.assertIsInstance(result.deserialized, Image)
        np.testing.assert_array_equal(result.deserialized.array, self.image.array)

    def test_round_trip_json(self) -> None:
        path = os.path.join(self.tmp, "x.json")
        write(self.image, path)
        result = read(path)
        self.assertIsInstance(result.deserialized, Image)
        np.testing.assert_array_equal(result.deserialized.array, self.image.array)

    @unittest.skipUnless(H5PY_AVAILABLE, "h5py not available.")
    def test_round_trip_ndf(self) -> None:
        path = os.path.join(self.tmp, "x.sdf")
        write(self.image, path)
        result = read(path)
        self.assertIsInstance(result.deserialized, Image)
        np.testing.assert_array_equal(result.deserialized.array, self.image.array)


if __name__ == "__main__":
    unittest.main()

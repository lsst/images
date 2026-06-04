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

from lsst.images import Box, Image, VisitImage
from lsst.images.serialization import ArchiveReadError, ReadResult, read, write
from lsst.utils.introspection import get_full_type_name

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

# Full Python type produced when each fixture is read through the generic
# read() API, keyed by the fixture's file name.  These are pinned here rather
# than derived from the schema registry so the test asserts the
# externally-observable type instead of re-running read()'s own lookup against
# itself.
EXPECTED_TYPES = {
    "aperture_correction_map.json": "dict",
    "background_map.json": "lsst.images.BackgroundMap",
    "cell_psf.json": "lsst.images.cells.CellPointSpreadFunction",
    "chebyshev_field.json": "lsst.images.fields.ChebyshevField",
    "coadd_provenance.json": "lsst.images.cells.CoaddProvenance",
    "color_image.json": "lsst.images.ColorImage",
    "detector.json": "lsst.images.cameras.Detector",
    "gaussian_psf.json": "lsst.images.psfs.GaussianPointSpreadFunction",
    "image.json": "lsst.images.Image",
    "mask.json": "lsst.images.Mask",
    "masked_image.json": "lsst.images.MaskedImage",
    "piff_psf.json": "lsst.images.psfs.PiffWrapper",
    "product_field.json": "lsst.images.fields.ProductField",
    "projection.json": "lsst.images.Projection",
    "sum_field.json": "lsst.images.fields.SumField",
    "transform.json": "lsst.images.Transform",
    "visit_image.json": "lsst.images.VisitImage",
    "cell_coadd.json": "lsst.images.cells.CellCoadd",
    "visit_image_dp1.json": "lsst.images.VisitImage",
    "visit_image_dp2.json": "lsst.images.VisitImage",
}


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
        # Write a JSON file with a fabricated schema name not in the
        # registry.
        path = os.path.join(self.tmp, "fake.json")
        with open(path, "w") as f:
            f.write(
                '{"schema_url": "https://images.lsst.io/schemas/no-such-schema-99.0.0",'
                ' "schema_version": "99.0.0", "min_read_version": 1, "indirect": []}'
            )
        with self.assertRaises(ArchiveReadError) as ctx:
            read(path)
        self.assertIn("no-such-schema", str(ctx.exception))


class FixtureSweepTestCase(unittest.TestCase):
    """Every schema_v1 JSON fixture reads through the generic API and
    produces the Python type pinned in ``EXPECTED_TYPES``.
    """

    def test_sweep(self) -> None:
        # piff is an optional dependency that PiffSerializationModel
        # imports lazily on deserialise; skip its fixture when missing.
        skip = set() if PIFF_AVAILABLE else {"piff_psf.json"}
        seen = set()
        roots = [DATA_DIR, os.path.join(DATA_DIR, "legacy")]
        for root in roots:
            if not os.path.isdir(root):
                continue
            for entry in sorted(os.listdir(root)):
                if not entry.endswith(".json") or entry in skip:
                    continue
                with self.subTest(entry=entry):
                    self.assertIn(entry, EXPECTED_TYPES, f"no expected type recorded for {entry!r}")
                    result = read(os.path.join(root, entry))
                    self.assertEqual(get_full_type_name(type(result.deserialized)), EXPECTED_TYPES[entry])
                seen.add(entry)
        # Fail if EXPECTED_TYPES drifts from the fixtures actually on disk.
        self.assertEqual(seen, set(EXPECTED_TYPES) - skip)


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


class GenericReadKwargsTestCase(unittest.TestCase):
    """**kwargs forwarded by read() reach the backend deserialize."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name

    def test_bbox_subset_fits(self) -> None:
        img = Image(np.arange(64, dtype=np.float32).reshape(8, 8), bbox=Box.factory[0:8, 0:8])
        path = os.path.join(self.tmp, "x.fits")
        write(img, path)
        # Read a 4x4 subset.  bbox is the FITS-specific kwarg understood
        # by Image.deserialize; the generic read must forward it.
        sub = read(path, bbox=Box.factory[2:6, 2:6])
        self.assertEqual(sub.deserialized.array.shape, (4, 4))
        np.testing.assert_array_equal(sub.deserialized.array, img.array[2:6, 2:6])


if __name__ == "__main__":
    unittest.main()

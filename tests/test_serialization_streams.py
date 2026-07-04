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
"""Tests for reading archives from in-memory bytes and binary streams."""

from __future__ import annotations

import io
import os
import tempfile
import unittest

import numpy as np

from lsst.images import Box, Image, Mask
from lsst.images.serialization import open as open_archive
from lsst.images.serialization import read, read_from_bytes, write

try:
    import h5py  # noqa: F401  -- detect availability for NDF stream skips

    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False


def _test_image() -> Image:
    return Image(np.arange(16, dtype=np.float32).reshape(4, 4), bbox=Box.factory[0:4, 0:4])


def _serialized_bytes(obj: object, extension: str) -> bytes:
    """Write ``obj`` to a temporary file and return the file's content."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, f"x{extension}")
        write(obj, path)
        with open(path, "rb") as f:
            return f.read()


class FitsOpenTreeStreamTestCase(unittest.TestCase):
    """FitsInputArchive.open_tree accepts a seekable binary stream."""

    def test_open_tree_stream(self) -> None:
        from lsst.images.fits import FitsInputArchive

        image = _test_image()
        stream = io.BytesIO(_serialized_bytes(image, ".fits"))
        with FitsInputArchive.open_tree(stream) as (archive, tree, info):
            self.assertEqual(info.schema_name, "image")
            result = tree.deserialize(archive)
        self.assertIsInstance(result, Image)
        np.testing.assert_array_equal(result.array, image.array)


class JsonOpenTreeStreamTestCase(unittest.TestCase):
    """JsonInputArchive.open_tree accepts a seekable binary stream."""

    def test_open_tree_stream(self) -> None:
        from lsst.images.json import JsonInputArchive

        image = _test_image()
        stream = io.BytesIO(_serialized_bytes(image, ".json"))
        with JsonInputArchive.open_tree(stream) as (archive, tree, info):
            result = tree.deserialize(archive)
        self.assertIsInstance(result, Image)
        np.testing.assert_array_equal(result.array, image.array)


@unittest.skipUnless(H5PY_AVAILABLE, "h5py not available.")
class NdfOpenTreeStreamTestCase(unittest.TestCase):
    """NdfInputArchive.open_tree accepts a seekable binary stream."""

    def test_open_tree_stream(self) -> None:
        from lsst.images.ndf import NdfInputArchive

        image = _test_image()
        stream = io.BytesIO(_serialized_bytes(image, ".sdf"))
        with NdfInputArchive.open_tree(stream) as (archive, tree, info):
            result = tree.deserialize(archive)
        self.assertIsInstance(result, Image)
        np.testing.assert_array_equal(result.array, image.array)


class ReadFromBytesTestCase(unittest.TestCase):
    """read_from_bytes turns in-memory data into objects, all backends."""

    def setUp(self) -> None:
        self.image = _test_image()

    def test_fits(self) -> None:
        result = read_from_bytes(_serialized_bytes(self.image, ".fits"))
        self.assertIsInstance(result, Image)
        np.testing.assert_array_equal(result.array, self.image.array)

    def test_json(self) -> None:
        result = read_from_bytes(_serialized_bytes(self.image, ".json"))
        self.assertIsInstance(result, Image)
        np.testing.assert_array_equal(result.array, self.image.array)

    @unittest.skipUnless(H5PY_AVAILABLE, "h5py not available.")
    def test_ndf(self) -> None:
        result = read_from_bytes(_serialized_bytes(self.image, ".sdf"))
        self.assertIsInstance(result, Image)
        np.testing.assert_array_equal(result.array, self.image.array)

    def test_buffer_protocol_inputs(self) -> None:
        data = _serialized_bytes(self.image, ".json")
        for buffer in (bytearray(data), memoryview(data)):
            with self.subTest(type=type(buffer).__name__):
                result = read_from_bytes(buffer)
                self.assertIsInstance(result, Image)

    def test_cls_match_and_mismatch(self) -> None:
        data = _serialized_bytes(self.image, ".fits")
        result = read_from_bytes(data, cls=Image)
        self.assertIsInstance(result, Image)
        with self.assertRaises(TypeError):
            read_from_bytes(data, cls=Mask)

    def test_kwargs_forwarded(self) -> None:
        big = Image(np.arange(64, dtype=np.float32).reshape(8, 8), bbox=Box.factory[0:8, 0:8])
        sub = read_from_bytes(_serialized_bytes(big, ".fits"), bbox=Box.factory[2:6, 2:6])
        self.assertEqual(sub.array.shape, (4, 4))
        np.testing.assert_array_equal(sub.array, big.array[2:6, 2:6])

    def test_format_override(self) -> None:
        result = read_from_bytes(_serialized_bytes(self.image, ".json"), format="json")
        self.assertIsInstance(result, Image)

    def test_wrong_format_override(self) -> None:
        # FITS bytes forced through the JSON backend fail in that backend.
        with self.assertRaises(ValueError):
            read_from_bytes(_serialized_bytes(self.image, ".fits"), format="json")

    def test_unrecognized_bytes(self) -> None:
        with self.assertRaises(ValueError) as cm:
            read_from_bytes(b"certainly not a supported format")
        self.assertIn("FITS", str(cm.exception))

    def test_bad_format_name(self) -> None:
        with self.assertRaises(ValueError):
            read_from_bytes(_serialized_bytes(self.image, ".json"), format="asdf")


class OpenStreamTestCase(unittest.TestCase):
    """open() accepts a seekable binary stream, with component reads."""

    DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "schema_v1")

    def test_open_stream_components(self) -> None:
        visit_image = read(os.path.join(self.DATA_DIR, "visit_image.json"))
        stream = io.BytesIO(_serialized_bytes(visit_image, ".fits"))
        with open_archive(stream) as reader:
            self.assertEqual(reader.info.schema_name, "visit_image")
            self.assertIsInstance(reader.metadata, dict)
            self.assertIsNotNone(reader.get_component("sky_projection"))
            full = reader.read()
        self.assertEqual(type(full).__name__, "VisitImage")


if __name__ == "__main__":
    unittest.main()

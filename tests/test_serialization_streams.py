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

import gzip
import io
import os
import tempfile
import unittest

import numpy as np

from lsst.images import Box, Image, Mask
from lsst.images.fits import FitsInputArchive
from lsst.images.json import JsonInputArchive
from lsst.images.serialization import open as open_archive
from lsst.images.serialization import read, write

try:
    import h5py  # noqa: F401  -- detect availability for NDF stream skips

    from lsst.images.ndf import NdfInputArchive

    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

try:
    from compression import zstd as _stdlib_zstd  # noqa: F401  -- detect zstd availability

    ZSTD_AVAILABLE = True
except ImportError:
    try:
        import zstandard  # noqa: F401

        ZSTD_AVAILABLE = True
    except ImportError:
        ZSTD_AVAILABLE = False


def _zstd_compress(data: bytes) -> bytes:
    """Compress with whichever zstd library is available."""
    try:
        from compression import zstd
    except ImportError:
        import zstandard

        return zstandard.ZstdCompressor().compress(data)
    return zstd.compress(data)


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
        image = _test_image()
        stream = io.BytesIO(_serialized_bytes(image, ".sdf"))
        with NdfInputArchive.open_tree(stream) as (archive, tree, info):
            result = tree.deserialize(archive)
        self.assertIsInstance(result, Image)
        np.testing.assert_array_equal(result.array, image.array)


class ReadStreamTestCase(unittest.TestCase):
    """read(io.BytesIO(data)) turns in-memory data into objects."""

    def setUp(self) -> None:
        self.image = _test_image()

    def test_fits(self) -> None:
        result = read(io.BytesIO(_serialized_bytes(self.image, ".fits")))
        self.assertIsInstance(result, Image)
        np.testing.assert_array_equal(result.array, self.image.array)

    def test_json(self) -> None:
        result = read(io.BytesIO(_serialized_bytes(self.image, ".json")))
        self.assertIsInstance(result, Image)
        np.testing.assert_array_equal(result.array, self.image.array)

    @unittest.skipUnless(H5PY_AVAILABLE, "h5py not available.")
    def test_ndf(self) -> None:
        result = read(io.BytesIO(_serialized_bytes(self.image, ".sdf")))
        self.assertIsInstance(result, Image)
        np.testing.assert_array_equal(result.array, self.image.array)

    def test_cls_match_and_mismatch(self) -> None:
        data = _serialized_bytes(self.image, ".fits")
        result = read(io.BytesIO(data), cls=Image)
        self.assertIsInstance(result, Image)
        with self.assertRaises(TypeError):
            read(io.BytesIO(data), cls=Mask)

    def test_kwargs_forwarded(self) -> None:
        big = Image(np.arange(64, dtype=np.float32).reshape(8, 8), bbox=Box.factory[0:8, 0:8])
        sub = read(io.BytesIO(_serialized_bytes(big, ".fits")), bbox=Box.factory[2:6, 2:6])
        self.assertEqual(sub.array.shape, (4, 4))
        np.testing.assert_array_equal(sub.array, big.array[2:6, 2:6])

    def test_format_override(self) -> None:
        result = read(io.BytesIO(_serialized_bytes(self.image, ".json")), format="json")
        self.assertIsInstance(result, Image)

    def test_wrong_format_override(self) -> None:
        # FITS bytes forced through the JSON backend fail in that backend.
        with self.assertRaises(ValueError):
            read(io.BytesIO(_serialized_bytes(self.image, ".fits")), format="json")

    def test_unrecognized_bytes(self) -> None:
        with self.assertRaises(ValueError) as cm:
            read(io.BytesIO(b"certainly not a supported format"))
        self.assertIn("FITS", str(cm.exception))

    def test_bad_format_name(self) -> None:
        with self.assertRaises(ValueError):
            read(io.BytesIO(_serialized_bytes(self.image, ".json")), format="asdf")

    def test_compressed_stream_raises(self) -> None:
        # Compressed streams are the caller's responsibility to decompress;
        # the sniff error says what to do.
        data = gzip.compress(_serialized_bytes(self.image, ".fits"))
        with self.assertRaises(ValueError) as cm:
            read(io.BytesIO(data))
        self.assertIn("gzip", str(cm.exception))


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


class CompressedPathTestCase(unittest.TestCase):
    """Compressed files read transparently through path-based read()."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.image = _test_image()

    def _write_compressed(self, extension: str, compress) -> str:
        data = compress(_serialized_bytes(self.image, extension))
        suffix = ".gz" if compress is gzip.compress else ".zst"
        path = os.path.join(self.tmp, f"x{extension}{suffix}")
        with open(path, "wb") as f:
            f.write(data)
        return path

    def test_fits_gz(self) -> None:
        # Regression: .fits.gz was dispatched to the FITS backend but
        # handed to it still compressed.
        path = self._write_compressed(".fits", gzip.compress)
        result = read(path)
        self.assertIsInstance(result, Image)
        np.testing.assert_array_equal(result.array, self.image.array)

    def test_json_gz(self) -> None:
        path = self._write_compressed(".json", gzip.compress)
        self.assertIsInstance(read(path), Image)

    @unittest.skipUnless(ZSTD_AVAILABLE, "no zstd decompressor available.")
    def test_fits_zst(self) -> None:
        path = self._write_compressed(".fits", _zstd_compress)
        self.assertIsInstance(read(path), Image)

    def test_open_fits_gz(self) -> None:
        path = self._write_compressed(".fits", gzip.compress)
        with open_archive(path) as reader:
            self.assertIsInstance(reader.read(), Image)


if __name__ == "__main__":
    unittest.main()

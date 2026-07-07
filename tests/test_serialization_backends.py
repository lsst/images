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

import gzip
import io
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from lsst.images import Box, Image
from lsst.images import fits as images_fits
from lsst.images.serialization import Backend, backend_for_name, backend_for_path, backend_for_stream
from lsst.images.serialization._backends import (
    _decompress_path_to_temp_file,
    _is_binary_stream,
    _path_is_compressed,
)
from lsst.resources import ResourcePath

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


class BackendForPathTestCase(unittest.TestCase):
    """Tests for suffix -> backend resolution."""

    def test_fits(self) -> None:
        from lsst.images.fits import FitsInputArchive

        b = backend_for_path("a/b/c.fits")
        self.assertIsInstance(b, Backend)
        self.assertEqual(b.name, "fits")
        self.assertIs(b.input_archive, FitsInputArchive)
        self.assertTrue(callable(b.write))

    def test_fits_gz(self) -> None:
        self.assertEqual(backend_for_path("c.fits.gz").name, "fits")
        self.assertEqual(backend_for_path("file://a/b/c.fits.gz?param=2").name, "fits")

    def test_json(self) -> None:
        from lsst.images.json import JsonInputArchive

        b = backend_for_path("c.json")
        self.assertEqual(b.name, "json")
        self.assertIs(b.input_archive, JsonInputArchive)

    def test_ndf(self) -> None:
        self.assertEqual(backend_for_path("c.sdf").name, "ndf")
        self.assertEqual(backend_for_path("c.h5").name, "ndf")

    def test_unknown(self) -> None:
        with self.assertRaises(ValueError) as cm:
            backend_for_path("c.txt")
        self.assertIn(".fits", str(cm.exception))


class MinifyDispatchTestCase(unittest.TestCase):
    """minify resolves backend and schema via the shared APIs."""

    def test_minify_unsupported_schema_uses_shared_dispatch(self) -> None:
        from lsst.images.tests._minify_for_fixtures import minify

        tmp = tempfile.mkdtemp()
        src = os.path.join(tmp, "plain.fits")
        out = os.path.join(tmp, "plain.json")
        images_fits.write(Image(np.zeros((4, 4), dtype=np.float32), bbox=Box.factory[0:4, 0:4]), src)
        # Reaching the "no subsetter" error proves backend_for_path and
        # get_basic_info ran and detected schema_name "image".
        with self.assertRaises(NotImplementedError) as cm:
            minify(src, out)
        self.assertIn("image", str(cm.exception))


class BackendForNameTestCase(unittest.TestCase):
    """Explicit format-name -> backend resolution."""

    def test_names(self) -> None:
        self.assertEqual(backend_for_name("fits").name, "fits")
        self.assertEqual(backend_for_name("ndf").name, "ndf")
        self.assertEqual(backend_for_name("json").name, "json")

    def test_unknown(self) -> None:
        with self.assertRaises(ValueError) as cm:
            backend_for_name("hdf")
        self.assertIn("hdf", str(cm.exception))


class BackendForStreamTestCase(unittest.TestCase):
    """Content sniffing -> backend resolution."""

    def test_fits_magic(self) -> None:
        stream = io.BytesIO(b"SIMPLE  =                    T / conforms")
        self.assertEqual(backend_for_stream(stream).name, "fits")
        # Sniffing must not consume the stream.
        self.assertEqual(stream.tell(), 0)

    def test_hdf5_magic(self) -> None:
        stream = io.BytesIO(b"\x89HDF\r\n\x1a\n" + b"\x00" * 8)
        self.assertEqual(backend_for_stream(stream).name, "ndf")

    def test_json(self) -> None:
        self.assertEqual(backend_for_stream(io.BytesIO(b'{"schema_url": "x"}')).name, "json")

    def test_json_leading_whitespace(self) -> None:
        self.assertEqual(backend_for_stream(io.BytesIO(b'  \n\t {"a": 1}')).name, "json")

    def test_unknown(self) -> None:
        with self.assertRaises(ValueError) as cm:
            backend_for_stream(io.BytesIO(b"not any known format"))
        msg = str(cm.exception)
        self.assertIn("FITS", msg)
        self.assertIn("format", msg)

    def test_gzip_compressed_stream_raises(self) -> None:
        with self.assertRaises(ValueError) as cm:
            backend_for_stream(io.BytesIO(gzip.compress(b"SIMPLE  = whatever")))
        self.assertIn("gzip", str(cm.exception))

    def test_zstd_compressed_stream_raises(self) -> None:
        with self.assertRaises(ValueError) as cm:
            backend_for_stream(io.BytesIO(b"\x28\xb5\x2f\xfd" + b"frame"))
        self.assertIn("zstd", str(cm.exception))


class BackendForPathCompressionTestCase(unittest.TestCase):
    """Compression suffixes are stripped before extension dispatch."""

    def test_gz(self) -> None:
        self.assertEqual(backend_for_path("c.json.gz").name, "json")
        self.assertEqual(backend_for_path("c.h5.gz").name, "ndf")

    def test_zst(self) -> None:
        self.assertEqual(backend_for_path("c.fits.zst").name, "fits")
        self.assertEqual(backend_for_path("c.sdf.zst").name, "ndf")

    def test_bare_compression_suffix(self) -> None:
        with self.assertRaises(ValueError):
            backend_for_path("c.gz")


class StreamHelpersTestCase(unittest.TestCase):
    """Stream detection and compression-suffix helpers."""

    def test_is_binary_stream(self) -> None:
        self.assertTrue(_is_binary_stream(io.BytesIO(b"")))
        self.assertFalse(_is_binary_stream("a.fits"))
        # ResourcePath has read() but no seek(); it must not look like a
        # stream.
        self.assertFalse(_is_binary_stream(ResourcePath("a.fits", forceAbsolute=False)))

    def test_path_is_compressed(self) -> None:
        self.assertTrue(_path_is_compressed("a/b.fits.gz"))
        self.assertTrue(_path_is_compressed("a/b.json.zst"))
        self.assertFalse(_path_is_compressed("a/b.fits"))


class DecompressPathToTempFileTestCase(unittest.TestCase):
    """Compressed paths stream-decompress into a temporary file on disk."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.payload = b"SIMPLE  = fake fits payload " * 1000

    def test_gzip(self) -> None:
        path = os.path.join(self.tmp, "x.fits.gz")
        with open(path, "wb") as f:
            f.write(gzip.compress(self.payload))
        with _decompress_path_to_temp_file(path) as handle:
            # The decompressed data lives in a real file, not in memory.
            self.assertNotIsInstance(handle, io.BytesIO)
            self.assertTrue(hasattr(handle, "fileno"))
            self.assertEqual(handle.read(), self.payload)

    @unittest.skipUnless(ZSTD_AVAILABLE, "no zstd decompressor available.")
    def test_zstd(self) -> None:
        path = os.path.join(self.tmp, "x.fits.zst")
        with open(path, "wb") as f:
            f.write(_zstd_compress(self.payload))
        with _decompress_path_to_temp_file(path) as handle:
            self.assertEqual(handle.read(), self.payload)

    def test_suffix_without_compression_raises(self) -> None:
        # The suffix selects the decompressor, so a .gz name whose content
        # is not gzip fails honestly.
        path = os.path.join(self.tmp, "x.fits.gz")
        with open(path, "wb") as f:
            f.write(self.payload)
        with self.assertRaises(gzip.BadGzipFile):
            _decompress_path_to_temp_file(path)

    def test_zstd_no_decompressor(self) -> None:
        path = os.path.join(self.tmp, "x.fits.zst")
        with open(path, "wb") as f:
            f.write(b"\x28\xb5\x2f\xfd" + b"pretend zstd frame")
        # None entries make both import routes raise ImportError.
        blocked = {"compression": None, "compression.zstd": None, "zstandard": None}
        with mock.patch.dict(sys.modules, blocked):
            with self.assertRaises(ValueError) as cm:
                _decompress_path_to_temp_file(path)
        self.assertIn("zstd", str(cm.exception))


if __name__ == "__main__":
    unittest.main()

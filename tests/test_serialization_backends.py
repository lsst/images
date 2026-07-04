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
from lsst.images.serialization import Backend, backend_for_path

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


class DecompressionTestCase(unittest.TestCase):
    """Magic-number detection and in-memory decompression."""

    def test_gzip(self) -> None:
        from lsst.images.serialization._backends import _maybe_decompress_stream

        stream = io.BytesIO(gzip.compress(b"payload"))
        out = _maybe_decompress_stream(stream)
        self.assertIsNot(out, stream)
        self.assertEqual(out.read(), b"payload")

    @unittest.skipUnless(ZSTD_AVAILABLE, "no zstd decompressor available.")
    def test_zstd(self) -> None:
        from lsst.images.serialization._backends import _maybe_decompress_stream

        stream = io.BytesIO(_zstd_compress(b"payload"))
        out = _maybe_decompress_stream(stream)
        self.assertIsNot(out, stream)
        self.assertEqual(out.read(), b"payload")

    def test_passthrough(self) -> None:
        from lsst.images.serialization._backends import _maybe_decompress_stream

        stream = io.BytesIO(b"SIMPLE  = plain uncompressed data")
        out = _maybe_decompress_stream(stream)
        self.assertIs(out, stream)
        # The stream must be left at the position it came in with.
        self.assertEqual(out.tell(), 0)

    def test_zstd_no_decompressor(self) -> None:
        from lsst.images.serialization._backends import _maybe_decompress_stream

        data = b"\x28\xb5\x2f\xfd" + b"pretend zstd frame"
        # None entries make both import routes raise ImportError.
        blocked = {"compression": None, "compression.zstd": None, "zstandard": None}
        with mock.patch.dict(sys.modules, blocked):
            with self.assertRaises(ValueError) as cm:
                _maybe_decompress_stream(io.BytesIO(data))
        self.assertIn("zstd", str(cm.exception))

    def test_is_binary_stream(self) -> None:
        from lsst.images.serialization._backends import _is_binary_stream
        from lsst.resources import ResourcePath

        self.assertTrue(_is_binary_stream(io.BytesIO(b"")))
        self.assertFalse(_is_binary_stream("a.fits"))
        # ResourcePath has read() but no seek(); it must not look like a
        # stream.
        self.assertFalse(_is_binary_stream(ResourcePath("a.fits", forceAbsolute=False)))

    def test_path_is_compressed(self) -> None:
        from lsst.images.serialization._backends import _path_is_compressed

        self.assertTrue(_path_is_compressed("a/b.fits.gz"))
        self.assertTrue(_path_is_compressed("a/b.json.zst"))
        self.assertFalse(_path_is_compressed("a/b.fits"))


if __name__ == "__main__":
    unittest.main()

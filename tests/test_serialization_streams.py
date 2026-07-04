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

from lsst.images import Box, Image
from lsst.images.serialization import write

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


if __name__ == "__main__":
    unittest.main()

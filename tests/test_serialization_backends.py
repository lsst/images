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

from lsst.images import Box, Image
from lsst.images import fits as images_fits
from lsst.images.serialization import Backend, backend_for_path


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


if __name__ == "__main__":
    unittest.main()

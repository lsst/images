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

import unittest

from lsst.images.serialization import Backend, backend_for_path


class BackendForPathTestCase(unittest.TestCase):
    """Tests for suffix -> backend resolution."""

    def test_fits(self) -> None:
        from lsst.images.fits import FitsInputArchive

        b = backend_for_path("a/b/c.fits")
        self.assertIsInstance(b, Backend)
        self.assertEqual(b.name, "fits")
        self.assertIs(b.input_archive, FitsInputArchive)
        self.assertTrue(callable(b.read) and callable(b.write))

    def test_fits_gz(self) -> None:
        self.assertEqual(backend_for_path("c.fits.gz").name, "fits")

    def test_json(self) -> None:
        from lsst.images.json import JsonInputArchive

        b = backend_for_path("c.json")
        self.assertEqual(b.name, "json")
        self.assertIs(b.input_archive, JsonInputArchive)

    def test_ndf(self) -> None:
        self.assertEqual(backend_for_path("c.sdf").name, "ndf")
        self.assertEqual(backend_for_path("c.ndf").name, "ndf")

    def test_unknown(self) -> None:
        with self.assertRaises(ValueError) as cm:
            backend_for_path("c.txt")
        self.assertIn(".fits", str(cm.exception))


if __name__ == "__main__":
    unittest.main()

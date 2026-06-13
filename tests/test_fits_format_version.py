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

import astropy.io.fits

from lsst.images import Image
from lsst.images.fits import FitsInputArchive
from lsst.images.serialization import ArchiveReadError, write


def _write_simple_image_fits(path: str) -> None:
    """Write a tiny Image to ``path`` via the high-level API."""
    image = Image(0.0, shape=(4, 4), dtype="float32")
    write(image, path)


class FitsFormatVersionTestCase(unittest.TestCase):
    """Tests for the FITS FMTVER and DATAMODL primary-header keywords."""

    def test_write_emits_fmtver_and_datamodl(self) -> None:
        """A freshly-written FITS carries FMTVER=1 and the root DATAMODL."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.fits")
            _write_simple_image_fits(path)
            with astropy.io.fits.open(path) as hdul:
                self.assertEqual(hdul[0].header["FMTVER"], 1)
                self.assertEqual(
                    hdul[0].header["DATAMODL"],
                    "https://images.lsst.io/schemas/image-1.0.0",
                )

    def test_read_succeeds_when_fmtver_matches(self) -> None:
        """Round-trip read of a freshly-written file succeeds without error."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.fits")
            _write_simple_image_fits(path)
            with FitsInputArchive.open(path):
                pass

    def test_read_fails_when_fmtver_too_high(self) -> None:
        """A file whose FMTVER is newer than this release raises."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.fits")
            _write_simple_image_fits(path)
            with astropy.io.fits.open(path, mode="update") as hdul:
                hdul[0].header["FMTVER"] = 2
                hdul.flush()
            with self.assertRaises(ArchiveReadError):
                with FitsInputArchive.open(path):
                    pass

    def test_read_succeeds_when_fmtver_absent(self) -> None:
        """A legacy file lacking FMTVER reads successfully (defaults to 1)."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.fits")
            _write_simple_image_fits(path)
            with astropy.io.fits.open(path, mode="update") as hdul:
                if "FMTVER" in hdul[0].header:
                    del hdul[0].header["FMTVER"]
                if "DATAMODL" in hdul[0].header:
                    del hdul[0].header["DATAMODL"]
                hdul.flush()
            with FitsInputArchive.open(path):
                pass


if __name__ == "__main__":
    unittest.main()

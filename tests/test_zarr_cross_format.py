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
import numpy as np

from lsst.images import Box, Image
from lsst.images.fits import write as fits_write
from lsst.images.serialization import read_archive

try:
    import zarr  # noqa: F401

    from lsst.images.zarr import write as zarr_write

    HAVE_ZARR = True
except ImportError:
    HAVE_ZARR = False


@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class FitsZarrCrossFormatTestCase(unittest.TestCase):
    """End-to-end FITS -> Zarr -> FITS preserves the primary header."""

    def test_fits_to_zarr_to_fits_preserves_primary_header(self) -> None:
        original = Image(
            np.arange(20, dtype=np.float32).reshape(4, 5),
            bbox=Box.factory[10:14, 20:25],
        )
        with tempfile.TemporaryDirectory() as tmp:
            fits_a = os.path.join(tmp, "a.fits")
            zarr_path = os.path.join(tmp, "b.zarr")
            fits_b = os.path.join(tmp, "c.fits")

            def update_header(header):
                header["ORIGIN"] = "RUBIN"
                header["EXPTIME"] = 30.0

            fits_write(original, fits_a, update_header=update_header)
            from_fits = read_archive(fits_a, Image)
            zarr_write(from_fits, zarr_path)
            from_zarr = read_archive(zarr_path, Image)
            fits_write(from_zarr, fits_b)

            with astropy.io.fits.open(fits_b) as hdul:
                self.assertEqual(hdul[0].header["ORIGIN"], "RUBIN")
                self.assertEqual(hdul[0].header["EXPTIME"], 30.0)


if __name__ == "__main__":
    unittest.main()

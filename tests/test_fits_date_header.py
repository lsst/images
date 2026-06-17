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
import astropy.time

from lsst.images import Image
from lsst.images.fits import ExtensionKey, FitsInputArchive


class FitsDateHeaderTestCase(unittest.TestCase):
    """Tests for the FITS ``DATE`` card the standard requires on every HDU."""

    def setUp(self) -> None:
        # Assume Image written correctly will be representative.
        self.image = Image(0.0, shape=(4, 4), dtype="float32")

    def test_every_hdu_has_a_fits_compliant_date(self) -> None:
        """Each HDU carries a DATE card whose value is a valid FITS datetime
        recording (approximately) when the file was written.
        """
        before = astropy.time.Time.now()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.fits")
            self.image.write(path)
            after = astropy.time.Time.now()
            with astropy.io.fits.open(path) as hdul:
                self.assertGreater(len(hdul), 1)
                for index, hdu in enumerate(hdul):
                    extname = hdu.header.get("EXTNAME", "PRIMARY")
                    with self.subTest(hdu=index, extname=extname):
                        self.assertIn("DATE", hdu.header)
                        # The value must parse in the FITS time format.
                        date = astropy.time.Time(hdu.header["DATE"], format="fits")
                        # And it must reflect this write, not a stale value.
                        self.assertGreaterEqual(date.jd, before.jd)
                        self.assertLessEqual(date.jd, after.jd)

    def test_date_is_not_captured_as_opaque_metadata(self) -> None:
        """DATE is regenerated on every write, so the value read back must not
        be carried in opaque metadata (which would propagate a stale date to
        the next write).
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.fits")
            self.image.write(path)
            with FitsInputArchive.open(path) as archive:
                opaque = archive.get_opaque_metadata()
                primary_header = opaque.headers[ExtensionKey()]
                self.assertNotIn("DATE", primary_header)


if __name__ == "__main__":
    unittest.main()

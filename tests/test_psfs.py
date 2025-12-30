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
import unittest

from lsst.images import Box, Image
from lsst.images.psfs import PointSpreadFunction
from lsst.images.psfs.legacy import LegacyPointSpreadFunction
from lsst.images.psfs.piff import PiffWrapper

DATA_DIR = os.environ.get("TESTDATA_IMAGES_DIR", None)


class PointSpreadFunctionTestCase(unittest.TestCase):
    """Tests for the PointSpreadFunction classes."""

    @unittest.skipUnless(DATA_DIR is not None, "TESTDATA_IMAGES_DIR is not in the environment.")
    def test_piff(self) -> None:
        """Test that we can read a legacy Piff PSF with afw and convert it to
        the new `PiffWrapper`, and then check that it behaves consistently with
        the legacy PSF type.

        This test is skipped if legacy modules cannot be imported.
        """
        try:
            from piff import PSF

            from lsst.afw.image import ExposureFitsReader
        except ImportError:
            raise unittest.SkipTest("'lsst.afw.image' could not be imported.") from None
        assert DATA_DIR is not None, "Guaranteed by decorator."
        filename = os.path.join(DATA_DIR, "extracted", "visit_image.fits")
        reader = ExposureFitsReader(filename)
        legacy_psf = reader.readPsf()
        domain = Box.from_legacy(reader.readBBox())
        psf = PointSpreadFunction.from_legacy(legacy_psf, domain)
        self.assertIsInstance(psf, PiffWrapper)
        self.assertEqual(psf.domain, domain)
        self.assertIsInstance(psf.piff_psf, PSF)

    @unittest.skipUnless(DATA_DIR is not None, "TESTDATA_IMAGES_DIR is not in the environment.")
    def test_psfex(self) -> None:
        """Test that we can:

        - read a legacy PSFEX PSF with afw;
        - wrap it inthe new `LegacyPointSpreadFunction` class;
        - get consistent behavior from the two.

        This test is skipped if legacy modules cannot be imported.
        """
        try:
            from lsst.afw.image import ExposureFitsReader
            from lsst.meas.extensions.psfex import PsfexPsf
        except ImportError:
            raise unittest.SkipTest("'lsst.afw.image' could not be imported.") from None
        assert DATA_DIR is not None, "Guaranteed by decorator."
        filename = os.path.join(DATA_DIR, "extracted", "preliminary_visit_image.fits")
        reader = ExposureFitsReader(filename)
        legacy_psf = reader.readPsf()
        domain = Box.from_legacy(reader.readBBox())
        psf = PointSpreadFunction.from_legacy(legacy_psf, domain)
        self.assertIsInstance(psf, LegacyPointSpreadFunction)
        self.assertEqual(psf.domain, domain)
        self.assertIsInstance(psf.legacy_psf, PsfexPsf)
        self.compare_to_legacy(legacy_psf, psf)

    def compare_to_legacy(self, legacy_psf: Any, psf: PointSpreadFunction) -> None:
        from lsst.geom import Point2D

        for p in [Point2D(50.0, 60.0), Point2D(801.2, 322.8), Point2D(33.5, 22.1)]:
            self.assertEqual(psf.kernel_bbox, Box.from_legacy(legacy_psf.computeKernelBBox(p)))
            self.assertEqual(
                psf.compute_kernel_image(x=p.x, y=p.y), Image.from_legacy(legacy_psf.computeKernelImage(p))
            )
            self.assertEqual(
                psf.compute_stellar_bbox(x=p.x, y=p.y), Box.from_legacy(legacy_psf.computeImageBBox(p))
            )
            self.assertEqual(
                psf.compute_stellar_image(x=p.x, y=p.y), Image.from_legacy(legacy_psf.computeImage(p))
            )


if __name__ == "__main__":
    unittest.main()

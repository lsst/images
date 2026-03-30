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

import numpy as np

from lsst.images import Box
from lsst.images.psfs import (
    GaussianPointSpreadFunction,
    PiffWrapper,
    PointSpreadFunction,
    PSFExWrapper,
)
from lsst.images.tests import RoundtripFits, compare_psf_to_legacy

DATA_DIR = os.environ.get("TESTDATA_IMAGES_DIR", None)


class PointSpreadFunctionTestCase(unittest.TestCase):
    """Tests for the PointSpreadFunction classes."""

    def test_gaussian(self) -> None:
        """Test the built-in Gaussian PSF implementation."""
        bounds = Box.factory[-1024:1024, -2048:2048]
        psf = GaussianPointSpreadFunction(2.5, bounds=bounds, stamp_size=33)
        self.assertEqual(psf.bounds, bounds)

        kernel = psf.compute_kernel_image(x=5.0, y=3.0)
        self.assertEqual(kernel.bbox, psf.kernel_bbox)
        self.assertAlmostEqual(float(kernel.array.sum()), 1.0)
        center = kernel.array.shape[0] // 2
        self.assertEqual(np.unravel_index(np.argmax(kernel.array), kernel.array.shape), (center, center))

        stellar = psf.compute_stellar_image(x=5.25, y=3.75)
        self.assertEqual(stellar.bbox, psf.compute_stellar_bbox(x=5.25, y=3.75))
        self.assertAlmostEqual(float(stellar.array.sum()), 1.0)
        self.assertGreater(stellar.array[center - 1, center], stellar.array[center + 1, center])
        self.assertGreater(stellar.array[center, center], stellar.array[center, center - 1])
        self.assertGreater(stellar.array[center, center], stellar.array[center - 1, center])

        with RoundtripFits(self, psf) as roundtrip:
            self.assertEqual(roundtrip.result, psf, f"{roundtrip.result} != {psf}")

        with self.assertRaises(ValueError):
            # Even stamp size.
            GaussianPointSpreadFunction(2.5, bounds=bounds, stamp_size=32)

        with self.assertRaises(ValueError):
            # Negative stamp size.
            GaussianPointSpreadFunction(2.5, bounds=bounds, stamp_size=-33)

        with self.assertRaises(ValueError):
            # Negative sigma.
            GaussianPointSpreadFunction(-2.5, bounds=bounds, stamp_size=33)

    @unittest.skipUnless(DATA_DIR is not None, "TESTDATA_IMAGES_DIR is not in the environment.")
    def test_piff(self) -> None:
        """Test that we can:

        - read a legacy Piff PSF with afw;
        - convert it to the new `PiffWrapper` class;
        - get consistent behavior from the two;
        - round-trip the new PSF through a FITS archive;
        - still get consistent behavior with the round-tripped PSF.

        This test is skipped if legacy modules cannot be imported.
        """
        try:
            from piff import PSF

            from lsst.afw.image import ExposureFitsReader
        except ImportError:
            raise unittest.SkipTest("'lsst.afw.image' could not be imported.") from None
        assert DATA_DIR is not None, "Guaranteed by decorator."
        filename = os.path.join(DATA_DIR, "dp2", "legacy", "visit_image.fits")
        reader = ExposureFitsReader(filename)
        legacy_psf = reader.readPsf()
        bounds = Box.from_legacy(reader.readBBox())
        psf = PointSpreadFunction.from_legacy(legacy_psf, bounds)
        self.assertIsInstance(psf, PiffWrapper)
        self.assertEqual(psf.bounds, bounds)
        self.assertIsInstance(psf.piff_psf, PSF)
        compare_psf_to_legacy(self, psf, legacy_psf)
        with RoundtripFits(self, psf) as roundtrip:
            pass
        compare_psf_to_legacy(self, roundtrip.result, legacy_psf)

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
        filename = os.path.join(DATA_DIR, "dp2", "legacy", "preliminary_visit_image.fits")
        reader = ExposureFitsReader(filename)
        legacy_psf = reader.readPsf()
        bounds = Box.from_legacy(reader.readBBox())
        psf = PointSpreadFunction.from_legacy(legacy_psf, bounds)
        self.assertIsInstance(psf, PSFExWrapper)
        self.assertEqual(psf.bounds, bounds)
        self.assertIsInstance(psf.legacy_psf, PsfexPsf)
        compare_psf_to_legacy(self, psf, legacy_psf)
        compare_psf_to_legacy(self, psf, legacy_psf)
        with RoundtripFits(self, psf) as roundtrip:
            pass
        compare_psf_to_legacy(self, roundtrip.result, legacy_psf)


if __name__ == "__main__":
    unittest.main()

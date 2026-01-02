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
from typing import Any

from lsst.images import Box, Image
from lsst.images.fits import FitsInputArchive, FitsOutputArchive
from lsst.images.psfs import PointSpreadFunction
from lsst.images.psfs.piff import PiffSerializationModel, PiffWrapper
from lsst.images.psfs.psfex import PSFExSerializationModel, PSFExWrapper

DATA_DIR = os.environ.get("TESTDATA_IMAGES_DIR", None)


class PointSpreadFunctionTestCase(unittest.TestCase):
    """Tests for the PointSpreadFunction classes."""

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
        filename = os.path.join(DATA_DIR, "extracted", "visit_image.fits")
        reader = ExposureFitsReader(filename)
        legacy_psf = reader.readPsf()
        domain = Box.from_legacy(reader.readBBox())
        psf = PointSpreadFunction.from_legacy(legacy_psf, domain)
        self.assertIsInstance(psf, PiffWrapper)
        self.assertEqual(psf.domain, domain)
        self.assertIsInstance(psf.piff_psf, PSF)
        self.compare_to_legacy(legacy_psf, psf)
        with tempfile.NamedTemporaryFile(suffix=".fits", delete_on_close=False, delete=True) as tmp:
            tmp.close()
            with FitsOutputArchive.open(tmp.name) as output_archive:
                tree = output_archive.serialize_direct("psf", psf.serialize)
                output_archive.add_tree(tree)
            with FitsInputArchive.open(tmp.name) as input_archive:
                tree = input_archive.get_tree(PiffSerializationModel)
                roundtripped = PiffWrapper.deserialize(tree, input_archive)
        self.compare_to_legacy(legacy_psf, roundtripped)

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
        self.assertIsInstance(psf, PSFExWrapper)
        self.assertEqual(psf.domain, domain)
        self.assertIsInstance(psf.legacy_psf, PsfexPsf)
        self.compare_to_legacy(legacy_psf, psf)
        with tempfile.NamedTemporaryFile(suffix=".fits", delete_on_close=False, delete=True) as tmp:
            tmp.close()
            with FitsOutputArchive.open(tmp.name) as output_archive:
                tree = output_archive.serialize_direct("psf", psf.serialize)
                output_archive.add_tree(tree)
            with FitsInputArchive.open(tmp.name) as input_archive:
                tree = input_archive.get_tree(PSFExSerializationModel)
                roundtripped = PSFExWrapper.deserialize(tree, input_archive)
        self.compare_to_legacy(legacy_psf, roundtripped)

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

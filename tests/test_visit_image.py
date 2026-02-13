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

from lsst.images import Box, DetectorFrame, VisitImage, get_legacy_visit_image_mask_planes
from lsst.images.fits import ExtensionKey
from lsst.images.tests import (
    DP2_VISIT_DETECTOR_DATA_ID,
    compare_projection_to_legacy_wcs,
    compare_psf_to_legacy,
)

DATA_DIR = os.environ.get("TESTDATA_IMAGES_DIR", None)


class VisitImageTestCase(unittest.TestCase):
    """Tests for the VisitImage class and the basics of the archive system."""

    @unittest.skipUnless(DATA_DIR is not None, "TESTDATA_IMAGES_DIR is not in the environment.")
    def test_legacy_rewrite(self) -> None:
        """Test that we can read a ``lsst.afw.image.Exposure`` into an
        `lsst.images.VisitImage` and write that out while preserving even
        lossy-compressed pixel values exactly.
        """
        assert DATA_DIR is not None, "Guaranteed by decorator."
        filename = os.path.join(DATA_DIR, "dp2", "legacy", "visit_image.fits")
        from_afw = VisitImage.read_legacy(
            filename, preserve_quantization=True, plane_map=get_legacy_visit_image_mask_planes()
        )
        # Check that we read the units from BUNIT.
        self.assertEqual(from_afw.unit, astropy.units.nJy)
        # Check that the primary header has the keys we want, and none of the
        # keys we don't want.
        header = from_afw._opaque_metadata.headers[ExtensionKey()]
        self.assertEqual(header["LSST BUTLER DATAID INSTRUMENT"], DP2_VISIT_DETECTOR_DATA_ID["instrument"])
        self.assertEqual(header["LSST BUTLER DATAID DETECTOR"], DP2_VISIT_DETECTOR_DATA_ID["detector"])
        self.assertEqual(header["LSST BUTLER DATAID VISIT"], DP2_VISIT_DETECTOR_DATA_ID["visit"])
        self.assertNotIn("LSST BUTLER ID", header)
        self.assertNotIn("AR HDU", header)
        self.assertNotIn("A_ORDER", header)
        # Check that the extension HDUs do not have any custom headers.
        self.assertFalse(from_afw._opaque_metadata.headers[ExtensionKey("IMAGE")])
        self.assertFalse(from_afw._opaque_metadata.headers[ExtensionKey("MASK")])
        self.assertFalse(from_afw._opaque_metadata.headers[ExtensionKey("VARIANCE")])
        with tempfile.NamedTemporaryFile(suffix=".fits", delete_on_close=False, delete=True) as tmp:
            tmp.close()
            from_afw.write_fits(tmp.name)
            # Check that we're still using the right compression, and that
            # wrote WCSs.
            with astropy.io.fits.open(tmp.name, disable_image_compression=True) as fits:
                self.assertEqual(fits[1].header["ZCMPTYPE"], "RICE_1")
                self.assertEqual(fits[1].header["CTYPE1"], "RA---TAN-SIP")
                self.assertEqual(fits[2].header["ZCMPTYPE"], "GZIP_2")
                self.assertEqual(fits[2].header["CTYPE1"], "RA---TAN-SIP")
                self.assertEqual(fits[3].header["ZCMPTYPE"], "RICE_1")
                self.assertEqual(fits[3].header["CTYPE1"], "RA---TAN-SIP")
            roundtripped = VisitImage.read_fits(tmp.name)
        self.assertEqual(roundtripped.bbox, from_afw.bbox)
        self.assertEqual(roundtripped.unit, from_afw.unit)
        self.assertEqual(roundtripped.mask.schema, from_afw.mask.schema)
        np.testing.assert_array_equal(roundtripped.image.array, from_afw.image.array)
        np.testing.assert_array_equal(roundtripped.mask.array, from_afw.mask.array)
        np.testing.assert_array_equal(roundtripped.variance.array, from_afw.variance.array)
        # Check that the round-tripped headers are the same (up to card order).
        self.assertEqual(dict(header), dict(roundtripped._opaque_metadata.headers[ExtensionKey()]))
        self.assertFalse(roundtripped._opaque_metadata.headers[ExtensionKey("IMAGE")])
        self.assertFalse(roundtripped._opaque_metadata.headers[ExtensionKey("MASK")])
        self.assertFalse(roundtripped._opaque_metadata.headers[ExtensionKey("VARIANCE")])
        # Check various components on the roundtripped image against the
        # legacy versions.
        try:
            from lsst.afw.image import ExposureFitsReader
        except ImportError:
            raise unittest.SkipTest("'lsst.afw.image' could not be imported.") from None
        reader = ExposureFitsReader(filename)
        detector_bbox = Box.from_legacy(reader.readDetector().getBBox())
        compare_projection_to_legacy_wcs(
            self,
            roundtripped.projection,
            reader.readWcs(),
            DetectorFrame(**DP2_VISIT_DETECTOR_DATA_ID, bbox=detector_bbox),
            roundtripped.bbox,
        )
        self.assertIs(roundtripped.projection, roundtripped.mask.projection)
        self.assertIs(roundtripped.projection, roundtripped.variance.projection)
        compare_psf_to_legacy(self, roundtripped.psf, reader.readPsf())


if __name__ == "__main__":
    unittest.main()

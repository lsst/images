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
from typing import Any

import astropy.io.fits

from lsst.images import Box, VisitImage, get_legacy_visit_image_mask_planes
from lsst.images.fits import ExtensionKey
from lsst.images.tests import (
    DP2_VISIT_DETECTOR_DATA_ID,
    RoundtripFits,
    TemporaryButler,
    assert_masked_images_equal,
    compare_visit_image_to_legacy,
)

DATA_DIR = os.environ.get("TESTDATA_IMAGES_DIR", None)


@unittest.skipUnless(DATA_DIR is not None, "TESTDATA_IMAGES_DIR is not in the environment.")
class VisitImageTestCase(unittest.TestCase):
    """Tests for the VisitImage class and the basics of the archive system."""

    @classmethod
    def setUpClass(cls) -> None:
        assert DATA_DIR is not None, "Guaranteed by decorator."
        cls.filename = os.path.join(DATA_DIR, "dp2", "legacy", "visit_image.fits")
        cls.plane_map = plane_map = get_legacy_visit_image_mask_planes()
        cls.visit_image = VisitImage.read_legacy(
            cls.filename, preserve_quantization=True, plane_map=plane_map
        )
        cls.legacy_exposure: Any = None
        try:
            from lsst.afw.image import ExposureFitsReader

            cls.legacy_exposure = ExposureFitsReader(cls.filename).read()
        except ImportError:
            pass

    def test_read_legacy_headers(self) -> None:
        """Test that headers were correctly stripped and interpreted in
        `VisitImage.read_legacy`.
        """
        # Check that we read the units from BUNIT.
        self.assertEqual(self.visit_image.unit, astropy.units.nJy)
        # Check that the primary header has the keys we want, and none of the
        # keys we don't want.
        header = self.visit_image._opaque_metadata.headers[ExtensionKey()]
        self.assertIn("INSTRUME", header)
        self.assertNotIn("LSST BUTLER ID", header)
        self.assertNotIn("AR HDU", header)
        self.assertNotIn("A_ORDER", header)
        # Check that the extension HDUs do not have any custom headers.
        self.assertFalse(self.visit_image._opaque_metadata.headers[ExtensionKey("IMAGE")])
        self.assertFalse(self.visit_image._opaque_metadata.headers[ExtensionKey("MASK")])
        self.assertFalse(self.visit_image._opaque_metadata.headers[ExtensionKey("VARIANCE")])

    def test_rewrite(self) -> None:
        """Test that we can rewrite the visit image and preserve both
        lossy-compressed pixel values and components exactly.
        """
        with RoundtripFits(self, self.visit_image, "VisitImage") as roundtrip:
            # Check that we're still using the right compression, and that we
            # wrote WCSs.
            fits = roundtrip.inspect()
            self.assertEqual(fits[1].header["ZCMPTYPE"], "RICE_1")
            self.assertEqual(fits[1].header["CTYPE1"], "RA---TAN-SIP")
            self.assertEqual(fits[2].header["ZCMPTYPE"], "GZIP_2")
            self.assertEqual(fits[2].header["CTYPE1"], "RA---TAN-SIP")
            self.assertEqual(fits[3].header["ZCMPTYPE"], "RICE_1")
            self.assertEqual(fits[3].header["CTYPE1"], "RA---TAN-SIP")
            # Check a subimage read.
            subbox = Box.factory[8:13, 9:30]
            subimage = roundtrip.get(bbox=subbox)
            assert_masked_images_equal(self, subimage, self.visit_image[subbox], expect_view=False)
            alternates: dict[str, Any] = {}
            with self.subTest():
                self.assertEqual(roundtrip.get("bbox"), self.visit_image.bbox)
                alternates = {k: roundtrip.get(k) for k in ["projection", "image", "mask", "variance", "psf"]}
        assert_masked_images_equal(self, roundtrip.result, self.visit_image, expect_view=False)
        # Check that the round-tripped headers are the same (up to card order).
        self.assertEqual(
            dict(self.visit_image._opaque_metadata.headers[ExtensionKey()]),
            dict(roundtrip.result._opaque_metadata.headers[ExtensionKey()]),
        )
        self.assertFalse(roundtrip.result._opaque_metadata.headers[ExtensionKey("IMAGE")])
        self.assertFalse(roundtrip.result._opaque_metadata.headers[ExtensionKey("MASK")])
        self.assertFalse(roundtrip.result._opaque_metadata.headers[ExtensionKey("VARIANCE")])
        with self.subTest():
            if self.legacy_exposure is None:
                raise unittest.SkipTest("'lsst.afw.image' could not be imported.") from None
            compare_visit_image_to_legacy(
                self,
                roundtrip.result,
                self.legacy_exposure,
                expect_view=False,
                plane_map=self.plane_map,
                **DP2_VISIT_DETECTOR_DATA_ID,
                alternates=alternates,
            )
            # Check converting from the legacy object in-memory.
            compare_visit_image_to_legacy(
                self,
                VisitImage.from_legacy(self.legacy_exposure, plane_map=self.plane_map),
                self.legacy_exposure,
                expect_view=True,
                plane_map=self.plane_map,
                **DP2_VISIT_DETECTOR_DATA_ID,
            )

    def test_butler_converters(self) -> None:
        """Test that we can read a VisitImage and its components from a butler
        dataset written as an `lsst.afw.image.Exposure`.
        """
        if self.legacy_exposure is None:
            raise unittest.SkipTest("lsst.afw.image.afw could not be imported.")
        with TemporaryButler(legacy="ExposureF") as helper:
            from lsst.daf.butler import FileDataset

            helper.butler.ingest(FileDataset(path=self.filename, refs=[helper.legacy]), transfer="symlink")
            visit_image_ref = helper.legacy.overrideStorageClass("VisitImage")
            visit_image = helper.butler.get(visit_image_ref)
            bbox = helper.butler.get(visit_image_ref.makeComponentRef("bbox"))
            self.assertEqual(bbox, visit_image.bbox)
            alternates = {
                k: helper.butler.get(visit_image_ref.makeComponentRef(k))
                # TODO: including "projection" here fails because there's
                # code in daf_butler that expects any component to be valid
                # for the *internal* storage class, not the requested one,
                # and that's difficult to fix because it's tied up with the
                # data ID standardization logic.
                for k in ["image", "mask", "variance", "psf"]
            }
            compare_visit_image_to_legacy(
                self,
                visit_image,
                self.legacy_exposure,
                expect_view=False,
                plane_map=self.plane_map,
                alternates=alternates,
                **DP2_VISIT_DETECTOR_DATA_ID,
            )


if __name__ == "__main__":
    unittest.main()

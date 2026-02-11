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
import astropy.units as u
import numpy as np

from lsst.images import Box, Image, MaskedImage, MaskPlane, MaskSchema
from lsst.images.fits import ExtensionKey, FitsCompressionOptions
from lsst.images.tests import DP2_VISIT_DETECTOR_DATA_ID

DATA_DIR = os.environ.get("TESTDATA_IMAGES_DIR", None)


class MaskedImageTestCase(unittest.TestCase):
    """Tests for the MaskedImage class and the basics of the archive system."""

    def setUp(self) -> None:
        self.maxDiff = None
        self.rng = np.random.default_rng(500)
        self.masked_image = MaskedImage(
            Image(self.rng.normal(100.0, 8.0, size=(200, 251)), dtype=np.float64, unit=u.nJy, start=(5, 8)),
            mask_schema=MaskSchema(
                [
                    MaskPlane("BAD", "Pixel is very bad, possibly downright evil."),
                    MaskPlane("HUNGRY", "Pixel hasn't had enough to eat today."),
                ]
            ),
        )
        self.masked_image.mask.array |= np.multiply.outer(
            self.masked_image.image.array < 102.0,
            self.masked_image.mask.schema.bitmask("BAD"),
        )
        self.masked_image.mask.array |= np.multiply.outer(
            self.masked_image.image.array > 98.0,
            self.masked_image.mask.schema.bitmask("HUNGRY"),
        )
        self.masked_image.variance.array = self.rng.normal(64.0, 0.5, size=self.masked_image.bbox.shape)

    def test_construction(self) -> None:
        """Test that the MaskedImage construction (in setUp) worked."""
        self.assertEqual(self.masked_image.bbox, Box.factory[5:205, 8:259])
        self.assertEqual(self.masked_image.mask.bbox, self.masked_image.bbox)
        self.assertEqual(self.masked_image.variance.bbox, self.masked_image.bbox)
        self.assertEqual(self.masked_image.image.array.shape, self.masked_image.bbox.shape)
        self.assertEqual(self.masked_image.mask.array.shape, self.masked_image.bbox.shape + (1,))
        self.assertEqual(self.masked_image.variance.array.shape, self.masked_image.bbox.shape)
        self.assertEqual(self.masked_image.unit, u.nJy)
        self.assertEqual(self.masked_image.variance.unit, u.nJy**2)
        # The checks below are subject to the vagaries of the RNG, but we want
        # the seed to be such that they all pass, or other tests will be
        # weaker.
        self.assertGreater(
            np.sum(self.masked_image.mask.array == self.masked_image.mask.schema.bitmask("BAD")), 0
        )
        self.assertGreater(
            np.sum(self.masked_image.mask.array == self.masked_image.mask.schema.bitmask("HUNGRY")), 0
        )
        self.assertGreater(
            np.sum(self.masked_image.mask.array == self.masked_image.mask.schema.bitmask("BAD", "HUNGRY")), 0
        )

    def test_fits_roundtrip(self) -> None:
        """Test that we can round-trip the MaskedImage through FITS, including
        subimage reads.
        """
        subbox = Box.factory[11:20, 25:30]
        subslices = (slice(6, 15), slice(17, 22))
        np.testing.assert_array_equal(
            self.masked_image.image.array[subslices], self.masked_image.image[subbox].array
        )
        with tempfile.NamedTemporaryFile(suffix=".fits", delete_on_close=False, delete=True) as tmp:
            tmp.close()
            self.masked_image.write_fits(tmp.name)
            roundtripped = MaskedImage.read_fits(tmp.name)
            subimage = MaskedImage.read_fits(tmp.name, bbox=subbox)
            # Check that we used lossless compression (the default).
            with astropy.io.fits.open(tmp.name, disable_image_compression=True) as fits:
                self.assertEqual(fits[1].header["ZCMPTYPE"], "GZIP_2")
                self.assertEqual(fits[2].header["ZCMPTYPE"], "GZIP_2")
                self.assertEqual(fits[3].header["ZCMPTYPE"], "GZIP_2")
        self.assertEqual(roundtripped.bbox, self.masked_image.bbox)
        self.assertEqual(roundtripped.unit, self.masked_image.unit)
        self.assertEqual(roundtripped.mask.schema, self.masked_image.mask.schema)
        np.testing.assert_array_equal(roundtripped.image.array, self.masked_image.image.array)
        np.testing.assert_array_equal(roundtripped.mask.array, self.masked_image.mask.array)
        np.testing.assert_array_equal(roundtripped.variance.array, self.masked_image.variance.array)
        self.assertEqual(subimage.bbox, subbox)
        self.assertEqual(subimage.unit, self.masked_image.unit)
        self.assertEqual(subimage.mask.schema, self.masked_image.mask.schema)
        np.testing.assert_array_equal(subimage.image.array, self.masked_image.image.array[subslices])
        np.testing.assert_array_equal(subimage.mask.array, self.masked_image.mask.array[subslices])
        np.testing.assert_array_equal(subimage.variance.array, self.masked_image.variance.array[subslices])

    def test_fits_roundtrip_lossy(self) -> None:
        """Test that we can round-trip the MaskedImage through FITS, including
        subimage reads, with lossy compression.
        """
        subbox = Box.factory[11:20, 25:30]
        subslices = (slice(6, 15), slice(17, 22))
        np.testing.assert_array_equal(
            self.masked_image.image.array[subslices], self.masked_image.image[subbox].array
        )
        with tempfile.NamedTemporaryFile(suffix=".fits", delete_on_close=False, delete=True) as tmp:
            tmp.close()
            self.masked_image.write_fits(
                tmp.name,
                image_compression=FitsCompressionOptions.LOSSY,
                variance_compression=FitsCompressionOptions.LOSSY,
            )
            roundtripped = MaskedImage.read_fits(tmp.name)
            subimage = MaskedImage.read_fits(tmp.name, bbox=subbox)
            with astropy.io.fits.open(tmp.name, disable_image_compression=True) as fits:
                self.assertEqual(fits[1].header["ZCMPTYPE"], "RICE_1")
                self.assertEqual(fits[2].header["ZCMPTYPE"], "GZIP_2")
                self.assertEqual(fits[3].header["ZCMPTYPE"], "RICE_1")
        self.assertEqual(roundtripped.bbox, self.masked_image.bbox)
        self.assertEqual(roundtripped.unit, self.masked_image.unit)
        self.assertEqual(roundtripped.mask.schema, self.masked_image.mask.schema)
        np.testing.assert_allclose(roundtripped.image.array, self.masked_image.image.array, rtol=0.01)
        np.testing.assert_array_equal(roundtripped.mask.array, self.masked_image.mask.array)
        np.testing.assert_allclose(roundtripped.variance.array, self.masked_image.variance.array, rtol=0.01)
        self.assertEqual(subimage.bbox, subbox)
        self.assertEqual(subimage.unit, self.masked_image.unit)
        self.assertEqual(subimage.mask.schema, self.masked_image.mask.schema)
        np.testing.assert_array_equal(subimage.image.array, roundtripped.image.array[subslices])
        np.testing.assert_array_equal(subimage.mask.array, roundtripped.mask.array[subslices])
        np.testing.assert_array_equal(subimage.variance.array, roundtripped.variance.array[subslices])

    @unittest.skipUnless(DATA_DIR is not None, "TESTDATA_IMAGES_DIR is not in the environment.")
    def test_legacy_rewrite(self) -> None:
        """Test that we can read a ``lsst.afw.image.MaskedImage`` into an
        `lsst.images.MaskedImage` and write that out while preserving even
        lossy-compressed pixel values exactly.
        """
        assert DATA_DIR is not None, "Guaranteed by decorator."
        filename = os.path.join(DATA_DIR, "dp2", "legacy", "visit_image.fits")
        from_afw = MaskedImage.read_legacy(filename, preserve_quantization=True)
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
            # Check that we're still using the right compression.
            with astropy.io.fits.open(tmp.name, disable_image_compression=True) as fits:
                self.assertEqual(fits[1].header["ZCMPTYPE"], "RICE_1")
                self.assertEqual(fits[2].header["ZCMPTYPE"], "GZIP_2")
                self.assertEqual(fits[3].header["ZCMPTYPE"], "RICE_1")
            roundtripped = MaskedImage.read_fits(tmp.name)
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


if __name__ == "__main__":
    unittest.main()

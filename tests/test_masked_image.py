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
from astro_metadata_translator import ObservationInfo

from lsst.images import Box, Image, MaskedImage, MaskPlane, MaskSchema, get_legacy_visit_image_mask_planes
from lsst.images.fits import FitsCompressionOptions
from lsst.images.tests import RoundtripFits, RoundtripNdf, assert_masked_images_equal, compare_masked_image_to_legacy

DATA_DIR = os.environ.get("TESTDATA_IMAGES_DIR", None)


class MaskedImageTestCase(unittest.TestCase):
    """Tests for the MaskedImage class and the basics of the archive system."""

    def setUp(self) -> None:
        self.maxDiff = None
        self.rng = np.random.default_rng(500)
        self.obs_info = ObservationInfo(instrument="LSSTCam", detector_num=4)
        self.masked_image = MaskedImage(
            Image(self.rng.normal(100.0, 8.0, size=(200, 251)), dtype=np.float64, unit=u.nJy, start=(5, 8)),
            mask_schema=MaskSchema(
                [
                    MaskPlane("BAD", "Pixel is very bad, possibly downright evil."),
                    MaskPlane("HUNGRY", "Pixel hasn't had enough to eat today."),
                ]
            ),
            metadata={"fifty": "5 * 10"},
            obs_info=self.obs_info,
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
        self.assertEqual(self.masked_image.metadata, {"fifty": "5 * 10"})
        self.assertEqual(self.masked_image.obs_info.instrument, "LSSTCam")
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

        self.assertIs(self.masked_image[...], self.masked_image)
        self.assertEqual(
            str(self.masked_image), "MaskedImage(Image([y=5:205, x=8:259], float64), ['BAD', 'HUNGRY'])"
        )
        self.assertEqual(
            repr(self.masked_image),
            "MaskedImage(Image(..., bbox=Box(y=Interval(start=5, stop=205), x=Interval(start=8, stop=259)), "
            "dtype=dtype('float64')), mask_schema=MaskSchema([MaskPlane(name='BAD', description='Pixel is "
            "very bad, possibly downright evil.'), MaskPlane(name='HUNGRY', description=\"Pixel hasn't had "
            "enough to eat today.\")], dtype=dtype('uint8')))",
        )
        copy = self.masked_image.copy()
        original = self.masked_image.image.array[0, 0]
        copy.image.array[0, 0] = 38.0
        self.assertEqual(self.masked_image.image.array[0, 0], original)
        self.assertEqual(copy.image.array[0, 0], 38.0)

        # Test error conditions.
        with self.assertRaises(ValueError):
            # Disagreement over mask bbox.
            MaskedImage(Image(42.0, shape=(5, 6)), mask=self.masked_image.mask)
        with self.assertRaises(TypeError):
            # No mask definition.
            MaskedImage(self.masked_image.image, variance=self.masked_image.variance)
        with self.assertRaises(TypeError):
            # Can not provide mask and mask schema.
            MaskedImage(
                Image(42.0, shape=(5, 5)),
                mask=self.masked_image.mask,
                mask_schema=self.masked_image.mask.schema,
            )
        with self.assertRaises(ValueError):
            # image and variance bbox disagreement.
            MaskedImage(
                Image(42.0, shape=(5, 5)),
                mask_schema=self.masked_image.mask.schema,
                variance=self.masked_image.variance,
            )
        with self.assertRaises(ValueError):
            # no image unit but there is variance unit.
            MaskedImage(
                Image(42.0, shape=(5, 5)),
                mask_schema=self.masked_image.mask.schema,
                variance=Image(1.0, shape=(5, 5), unit=u.nJy),
            )
        with self.assertRaises(ValueError):
            # image and variance units disagree.
            MaskedImage(
                Image(42.0, shape=(5, 5), unit=u.nJy),
                mask_schema=self.masked_image.mask.schema,
                variance=Image(1.0, shape=(5, 5), unit=u.nJy),
            )

    def test_subset(self) -> None:
        """Test assignment of subset."""
        copy = self.masked_image.copy()
        subset = copy.local[0:10, 20:30].copy()
        subset.image[...] = Image(42.0, shape=(10, 10), unit=u.nJy)
        copy[subset.bbox] = subset
        self.assertEqual(copy.image.array[0, 20], 42.0)
        self.assertEqual(copy.image.array[0, 0], self.masked_image.image.array[0, 0])

    def test_fits_roundtrip(self) -> None:
        """Test that we can round-trip the MaskedImage through FITS, including
        subimage reads.
        """
        subbox = Box.factory[11:20, 25:30]
        subslices = (slice(6, 15), slice(17, 22))
        np.testing.assert_array_equal(
            self.masked_image.image.array[subslices], self.masked_image.image[subbox].array
        )
        with RoundtripFits(self, self.masked_image, "MaskedImageV2") as roundtrip:
            subimage = roundtrip.get(bbox=subbox)
            # Check that we used lossless compression (the default).
            fits = roundtrip.inspect()
            self.assertEqual(fits[1].header["ZCMPTYPE"], "GZIP_2")
            self.assertEqual(fits[2].header["ZCMPTYPE"], "GZIP_2")
            self.assertEqual(fits[3].header["ZCMPTYPE"], "GZIP_2")
        assert_masked_images_equal(self, roundtrip.result, self.masked_image, expect_view=False)
        assert_masked_images_equal(self, subimage, roundtrip.result[subbox], expect_view=False)

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
                compression_seed=50,
            )
            roundtripped = MaskedImage.read_fits(tmp.name)
            subimage = MaskedImage.read_fits(tmp.name, bbox=subbox)
            with astropy.io.fits.open(tmp.name, disable_image_compression=True) as fits:
                self.assertEqual(fits[1].header["ZCMPTYPE"], "RICE_1")
                self.assertEqual(fits[2].header["ZCMPTYPE"], "GZIP_2")
                self.assertEqual(fits[3].header["ZCMPTYPE"], "RICE_1")
        assert_masked_images_equal(self, roundtripped, self.masked_image, expect_view=False, rtol=0.01)
        assert_masked_images_equal(self, subimage, roundtripped[subbox], expect_view=False)

    def test_round_trip_ndf_compatible_mask(self):
        """NDF round-trip for the default-setup MaskedImage (2 planes ≤ 8)."""
        with RoundtripNdf(self, self.masked_image) as roundtrip:
            assert_masked_images_equal(
                self, roundtrip.result, self.masked_image, expect_view=False
            )

    def test_round_trip_ndf_incompatible_mask(self):
        """NDF round-trip for a >8-plane mask (forces 3D mask array, hoisted to MORE/LSST/MASK)."""
        rng = np.random.default_rng(7)
        planes = [MaskPlane(f"P{i}", f"plane {i}") for i in range(12)]
        wide = MaskedImage(
            Image(
                rng.normal(100.0, 8.0, size=(50, 60)),
                dtype=np.float64,
                unit=u.nJy,
                start=(0, 0),
            ),
            mask_schema=MaskSchema(planes),
            obs_info=self.obs_info,
        )
        wide.variance.array = rng.normal(64.0, 0.5, size=wide.bbox.shape)
        with RoundtripNdf(self, wide) as roundtrip:
            assert_masked_images_equal(self, roundtrip.result, wide, expect_view=False)

    def test_fits_ndf_consistency(self):
        """FITS and NDF backends produce equal MaskedImages on round-trip."""
        with RoundtripFits(self, self.masked_image) as fits_rt, \
             RoundtripNdf(self, self.masked_image) as ndf_rt:
            assert_masked_images_equal(self, self.masked_image, fits_rt.result, expect_view=False)
            assert_masked_images_equal(self, self.masked_image, ndf_rt.result, expect_view=False)
            assert_masked_images_equal(self, fits_rt.result, ndf_rt.result, expect_view=False)

    @unittest.skipUnless(DATA_DIR is not None, "TESTDATA_IMAGES_DIR is not in the environment.")
    def test_legacy(self) -> None:
        """Test MaskedImage.read_legacy, MaskedImage.to_legacy, and
        MaskedImage.from_legacy.
        """
        assert DATA_DIR is not None, "Guaranteed by decorator."
        filename = os.path.join(DATA_DIR, "dp2", "legacy", "visit_image.fits")
        plane_map = get_legacy_visit_image_mask_planes()
        masked_image = MaskedImage.read_legacy(filename, plane_map=plane_map)
        try:
            from lsst.afw.image import MaskedImageFitsReader
        except ImportError:
            raise unittest.SkipTest("'lsst.afw.image' could not be imported.") from None
        reader = MaskedImageFitsReader(filename)
        legacy_masked_image = reader.read()
        compare_masked_image_to_legacy(
            self, masked_image, legacy_masked_image, plane_map=plane_map, expect_view=False
        )
        compare_masked_image_to_legacy(
            self,
            masked_image,
            masked_image.to_legacy(plane_map=plane_map),
            plane_map=plane_map,
            expect_view=True,
        )
        compare_masked_image_to_legacy(
            self,
            MaskedImage.from_legacy(legacy_masked_image, plane_map=plane_map),
            legacy_masked_image,
            expect_view=True,
            plane_map=plane_map,
        )


if __name__ == "__main__":
    unittest.main()

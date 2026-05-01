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

import astropy.io.fits
import astropy.units as u
import numpy as np
from astro_metadata_translator import ObservationInfo

import lsst.utils.tests
from lsst.images import Box, DetectorFrame, Image
from lsst.images.tests import (
    RoundtripFits,
    RoundtripJson,
    RoundtripNdf,
    assert_close,
    assert_images_equal,
    assert_projections_equal,
    compare_image_to_legacy,
    make_random_projection,
)

DATA_DIR = os.environ.get("TESTDATA_IMAGES_DIR", None)


class ImageTestCase(unittest.TestCase):
    """Tests for the Image class."""

    def test_basics(self):
        """Test basic constructor patterns."""
        image = Image(42, shape=(5, 5), metadata={"three": 3})
        assert_close(self, image.array, np.zeros([5, 5], dtype=np.int64) + 42)
        self.assertEqual(image.metadata["three"], 3)

        data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
        image = Image(data)
        subset = image[Box.factory[:3, 1:3]]
        subset2 = image.absolute[:3, 1:3]
        assert_images_equal(self, subset2, subset, expect_view=True)

        assert_images_equal(self, image.copy(), image, expect_view=False)

        # Add an explicit bounding box and then slice it.
        image = Image(data, bbox=Box.factory[-2:1, 10:14])
        with self.assertRaises(IndexError):
            # Same slice no longer works in absolute slicing because we have
            # moved origin.
            image.absolute[:3, 1:3]
        # That slice does still work in local coordinates.
        assert_close(self, image.local[:3, 1:3].array, subset2.array)
        # And we can write an equivalent slice in absolute coordinates.
        assert_close(self, image.absolute[:0, 11:13].array, np.array([[2, 3], [6, 7]]))

        # Test __eq__ behavior.
        self.assertEqual(image[...], image)
        self.assertEqual(image.__eq__(data), NotImplemented)
        self.assertNotEqual(image, list(data))

        with self.assertRaises(ValueError):
            # bbox does not match array shape.
            Image(np.array([[1, 2, 3], [4, 5, 6]]), bbox=Box.factory[0:2, 0:4])

        with self.assertRaises(ValueError):
            # shape does not match array shape.
            Image(np.array([[2, 3, 4], [6, 7, 8]]), shape=[5, 2])

        with self.assertRaises(TypeError):
            # shape and bbox both None.
            Image()

        with self.assertRaises(ValueError):
            # Shape mismatch.
            Image(shape=[3, 6], bbox=Box.factory[-5:10, 0:10])

    def test_json_roundtrip(self) -> None:
        """Test saving a tiny image to pure JSON."""
        image = Image(
            np.arange(15).reshape(5, 3),
            start=(2, -1),
        )
        with RoundtripJson(self, image) as roundtrip:
            pass
        assert_images_equal(self, image, roundtrip.result)

    def test_fits_roundtrip(self) -> None:
        """Test saving a tiny image to FITS generically."""
        image = Image(
            np.arange(15).reshape(5, 3),
            start=(2, -1),
        )
        with RoundtripFits(self, image) as roundtrip:
            subbox = Box.factory[3:5, 0:1]
            assert_images_equal(self, image[subbox], roundtrip.get(bbox=subbox))
        assert_images_equal(self, image, roundtrip.result)

    def test_ndf_roundtrip(self) -> None:
        """Test saving a tiny image to NDF."""
        image = Image(
            np.arange(15).reshape(5, 3),
            start=(2, -1),
        )
        with RoundtripNdf(self, image) as roundtrip:
            pass
        assert_images_equal(self, image, roundtrip.result)

    def test_fits_ndf_consistency(self):
        """Writing via FITS and via NDF, then reading back, produces equal Images."""
        rng = np.random.default_rng(321)
        image = Image(
            rng.normal(100.0, 8.0, size=(60, 80)),
            dtype=np.float64,
            unit=u.nJy,
            start=(0, 0),
        )
        with RoundtripFits(self, image) as fits_rt, RoundtripNdf(self, image) as ndf_rt:
            assert_images_equal(self, image, fits_rt.result)
            assert_images_equal(self, image, ndf_rt.result)
            assert_images_equal(self, fits_rt.result, ndf_rt.result)

    def test_quantity(self):
        """Test quantities."""
        data = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])
        data2 = data.copy() * 2.0
        image = Image(data, unit=u.mJy, bbox=Box.factory[-2:1, 3:7])

        q = image.quantity
        self.assertEqual(q[1, 0], 5.0 * u.mJy)
        image.quantity = image.array * 10.0 * u.uJy
        q = image.quantity
        self.assertEqual(q[1, 0], 0.05 * u.mJy)

        image2 = Image(data2, unit=u.Jy)
        image[Box.factory[-1:0, 5:7]] = image2.local[1:2, 2:4]
        assert_close(
            self,
            image.array,
            np.array([[0.01, 0.02, 0.03, 0.04], [0.05, 0.06, 14000.0, 16000.0], [0.09, 0.1, 0.11, 0.12]]),
        )

    def test_read_write(self):
        """Round trip through file.

        This uses the read_fits and write_fits methods (which RoundtripFits
        does not use).
        """
        data = np.array([[1.0, 2.0, np.nan, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])
        md = {"int": 1, "float": 42.0, "bool": False, "long string header": "This is a string"}
        obsinfo = ObservationInfo(telescope="Simonyi", instrument="LSSTCam", relative_humidity=23.5)
        det_frame = DetectorFrame(instrument="Inst", visit=1234, detector=1, bbox=Box.factory[1:4096, 1:4096])
        rng = np.random.default_rng(500)
        projection = make_random_projection(rng, det_frame, Box.factory[1:4096, 1:4096])

        image = Image(
            data,
            unit=u.nJy,
            metadata=md,
            obs_info=obsinfo,
            bbox=Box.factory[-2:1, 3:7],
            projection=projection,
        )

        with lsst.utils.tests.getTempFilePath(".fits") as tmpFile:
            image.write_fits(tmpFile)

            new = Image.read_fits(tmpFile)
            self.assertEqual(new, image)

            # __eq__ does not test all components.
            self.assertEqual(new.obs_info, image.obs_info)
            self.assertEqual(new.metadata, image.metadata)
            self.maxDiff = None
            assert_projections_equal(self, new.projection, image.projection, expect_identity=False)

            # Read subset.
            subset = Image.read_fits(tmpFile, bbox=Box.factory[-2:0, 5:7])
            self.assertEqual(subset, image.absolute[-2:0, 5:7])
            self.assertEqual(subset, image.local[0:2, 2:4])
            self.assertEqual(str(subset), "Image([y=-2:0, x=5:7], float64)")
            self.assertEqual(
                repr(subset),
                "Image(..., bbox=Box(y=Interval(start=-2, stop=0), x=Interval(start=5, stop=7)), "
                "dtype=dtype('float64'))",
            )

            # Check that WCS headers were written out.
            with astropy.io.fits.open(tmpFile) as hdul:
                hdu1 = hdul[1]
                hdr1 = hdu1.header
                self.assertEqual(hdr1["CTYPE1"], "RA---TAN")

    @unittest.skipUnless(DATA_DIR is not None, "TESTDATA_IMAGES_DIR is not in the environment.")
    def test_legacy(self) -> None:
        """Test Image.read_legacy, Image.to_legacy, and Image.from_legacy."""
        assert DATA_DIR is not None, "Guaranteed by decorator."
        filename = os.path.join(DATA_DIR, "dp2", "legacy", "visit_image.fits")
        det_frame = DetectorFrame(instrument="Inst", visit=1234, detector=1, bbox=Box.factory[1:4096, 1:4096])
        image = Image.read_legacy(filename, preserve_quantization=True, fits_wcs_frame=det_frame)
        try:
            from lsst.afw.image import MaskedImageFitsReader
        except ImportError:
            raise unittest.SkipTest("'lsst.afw.image' could not be imported.") from None
        reader = MaskedImageFitsReader(filename)
        legacy_image = reader.readImage()
        compare_image_to_legacy(self, image, legacy_image, expect_view=False)
        # Converting back to afw will not share memory, because
        # preserve_quantization=True makes the array read-only and to_legacy
        # has to copy in that case.
        compare_image_to_legacy(self, image, image.to_legacy(), expect_view=False)
        # Converting from afw will always share memory.
        image_view = Image.from_legacy(legacy_image)
        compare_image_to_legacy(self, image_view, legacy_image, expect_view=True)
        # Converting back to afw from the in-memory view will be another view.
        compare_image_to_legacy(self, image_view, image_view.to_legacy(), expect_view=True)


if __name__ == "__main__":
    unittest.main()

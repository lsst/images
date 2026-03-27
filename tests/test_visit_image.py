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
import warnings
from typing import Any

import astropy.io.fits
import astropy.units as u
import astropy.wcs
import numpy as np
from astro_metadata_translator import ObservationInfo

from lsst.images import (
    Box,
    DetectorFrame,
    Image,
    MaskPlane,
    MaskSchema,
    ProjectionAstropyView,
    TractFrame,
    VisitImage,
    get_legacy_visit_image_mask_planes,
)
from lsst.images.fits import ExtensionKey, FitsOpaqueMetadata
from lsst.images.psfs import ConstantPointSpreadFunction, PointSpreadFunction
from lsst.images.tests import (
    DP2_VISIT_DETECTOR_DATA_ID,
    RoundtripFits,
    TemporaryButler,
    assert_masked_images_equal,
    assert_projections_equal,
    compare_visit_image_to_legacy,
    make_random_projection,
)

DATA_DIR = os.environ.get("TESTDATA_IMAGES_DIR", None)


class VisitImageTestCase(unittest.TestCase):
    """Basic Tests for VisitImage."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.rng = np.random.default_rng(500)
        det_frame = DetectorFrame(instrument="Inst", visit=1234, detector=1, bbox=Box.factory[1:4096, 1:4096])
        cls.projection = make_random_projection(cls.rng, det_frame, Box.factory[1:4096, 1:4096])
        cls.mask_schema = MaskSchema([MaskPlane("M1", "D1")])
        cls.obs_info = ObservationInfo(instrument="LSSTCam", detector_num=4)
        cls.constant_psf = ConstantPointSpreadFunction(
            constant=42.0, stamp_size=33, bounds=Box.factory[-10:10, -12:13]
        )

        opaque = FitsOpaqueMetadata()
        hdr = astropy.io.fits.Header()
        with warnings.catch_warnings():
            # Silence warnings about long keys becoming HIERARCH.
            warnings.simplefilter("ignore", category=astropy.io.fits.verify.VerifyWarning)
            hdr.update({"PLATFORM": "lsstcam", "LSST BUTLER ID": "123456789"})
        opaque.extract_legacy_primary_header(hdr)

        cls.image = Image(42, shape=(1024, 1024), unit=u.nJy)
        cls.variance = Image(5.0, shape=(1024, 1024), unit=u.nJy * u.nJy)
        # API signature suggests projection and obs_info can be None but they
        # are required.
        cls.visit_image = VisitImage(
            cls.image,
            variance=cls.variance,
            psf=ConstantPointSpreadFunction(constant=42.0, stamp_size=33, bounds=Box.factory[-10:10, -12:13]),
            mask_schema=cls.mask_schema,
            projection=cls.projection,
            obs_info=cls.obs_info,
        )
        cls.visit_image._opaque_metadata = opaque
        cls.simplest_visit_image = VisitImage(
            cls.image,
            psf=ConstantPointSpreadFunction(constant=42.0, stamp_size=33, bounds=Box.factory[-10:10, -12:13]),
            mask_schema=cls.mask_schema,
            projection=cls.projection,
            obs_info=cls.obs_info,
        )

    def test_basics(self) -> None:
        """Test basic constructor patterns."""
        # Test default fill of variance.
        visit = self.simplest_visit_image
        self.assertEqual(visit.variance.array[0, 0], 1.0)
        self.assertIs(visit[...], visit)
        self.assertEqual(str(visit), "VisitImage(Image([y=0:1024, x=0:1024], int64), ['M1'])")
        self.assertEqual(
            repr(visit),
            "VisitImage(Image(..., bbox=Box(y=Interval(start=0, stop=1024), x=Interval(start=0, stop=1024)),"
            " dtype=dtype('int64')), mask_schema=MaskSchema([MaskPlane(name='M1', description='D1')],"
            " dtype=dtype('uint8')))",
        )

        astropy_wcs = visit.astropy_wcs
        self.assertIsInstance(astropy_wcs, ProjectionAstropyView)
        approx_wcs = visit.fits_wcs
        self.assertIsInstance(approx_wcs, astropy.wcs.WCS)

        # Check that it is a deep copy.
        copy = visit.copy()
        copy.image.array[0, 0] = 30.0
        self.assertEqual(visit.image.array[0, 0], 42.0)
        self.assertEqual(copy.image.array[0, 0], 30.0)

        with self.assertRaises(TypeError):
            # Requires a PSF.
            VisitImage(
                self.image,
                mask_schema=self.mask_schema,
                projection=self.projection,
                obs_info=self.obs_info,
            )

        with self.assertRaises(TypeError):
            # Requires ObservationInfo.
            VisitImage(
                self.image,
                psf=self.constant_psf,
                mask_schema=self.mask_schema,
                projection=self.projection,
            )

        with self.assertRaises(TypeError):
            # Requires a projection.
            VisitImage(
                self.image,
                psf=self.constant_psf,
                mask_schema=self.mask_schema,
                obs_info=self.obs_info,
            )

        with self.assertRaises(TypeError):
            # Requires some form of mask.
            VisitImage(
                self.image,
                psf=self.constant_psf,
                projection=self.projection,
                obs_info=self.obs_info,
            )

        with self.assertRaises(TypeError):
            VisitImage(
                Image(42, shape=(5, 5)),
                psf=self.constant_psf,
                mask_schema=self.mask_schema,
                projection=self.projection,
                obs_info=self.obs_info,
            )

        # Requires a DetectorFrame.
        tract_frame = TractFrame(skymap="Skymap", tract=1, bbox=Box.factory[1:10, 1:10])
        tract_proj = make_random_projection(self.rng, tract_frame, Box.factory[1:4096, 1:4096])
        with self.assertRaises(TypeError):
            VisitImage(
                self.image,
                projection=tract_proj,
                psf=self.constant_psf,
                mask_schema=self.mask_schema,
                obs_info=self.obs_info,
            )

        # Variance unit mismatch.
        with self.assertRaises(ValueError):
            VisitImage(
                self.image,
                variance=self.image,
                psf=self.constant_psf,
                mask_schema=self.mask_schema,
                projection=self.projection,
                obs_info=self.obs_info,
            )

    def test_obs_info(self) -> None:
        """Check that ObservationInfo has been constructured."""
        visit = self.visit_image
        self.assertIsNotNone(visit.obs_info)
        self.maxDiff = None
        assert visit.obs_info is not None  # for mypy.
        self.assertEqual(visit.obs_info.instrument, "LSSTCam")

    def test_read_write(self) -> None:
        """Test that a visit can round trip through a FITS file."""
        with RoundtripFits(self, self.visit_image, "VisitImage") as roundtrip:
            # Check that we're still using the right compression, and that we
            # wrote WCSs.
            fits = roundtrip.inspect()
            self.assertEqual(fits[1].header["ZCMPTYPE"], "GZIP_2")
            self.assertEqual(fits[1].header["CTYPE1"], "RA---TAN")
            self.assertEqual(fits[2].header["ZCMPTYPE"], "GZIP_2")
            self.assertEqual(fits[2].header["CTYPE1"], "RA---TAN")
            self.assertEqual(fits[3].header["ZCMPTYPE"], "GZIP_2")
            self.assertEqual(fits[3].header["CTYPE1"], "RA---TAN")
            # Check a subimage read.
            subbox = Box.factory[8:13, 9:30]
            subimage = roundtrip.get(bbox=subbox)
            assert_masked_images_equal(self, subimage, self.visit_image[subbox], expect_view=False)
            with self.subTest():
                self.assertEqual(roundtrip.get("bbox"), self.visit_image.bbox)
        assert_masked_images_equal(self, roundtrip.result, self.visit_image, expect_view=False)
        # Check that the round-tripped headers are the same (up to card order).
        self.assertEqual(len(roundtrip.result._opaque_metadata.headers[ExtensionKey()]), 1)
        self.assertEqual(
            dict(self.visit_image._opaque_metadata.headers[ExtensionKey()]),
            dict(roundtrip.result._opaque_metadata.headers[ExtensionKey()]),
        )
        self.assertFalse(roundtrip.result._opaque_metadata.headers[ExtensionKey("IMAGE")])
        self.assertFalse(roundtrip.result._opaque_metadata.headers[ExtensionKey("MASK")])
        self.assertFalse(roundtrip.result._opaque_metadata.headers[ExtensionKey("VARIANCE")])
        self.assertEqual(roundtrip.result.obs_info, self.visit_image.obs_info)


@unittest.skipUnless(DATA_DIR is not None, "TESTDATA_IMAGES_DIR is not in the environment.")
class VisitImageLegacyTestCase(unittest.TestCase):
    """Tests for the VisitImage class and the basics of the archive system.

    Requires legacy code.
    """

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

    def test_legacy_errors(self) -> None:
        """Legacy read failure modes."""
        with self.assertRaises(ValueError):
            VisitImage.from_legacy(self.legacy_exposure, instrument="HSC")
        with self.assertRaises(ValueError):
            VisitImage.from_legacy(self.legacy_exposure, visit=123456)
        with self.assertRaises(ValueError):
            VisitImage.from_legacy(self.legacy_exposure, unit=u.mJy)
        visit = VisitImage.from_legacy(
            self.legacy_exposure, instrument="LSSTCam", unit=u.nJy, visit=2025052000177
        )
        self.assertEqual(visit.unit, u.nJy)

        with self.assertRaises(ValueError):
            VisitImage.read_legacy(self.filename, instrument="HSC")
        with self.assertRaises(ValueError):
            VisitImage.read_legacy(self.filename, visit=123456)

    def test_component_reads(self) -> None:
        """Test reads of components from legacy file."""
        visit = VisitImage.read_legacy(self.filename)
        proj = VisitImage.read_legacy(self.filename, component="projection")
        assert_projections_equal(self, proj, visit.projection, expect_identity=False)
        image = VisitImage.read_legacy(self.filename, component="image")
        self.assertEqual(image, visit.image)
        variance = VisitImage.read_legacy(self.filename, component="variance")
        self.assertEqual(variance, visit.variance)
        mask = VisitImage.read_legacy(self.filename, component="mask")
        self.assertEqual(mask, visit.mask)
        psf = VisitImage.read_legacy(self.filename, component="psf")
        self.assertIsInstance(psf, PointSpreadFunction)

    def test_obs_info(self) -> None:
        """Check that ObservationInfo has been constructed."""
        legacy = VisitImage.from_legacy(self.legacy_exposure, plane_map=self.plane_map)
        self.assertIsNotNone(legacy.obs_info)
        self.maxDiff = None
        self.assertEqual(legacy.obs_info, self.visit_image.obs_info)
        assert legacy.obs_info is not None  # for mypy.
        self.assertEqual(legacy.obs_info.instrument, "LSSTCam")

    def test_read_legacy_headers(self) -> None:
        """Test that headers were correctly stripped and interpreted in
        `VisitImage.read_legacy`.
        """
        # Check that we read the units from BUNIT.
        self.assertEqual(self.visit_image.unit, astropy.units.nJy)
        # Check that the primary header has the keys we want, and none of the
        # keys we don't want.
        header = self.visit_image._opaque_metadata.headers[ExtensionKey()]
        self.assertIn("EXPTIME", header)
        self.assertEqual(header["PLATFORM"], "lsstcam")
        self.assertNotIn("LSST BUTLER ID", header)
        self.assertNotIn("AR HDU", header)
        self.assertNotIn("A_ORDER", header)
        # Check that the extension HDUs do not have any custom headers.
        self.assertFalse(self.visit_image._opaque_metadata.headers[ExtensionKey("IMAGE")])
        self.assertFalse(self.visit_image._opaque_metadata.headers[ExtensionKey("MASK")])
        self.assertFalse(self.visit_image._opaque_metadata.headers[ExtensionKey("VARIANCE")])

    def test_from_legacy_headers(self) -> None:
        """Test that from_legacy handles headers properly."""
        legacy = VisitImage.from_legacy(self.legacy_exposure, plane_map=self.plane_map)
        header = legacy._opaque_metadata.headers[ExtensionKey()]
        self.assertIn("EXPTIME", header)
        self.assertEqual(header["PLATFORM"], "lsstcam")
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
        self.assertEqual(roundtrip.result._opaque_metadata.headers[ExtensionKey()]["PLATFORM"], "lsstcam")
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

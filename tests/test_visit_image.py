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
    BackgroundMap,
    Box,
    DetectorFrame,
    Image,
    MaskPlane,
    MaskSchema,
    ObservationSummaryStats,
    Polygon,
    ProjectionAstropyView,
    TractFrame,
    VisitImage,
    get_legacy_visit_image_mask_planes,
)
from lsst.images.aperture_corrections import ApertureCorrectionMap, aperture_corrections_to_legacy
from lsst.images.cameras import Detector
from lsst.images.fields import ChebyshevField, field_from_legacy_photo_calib
from lsst.images.fits import ExtensionKey, FitsOpaqueMetadata
from lsst.images.json import read as read_json
from lsst.images.psfs import GaussianPointSpreadFunction, PointSpreadFunction
from lsst.images.tests import (
    DP2_VISIT_DETECTOR_DATA_ID,
    RoundtripFits,
    RoundtripNdf,
    TemporaryButler,
    assert_close,
    assert_masked_images_equal,
    assert_projections_equal,
    compare_aperture_corrections_to_legacy,
    compare_detector_to_legacy,
    compare_photo_calib_to_legacy,
    compare_visit_image_to_legacy,
    make_random_projection,
)

try:
    import h5py  # noqa: F401

    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False

EXTERNAL_DATA_DIR = os.environ.get("TESTDATA_IMAGES_DIR", None)
LOCAL_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


class VisitImageTestCase(unittest.TestCase):
    """Basic Tests for VisitImage."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.rng = np.random.default_rng(500)
        det_frame = DetectorFrame(instrument="Inst", visit=1234, detector=1, bbox=Box.factory[1:4096, 1:4096])
        cls.mask_schema = MaskSchema([MaskPlane("M1", "D1")])
        cls.obs_info = ObservationInfo(instrument="LSSTCam", detector_num=4)
        cls.summary_stats = ObservationSummaryStats(psfSigma=2.5, zeroPoint=31.4)
        cls.gaussian_psf = GaussianPointSpreadFunction(2.5, stamp_size=33, bounds=Box.factory[-10:10, -12:13])
        cls.aperture_corrections: ApertureCorrectionMap = {
            "flux1": ChebyshevField(det_frame.bbox, np.array([0.75])),
            "flux2": ChebyshevField(det_frame.bbox, np.array([0.625])),
        }
        cls.detector, _, _ = read_json(Detector, os.path.join(LOCAL_DATA_DIR, "detector.json"))

        opaque = FitsOpaqueMetadata()
        hdr = astropy.io.fits.Header()
        with warnings.catch_warnings():
            # Silence warnings about long keys becoming HIERARCH.
            warnings.simplefilter("ignore", category=astropy.io.fits.verify.VerifyWarning)
            hdr.update({"PLATFORM": "lsstcam", "LSST BUTLER ID": "123456789"})
        opaque.extract_legacy_primary_header(hdr)

        cls.image = Image(42, shape=(1024, 1024), unit=u.nJy)
        cls.variance = Image(5.0, shape=(1024, 1024), unit=u.nJy * u.nJy)
        # polygon is the lower triangle of the image.
        cls.polygon = Polygon(x_vertices=[-0.5, 1023.5, -0.5], y_vertices=[-0.5, -0.5, 1023.5])
        cls.projection = make_random_projection(cls.rng, det_frame, Box.factory[1:4096, 1:4096])
        # API signature suggests projection and obs_info can be None but they
        # are required (unless you pass them in via the image plane).
        cls.visit_image = VisitImage(
            cls.image,
            variance=cls.variance,
            psf=GaussianPointSpreadFunction(2.5, stamp_size=33, bounds=Box.factory[-10:10, -12:13]),
            mask_schema=cls.mask_schema,
            projection=cls.projection,
            obs_info=cls.obs_info,
            summary_stats=cls.summary_stats,
            detector=cls.detector,
            bounds=cls.polygon,
            aperture_corrections=cls.aperture_corrections,
        )
        cls.visit_image.backgrounds.add(
            "standard",
            ChebyshevField(det_frame.bbox, np.array([[2.0]])),
            description="Background subtracted from the image.",
            is_subtracted=True,
        )
        cls.visit_image._opaque_metadata = opaque
        cls.simplest_visit_image = VisitImage(
            cls.image,
            psf=GaussianPointSpreadFunction(2.5, stamp_size=33, bounds=Box.factory[-10:10, -12:13]),
            mask_schema=cls.mask_schema,
            projection=cls.projection,
            detector=cls.detector,
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

        with self.assertRaises(TypeError):
            # Requires a PSF.
            VisitImage(
                self.image,
                mask_schema=self.mask_schema,
                projection=self.projection,
                obs_info=self.obs_info,
                detector=self.detector,
            )

        with self.assertRaises(TypeError):
            # Requires ObservationInfo.
            VisitImage(
                self.image,
                psf=self.gaussian_psf,
                mask_schema=self.mask_schema,
                projection=self.projection,
                detector=self.detector,
            )

        with self.assertRaises(TypeError):
            # Requires a projection.
            VisitImage(
                self.image,
                psf=self.gaussian_psf,
                mask_schema=self.mask_schema,
                obs_info=self.obs_info,
                detector=self.detector,
            )

        with self.assertRaises(TypeError):
            # Requires a detector.
            VisitImage(
                self.image,
                psf=self.gaussian_psf,
                mask_schema=self.mask_schema,
                projection=self.projection,
                obs_info=self.obs_info,
            )

        with self.assertRaises(TypeError):
            # Requires some form of mask.
            VisitImage(
                self.image,
                psf=self.gaussian_psf,
                projection=self.projection,
                obs_info=self.obs_info,
                detector=self.detector,
            )

        with self.assertRaises(TypeError):
            VisitImage(
                Image(42, shape=(5, 5)),
                psf=self.gaussian_psf,
                mask_schema=self.mask_schema,
                projection=self.projection,
                obs_info=self.obs_info,
                detector=self.detector,
            )

        # Requires a DetectorFrame.
        tract_frame = TractFrame(skymap="Skymap", tract=1, bbox=Box.factory[1:10, 1:10])
        tract_proj = make_random_projection(self.rng, tract_frame, Box.factory[1:4096, 1:4096])
        with self.assertRaises(TypeError):
            VisitImage(
                self.image,
                projection=tract_proj,
                psf=self.gaussian_psf,
                mask_schema=self.mask_schema,
                obs_info=self.obs_info,
                detector=self.detector,
            )

        # Variance unit mismatch.
        with self.assertRaises(ValueError):
            VisitImage(
                self.image,
                variance=self.image,
                psf=self.gaussian_psf,
                mask_schema=self.mask_schema,
                projection=self.projection,
                obs_info=self.obs_info,
                detector=self.detector,
            )

    def test_copy_and_slice(self) -> None:
        """Test that arrays and components are copied (when not immutable) by
        'copy' and referenced by 'slice'.
        """
        visit = self.visit_image
        copy = visit.copy()
        copy.image.array[0, 0] = 30.0
        self.assertEqual(visit.image.array[0, 0], 42.0)
        self.assertEqual(copy.image.array[0, 0], 30.0)
        subvisit = visit[Box.factory[0:5, 0:5]]
        # Check summary stats.
        self.assertEqual(copy.summary_stats, visit.summary_stats)
        self.assertIsNot(copy.summary_stats, visit.summary_stats)
        self.assertEqual(subvisit.summary_stats, visit.summary_stats)
        self.assertIs(subvisit.summary_stats, visit.summary_stats)
        # Check aperture corrections.
        self.assertEqual(copy.aperture_corrections.keys(), visit.aperture_corrections.keys())
        self.assertIsNot(copy.aperture_corrections, visit.aperture_corrections)
        self.assertEqual(subvisit.aperture_corrections.keys(), visit.aperture_corrections.keys())
        self.assertIs(subvisit.aperture_corrections, visit.aperture_corrections)
        # Check backgrounds.
        self.assertEqual(copy.backgrounds.keys(), visit.backgrounds.keys())
        self.assertIsNot(copy.backgrounds, visit.backgrounds)
        self.assertEqual(subvisit.backgrounds.keys(), visit.backgrounds.keys())
        self.assertIs(subvisit.backgrounds, visit.backgrounds)
        # Check bounds.
        self.assertIs(copy.bounds, self.polygon)
        self.assertEqual(subvisit.bounds, subvisit.bbox)  # original polygon wholly encloses subvisit.bbox

    def test_obs_info(self) -> None:
        """Check that ObservationInfo has been constructed."""
        visit = self.visit_image
        self.assertIsNotNone(visit.obs_info)
        self.maxDiff = None
        assert visit.obs_info is not None  # for mypy.
        self.assertEqual(visit.obs_info.instrument, "LSSTCam")

    def test_summary_stats(self) -> None:
        """Test the comparisons and attributes of ObservationSummaryStats."""
        self.assertEqual(self.summary_stats, ObservationSummaryStats(psfSigma=2.5, zeroPoint=31.4))
        self.assertNotEqual(self.summary_stats, ObservationSummaryStats(psfSigma=2.5))
        self.assertNotEqual(
            self.summary_stats, ObservationSummaryStats(psfSigma=2.5, raCorners=(5.2, 5.4, 5.4, 5.2))
        )

    @unittest.skipUnless(HAVE_H5PY, "h5py is not installed")
    def test_round_trip_ndf(self):
        """NDF round-trip for VisitImage."""
        with RoundtripNdf(self, self.visit_image) as roundtrip:
            assert_masked_images_equal(self, roundtrip.result, self.visit_image, expect_view=False)
            self.assertEqual(roundtrip.result.summary_stats, self.visit_image.summary_stats)
            self.assertEqual(type(roundtrip.result.psf), type(self.visit_image.psf))

    @unittest.skipUnless(HAVE_H5PY, "h5py is not installed")
    def test_fits_ndf_consistency(self):
        """FITS and NDF backends produce equal VisitImages on round-trip."""
        with RoundtripFits(self, self.visit_image) as fits_rt, RoundtripNdf(self, self.visit_image) as ndf_rt:
            assert_masked_images_equal(self, self.visit_image, fits_rt.result, expect_view=False)
            assert_masked_images_equal(self, self.visit_image, ndf_rt.result, expect_view=False)
            assert_masked_images_equal(self, fits_rt.result, ndf_rt.result, expect_view=False)

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
            with self.subTest():
                obs_info = roundtrip.get("obs_info")
                self.assertIsInstance(obs_info, ObservationInfo)
                self.assertEqual(obs_info, self.visit_image.obs_info)
            with self.subTest():
                summary_stats = roundtrip.get("summary_stats")
                self.assertIsInstance(summary_stats, ObservationSummaryStats)
                self.assertEqual(summary_stats, self.visit_image.summary_stats)
            with self.subTest():
                psf = roundtrip.get("psf")
                self.assertIsInstance(psf, GaussianPointSpreadFunction)
                self.assertEqual(psf.kernel_bbox, self.gaussian_psf.kernel_bbox)
            with self.subTest():
                backgrounds = roundtrip.get("backgrounds")
                self.assertIsInstance(backgrounds, BackgroundMap)
                self.assertEqual(backgrounds.keys(), {"standard"})
                self.assertIsInstance(backgrounds["standard"].field, ChebyshevField)
                self.assertEqual(backgrounds.subtracted.name, "standard")
                self.assertEqual(
                    roundtrip.result.backgrounds.subtracted.description,
                    "Background subtracted from the image.",
                )

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
        self.assertIsNotNone(roundtrip.result.summary_stats)
        self.assertEqual(
            roundtrip.result.summary_stats.psfSigma,
            self.visit_image.summary_stats.psfSigma,
        )
        self.assertEqual(
            roundtrip.result.summary_stats.zeroPoint,
            self.visit_image.summary_stats.zeroPoint,
        )
        self.assertEqual(roundtrip.result.bounds, self.polygon)
        self.assertIsInstance(roundtrip.result.backgrounds, BackgroundMap)
        self.assertEqual(roundtrip.result.backgrounds.keys(), {"standard"})
        self.assertIsInstance(roundtrip.result.backgrounds["standard"].field, ChebyshevField)
        self.assertEqual(roundtrip.result.backgrounds.subtracted.name, "standard")
        self.assertEqual(
            roundtrip.result.backgrounds.subtracted.description, "Background subtracted from the image."
        )


class VisitImageLegacyTestMixin:
    """Tests for the VisitImage class and the basics of the archive, to be
    specialized for a particular test image.

    `setUp` or `setUpClass` must be implemented to set the attributes declared
    in the class.
    """

    filename: str
    legacy_exposure: Any
    plane_map: dict[str, MaskPlane]
    visit_image: VisitImage
    unit: u.UnitBase

    def test_legacy_errors(self) -> None:
        """Legacy read failure modes."""
        with self.assertRaises(ValueError):
            VisitImage.from_legacy(self.legacy_exposure, instrument="HSC")
        with self.assertRaises(ValueError):
            VisitImage.from_legacy(self.legacy_exposure, visit=123456)
        with self.assertRaises(ValueError):
            VisitImage.from_legacy(self.legacy_exposure, unit=u.mJy)
        visit = VisitImage.from_legacy(
            self.legacy_exposure, instrument="LSSTCam", unit=self.unit, visit=2025052000177
        )
        self.assertEqual(visit.unit, self.unit)

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
        assert_projections_equal(self, proj, image.projection, expect_identity=False)
        variance = VisitImage.read_legacy(self.filename, component="variance")
        self.assertEqual(variance, visit.variance)
        assert_projections_equal(self, proj, variance.projection, expect_identity=False)
        mask = VisitImage.read_legacy(self.filename, component="mask")
        self.assertEqual(mask, visit.mask)
        assert_projections_equal(self, proj, mask.projection, expect_identity=False)
        psf = VisitImage.read_legacy(self.filename, component="psf")
        self.assertIsInstance(psf, PointSpreadFunction)
        obs_info = VisitImage.read_legacy(self.filename, component="obs_info")
        self.check_legacy_obs_info(obs_info)
        summary_stats = VisitImage.read_legacy(self.filename, component="summary_stats")
        self.assertIsInstance(summary_stats, ObservationSummaryStats)
        self.assertEqual(summary_stats.nPsfStar, self.legacy_exposure.info.getSummaryStats().nPsfStar)
        compare_aperture_corrections_to_legacy(
            self,
            VisitImage.read_legacy(self.filename, component="aperture_corrections"),
            self.legacy_exposure.info.getApCorrMap(),
            visit.bbox,
        )
        detector = VisitImage.read_legacy(self.filename, component="detector")
        compare_detector_to_legacy(self, detector, self.legacy_exposure.getDetector(), is_raw_assembled=True)
        photometric_scaling = VisitImage.read_legacy(self.filename, component="photometric_scaling")
        compare_photo_calib_to_legacy(
            self,
            photometric_scaling,
            self.legacy_exposure.getPhotoCalib(),
            subimage_bbox=visit.bbox,
        )

    def check_legacy_obs_info(self, obs_info: ObservationInfo | None) -> None:
        """Check that an `ObservationInfo` instance is not `None`, and that it
        matches the one in the legacy test data file.
        """
        self.assertIsInstance(obs_info, ObservationInfo)
        self.assertEqual(obs_info.instrument, "LSSTCam")
        self.assertEqual(obs_info.detector_num, 85, obs_info)
        self.assertEqual(obs_info.detector_unique_name, "R21_S11", obs_info)
        self.assertEqual(obs_info.physical_filter, "r_57", obs_info)

    def test_obs_info(self) -> None:
        """Check that ObservationInfo has been constructed."""
        legacy = VisitImage.from_legacy(self.legacy_exposure, plane_map=self.plane_map)
        self.assertIsNotNone(legacy.obs_info)
        self.maxDiff = None
        self.assertEqual(legacy.obs_info, self.visit_image.obs_info)
        assert legacy.obs_info is not None  # for mypy.
        self.assertEqual(legacy.obs_info.instrument, "LSSTCam")
        self.assertEqual(legacy.obs_info.detector_num, 85, legacy.obs_info)
        self.assertEqual(legacy.obs_info.detector_unique_name, "R21_S11", legacy.obs_info)
        self.assertEqual(legacy.obs_info.physical_filter, "r_57", legacy.obs_info)

    def test_aperture_corrections_to_legacy(self) -> None:
        """Test that we can convert an aperture correction map back to a
        legacy `lsst.afw.image.ApCorrMap`.
        """
        legacy_ap_corr_map = aperture_corrections_to_legacy(self.visit_image.aperture_corrections)
        compare_aperture_corrections_to_legacy(
            self, self.visit_image.aperture_corrections, legacy_ap_corr_map, self.visit_image.bbox
        )

    def test_read_legacy_headers(self) -> None:
        """Test that headers were correctly stripped and interpreted in
        `VisitImage.read_legacy`.
        """
        # Check that we read the units from BUNIT.
        self.assertEqual(self.visit_image.unit, self.unit)
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
                alternates = {
                    k: roundtrip.get(k)
                    for k in [
                        "projection",
                        "image",
                        "mask",
                        "variance",
                        "psf",
                        "obs_info",
                        "summary_stats",
                        "aperture_corrections",
                        "detector",
                        "photometric_scaling",
                    ]
                }
            # Test reading back in as an Exposure.
            with self.subTest():
                legacy_exposure = roundtrip.get(storageClass="Exposure")
                self.assertIsInstance(legacy_exposure, lsst.afw.image.Exposure)
            # Try to do a butler get of a component with storage class
            # override.
            with self.subTest():
                import lsst.afw.image

                # We have VisitInfo available.
                visit_info = roundtrip.get("obs_info", storageClass="VisitInfo")
                self.assertIsInstance(visit_info, lsst.afw.image.VisitInfo)
                self.assertEqual(visit_info.getInstrumentLabel(), "LSSTCam")

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
            with warnings.catch_warnings():
                # Silence warnings about data ID and filter label disagreeing.
                warnings.simplefilter("ignore", category=UserWarning)
                visit_image = helper.butler.get(visit_image_ref)
            bbox = helper.butler.get(visit_image_ref.makeComponentRef("bbox"))
            self.assertEqual(bbox, visit_image.bbox)
            alternates = {
                k: helper.butler.get(visit_image_ref.makeComponentRef(k))
                # TODO: including "projection" or "obs_info" here fails because
                # there's code in daf_butler that expects any component to be
                # valid for the *internal* storage class, not the requested
                # one, and that's difficult to fix because it's tied up with
                # the data ID standardization logic.
                for k in ["image", "mask", "variance", "psf", "detector"]
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


@unittest.skipUnless(EXTERNAL_DATA_DIR is not None, "TESTDATA_IMAGES_DIR is not in the environment.")
class VisitImageLegacyTestCase(unittest.TestCase, VisitImageLegacyTestMixin):
    """Tests for the VisitImage class using a DRP-final visit_image dataset.

    Requires legacy code.
    """

    @classmethod
    def setUpClass(cls) -> None:
        assert EXTERNAL_DATA_DIR is not None, "Guaranteed by decorator."
        cls.filename = os.path.join(EXTERNAL_DATA_DIR, "dp2", "legacy", "visit_image.fits")
        try:
            from lsst.afw.image import ExposureFitsReader

            cls.legacy_exposure = ExposureFitsReader(cls.filename).read()
        except ImportError:
            raise unittest.SkipTest("afw not available; cannot read legacy visit images") from None
        cls.plane_map = get_legacy_visit_image_mask_planes()
        cls.visit_image = VisitImage.read_legacy(
            cls.filename, preserve_quantization=True, plane_map=cls.plane_map
        )
        cls.unit = u.nJy

    def test_convert_unit(self) -> None:
        """Test using the ``photometric_scaling`` to swap between
        calibrated and instrumental units.
        """
        from lsst.afw.table import ExposureCatalog

        assert EXTERNAL_DATA_DIR is not None, "Guaranteed by decorator."
        # Make a copy of class state so we can modify it without breaking
        # other tests.
        original = self.visit_image.copy()
        # We should not be able to convert to instrumental units when there is
        # no photometric scaling.
        with self.assertRaises(u.UnitConversionError):
            original.convert_unit(u.electron)
        # Converting to the current unit should be a no-op that does not need
        # to copy.
        visit_image_nJy = original.convert_unit(u.nJy, copy=False)
        self.assertTrue(np.may_share_memory(visit_image_nJy.image.array, original.image.array))
        self.assertTrue(np.may_share_memory(visit_image_nJy.variance.array, original.variance.array))
        # Even without a photometric_scaling attached, we should be able to
        # convert to a compatible unit, but only if we allow copies.
        with self.assertRaises(u.UnitConversionError):
            original.convert_unit(u.mJy, copy=False)
        visit_image_mJy = original.convert_unit(u.mJy, copy="as-needed")
        self.assertEqual(visit_image_mJy.unit, u.mJy)
        assert_close(self, visit_image_mJy.image.array, original.image.array * 1e-6)
        self.assertTrue(np.may_share_memory(visit_image_nJy.mask.array, original.mask.array))
        assert_close(self, visit_image_mJy.variance.array, original.variance.array * 1e-12)
        # Test that we haven't dropped any component objects along the way,
        # and that they're all still the same objects or thin views.
        self.assertTrue(np.may_share_memory(visit_image_mJy.mask.array, original.mask.array))
        self.assertIs(visit_image_mJy.projection, original.projection)
        self.assertIs(visit_image_mJy.obs_info, original.obs_info)
        self.assertIs(visit_image_mJy.summary_stats, original.summary_stats)
        self.assertIs(visit_image_mJy.psf, original.psf)
        self.assertIs(visit_image_mJy.detector, original.detector)
        self.assertIs(visit_image_mJy.bounds, original.bounds)
        self.assertIs(visit_image_mJy.aperture_corrections, original.aperture_corrections)
        self.assertIs(visit_image_mJy.photometric_scaling, original.photometric_scaling)
        # Attach the final PhotoCalib (this isn't stored with the legacy file
        # because that is the mapping to nJy, which is trivial).
        visit_summary = ExposureCatalog.readFits(
            os.path.join(EXTERNAL_DATA_DIR, "dp2", "legacy", "visit_summary.fits")
        )
        legacy_photo_calib = visit_summary.find(DP2_VISIT_DETECTOR_DATA_ID["detector"]).getPhotoCalib()
        visit_image_nJy.photometric_scaling = field_from_legacy_photo_calib(
            legacy_photo_calib, bounds=original.detector.bbox, instrumental_unit=u.electron
        )
        compare_photo_calib_to_legacy(
            self,
            visit_image_nJy.photometric_scaling,
            self.legacy_exposure.getPhotoCalib(),
            applied_legacy_photo_calib=legacy_photo_calib,
            subimage_bbox=visit_image_nJy.bbox,
        )
        # We still can't convert to completely unrelated units.
        with self.assertRaises(u.UnitConversionError):
            visit_image_nJy.convert_unit(u.mm)
        # Uncalibrating via the photometric_scaling matches what legacy code
        # does, and by default it copies everything.
        with self.assertRaises(u.UnitConversionError):
            visit_image_nJy.convert_unit(u.electron, copy=False)
        legacy_masked_image_e = legacy_photo_calib.uncalibrateImage(self.legacy_exposure.maskedImage)
        visit_image_e = visit_image_nJy.convert_unit(u.electron)
        assert_close(self, visit_image_e.image.array, legacy_masked_image_e.image.array)
        assert_close(self, visit_image_e.variance.array, legacy_masked_image_e.variance.array)
        self.assertFalse(np.may_share_memory(visit_image_e.mask.array, visit_image_nJy.mask.array))
        # We can also uncalibrate if we start with an image that has units
        # that are compatible with the photometric_scaling but not identical
        # to it.
        visit_image_mJy.photometric_scaling = visit_image_nJy.photometric_scaling
        visit_image_e = visit_image_mJy.convert_unit(u.electron)
        assert_close(self, visit_image_e.image.array, legacy_masked_image_e.image.array)
        assert_close(self, visit_image_e.variance.array, legacy_masked_image_e.variance.array)
        # We can re-apply the scaling go go back to calibrated units.
        visit_image_nJy_2 = visit_image_e.convert_unit(u.nJy)
        assert_close(self, visit_image_nJy_2.image.array, visit_image_nJy.image.array)
        assert_close(self, visit_image_nJy_2.variance.array, original.variance.array)


@unittest.skipUnless(EXTERNAL_DATA_DIR is not None, "TESTDATA_IMAGES_DIR is not in the environment.")
class PreliminaryVisitImageLegacyTestCase(unittest.TestCase, VisitImageLegacyTestMixin):
    """Tests for the VisitImage class using a DRP preliminary_visit_image
    dataset.

    Requires legacy code.
    """

    @classmethod
    def setUpClass(cls) -> None:
        assert EXTERNAL_DATA_DIR is not None, "Guaranteed by decorator."
        cls.filename = os.path.join(EXTERNAL_DATA_DIR, "dp2", "legacy", "preliminary_visit_image.fits")
        try:
            from lsst.afw.image import ExposureFitsReader

            cls.legacy_exposure = ExposureFitsReader(cls.filename).read()
        except ImportError:
            raise unittest.SkipTest("afw not available; cannot read legacy visit images") from None
        cls.plane_map = get_legacy_visit_image_mask_planes()
        cls.visit_image = VisitImage.read_legacy(
            cls.filename, preserve_quantization=True, plane_map=cls.plane_map
        )
        cls.unit = u.electron


if __name__ == "__main__":
    unittest.main()

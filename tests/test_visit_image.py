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

import dataclasses
import os
import warnings
from typing import Any, Literal

import astropy.io.fits
import astropy.units as u
import astropy.wcs
import numpy as np
import pytest
from astro_metadata_translator import ObservationInfo

from lsst.images import (
    BackgroundMap,
    Box,
    DetectorFrame,
    DifferenceImage,
    Image,
    MaskPlane,
    MaskSchema,
    ObservationSummaryStats,
    Polygon,
    SkyProjectionAstropyView,
    TractFrame,
    VisitImage,
    get_legacy_difference_image_mask_planes,
    get_legacy_visit_image_mask_planes,
)
from lsst.images.aperture_corrections import ApertureCorrectionMap, aperture_corrections_to_legacy
from lsst.images.cameras import Detector
from lsst.images.fields import ChebyshevField, SplineField, SumField, field_from_legacy_photo_calib
from lsst.images.fits import ExtensionKey, FitsOpaqueMetadata
from lsst.images.psfs import GaussianPointSpreadFunction, PointSpreadFunction
from lsst.images.serialization import read
from lsst.images.tests import (
    DP2_VISIT_DETECTOR_DATA_ID,
    RoundtripFits,
    RoundtripJson,
    RoundtripNdf,
    TemporaryButler,
    assert_close,
    assert_masked_images_equal,
    assert_sky_projections_equal,
    assert_visit_images_equal,
    compare_aperture_corrections_to_legacy,
    compare_detector_to_legacy,
    compare_photo_calib_to_legacy,
    compare_visit_image_to_legacy,
    make_random_sky_projection,
)

try:
    import h5py  # noqa: F401

    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False

try:
    from lsst.afw.image import Exposure as LegacyExposure
    from lsst.afw.image import VisitInfo as LegacyVisitInfo
except ImportError:
    type LegacyExposure = Any  # type: ignore[no-redef]
    type LegacyVisitInfo = Any  # type: ignore[no-redef]

EXTERNAL_DATA_DIR = os.environ.get("TESTDATA_IMAGES_DIR", None)
LOCAL_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

skip_no_h5py = pytest.mark.skipif(not HAVE_H5PY, reason="h5py is not installed")


@pytest.fixture(scope="session")
def visit_image_components() -> dict[str, Any]:
    """Return a dictionary of VisitImage components."""
    rng = np.random.default_rng(500)
    det_frame = DetectorFrame(instrument="Inst", visit=1234, detector=1, bbox=Box.factory[1:4096, 1:4096])
    mask_schema = MaskSchema([MaskPlane("M1", "D1")])
    obs_info = ObservationInfo(instrument="LSSTCam", detector_num=4, physical_filter="r1")
    summary_stats = ObservationSummaryStats(psfSigma=2.5, zeroPoint=31.4)
    gaussian_psf = GaussianPointSpreadFunction(2.5, stamp_size=33, bounds=Box.factory[-10:10, -12:13])
    aperture_corrections: ApertureCorrectionMap = {
        "flux1": ChebyshevField(det_frame.bbox, np.array([0.75])),
        "flux2": ChebyshevField(det_frame.bbox, np.array([0.625])),
    }
    detector = read(os.path.join(LOCAL_DATA_DIR, "detector.json"), Detector)
    image = Image(42, shape=(1024, 1024), unit=u.nJy)
    variance = Image(5.0, shape=(1024, 1024), unit=u.nJy * u.nJy)
    # polygon is the lower triangle of the image.
    polygon = Polygon(x_vertices=[-0.5, 1023.5, -0.5], y_vertices=[-0.5, -0.5, 1023.5])
    sky_projection = make_random_sky_projection(rng, det_frame, det_frame.bbox)
    return {
        "mask_schema": mask_schema,
        "obs_info": obs_info,
        "summary_stats": summary_stats,
        "gaussian_psf": gaussian_psf,
        "aperture_corrections": aperture_corrections,
        "detector": detector,
        "image": image,
        "variance": variance,
        "polygon": polygon,
        "sky_projection": sky_projection,
    }


def make_visit_image(components: dict[str, Any]) -> VisitImage:
    """Construct a new VisitImage with most components populated."""
    det_frame = components["sky_projection"].pixel_frame
    opaque = FitsOpaqueMetadata()
    hdr = astropy.io.fits.Header()
    with warnings.catch_warnings():
        # Silence warnings about long keys becoming HIERARCH.
        warnings.simplefilter("ignore", category=astropy.io.fits.verify.VerifyWarning)
        hdr.update({"PLATFORM": "lsstcam", "LSST BUTLER ID": "123456789"})
    opaque.extract_legacy_primary_header(hdr)
    # API signature suggests sky_projection and obs_info can be None but
    # they are required (unless you pass them in via the image plane).
    vi = VisitImage(
        components["image"],
        variance=components["variance"],
        psf=GaussianPointSpreadFunction(2.5, stamp_size=33, bounds=Box.factory[-10:10, -12:13]),
        mask_schema=components["mask_schema"],
        sky_projection=components["sky_projection"],
        obs_info=components["obs_info"],
        summary_stats=components["summary_stats"],
        detector=components["detector"],
        bounds=components["polygon"],
        aperture_corrections=components["aperture_corrections"],
        band="r",
    )
    vi.backgrounds.add(
        "standard",
        ChebyshevField(det_frame.bbox, np.array([[2.0]])),
        description="Background subtracted from the image.",
        is_subtracted=True,
    )
    vi._opaque_metadata = opaque
    return vi


def make_simplest_visit_image(components: dict[str, Any]) -> VisitImage:
    """Construct a VisitImage with the minimal set of components populated."""
    return VisitImage(
        components["image"],
        psf=GaussianPointSpreadFunction(2.5, stamp_size=33, bounds=Box.factory[-10:10, -12:13]),
        mask_schema=components["mask_schema"],
        sky_projection=components["sky_projection"],
        detector=components["detector"],
        obs_info=components["obs_info"],
        band="r",
    )


def _make_sum_background_visit_image(components: dict[str, Any], visit_image: VisitImage) -> VisitImage:
    """Return a VisitImage whose subtracted background is a SumField."""
    rng = np.random.default_rng(42)
    bbox = visit_image.image.sky_projection.pixel_frame.bbox
    bin_y = bbox.y.linspace(6)
    bin_x = bbox.x.linspace(7)
    spline_a = SplineField(
        bbox,
        rng.standard_normal(size=(bin_y.size, bin_x.size)),
        y=bin_y,
        x=bin_x,
    )
    spline_b = SplineField(
        bbox,
        rng.standard_normal(size=(bin_y.size, bin_x.size)),
        y=bin_y,
        x=bin_x,
    )
    sum_field = SumField([spline_a, spline_b])
    bg_map = BackgroundMap()
    bg_map.add(
        "stacked",
        sum_field,
        description="Two-operand SumField subtracted background.",
        is_subtracted=True,
    )
    return VisitImage(
        components["image"],
        variance=components["variance"],
        psf=components["gaussian_psf"],
        mask_schema=components["mask_schema"],
        sky_projection=components["sky_projection"],
        obs_info=components["obs_info"],
        summary_stats=components["summary_stats"],
        detector=components["detector"],
        band="r",
        backgrounds=bg_map,
    )


def _check_sum_background_round_trip(result: VisitImage, original: VisitImage) -> None:
    """Assert that a round-tripped SumField background matches the original."""
    subtracted = result.backgrounds.subtracted
    assert subtracted is not None
    assert isinstance(subtracted.field, SumField)
    original_subtracted = original.backgrounds.subtracted
    assert original_subtracted is not None
    original_field = original_subtracted.field
    assert isinstance(original_field, SumField)
    round_field = subtracted.field
    assert isinstance(round_field, SumField)
    assert len(round_field.operands) == len(original_field.operands)
    for round_op, orig_op in zip(round_field.operands, original_field.operands, strict=True):
        assert round_op == orig_op


def test_basics(visit_image_components: dict[str, Any]) -> None:
    """Verify VisitImage constructor patterns and required-argument checks."""
    c = visit_image_components
    # Test default fill of variance.
    visit = make_simplest_visit_image(c)
    assert visit.variance.array[0, 0] == 1.0
    assert visit[...] is visit
    assert str(visit) == "VisitImage(Image([y=0:1024, x=0:1024], int64), ['M1'])"
    assert repr(visit) == (
        "VisitImage(Image(..., bbox=Box(y=Interval(start=0, stop=1024), x=Interval(start=0, stop=1024)),"
        " dtype=dtype('int64')), mask_schema=MaskSchema([MaskPlane(name='M1', description='D1')],"
        " dtype=dtype('uint8')))"
    )

    astropy_wcs = visit.astropy_wcs
    assert isinstance(astropy_wcs, SkyProjectionAstropyView)
    approx_wcs = visit.fits_wcs
    assert isinstance(approx_wcs, astropy.wcs.WCS)

    with pytest.raises(TypeError):
        # Requires a PSF.
        VisitImage(
            c["image"],
            mask_schema=c["mask_schema"],
            sky_projection=c["sky_projection"],
            obs_info=c["obs_info"],
            detector=c["detector"],
            band="r",
        )

    with pytest.raises(TypeError):
        # Requires ObservationInfo.
        VisitImage(
            c["image"],
            psf=c["gaussian_psf"],
            mask_schema=c["mask_schema"],
            sky_projection=c["sky_projection"],
            detector=c["detector"],
            band="r",
        )

    with pytest.raises(TypeError):
        # Requires a sky_projection.
        VisitImage(
            c["image"],
            psf=c["gaussian_psf"],
            mask_schema=c["mask_schema"],
            obs_info=c["obs_info"],
            detector=c["detector"],
            band="r",
        )

    with pytest.raises(TypeError):
        # Requires a detector.
        VisitImage(
            c["image"],
            psf=c["gaussian_psf"],
            mask_schema=c["mask_schema"],
            sky_projection=c["sky_projection"],
            obs_info=c["obs_info"],
            band="r",
        )

    with pytest.raises(TypeError):
        # Requires some form of mask.
        VisitImage(
            c["image"],
            psf=c["gaussian_psf"],
            sky_projection=c["sky_projection"],
            obs_info=c["obs_info"],
            detector=c["detector"],
            band="r",
        )

    with pytest.raises(TypeError):
        VisitImage(
            Image(42, shape=(5, 5)),
            psf=c["gaussian_psf"],
            mask_schema=c["mask_schema"],
            sky_projection=c["sky_projection"],
            obs_info=c["obs_info"],
            detector=c["detector"],
            band="r",
        )

    # Requires a DetectorFrame.
    rng = np.random.default_rng(501)
    tract_frame = TractFrame(skymap="Skymap", tract=1, bbox=Box.factory[1:10, 1:10])
    tract_proj = make_random_sky_projection(rng, tract_frame, Box.factory[1:4096, 1:4096])
    with pytest.raises(TypeError):
        VisitImage(
            c["image"],
            sky_projection=tract_proj,
            psf=c["gaussian_psf"],
            mask_schema=c["mask_schema"],
            obs_info=c["obs_info"],
            detector=c["detector"],
            band="r",
        )

    # Variance unit mismatch.
    with pytest.raises(ValueError):
        VisitImage(
            c["image"],
            variance=c["image"],
            psf=c["gaussian_psf"],
            mask_schema=c["mask_schema"],
            sky_projection=c["sky_projection"],
            obs_info=c["obs_info"],
            detector=c["detector"],
            band="r",
        )


def test_copy_and_slice(visit_image_components: dict[str, Any]) -> None:
    """Verify that copy deep-copies arrays and components while slice shares
    them.
    """
    c = visit_image_components
    visit_image = make_visit_image(c)
    copy = visit_image.copy()
    copy.image.array[0, 0] = 30.0
    assert visit_image.image.array[0, 0] == 42.0
    assert copy.image.array[0, 0] == 30.0
    subvisit = visit_image[Box.factory[0:5, 0:5]]
    # Check summary stats.
    assert copy.summary_stats == visit_image.summary_stats
    assert copy.summary_stats is not visit_image.summary_stats
    assert subvisit.summary_stats == visit_image.summary_stats
    assert subvisit.summary_stats is visit_image.summary_stats
    # Check aperture corrections.
    assert copy.aperture_corrections.keys() == visit_image.aperture_corrections.keys()
    assert copy.aperture_corrections is not visit_image.aperture_corrections
    assert subvisit.aperture_corrections.keys() == visit_image.aperture_corrections.keys()
    assert subvisit.aperture_corrections is visit_image.aperture_corrections
    # Check backgrounds.
    assert copy.backgrounds.keys() == visit_image.backgrounds.keys()
    assert copy.backgrounds is not visit_image.backgrounds
    assert subvisit.backgrounds.keys() == visit_image.backgrounds.keys()
    assert subvisit.backgrounds is visit_image.backgrounds
    # Check bounds.
    assert copy.bounds is c["polygon"]
    assert subvisit.bounds == subvisit.bbox  # original polygon wholly encloses subvisit.bbox


def test_obs_info(visit_image_components: dict[str, Any]) -> None:
    """Verify that ObservationInfo is present and carries the expected
    instrument.
    """
    visit_image = make_visit_image(visit_image_components)
    assert visit_image.obs_info is not None
    assert visit_image.obs_info.instrument == "LSSTCam"


def test_summary_stats(visit_image_components: dict[str, Any]) -> None:
    """Verify ObservationSummaryStats equality and inequality comparisons."""
    summary_stats = visit_image_components["summary_stats"]
    assert summary_stats == ObservationSummaryStats(psfSigma=2.5, zeroPoint=31.4)
    assert summary_stats != ObservationSummaryStats(psfSigma=2.5)
    assert summary_stats != ObservationSummaryStats(psfSigma=2.5, raCorners=(5.2, 5.4, 5.4, 5.2))


@skip_no_h5py
def test_round_trip_ndf(visit_image_components: dict[str, Any]) -> None:
    """Verify NDF round-trip produces a VisitImage equal to the original."""
    visit_image = make_visit_image(visit_image_components)
    with RoundtripNdf(visit_image, "VisitImage") as roundtrip:
        assert_visit_images_equal(roundtrip.result, visit_image, expect_view=False)


@skip_no_h5py
def test_fits_ndf_consistency(visit_image_components: dict[str, Any]) -> None:
    """Verify FITS and NDF backends produce equal VisitImages on round-trip."""
    visit_image = make_visit_image(visit_image_components)
    with RoundtripFits(visit_image) as fits_rt, RoundtripNdf(visit_image) as ndf_rt:
        assert_visit_images_equal(visit_image, fits_rt.result, expect_view=False)
        assert_visit_images_equal(visit_image, ndf_rt.result, expect_view=False)
        assert_visit_images_equal(fits_rt.result, ndf_rt.result, expect_view=False)


def test_fits_json_consistency(visit_image_components: dict[str, Any]) -> None:
    """Verify FITS and JSON backends produce equal VisitImages."""
    visit_image = make_visit_image(visit_image_components)
    with (
        RoundtripFits(visit_image) as fits_rt,
        RoundtripJson(visit_image) as json_rt,
    ):
        assert_visit_images_equal(visit_image, fits_rt.result, expect_view=False)
        assert_visit_images_equal(visit_image, json_rt.result, expect_view=False)
        assert_visit_images_equal(fits_rt.result, json_rt.result, expect_view=False)


def test_read_write(visit_image_components: dict[str, Any]) -> None:
    """Verify a VisitImage round-trips through FITS with correct compression.

    Checks compression headers, subimage reads, equality, and opaque
    metadata.  Contains only butler-free assertions; component reads live
    in `test_read_write_components`.
    """
    visit_image = make_visit_image(visit_image_components)
    with RoundtripFits(visit_image, "VisitImage") as roundtrip:
        # Check that we're still using the right compression, and that we
        # wrote WCSs.
        fits = roundtrip.inspect()
        assert fits[1].header["ZCMPTYPE"] == "GZIP_2"
        assert fits[1].header["CTYPE1"] == "RA---TAN"
        assert fits[2].header["ZCMPTYPE"] == "GZIP_2"
        assert fits[2].header["CTYPE1"] == "RA---TAN"
        assert fits[3].header["ZCMPTYPE"] == "GZIP_2"
        assert fits[3].header["CTYPE1"] == "RA---TAN"
        # Check a subimage read (no component arg — does not trigger a skip).
        subbox = Box.factory[8:13, 9:30]
        subimage = roundtrip.get(bbox=subbox)
        assert_masked_images_equal(subimage, visit_image[subbox], expect_view=False)

    assert_visit_images_equal(roundtrip.result, visit_image, expect_view=False)
    # Check that the round-tripped headers are the same (up to card order).
    assert len(roundtrip.result._opaque_metadata.headers[ExtensionKey()]) == 1
    assert dict(visit_image._opaque_metadata.headers[ExtensionKey()]) == dict(
        roundtrip.result._opaque_metadata.headers[ExtensionKey()]
    )
    assert not roundtrip.result._opaque_metadata.headers[ExtensionKey("IMAGE")]
    assert not roundtrip.result._opaque_metadata.headers[ExtensionKey("MASK")]
    assert not roundtrip.result._opaque_metadata.headers[ExtensionKey("VARIANCE")]
    # Spot-check the concrete background contents (names, field types,
    # subtracted entry) against the known fixture, so the equality check
    # above is not vacuously satisfied by empty background maps.
    assert isinstance(roundtrip.result.backgrounds, BackgroundMap)
    assert roundtrip.result.backgrounds.keys() == {"standard"}
    assert isinstance(roundtrip.result.backgrounds["standard"].field, ChebyshevField)
    assert roundtrip.result.backgrounds.subtracted.name == "standard"
    assert roundtrip.result.backgrounds.subtracted.description == "Background subtracted from the image."


def test_read_write_components(visit_image_components: dict[str, Any]) -> None:
    """Verify component reads and storage-class overrides round-trip correctly.

    Requires a butler; skips when `lsst.daf.butler` is absent.
    Butler-free assertions live in `test_read_write`.
    """
    c = visit_image_components
    visit_image = make_visit_image(c)
    with RoundtripFits(visit_image, "VisitImage") as roundtrip:
        subbox = Box.factory[8:13, 9:30]
        subimage = roundtrip.get(bbox=subbox)

        # Get an explicit masked image to compare with the subimage.
        subimage_masked = roundtrip.get("masked_image", bbox=subbox)
        assert_masked_images_equal(subimage_masked, subimage, expect_view=False)

        # Get the same masked image in a multi-component get and ensure
        # it is the same thing.
        components = roundtrip.get("components", components=["masked_image", "psf"], bbox=subbox)
        assert set(components) == {"masked_image", "psf"}
        assert_masked_images_equal(components["masked_image"], subimage_masked, expect_view=False)

        assert roundtrip.get("bbox") == visit_image.bbox

        obs_info = roundtrip.get("obs_info")
        assert isinstance(obs_info, ObservationInfo)
        assert obs_info == visit_image.obs_info

        summary_stats = roundtrip.get("summary_stats")
        assert isinstance(summary_stats, ObservationSummaryStats)
        assert summary_stats == visit_image.summary_stats

        psf = roundtrip.get("psf")
        assert isinstance(psf, GaussianPointSpreadFunction)
        assert psf.kernel_bbox == c["gaussian_psf"].kernel_bbox

        backgrounds = roundtrip.get("backgrounds")
        assert isinstance(backgrounds, BackgroundMap)
        assert backgrounds.keys() == {"standard"}
        assert isinstance(backgrounds["standard"].field, ChebyshevField)
        assert backgrounds.subtracted.name == "standard"
        assert roundtrip.result.backgrounds.subtracted.description == "Background subtracted from the image."

        # Test some components get edge cases.
        components = roundtrip.get("components", components="image")
        assert isinstance(components["image"], Image)

        components = roundtrip.get("components")
        assert set(components) == {
            "image",
            "variance",
            "psf",
            "bbox",
            "mask",
            "obs_info",
            "backgrounds",
            "detector",
            "aperture_corrections",
            "sky_projection",
            "summary_stats",
            "photometric_scaling",
        }

        # Butler morphs RuntimeError to ValueError.
        with pytest.raises(ValueError):
            roundtrip.get("components", components=["image", "nonexistent"])

        with pytest.raises(ValueError):
            roundtrip.get("components", components=["image", "components"])

        with pytest.raises(ValueError):
            roundtrip.get("components", components=[])

        with pytest.raises(ValueError):
            # PSF does not know how to use bbox so this fails.
            roundtrip.get("components", components="psf", bbox=subbox)


def test_sum_background_round_trip_fits(visit_image_components: dict[str, Any]) -> None:
    """Verify FITS backend keeps two same-named SumField operands as distinct
    EXTVERs.
    """
    visit_image = make_visit_image(visit_image_components)
    visit = _make_sum_background_visit_image(visit_image_components, visit_image)
    with RoundtripFits(visit) as roundtrip:
        _check_sum_background_round_trip(roundtrip.result, visit)


@skip_no_h5py
def test_sum_background_round_trip_ndf(visit_image_components: dict[str, Any]) -> None:
    """Verify NDF backend disambiguates the repeated ``data`` leaf, just as
    the FITS backend does.
    """
    visit_image = make_visit_image(visit_image_components)
    visit = _make_sum_background_visit_image(visit_image_components, visit_image)
    with RoundtripNdf(visit) as roundtrip:
        _check_sum_background_round_trip(roundtrip.result, visit)


@dataclasses.dataclass
class _LegacyTestData:
    filename: str
    plane_map: dict[str, MaskPlane] = dataclasses.field(default_factory=get_legacy_visit_image_mask_planes)
    unit: u.Unit = u.nJy
    storage_class: str = "VisitImage"
    read_cls: type[VisitImage] = VisitImage
    legacy_exposure: LegacyExposure = dataclasses.field(init=False)

    @classmethod
    def get(
        cls, which: Literal["visit_image", "preliminary_visit_image", "difference_image"]
    ) -> _LegacyTestData:
        if EXTERNAL_DATA_DIR is None:
            pytest.skip("TESTDATA_IMAGES is not set up.")
        result = cls(
            os.path.join(EXTERNAL_DATA_DIR, "dp2", "legacy", f"{which}.fits"),
        )
        match which:
            case "preliminary_visit_image":
                result.unit = u.electron
            case "difference_image":
                result.storage_class = "DifferenceImage"
                result.read_cls = DifferenceImage
                result.plane_map = get_legacy_difference_image_mask_planes()
            case "visit_image":
                pass
        try:
            from lsst.afw.image import ExposureFitsReader

            result.legacy_exposure = ExposureFitsReader(result.filename).read()
        except ImportError:
            pytest.skip("lsst.afw.image is not available; cannot read legacy exposures")
        result.visit_image = result.read_cls.read_legacy(
            result.filename, preserve_quantization=True, plane_map=result.plane_map
        )
        return result


@pytest.fixture(scope="session", params=["visit_image", "preliminary_visit_image", "difference_image"])
def legacy_test_data(request: pytest.FixtureRequest) -> _LegacyTestData:
    """Return legacy test data.

    Tests that depend on this parameterized fixture run on all of the legacy
    test images.
    """
    return _LegacyTestData.get(request.param)


@pytest.fixture(scope="session", params=["visit_image", "difference_image"])
def legacy_test_data_calibrated(request: pytest.FixtureRequest) -> _LegacyTestData:
    """Return legacy test data for calibrated images only.

    Tests that depend on this parameterized fixture do not run on
    preliminary_visit_image, since that has 'electron' pixel units
    """
    return _LegacyTestData.get(request.param)


def _check_legacy_obs_info(obs_info: ObservationInfo | None) -> None:
    """Assert obs_info carries expected LSSTCam/DP2 field values."""
    assert isinstance(obs_info, ObservationInfo)
    assert obs_info.instrument == "LSSTCam"
    assert obs_info.detector_num == 85, obs_info
    assert obs_info.detector_unique_name == "R21_S11", obs_info
    assert obs_info.physical_filter == "r_57", obs_info


def test_legacy_errors(legacy_test_data: _LegacyTestData) -> None:
    """Verify that from_legacy and read_legacy raise ValueError on
    conflicting arguments.
    """
    with pytest.raises(ValueError):
        VisitImage.from_legacy(legacy_test_data.legacy_exposure, instrument="HSC")
    with pytest.raises(ValueError):
        VisitImage.from_legacy(legacy_test_data.legacy_exposure, visit=123456)
    with pytest.raises(ValueError):
        VisitImage.from_legacy(legacy_test_data.legacy_exposure, unit=u.mJy)
    visit = VisitImage.from_legacy(
        legacy_test_data.legacy_exposure,
        instrument="LSSTCam",
        unit=legacy_test_data.unit,
        visit=2025052000177,
    )
    assert visit.unit == legacy_test_data.unit

    with pytest.raises(ValueError):
        legacy_test_data.read_cls.read_legacy(legacy_test_data.filename, instrument="HSC")
    with pytest.raises(ValueError):
        legacy_test_data.read_cls.read_legacy(legacy_test_data.filename, visit=123456)


def test_component_reads(legacy_test_data: _LegacyTestData) -> None:
    """Verify that individual components can be read from a legacy FITS
    file.
    """
    visit = VisitImage.read_legacy(legacy_test_data.filename)
    proj = VisitImage.read_legacy(legacy_test_data.filename, component="sky_projection")
    assert_sky_projections_equal(proj, visit.sky_projection, expect_identity=False)
    image = VisitImage.read_legacy(legacy_test_data.filename, component="image")
    assert image == visit.image
    assert_sky_projections_equal(proj, image.sky_projection, expect_identity=False)
    variance = VisitImage.read_legacy(legacy_test_data.filename, component="variance")
    assert variance == visit.variance
    assert_sky_projections_equal(proj, variance.sky_projection, expect_identity=False)
    mask = VisitImage.read_legacy(legacy_test_data.filename, component="mask")
    assert mask == visit.mask
    assert_sky_projections_equal(proj, mask.sky_projection, expect_identity=False)
    psf = VisitImage.read_legacy(legacy_test_data.filename, component="psf")
    assert isinstance(psf, PointSpreadFunction)
    obs_info = VisitImage.read_legacy(legacy_test_data.filename, component="obs_info")
    _check_legacy_obs_info(obs_info)
    summary_stats = VisitImage.read_legacy(legacy_test_data.filename, component="summary_stats")
    assert isinstance(summary_stats, ObservationSummaryStats)
    assert summary_stats.nPsfStar == legacy_test_data.legacy_exposure.info.getSummaryStats().nPsfStar
    compare_aperture_corrections_to_legacy(
        VisitImage.read_legacy(legacy_test_data.filename, component="aperture_corrections"),
        legacy_test_data.legacy_exposure.info.getApCorrMap(),
        visit.bbox,
    )
    detector = VisitImage.read_legacy(legacy_test_data.filename, component="detector")
    compare_detector_to_legacy(
        detector, legacy_test_data.legacy_exposure.getDetector(), is_raw_assembled=True
    )
    photometric_scaling = VisitImage.read_legacy(legacy_test_data.filename, component="photometric_scaling")
    compare_photo_calib_to_legacy(
        photometric_scaling,
        legacy_test_data.legacy_exposure.getPhotoCalib(),
        subimage_bbox=visit.bbox,
    )


def test_legacy_obs_info(legacy_test_data: _LegacyTestData) -> None:
    """Verify that ObservationInfo is constructed correctly from a legacy
    exposure.
    """
    legacy = VisitImage.from_legacy(legacy_test_data.legacy_exposure, plane_map=legacy_test_data.plane_map)
    assert legacy.obs_info is not None
    assert legacy.obs_info == legacy_test_data.visit_image.obs_info
    assert legacy.obs_info is not None  # for mypy
    assert legacy.obs_info.instrument == "LSSTCam"
    assert legacy.obs_info.detector_num == 85, legacy.obs_info
    assert legacy.obs_info.detector_unique_name == "R21_S11", legacy.obs_info
    assert legacy.obs_info.physical_filter == "r_57", legacy.obs_info


def test_aperture_corrections_to_legacy(legacy_test_data: _LegacyTestData) -> None:
    """Verify that aperture corrections round-trip through a legacy
    ApCorrMap.
    """
    ap_corrections = legacy_test_data.visit_image.aperture_corrections
    legacy_ap_corr_map = aperture_corrections_to_legacy(ap_corrections)
    compare_aperture_corrections_to_legacy(
        ap_corrections,
        legacy_ap_corr_map,
        legacy_test_data.visit_image.bbox,
    )


def _check_legacy_headers(visit_image: VisitImage) -> None:
    """Assert that primary and extension headers are stripped correctly."""
    header = visit_image._opaque_metadata.headers[ExtensionKey()]
    assert "EXPTIME" in header
    assert header["PLATFORM"] == "lsstcam"
    assert "LSST BUTLER ID" not in header
    assert "AR HDU" not in header
    assert "A_ORDER" not in header
    assert not visit_image._opaque_metadata.headers.get(ExtensionKey("IMAGE"), astropy.io.fits.Header())
    assert not visit_image._opaque_metadata.headers.get(ExtensionKey("MASK"), astropy.io.fits.Header())
    assert not visit_image._opaque_metadata.headers.get(ExtensionKey("VARIANCE"), astropy.io.fits.Header())


def test_read_legacy_headers(legacy_test_data: _LegacyTestData) -> None:
    """Verify that headers were stripped and interpreted correctly in
    read_legacy.
    """
    assert legacy_test_data.visit_image.unit == legacy_test_data.unit
    _check_legacy_headers(legacy_test_data.visit_image)


def test_from_legacy_headers(legacy_test_data: _LegacyTestData) -> None:
    """Verify that from_legacy handles primary and extension headers
    correctly.
    """
    legacy = VisitImage.from_legacy(legacy_test_data.legacy_exposure, plane_map=legacy_test_data.plane_map)
    assert legacy.unit == legacy_test_data.unit
    _check_legacy_headers(legacy)


def test_rewrite(legacy_test_data: _LegacyTestData) -> None:
    """Verify that a legacy VisitImage can be rewritten and round-trips both
    pixel values and all components.
    """
    with RoundtripFits(legacy_test_data.visit_image, legacy_test_data.storage_class) as roundtrip:
        fits = roundtrip.inspect()
        assert fits[1].header["ZCMPTYPE"] == "RICE_1"
        assert fits[1].header["CTYPE1"] == "RA---TAN-SIP"
        assert fits[2].header["ZCMPTYPE"] == "GZIP_2"
        assert fits[2].header["CTYPE1"] == "RA---TAN-SIP"
        assert fits[3].header["ZCMPTYPE"] == "RICE_1"
        assert fits[3].header["CTYPE1"] == "RA---TAN-SIP"
        subbox = Box.factory[8:13, 9:30]
        subimage = roundtrip.get(bbox=subbox)
        assert_masked_images_equal(subimage, legacy_test_data.visit_image[subbox], expect_view=False)
        alternates: dict[str, Any] = {}
        assert roundtrip.get("bbox") == legacy_test_data.visit_image.bbox
        alternates = {
            k: roundtrip.get(k)
            for k in [
                "sky_projection",
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
        legacy_exposure = roundtrip.get(storageClass="Exposure")
        assert isinstance(legacy_exposure, LegacyExposure)
        compare_visit_image_to_legacy(
            legacy_test_data.visit_image,
            legacy_exposure,
            expect_view=False,
            plane_map=legacy_test_data.plane_map,
            **DP2_VISIT_DETECTOR_DATA_ID,
        )
        if legacy_test_data.visit_image.unit == u.nJy:
            assert legacy_exposure.getPhotoCalib()._isConstant
            assert legacy_exposure.getPhotoCalib().getCalibrationMean() == 1.0
        else:
            compare_photo_calib_to_legacy(
                legacy_test_data.visit_image.photometric_scaling,
                legacy_exposure.getPhotoCalib(),
                subimage_bbox=subbox,
            )
        assert legacy_exposure.info.getId() == legacy_test_data.legacy_exposure.info.getId()
        visit_info = roundtrip.get("obs_info", storageClass="VisitInfo")
        assert isinstance(visit_info, LegacyVisitInfo)
        assert visit_info.getInstrumentLabel() == "LSSTCam"

    assert_visit_images_equal(roundtrip.result, legacy_test_data.visit_image, expect_view=False)
    assert dict(legacy_test_data.visit_image._opaque_metadata.headers[ExtensionKey()]) == dict(
        roundtrip.result._opaque_metadata.headers[ExtensionKey()]
    )
    assert not roundtrip.result._opaque_metadata.headers[ExtensionKey("IMAGE")]
    assert not roundtrip.result._opaque_metadata.headers[ExtensionKey("MASK")]
    assert not roundtrip.result._opaque_metadata.headers[ExtensionKey("VARIANCE")]
    assert roundtrip.result._opaque_metadata.headers[ExtensionKey()]["PLATFORM"] == "lsstcam"
    compare_visit_image_to_legacy(
        roundtrip.result,
        legacy_test_data.legacy_exposure,
        expect_view=False,
        plane_map=legacy_test_data.plane_map,
        **DP2_VISIT_DETECTOR_DATA_ID,
        alternates=alternates,
    )
    compare_visit_image_to_legacy(
        legacy_test_data.read_cls.from_legacy(
            legacy_test_data.legacy_exposure, plane_map=legacy_test_data.plane_map
        ),
        legacy_test_data.legacy_exposure,
        expect_view=True,
        plane_map=legacy_test_data.plane_map,
        **DP2_VISIT_DETECTOR_DATA_ID,
    )


def test_butler_converters(legacy_test_data: _LegacyTestData) -> None:
    """Verify that a VisitImage can be read from a Butler dataset written as
    an Exposure.
    """
    try:
        from lsst.daf.butler import FileDataset
    except ImportError:
        pytest.skip("lsst.daf.butler could not be imported.")

    with TemporaryButler(legacy="ExposureF") as helper:
        helper.butler.ingest(
            FileDataset(path=legacy_test_data.filename, refs=[helper.legacy]), transfer="symlink"
        )
        visit_image_ref = helper.legacy.overrideStorageClass(legacy_test_data.storage_class)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*filter label mismatch.*", category=UserWarning)
            visit_image = helper.butler.get(visit_image_ref)
            assert visit_image._opaque_metadata.precompressed.keys() == set()
            visit_image = helper.butler.get(visit_image_ref, parameters={"preserve_quantization": True})
            assert visit_image._opaque_metadata.precompressed.keys() == {"IMAGE", "VARIANCE"}
        bbox = helper.butler.get(visit_image_ref.makeComponentRef("bbox"))
        assert bbox == visit_image.bbox
        alternates = {
            k: helper.butler.get(visit_image_ref.makeComponentRef(k))
            for k in ["image", "mask", "variance", "bbox", "psf", "detector"]
        }
        compare_visit_image_to_legacy(
            visit_image,
            legacy_test_data.legacy_exposure,
            expect_view=False,
            plane_map=legacy_test_data.plane_map,
            alternates=alternates,
            **DP2_VISIT_DETECTOR_DATA_ID,
        )
        helper.butler.pruneDatasets([helper.legacy], purge=True, unstore=True, disassociate=True)
        visit_image.metadata["MixedCaseKey"] = 52
        helper.butler.put(visit_image, visit_image_ref)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*filter label mismatch.*", category=UserWarning)
            legacy_exposure = helper.butler.get(helper.legacy)
        compare_visit_image_to_legacy(
            visit_image,
            legacy_exposure,
            expect_view=False,
            plane_map=legacy_test_data.plane_map,
            alternates=alternates,
            **DP2_VISIT_DETECTOR_DATA_ID,
        )
        visit_image_2 = helper.butler.get(visit_image_ref)
        compare_visit_image_to_legacy(
            visit_image_2,
            legacy_exposure,
            expect_view=False,
            plane_map=legacy_test_data.plane_map,
            alternates=alternates,
            **DP2_VISIT_DETECTOR_DATA_ID,
        )
        assert visit_image_2.metadata["MixedCaseKey"] == 52


def test_convert_unit(legacy_test_data_calibrated: _LegacyTestData) -> None:
    """Verify convert_unit round-trips between nJy, mJy, and electron via
    photometric_scaling.
    """
    from lsst.afw.table import ExposureCatalog

    legacy_test_data = legacy_test_data_calibrated
    original = legacy_test_data.visit_image.copy()
    with pytest.raises(u.UnitConversionError):
        original.convert_unit(u.electron)
    visit_image_nJy = original.convert_unit(u.nJy, copy=False)
    assert np.may_share_memory(visit_image_nJy.image.array, original.image.array)
    assert np.may_share_memory(visit_image_nJy.variance.array, original.variance.array)
    with pytest.raises(u.UnitConversionError):
        original.convert_unit(u.mJy, copy=False)
    visit_image_mJy = original.convert_unit(u.mJy, copy="as-needed")
    assert visit_image_mJy.unit == u.mJy
    assert_close(visit_image_mJy.image.array, original.image.array * 1e-6)
    assert np.may_share_memory(visit_image_nJy.mask.array, original.mask.array)
    assert_close(visit_image_mJy.variance.array, original.variance.array * 1e-12)
    legacy_exposure_mJy = visit_image_mJy.to_legacy()
    assert_close(legacy_exposure_mJy.getPhotoCalib().getCalibrationMean(), 1e6)
    legacy_masked_image_nJy = legacy_exposure_mJy.getPhotoCalib().calibrateImage(
        legacy_exposure_mJy.maskedImage
    )
    assert_close(visit_image_nJy.image.array, legacy_masked_image_nJy.image.array)
    assert_close(visit_image_nJy.variance.array, legacy_masked_image_nJy.variance.array)
    assert np.may_share_memory(visit_image_mJy.mask.array, original.mask.array)
    assert visit_image_mJy.sky_projection is original.sky_projection
    assert visit_image_mJy.obs_info is original.obs_info
    assert visit_image_mJy.summary_stats is original.summary_stats
    assert visit_image_mJy.psf is original.psf
    assert visit_image_mJy.detector is original.detector
    assert visit_image_mJy.bounds is original.bounds
    assert visit_image_mJy.aperture_corrections is original.aperture_corrections
    assert visit_image_mJy.photometric_scaling is original.photometric_scaling
    visit_summary = ExposureCatalog.readFits(
        os.path.join(EXTERNAL_DATA_DIR, "dp2", "legacy", "visit_summary.fits")
    )
    legacy_photo_calib = visit_summary.find(DP2_VISIT_DETECTOR_DATA_ID["detector"]).getPhotoCalib()
    visit_image_nJy.photometric_scaling = field_from_legacy_photo_calib(
        legacy_photo_calib, bounds=original.detector.bbox, instrumental_unit=u.electron
    )
    compare_photo_calib_to_legacy(
        visit_image_nJy.photometric_scaling,
        legacy_test_data.legacy_exposure.getPhotoCalib(),
        applied_legacy_photo_calib=legacy_photo_calib,
        subimage_bbox=visit_image_nJy.bbox,
    )
    with pytest.raises(u.UnitConversionError):
        visit_image_nJy.convert_unit(u.mm)
    with pytest.raises(u.UnitConversionError):
        visit_image_nJy.convert_unit(u.electron, copy=False)
    legacy_masked_image_e = legacy_photo_calib.uncalibrateImage(legacy_test_data.legacy_exposure.maskedImage)
    visit_image_e = visit_image_nJy.convert_unit(u.electron)
    assert_close(visit_image_e.image.array, legacy_masked_image_e.image.array)
    assert_close(visit_image_e.variance.array, legacy_masked_image_e.variance.array)
    assert not np.may_share_memory(visit_image_e.mask.array, visit_image_nJy.mask.array)
    visit_image_mJy.photometric_scaling = visit_image_nJy.photometric_scaling
    visit_image_e = visit_image_mJy.convert_unit(u.electron)
    assert_close(visit_image_e.image.array, legacy_masked_image_e.image.array)
    assert_close(visit_image_e.variance.array, legacy_masked_image_e.variance.array)
    visit_image_nJy_2 = visit_image_e.convert_unit(u.nJy)
    assert_close(visit_image_nJy_2.image.array, visit_image_nJy.image.array)
    assert_close(visit_image_nJy_2.variance.array, original.variance.array)
    visit_image_e.photometric_scaling = visit_image_nJy.photometric_scaling * (1e-6 * u.mJy / u.nJy)
    visit_image_nJy_3 = visit_image_e.convert_unit(u.nJy)
    assert_close(visit_image_nJy_3.image.array, visit_image_nJy.image.array)
    assert_close(visit_image_nJy_3.variance.array, original.variance.array)
    legacy_exposure_e = visit_image_e.to_legacy()
    assert_close(
        legacy_exposure_e.getPhotoCalib().getCalibrationMean(),
        legacy_photo_calib.getCalibrationMean(),
    )
    legacy_masked_image_nJy = legacy_exposure_e.getPhotoCalib().calibrateImage(legacy_exposure_e.maskedImage)
    assert_close(visit_image_nJy.image.array, legacy_masked_image_nJy.image.array)
    assert_close(visit_image_nJy.variance.array, legacy_masked_image_nJy.variance.array)

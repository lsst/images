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
from pathlib import Path
from typing import Any

import astropy.io.fits
import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import Angle, SkyCoord

from lsst.images import (
    Box,
    GeneralFrame,
    Image,
    MaskedImage,
    MaskPlane,
    MaskSchema,
    NoOverlapError,
    SkyProjection,
    get_legacy_visit_image_mask_planes,
)
from lsst.images.fits import FitsCompressionOptions
from lsst.images.tests import (
    RoundtripFits,
    RoundtripJson,
    RoundtripNdf,
    assert_masked_images_equal,
    compare_masked_image_to_legacy,
)

try:
    import h5py  # noqa: F401

    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False

try:
    from lsst.afw.image import MaskedImageReader as LegacyMaskedImageReader

except ImportError:
    type LegacyMaskedImageReader = Any  # type: ignore[no-redef]

EXTERNAL_DATA_DIR = os.environ.get("TESTDATA_IMAGES_DIR", None)

skip_no_h5py = pytest.mark.skipif(not HAVE_H5PY, reason="h5py is not installed")


@dataclasses.dataclass
class _LegacyTestData:
    masked_image: MaskedImage
    reader: LegacyMaskedImageReader
    plane_map: dict[str, MaskPlane]


@pytest.fixture(scope="session")
def legacy_test_data() -> _LegacyTestData:
    """Return a Mask read directly from the legacy test dataset and a legacy
    reader for that image.

    Skips if TESTDATA_IMAGES_DIR is unset or lsst.afw.image is unavailable.
    """
    if EXTERNAL_DATA_DIR is None:
        pytest.skip("TESTDATA_IMAGES_DIR is not in the environment.")
    try:
        from lsst.afw.image import MaskedImageFitsReader
    except ImportError:
        pytest.skip("'lsst.afw.image' could not be imported.")
    filename = os.path.join(EXTERNAL_DATA_DIR, "dp2", "legacy", "visit_image.fits")
    plane_map = get_legacy_visit_image_mask_planes()
    masked_image = MaskedImage.read_legacy(filename, plane_map=plane_map)
    reader = MaskedImageFitsReader(filename)
    return _LegacyTestData(masked_image=masked_image, reader=reader, plane_map=plane_map)


def _make_wcs() -> astropy.wcs.WCS:
    """Build a gnomonic FITS WCS with 0.1 arcsec pixels at (12, 13) deg.

    The reference pixel is at 0-based pixel (x=5, y=6).
    """
    wcs = astropy.wcs.WCS(naxis=2)
    # FITS CRPIX is 1-based, so CRPIX (6, 7) is 0-based pixel (x=5, y=6).
    wcs.wcs.crpix = [6.0, 7.0]
    wcs.wcs.crval = [12.0, 13.0]
    scale = 0.1 / 3600.0
    wcs.wcs.cd = [[-scale, 0.0], [0.0, scale]]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs


def make_masked_image() -> MaskedImage:
    """Return a freshly-constructed MaskedImage with BAD and HUNGRY mask
    planes set.
    """
    rng = np.random.default_rng(500)
    masked_image = MaskedImage(
        Image(rng.normal(100.0, 8.0, size=(200, 251)), dtype=np.float64, unit=u.nJy, yx0=(5, 8)),
        mask_schema=MaskSchema(
            [
                MaskPlane("BAD", "Pixel is very bad, possibly downright evil."),
                MaskPlane("HUNGRY", "Pixel hasn't had enough to eat today."),
            ]
        ),
        metadata={"fifty": "5 * 10"},
        sky_projection=SkyProjection.from_fits_wcs(_make_wcs(), GeneralFrame(unit=u.pix)),
    )
    masked_image.mask.array |= np.multiply.outer(
        masked_image.image.array < 102.0,
        masked_image.mask.schema.bitmask("BAD"),
    )
    masked_image.mask.array |= np.multiply.outer(
        masked_image.image.array > 98.0,
        masked_image.mask.schema.bitmask("HUNGRY"),
    )
    masked_image.variance.array = rng.normal(64.0, 0.5, size=masked_image.bbox.shape)
    return masked_image


def test_construction() -> None:
    """Verify the MaskedImage constructed by make_masked_image has the
    expected attributes.
    """
    mi = make_masked_image()
    assert mi.bbox == Box.factory[5:205, 8:259]
    assert mi.mask.bbox == mi.bbox
    assert mi.variance.bbox == mi.bbox
    assert mi.image.array.shape == mi.bbox.shape
    assert mi.mask.array.shape == mi.bbox.shape + (1,)
    assert mi.variance.array.shape == mi.bbox.shape
    assert mi.unit == u.nJy
    assert mi.variance.unit == u.nJy**2
    assert mi.metadata == {"fifty": "5 * 10"}
    # The checks below are subject to the vagaries of the RNG, but we want
    # the seed to be such that they all pass, or other tests will be weaker.
    assert np.sum(mi.mask.array == mi.mask.schema.bitmask("BAD")) > 0
    assert np.sum(mi.mask.array == mi.mask.schema.bitmask("HUNGRY")) > 0
    assert np.sum(mi.mask.array == mi.mask.schema.bitmask("BAD", "HUNGRY")) > 0

    assert mi[...] is not mi
    assert str(mi) == "MaskedImage(Image([y=5:205, x=8:259], float64), ['BAD', 'HUNGRY'])"
    assert (
        repr(mi)
        == "MaskedImage(Image(..., bbox=Box(y=Interval(start=5, stop=205), x=Interval(start=8, stop=259)), "
        "dtype=dtype('float64')), mask_schema=MaskSchema([MaskPlane(name='BAD', description='Pixel is "
        "very bad, possibly downright evil.'), MaskPlane(name='HUNGRY', description=\"Pixel hasn't had "
        "enough to eat today.\")], dtype=dtype('uint8')))"
    )
    copy = mi.copy()
    original = mi.image.array[0, 0]
    copy.image.array[0, 0] = 38.0
    assert mi.image.array[0, 0] == original
    assert copy.image.array[0, 0] == 38.0

    # Test error conditions.
    with pytest.raises(ValueError):
        # Disagreement over mask bbox.
        MaskedImage(Image(42.0, shape=(5, 6)), mask=mi.mask)
    with pytest.raises(TypeError):
        # No mask definition.
        MaskedImage(mi.image, variance=mi.variance)
    with pytest.raises(TypeError):
        # Can not provide mask and mask schema.
        MaskedImage(
            Image(42.0, shape=(5, 5)),
            mask=mi.mask,
            mask_schema=mi.mask.schema,
        )
    with pytest.raises(ValueError):
        # image and variance bbox disagreement.
        MaskedImage(
            Image(42.0, shape=(5, 5)),
            mask_schema=mi.mask.schema,
            variance=mi.variance,
        )
    with pytest.raises(ValueError):
        # no image unit but there is variance unit.
        MaskedImage(
            Image(42.0, shape=(5, 5)),
            mask_schema=mi.mask.schema,
            variance=Image(1.0, shape=(5, 5), unit=u.nJy),
        )
    with pytest.raises(ValueError):
        # image and variance units disagree.
        MaskedImage(
            Image(42.0, shape=(5, 5), unit=u.nJy),
            mask_schema=mi.mask.schema,
            variance=Image(1.0, shape=(5, 5), unit=u.nJy),
        )


def test_subset() -> None:
    """Verify assignment of a subset into a MaskedImage copy."""
    mi = make_masked_image()
    copy = mi.copy()
    subset = copy.local[0:10, 20:30].copy()
    subset.image[...] = Image(42.0, shape=(10, 10), unit=u.nJy)
    copy[subset.bbox] = subset
    assert copy.image.array[0, 20] == 42.0
    assert copy.image.array[0, 0] == mi.image.array[0, 0]


def test_mask_setter() -> None:
    """Verify the mask plane can be replaced with one grown by add_plane."""
    mi = make_masked_image()
    bad = mi.mask.get("BAD")
    mi.mask = mi.mask.add_plane("OUTSIDE_STENCIL", "Pixel lies outside the stencil.")
    assert "OUTSIDE_STENCIL" in mi.mask.schema.names
    assert mi.mask.bbox == mi.image.bbox
    np.testing.assert_array_equal(mi.mask.get("BAD"), bad)
    assert not mi.mask.get("OUTSIDE_STENCIL").any()
    # A mask whose bounding box disagrees with the image is rejected.
    with pytest.raises(ValueError):
        mi.mask = mi.mask[Box.factory[10:20, 12:22]]


def test_fits_roundtrip() -> None:
    """Verify MaskedImage round-trips correctly through FITS, including
    subimage reads.
    """
    mi = make_masked_image()
    subbox = Box.factory[11:20, 25:30]
    subslices = (slice(6, 15), slice(17, 22))
    np.testing.assert_array_equal(mi.image.array[subslices], mi.image[subbox].array)
    with RoundtripFits(mi, "MaskedImageV2") as roundtrip:
        subimage = roundtrip.get(bbox=subbox)
        # Check that we used lossless compression (the default).
        fits = roundtrip.inspect()
        assert fits[1].header["ZCMPTYPE"] == "GZIP_2"
        assert fits[2].header["ZCMPTYPE"] == "GZIP_2"
        assert fits[3].header["ZCMPTYPE"] == "GZIP_2"
    assert_masked_images_equal(roundtrip.result, mi, expect_view=False)
    assert_masked_images_equal(subimage, roundtrip.result[subbox], expect_view=False)


def test_fits_roundtrip_legacy_read() -> None:
    """Verify a round-tripped MaskedImageV2 can be read back as a legacy afw
    MaskedImage.
    """
    try:
        import lsst.afw.image
    except ImportError:
        pytest.skip("afw could not be imported")
    mi = make_masked_image()
    with RoundtripFits(mi, "MaskedImageV2") as roundtrip:
        legacy_masked_image = roundtrip.get(storageClass="MaskedImage")
        assert isinstance(legacy_masked_image, lsst.afw.image.MaskedImage)
        compare_masked_image_to_legacy(mi, legacy_masked_image, expect_view=False)


def test_fits_roundtrip_lossy(tmp_path: Path) -> None:
    """Verify MaskedImage round-trips correctly through FITS with lossy
    compression.
    """
    mi = make_masked_image()
    subbox = Box.factory[11:20, 25:30]
    subslices = (slice(6, 15), slice(17, 22))
    np.testing.assert_array_equal(mi.image.array[subslices], mi.image[subbox].array)
    path = tmp_path / "lossy.fits"
    mi.write(
        path,
        compression_options={
            "image": FitsCompressionOptions.LOSSY,
            "variance": FitsCompressionOptions.LOSSY,
        },
        compression_seed=50,
    )
    roundtripped = MaskedImage.read(path)
    subimage = MaskedImage.read(path, bbox=subbox)
    with astropy.io.fits.open(path, disable_image_compression=True) as fits:
        assert fits[1].header["ZCMPTYPE"] == "RICE_1"
        assert fits[2].header["ZCMPTYPE"] == "GZIP_2"
        assert fits[3].header["ZCMPTYPE"] == "RICE_1"
    assert_masked_images_equal(roundtripped, mi, expect_view=False, rtol=0.01)
    assert_masked_images_equal(subimage, roundtripped[subbox], expect_view=False)


@skip_no_h5py
def test_round_trip_ndf_compatible_mask() -> None:
    """Verify NDF round-trip for a MaskedImage with ≤8 mask planes."""
    mi = make_masked_image()
    with RoundtripNdf(mi, "MaskedImageV2") as roundtrip:
        assert_masked_images_equal(roundtrip.result, mi, expect_view=False)


@skip_no_h5py
def test_round_trip_ndf_incompatible_mask() -> None:
    """Verify NDF round-trip for a MaskedImage with more than 8 mask planes."""
    rng = np.random.default_rng(7)
    planes = [MaskPlane(f"P{i}", f"plane {i}") for i in range(12)]
    wide = MaskedImage(
        Image(
            rng.normal(100.0, 8.0, size=(50, 60)),
            dtype=np.float64,
            unit=u.nJy,
            yx0=(0, 0),
        ),
        mask_schema=MaskSchema(planes),
    )
    wide.variance.array = rng.normal(64.0, 0.5, size=wide.bbox.shape)
    with RoundtripNdf(wide, "MaskedImageV2") as roundtrip:
        assert_masked_images_equal(roundtrip.result, wide, expect_view=False)


@skip_no_h5py
def test_round_trip_ndf_many_plane_mask() -> None:
    """Verify NDF round-trip for a mask that needs more than one int32
    chunk.
    """
    rng = np.random.default_rng(11)
    planes = [MaskPlane(f"P{i}", f"plane {i}") for i in range(40)]
    wide = MaskedImage(
        Image(
            rng.normal(100.0, 8.0, size=(10, 12)),
            dtype=np.float64,
            unit=u.nJy,
            yx0=(0, 0),
        ),
        mask_schema=MaskSchema(planes),
    )
    wide.mask.set("P0", wide.image.array > 100.0)
    wide.mask.set("P17", wide.image.array < 95.0)
    wide.mask.set("P39", wide.image.array > 110.0)
    wide.variance.array = rng.normal(64.0, 0.5, size=wide.bbox.shape)
    with RoundtripNdf(wide, "MaskedImageV2") as roundtrip:
        assert_masked_images_equal(roundtrip.result, wide, expect_view=False)


@skip_no_h5py
def test_fits_ndf_consistency() -> None:
    """Verify FITS and NDF backends produce equal MaskedImages on
    round-trip.
    """
    mi = make_masked_image()
    with (
        RoundtripFits(mi) as fits_rt,
        RoundtripNdf(mi) as ndf_rt,
    ):
        assert_masked_images_equal(mi, fits_rt.result, expect_view=False)
        assert_masked_images_equal(mi, ndf_rt.result, expect_view=False)
        assert_masked_images_equal(fits_rt.result, ndf_rt.result, expect_view=False)


def test_fits_json_consistency() -> None:
    """Verify FITS and JSON backends produce equal MaskedImages on
    round-trip.
    """
    mi = make_masked_image()
    with (
        RoundtripFits(mi) as fits_rt,
        RoundtripJson(mi) as json_rt,
    ):
        assert_masked_images_equal(mi, fits_rt.result, expect_view=False)
        assert_masked_images_equal(mi, json_rt.result, expect_view=False)
        assert_masked_images_equal(fits_rt.result, json_rt.result, expect_view=False)


def test_legacy(legacy_test_data: _LegacyTestData) -> None:
    """Test MaskedImage.read_legacy, MaskedImage.to_legacy, and
    MaskedImage.from_legacy.
    """
    legacy_masked_image = legacy_test_data.reader.read()
    compare_masked_image_to_legacy(
        legacy_test_data.masked_image,
        legacy_masked_image,
        plane_map=legacy_test_data.plane_map,
        expect_view=False,
    )
    compare_masked_image_to_legacy(
        legacy_test_data.masked_image,
        legacy_test_data.masked_image.to_legacy(plane_map=legacy_test_data.plane_map),
        plane_map=legacy_test_data.plane_map,
        expect_view=True,
    )
    compare_masked_image_to_legacy(
        MaskedImage.from_legacy(legacy_masked_image, plane_map=legacy_test_data.plane_map),
        legacy_masked_image,
        expect_view=True,
        plane_map=legacy_test_data.plane_map,
    )


def test_sky_circle_bbox() -> None:
    """Test that we can extract a bounding box from a sky circle."""
    mi = make_masked_image()

    # This position is on the reference pixel (x=5, y=6), which is just
    # outside the image (the bbox starts at x=8, y=5), so the box must be
    # clipped on the low-x and low-y sides. 0.1 arcsec pixels.
    bbox = mi.bbox_from_sky_circle(
        SkyCoord(ra=12.0 * u.deg, dec=13.0 * u.deg, frame="icrs"), Angle(1.0 * u.arcsec), clip=True
    )
    # The circle has a ~10 pixel radius (1 arcsec at 0.1 arcsec per pixel),
    # spanning x [-5, 15] and y [-4, 16] before clipping to the image bounds.
    assert bbox == Box.factory[5:17, 8:16]

    with pytest.raises(NoOverlapError):
        # Partially off the edge but clipping not requested.
        mi.bbox_from_sky_circle(
            SkyCoord(ra=12.0 * u.deg, dec=13.0 * u.deg, frame="icrs"), Angle(1.0 * u.arcsec)
        )

    # Fully inside the image. The image is only ~200 pixels across or
    # ~20 arcsec.
    bbox = mi.bbox_from_sky_circle(
        SkyCoord(ra=12.0 * u.deg - 5.0 * u.arcsec, dec=13.0 * u.deg + 5.0 * u.arcsec, frame="icrs"),
        Angle(1.0 * u.arcsec),
    )
    # The center is offset from the reference pixel by +50 pixels in y and
    # by +48.7 pixels in x (the 5 arcsec RA offset scales by cos(dec)),
    # placing it at (x=53.7, y=56) with a ~10 pixel radius.
    assert bbox == Box.factory[46:67, 44:65]

    # Fully off the image.
    with pytest.raises(NoOverlapError):
        mi.bbox_from_sky_circle(
            SkyCoord(ra=13.0 * u.deg, dec=13.0 * u.deg, frame="icrs"), Angle(1.0 * u.arcsec)
        )

    # Fully off the image with clipping requested.
    with pytest.raises(NoOverlapError):
        mi.bbox_from_sky_circle(
            SkyCoord(ra=13.0 * u.deg, dec=13.0 * u.deg, frame="icrs"), Angle(1.0 * u.arcsec), clip=True
        )

    # Non-scalar center and radius are rejected.
    with pytest.raises(ValueError, match="scalar SkyCoord"):
        mi.bbox_from_sky_circle(
            SkyCoord(ra=[12.0, 12.1] * u.deg, dec=[13.0, 13.1] * u.deg, frame="icrs"),
            Angle(1.0 * u.arcsec),
        )
    with pytest.raises(ValueError, match="scalar Angle"):
        mi.bbox_from_sky_circle(
            SkyCoord(ra=12.0 * u.deg, dec=13.0 * u.deg, frame="icrs"), Angle([1.0, 2.0] * u.arcsec)
        )

    # An image without a sky projection cannot calculate a bounding box.
    no_wcs = Image(0.0, shape=(10, 10), dtype=np.float64)
    with pytest.raises(ValueError, match="sky projection"):
        no_wcs.bbox_from_sky_circle(
            SkyCoord(ra=12.0 * u.deg, dec=13.0 * u.deg, frame="icrs"), Angle(1.0 * u.arcsec)
        )

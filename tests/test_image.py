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
from typing import TYPE_CHECKING, Any

import astropy.io.fits
import astropy.units as u
import numpy as np
import pytest

import lsst.utils.tests
from lsst.images import Box, DetectorFrame, Image
from lsst.images.tests import (
    RoundtripFits,
    RoundtripJson,
    RoundtripNdf,
    assert_close,
    assert_images_equal,
    assert_sky_projections_equal,
    compare_image_to_legacy,
    make_random_sky_projection,
)

try:
    import h5py  # noqa: F401

    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False

if TYPE_CHECKING:
    try:
        from lsst.afw.image import MaskedImageReader as LegacyMaskedImageReader
    except ImportError:
        type LegacyMaskedImageReader = Any  # type: ignore[no-redef]

EXTERNAL_DATA_DIR = os.environ.get("TESTDATA_IMAGES_DIR", None)

skip_no_h5py = pytest.mark.skipif(not HAVE_H5PY, reason="h5py is not installed")


@dataclasses.dataclass
class _LegacyTestData:
    image: Image
    reader: LegacyMaskedImageReader


@pytest.fixture(scope="session")
def legacy_test_data() -> _LegacyTestData:
    """Return an Image read directly from the legacy test dataset and a legacy
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
    det_frame = DetectorFrame(instrument="Inst", visit=1234, detector=1, bbox=Box.factory[1:4096, 1:4096])
    image = Image.read_legacy(filename, preserve_quantization=True, fits_wcs_frame=det_frame)
    reader = MaskedImageFitsReader(filename)
    return _LegacyTestData(image, reader)


def test_basics() -> None:
    """Test basic Image constructor patterns and slicing."""
    image = Image(42, shape=(5, 5), metadata={"three": 3})
    assert_close(image.array, np.zeros([5, 5], dtype=np.int64) + 42)
    assert image.metadata["three"] == 3

    data = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    image = Image(data)
    subset = image[Box.factory[:3, 1:3]]
    subset2 = image.absolute[:3, 1:3]
    assert_images_equal(subset2, subset, expect_view=True)

    assert_images_equal(image.copy(), image, expect_view=False)

    # Add an explicit bounding box and then slice it.
    image = Image(data, bbox=Box.factory[-2:1, 10:14])
    with pytest.raises(IndexError):
        # Same slice no longer works in absolute slicing because we have
        # moved origin.
        image.absolute[:3, 1:3]
    # That slice does still work in local coordinates.
    assert_close(image.local[:3, 1:3].array, subset2.array)
    # And we can write an equivalent slice in absolute coordinates.
    assert_close(image.absolute[:0, 11:13].array, np.array([[2, 3], [6, 7]]))

    # Test __eq__ behavior.
    assert image[...] == image
    assert image.__eq__(data) == NotImplemented
    assert image != list(data)

    with pytest.raises(ValueError):
        # bbox does not match array shape.
        Image(np.array([[1, 2, 3], [4, 5, 6]]), bbox=Box.factory[0:2, 0:4])

    with pytest.raises(ValueError):
        # shape does not match array shape.
        Image(np.array([[2, 3, 4], [6, 7, 8]]), shape=[5, 2])

    with pytest.raises(TypeError):
        # shape and bbox both None.
        Image()

    with pytest.raises(ValueError):
        # Shape mismatch.
        Image(shape=[3, 6], bbox=Box.factory[-5:10, 0:10])


def test_json_roundtrip() -> None:
    """Test saving a tiny image to pure JSON."""
    image = Image(
        np.arange(15).reshape(5, 3),
        yx0=(2, -1),
    )
    with RoundtripJson(image, "ImageV2") as roundtrip:
        pass
    assert_images_equal(image, roundtrip.result)


def test_fits_roundtrip() -> None:
    """Test saving a tiny image to FITS generically."""
    image = Image(
        np.arange(15).reshape(5, 3),
        yx0=(2, -1),
    )
    with RoundtripFits(image, "ImageV2") as roundtrip:
        subbox = Box.factory[3:5, 0:1]
        assert_images_equal(image[subbox], roundtrip.get(bbox=subbox))
    assert_images_equal(image, roundtrip.result)


@skip_no_h5py
def test_ndf_roundtrip() -> None:
    """Test saving a tiny image to NDF."""
    image = Image(
        np.arange(15).reshape(5, 3),
        yx0=(2, -1),
    )
    with RoundtripNdf(image, "ImageV2") as roundtrip:
        pass
    assert_images_equal(image, roundtrip.result)


@skip_no_h5py
def test_fits_ndf_consistency() -> None:
    """Verify FITS and NDF round-trips produce equal Images."""
    rng = np.random.default_rng(321)
    image = Image(
        rng.normal(100.0, 8.0, size=(60, 80)),
        dtype=np.float64,
        unit=u.nJy,
        yx0=(0, 0),
    )
    with RoundtripFits(image) as fits_rt, RoundtripNdf(image) as ndf_rt:
        assert_images_equal(image, fits_rt.result)
        assert_images_equal(image, ndf_rt.result)
        assert_images_equal(fits_rt.result, ndf_rt.result)


def test_fits_json_consistency() -> None:
    """Verify FITS and JSON round-trips produce equal Images."""
    rng = np.random.default_rng(321)
    image = Image(
        rng.normal(100.0, 8.0, size=(60, 80)),
        dtype=np.float64,
        unit=u.nJy,
        yx0=(0, 0),
    )
    with RoundtripFits(image) as fits_rt, RoundtripJson(image) as json_rt:
        assert_images_equal(image, fits_rt.result)
        assert_images_equal(image, json_rt.result)
        assert_images_equal(fits_rt.result, json_rt.result)


def test_quantity() -> None:
    """Test quantity getter and setter on Image."""
    data = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])
    data2 = data.copy() * 2.0
    image = Image(data, unit=u.mJy, bbox=Box.factory[-2:1, 3:7])

    q = image.quantity
    assert q[1, 0] == 5.0 * u.mJy
    image.quantity = image.array * 10.0 * u.uJy
    q = image.quantity
    assert q[1, 0] == 0.05 * u.mJy

    image2 = Image(data2, unit=u.Jy)
    image[Box.factory[-1:0, 5:7]] = image2.local[1:2, 2:4]
    assert_close(
        image.array,
        np.array([[0.01, 0.02, 0.03, 0.04], [0.05, 0.06, 14000.0, 16000.0], [0.09, 0.1, 0.11, 0.12]]),
    )


def test_read_write() -> None:
    """Test round-trip through file using GeneralizedImage.write / read."""
    data = np.array([[1.0, 2.0, np.nan, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])
    md: dict[str, Any] = {"int": 1, "float": 42.0, "bool": False, "long string header": "This is a string"}
    det_frame = DetectorFrame(instrument="Inst", visit=1234, detector=1, bbox=Box.factory[1:4096, 1:4096])
    rng = np.random.default_rng(500)
    sky_projection = make_random_sky_projection(rng, det_frame, Box.factory[1:4096, 1:4096])

    image = Image(
        data,
        unit=u.dn,
        metadata=md,
        bbox=Box.factory[-2:1, 3:7],
        sky_projection=sky_projection,
    )

    with lsst.utils.tests.getTempFilePath(".fits") as tmpFile:
        image.write(tmpFile)

        new = Image.read(tmpFile)
        assert new == image

        # __eq__ does not test all components.
        assert new.metadata == image.metadata
        assert_sky_projections_equal(new.sky_projection, image.sky_projection, expect_identity=False)

        # Read subset.
        subset = Image.read(tmpFile, bbox=Box.factory[-2:0, 5:7])
        assert subset == image.absolute[-2:0, 5:7]
        assert subset == image.local[0:2, 2:4]
        assert str(subset) == "Image([y=-2:0, x=5:7], float64)"
        assert (
            repr(subset) == "Image(..., bbox=Box(y=Interval(start=-2, stop=0), x=Interval(start=5, stop=7)), "
            "dtype=dtype('float64'))"
        )

        # Check that WCS headers were written out.
        with astropy.io.fits.open(tmpFile) as hdul:
            hdu1 = hdul[1]
            hdr1 = hdu1.header
            assert hdr1["CTYPE1"] == "RA---TAN"


def test_legacy(legacy_test_data: _LegacyTestData) -> None:
    """Test Image.read_legacy, Image.to_legacy, and Image.from_legacy."""
    legacy_image = legacy_test_data.reader.readImage()
    compare_image_to_legacy(legacy_test_data.image, legacy_image, expect_view=False)
    # Converting back to afw will not share memory, because
    # preserve_quantization=True makes the array read-only and to_legacy
    # has to copy in that case.
    compare_image_to_legacy(legacy_test_data.image, legacy_test_data.image.to_legacy(), expect_view=False)
    # Converting from afw will always share memory.
    image_view = Image.from_legacy(legacy_image)
    compare_image_to_legacy(image_view, legacy_image, expect_view=True)
    # Converting back to afw from the in-memory view will be another view.
    compare_image_to_legacy(image_view, image_view.to_legacy(), expect_view=True)
    # Write the image out in the new format, and test that we can read it
    # back either way.
    with RoundtripFits(legacy_test_data.image, storage_class="ImageV2") as roundtrip:
        pass
    assert_images_equal(roundtrip.result, legacy_test_data.image, expect_view=False)


def test_legacy_butler_read(legacy_test_data: _LegacyTestData) -> None:
    """Test that a round-tripped ImageV2 can be read back as a legacy afw
    Image via Butler.
    """
    with RoundtripFits(legacy_test_data.image, storage_class="ImageV2") as roundtrip:
        legacy_image = roundtrip.get(storageClass="Image")
        assert isinstance(legacy_image, lsst.afw.image.Image)
        compare_image_to_legacy(legacy_test_data.image, legacy_image, expect_view=False)

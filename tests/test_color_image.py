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

import numpy as np
import pytest

from lsst.images import Box, ColorImage, Image, TractFrame
from lsst.images.tests import (
    RoundtripFits,
    RoundtripJson,
    RoundtripNdf,
    assert_images_equal,
    assert_sky_projections_equal,
    make_random_sky_projection,
)

try:
    import h5py

    from lsst.images.ndf import _hds

    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False

skip_no_h5py = pytest.mark.skipif(not HAVE_H5PY, reason="h5py is not installed")


def _make_pixel_frame() -> TractFrame:
    """Return a TractFrame for use in ColorImage tests."""
    return TractFrame(skymap="test_skymap", tract=33, bbox=Box.factory[:50, :64])


def _make_color_image() -> tuple[ColorImage, np.ndarray]:
    """Return a ColorImage and its backing uint8 array."""
    rng = np.random.default_rng(500)
    pixel_frame = _make_pixel_frame()
    bbox = Box.factory[20:25, 40:48]
    sky_projection = make_random_sky_projection(rng, pixel_frame, pixel_frame.bbox)
    array = rng.integers(low=0, high=255, size=bbox.shape + (3,), dtype=np.uint8)
    return ColorImage(array, bbox=bbox, sky_projection=sky_projection), array


def assert_color_images_equal(a: ColorImage, b: ColorImage, expect_view: bool | None = None) -> None:
    """Assert that two ColorImages have equal sky projections and
    pixel data.
    """
    assert_sky_projections_equal(a.sky_projection, b.sky_projection)
    if expect_view is not None:
        assert np.may_share_memory(a.array, b.array) == expect_view
    if not expect_view:
        np.testing.assert_array_equal(a.array, b.array)


def test_properties() -> None:
    """Test that ColorImage properties match the construction arguments."""
    color_image, array = _make_color_image()
    bbox = Box.factory[20:25, 40:48]
    assert color_image.bbox == bbox
    assert np.may_share_memory(color_image.array, array)
    assert_images_equal(
        color_image.red,
        Image(array[:, :, 0], bbox=bbox, sky_projection=color_image.sky_projection),
        expect_view="array",
    )
    assert_images_equal(
        color_image.green,
        Image(array[:, :, 1], bbox=bbox, sky_projection=color_image.sky_projection),
        expect_view="array",
    )
    assert_images_equal(
        color_image.blue,
        Image(array[:, :, 2], bbox=bbox, sky_projection=color_image.sky_projection),
        expect_view="array",
    )


def test_constructor() -> None:
    """Test that alternate ColorImage constructors produce equivalent
    objects.
    """
    color_image, array = _make_color_image()
    assert_color_images_equal(
        ColorImage(array, yx0=color_image.bbox.start, sky_projection=color_image.sky_projection),
        color_image,
        expect_view=True,
    )
    assert_color_images_equal(
        ColorImage.from_channels(
            color_image.red,
            color_image.green,
            color_image.blue,
            sky_projection=color_image.sky_projection,
        ),
        color_image,
        expect_view=False,
    )


def test_fits_roundtrip() -> None:
    """Test that ColorImage round-trips correctly through FITS."""
    color_image, _ = _make_color_image()
    with RoundtripFits(color_image, "ColorImage") as roundtrip:
        pass
    assert_color_images_equal(roundtrip.result, color_image, expect_view=False)


@skip_no_h5py
def test_ndf_roundtrip() -> None:
    """Test that ColorImage round-trips correctly through NDF."""
    color_image, _ = _make_color_image()
    with RoundtripNdf(color_image, "ColorImage") as roundtrip:
        pass
    assert_color_images_equal(roundtrip.result, color_image, expect_view=False)


# TODO[DM-54956]: instead of checking for consistency, we should just have
# an independent test of the JSON archive, as we do for the others.
def test_fits_json_consistency() -> None:
    """Test that the FITS and JSON backends produce equal ColorImages on
    round-trip.
    """
    color_image, _ = _make_color_image()
    with (
        RoundtripFits(color_image) as fits_rt,
        RoundtripJson(color_image) as json_rt,
    ):
        assert_color_images_equal(fits_rt.result, color_image, expect_view=False)
        assert_color_images_equal(json_rt.result, color_image, expect_view=False)
        assert_color_images_equal(fits_rt.result, json_rt.result, expect_view=False)


@skip_no_h5py
def test_ndf_layout() -> None:
    """Test that NDF output has the expected top-level structure with RGB
    child NDFs.
    """
    color_image, array = _make_color_image()
    with RoundtripNdf(color_image, "ColorImage") as roundtrip:
        f = roundtrip.inspect()
        assert _cls(f["/"]) == "EXT"
        assert "LSST" in f
        assert "JSON" in f["/LSST"]
        assert "MORE" not in f
        for channel, index in (("RED", 0), ("GREEN", 1), ("BLUE", 2)):
            assert channel in f
            assert _cls(f[channel]) == "NDF"
            assert _cls(f[f"{channel}/DATA_ARRAY"]) == "ARRAY"
            np.testing.assert_array_equal(
                f[f"{channel}/DATA_ARRAY/DATA"][()],
                array[:, :, index],
            )
            assert list(f[f"{channel}/DATA_ARRAY/ORIGIN"][()]) == [40, 20]
            assert "WCS" in f[channel]
            assert _cls(f[f"{channel}/WCS"]) == "WCS"


def _cls(node: h5py.Group) -> str:
    """Return the HDS CLASS attribute of an h5py group as a string."""
    val = node.attrs.get(_hds.ATTR_CLASS)
    if isinstance(val, bytes):
        return val.decode("ascii")
    return str(val)

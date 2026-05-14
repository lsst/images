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

import unittest

import numpy as np

from lsst.images import Box, ColorImage, Image, TractFrame
from lsst.images.tests import (
    RoundtripFits,
    RoundtripNdf,
    assert_images_equal,
    assert_projections_equal,
    make_random_projection,
)

try:
    import h5py

    from lsst.images.ndf import _hds

    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False


class ColorImageTestCase(unittest.TestCase):
    """Tests for ColorImage."""

    def setUp(self) -> None:
        self.maxDiff = None
        self.rng = np.random.default_rng(500)
        self.pixel_frame = TractFrame(skymap="test_skymap", tract=33, bbox=Box.factory[:50, :64])
        self.bbox = Box.factory[20:25, 40:48]
        self.projection = make_random_projection(self.rng, self.pixel_frame, self.pixel_frame.bbox)
        self.array = self.rng.integers(low=0, high=255, size=self.bbox.shape + (3,), dtype=np.uint8)
        self.color_image = ColorImage(self.array, bbox=self.bbox, projection=self.projection)

    def test_properties(self) -> None:
        """Test the properties of the nominal ColorImage constructed in
        setUp.
        """
        self.assertEqual(self.color_image.bbox, self.bbox)
        self.assertTrue(np.may_share_memory(self.color_image.array, self.array))
        assert_images_equal(
            self,
            self.color_image.red,
            Image(self.array[:, :, 0], bbox=self.bbox, projection=self.projection),
            expect_view="array",
        )
        assert_images_equal(
            self,
            self.color_image.green,
            Image(self.array[:, :, 1], bbox=self.bbox, projection=self.projection),
            expect_view="array",
        )
        assert_images_equal(
            self,
            self.color_image.blue,
            Image(self.array[:, :, 2], bbox=self.bbox, projection=self.projection),
            expect_view="array",
        )
        assert_projections_equal(self, self.color_image.projection, self.projection, expect_identity=True)

    def test_constructor(self) -> None:
        """Test alternate constructor arguments."""
        self.assert_color_images_equal(
            ColorImage(self.array, start=self.bbox.start, projection=self.projection),
            self.color_image,
            expect_view=True,
        )
        self.assert_color_images_equal(
            ColorImage.from_channels(
                self.color_image.red,
                self.color_image.green,
                self.color_image.blue,
                projection=self.projection,
            ),
            self.color_image,
            expect_view=False,
        )

    def test_fits_roundtrip(self) -> None:
        """Test round-tripping through FITS, via the butler if available."""
        with RoundtripFits(self, self.color_image, "ColorImage") as roundtrip:
            pass
        self.assert_color_images_equal(roundtrip.result, self.color_image, expect_view=False)

    @unittest.skipUnless(HAVE_H5PY, "h5py is not installed")
    def test_ndf_roundtrip(self) -> None:
        """Test round-tripping through NDF."""
        with RoundtripNdf(self, self.color_image) as roundtrip:
            pass
        self.assert_color_images_equal(roundtrip.result, self.color_image, expect_view=False)

    @unittest.skipUnless(HAVE_H5PY, "h5py is not installed")
    def test_ndf_layout(self) -> None:
        """ColorImage writes a top-level container with RGB child NDFs."""
        with RoundtripNdf(self, self.color_image) as roundtrip:
            f = roundtrip.inspect()
            self.assertEqual(_cls(f["/"]), "EXT")
            self.assertIn("LSST", f)
            self.assertIn("JSON", f["/LSST"])
            self.assertNotIn("MORE", f)
            for channel, index in (("RED", 0), ("GREEN", 1), ("BLUE", 2)):
                with self.subTest(channel=channel):
                    self.assertIn(channel, f)
                    self.assertEqual(_cls(f[channel]), "NDF")
                    self.assertEqual(_cls(f[f"{channel}/DATA_ARRAY"]), "ARRAY")
                    np.testing.assert_array_equal(
                        f[f"{channel}/DATA_ARRAY/DATA"][()],
                        self.array[:, :, index],
                    )
                    self.assertEqual(list(f[f"{channel}/DATA_ARRAY/ORIGIN"][()]), [40, 20])
                    self.assertIn("WCS", f[channel])
                    self.assertEqual(_cls(f[f"{channel}/WCS"]), "WCS")

    def assert_color_images_equal(
        self, a: ColorImage, b: ColorImage, expect_view: bool | None = None
    ) -> None:
        """Check that the given ColorImage matches the nominal one constructed
        in setUp.
        """
        assert_projections_equal(self, a.projection, b.projection)
        if expect_view is not None:
            self.assertEqual(np.may_share_memory(a.array, b.array), expect_view)
        if not expect_view:
            np.testing.assert_array_equal(a.array, b.array)


def _cls(node: h5py.Group) -> str:
    val = node.attrs.get(_hds.ATTR_CLASS)
    if isinstance(val, bytes):
        return val.decode("ascii")
    return str(val)


if __name__ == "__main__":
    unittest.main()

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

from lsst.images import Box, Image, Interval

try:
    import lsst.afw.image
    import lsst.geom
except ImportError:
    skip_all = True
else:
    skip_all = False


@unittest.skipIf(skip_all, "lsst legacy packages could not be imported.")
class LegacyConversionTestCase(unittest.TestCase):
    """Tests for conversions between lsst.images and their corresponding
    legacy types.
    """

    def setUp(self):
        self.maxDiff = None
        self.rng = np.random.default_rng(500)

    def test_interval(self) -> None:
        i = Interval.factory[3:6]
        j = i.to_legacy()
        self.assertIsInstance(j, lsst.geom.IntervalI)
        self.assertEqual(j.min, 3)
        self.assertEqual(j.max, 5)
        k = Interval.from_legacy(j)
        self.assertEqual(i, k)

    def test_box(self) -> None:
        b = Box.factory[3:6, -2:1]
        c = b.to_legacy()
        self.assertIsInstance(c, lsst.geom.Box2I)
        self.assertEqual(c.y.min, 3)
        self.assertEqual(c.y.max, 5)
        self.assertEqual(c.x.min, -2)
        self.assertEqual(c.x.max, 0)
        d = Box.from_legacy(c)
        self.assertEqual(b, d)

    def test_image(self) -> None:
        i = Image(self.rng.normal(100.0, 8.0, size=(200, 251)), dtype=np.float64, start=(5, 8))
        j = i.to_legacy()
        self.assertIsInstance(j, lsst.afw.image.ImageD)
        self.assertEqual(Box.from_legacy(j.getBBox()), i.bbox)
        np.testing.assert_array_equal(i.array, j.array)
        k = Image.from_legacy(j)
        self.assertEqual(i, k)


if __name__ == "__main__":
    unittest.main()

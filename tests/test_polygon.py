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

from lsst.images import Box, Polygon

try:
    import lsst.afw.geom  # noqa: F401

    have_legacy = True
except ImportError:
    have_legacy = False


class SimplePolygonTestCase(unittest.TestCase):
    """Tests for the SimplePolygon class."""

    def setUp(self) -> None:
        # A quadrilateral that's almost a box, so it's easy to reason about.
        self.x_vertices = [32.0, 31.0, 50.0, 53.0]
        self.y_vertices = [-5.0, 7.0, 7.2, -4.8]
        self.polygon = Polygon(x_vertices=self.x_vertices, y_vertices=self.y_vertices)

    def test_vertices(self) -> None:
        """Test the vertices accessors."""
        self.assertEqual(self.polygon.n_vertices, 4)
        np.testing.assert_array_equal(self.polygon.x_vertices, np.asarray(self.x_vertices))
        np.testing.assert_array_equal(self.polygon.y_vertices, np.asarray(self.y_vertices))
        with self.assertRaises(ValueError):
            self.polygon.x_vertices[0] = 0.0
        with self.assertRaises(ValueError):
            self.polygon.y_vertices[0] = 0.0

    def test_boxes(self) -> None:
        """Test 'from_box', the `area` property, and the 'contains' method
        with polygon arguments.
        """
        small = Polygon.from_box(Box.factory[-3:3, 40:45])
        self.assertEqual(small.area, 30.0)
        self.assertTrue(self.polygon.contains(small))
        self.assertFalse(small.contains(self.polygon))
        big = Polygon.from_box(Box.factory[-10:10, 20:60])
        self.assertEqual(big.area, 800.0)
        self.assertFalse(self.polygon.contains(big))
        self.assertTrue(big.contains(self.polygon))
        medium = Polygon.from_box(Box.factory[-4:8, 31:52])
        self.assertEqual(medium.area, 252.0)
        self.assertFalse(self.polygon.contains(medium))
        self.assertFalse(medium.contains(self.polygon))
        self.assertTrue(self.polygon.contains(self.polygon))

    def test_contains_points(self) -> None:
        """Test the 'contains' method with points."""
        self.assertTrue(self.polygon.contains(x=40.0, y=0.0))
        self.assertFalse(self.polygon.contains(x=0.0, y=0.0))
        self.assertFalse(self.polygon.contains(x=40.0, y=10.0))
        np.testing.assert_array_equal(
            self.polygon.contains(x=np.array([40.0, 0.0, 40.0]), y=np.array([0.0, 0.0, 10.0])),
            np.array([True, False, False]),
        )

    @unittest.skipUnless(have_legacy, "lsst legacy packages could not be imported.")
    def test_legacy(self) -> None:
        """Test conversion to/from lsst.afw.geom.Polygon."""
        legacy_polygon = self.polygon.to_legacy()
        self.assertEqual(legacy_polygon.calculateArea(), self.polygon.area)
        self.assertEqual(Polygon.from_legacy(legacy_polygon), self.polygon)


if __name__ == "__main__":
    unittest.main()

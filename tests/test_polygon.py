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

from lsst.images import Box, NoOverlapError, Polygon, Region, RegionSerializationModel

try:
    import lsst.afw.geom  # noqa: F401

    have_legacy = True
except ImportError:
    have_legacy = False


class SimplePolygonTestCase(unittest.TestCase):
    """Tests for the Polygon class.

    This includes most of the test coverage for the Region base class.
    """

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
        self.assertEqual(small.bbox, Box.factory[-3:3, 40:45])
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

    def test_io(self) -> None:
        """Test serialization and stringification."""
        self.assertEqual(
            Polygon.deserialize(
                RegionSerializationModel.model_validate_json(self.polygon.serialize().model_dump_json())
            ),
            self.polygon,
        )
        self.assertEqual(Polygon.from_wkt(self.polygon.wkt), self.polygon)
        self.assertEqual(Polygon.from_wkt(str(self.polygon)), self.polygon)
        self.assertEqual(eval(repr(self.polygon), {"array": np.array, "Polygon": Polygon}), self.polygon)

    @unittest.skipUnless(have_legacy, "lsst legacy packages could not be imported.")
    def test_legacy(self) -> None:
        """Test conversion to/from lsst.afw.geom.Polygon."""
        legacy_polygon = self.polygon.to_legacy()
        self.assertEqual(legacy_polygon.calculateArea(), self.polygon.area)
        self.assertEqual(Polygon.from_legacy(legacy_polygon), self.polygon)


class RegionTestCase(unittest.TestCase):
    """Tests for `Region` objects that are not necessarily polygons, including
    point-set operations.

    Notes
    -----
    This test uses test geometries (all boxes) with the following rough layout
    (with y increasing upwards):

    .. _code-block::
            ┌─────┐
        ┌───┼─┐B┌─┼────┐
        │  A└─┼─┼─┘ ┌─┐│
        └─────┘ │ C │D││
                │   └─┘│
                └──────┘
    """

    def setUp(self) -> None:
        self.a = Polygon.from_box(Box.factory[3:6, 0:5])
        self.b = Polygon.from_box(Box.factory[4:7, 3:8])
        self.c = Polygon.from_box(Box.factory[0:6, 6:12])
        self.d = Polygon.from_box(Box.factory[1:4, 9:10])

    def test_intersection(self) -> None:
        """Test region intersection."""
        # Usual case:
        self.assertEqual(self.a.intersection(self.b), Polygon.from_box(Box.factory[4:6, 3:5]))
        # No-overlap case:
        with self.assertRaises(NoOverlapError):
            self.a.intersection(self.c)
        # LHS fully contains RHS:
        self.assertEqual(self.c.intersection(self.d), self.d)

    def test_union(self) -> None:
        """Test region union."""
        # Usual case:
        self.assertEqual(self.a.union(self.b).bbox, Box.factory[3:7, 0:8])
        self.assertEqual(self.a.union(self.b).area, 15 + 15 - 4)
        # Operands are disjoint, so union is not a single Polygon:
        self.assertNotIsInstance(self.a.union(self.c), Polygon)
        self.assertEqual(self.a.union(self.c).area, self.a.area + self.c.area)
        # LHS fully contains RHS:
        self.assertEqual(self.c.union(self.d), self.c)

    def test_difference(self) -> None:
        """Test region difference."""
        # Usual case:
        self.assertEqual(self.a.difference(self.b).bbox, self.a.bbox)
        self.assertEqual(self.a.difference(self.b).area, 15 - 4)
        # Operands are disjoint, so difference is just the LHS.
        self.assertEqual(self.a.difference(self.c), self.a)
        # LHS fully contains RHS -> polygon with hole is not a Polygon:
        self.assertNotIsInstance(self.c.difference(self.d), Polygon)
        self.assertEqual(self.c.difference(self.d).bbox, self.c.bbox)
        self.assertEqual(self.c.difference(self.d).area, self.c.area - self.d.area)
        # RHS fully contains LHS -> region is empty.
        self.assertEqual(self.d.difference(self.d).area, 0)

    def test_io(self) -> None:
        """Test serialization and stringification of non-polygon regions."""
        # A two-polygon region with a hole:
        region = self.a.union(self.c).difference(self.d)
        self.assertEqual(
            Region.deserialize(
                RegionSerializationModel.model_validate_json(region.serialize().model_dump_json())
            ),
            region,
        )
        self.assertEqual(Region.from_wkt(region.wkt), region)
        self.assertEqual(Region.from_wkt(str(region)), region)
        self.assertEqual(eval(repr(region), {"Region": Region}), region)


if __name__ == "__main__":
    unittest.main()

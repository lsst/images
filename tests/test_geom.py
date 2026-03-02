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

import pickle
import unittest

import numpy as np
import pydantic

from lsst.images import XY, YX, Box, Interval
from lsst.images.tests import assert_close


class IntervalModel(pydantic.BaseModel):
    """Test Pydantic model with an Interval."""

    interval1: Interval
    interval2: Interval


class BoxModel(pydantic.BaseModel):
    """Test Pydantic model with a box."""

    box: Box


class XYYXTestCase(unittest.TestCase):
    """Test the XY and YX classes."""

    def test_yx(self) -> None:
        """Test YX."""
        yx = YX(5, 7)
        self.assertEqual(yx, (5, 7))
        self.assertEqual(yx.y, 5)
        self.assertEqual(yx.x, 7)

        def _plus_one(v: int) -> int:
            return v + 1

        new = yx.map(_plus_one)
        self.assertEqual(new, (6, 8))

        xy = yx.xy
        self.assertEqual(xy, (7, 5))

    def test_xy(self) -> None:
        """Test XY."""
        xy = XY(5, 7)
        self.assertEqual(xy, (5, 7))
        self.assertEqual(xy.y, 7)
        self.assertEqual(xy.x, 5)

        def _plus_one(v: int) -> int:
            return v + 1

        new = xy.map(_plus_one)
        self.assertEqual(new, (6, 8))

        yx = xy.yx
        self.assertEqual(yx, (7, 5))


class IntervalTestCase(unittest.TestCase):
    """Test the Interval class."""

    def test_constructor(self) -> None:
        """Simple construction."""
        i = Interval(start=1, stop=10)
        self.assertEqual(i.start, 1)
        self.assertEqual(i.stop, 10)
        self.assertEqual(i.min, 1)
        self.assertEqual(i.max, 9)
        self.assertEqual(i.size, 9)
        self.assertEqual(i.center, 5.0)
        self.assertEqual(Interval(1, 10), i)

        self.assertEqual(str(i), "1:10")

        shifted = i + 10
        self.assertEqual(shifted.start, 11)
        self.assertEqual(shifted.stop, 20)

        shifted = i - 10
        self.assertEqual(shifted.start, -9)
        self.assertEqual(shifted.stop, 0)

        self.assertEqual(shifted, i - 10)
        self.assertNotEqual(shifted, i)

        sized = Interval.from_size(10)
        self.assertEqual(sized, Interval(0, 10))
        sized = Interval.from_size(10, 5)
        self.assertEqual(sized, Interval(5, 15))

        h = Interval.hull(2, -1, 3, 6)
        self.assertEqual(h, Interval(start=-1, stop=7))
        h2 = Interval.hull(h, 3, 40, Interval(start=-10, stop=2))
        self.assertEqual(h2, Interval(start=-10, stop=41))

    def test_contains(self) -> None:
        """Test containment."""
        i = Interval(start=1, stop=10)
        self.assertIn(5, i)
        self.assertNotIn(10, i)

        i2 = Interval(start=2, stop=4)
        self.assertTrue(i.contains(i2))

        i3 = Interval(start=-1, stop=5)
        self.assertFalse(i.contains(i3))

        self.assertTrue(i3.contains(2.5))

        containment = i3.contains(np.array([-2, -1, 0, 1, 2, 3, 4, 5, 6]))
        self.assertEqual(list(containment), [False, True, True, True, True, True, True, True, False])

        inter = i2.intersection(i)
        self.assertEqual(inter, Interval(start=2, stop=4), msg=f"Intersection of {i2} with {i}")
        self.assertIsNone(i.intersection(Interval.factory[20:30]))

        self.assertNotEqual(i, [])

    def test_slice(self) -> None:
        """Interval construction with slicing."""
        i = Interval.factory[3:20]
        self.assertEqual(i.start, 3)
        self.assertEqual(i.stop, 20)
        self.assertEqual(i.absolute[::], i)
        self.assertEqual(i.local[::], i)

        subset = i.absolute[5:10]
        self.assertEqual(subset.start, 5)
        self.assertEqual(subset.stop, 10)

        subset = i.local[5:10]
        self.assertEqual(subset.start, 8)
        self.assertEqual(subset.stop, 13)

        subset = i.absolute[:10]
        self.assertEqual(subset.start, 3)
        self.assertEqual(subset.stop, 10)

        subset = i.local[:10]
        self.assertEqual(subset.start, 3)
        self.assertEqual(subset.stop, 13)

        subset = i.absolute[10:]
        self.assertEqual(subset.start, 10)
        self.assertEqual(subset.stop, 20)

        subset = i.local[10:]
        self.assertEqual(subset.start, 13)
        self.assertEqual(subset.stop, 20)

        subset = i.local[3:-2]
        self.assertEqual(subset.start, 6)
        self.assertEqual(subset.stop, 18)

        subset = i.local[-5:-2]
        self.assertEqual(subset.start, 15)
        self.assertEqual(subset.stop, 18)

        with self.assertRaises(IndexError):
            i.absolute[:30]

        # It might seem surprising that this does not raise, but it's exactly
        # what what list(range(3, 20))[:30] does:
        subset = i.local[:30]
        self.assertEqual(subset.start, 3)
        self.assertEqual(subset.stop, 20)

        with self.assertRaises(IndexError):
            i.absolute[30:]

        with self.assertRaises(IndexError):
            i.local[30:]

        with self.assertRaises(IndexError):
            i.absolute[-1:10]

        with self.assertRaises(IndexError):
            i.local[-1:10]

        with self.assertRaises(ValueError):
            i.absolute[::2]

        with self.assertRaises(ValueError):
            i.local[::2]

        with self.assertRaises(ValueError):
            Interval.factory[1:2:2]

    def test_usage(self) -> None:
        """Test using intervals."""
        i = Interval(start=1, stop=10)
        d = i.dilated_by(5)
        self.assertEqual(d, Interval(start=-4, stop=15))

        s = i.slice_within(Interval(start=-1, stop=12))
        self.assertEqual(s.start, 2)
        self.assertEqual(s.stop, 11)

        with self.assertRaises(IndexError):
            i.slice_within(Interval(start=3, stop=5))

        val = i.linspace()
        assert_close(self, val, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]))
        val = i.linspace(step=2.0)
        assert_close(self, val, np.array([1.0, 3.0, 5.0, 7.0, 9.0]))
        val = i.linspace(n=3)
        assert_close(self, val, np.array([1.0, 5.0, 9.0]))
        with self.assertRaises(TypeError):
            i.linspace(n=2, step=3.0)

        self.assertEqual(list(i.range), [1, 2, 3, 4, 5, 6, 7, 8, 9])
        val = i.arange
        assert_close(self, val, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]))

    def test_pydantic(self) -> None:
        """Test roundtrip through pydantic serialization."""
        i1 = Interval(start=2, stop=5)
        i2 = Interval(start=-5, stop=10)
        model = IntervalModel(interval1=i1, interval2=i2)
        j_str = model.model_dump_json()
        jmodel = IntervalModel.model_validate_json(j_str)
        self.assertEqual(jmodel, model)
        self.assertEqual(jmodel.interval1, i1)

    def test_pickle(self) -> None:
        """Test pickle roundtrip."""
        i = Interval(start=5, stop=10)
        d = pickle.dumps(i)
        copy = pickle.loads(d)
        self.assertEqual(copy, i)


class BoxTestCase(unittest.TestCase):
    """Test the Box implementation."""

    def test_constructor(self) -> None:
        """Test Box construction."""
        y = Interval(-5, 5)
        x = Interval(32, 64)
        box = Box(y, x)

        self.assertEqual(box.start, YX(-5, 32))
        self.assertEqual(box.shape, YX(10, 32))
        self.assertEqual(box.x, x)
        self.assertEqual(box.y, y)
        self.assertNotEqual(box, [])

        box2 = Box.factory[-5:5, 32:64]
        self.assertEqual(box2, box)

        sbox = Box.from_shape((10, 5))
        self.assertEqual(sbox, Box.factory[0:10, 0:5])
        sbox = Box.from_shape((10, 5), start=(2, 3))
        self.assertEqual(sbox, Box.factory[2:12, 3:8])

        sbox = Box.from_shape(YX(10, 5))
        self.assertEqual(sbox, Box.factory[0:10, 0:5])
        sbox = Box.from_shape((10, 5), start=YX(2, 3))
        self.assertEqual(sbox, Box.factory[2:12, 3:8])

        sbox = Box.from_shape(XY(5, 10))
        self.assertEqual(sbox, Box.factory[0:10, 0:5])
        sbox = Box.from_shape((10, 5), start=XY(3, 2))
        self.assertEqual(sbox, Box.factory[2:12, 3:8])

        with self.assertRaises(TypeError):
            Box.from_shape(42)
        with self.assertRaises(ValueError):
            Box.from_shape([42])
        with self.assertRaises(ValueError):
            Box.from_shape([42, 33], start=[1, 2, 3])

        box = Box.factory[1:2, -1:1]
        grown = box.dilated_by(2)
        self.assertEqual(grown, Box.factory[-1:4, -3:3])

    def test_contains(self) -> None:
        """Does a box fit inside another or not."""
        box = Box.factory[0:20, 10:40]

        self.assertTrue(box.contains(Box.factory[4:10, 20:25]))
        self.assertFalse(box.contains(Box.factory[4:10, 35:45]))
        self.assertTrue(box.contains(y=4, x=15))
        self.assertFalse(box.contains(x=4, y=15))

        contains = box.contains(
            # Half pixel leeway.
            x=np.array([-1, 10, 20, 30, 40, 41]),
            y=np.array([-1, 5, 19, 20, 20, 20]),
        )
        self.assertEqual(list(contains), [False, True, True, True, True, False])

        with self.assertRaises(TypeError):
            box.contains(box, x=3, y=2)
        with self.assertRaises(TypeError):
            box.contains()

    def test_intersection(self) -> None:
        """Test box intersection."""
        box1 = Box.factory[0:20, 30:50]
        box2 = Box.factory[10:30, 40:42]
        self.assertEqual(box1.intersection(box2), Box.factory[10:20, 40:42])
        self.assertIsNone(box1.intersection(Box.factory[50:70, -10:-5]))

    def test_slicing(self) -> None:
        """Test slicing."""
        box = Box.factory[:10, :20]
        sbox = box.absolute[4:6, :3]
        self.assertEqual(sbox, Box.factory[4:6, 0:3])
        sbox = box.local[4:6, :3]
        self.assertEqual(sbox, Box.factory[4:6, 0:3])
        sbox = box.absolute[4:, 5:]
        self.assertEqual(sbox, Box.factory[4:10, 5:20])
        sbox = box.local[4:, 5:]
        self.assertEqual(sbox, Box.factory[4:10, 5:20])
        sbox = box.absolute[XY(slice(5, None), slice(4, None))]
        self.assertEqual(sbox, Box.factory[4:10, 5:20])
        sbox = box.local[XY(slice(5, None), slice(4, None))]
        self.assertEqual(sbox, Box.factory[4:10, 5:20])

        self.assertEqual(Box.factory[4:10, -1:2], Box.factory[XY(slice(-1, 2), slice(4, 10))])

        slices = sbox.slice_within(box)
        self.assertEqual(slices.x.start, 5)
        self.assertEqual(slices.y.start, 4)
        self.assertEqual(slices.x.stop, 20)
        self.assertEqual(slices.y.stop, 10)

        slices = Box.factory[:5, 110:119].slice_within(Box.factory[-15:12, 90:120])
        self.assertEqual(slices.x.start, 20)
        self.assertEqual(slices.y.start, 15)
        self.assertEqual(slices.x.stop, 29)
        self.assertEqual(slices.y.stop, 20)

        with self.assertRaises(IndexError):
            box.absolute[-1:5, 3:]
        with self.assertRaises(IndexError):
            box.local[-1:5, 3:]
        with self.assertRaises(TypeError):
            box.absolute[3:5, :5, 4:]
        with self.assertRaises(TypeError):
            box.local[3:5, :5, 4:]
        with self.assertRaises(TypeError):
            Box.factory[3:5, :6, 4:]

    def test_mesh(self) -> None:
        """Test grid creation."""
        box = Box.factory[0:2, 0:3]

        grid = box.meshgrid()
        assert_close(self, grid.x, np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]))
        assert_close(self, grid.y, np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))

        grid = box.meshgrid(2)
        assert_close(self, grid.x, np.array([[0.0, 2.0], [0.0, 2.0]]))
        assert_close(self, grid.y, np.array([[0.0, 0.0], [1.0, 1.0]]))

        grid = box.meshgrid([2, 1])
        assert_close(self, grid.x, np.array([[0.0, 2.0]]))
        assert_close(self, grid.y, np.array([[0.0, 0.0]]))

        grid = box.meshgrid(XY(2, 1))
        assert_close(self, grid.x, np.array([[0.0, 2.0]]))
        assert_close(self, grid.y, np.array([[0.0, 0.0]]))

        grid = box.meshgrid(YX(1, 2))
        assert_close(self, grid.x, np.array([[0.0, 2.0]]))
        assert_close(self, grid.y, np.array([[0.0, 0.0]]))

        grid = box.meshgrid(step=3)
        assert_close(self, grid.x, np.array([[0.0]]))
        assert_close(self, grid.y, np.array([[0.0]]))

        with self.assertRaises(TypeError):
            box.meshgrid(2, step=3)

        with self.assertRaises(ValueError):
            box.meshgrid("n")

    def test_boundary(self) -> None:
        """Test we can found the boundary."""
        box = Box.factory[-1:9, 7:15]
        corners = list(box.boundary())
        self.assertEqual(corners[0], (-1, 7))
        self.assertEqual(corners[1], (-1, 14))
        self.assertEqual(corners[2], (8, 14))
        self.assertEqual(corners[3], (8, 7))

    def test_pydantic(self) -> None:
        """Test roundtrip through pydantic serialization."""
        box = Box.factory[-1:1, 5:10]
        model = BoxModel(box=box)
        j_str = model.model_dump_json()
        jmodel = BoxModel.model_validate_json(j_str)
        self.assertEqual(jmodel, model)
        self.assertEqual(jmodel.box, box)

    def test_pickle(self) -> None:
        """Test pickle roundtrip."""
        box = Box.factory[-1:1, 5:10]
        d = pickle.dumps(box)
        copy = pickle.loads(d)
        self.assertEqual(copy, box)


if __name__ == "__main__":
    unittest.main()

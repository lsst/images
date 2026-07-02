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

import numpy as np
import pydantic
import pytest

from lsst.images import XY, YX, Box, Interval, NoOverlapError
from lsst.images.tests import assert_close


class IntervalModel(pydantic.BaseModel):
    """Test Pydantic model with an Interval."""

    interval1: Interval
    interval2: Interval


class BoxModel(pydantic.BaseModel):
    """Test Pydantic model with a box."""

    box: Box


def test_yx() -> None:
    """Test YX."""
    yx = YX(5, 7)
    assert yx == (5, 7)
    assert yx.y == 5
    assert yx.x == 7

    def _plus_one(v: int) -> int:
        return v + 1

    new = yx.map(_plus_one)
    assert new == (6, 8)

    xy = yx.xy
    assert xy == (7, 5)


def test_xy() -> None:
    """Test XY."""
    xy = XY(5, 7)
    assert xy == (5, 7)
    assert xy.y == 7
    assert xy.x == 5

    def _plus_one(v: int) -> int:
        return v + 1

    new = xy.map(_plus_one)
    assert new == (6, 8)

    yx = xy.yx
    assert yx == (7, 5)


def test_interval_constructor() -> None:
    """Test simple Interval construction and arithmetic."""
    i = Interval(start=1, stop=10)
    assert i.start == 1
    assert i.stop == 10
    assert i.min == 1
    assert i.max == 9
    assert i.size == 9
    assert i.center == 5.0
    assert Interval(1, 10) == i

    assert str(i) == "1:10"

    shifted = i + 10
    assert shifted.start == 11
    assert shifted.stop == 20

    shifted = i - 10
    assert shifted.start == -9
    assert shifted.stop == 0

    assert shifted == i - 10
    assert shifted != i

    sized = Interval.from_size(10)
    assert sized == Interval(0, 10)
    sized = Interval.from_size(10, 5)
    assert sized == Interval(5, 15)

    h = Interval.hull(2, -1, 3, 6)
    assert h == Interval(start=-1, stop=7)
    h2 = Interval.hull(h, 3, 40, Interval(start=-10, stop=2))
    assert h2 == Interval(start=-10, stop=41)


def test_interval_contains() -> None:
    """Test containment."""
    i = Interval(start=1, stop=10)
    assert 5 in i
    assert 10 not in i

    i2 = Interval(start=2, stop=4)
    assert i.contains(i2)

    i3 = Interval(start=-1, stop=5)
    assert not i.contains(i3)

    assert i3.contains(2.5)

    containment = i3.contains(np.array([-2, -1, 0, 1, 2, 3, 4, 5, 6]))
    assert list(containment) == [False, True, True, True, True, True, True, False, False]

    inter = i2.intersection(i)
    assert inter == Interval(start=2, stop=4), f"Intersection of {i2} with {i}"
    with pytest.raises(NoOverlapError):
        i.intersection(Interval.factory[20:30])
    assert i != []


def test_interval_slice() -> None:
    """Test Interval construction and subsetting with slice notation."""
    i = Interval.factory[3:20]
    assert i.start == 3
    assert i.stop == 20
    assert i.absolute[::] == i
    assert i.local[::] == i

    subset = i.absolute[5:10]
    assert subset.start == 5
    assert subset.stop == 10

    subset = i.local[5:10]
    assert subset.start == 8
    assert subset.stop == 13

    subset = i.absolute[:10]
    assert subset.start == 3
    assert subset.stop == 10

    subset = i.local[:10]
    assert subset.start == 3
    assert subset.stop == 13

    subset = i.absolute[10:]
    assert subset.start == 10
    assert subset.stop == 20

    subset = i.local[10:]
    assert subset.start == 13
    assert subset.stop == 20

    subset = i.local[3:-2]
    assert subset.start == 6
    assert subset.stop == 18

    subset = i.local[-5:-2]
    assert subset.start == 15
    assert subset.stop == 18

    with pytest.raises(IndexError):
        i.absolute[:30]

    # It might seem surprising that this does not raise, but it's exactly
    # what what list(range(3, 20))[:30] does:
    subset = i.local[:30]
    assert subset.start == 3
    assert subset.stop == 20

    with pytest.raises(IndexError):
        i.absolute[30:]

    with pytest.raises(IndexError):
        i.local[30:]

    with pytest.raises(IndexError):
        i.absolute[-1:10]

    with pytest.raises(IndexError):
        i.local[-1:10]

    with pytest.raises(ValueError):
        i.absolute[::2]

    with pytest.raises(ValueError):
        i.local[::2]

    with pytest.raises(ValueError):
        Interval.factory[1:2:2]


def test_interval_usage() -> None:
    """Test using intervals."""
    i = Interval(start=1, stop=10)
    d = i.dilated_by(5)
    assert d == Interval(start=-4, stop=15)

    s = i.slice_within(Interval(start=-1, stop=12))
    assert s.start == 2
    assert s.stop == 11

    with pytest.raises(IndexError):
        i.slice_within(Interval(start=3, stop=5))

    val = i.linspace()
    assert_close(val, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]))
    val = i.linspace(step=2.0)
    assert_close(val, np.array([1.0, 3.0, 5.0, 7.0, 9.0]))
    val = i.linspace(n=3)
    assert_close(val, np.array([1.0, 5.0, 9.0]))
    with pytest.raises(TypeError):
        i.linspace(n=2, step=3.0)

    assert list(i.range) == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    val = i.arange
    assert_close(val, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]))


def test_interval_pydantic() -> None:
    """Test roundtrip through pydantic serialization."""
    i1 = Interval(start=2, stop=5)
    i2 = Interval(start=-5, stop=10)
    model = IntervalModel(interval1=i1, interval2=i2)
    j_str = model.model_dump_json()
    jmodel = IntervalModel.model_validate_json(j_str)
    assert jmodel == model
    assert jmodel.interval1 == i1


def test_interval_pickle() -> None:
    """Test pickle roundtrip."""
    i = Interval(start=5, stop=10)
    d = pickle.dumps(i)
    copy = pickle.loads(d)
    assert copy == i


def test_box_constructor() -> None:
    """Test Box construction and factory methods."""
    y = Interval(-5, 5)
    x = Interval(32, 64)
    box = Box(y, x)

    assert box.start == YX(-5, 32)
    assert box.shape == YX(10, 32)
    assert box.area == 320
    assert box.x == x
    assert box.y == y
    assert box != []

    box2 = Box.factory[-5:5, 32:64]
    assert box2 == box

    sbox = Box.from_shape((10, 5))
    assert sbox == Box.factory[0:10, 0:5]
    sbox = Box.from_shape((10, 5), start=(2, 3))
    assert sbox == Box.factory[2:12, 3:8]

    sbox = Box.from_shape(YX(10, 5))
    assert sbox == Box.factory[0:10, 0:5]
    sbox = Box.from_shape((10, 5), start=YX(2, 3))
    assert sbox == Box.factory[2:12, 3:8]

    sbox = Box.from_shape(XY(5, 10))
    assert sbox == Box.factory[0:10, 0:5]
    sbox = Box.from_shape((10, 5), start=XY(3, 2))
    assert sbox == Box.factory[2:12, 3:8]

    with pytest.raises(TypeError):
        Box.from_shape(42)
    with pytest.raises(ValueError):
        Box.from_shape([42])
    with pytest.raises(ValueError):
        Box.from_shape([42, 33], start=[1, 2, 3])

    box = Box.factory[1:2, -1:1]
    grown = box.dilated_by(2)
    assert grown == Box.factory[-1:4, -3:3]


def test_box_contains() -> None:
    """Test box containment against other boxes and points."""
    box = Box.factory[0:20, 10:40]

    assert box.contains(Box.factory[4:10, 20:25])
    assert not box.contains(Box.factory[4:10, 35:45])
    assert box.contains(y=4, x=15)
    assert not box.contains(x=4, y=15)

    contains = box.contains(
        # Half pixel leeway.
        x=np.array([-1, 10, 20, 30, 40, 41]),
        y=np.array([-1, 5, 19, 20, 20, 20]),
    )
    assert list(contains) == [False, True, True, False, False, False]

    with pytest.raises(TypeError):
        box.contains(box, x=3, y=2)
    with pytest.raises(TypeError):
        box.contains()


def test_box_intersection() -> None:
    """Test box intersection."""
    box1 = Box.factory[0:20, 30:50]
    box2 = Box.factory[10:30, 40:42]
    assert box1.intersection(box2) == Box.factory[10:20, 40:42]
    with pytest.raises(NoOverlapError):
        box1.intersection(Box.factory[50:70, -10:-5])


def test_box_slicing() -> None:
    """Test slicing."""
    box = Box.factory[:10, :20]
    sbox = box.absolute[4:6, :3]
    assert sbox == Box.factory[4:6, 0:3]
    sbox = box.local[4:6, :3]
    assert sbox == Box.factory[4:6, 0:3]
    sbox = box.absolute[4:, 5:]
    assert sbox == Box.factory[4:10, 5:20]
    sbox = box.local[4:, 5:]
    assert sbox == Box.factory[4:10, 5:20]
    sbox = box.absolute[XY(slice(5, None), slice(4, None))]
    assert sbox == Box.factory[4:10, 5:20]
    sbox = box.local[XY(slice(5, None), slice(4, None))]
    assert sbox == Box.factory[4:10, 5:20]

    assert Box.factory[4:10, -1:2] == Box.factory[XY(slice(-1, 2), slice(4, 10))]

    slices = sbox.slice_within(box)
    assert slices.x.start == 5
    assert slices.y.start == 4
    assert slices.x.stop == 20
    assert slices.y.stop == 10

    slices = Box.factory[:5, 110:119].slice_within(Box.factory[-15:12, 90:120])
    assert slices.x.start == 20
    assert slices.y.start == 15
    assert slices.x.stop == 29
    assert slices.y.stop == 20

    with pytest.raises(IndexError):
        box.absolute[-1:5, 3:]
    with pytest.raises(IndexError):
        box.local[-1:5, 3:]
    with pytest.raises(TypeError):
        box.absolute[3:5, :5, 4:]
    with pytest.raises(TypeError):
        box.local[3:5, :5, 4:]
    with pytest.raises(TypeError):
        Box.factory[3:5, :6, 4:]


def test_box_mesh() -> None:
    """Test grid creation."""
    box = Box.factory[0:2, 0:3]

    grid = box.meshgrid()
    assert_close(grid.x, np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]]))
    assert_close(grid.y, np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))

    grid = box.meshgrid(2)
    assert_close(grid.x, np.array([[0.0, 2.0], [0.0, 2.0]]))
    assert_close(grid.y, np.array([[0.0, 0.0], [1.0, 1.0]]))

    grid = box.meshgrid([2, 1])
    assert_close(grid.x, np.array([[0.0, 2.0]]))
    assert_close(grid.y, np.array([[0.0, 0.0]]))

    grid = box.meshgrid(XY(2, 1))
    assert_close(grid.x, np.array([[0.0, 2.0]]))
    assert_close(grid.y, np.array([[0.0, 0.0]]))

    grid = box.meshgrid(YX(1, 2))
    assert_close(grid.x, np.array([[0.0, 2.0]]))
    assert_close(grid.y, np.array([[0.0, 0.0]]))

    grid = box.meshgrid(step=3)
    assert_close(grid.x, np.array([[0.0]]))
    assert_close(grid.y, np.array([[0.0]]))

    with pytest.raises(TypeError):
        box.meshgrid(2, step=3)

    with pytest.raises(ValueError):
        box.meshgrid("n")


def test_box_boundary() -> None:
    """Test we can find the boundary."""
    box = Box.factory[-1:9, 7:15]
    corners = list(box.boundary())
    assert corners[0] == (-1, 7)
    assert corners[1] == (-1, 14)
    assert corners[2] == (8, 14)
    assert corners[3] == (8, 7)


def test_box_pydantic() -> None:
    """Test roundtrip through pydantic serialization."""
    box = Box.factory[-1:1, 5:10]
    model = BoxModel(box=box)
    j_str = model.model_dump_json()
    jmodel = BoxModel.model_validate_json(j_str)
    assert jmodel == model
    assert jmodel.box == box


def test_box_pickle() -> None:
    """Test pickle roundtrip."""
    box = Box.factory[-1:1, 5:10]
    d = pickle.dumps(box)
    copy = pickle.loads(d)
    assert copy == box


def test_box_from_float_bounds() -> None:
    """Test Box.from_float_bounds rounds outward to integer pixel bounds."""
    # x in [4.6, 9.4], y in [2.6, 5.4] -> box [3:6, 5:10] in [y, x].
    box = Box.from_float_bounds(x_min=4.6, x_max=9.4, y_min=2.6, y_max=5.4)
    assert box == Box.factory[3:6, 5:10]

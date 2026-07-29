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

import astropy.units as u
import numpy as np
import pydantic
import pytest
import shapely

from lsst.images import (
    XY,
    YX,
    Box,
    GeneralFrame,
    NoOverlapError,
    Polygon,
    Region,
    RegionSerializationModel,
    Transform,
)
from lsst.images.tests import assert_close, check_bounds_contains_broadcasting

try:
    import lsst.afw.geom  # noqa: F401

    have_legacy = True
except ImportError:
    have_legacy = False

skip_no_legacy = pytest.mark.skipif(not have_legacy, reason="lsst legacy packages could not be imported.")


class _PolygonHolder(pydantic.BaseModel):
    polygon: Polygon


class _RegionHolder(pydantic.BaseModel):
    region: Region


def _make_polygon() -> Polygon:
    """Return a near-box quadrilateral that is easy to reason about."""
    x_vertices = [32.0, 31.0, 50.0, 53.0]
    y_vertices = [-5.0, 7.0, 7.2, -4.8]
    return Polygon(x_vertices=x_vertices, y_vertices=y_vertices)


@dataclasses.dataclass
class _TestRegions:
    """Four overlapping box-polygons for region operation tests.

    Rough layout (y increasing upwards)::

            ┌─────┐
        ┌───┼─┐B┌─┼────┐
        │  A└─┼─┼─┘ ┌─┐│
        └─────┘ │ C │D││
                │   └─┘│
                └──────┘
    """

    a: Polygon
    b: Polygon
    c: Polygon
    d: Polygon

    def __init__(self) -> None:
        self.a = Polygon.from_box(Box.factory[3:6, 0:5])
        self.b = Polygon.from_box(Box.factory[4:7, 3:8])
        self.c = Polygon.from_box(Box.factory[0:6, 6:12])
        self.d = Polygon.from_box(Box.factory[1:4, 9:10])


def test_vertices() -> None:
    """Test the vertices accessors."""
    polygon = _make_polygon()
    x_vertices = [32.0, 31.0, 50.0, 53.0]
    y_vertices = [-5.0, 7.0, 7.2, -4.8]
    assert polygon.n_vertices == 4
    np.testing.assert_array_equal(polygon.x_vertices, np.asarray(x_vertices))
    np.testing.assert_array_equal(polygon.y_vertices, np.asarray(y_vertices))
    with pytest.raises(ValueError):
        polygon.x_vertices[0] = 0.0
    with pytest.raises(ValueError):
        polygon.y_vertices[0] = 0.0


def test_boxes() -> None:
    """Test 'from_box', the ``area`` property, and the 'contains' method
    with polygon arguments.
    """
    polygon = _make_polygon()
    small = Polygon.from_box(Box.factory[-3:3, 40:45])
    assert small.area == 30.0
    assert small.bbox == Box.factory[-3:3, 40:45]
    assert polygon.contains(small)
    assert not small.contains(polygon)
    assert small.centroid == XY(x=42.0, y=-0.5)
    big = Polygon.from_box(Box.factory[-10:10, 20:60])
    assert big.area == 800.0
    assert not polygon.contains(big)
    assert big.contains(polygon)
    medium = Polygon.from_box(Box.factory[-4:8, 31:52])
    assert medium.area == 252.0
    assert not polygon.contains(medium)
    assert not medium.contains(polygon)
    assert polygon.contains(polygon)


def test_transform() -> None:
    """Test applying a coordinate transform to a polygon."""
    polygon = _make_polygon()
    matrix = np.array([[0.4, 0.25], [-0.20, 0.6]])
    t = Transform.affine(GeneralFrame(unit=u.pix), GeneralFrame(unit=u.pix), matrix)
    tp = polygon.transform(t)
    assert isinstance(tp, Polygon)
    assert_close(tp.area, polygon.area * np.linalg.det(matrix))
    xyt = t.apply_forward(x=polygon.x_vertices, y=polygon.y_vertices)
    # Slicing below is because shapely sometimes adds a duplicate closing
    # vertex.
    assert_close(tp.x_vertices[: len(xyt.x)], xyt.x)
    assert_close(tp.y_vertices[: len(xyt.y)], xyt.y)


def test_contains_points() -> None:
    """Test the 'contains' method with points."""
    polygon = _make_polygon()
    assert polygon.contains(x=40.0, y=0.0)
    assert not polygon.contains(x=0.0, y=0.0)
    assert not polygon.contains(x=40.0, y=10.0)
    np.testing.assert_array_equal(
        polygon.contains(x=np.array([40.0, 0.0, 40.0]), y=np.array([0.0, 0.0, 10.0])),
        np.array([True, False, False]),
    )


def test_contains_points_xy_yx() -> None:
    """Verify that Region.contains accepts XY and YX positional arguments."""
    polygon = _make_polygon()
    # Scalar XY and YX — results must match the keyword form.
    assert polygon.contains(XY(x=40.0, y=0.0)) == polygon.contains(x=40.0, y=0.0)
    assert polygon.contains(YX(y=0.0, x=40.0)) == polygon.contains(x=40.0, y=0.0)
    assert not polygon.contains(XY(x=0.0, y=0.0))
    assert not polygon.contains(YX(y=0.0, x=0.0))
    # Array XY and YX.
    xv = np.array([40.0, 0.0, 40.0])
    yv = np.array([0.0, 0.0, 10.0])
    np.testing.assert_array_equal(polygon.contains(XY(xv, yv)), polygon.contains(x=xv, y=yv))
    np.testing.assert_array_equal(polygon.contains(YX(yv, xv)), polygon.contains(x=xv, y=yv))
    # Mixing positional point with keyword x= or y= must raise TypeError.
    with pytest.raises(TypeError):
        polygon.contains(XY(x=40.0, y=0.0), x=40.0)
    with pytest.raises(TypeError):
        polygon.contains(YX(y=0.0, x=40.0), y=0.0)


def test_region_contains_broadcasting() -> None:
    """Test that Region.contains broadcasts like a numpy ufunc."""
    check_bounds_contains_broadcasting(_make_polygon())


def test_polygon_io() -> None:
    """Test serialization and stringification."""
    polygon = _make_polygon()
    assert (
        RegionSerializationModel.model_validate_json(polygon.serialize().model_dump_json()).deserialize()
        == polygon
    )
    assert Polygon.from_wkt(polygon.wkt) == polygon
    assert Polygon.from_wkt(str(polygon)) == polygon
    assert eval(repr(polygon), {"array": np.array, "Polygon": Polygon}) == polygon


def test_polygon_model_field() -> None:
    """Test that we can use a Polygon directly as a Pydantic model field."""
    polygon = _make_polygon()
    holder = _PolygonHolder(polygon=polygon)
    assert polygon == holder.model_validate_json(holder.model_dump_json()).polygon
    assert (
        _PolygonHolder.model_json_schema()["properties"]["polygon"]
        == RegionSerializationModel.model_json_schema()
    )


@skip_no_legacy
def test_polygon_legacy() -> None:
    """Test conversion to/from lsst.afw.geom.Polygon."""
    polygon = _make_polygon()
    legacy_polygon = polygon.to_legacy()
    assert legacy_polygon.calculateArea() == polygon.area
    assert Polygon.from_legacy(legacy_polygon) == polygon


def test_intersection() -> None:
    """Test region intersection."""
    regions = _TestRegions()
    # Usual case:
    assert regions.a.intersection(regions.b) == Polygon.from_box(Box.factory[4:6, 3:5])
    # No-overlap case:
    with pytest.raises(NoOverlapError):
        regions.a.intersection(regions.c)
    # LHS fully contains RHS:
    assert regions.c.intersection(regions.d) == regions.d
    # Intersections with the boxes themselves should return boxes:
    assert regions.a.intersection(regions.b.bbox) == Box.factory[4:6, 3:5]
    assert regions.a.bbox.intersection(regions.b) == Box.factory[4:6, 3:5]
    assert (
        # A Box is not possible when the result is not simple.
        regions.a.union(regions.c).intersection(regions.b.bbox)
        == regions.a.union(regions.c).intersection(regions.b)
    )


def test_union() -> None:
    """Test region union."""
    regions = _TestRegions()
    # Usual case:
    assert regions.a.union(regions.b).bbox == Box.factory[3:7, 0:8]
    assert regions.a.union(regions.b).area == 15 + 15 - 4
    # Operands are disjoint, so union is not a single Polygon:
    assert not isinstance(regions.a.union(regions.c), Polygon)
    assert regions.a.union(regions.c).area == regions.a.area + regions.c.area
    # LHS fully contains RHS:
    assert regions.c.union(regions.d) == regions.c


def test_difference() -> None:
    """Test region difference."""
    regions = _TestRegions()
    # Usual case:
    assert regions.a.difference(regions.b).bbox == regions.a.bbox
    assert regions.a.difference(regions.b).area == 15 - 4
    # Operands are disjoint, so difference is just the LHS.
    assert regions.a.difference(regions.c) == regions.a
    # LHS fully contains RHS -> polygon with hole is not a Polygon:
    assert not isinstance(regions.c.difference(regions.d), Polygon)
    assert regions.c.difference(regions.d).bbox == regions.c.bbox
    assert regions.c.difference(regions.d).area == regions.c.area - regions.d.area
    # RHS fully contains LHS -> region is empty.
    assert regions.d.difference(regions.d).area == 0


def test_region_io() -> None:
    """Test serialization and stringification of non-polygon regions."""
    regions = _TestRegions()
    # A two-polygon region with a hole:
    region = regions.a.union(regions.c).difference(regions.d)
    assert (
        RegionSerializationModel.model_validate_json(region.serialize().model_dump_json()).deserialize()
        == region
    )
    assert Region.from_wkt(region.wkt) == region
    assert Region.from_wkt(str(region)) == region
    assert eval(repr(region), {"Region": Region}) == region


def test_region_model_field() -> None:
    """Test that we can use a Region directly as a Pydantic model field."""
    regions = _TestRegions()
    region = regions.a.union(regions.c).difference(regions.d)
    holder = _RegionHolder(region=region)
    assert region == holder.model_validate_json(holder.model_dump_json()).region
    assert (
        _RegionHolder.model_json_schema()["properties"]["region"]
        == RegionSerializationModel.model_json_schema()
    )


def test_region_polygon_repr_str_pinned() -> None:
    """Region and Polygon str/repr match their documented forms."""
    poly = Polygon(x_vertices=[0.0, 4.0, 4.0, 0.0], y_vertices=[0.0, 0.0, 5.0, 5.0])
    assert repr(poly) == f"Polygon(x_vertices={poly.x_vertices!r}, y_vertices={poly.y_vertices!r})"
    # str is inherited from Region and uses WKT.
    assert str(poly) == poly.wkt

    # Region.from_wkt on a simple polygon returns a Polygon via try_to_polygon.
    # To test Region.__str__/__repr__ directly, construct a true Region
    # (multi-polygon), which is not coerced to Polygon.
    multi = shapely.MultiPolygon(
        [
            shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            shapely.Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
        ]
    )
    region = Region(multi)
    assert str(region) == region.wkt
    assert repr(region) == f"Region.from_wkt({region.wkt!r})"

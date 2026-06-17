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

__all__ = ("Polygon", "Region", "RegionSerializationModel")

from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
import numpy.typing as npt
import pydantic
import pydantic_core.core_schema as pcs
import shapely
from pydantic.json_schema import JsonSchemaValue

from ._geom import XY, Bounds, Box
from .utils import round_half_down, round_half_up

if TYPE_CHECKING:
    from ._transforms import Transform

    try:
        from lsst.afw.geom import Polygon as LegacyPolygon
    except ImportError:
        type LegacyPolygon = Any  # type: ignore[no-redef]


class Region:
    """A 2-d Euclidean region represented as one or more polygons with
    optional holes.

    Parameters
    ----------
    geometry
        A polygon or multi-polygon from the Shapely library.
    """

    def __init__(self, geometry: shapely.Polygon | shapely.MultiPolygon) -> None:
        self._impl = geometry

    @property
    def area(self) -> float:
        """The area of the region (`float`)."""
        return self._impl.area

    @property
    def bbox(self) -> Box:
        """The integer-coordinate bounding box of the region (`Box`).

        Because a `Box` logically contains the entirety of the pixels on its
        boundary, but the centers of those pixels are the numerical values of
        its bounds, the region may contain vertices that are up to 0.5 beyond
        the integer box coordinates in either dimension.
        """
        x_min, y_min, x_max, y_max = self._impl.bounds
        return Box.factory[
            round_half_up(y_min) : round_half_down(y_max) + 1,
            round_half_up(x_min) : round_half_down(x_max) + 1,
        ]

    @property
    def wkt(self) -> str:
        """The 'Well-Known Text' representation of this region (`str`)."""
        return self._impl.wkt

    @staticmethod
    def from_wkt(wkt: str) -> Region:
        """Construct from a 'Well-Known Text' string.

        Parameters
        ----------
        wkt
            Well-Known Text representation of the region.
        """
        impl = shapely.from_wkt(wkt)
        if not isinstance(impl, shapely.Polygon | shapely.MultiPolygon):
            raise ValueError("Only Polygon and MultiPolygon geometries can be converted to Regions.")
        return Region(impl).try_to_polygon()

    def __str__(self) -> str:
        return self._impl.wkt

    def __repr__(self) -> str:
        return f"Region.from_wkt({self._impl.wkt!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Region):
            return bool(shapely.equals(self._impl, other._impl))
        return NotImplemented

    @overload
    def contains(self, other: Polygon) -> bool: ...

    @overload
    def contains(self, *, x: int, y: int) -> bool: ...

    @overload
    def contains(self, *, x: float, y: float) -> bool: ...

    @overload
    def contains(self, *, x: np.ndarray, y: np.ndarray) -> np.ndarray: ...

    def contains(
        self,
        other: Region | None = None,
        *,
        x: float | int | np.ndarray | None = None,
        y: float | int | np.ndarray | None = None,
    ) -> bool | np.ndarray:
        """Test whether the geometry contains the given points or another
        geometry.

        Parameters
        ----------
        other
            Another geometry to compare to.  Not compatible with the ``y`` and
            ``x`` arguments.
        x
            One or more floating-point or integer X coordinates to test for
            containment.  If an array, an array of results will be returned.
        y
            One or more floating-point or integer Y coordinates to test for
            containment.  If an array, an array of results will be returned.
        """
        if other is not None:
            if x is not None or y is not None:
                raise TypeError("Too many arguments to 'Region.contains'.")
            return self._impl.contains(other._impl)
        elif x is None or y is None:
            raise TypeError("Not enough arguments to 'Region.contains'.")
        else:
            # Quibbles about bool vs numpy.bool_ as the return type.
            return shapely.contains_xy(self._impl, x=x, y=y)  # type: ignore[return-value]

    def intersection(self, other: Bounds) -> Bounds:
        """Compute the intersection of this region with a `Bounds` object.

        Notes
        -----
        Because `Region` implements the `Bounds` interface, its intersections
        need to support all other `Bounds` objects.  This is not true of other
        `Region` point-set operations like `union` and `difference`.

        Parameters
        ----------
        other
            Bounds to intersect with this region.
        """
        from ._concrete_bounds import _intersect_region

        return _intersect_region(self, other)

    def union(self, other: Region) -> Region:
        """Compute the point-set union of this region with another.

        Parameters
        ----------
        other
            Region to union with this one.
        """
        impl = shapely.union(self._impl, other._impl)
        assert isinstance(impl, shapely.Polygon | shapely.MultiPolygon), (
            "A union of Polygons and MultiPolygons should be one of those."
        )
        return Region(impl).try_to_polygon()

    def difference(self, other: Region) -> Region:
        """Compute the point-set difference of this region with another.

        Parameters
        ----------
        other
            Region to subtract from this one.
        """
        impl = shapely.difference(self._impl, other._impl)
        assert isinstance(impl, shapely.Polygon | shapely.MultiPolygon), (
            "A difference of Polygons and MultiPolygons should be one of those."
        )
        return Region(impl).try_to_polygon()

    def try_to_polygon(self) -> Region:
        """If the underlying geometry is a single polygon with no holes,
        return a `Polygon` instance holding it.

        In all other cases ``self`` is returned.
        """
        impl = self._impl
        if isinstance(impl, shapely.MultiPolygon) and len(impl.geoms) == 1:
            impl = impl.geoms[0]
        if isinstance(impl, shapely.Polygon) and not impl.interiors:
            vertices = np.array(impl.exterior.coords)
            return Polygon(x_vertices=vertices[:, 0], y_vertices=vertices[:, 1])
        return self

    def try_to_box(self) -> Region | Box:
        """If the underlying geometry is a rectangle that fully covers integer
        pixels (i.e. has all vertices at half-integer positions), return the
        equivalent `Box`.

        In all other cases ``self`` is returned.
        """
        if Polygon.from_box(self.bbox) == self:
            return self.bbox
        return self

    def to_shapely(self) -> shapely.Polygon | shapely.MultiPolygon:
        """Convert to a `shapely.Polygon` or `shapely.MultiPolygon` object."""
        return self._impl

    def transform(self, transform: Transform[Any, Any]) -> Region:
        """Transform the coordinates of the region, returning a new one.

        Parameters
        ----------
        transform
            Coordinate transform to apply (in the forward direction).

        Notes
        -----
        This applies the transform to all vertices, assuming that the
        transform is close enough to affine that the topology of the geometry
        does not change and straight-line edges do not need to be subsampled.
        """

        def wrapper(x: np.ndarray, y: np.ndarray) -> XY[np.ndarray]:
            return transform.apply_forward(x=x, y=y)

        return Region(
            # Shapely overloads don't seem to have been annotated rigorously
            shapely.transform(self._impl, wrapper, interleaved=False)  # type: ignore[arg-type]
        ).try_to_polygon()

    def serialize(self) -> RegionSerializationModel:
        """Serialize the region to a Pydantic model.

        Region serialization uses a subset of the GeoJSON specification (IETF
        RFC 7946).
        """
        return RegionSerializationModel.model_validate_json(shapely.to_geojson(self._impl))

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: pydantic.GetCoreSchemaHandler
    ) -> pcs.CoreSchema:
        from_model_schema = pcs.chain_schema(
            [
                handler(RegionSerializationModel),
                pcs.no_info_plain_validator_function(RegionSerializationModel.deserialize),
            ]
        )
        return pcs.json_or_python_schema(
            json_schema=from_model_schema,
            python_schema=pcs.union_schema([pcs.is_instance_schema(cls), from_model_schema]),
            serialization=pcs.plain_serializer_function_ser_schema(cls.serialize),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, schema: pcs.CoreSchema, handler: pydantic.GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        return handler(RegionSerializationModel.__pydantic_core_schema__)


class Polygon(Region):
    """A simple 2-d polygon in Euclidean coordinates, with no holes.

    Parameters
    ----------
    x_vertices
        The x coordinates of the vertices of the polygon.
    y_vertices
        The y coordinate of the vertices of the polygon.
    """

    def __init__(self, *, x_vertices: npt.ArrayLike, y_vertices: npt.ArrayLike) -> None:
        self._vertices = np.stack(
            [np.asarray(x_vertices).flat, np.asarray(y_vertices).flat], dtype=np.float64
        ).transpose()
        self._vertices.flags.writeable = False
        super().__init__(shapely.Polygon(self._vertices))

    @staticmethod
    def from_box(box: Box) -> Polygon:
        """Construct from an integer-coordinate box.

        Parameters
        ----------
        box
            Integer-coordinate box to convert to a polygon.

        Notes
        -----
        Because the integer min and max coordinates of the box are
        interpreted as pixel centers, these are expanded by 0.5 on all sides
        before using them to form the polygon vertices.
        """
        return Polygon(
            x_vertices=[box.x.min - 0.5, box.x.min - 0.5, box.x.max + 0.5, box.x.max + 0.5],
            y_vertices=[box.y.min - 0.5, box.y.max + 0.5, box.y.max + 0.5, box.y.min - 0.5],
        )

    @property
    def n_vertices(self) -> int:
        """The number of vertices in the polygon."""
        return self._vertices.shape[0]

    @property
    def x_vertices(self) -> np.ndarray:
        """The x coordinates of the vertices of the polygon.

        This is a read-only array; polygons are immutable.
        """
        return self._vertices[:, 0]

    @property
    def y_vertices(self) -> np.ndarray:
        """The y coordinates of the vertices of the polygon.

        This is a read-only array; polygons are immutable.
        """
        return self._vertices[:, 1]

    @property
    def centroid(self) -> XY[float]:
        """The centroid of the polygon (`XY` [`float`])."""
        c = self._impl.centroid
        return XY(x=c.x, y=c.y)

    def __repr__(self) -> str:
        return f"Polygon(x_vertices={self.x_vertices!r}, y_vertices={self.y_vertices!r})"

    @staticmethod
    def from_legacy(legacy: LegacyPolygon) -> Polygon:
        """Convert from a legacy `lsst.afw.geom.Polygon` instance.

        Parameters
        ----------
        legacy
            Legacy `lsst.afw.geom.Polygon` to convert.
        """
        vertices = legacy.getVertices()
        x_vertices = np.zeros(len(vertices), dtype=np.float64)
        y_vertices = np.zeros(len(vertices), dtype=np.float64)
        for n, point in enumerate(vertices):
            x_vertices[n] = point.x
            y_vertices[n] = point.y
        return Polygon(x_vertices=x_vertices, y_vertices=y_vertices)

    def to_legacy(self) -> LegacyPolygon:
        """Convert to a legacy `lsst.afw.geom.Polygon` instance."""
        from lsst.afw.geom import Polygon as LegacyPolygon
        from lsst.geom import Point2D

        return LegacyPolygon([Point2D(x, y) for x, y in zip(self.x_vertices, self.y_vertices)])


class RegionSerializationModel(pydantic.BaseModel):
    """Serialization model for `Region` and `Polygon`.

    This model is a subset of the GeoJSON specification (IETF RFC 7946).
    """

    type: Literal["Polygon", "MultiPolygon"] = pydantic.Field(description="Geometry type.")

    coordinates: list[list[tuple[float, float] | list[tuple[float, float]]]] = pydantic.Field(
        description="Vertices of the polygon or polygons."
    )

    def deserialize(self) -> Region:
        """Deserialize into a `Region` (a `Polygon`, if possible)."""
        region_impl = shapely.from_geojson(self.model_dump_json())
        assert isinstance(region_impl, shapely.Polygon | shapely.MultiPolygon), (
            "Other geometry types are not used."
        )
        return Region(region_impl).try_to_polygon()

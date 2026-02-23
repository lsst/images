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

__all__ = ("Polygon",)

from typing import TYPE_CHECKING, Any, overload

import numpy as np
import numpy.typing as npt

from ._geom import Box

if TYPE_CHECKING:
    try:
        from shapely import Polygon as _ImplPolygon
    except ImportError:
        type _ImplPolygon = Any  # type: ignore[no-redef]
    try:
        from lsst.afw.geom import Polygon as LegacyPolygon
    except ImportError:
        type LegacyPolygon = Any  # type: ignore[no-redef]


class Polygon:
    """A simple 2-d polygon in Euclidean coordinates, with no holes.

    Parameters
    ----------
    x_vertices
        The x coordinates of the vertices of the polygon.
    y_vertices
        The y coordinate of the vertices of the polygon.
    """

    def __init__(self, *, x_vertices: npt.ArrayLike, y_vertices: npt.ArrayLike):
        self._vertices = np.stack(
            [np.asarray(x_vertices).flat, np.asarray(y_vertices).flat], dtype=np.float64
        ).transpose()
        self._vertices.flags.writeable = False
        self._impl: _ImplPolygon | None = None

    @staticmethod
    def from_box(box: Box) -> Polygon:
        """Construct from an integer-coordinate box.

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
    def area(self) -> float:
        """The area of the polygon (`float`)."""
        return self._get_impl().area

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Polygon):
            import shapely

            return bool(shapely.equals(self._get_impl(), other._get_impl()))
        return NotImplemented

    @overload
    def contains(self, other: Polygon) -> bool: ...

    @overload
    def contains(self, *, x: float, y: float) -> bool: ...

    @overload
    def contains(self, *, x: np.ndarray, y: np.ndarray) -> np.ndarray: ...

    def contains(
        self,
        other: Polygon | None = None,
        *,
        x: float | np.ndarray | None = None,
        y: float | np.ndarray | None = None,
    ) -> bool | np.ndarray:
        """Test whether the polygon contains the given points or polygon.

        Parameters
        ----------
        other
            Another polygon to compare to.  Not compatible with the ``y`` and
            ``x`` arguments.
        x
            One or more floating-point X coordinates to test for containment.
            If an array, an array of results will be returned.
        y
            One or more floating-point Y coordinates to test for containment.
            If an array, an array of results will be returned.
        """
        import shapely

        impl = self._get_impl()
        if other is not None:
            if x is not None or y is not None:
                raise TypeError("Too many arguments to 'SimplePolygon.contains'.")
            return impl.contains(other._get_impl())
        elif x is None or y is None:
            raise TypeError("Not enough arguments to 'SimplePolygon.contains'.")
        else:
            # Quibbles about bool vs numpy.bool_ as the return type.
            return shapely.contains_xy(impl, x=x, y=y)  # type: ignore[return-value]

    @staticmethod
    def from_legacy(legacy: LegacyPolygon) -> Polygon:
        """Convert from a legacy `lsst.afw.geom.Polygon` instance."""
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

    def _get_impl(self) -> _ImplPolygon:
        if self._impl is None:
            import shapely

            self._impl = shapely.Polygon(self._vertices)
            # 'prepare' preps whatever index structures etc. might be useful
            # for accelerating various predicates.
            shapely.prepare(self._impl)
        return self._impl

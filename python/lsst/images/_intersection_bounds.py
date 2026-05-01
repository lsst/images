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

__all__ = ("IntersectionBounds",)

from typing import TYPE_CHECKING, Any, Self, cast, overload

import numpy as np

from ._geom import Bounds, Box

if TYPE_CHECKING:
    from ._concrete_bounds import SerializableBounds


class IntersectionBounds:
    """An implementation of the `Bounds` protocol that acts as a lazy
    intersection of two other `Bounds` objects.
    """

    def __init__(self, a: Bounds, b: Bounds):
        self._a = a
        self._b = b

    @property
    def bbox(self) -> Box:
        """The intersection of the bounding boxes of the operands (`.Box`)."""
        from ._concrete_bounds import _intersect_box_box

        return _intersect_box_box(self._a.bbox, self._b.bbox)

    @overload
    def contains(self, *, x: int, y: int) -> bool: ...

    @overload
    def contains(self, *, x: np.ndarray, y: np.ndarray) -> np.ndarray: ...

    def contains(self, *, x: Any, y: Any) -> Any:
        """Test whether these bounds contain one or more points.

        Parameters
        ----------
        x
            One or more integer X coordinates to test for containment.
            If an array, an array of results will be returned.
        y
            One or more integer Y coordinates to test for containment.
            If an array, an array of results will be returned.

        Returns
        -------
        `bool` | `numpy.ndarray`
            If ``x`` and ``y`` are both scalars, a single `bool` value.  If
            ``x`` and ``y`` are arrays, a boolean array with their broadcasted
            shape.
        """
        return np.logical_and(self._a.contains(x=x, y=y), self._b.contains(x=x, y=y))

    def intersection(self, other: Bounds) -> Bounds:
        """Compute the intersection of this bounds object with another.

        Notes
        -----
        Bounds intersection is guaranteed to raise `NoOverlapError` when the
        operand bounding boxes do not overlap, but it may return a bounds
        implementation that contains no points in more complex cases.
        """
        from ._concrete_bounds import _intersect_ib

        return _intersect_ib(self, other)

    def serialize(self) -> SerializableBounds:
        """Convert a bounds instance into a serializable object."""
        # Cyclic dependencies prevent IntersectionBoundsSerializationModel
        # from being defined here.
        from ._concrete_bounds import IntersectionBoundsSerializationModel

        return IntersectionBoundsSerializationModel(a=self._a.serialize(), b=self._b.serialize())

    @classmethod
    def deserialize(cls, serialized: SerializableBounds) -> Self:
        """Convert a serialized bounds object into its in-memory form."""
        from ._concrete_bounds import deserialize_bounds

        return cast(Self, deserialize_bounds(serialized))

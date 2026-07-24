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

from typing import TYPE_CHECKING, Any, assert_type, overload

import numpy as np
import numpy.typing as npt

from ._geom import XY, YX, Bounds, Box

if TYPE_CHECKING:
    from ._concrete_bounds import IntersectionBoundsSerializationModel


class IntersectionBounds:
    """An implementation of the `Bounds` protocol that acts as a lazy
    intersection of two other `Bounds` objects.

    Parameters
    ----------
    a
        First operand of the intersection.
    b
        Second operand of the intersection.
    """

    def __init__(self, a: Bounds, b: Bounds) -> None:
        self._a = a
        self._b = b

    def __str__(self) -> str:
        return f"({self._a}) ∩ ({self._b})"

    def __repr__(self) -> str:
        return f"IntersectionBounds(a={self._a!r}, b={self._b!r})"

    @property
    def bbox(self) -> Box:
        """The intersection of the bounding boxes of the operands (`.Box`)."""
        from ._concrete_bounds import _intersect_box_box

        return _intersect_box_box(self._a.bbox, self._b.bbox)

    @overload
    def contains(self, point: XY[int | float] | YX[int | float], /) -> bool: ...

    @overload
    def contains(self, point: XY[npt.ArrayLike] | YX[npt.ArrayLike], /) -> np.ndarray: ...

    @overload
    def contains(self, /, *, x: int | float, y: int | float) -> bool: ...

    @overload
    def contains(self, /, *, x: npt.ArrayLike, y: npt.ArrayLike) -> np.ndarray: ...

    def contains(self, point: XY[Any] | YX[Any] | None = None, /, *, x: Any = None, y: Any = None) -> Any:
        """Test whether these bounds contain one or more points.

        Parameters
        ----------
        point
            An `XY` or `YX` coordinate pair to test for containment.
            Mutually exclusive with ``x`` and ``y``.
        x
            One or more X coordinates to test for containment, as a scalar or
            any array-like.  Results are broadcast against ``y``.
            Mutually exclusive with ``point``.
        y
            One or more Y coordinates to test for containment, as a scalar or
            any array-like.  Results are broadcast against ``x``.
            Mutually exclusive with ``point``.

        Returns
        -------
        `bool` | `numpy.ndarray`
            If ``x`` and ``y`` are both scalars, a single `bool` value.  If
            ``x`` and ``y`` are array-like, a boolean array with their
            broadcasted shape.
        """
        match point:
            case None:
                if x is None or y is None:
                    raise TypeError("Pass either a point or both x= and y= to 'IntersectionBounds.contains'.")
            case XY() | YX():
                if x is not None or y is not None:
                    raise TypeError(
                        "'IntersectionBounds.contains' point argument is mutually exclusive with x= and y=."
                    )
                x, y = point.x, point.y
            case _:
                raise TypeError(f"Unexpected positional argument type: {type(point)!r}.")
        return np.logical_and(self._a.contains(x=x, y=y), self._b.contains(x=x, y=y))

    def intersection(self, other: Bounds) -> Bounds:
        """Compute the intersection of this bounds object with another.

        Parameters
        ----------
        other
            Bounds to intersect with this one.

        Notes
        -----
        Bounds intersection is guaranteed to raise `NoOverlapError` when the
        operand bounding boxes do not overlap, but it may return a bounds
        implementation that contains no points in more complex cases.
        """
        from ._concrete_bounds import _intersect_ib

        return _intersect_ib(self, other)

    def serialize(self) -> IntersectionBoundsSerializationModel:
        """Convert a bounds instance into a serializable object."""
        # Cyclic dependencies prevent IntersectionBoundsSerializationModel
        # from being defined here.
        from ._concrete_bounds import IntersectionBoundsSerializationModel

        return IntersectionBoundsSerializationModel(a=self._a.serialize(), b=self._b.serialize())


if TYPE_CHECKING:

    def _test_types() -> None:
        arr = np.zeros(3)
        a = Box.from_shape((10, 20))
        ib = IntersectionBounds(a, a)

        # IntersectionBounds satisfies the Bounds Protocol.
        bounds: Bounds = ib

        # IntersectionBounds.contains: XY/YX, scalar, array-like
        assert_type(ib.contains(x=1, y=2), bool)
        assert_type(ib.contains(x=1.0, y=2.0), bool)
        assert_type(ib.contains(x=arr, y=arr), np.ndarray)
        assert_type(ib.contains(XY(1, 2)), bool)
        assert_type(ib.contains(YX(2, 1)), bool)
        assert_type(ib.contains(XY(arr, arr)), np.ndarray)
        assert_type(ib.contains(YX(arr, arr)), np.ndarray)

        # Via the Bounds Protocol view, same signatures hold.
        assert_type(bounds.contains(x=1, y=1), bool)
        assert_type(bounds.contains(x=1.0, y=1.0), bool)
        assert_type(bounds.contains(x=arr, y=arr), np.ndarray)
        assert_type(bounds.contains(XY(1, 1)), bool)
        assert_type(bounds.contains(YX(1, 1)), bool)
        assert_type(bounds.contains(XY(arr, arr)), np.ndarray)
        assert_type(bounds.contains(YX(arr, arr)), np.ndarray)

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

__all__ = ("BaseField",)

from abc import ABC, abstractmethod
from typing import Literal, Self, overload

import astropy.units
import numpy as np

from .._geom import Bounds, Box
from .._image import Image


class BaseField(ABC):
    """An abstract base class for parametric or interpolated 2-d functions,
    generally representing some sort of calculated image.

    Notes
    -----
    The field hierarchy is closed to the types in this package, so we can
    enumerate all of the serializations and avoid any kind of extension system.
    All field types are immutable.

    Field types implement the function call operator and both multiplication
    and division by a constant via operator overloading.  See the named
    `evaluate` and `multiply_constant` methods (respectively) for more
    information about those operations.

    This interface will probably change in the future to incorporate options
    for dealing with out-of-bounds positions.  At present the behavior for
    such positions is implementation-specific and should not be relied upon.
    """

    @property
    @abstractmethod
    def bounds(self) -> Bounds:
        """The region over which this field can be evaluated (`.Bounds`)."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def unit(self) -> astropy.units.UnitBase | None:
        """The units of the field (`astropy.units.UnitBase` or `None`)."""
        raise NotImplementedError()

    @overload
    def __call__(self, *, x: np.ndarray, y: np.ndarray, quantity: Literal[False] = False) -> np.ndarray: ...

    @overload
    def __call__(
        self, *, x: np.ndarray, y: np.ndarray, quantity: Literal[True]
    ) -> astropy.units.Quantity: ...

    @overload
    def __call__(
        self, *, x: np.ndarray, y: np.ndarray, quantity: bool
    ) -> np.ndarray | astropy.units.Quantity: ...

    def __call__(
        self, *, x: np.ndarray, y: np.ndarray, quantity: bool = False
    ) -> np.ndarray | astropy.units.Quantity:
        return self.evaluate(x=x, y=y, quantity=quantity)

    @abstractmethod
    def render(
        self,
        bbox: Box | None = None,
        *,
        dtype: np.typing.DTypeLike | None = None,
    ) -> Image:
        """Create an image realization of the field.

        Parameters
        ----------
        bbox
            Bounding box of the image.  If not provided, ``self.bounds.bbox``
            will be used.
        dtype
            Pixel data type for the returned image.
        """
        raise NotImplementedError()

    def __mul__(self, factor: float | astropy.units.Quantity | astropy.units.UnitBase) -> Self:
        return self.multiply_constant(factor)

    def __rmul__(self, factor: float | astropy.units.Quantity | astropy.units.UnitBase) -> Self:
        return self.multiply_constant(factor)

    def __truediv__(self, factor: float | astropy.units.Quantity | astropy.units.UnitBase) -> Self:
        return self.multiply_constant(1.0 / factor)

    @abstractmethod
    def evaluate(
        self, *, x: np.ndarray, y: np.ndarray, quantity: bool
    ) -> np.ndarray | astropy.units.Quantity:
        """Evaluate at non-gridded points.

        Parameters
        ----------
        x
            X coordinates to evaluate at.
        y
            Y coordinates to evaluate at; must be broadcast-compatible with
            ``x``.
        quantity
            If `True`, return an `astropy.units.Quantity` instead of a
            `numpy.ndarray`.  If `unit` is `None`, the returned object will
            be a dimensionless `~astropy.units.Quantity`.
        """
        raise NotImplementedError()

    @abstractmethod
    def multiply_constant(self, factor: float | astropy.units.Quantity | astropy.units.UnitBase) -> Self:
        """Multiply by a constant, returning a new field of the same type.

        Parameters
        ----------
        factor
            Factor to multiply by.  When this has units, those should multiply
            ``self.unit`` or set the units of the returned field if
            ``self.unit is None``.
        """
        raise NotImplementedError()

    def _handle_factor_units(
        self, factor: float | astropy.units.Quantity | astropy.units.UnitBase
    ) -> tuple[float, astropy.units.UnitBase | None]:
        """Interpret the ``factor`` argument to `multiply_constant` and apply
        any units it carries to this field's units.

        This is a convenience function for subclass implementations of
        `multiply_constant`.

        Parameters
        ----------
        factor
            Factor passed by the caller.

        Returns
        -------
        `float`
            The factor to multiply by as a pure `float`
        `astropy.units.UnitBase` | `None`
            The units for the new field returned by `multiply_constant`.
        """
        unit = self.unit
        factor_unit = None
        if isinstance(factor, astropy.units.Quantity):
            factor_unit = factor.unit
            factor = factor.to_value()
        elif isinstance(factor, astropy.units.UnitBase):
            factor_unit = factor
            factor = 1.0
        if factor_unit is not None:
            if unit is None:
                unit = factor_unit
            else:
                unit *= factor_unit
        return factor, unit

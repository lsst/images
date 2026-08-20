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

__all__ = ("Ellipse", "EllipseArray", "InvalidEllipseError")

import dataclasses
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, Literal, Self, overload

import astropy.units
import numpy as np

from ._geom import XY, YX
from ._image import Image


class InvalidEllipseError(ValueError):
    """Exception raised when parameters do not define a valid ellipse."""


# _GenericEllipse provides common implementation for Ellipse (scalar) and
# EllipseArray.  Unfortunately we have to add trivial forwarders for most
# attributes and methods anyway because:
#
# - the generics can't quite capture the differences (especially w.r.t. types
#   can be coerced to float or numpy.ndarray)
# - Sphinx doesn't handle generics well, and some docstrings need to differ
#   anyway.
#
# But this base class at least centralizes all of the mathematical logic, so
# the duplication is just boilerplate.
class _GenericEllipse[T: float | np.ndarray](ABC):
    def __init__(
        self,
        *,
        xx: Any | None = None,
        yy: Any | None = None,
        xy: Any | None = None,
        a: Any | None = None,
        b: Any | None = None,
        theta: astropy.units.Quantity | None = None,
    ):
        self._moments: _EllipseMoments | None = None
        self._axes: _EllipseAxes | None = None
        if xx is not None and yy is not None and xy is not None:
            self._moments = _EllipseMoments(self._coerce(xx), self._coerce(yy), self._coerce(xy))
        if a is not None and b is not None and theta is not None:
            self._axes = _EllipseAxes(
                self._coerce(a), self._coerce(b), self._coerce(theta.to_value(astropy.units.rad))
            )
        if self._moments is None and self._axes is None:
            raise TypeError("Either all of (xx, yy, xy) or all of (a, b, theta) must be provided.")

    @classmethod
    def from_reduced_shear(
        cls,
        g: Any | None = None,
        *,
        g1: Any | None = None,
        g2: Any | None = None,
        r_tr: Any | None = None,
        r_det: Any | None = None,
        tr: Any | None = None,
        det: Any | None = None,
    ) -> Self:
        if g is not None:
            if g1 is not None or g2 is not None:
                raise TypeError("g is mutually exclusive with g1 and g2.")
            g1 = g.real
            g2 = g.imag
        if g1 is None or g2 is None:
            raise TypeError("Either g or both g1 and g2 must be provided.")
        if tr is not None:
            if r_tr is not None:
                raise TypeError("tr and r_tr may not both be provided.")
            r_tr = (0.5 * tr) ** 0.5
        r_det_sq = None
        if det is not None:
            if r_det is not None:
                raise TypeError("det and r_det may not both be provided.")
            r_det_sq = det**0.25
        elif r_det is not None:
            r_det_sq = r_det**2
        g = (g1**2 + g2**2) ** 0.5
        q = (1 - g) / (1 + g)
        if r_tr is None:
            if r_det_sq is None:
                r_det_sq = 1.0
            a = (r_det_sq / q) ** 0.5
        elif r_det_sq is not None:
            raise TypeError("At most one of r_tr, tr, r_det, and det may be provided.")
        else:
            a = r_tr * (2.0 / (1.0 + q**2)) ** 0.5
        b = q * a
        theta = 0.5 * np.atan2(g2, g1)
        return cls(a=a, b=b, theta=theta * astropy.units.rad)

    def _require_moments(self) -> _EllipseMoments:
        if self._moments is None:
            assert self._axes is not None, "Guaranteed at construction."
            self._moments = self._axes.to_moments()
        return self._moments

    def _require_axes(self) -> _EllipseAxes:
        if self._axes is None:
            assert self._moments is not None, "Guaranteed at construction."
            self._axes = self._moments.to_axes()
        return self._axes

    @classmethod
    @abstractmethod
    def _coerce(cls, value: Any) -> T:
        """Convert a number- or array-like argument to the correct type."""
        raise NotImplementedError()

    @property
    def xx(self) -> T:
        return self._require_moments().xx

    @property
    def yy(self) -> T:
        return self._require_moments().yy

    @property
    def xy(self) -> T:
        return self._require_moments().xy

    @property
    def a(self) -> T:
        return self._require_axes().a

    @property
    def b(self) -> T:
        return self._require_axes().b

    @property
    def theta(self) -> astropy.units.Quantity:
        return self._require_axes().theta * astropy.units.rad

    @cached_property
    def matrix(self) -> np.ndarray:
        array = np.array([[self.xx, self.xy], [self.xy, self.yy]], dtype=np.float64)
        array.flags.writeable = False
        return array

    @cached_property
    def det(self) -> T:
        if self._moments is not None:
            return self._moments.xx * self._moments.yy - self._moments.xy**2
        else:
            assert self._axes is not None, "Guaranteed at construction."
            return (self._axes.a * self._axes.b) ** 2

    @cached_property
    def tr(self) -> T:
        if self._moments is not None:
            return self._moments.xx + self._moments.yy
        else:
            assert self._axes is not None, "Guaranteed at construction."
            return self._axes.a**2 + self._axes.b**2

    @cached_property
    def area(self) -> T:
        return self._coerce(np.pi * self.det**0.5)

    @cached_property
    def r_det(self) -> T:
        return self._coerce(self.det**0.25)

    @cached_property
    def r_tr(self) -> T:
        return self._coerce((0.5 * self.tr) ** 0.5)

    @cached_property
    def reduced_shear(self) -> Any:
        if self._moments is not None:
            d = self.tr + 2.0 * (self.det**0.5)
            return (self._moments.xx - self._moments.yy + 2j * self._moments.xy) / d
        else:
            assert self._axes is not None, "Guaranteed at construction."
            g = (self._axes.a - self._axes.b) / (self._axes.a + self._axes.b)
            return g * np.exp(2j * self._axes.theta)


class Ellipse(_GenericEllipse[float]):
    """An ellipse on a Cartesian coordinate grid.

    Parameters
    ----------
    xx
        The x^2 moment of the image, or the corresponding element in a
        covariance or moments matrix.  Mutually exclusive with
        ``(a, b, theta)``.
    yy
        The y^2 moment of the image, or the corresponding element in a
        covariance or moments matrix.  Mutually exclusive with
        ``(a, b, theta)``.
    xy
        The x*y moment of the image, or the corresponding element in a
        covariance or moments matrix.  Mutually exclusive with
        ``(a, b, theta)``.
    a
        The semi-major axis of the ellipse.  Mutually exclusive with
        ``(xx, yy, xy)``.  If smaller than ``b``, ``a`` and ``b``` will be
        swapped and ``theta`` will be offset by 90 degrees.
    b
        The semi-minor axis of the ellipse.  Mutually exclusive with
        ``(xx, yy, xy)``.
    theta
        The orientation of the ellipse, measured counterclockwise from the
        positive X axis to the semi-major axis, in the canonical range
        ``[-pi/2, pi/2)``.  Mutually exclusive with
        ``(xx, yy, xy)``.
    center
        Center of the ellipse.  Mutually exclusive with ``(x, y)``.
    x
        Center X coordinate of the ellipse.  Mutually exclusive with
        ``center``.
    y
        Center Y coordinate of the ellipse.  Mutually exclusive with
        ``center``.

    Notes
    -----
    See `EllipseArray` for vectorized array operations on ellipse parameters.
    """

    @overload
    def __init__(
        self,
        *,
        xx: float,
        yy: float,
        xy: float,
        center: XY[float] | YX[float] | None = None,
        x: float | None = None,
        y: float | None = None,
    ): ...
    @overload
    def __init__(
        self,
        *,
        a: float,
        b: float,
        theta: astropy.units.Quantity,
        center: XY[float] | YX[float] | None = None,
        x: float | None = None,
        y: float | None = None,
    ): ...
    def __init__(
        self,
        *,
        xx: float | None = None,
        yy: float | None = None,
        xy: float | None = None,
        a: float | None = None,
        b: float | None = None,
        theta: astropy.units.Quantity | None = None,
        center: XY[float] | YX[float] | None = None,
        x: float | None = None,
        y: float | None = None,
    ):
        super().__init__(xx=xx, yy=yy, xy=xy, a=a, b=b, theta=theta)
        self._init_center(center, x=x, y=y)
        if self._moments is not None:
            if not np.isfinite(self._moments.xx):
                raise InvalidEllipseError(f"xx={self._moments.xx} must be finite.")
            if not np.isfinite(self._moments.yy):
                raise InvalidEllipseError(f"yy={self._moments.yy} must be finite.")
            if not np.isfinite(self._moments.xy):
                raise InvalidEllipseError(f"xy={self._moments.xy} must be finite.")
            if self._moments.xx < 0:
                raise InvalidEllipseError(f"xx={self._moments.xx} must be non-negative.")
            if self._moments.yy < 0:
                raise InvalidEllipseError(f"yy={self._moments.yy} must be non-negative.")
            if self._moments.xx * self._moments.yy < self._moments.xy**2:
                raise InvalidEllipseError("Determinant must be non-negative.")
        if self._axes is not None:
            if not np.isfinite(self._axes.a):
                raise InvalidEllipseError(f"a={self._axes.a} must be finite.")
            if not np.isfinite(self._axes.b):
                raise InvalidEllipseError(f"b={self._axes.b} must be finite.")
            if not np.isfinite(self._axes.theta):
                raise InvalidEllipseError(f"theta={self._axes.theta} must be finite.")
            if self._axes.a < 0:
                raise InvalidEllipseError(f"a={self._axes.a} must be non-negative.")
            if self._axes.b < 0:
                raise InvalidEllipseError(f"b={self._axes.b} must be non-negative.")
            if self._axes.a < self._axes.b:
                self._axes.a, self._axes.b = self._axes.b, self._axes.a
                self._axes.theta += 0.5 * np.pi
            self._axes.theta = (self._axes.theta + 0.5 * np.pi) % np.pi - 0.5 * np.pi

    @classmethod
    def from_reduced_shear(
        cls,
        g: complex | None = None,
        *,
        g1: float | None = None,
        g2: float | None = None,
        r_tr: float | None = None,
        r_det: float | None = None,
        tr: float | None = None,
        det: float | None = None,
        center: XY[float] | YX[float] | None = None,
        x: float | None = None,
        y: float | None = None,
    ) -> Self:
        r"""Construct from reduced shear and a radius.

        Parameters
        ----------
        g
            Complex ellipticity in "reduced shear" form.  Mutually exclusive
            with ``g1`` and ``g2``.
        g1
            Real component of ellipticity (axis-aligned), in "reduced shear"
            form.  Mutually exclusive with ``g``.
        g2
            Imaginary component of ellipticity (45 degrees off-axis), in
            "reduced shear" form.  Mutually exclusive with ``g``.
        r_tr
            Trace-based radius for the ellipse (see the `r_tr` property).
        r_det
            Determinant-based radius for the ellipse (see the `r_det`
            property).  At most one of ``r_tr`` and ``r_det`` may be provided.
            If neither is provided, an ellipse with ``r_det = 1`` is created.
        tr
            Trace of the ellipse's moments matrix.
        det
            Determinant of the ellipse's moments matrix.
        center
            Center of the ellipse.  Mutually exclusive with ``(x, y)``.
        x
            Center X coordinate of the ellipse.  Mutually exclusive with
            ``center``.
        y
            Center Y coordinate of the ellipse.  Mutually exclusive with
            ``center``.

        Notes
        -----
        The complex reduced shear ``g`` is related to the semi-major axis `a`,
        the semi-minor axis `b`, and the positional angle `theta` by

        .. math::
            g = \frac{a - b}{a + b} e^{2i\theta}

        and to the moments by

        .. math::
            g = \frac{I_{xx} - I_{yy} + 2i I_{xy}}}{I_{xx} + I_{yy}
                + 2 \sqrt{I_{xx} I_{yy} - I_{xy}^2}}
        """
        result = super().from_reduced_shear(g, g1=g1, g2=g2, r_tr=r_tr, r_det=r_det, tr=tr, det=det)
        result._init_center(center, x=x, y=y)
        return result

    @classmethod
    def from_image_unweighted(
        cls,
        image: Image,
        *,
        center: XY[float] | YX[float] | None = None,
        x: float | None = None,
        y: float | None = None,
    ) -> Ellipse:
        """Measure the unweighted second moments of an image.

        Parameters
        ----------
        image
            The image to measure.
        center
            Fixed pixel position to measure the second moments about (mutually
            exclusive with ``x`` and ``y``).
        x
            Fixed X pixel coordinate to measure the second moments about.  If
            not provided, the unweighted centroid is measured.
        y
            Fixed Y pixel coordinate to measure the second moments about.  If
            not provided, the unweighted centroid is measured.
        """
        m0 = image.array.sum()
        xj = image.bbox.x.arange.astype(float)
        yi = image.bbox.y.arange.astype(float)
        if center is not None:
            if x is not None or y is not None:
                raise TypeError("At most one of '(x, y)' or 'center' can be provided.")
            x = center.x
            y = center.y
        else:
            if x is None:
                x = np.einsum("ij,j->", image.array, xj) / m0
            if y is None:
                y = np.einsum("ij,i->", image.array, yi) / m0
        x = float(x)
        y = float(y)
        xj -= x
        yi -= y
        return cls(
            xx=np.einsum("ij,j,j->", image.array, xj, xj) / m0,
            yy=np.einsum("ij,i,i->", image.array, yi, yi) / m0,
            xy=np.einsum("ij,i,j->", image.array, yi, xj) / m0,
            center=XY(x=x, y=y),
        )

    @classmethod
    def _coerce(cls, value: Any) -> float:
        return float(value)

    def _init_center(
        self,
        center: XY[float] | YX[float] | None = None,
        *,
        x: float | None = None,
        y: float | None = None,
    ) -> None:
        match center, x, y:
            case [None, None, None]:
                self._center = None
            case [XY(), None, None]:
                self._center = center.map(float)
            case [YX(), None, None]:
                self._center = center.xy.map(float)
            case [None, float(), float()]:
                self._center = XY(x=float(x), y=float(y))
            case _:
                raise TypeError(f"Invalid center arguments x={x!r}, y={y!r}, center={center!r} for ellipse.")

    @property
    def xx(self) -> float:
        """The x^2 moment of the image, or the corresponding element in a
        bivariate covariance matrix (`float`).
        """
        return self._require_moments().xx

    @property
    def yy(self) -> float:
        """The y^2 moment of the image, or the corresponding element in a
        bivariate covariance matrix (`float`).
        """
        return self._require_moments().yy

    @property
    def xy(self) -> float:
        """The x*y moment of the image, or the corresponding element in a
        bivariate covariance matrix (`float`).
        """
        return self._require_moments().xy

    @property
    def a(self) -> float:
        """The semi-major axis of the ellipse (`float`)."""
        return self._require_axes().a

    @property
    def b(self) -> float:
        """The semi-minor axis of the ellipse (`float`)."""
        return self._require_axes().b

    @property
    def theta(self) -> astropy.units.Quantity:
        """The position angle of the ellipse, measured counterclockwise from
        the positive X axis to the semi-major axis, in the canonical range
        ``[-pi/2, pi/2)`` (`astropy.units.Quantity`).
        """
        return super().theta

    @cached_property
    def matrix(self) -> np.ndarray:
        """The 2x2 covariance matrix that corresponds to this ellipse
        (`numpy.ndarray`).

        For array-valued ellipses, this has shape ``(2, 2, ...)``.
        """
        return super().matrix

    @property
    def det(self) -> float:
        """The determinant of the covariance or moments matrix (`float`)."""
        return super().det

    @property
    def tr(self) -> float:
        """The trace of the covariance or moments matrix (`float`)."""
        return super().tr

    @property
    def area(self) -> float:
        """The area of the ellipse (`float`)."""
        return super().area

    @property
    def r_det(self) -> float:
        """A radius equal to ``det**0.25`` and proportional to ``sqrt(area)``
        (`float`).

        This radius goes to zero as the ellipse approaches a line segment.
        """
        return super().r_det

    @property
    def r_tr(self) -> float:
        """A radius equal to ``sqrt(0.5*tr)`` (`float`).

        This radius approaches the semi-major axis size as the ellipse
        approaches a line segment.
        """
        return super().r_tr

    @property
    def reduced_shear(self) -> complex:
        """The complex ellipticity of the ellipse in the reduced shear
        parameterization (`float`).
        """
        return super().reduced_shear

    @property
    def center(self) -> XY[float]:
        """The center of the ellipse (`XY`).

        An ellipse can be constructed without a center, this can raise
        `AttributeError`.  Use `hasattr` to test whether the attribute exists.
        """
        if self._center is None:
            raise AttributeError("center")
        return self._center

    def remeasure_adaptive(self, image: Image) -> Ellipse:
        """Measure the adaptive Gaussian-weighted moments of an image, using
        ``self`` as an initial guess.

        This requires the GalSim package to be importable.

        Parameters
        ----------
        image
            The image to measure.
        """
        import galsim.hsm

        galsim_image = image.to_galsim()
        guess_centroid = self.center.to_galsim_float_position() if self._center is not None else None
        result = galsim.hsm.FindAdaptiveMom(galsim_image, guess_sig=self.r_tr, guess_centroid=guess_centroid)
        return Ellipse.from_reduced_shear(
            g1=result.observed_shape.g1,
            g2=result.observed_shape.g2,
            r_det=result.moments_sigma,
            center=XY(x=result.moments_centroid.x, y=result.moments_centroid.y),
        )


class EllipseArray(_GenericEllipse[np.ndarray]):
    """A container of ellipses backed by arrays.

    Parameters
    ----------
    xx
        The x^2 moment of the image, or the corresponding element in a
        covariance or moments matrix.  Mutually exclusive with
        ``(a, b, theta)``.
    yy
        The y^2 moment of the image, or the corresponding element in a
        covariance or moments matrix.  Mutually exclusive with
        ``(a, b, theta)``.
    xy
        The x*y moment of the image, or the corresponding element in a
        covariance or moments matrix.  Mutually exclusive with
        ``(a, b, theta)``.
    a
        The semi-major axis of the ellipse.  Mutually exclusive with
        ``(xx, yy, xy)``.
    b
        The semi-minor axis of the ellipse.  Mutually exclusive with
        ``(xx, yy, xy)``.
    theta
        The orientation of the ellipse, measured counterclockwise from the
        positive X axis to the semi-major axis, in the canonical range
        ``[-pi/2, pi/2)``.  Mutually exclusive with
        ``(xx, yy, xy)``.
    copy
        Whether to copy the input arrays:

        - ``"as-needed"`` (default) copies only arrays whose shapes need to
          change for broadcasting;
        - `True` always makes independent copies;
        - `False` never copies, raising `ValueError` if the arrays do not
          already have the same shape.

    Notes
    -----
    In order to support arrays that have a mix of valid and invalid ellipses,
    this class does not automatically check validity and never raises
    `InvalidEllipseError`.  Instead, use `standardize` to obtain a mask
    of invalid ellipses.

    The input arrays are broadcast to the same shape.

    This class aggressively caches arrays, without checking for changes to
    either the arrays it is constructed with or any arrays it returns as
    properties.  Users should never modify these arrays in place.
    """

    @overload
    def __init__(
        self,
        *,
        xx: np.typing.ArrayLike,
        yy: np.typing.ArrayLike,
        xy: np.typing.ArrayLike,
        copy: Literal["as-needed"] | bool = "as-needed",
    ): ...

    @overload
    def __init__(
        self,
        *,
        a: np.typing.ArrayLike,
        b: np.typing.ArrayLike,
        theta: astropy.units.Quantity,
        copy: Literal["as-needed"] | bool = "as-needed",
    ): ...

    def __init__(
        self,
        *,
        xx: np.typing.ArrayLike | None = None,
        yy: np.typing.ArrayLike | None = None,
        xy: np.typing.ArrayLike | None = None,
        a: np.typing.ArrayLike | None = None,
        b: np.typing.ArrayLike | None = None,
        theta: astropy.units.Quantity | None = None,
        copy: Literal["as-needed"] | bool = "as-needed",
    ):
        if not isinstance(copy, bool) and copy != "as-needed":
            raise TypeError("'copy' must be a bool or 'as-needed'.")
        super().__init__(xx=xx, yy=yy, xy=xy, a=a, b=b, theta=theta)
        shapes = []
        if self._moments is not None:
            shapes.extend([self._moments.xx.shape, self._moments.yy.shape, self._moments.xy.shape])
        if self._axes is not None:
            shapes.extend([self._axes.a.shape, self._axes.b.shape, self._axes.theta.shape])
        self._shape = np.broadcast_shapes(*shapes)
        if self._moments is not None:
            self._moments = _EllipseMoments(
                self._broadcast_or_copy(self._moments.xx, self._shape, copy=copy),
                self._broadcast_or_copy(self._moments.yy, self._shape, copy=copy),
                self._broadcast_or_copy(self._moments.xy, self._shape, copy=copy),
            )
        if self._axes is not None:
            self._axes = _EllipseAxes(
                self._broadcast_or_copy(self._axes.a, self._shape, copy=copy),
                self._broadcast_or_copy(self._axes.b, self._shape, copy=copy),
                self._broadcast_or_copy(self._axes.theta, self._shape, copy=copy),
            )

    @classmethod
    def from_reduced_shear(
        cls,
        g: np.typing.ArrayLike | None = None,
        *,
        g1: np.typing.ArrayLike | None = None,
        g2: np.typing.ArrayLike | None = None,
        r_tr: np.typing.ArrayLike | None = None,
        r_det: np.typing.ArrayLike | None = None,
        tr: np.typing.ArrayLike | None = None,
        det: np.typing.ArrayLike | None = None,
    ) -> Self:
        r"""Construct from reduced shear and a radius.

        Parameters
        ----------
        g
            Complex ellipticity in "reduced shear" form.  Mutually exclusive
            with ``g1`` and ``g2``.
        g1
            Real component of ellipticity (axis-aligned), in "reduced shear"
            form.  Mutually exclusive with ``g``.
        g2
            Imaginary component of ellipticity (45 degrees off-axis), in
            "reduced shear" form.  Mutually exclusive with ``g``.
        r_tr
            Trace-based radius for the ellipse (see the `r_tr` property).
        r_det
            Determinant-based radius for the ellipse (see the `r_det`
            property).  At most one of ``r_tr`` and ``r_det`` may be provided.
            If neither is provided, an ellipse with ``r_det = 1`` is created.
        tr
            Trace of the ellipse's moments matrix.
        det
            Determinant of the ellipse's moments matrix.

        Notes
        -----
        The complex reduced shear ``g`` is related to the semi-major axis `a`,
        the semi-minor axis `b`, and the positional angle `theta` by

        .. math::
            g = \frac{a - b}{a + b} e^{2i\theta}

        and to the moments by

        .. math::
            g = \frac{I_{xx} - I_{yy} + 2i I_{xy}}}{I_{xx} + I_{yy}
                + 2 \sqrt{I_{xx} I_{yy} - I_{xy}^2}}
        """
        return super().from_reduced_shear(g, g1=g1, g2=g2, r_tr=r_tr, r_det=r_det, tr=tr, det=det)

    @classmethod
    def _coerce(cls, value: Any) -> np.ndarray:
        return np.asarray(value)

    @staticmethod
    def _broadcast_or_copy(
        array: np.ndarray, shape: tuple[int, ...], *, copy: Literal["as-needed"] | bool
    ) -> np.ndarray:
        """Broadcast ``array`` to the given shape."""
        if array.shape == shape:
            if copy is True:
                return array.copy()
            return array
        if not copy:
            raise ValueError(
                "Cannot avoid a copy while broadcasting an EllipseArray input to "
                f"shape {shape}; use copy=True or the default copy='as-needed' instead."
            )
        return np.broadcast_to(array, shape).copy()

    @property
    def shape(self) -> tuple[int, ...]:
        """The broadcasted shape of all input arrays."""
        return self._shape

    @overload
    def __getitem__(self, index: int) -> Ellipse: ...

    @overload
    def __getitem__(self, index: slice | tuple[slice, ...] | np.ndarray) -> EllipseArray: ...

    def __getitem__(self, index: Any) -> Any:
        kwargs: dict[str, Any] = {}
        is_scalar = True
        if self._moments is not None:
            kwargs["xx"] = self._moments.xx[index]
            kwargs["yy"] = self._moments.yy[index]
            kwargs["xy"] = self._moments.xy[index]
            is_scalar = np.ndim(kwargs["xx"]) == 0
        if self._axes is not None:
            kwargs["a"] = self._axes.a[index]
            kwargs["b"] = self._axes.b[index]
            kwargs["theta"] = self._axes.theta[index] * astropy.units.rad
            is_scalar = np.ndim(kwargs["a"]) == 0
        if is_scalar:
            return Ellipse(**kwargs)
        else:
            return EllipseArray(**kwargs)

    def __len__(self) -> int:
        return self._shape[0]

    @property
    def xx(self) -> np.ndarray:
        """The x^2 moment of the image, or the corresponding element in a
        bivariate covariance matrix (`numpy.ndarray`).
        """
        return self._require_moments().xx

    @property
    def yy(self) -> np.ndarray:
        """The y^2 moment of the image, or the corresponding element in a
        bivariate covariance matrix (`numpy.ndarray`).
        """
        return self._require_moments().yy

    @property
    def xy(self) -> np.ndarray:
        """The x*y moment of the image, or the corresponding element in a
        bivariate covariance matrix (`numpy.ndarray`).
        """
        return self._require_moments().xy

    @property
    def a(self) -> np.ndarray:
        """The semi-major axis of the ellipse (`numpy.ndarray`)."""
        return self._require_axes().a

    @property
    def b(self) -> np.ndarray:
        """The semi-minor axis of the ellipse (`numpy.ndarray`)."""
        return self._require_axes().b

    @property
    def det(self) -> np.ndarray:
        """The determinant of the covariance or moments matrix
        (`numpy.ndarray`).
        """
        return super().det

    @property
    def tr(self) -> np.ndarray:
        """The trace of the covariance or moments matrix (`numpy.ndarray`)."""
        return super().tr

    @property
    def theta(self) -> astropy.units.Quantity:
        """The position angle of the ellipse, measured counterclockwise from
        the positive X axis to the semi-major axis, in the canonical range
        ``[-pi/2, pi/2)`` (`astropy.units.Quantity`).
        """
        return super().theta

    @cached_property
    def matrix(self) -> np.ndarray:
        """The array-valued covariance matrix that corresponds to this ellipse
        (`numpy.ndarray`).

        This has shape ``(2, 2, ...)``, where ``...`` is the broadcasted shape
        of the arrays passed at construction.
        """
        return super().matrix

    @property
    def area(self) -> np.ndarray:
        """The area of the ellipse (`numpy.ndarray`)."""
        return super().area

    @property
    def r_det(self) -> np.ndarray:
        """A radius equal to ``det**0.25`` and proportional to ``sqrt(area)``
        (`numpy.ndarray`).

        This radius goes to zero as the ellipse approaches a line segment.
        """
        return super().r_det

    @property
    def r_tr(self) -> np.ndarray:
        """A radius equal to ``sqrt(0.5*tr)`` (`numpy.ndarray`).

        This radius approaches the semi-major axis size as the ellipse
        approaches a line segment.
        """
        return super().r_tr

    @property
    def reduced_shear(self) -> np.ndarray:
        """The complex ellipticity of the ellipse in the reduced shear
        parameterization (`numpy.ndarray`).
        """
        return super().reduced_shear

    def standardize(self) -> np.ndarray:
        """Check and standardize all ellipses.

        Returns
        -------
        numpy.ndarray
            A boolean array with the broadcasted shape of all input arrays
            that is `True` for valid arrays and `False` for invalid arrays.

        Notes
        -----
        If initialized with the axes parameterization, this will swap ``a``
        and ``b`` for any rows where ``a < b``, offsetting ``theta`` by 90
        degrees.
        """
        is_valid = np.ones(self.shape, dtype=bool)
        if self._moments is not None:
            is_valid[np.logical_not(np.isfinite(self._moments.xx))] = False
            is_valid[np.logical_not(np.isfinite(self._moments.yy))] = False
            is_valid[np.logical_not(np.isfinite(self._moments.xy))] = False
            is_valid[self._moments.xx < 0] = False
            is_valid[self._moments.yy < 0] = False
            is_valid[self._moments.xx * self._moments.yy < self._moments.xy**2] = False
        if self._axes is not None:
            is_valid[np.logical_not(np.isfinite(self._axes.a))] = False
            is_valid[np.logical_not(np.isfinite(self._axes.b))] = False
            is_valid[np.logical_not(np.isfinite(self._axes.theta))] = False
            is_valid[self._axes.a < 0] = False
            is_valid[self._axes.b < 0] = False
            to_flip = self._axes.a < self._axes.b
            self._axes.a[to_flip], self._axes.b[to_flip] = self._axes.b[to_flip], self._axes.a[to_flip]
            self._axes.theta[to_flip] += 0.5 * np.pi
            self._axes.theta[:] = (self._axes.theta + 0.5 * np.pi) % np.pi - 0.5 * np.pi
        return is_valid


# We use typing.Any for ellipse storage because NumPy' type stubs resolve to
# that most of the time anyway, and the typing system really can't handle
# float-array arithmetic operators.


@dataclasses.dataclass
class _EllipseAxes:
    a: Any
    b: Any
    theta: Any

    def to_moments[T](self) -> _EllipseMoments:
        aa = self.a**2
        bb = self.b**2
        c = np.cos(self.theta)
        s = np.sin(self.theta)
        xy = (aa - bb) * c * s
        cc = c**2
        ss = s**2
        xx = cc * aa + ss * bb
        yy = ss * aa + cc * bb
        return _EllipseMoments(xx=xx, yy=yy, xy=xy)


@dataclasses.dataclass
class _EllipseMoments:
    xx: Any
    yy: Any
    xy: Any

    def to_axes(self) -> _EllipseAxes:
        p = self.xx + self.yy
        m = self.xx - self.yy
        t = np.sqrt(4.0 * self.xy**2 + m**2)
        a = np.sqrt(0.5 * (p + t))
        b = np.sqrt(0.5 * (p - t))
        theta = 0.5 * np.atan2(2.0 * self.xy, m)
        return _EllipseAxes(a=a, b=b, theta=theta)

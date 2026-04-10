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

__all__ = ("ChebyshevField", "ChebyshevFieldSerializationModel")

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, final

import astropy.units
import numpy as np
import pydantic

from .._concrete_bounds import SerializableBounds, deserialize_bounds
from .._geom import YX, Bounds, Box
from .._image import Image
from ..serialization import ArchiveTree, InlineArray, InputArchive, OutputArchive, Unit
from ._base import BaseField

if TYPE_CHECKING:
    try:
        from lsst.afw.math import BackgroundMI as LegacyBackground
        from lsst.afw.math import ChebyshevBoundedField as LegacyChebyshevBoundedField
    except ImportError:
        type LegacyBackground = Any  # type: ignore[no-redef]
        type LegacyChebyshevBoundedField = Any  # type: ignore[no-redef]


@final
class ChebyshevField(BaseField):
    """A 2-d Chebyshev polynomial over a rectangular region.

    Parameters
    ----------
    bounds
        The region where this field can be evaluated.  The ``bbox`` of this
        region is grown by half a pixel on all sides and then used to remap
        coordinates to ``[-1, 1]x[-1, 1]``, which is the natural domain of a
        2-d Chebyshev polynomial.
    coefficients
        Coefficients for the 2-d Chebyshev polynomial of the first kind, as a
        2-d matrix in which element ``[p, q]`` corresponds to the coefficient
        of ``T_p(y) T_q(x)``.  Will be set to read-only in place.
    unit
        Units of the field.
    """

    def __init__(
        self, bounds: Bounds, coefficients: np.ndarray, *, unit: astropy.units.UnitBase | None = None
    ):
        self._bounds = bounds
        self._coefficients = coefficients
        self._coefficients.flags.writeable = False
        self._unit = unit
        # Compute the scaling and translation that map points in the bbox
        # (including an extra 0.5 on all sides, since the bbox is int-based)
        # to [-1, 1].
        bbox = bounds.bbox
        self._xs = 2.0 / bbox.x.size
        self._xt = bbox.x.min + 0.5 * bbox.x.size - 0.5
        self._ys = 2.0 / bbox.y.size
        self._yt = bbox.y.min + 0.5 * bbox.y.size - 0.5

    @staticmethod
    def fit(
        bounds: Bounds,
        data: np.ndarray | astropy.units.Quantity,
        order: int | None = None,
        *,
        y: np.ndarray,
        x: np.ndarray,
        weight: np.ndarray | None = None,
        y_order: int | None = None,
        x_order: int | None = None,
        triangular: bool = True,
        unit: astropy.units.UnitBase | None = None,
    ) -> ChebyshevField:
        """Fit a Chebyshev field to data points using linear least squares.

        Parameters
        ----------
        data
            Data points to fit.  If this is an `astropy.units.Quantity`,
            this sets the units of the field and the ``unit`` argument cannot
            also be provided.
        order
            Maximum order for the Chebyshev polynomial in both dimensions.
        y
            Y coordinates of the data points.  Must have either the same
            shape as ``data`` (providing the coordinates for all points
            directly), or be a 1-d array with the same size as
            ``data.shape[0]`` (when ``data`` is a 2-d image and ``y`` provides
            the coordinates of the rows).
        x
            X coordinates of the data points.  Must have either the same
            shape as ``data`` (providing the coordinates for all points
            directly), or be a 1-d array with the same size as
            ``data.shape[1]`` (when ``data`` is a 2-d image and ``x`` provides
            the coordinates of the columns).
        weight
            Weights to apply to the data points.  Must have the same shape as
            ``data``.
        y_order
            Maximum order for the Chebyshev polynomial in ``y``.  Requires
            ``x_order`` to also be provided.  Incompatible with ``order``.
        x_order
            Maximum order for the Chebyshev polynomial in ``x``.  Requires
            ``y_order`` to also be provided.  Incompatible with ``order``.
        triangular
            If `True`, only fit for coefficients of ``T_p(y) T_q(x)`` where
            ``p + q <= max(y_order, x_order)``.
        unit
            Units of the returned field.
        """
        match (order, x_order, y_order):
            case (int(), None, None):
                x_order = order
                y_order = order
            case (None, int(), int()):
                pass
            case _:
                raise TypeError("Either 'order' (only) or both 'x_order' and 'y_order' must be provided.")
        if weight is not None and weight.shape != data.shape:
            raise ValueError(f"Shape of 'data' {data.shape} does not match 'weight' {weight.shape}.")
        if isinstance(data, astropy.units.Quantity):
            if unit is not None:
                raise TypeError("If 'data' is a Quantity, 'unit' cannot be provided separately.")
            unit = data.unit
            data = data.to_value()
        result = ChebyshevField(bounds, np.zeros((y_order + 1, x_order + 1), dtype=np.float64), unit=unit)
        if len(data.shape) == 2 and len(x.shape) == 1 and len(y.shape) == 1:
            if data.shape != y.shape + x.shape:
                raise ValueError(
                    f"Shape of 2-d 'data' {data.shape} does not match 1-d 'y' {y.shape} and/or 'x' {x.shape}."
                )
            matrix = result._make_grid_matrix(x=x, y=y, triangular=triangular)
        else:
            if data.shape != y.shape:
                raise ValueError(f"Shape of 'data' {data.shape} does not match 'y' {y.shape}.")
            if data.shape != x.shape:
                raise ValueError(f"Shape of 'data' {data.shape} does not match 'x' {x.shape}.")
            matrix = result._make_general_matrix(x=x, y=y, triangular=triangular)
        if weight is not None:
            weight = weight.ravel()  # copies only if needed
            matrix *= weight[:, np.newaxis]
            data = data.flatten()  # always copies
            data *= weight
            mask = np.logical_and(weight > 0, np.isfinite(data))
        else:
            data = data.ravel()
            mask = np.isfinite(data)
        n_good = mask.sum()
        if n_good == 0:
            raise ValueError("No good data points.")
        if n_good < data.size:
            data = data[mask]
            matrix = matrix[mask, :]
        packed_coefficients, *_ = np.linalg.lstsq(matrix, data)
        result._coefficients.flags.writeable = True
        for i, pq in result._packing_indices(triangular):
            result._coefficients[pq.y, pq.x] = packed_coefficients[i]
        result._coefficients.flags.writeable = False
        return result

    @property
    def bounds(self) -> Bounds:
        return self._bounds

    @property
    def unit(self) -> astropy.units.UnitBase | None:
        return self._unit

    @property
    def x_order(self) -> int:
        """Maximum polynomial order in the column dimension (`int`)."""
        return self._coefficients.shape[1] - 1

    @property
    def y_order(self) -> int:
        """Maximum polynomial order in the row dimension (`int`)."""
        return self._coefficients.shape[0] - 1

    @property
    def order(self) -> int:
        """Maximum polynomial order in either dimension (`int`)."""
        return max(self.x_order, self.y_order)

    @property
    def coefficients(self) -> np.ndarray:
        """Coefficients for the 2-d Chebyshev polynomial of the first kind,
        as a 2-d matrix in which element ``[p, q]`` corresponds to the
        coefficient of ``T_p(y) T_q(x)``.
        """
        return self._coefficients

    def evaluate(
        self, *, x: np.ndarray, y: np.ndarray, quantity: bool
    ) -> np.ndarray | astropy.units.Quantity:
        m = self._remap(x=x.copy(), y=y.copy())
        # We swap x and y relative to Numpy's docs because that's how our
        # coefficients are ordered.
        v = np.polynomial.chebyshev.chebval2d(m.y, m.x, self._coefficients)
        if quantity:
            return astropy.units.Quantity(v, self.unit)
        return v

    def render(self, bbox: Box | None = None, *, dtype: np.typing.DTypeLike | None = None) -> Image:
        if bbox is None:
            bbox = self.bounds.bbox
        m = self._remap(
            x=bbox.x.arange.astype(np.float64),
            y=bbox.y.arange.astype(np.float64),
        )
        # We swap x and y relative to Numpy's docs because that's how our
        # coefficients and images are ordered.
        v = np.polynomial.chebyshev.chebgrid2d(m.y, m.x, self._coefficients)
        return Image(v, bbox=bbox, unit=self.unit, dtype=dtype)

    def multiply_constant(
        self, factor: float | astropy.units.Quantity | astropy.units.UnitBase
    ) -> ChebyshevField:
        factor, unit = self._handle_factor_units(factor)
        return ChebyshevField(self.bounds, self.coefficients * factor, unit=unit)

    def serialize(self, archive: OutputArchive[Any]) -> ChebyshevFieldSerializationModel:
        """Serialize the Chebyshev field to an output archive."""
        return ChebyshevFieldSerializationModel(
            bounds=self.bounds.serialize(),
            coefficients=self.coefficients,
            unit=self.unit,
        )

    @staticmethod
    def deserialize(model: ChebyshevFieldSerializationModel, archive: InputArchive[Any]) -> ChebyshevField:
        """Deserialize the Chebyshev field from an input archive."""
        return ChebyshevField(deserialize_bounds(model.bounds), model.coefficients, unit=model.unit)

    @staticmethod
    def _get_archive_tree_type(
        pointer_type: type[Any],
    ) -> type[ChebyshevFieldSerializationModel]:
        """Return the serialization model type for this object for an archive
        type that uses the given pointer type.
        """
        return ChebyshevFieldSerializationModel

    @staticmethod
    def from_legacy(
        legacy: LegacyChebyshevBoundedField, unit: astropy.units.UnitBase | None = None
    ) -> ChebyshevField:
        """Convert from a legacy `lsst.afw.math.ChebyshevBoundedField`."""
        return ChebyshevField(
            bounds=Box.from_legacy(legacy.getBBox()), coefficients=legacy.getCoefficients(), unit=unit
        )

    def to_legacy(self) -> LegacyChebyshevBoundedField:
        """Convert to a legacy `lsst.afw.math.ChebyshevBoundedField`."""
        from lsst.afw.math import ChebyshevBoundedField as LegacyChebyshevBoundedField

        return LegacyChebyshevBoundedField(self.bounds.bbox.to_legacy(), self.coefficients)

    @staticmethod
    def from_legacy_background(
        legacy_background: LegacyBackground,
        unit: astropy.units.UnitBase | None = None,
    ) -> ChebyshevField:
        """Convert from a legacy `lsst.afw.math.BackgroundMI` instance."""
        from lsst.afw.math import ApproximateControl

        approx_control = legacy_background.getBackgroundControl().getApproximateControl()
        stats_image = legacy_background.getStatsImage()
        if approx_control.getStyle() != ApproximateControl.CHEBYSHEV:
            raise TypeError("Legacy background does not use Chebyshev approximation.")
        if approx_control.getWeighting():
            weight = stats_image.variance.array ** (-0.5)
        else:
            weight = None
        x = legacy_background.getBinCentersX()
        y = legacy_background.getBinCentersY()
        return ChebyshevField.fit(
            Box.from_legacy(legacy_background.getImageBBox()),
            stats_image.image.array,
            x=x,
            y=y,
            x_order=approx_control.getOrderX(),
            y_order=approx_control.getOrderY(),
            weight=weight,
            unit=unit,
        )

    def _remap(self, *, x: np.ndarray, y: np.ndarray) -> YX[np.ndarray]:
        x -= self._xt
        x *= self._xs
        y -= self._yt
        y *= self._ys
        return YX(y=y, x=x)

    def _packing_indices(self, triangular: bool) -> Iterator[tuple[int, YX[int]]]:
        i = 0
        for p in range(self.y_order + 1):
            for q in range(self.x_order + 1):
                if not triangular or p + q <= self.order:
                    yield i, YX(y=p, x=q)
                    i += 1

    def _make_grid_matrix(self, *, x: np.ndarray, y: np.ndarray, triangular: bool) -> np.ndarray:
        r = self._remap(
            x=np.asarray(x, dtype=np.float64, copy=True),
            y=np.asarray(y, dtype=np.float64, copy=True),
        )
        yv = np.polynomial.chebyshev.chebvander(r.y, self.y_order)
        xv = np.polynomial.chebyshev.chebvander(r.x, self.x_order)
        indices = list(self._packing_indices(triangular))
        tensor = np.zeros(r.y.shape + r.x.shape + (len(indices),), dtype=np.float64)
        for i, pq in indices:
            tensor[:, :, i] = np.multiply.outer(yv[:, pq.y], xv[:, pq.x])
        return tensor.reshape(y.shape[0] * x.shape[0], len(indices))

    def _make_general_matrix(self, *, x: np.ndarray, y: np.ndarray, triangular: bool) -> np.ndarray:
        r = self._remap(
            x=np.asarray(x, dtype=np.float64, copy=True).ravel(),
            y=np.asarray(y, dtype=np.float64, copy=True).ravel(),
        )
        yv = np.polynomial.chebyshev.chebvander(r.y, self.y_order)
        xv = np.polynomial.chebyshev.chebvander(r.x, self.x_order)
        indices = list(self._packing_indices(triangular))
        matrix = np.zeros(r.y.shape + (len(indices),), dtype=np.float64)
        for i, pq in indices:
            matrix[:, i] = yv[:, pq.y] * xv[:, pq.x]
        return matrix


class ChebyshevFieldSerializationModel(ArchiveTree):
    """Serialization model for `ChebyshevField`."""

    bounds: SerializableBounds = pydantic.Field(
        description=(
            "The region where this field can be evaluated. "
            "The bbox of this region is grown by half a pixel on all sides and then used to remap "
            "coordinates to [-1, 1]x[-1, 1], which is the natural domain of a 2-d Chebyshev polynomial."
        )
    )

    coefficients: InlineArray = pydantic.Field(
        description=(
            "Coefficients for a 2-d Chebyshev polynomial of the first kind, as a 2-d matrix in which "
            "element [p, q] corresponds to the coefficient of T_p(y) T_q(x)."
        )
    )

    unit: Unit | None = pydantic.Field(default=None, description="Units of the field.")

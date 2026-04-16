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

__all__ = ("SplineField", "SplineFieldSerializationModel")

from typing import TYPE_CHECKING, Any, final

import astropy.units
import numpy as np
import pydantic
from scipy.interpolate import Akima1DInterpolator

from .._concrete_bounds import SerializableBounds, deserialize_bounds
from .._geom import Bounds, Box
from .._image import Image
from ..serialization import ArchiveTree, ArrayReferenceModel, InlineArray, InputArchive, OutputArchive, Unit
from ._base import BaseField

if TYPE_CHECKING:
    try:
        from lsst.afw.math import BackgroundMI as LegacyBackground
    except ImportError:
        type LegacyBackground = Any  # type: ignore[no-redef]


@final
class SplineField(BaseField):
    """A 2-d Akima spline interpolation of data on a regular grid.

    Parameters
    ----------
    bounds
        The region where this field can be evaluated.
    data
        The data points to be interpolated.  Missing values (indicated by NaN)
        are allowed.  Will be set to read-only in place.
    y
        Coordinates for the first dimension of ``data``.  Will be set to
        read-only in place.
    x
        Coordinates for the second dimension of ``data``.  Will be set to
        read-only in place.
    unit
        Units of the field.

    Notes
    -----
    This field is much faster to evaluate on a grid via `render` than at
    arbitrary points via the function-call operator.
    """

    def __init__(
        self,
        bounds: Bounds,
        data: np.ndarray,
        *,
        y: np.ndarray,
        x: np.ndarray,
        unit: astropy.units.UnitBase | None = None,
    ):
        if isinstance(data, astropy.units.Quantity):
            if unit is not None:
                raise TypeError("If 'data' is a Quantity, 'unit' cannot be provided separately.")
            unit = data.unit
            data = data.to_value()
        if data.ndim != 2:
            raise ValueError("'data' must be 2-d.")
        if y.ndim != 1:
            raise ValueError("'y' must be 1-d.")
        if x.ndim != 1:
            raise ValueError("'x' must be 1-d.")
        if data.shape != y.shape + x.shape:
            raise ValueError(
                f"Shape of 2-d 'data' {data.shape} does not match "
                f"expected 1-d 'y' {y.shape} and/or 'x' {x.shape}."
            )
        self._bounds = bounds
        self._data = data
        self._data.flags.writeable = False
        self._x = x
        self._x.flags.writeable = False
        self._y = y
        self._y.flags.writeable = False
        self._unit = unit

    @property
    def bounds(self) -> Bounds:
        return self._bounds

    @property
    def unit(self) -> astropy.units.UnitBase | None:
        return self._unit

    @property
    def data(self) -> np.ndarray:
        """The data points to be interpolated (`numpy.ndarray`).

        May have missing values indicated by NaNs.
        """
        return self._data

    @property
    def x(self) -> np.ndarray:
        """Coordinates for the second dimension of `data` (`numpy.ndarray`)."""
        return self._x

    @property
    def y(self) -> np.ndarray:
        """Coordinates for the first dimension of `data` (`numpy.ndarray`)."""
        return self._y

    def evaluate(
        self, *, x: np.ndarray, y: np.ndarray, quantity: bool = False
    ) -> np.ndarray | astropy.units.Quantity:
        y, x = np.broadcast_arrays(y, x)
        xg = self._x
        y_render = np.zeros(xg.shape + y.shape, dtype=np.float64)
        mask = np.zeros(xg.size, dtype=bool)
        for j in range(xg.size):
            if (y_interpolator := self._make_y_interpolator(j)) is not None:
                y_render[j, ...] = y_interpolator(y)
                mask[j] = True
        if not np.all(mask):
            y_render = y_render[mask, ...]
            xg = xg[mask]
        result = np.zeros(y.shape, dtype=np.float64)
        # There doesn't seem to be a way to avoid looping in Python here;
        # maybe someday we'll push this down to a compiled language.
        for i, xv in np.ndenumerate(x):
            if (x_interpolator := self._make_1d_interpolator(xg, y_render[:, *i])) is None:
                raise ValueError("No valid data points.")
            v = x_interpolator(xv)
            result[*i] = v
        if quantity:
            return astropy.units.Quantity(result, self._unit)
        return result

    def render(self, bbox: Box | None = None, *, dtype: np.typing.DTypeLike | None = None) -> Image:
        if bbox is None:
            bbox = self.bounds.bbox
        xg = self._x
        y_render = np.zeros((xg.size, bbox.y.size), dtype=dtype)
        mask = np.zeros(xg.size, dtype=bool)
        for j in range(xg.size):  # we have to loop, but only over bins, not evaluation points.
            if (y_interpolator := self._make_y_interpolator(j)) is not None:
                y_render[j, :] = y_interpolator(bbox.y.arange)
                mask[j] = True
        if not np.all(mask):
            y_render = y_render[mask, :]
            xg = xg[mask]
        if (x_interpolator := self._make_1d_interpolator(xg, y_render)) is None:
            raise ValueError("No valid data points.")
        rendered_array = x_interpolator(bbox.x.arange)
        return Image(rendered_array.transpose().copy(), bbox=bbox, unit=self._unit, dtype=dtype)

    def multiply_constant(
        self, factor: float | astropy.units.Quantity | astropy.units.UnitBase
    ) -> SplineField:
        factor, unit = self._handle_factor_units(factor)
        return SplineField(self._bounds, self._data * factor, y=self._y, x=self._x, unit=unit)

    def serialize(self, archive: OutputArchive[Any]) -> SplineFieldSerializationModel:
        """Serialize the spline field to an output archive."""
        return SplineFieldSerializationModel(
            bounds=self.bounds.serialize(),
            data=archive.add_array(self._data, name="data"),
            y=self._y,
            x=self._x,
            unit=self._unit,
        )

    @staticmethod
    def deserialize(model: SplineFieldSerializationModel, archive: InputArchive[Any]) -> SplineField:
        """Deserialize the spline field from an input archive."""
        return SplineField(
            deserialize_bounds(model.bounds),
            archive.get_array(model.data),
            y=model.y,
            x=model.x,
            unit=model.unit,
        )

    @staticmethod
    def _get_archive_tree_type(
        pointer_type: type[Any],
    ) -> type[SplineFieldSerializationModel]:
        """Return the serialization model type for this object for an archive
        type that uses the given pointer type.
        """
        return SplineFieldSerializationModel

    @staticmethod
    def from_legacy_background(
        legacy_background: LegacyBackground,
        unit: astropy.units.UnitBase | None = None,
    ) -> SplineField:
        """Convert from a legacy `lsst.afw.math.BackgroundMI` instance.

        Notes
        -----
        `SplineField.render` and the `lsst.afw` background interpolator both
        use Akima splines, but with slightly different boundary conditions.
        They should produce equivalent to single-precision round-off error
        when evaluated within the region enclosed by bin centers (i.e. where
        no extrapolation is necessary) and when there are five or more
        points to be interpolated in each row and column.
        """
        from lsst.afw.math import ApproximateControl, Interpolate

        bg_control = legacy_background.getBackgroundControl()
        approx_control = bg_control.getApproximateControl()
        stats_image = legacy_background.getStatsImage()
        if approx_control.getStyle() != ApproximateControl.UNKNOWN:
            raise TypeError("Legacy background uses Chebyshev approximation, not splines.")
        if bg_control.getInterpStyle() != Interpolate.AKIMA_SPLINE:
            raise TypeError("Legacy background does not use Akima spline interpolation.")
        x = legacy_background.getBinCentersX()
        y = legacy_background.getBinCentersY()
        return SplineField(
            Box.from_legacy(legacy_background.getImageBBox()), stats_image.image.array, x=x, y=y, unit=unit
        )

    def _make_1d_interpolator(self, loc: np.ndarray, val: np.ndarray) -> Akima1DInterpolator | None:
        match len(loc):
            case 0:
                return None
            case 1:
                # SciPy can handle only two points by downgrading to linear
                # interpolation, but it raises if given only one.  Mock up
                # two for the nearest-neighbor fallback.
                return Akima1DInterpolator(np.array([loc[0], loc[0]]), np.array([val[0], val[0]]))
            case _:
                return Akima1DInterpolator(loc, val, extrapolate=True)

    def _make_y_interpolator(self, j: int) -> Akima1DInterpolator | None:
        y = self._y
        z = self._data[:, j]
        mask = np.isfinite(z)
        if not np.all(mask):
            y = y[mask]
            z = z[mask]
        del mask
        return self._make_1d_interpolator(y, z)


class SplineFieldSerializationModel(ArchiveTree):
    """Serialization model for `SplineField`."""

    bounds: SerializableBounds = pydantic.Field(description=("The region where this field can be evaluated."))

    data: ArrayReferenceModel = pydantic.Field(
        description="2-d data to interpolate.  NaNs indicate missing values."
    )

    y: InlineArray = pydantic.Field(description="Row positions of the data points.")

    x: InlineArray = pydantic.Field(description="Column positions of the data points.")

    unit: Unit | None = pydantic.Field(default=None, description="Units of the field.")

    def finish_deserialize(self, archive: InputArchive) -> SplineField:
        return SplineField.deserialize(self, archive)

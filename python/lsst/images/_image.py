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

__all__ = ("Image", "ImageModel")

from collections.abc import Sequence
from types import EllipsisType
from typing import Any, final

import astropy.io.fits
import astropy.units
import numpy as np
import numpy.typing as npt
import pydantic

from . import asdf_utils
from ._geom import Box, Interval


@final
class Image:
    """A 2-d array that may be augmented with units and a nonzero origin.

    Parameters
    ----------
    array_or_fill, optional
        Array or fill value for the image.  If a fill value, ``bbox`` or
        ``shape`` must be provided.
    bbox, optional
        Bounding box for the image.
    start, optional
        Logical coordinates of the first pixel in the array.  Ignored if
        ``bbox`` is provided.  Defaults to zeros.
    shape, optional
        Dimensions of the array.  Only needed if ``array_or_fill` is not an
        array and ``bbox`` is not provided.
    unit, optional
        Units for the image's pixel values.
    dtype, optional
        Pixel data type override.

    Notes
    -----
    Indexing the `array` attribute of an `Image` does not take into account its
    `start` offset, but accessing a subimage by indexing an `Image` with a
    `Box` does, and the `bbox` of the subimage is set to match its location
    within the original image.

    Indexed assignment to a subimage requires consistency between the
    coordinate systems and units of both operands, but it will automatically
    select a subimage of the right-hand side and convert compatible units when
    possible.  In other words::

        a[box] = b

    is a shortcut for

        a[box].quantity = b[box].quantity

    An ellipsis (``...``) can be used instead of a `Box` to assign to the full
    image.
    """

    def __init__(
        self,
        array_or_fill: np.ndarray | int | float = 0,
        /,
        *,
        bbox: Box | None = None,
        start: Sequence[int] | None = None,
        shape: Sequence[int] | None = None,
        unit: astropy.units.Unit | None = None,
        dtype: npt.DTypeLike | None = None,
    ):
        if isinstance(array_or_fill, np.ndarray):
            if dtype is not None:
                array = np.array(array_or_fill, dtype=dtype)
            else:
                array = array_or_fill
            if bbox is None:
                bbox = Box.from_shape(array.shape, start=start)
            elif bbox.shape != array.shape:
                raise ValueError(
                    f"Explicit bbox shape {bbox.shape} does not match array with shape {array.shape}."
                )
            if shape is not None and shape != array.shape:
                raise ValueError(f"Explicit shape {shape} does not match array with shape {array.shape}.")

        else:
            if bbox is None:
                if shape is None:
                    raise TypeError("No bbox, shape, or array provided.")
                bbox = Box.from_shape(shape, start=start)
            elif shape is not None and shape != bbox.shape:
                raise ValueError(f"Explicit shape {shape} does not match bbox shape {bbox.shape}.")
            array = np.full(bbox.shape, array_or_fill, dtype=dtype)
        self._array = array
        self._bbox = bbox
        self._unit = unit

    @property
    def array(self) -> np.ndarray:
        """The low-level array.

        Assigning to this attribute modifies the existing array in place; the
        bounding box and underlying data pointer are never changed.
        """
        return self._array

    @array.setter
    def array(self, value: np.ndarray | int | float) -> None:
        self._array[...] = value

    @property
    def quantity(self) -> astropy.units.Quantity:
        """The low-level array with units.

        Assigning to this attribute modifies the existing array in place; the
        bounding box and underlying data pointer are never changed.
        """
        return astropy.units.Quantity(self._array, self._unit, copy=False)

    @quantity.setter
    def quantity(self, value: astropy.units.Quantity) -> None:
        self.quantity[...] = value

    @property
    def unit(self) -> astropy.units.Unit | None:
        """Units for the image's pixel values."""
        return self._unit

    @property
    def bbox(self) -> Box:
        """Bounding box for the image."""
        return self._bbox

    def __getitem__(self, bbox: Box | EllipsisType) -> Image:
        indices: EllipsisType | tuple[slice, ...]
        if bbox is ...:
            indices = ...
            bbox = self._bbox
        else:
            indices = bbox.slice_within(self._bbox)
        return Image(self._array[indices], bbox=bbox, unit=self._unit)

    def __setitem__(self, bbox: Box | EllipsisType, value: Image) -> None:
        if bbox is ...:
            bbox = self._bbox
        if value._bbox != bbox:
            value = value[bbox]
        # Use the quantity property to handle unit conversions and checks.
        self.quantity[bbox.slice_within(self._bbox)] = value.quantity

    def __str__(self) -> str:
        return f"Image({self.bbox!s}, {self.array.dtype.type.__name__})"

    def __repr__(self) -> str:
        return f"Image(..., bbox={self.bbox!r}, dtype={self.array.dtype!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Image):
            return NotImplemented
        return (
            self._bbox == other._bbox
            and self._unit == other._unit
            and np.array_equal(self._array, other._array, equal_nan=True)
        )

    def copy(self) -> Image:
        """Deep-copy the image."""
        return Image(self._array.copy(), bbox=self._bbox, unit=self._unit)

    @classmethod
    def from_legacy(cls, legacy: Any, unit: astropy.units.Unit | None = None) -> Image:
        """Convert from an `lsst.afw.image.Image` instance."""
        return cls(legacy.array, start=(legacy.getY0(), legacy.getX0()), unit=unit)

    def to_legacy(self) -> Any:
        """Convert to an `lsst.afw.image.Image` instance."""
        import lsst.afw.image
        import lsst.geom

        result = lsst.afw.image.Image(self._array, dtype=self._array.dtype)
        result.setXY0(lsst.geom.Point2I(self._bbox.x.min, self._bbox.y.min))
        return result

    @classmethod
    def read_legacy(cls, hdu: astropy.io.fits.ImageHDU | astropy.io.fits.CompImageHDU) -> Image:
        """Read a FITS file written by `lsst.afw.image.Image.writeFits`.

        Parameters
        ----------
        hdu
            An astropy image object.

        Returns
        -------
        image
            A new `Image` object.
        """
        unit: astropy.units.BaseUnit | None = None
        if (fits_unit := hdu.header.pop("BUNIT", None)) is not None:
            unit = astropy.units.Unit(fits_unit, format="fits")
        dx: int = hdu.header.pop("LTV1")
        dy: int = hdu.header.pop("LTV2")
        start = (-dy, -dx)
        image = Image(hdu.data, start=start, unit=unit)
        return image


class ImageModel(pydantic.BaseModel):
    """Pydantic model used to represent the serialized form of an `Image`."""

    data: asdf_utils.ArrayReferenceQuantityModel | asdf_utils.ArrayReferenceModel
    start: list[int]

    @property
    def bbox(self) -> Box:
        """The bounding box of the image."""
        if isinstance(self.data, asdf_utils.ArrayReferenceQuantityModel):
            shape = self.data.value.shape
        else:
            shape = self.data.shape
        return Box(*[Interval.factory[begin : begin + size] for begin, size in zip(self.start, shape)])

    @classmethod
    def pack(
        cls, array_model: asdf_utils.ArrayReferenceModel, start: Sequence[int], unit: asdf_utils.Unit | None
    ) -> ImageModel:
        """Construct an `ImageReference` from the components of a serialized
        image.

        Parameters
        ----------
        array_model
            Serialized form of the underlying array.
        start
            Logical coordinates of the first pixel in the array.
        unit, optional
            Units for the image's pixel values.
        """
        if unit is None:
            return cls.model_construct(data=array_model, start=list(start))
        return cls.model_construct(
            data=asdf_utils.ArrayReferenceQuantityModel.model_construct(value=array_model, unit=unit),
            start=list(start),
        )

    def unpack(self) -> tuple[asdf_utils.ArrayReferenceModel, asdf_utils.Unit | None]:
        """Return the components of a serialized image from this model.

        Returns
        -------
        array_model
            Serialized form of the underlying array.
        unit
            Units for the image's pixel values.

        Notes
        -----
        The ``start`` attribute is not included in the results because it is
        directly accessible, rather than possibly nested under `data` as with
        the other attributes.
        """
        if isinstance(self.data, asdf_utils.ArrayReferenceQuantityModel):
            return self.data.value, self.data.unit
        return self.data, None

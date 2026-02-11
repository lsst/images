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

__all__ = ("Image", "ImageSerializationModel")

from collections.abc import Callable, Sequence
from types import EllipsisType
from typing import Any, final

import astropy.io.fits
import astropy.units
import astropy.wcs
import numpy as np
import numpy.typing as npt
import pydantic

from . import fits
from ._geom import Box
from ._transforms import Projection, ProjectionAstropyView, ProjectionSerializationModel
from .serialization import (
    ArrayReferenceModel,
    ArrayReferenceQuantityModel,
    InputArchive,
    OutputArchive,
    no_header_updates,
)
from .utils import is_none


@final
class Image:
    """A 2-d array that may be augmented with units and a nonzero origin.

    Parameters
    ----------
    array_or_fill
        Array or fill value for the image.  If a fill value, ``bbox`` or
        ``shape`` must be provided.
    bbox
        Bounding box for the image.
    start
        Logical coordinates of the first pixel in the array, ordered ``y``,
        ``x`` (unless an `XY` instance is passed).  Ignored if
        ``bbox`` is provided.  Defaults to zeros.
    shape
        Leading dimensions of the array, ordered ``y``, ``x`` (unless an `XY`
        instance is passed).   Only needed if ``array_or_fill`` is not an
        array and ``bbox`` is not provided.  Like the bbox, this does not
        include the last dimension of the array.
    dtype
        Pixel data type override.
    unit
        Units for the image's pixel values.
    projection
        Projection that maps the pixel grid to the sky.

    Notes
    -----
    Indexing the `array` attribute of an `Image` does not take into account its
    ``start`` offset, but accessing a subimage by indexing an `Image` with a
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
        dtype: npt.DTypeLike | None = None,
        unit: astropy.units.UnitBase | None = None,
        projection: Projection[Any] | None = None,
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
        self._array: np.ndarray = array
        self._bbox: Box = bbox
        self._unit = unit
        self._projection = projection

    @property
    def array(self) -> np.ndarray:
        """The low-level array (`numpy.ndarray`).

        Assigning to this attribute modifies the existing array in place; the
        bounding box and underlying data pointer are never changed.
        """
        return self._array

    @array.setter
    def array(self, value: np.ndarray | int | float) -> None:
        self._array[...] = value

    @property
    def quantity(self) -> astropy.units.Quantity:
        """The low-level array with units (`astropy.units.Quantity`).

        Assigning to this attribute modifies the existing array in place; the
        bounding box and underlying data pointer are never changed.
        """
        return astropy.units.Quantity(self._array, self._unit, copy=False)

    @quantity.setter
    def quantity(self, value: astropy.units.Quantity) -> None:
        self.quantity[...] = value

    @property
    def bbox(self) -> Box:
        """Bounding box for the image (`Box`)."""
        return self._bbox

    @property
    def unit(self) -> astropy.units.UnitBase | None:
        """Units for the image's pixel values (`astropy.units.Unit` or
        `None`).
        """
        return self._unit

    @property
    def projection(self) -> Projection[Any] | None:
        """The projection that maps this image's pixel grid to the sky
        (`Projection` | `None`).

        Notes
        -----
        The pixel coordinates used by this projection account for the bounding
        box ``start``; they are not just array indices.
        """
        return self._projection

    @property
    def astropy_wcs(self) -> ProjectionAstropyView | None:
        """An Astropy WCS for this image's pixel array.

        Notes
        -----
        As expected for Astropy WCS objects, this defines pixel coordinates
        such that the first row and column in `array` are ``(0, 0)``, not
        ``bbox.start``, as is the case for `projection`.

        This object satisfies the `astropy.wcs.wcsapi.BaseHighLevelWCS` and
        `astropy.wcs.wcsapi.BaseLowLevelWCS` interfaces, but it is not an
        `astropy.wcs.WCS` (use `fits_wcs` for that).
        """
        return self._projection.as_astropy(self.bbox) if self._projection is not None else None

    @property
    def fits_wcs(self) -> astropy.wcs.WCS | None:
        """An Astropy FITS WCS for this image's pixel array.

        Notes
        -----
        As expected for Astropy WCS objects, this defines pixel coordinates
        such that the first row and column in `array` are ``(0, 0)``, not
        ``bbox.start``, as is the case for `projection`.

        This may be an approximation or absent if `projection` is not
        naturally representable as a FITS WCS.
        """
        return self._projection.as_fits_wcs(self.bbox) if self._projection is not None else None

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

    def copy(
        self,
        *,
        unit: astropy.units.UnitBase | None | EllipsisType = ...,
        projection: Projection | None | EllipsisType = ...,
        start: Sequence[int] | EllipsisType = ...,
    ) -> Image:
        """Deep-copy the image, with optional updates."""
        if unit is ...:
            unit = self._unit
        if projection is ...:
            projection = self._projection
        if start is ...:
            start = self._bbox.start
        return Image(self._array.copy(), start=start, unit=unit, projection=projection)

    def view(
        self,
        *,
        unit: astropy.units.UnitBase | None | EllipsisType = ...,
        projection: Projection | None | EllipsisType = ...,
        start: Sequence[int] | EllipsisType = ...,
    ) -> Image:
        """Make a view of the image, with optional updates."""
        if unit is ...:
            unit = self._unit
        if projection is ...:
            projection = self._projection
        if start is ...:
            start = self._bbox.start
        return Image(self._array, start=start, unit=unit, projection=projection)

    def serialize[P: pydantic.BaseModel](
        self,
        archive: OutputArchive[P],
        *,
        update_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
        save_projection: bool = True,
        add_offset_wcs: str | None = "A",
    ) -> ImageSerializationModel[P]:
        """Serialize the image to an output archive.

        Parameters
        ----------
        archive
            `~serialization.OutputArchive` instance to write to.
        update_header
            A callback that will be given the FITS header for the HDU
            containing this image in order to add keys to it.  This callback
            may be provided but will not be called if the output format is not
            FITS.
        save_projection
            If `True`, save the `Projection` attached to the image, if there
            is one.
        add_offset_wcs
            A FITS WCS single-character suffix to use when adding a linear
            WCS that maps the FITS array to the logical pixel coordinates
            defined by ``bbox.start``.  Set to `None` to not write this WCS.

        Returns
        -------
        ImageSerializationModel
            A Pydantic model representation of the image, holding references
            to data stored in the archive.
        """
        if save_projection and add_offset_wcs == "":
            raise TypeError("save_projection=True is not compatible with add_offset_wcs=''.")

        def _update_header(header: astropy.io.fits.Header) -> None:
            update_header(header)
            if self.unit is not None:
                header["BUNIT"] = self.unit.to_string(format="fits")
            if self.projection is not None:
                fits_wcs = self.projection.as_fits_wcs(self.bbox)
                if fits_wcs:
                    header.update(fits_wcs.to_header(relax=True))
            if add_offset_wcs is not None:
                fits.add_offset_wcs(header, x=self.bbox.x.start, y=self.bbox.y.start, key=add_offset_wcs)

        ref = archive.add_array(self.array, update_header=_update_header)
        serialized_projection: ProjectionSerializationModel[P] | None = None
        if save_projection and self.projection is not None:
            serialized_projection = archive.serialize_direct("projection", self.projection.serialize)
        if self.unit is None:
            return ImageSerializationModel.model_construct(
                data=ref, start=list(self.bbox.start), projection=serialized_projection
            )
        else:
            return ImageSerializationModel.model_construct(
                data=ArrayReferenceQuantityModel.model_construct(value=ref, unit=self.unit),
                start=list(self.bbox.start),
                projection=serialized_projection,
            )

    @classmethod
    def deserialize(
        cls,
        model: ImageSerializationModel[Any],
        archive: InputArchive[Any],
        *,
        bbox: Box | None = None,
        strip_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
    ) -> Image:
        """Deserialize an image from an input archive.

        Parameters
        ----------
        model
            A Pydantic model representation of the image, holding references
            to data stored in the archive.
        archive
            `~serialization.InputArchive` instance to read from.
        bbox
            Bounding box of a subimage to read instead.
        strip_header
            A callable that strips out any FITS header cards added by the
            ``update_header`` argument in the corresponding call to
            `serialize`.
        """
        ref: ArrayReferenceModel
        unit: astropy.units.UnitBase | None = None
        if isinstance(model.data, ArrayReferenceQuantityModel):
            ref = model.data.value
            unit = model.data.unit
        else:
            ref = model.data

        def _strip_header(header: astropy.io.fits.Header) -> None:
            if unit is not None:
                header.pop("BUNIT", None)
            fits.strip_wcs_cards(header)
            strip_header(header)

        slices = bbox.slice_within(model.bbox) if bbox is not None else ...
        array = archive.get_array(ref, strip_header=_strip_header, slices=slices)
        projection = (
            Projection.deserialize(model.projection, archive) if model.projection is not None else None
        )
        return cls(array, start=model.start if bbox is None else bbox.start, unit=unit, projection=projection)

    @classmethod
    def from_legacy(cls, legacy: Any, unit: astropy.units.Unit | None = None) -> Image:
        """Convert from an `lsst.afw.image.Image` instance.

        Parameters
        ----------
        legacy
            An `lsst.afw.image.Image` instance.
        unit
            Units of the image.
        """
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
            An Astropy image HDU.
        """
        unit: astropy.units.UnitBase | None = None
        if (fits_unit := hdu.header.pop("BUNIT", None)) is not None:
            unit = astropy.units.Unit(fits_unit, format="fits")
        dx: int = hdu.header.pop("LTV1")
        dy: int = hdu.header.pop("LTV2")
        start = (-dy, -dx)
        image = Image(hdu.data, start=start, unit=unit)
        return image


class ImageSerializationModel[P: pydantic.BaseModel](pydantic.BaseModel):
    """Pydantic model used to represent the serialized form of an `.Image`."""

    data: ArrayReferenceQuantityModel | ArrayReferenceModel = pydantic.Field(
        description="Reference to pixel data."
    )
    start: list[int] = pydantic.Field(
        description="Coordinate of the first pixels in the array, ordered (y, x)."
    )
    projection: ProjectionSerializationModel[P] | None = pydantic.Field(
        default=None,
        exclude_if=is_none,
        description="Projection that maps the logical pixel grid onto the sky.",
    )

    @property
    def bbox(self) -> Box:
        """The bounding box of the image."""
        if isinstance(self.data, ArrayReferenceQuantityModel):
            shape = self.data.value.shape
        else:
            shape = self.data.shape
        return Box.from_shape(shape, self.start)

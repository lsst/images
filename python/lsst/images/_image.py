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
from contextlib import ExitStack
from types import EllipsisType
from typing import TYPE_CHECKING, Any, ClassVar, final

import astropy.io.fits
import astropy.units
import astropy.wcs
import numpy as np
import numpy.typing as npt
import pydantic

from lsst.resources import ResourcePath, ResourcePathExpression

from . import fits
from ._generalized_image import GeneralizedImage
from ._geom import YX, Box
from ._transforms import Frame, SkyProjection, SkyProjectionSerializationModel
from .serialization import (
    ArchiveTree,
    ArrayReferenceModel,
    ArrayReferenceQuantityModel,
    InlineArrayModel,
    InlineArrayQuantityModel,
    InputArchive,
    InvalidParameterError,
    MetadataValue,
    OutputArchive,
    no_header_updates,
)
from .utils import is_none

if TYPE_CHECKING:
    try:
        from lsst.afw.image import Image as LegacyImage
    except ImportError:
        type LegacyImage = Any  # type: ignore[no-redef]


@final
class Image(GeneralizedImage):
    """A 2-d array that may be augmented with units and a nonzero origin.

    Parameters
    ----------
    array_or_fill
        Array or fill value for the image.  If a fill value, ``bbox`` or
        ``shape`` must be provided.
    bbox
        Bounding box for the image.
    yx0
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
    sky_projection
        Projection that maps the pixel grid to the sky.
    metadata
        Arbitrary flexible metadata to associate with the image.

    Notes
    -----
    Indexing the `array` attribute of an `Image` does not take into account its
    ``yx0`` offset, but accessing a subimage by indexing an `Image` with a
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
        yx0: Sequence[int] | None = None,
        shape: Sequence[int] | None = None,
        dtype: npt.DTypeLike | None = None,
        unit: astropy.units.UnitBase | None = None,
        sky_projection: SkyProjection[Any] | None = None,
        metadata: dict[str, MetadataValue] | None = None,
    ) -> None:
        super().__init__(metadata)
        if isinstance(array_or_fill, np.ndarray):
            if dtype is not None:
                array = np.array(array_or_fill, dtype=dtype, copy=None)
            else:
                array = array_or_fill
            if bbox is None:
                bbox = Box.from_shape(array.shape, start=yx0)
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
                bbox = Box.from_shape(shape, start=yx0)
            elif shape is not None and shape != bbox.shape:
                raise ValueError(f"Explicit shape {shape} does not match bbox shape {bbox.shape}.")
            array = np.full(bbox.shape, array_or_fill, dtype=dtype)
        self._array: np.ndarray = array
        self._bbox: Box = bbox
        self._unit = unit
        self._sky_projection = sky_projection

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
    def sky_projection(self) -> SkyProjection[Any] | None:
        """The projection that maps this image's pixel grid to the sky
        (`SkyProjection` | `None`).

        Notes
        -----
        The pixel coordinates used by this projection account for the bounding
        box ``start``; they are not just array indices.
        """
        return self._sky_projection

    def __getitem__(self, bbox: Box | EllipsisType) -> Image:
        if bbox is ...:
            return self
        super().__getitem__(bbox)
        indices = bbox.slice_within(self._bbox)
        return self._transfer_metadata(
            Image(self._array[indices], bbox=bbox, unit=self._unit, sky_projection=self._sky_projection),
            bbox=bbox,
        )

    def __setitem__(self, bbox: Box | EllipsisType, value: Image) -> None:
        self[bbox].quantity[...] = value.quantity

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
        return self._transfer_metadata(
            Image(self._array.copy(), bbox=self._bbox, unit=self._unit, sky_projection=self._sky_projection),
            copy=True,
        )

    def view(
        self,
        *,
        unit: astropy.units.UnitBase | None | EllipsisType = ...,
        sky_projection: SkyProjection | None | EllipsisType = ...,
        yx0: Sequence[int] | EllipsisType = ...,
    ) -> Image:
        """Make a view of the image, with optional updates."""
        if unit is ...:
            unit = self._unit
        if sky_projection is ...:
            sky_projection = self._sky_projection
        if yx0 is ...:
            yx0 = self._bbox.start
        return self._transfer_metadata(Image(self._array, yx0=yx0, unit=unit, sky_projection=sky_projection))

    def serialize[P: pydantic.BaseModel](
        self,
        archive: OutputArchive[P],
        *,
        update_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
        save_projection: bool = True,
        add_offset_wcs: str | None = "A",
        tile_shape: tuple[int, ...] | None = None,
        options_name: str | None = None,
    ) -> ImageSerializationModel[P]:
        """Serialize the image to an output archive.

        Parameters
        ----------
        archive
            Archive to write to.
        update_header
            A callback that will be given the FITS header for the HDU
            containing this image in order to add keys to it.  This callback
            may be provided but will not be called if the output format is not
            FITS.
        save_projection
            If `True`, save the `SkyProjection` attached to the image, if there
            is one.  This does not affect whether a FITS WCS corresponding to
            the projection is written (it always is, if available, and if
            ``add_offset_wcs`` is not ``" "``).
        add_offset_wcs
            A FITS WCS single-character suffix to use when adding a linear
            WCS that maps the FITS array to the logical pixel coordinates
            defined by ``bbox.start``.  Set to `None` to not write this WCS.
            If this is set to ``" "``, it will prevent the `SkyProjection` from
            being saved as a FITS WCS.
        tile_shape
            The recommended shape of each tile, if the archive will save
            the array in distinct tiles for faster subarray retrieval.
            This is a hint; archives are not required to use this value.
        options_name
            Use this name to look up archive options.
        """

        def _update_header(header: astropy.io.fits.Header) -> None:
            update_header(header)
            if self.unit is not None:
                try:
                    header["BUNIT"] = self.unit.to_string(format="fits")
                except ValueError:
                    # Units not supported by FITS; write it anyway because
                    # the accepted units are just a recommendation in the
                    # standard.
                    header["BUNIT"] = self.unit.to_string()
            if self.sky_projection is not None and add_offset_wcs != " ":
                if self.fits_wcs:
                    header.update(self.fits_wcs.to_header(relax=True))
            if add_offset_wcs is not None:
                fits.add_offset_wcs(header, x=self.bbox.x.start, y=self.bbox.y.start, key=add_offset_wcs)

        array_model = archive.add_array(
            self.array, update_header=_update_header, tile_shape=tile_shape, options_name=options_name
        )
        serialized_projection: SkyProjectionSerializationModel[P] | None = None
        if save_projection and self.sky_projection is not None:
            serialized_projection = archive.serialize_direct("sky_projection", self.sky_projection.serialize)
        data = array_model if self.unit is None else array_model.with_units(self.unit)
        return ImageSerializationModel.model_construct(
            data=data,
            yx0=list(self.bbox.start),
            sky_projection=serialized_projection,
            metadata=self.metadata,
        )

    @staticmethod
    def _get_archive_tree_type[P: pydantic.BaseModel](
        pointer_type: type[P],
    ) -> type[ImageSerializationModel[P]]:
        """Return the serialization model type for this object for an archive
        type that uses the given pointer type.
        """
        return ImageSerializationModel[pointer_type]  # type: ignore

    _archive_default_name: ClassVar[str] = "image"
    """The name this object should be serialized with when written as the
    top-level object.
    """

    @staticmethod
    def from_legacy(legacy: LegacyImage, unit: astropy.units.UnitBase | None = None) -> Image:
        """Convert from an `lsst.afw.image.Image` instance.

        Parameters
        ----------
        legacy
            An `lsst.afw.image.Image` instance that will share pixel data with
           the returned object.
        unit
            Units of the image.
        """
        return Image(legacy.array, yx0=YX(y=legacy.getY0(), x=legacy.getX0()), unit=unit)

    def to_legacy(self, *, copy: bool | None = None) -> LegacyImage:
        """Convert to an `lsst.afw.image.Image` instance.

        Parameters
        ----------
        copy
            If `True`, always copy the pixel data.  If `False`, return a view,
            and raise `TypeError` if the pixel data is read-only (this is not
            supported by afw).  If `None`, onyl if the pixel data is
            read-only.
        """
        import lsst.afw.image
        import lsst.geom

        array = self._array
        if copy:
            array = array.copy()
        elif not self._array.flags.writeable:
            if copy is None:
                array = array.copy()
            else:
                raise TypeError("Cannot create a legacy lsst.afw.image.Image view into a read-only array.")

        return lsst.afw.image.Image(
            array,
            deep=False,
            dtype=array.dtype.type,
            xy0=lsst.geom.Point2I(self._bbox.x.min, self._bbox.y.min),
        )

    @staticmethod
    def read_legacy(
        uri: ResourcePathExpression,
        *,
        preserve_quantization: bool = False,
        ext: str | int = 1,
        fits_wcs_frame: Frame | None = None,
    ) -> Image:
        """Read a FITS file written by `lsst.afw.image.Image.writeFits`.

        Parameters
        ----------
        uri
            URI or file name.
        preserve_quantization
            If `True`, ensure that writing the image back out again will
            exactly preserve quantization-compressed pixel values.  This causes
            the arrays to be marked as read-only and stores the original binary
            table data for those planes in memory. If the `Image` is copied,
            the precompressed pixel values are not transferred to the copy.
        ext
            Name or index of the FITS HDU to read.
        fits_wcs_frame
            If not `None` and the HDU containing the image has a FITS WCS,
            attach a `SkyProjection` to the returned image by converting that
            WCS.
        """
        opaque_metadata = fits.FitsOpaqueMetadata()
        with ExitStack() as exit_stack:
            fs, fspath = ResourcePath(uri).to_fsspec()
            stream = exit_stack.enter_context(fs.open(fspath))
            hdu_list = exit_stack.enter_context(astropy.io.fits.open(stream))
            opaque_metadata.extract_legacy_primary_header(hdu_list[0].header)
            bintable_hdu: astropy.io.fits.BinTableHDU | None = None
            if preserve_quantization:
                bintable_stream = exit_stack.enter_context(fs.open(fspath))
                bintable_hdu_list = exit_stack.enter_context(
                    astropy.io.fits.open(bintable_stream, disable_image_compression=True)
                )
                bintable_hdu = bintable_hdu_list[ext]
            result = Image._read_legacy_hdu(
                hdu_list[ext], opaque_metadata, preserve_bintable=bintable_hdu, fits_wcs_frame=fits_wcs_frame
            )
            result._opaque_metadata = opaque_metadata
        return result

    @staticmethod
    def _read_legacy_hdu(
        hdu: astropy.io.fits.ImageHDU | astropy.io.fits.CompImageHDU,
        opaque_metadata: fits.FitsOpaqueMetadata,
        *,
        preserve_bintable: astropy.io.fits.BinTableHDU | None,
        fits_wcs_frame: Frame | None = None,
    ) -> Image:
        unit: astropy.units.UnitBase | None = None
        if (fits_unit := hdu.header.pop("BUNIT", None)) is not None:
            try:
                unit = astropy.units.Unit(fits_unit, format="fits")
            except ValueError:
                # Accept non-FITS units by assuming Astropy can still figure
                # them out if we don't specify the format.
                unit = astropy.units.Unit(fits_unit)
            if opaque_metadata.get_instrumental_unit() == astropy.units.electron:
                # Fix incorrect BUNIT='adu' in LSST preliminary_visit_image.
                if unit == astropy.units.adu:
                    unit = astropy.units.electron
                if unit == astropy.units.adu**2:
                    unit = astropy.units.electron**2
        dx: int = hdu.header.pop("LTV1")
        dy: int = hdu.header.pop("LTV2")
        yx0 = YX(y=-dy, x=-dx)
        read_only: bool = False
        if preserve_bintable is not None:
            opaque_metadata.precompressed[hdu.name] = fits.PrecompressedImage.from_bintable(preserve_bintable)
            read_only = True
        sky_projection: SkyProjection | None = None
        if fits_wcs_frame is not None:
            try:
                fits_wcs = astropy.wcs.WCS(hdu.header)
            except KeyError:
                pass
            else:
                sky_projection = SkyProjection.from_fits_wcs(
                    fits_wcs, pixel_frame=fits_wcs_frame, x0=yx0.x, y0=yx0.y
                )
        image = Image(hdu.data, yx0=yx0, unit=unit, sky_projection=sky_projection)
        if read_only:
            image._array.flags["WRITEABLE"] = False
        fits.strip_wcs_cards(hdu.header)
        hdu.header.strip()
        hdu.header.remove("EXTTYPE", ignore_missing=True)
        hdu.header.remove("INHERIT", ignore_missing=True)
        hdu.header.remove("UZSCALE", ignore_missing=True)
        opaque_metadata.add_header(hdu.header)
        return image


class ImageSerializationModel[P: pydantic.BaseModel](ArchiveTree):
    """Pydantic model used to represent the serialized form of an `.Image`."""

    SCHEMA_NAME: ClassVar[str] = "image"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"
    MIN_READ_VERSION: ClassVar[int] = 1
    PUBLIC_TYPE: ClassVar[type] = Image

    data: ArrayReferenceQuantityModel | ArrayReferenceModel | InlineArrayModel | InlineArrayQuantityModel = (
        pydantic.Field(description="Reference to pixel data.")
    )
    yx0: list[int] = pydantic.Field(
        description="Coordinate of the first pixels in the array, ordered (y, x)."
    )
    sky_projection: SkyProjectionSerializationModel[P] | None = pydantic.Field(
        default=None,
        exclude_if=is_none,
        description="Projection that maps the logical pixel grid onto the sky.",
    )

    @property
    def bbox(self) -> Box:
        """The bounding box of the image."""
        match self.data:
            case ArrayReferenceQuantityModel() | InlineArrayQuantityModel():
                shape = self.data.value.shape
            case ArrayReferenceModel() | InlineArrayModel():
                shape = self.data.shape
        return Box.from_shape(shape, self.yx0)

    def deserialize(
        self,
        archive: InputArchive[Any],
        *,
        bbox: Box | None = None,
        strip_header: Callable[[astropy.io.fits.Header], None] = no_header_updates,
        **kwargs: Any,
    ) -> Image:
        """Deserialize an image from an input archive.

        Parameters
        ----------
        archive
            Archive to read from.
        bbox
            Bounding box of a subimage to read instead.
        strip_header
            A callable that strips out any FITS header cards added by the
            ``update_header`` argument in the corresponding call to
            `Image.serialize`.
        **kwargs
            Unsupported keyword arguments are accepted only to provide better
            error messages (raising `serialization.InvalidParameterError`).
        """
        if kwargs:
            raise InvalidParameterError(f"Unrecognized parameters for Image: {set(kwargs.keys())}.")
        array_model: ArrayReferenceModel | InlineArrayModel
        unit: astropy.units.UnitBase | None = None
        if isinstance(self.data, ArrayReferenceQuantityModel | InlineArrayQuantityModel):
            array_model = self.data.value
            unit = self.data.unit
        else:
            array_model = self.data

        def _strip_header(header: astropy.io.fits.Header) -> None:
            if unit is not None:
                header.pop("BUNIT", None)
            fits.strip_wcs_cards(header)
            strip_header(header)

        slices = bbox.slice_within(self.bbox) if bbox is not None else ...
        array = archive.get_array(array_model, strip_header=_strip_header, slices=slices)
        sky_projection = self.sky_projection.deserialize(archive) if self.sky_projection is not None else None
        return Image(
            array,
            yx0=self.yx0 if bbox is None else bbox.start,
            unit=unit,
            sky_projection=sky_projection,
        )._finish_deserialize(self)

    def deserialize_component(self, component: str, archive: InputArchive[Any], **kwargs: Any) -> Any:
        if kwargs:
            raise InvalidParameterError(f"Unsupported parameters for Image components: {set(kwargs.keys())}.")
        return super().deserialize_component(component, archive)

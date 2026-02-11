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

__all__ = ("MaskedImage", "MaskedImageSerializationModel")

import functools
from collections.abc import Sequence
from types import EllipsisType
from typing import Any

import astropy.io.fits
import astropy.units
import astropy.wcs
import numpy as np
import pydantic

from lsst.resources import ResourcePathExpression

from ._geom import Box
from ._image import Image, ImageSerializationModel
from ._mask import Mask, MaskSchema, MaskSerializationModel
from ._transforms import Projection, ProjectionAstropyView, ProjectionSerializationModel
from .fits import (
    ExtensionHDU,
    ExtensionKey,
    FitsCompressionOptions,
    FitsInputArchive,
    FitsOpaqueMetadata,
    FitsOutputArchive,
    PrecompressedImage,
    strip_wcs_cards,
)
from .serialization import InputArchive, OpaqueArchiveMetadata, OutputArchive, TableCellReferenceModel
from .utils import is_none


class MaskedImage:
    """A multi-plane image with data (image), mask, and variance planes.

    Parameters
    ----------
    image
        The main image plane.  If this has a `Projection`, it will be used
        for all planes unless a ``projection`` is passed separately.
    mask
        A bitmask image that annotates the main image plane.  Must have the
        same bounding box as ``image`` if provided.  Any attached projection
        is replaced (possibly by `None`).
    variance
        The per-pixel uncertainty of the main image as an image of variance
        values.  Must have the same bounding box as ``image`` if provided, and
        its units must be the square of ``image.unit`` or `None`.
        Values default to ``1.0``.  Any attached projection is replaced
        (possibly by `None`).
    mask_schema
        Schema for the mask plane.  Must be provided if and only if ``mask`` is
        not provided.
    opaque_metadata
        Opaque metadata obtained from reading this object from storage.  It may
        be provided when writing to storage to propagate that metadata and/or
        preserve file-format-specific options (e.g. compression parameters).
    projection
        Projection that maps the pixel grid to the sky.
    """

    def __init__(
        self,
        image: Image,
        *,
        mask: Mask | None = None,
        variance: Image | None = None,
        mask_schema: MaskSchema | None = None,
        opaque_metadata: OpaqueArchiveMetadata | None = None,
        projection: Projection | None = None,
    ):
        if projection is None:
            projection = image.projection
        else:
            image = image.view(projection=projection)
        if mask is None:
            if mask_schema is None:
                raise TypeError("'mask_schema' must be provided if 'mask' is not.")
            mask = Mask(schema=mask_schema, bbox=image.bbox, projection=projection)
        elif mask_schema is not None:
            raise TypeError("'mask_schema' may not be provided if 'mask' is.")
        else:
            if image.bbox != mask.bbox:
                raise ValueError(f"Image ({image.bbox}) and mask ({mask.bbox}) bboxes do not agree.")
            mask = mask.view(projection=projection)
        if variance is None:
            variance = Image(
                1.0,
                dtype=np.float32,
                bbox=image.bbox,
                unit=None if image.unit is None else image.unit**2,
                projection=projection,
            )
        else:
            if image.bbox != variance.bbox:
                raise ValueError(f"Image ({image.bbox}) and variance ({variance.bbox}) bboxes do not agree.")
            variance = variance.view(projection=projection)
            if image.unit is None:
                if variance.unit is not None:
                    raise ValueError(f"Image has no units but variance does ({variance.unit}).")
            elif variance.unit is None:
                variance = variance.view(unit=image.unit**2)
            elif variance.unit != image.unit**2:
                raise ValueError(
                    f"Variance unit ({variance.unit}) should be the square of the image unit ({image.unit})."
                )
        self._image = image
        self._mask = mask
        self._variance = variance
        self._opaque_metadata = opaque_metadata

    @property
    def image(self) -> Image:
        """The main image plane (`Image`)."""
        return self._image

    @property
    def mask(self) -> Mask:
        """The mask plane (`Mask`)."""
        return self._mask

    @property
    def variance(self) -> Image:
        """The variance plane (`Image`)."""
        return self._variance

    @property
    def bbox(self) -> Box:
        """The bounding box shared by all three image planes (`Box`)."""
        return self._image.bbox

    @property
    def unit(self) -> astropy.units.UnitBase | None:
        """The units of the image plane (`astropy.units.Unit` | `None`)."""
        return self._image.unit

    @property
    def projection(self) -> Projection[Any] | None:
        """The projection that maps the pixel grid to the sky
        (`Projection` | `None`).
        """
        return self._image.projection

    @property
    def astropy_wcs(self) -> ProjectionAstropyView | None:
        """An Astropy WCS for the pixel arrays
        (`ProjectionAstropyView` | `None`).

        Notes
        -----
        As expected for Astropy WCS objects, this defines pixel coordinates
        such that the first row and column in the arrays are ``(0, 0)``, not
        ``bbox.start``, as is the case for `projection`.

        This object satisfies the `astropy.wcs.wcsapi.BaseHighLevelWCS` and
        `astropy.wcs.wcsapi.BaseLowLevelWCS` interfaces, but it is not an
        `astropy.wcs.WCS` (use `fits_wcs` for that).
        """
        return self.projection.as_astropy(self.bbox) if self.projection is not None else None

    @property
    def fits_wcs(self) -> astropy.wcs.WCS | None:
        """An Astropy FITS WCS for this mask's pixel array
        (`astropy.wcs.WCS` | `None`).

        Notes
        -----
        As expected for Astropy WCS objects, this defines pixel coordinates
        such that the first row and column in the arrays are ``(0, 0)``, not
        ``bbox.start``, as is the case for `projection`.

        This may be an approximation or absent if `projection` is not
        naturally representable as a FITS WCS.
        """
        return self.projection.as_fits_wcs(self.bbox) if self.projection is not None else None

    def __getitem__(self, bbox: Box) -> MaskedImage:
        return MaskedImage(
            self.image[bbox],
            mask=self.mask[bbox],
            variance=self.variance[bbox],
            opaque_metadata=(
                self._opaque_metadata.subset(bbox) if self._opaque_metadata is not None else None
            ),
            projection=self.projection,
        )

    def __str__(self) -> str:
        return f"MaskedImage({self.image!s}, {list(self.mask.schema.names)})"

    def __repr__(self) -> str:
        return f"MaskedImage({self.image!r}, mask_schema={self.mask.schema!r})"

    def copy(
        self,
        *,
        unit: astropy.units.UnitBase | None | EllipsisType = ...,
        mask_schema: MaskSchema | EllipsisType = ...,
        projection: Projection | None | EllipsisType = ...,
        start: Sequence[int] | EllipsisType = ...,
    ) -> MaskedImage:
        """Deep-copy the masked image, with optional updates.

        Notes
        -----
        This can also be used to rewrite the mask with a new related schema
        (e.g. adding or dropping mask planes, or changing ``dtype``; all
        planes with names in both schemas will be copied.).
        """
        return MaskedImage(
            image=self._image.copy(unit=unit, projection=projection, start=start),
            # We let the constructor take care of propagating projection and
            # unit updates.
            mask=self._mask.copy(schema=mask_schema, projection=None, start=start),
            variance=self._variance.copy(unit=None, projection=None, start=start),
            opaque_metadata=(self._opaque_metadata.copy() if self._opaque_metadata is not None else None),
        )

    def view(
        self,
        *,
        unit: astropy.units.UnitBase | None | EllipsisType = ...,
        mask_schema: MaskSchema | EllipsisType = ...,
        projection: Projection | None | EllipsisType = ...,
        start: Sequence[int] | EllipsisType = ...,
    ) -> MaskedImage:
        """Deep-copy the masked image, with optional updates.

        Notes
        -----
        This can only be used to make changes to schema descriptions; plane
        names must remain the same (in the same order).
        """
        return MaskedImage(
            image=self._image.view(unit=unit, projection=projection, start=start),
            # We let the constructor take care of propagating projection and
            # unit updates.
            mask=self._mask.view(schema=mask_schema, projection=None, start=start),
            variance=self._variance.view(unit=None, projection=None, start=start),
            opaque_metadata=(self._opaque_metadata.copy() if self._opaque_metadata is not None else None),
        )

    def serialize(self, archive: OutputArchive[Any]) -> MaskedImageSerializationModel:
        """Serialize the masked image to an output archive.

        Parameters
        ----------
        archive
            Archive to write to.
        """
        serialized_image = archive.serialize_direct(
            "image", functools.partial(self.image.serialize, save_projection=False)
        )
        serialized_mask = archive.serialize_direct(
            "mask", functools.partial(self.mask.serialize, save_projection=False)
        )
        serialized_variance = archive.serialize_direct(
            "variance", functools.partial(self.variance.serialize, save_projection=False)
        )
        serialized_projection = (
            archive.serialize_direct("projection", self.projection.serialize)
            if self.projection is not None
            else None
        )
        return MaskedImageSerializationModel(
            image=serialized_image,
            mask=serialized_mask,
            variance=serialized_variance,
            projection=serialized_projection,
        )

    @classmethod
    def deserialize(
        cls, model: MaskedImageSerializationModel[Any], archive: InputArchive[Any], *, bbox: Box | None = None
    ) -> MaskedImage:
        """Deserialize a image from an input archive.

        Parameters
        ----------
        model
            A Pydantic model representation of the image, holding references
            to data stored in the archive.
        archive
            Archive to read from.
        bbox
            Bounding box of a subimage to read instead.
        """
        image = Image.deserialize(model.image, archive, bbox=bbox)
        mask = Mask.deserialize(model.mask, archive, bbox=bbox)
        variance = Image.deserialize(model.variance, archive, bbox=bbox)
        return MaskedImage(image, mask=mask, variance=variance)

    def write_fits(
        self,
        filename: str,
        *,
        image_compression: FitsCompressionOptions | None = FitsCompressionOptions.DEFAULT,
        mask_compression: FitsCompressionOptions | None = FitsCompressionOptions.DEFAULT,
        variance_compression: FitsCompressionOptions | None = FitsCompressionOptions.DEFAULT,
    ) -> None:
        """Write the image to a FITS file.

        Parameters
        ----------
        filename
            Name of the file to write to.  Must be a local file.
        image_compression
            Compression options for the `image` plane.
        mask_compression
            Compression options for the `mask` plane.
        variance_compression
            Compression options for the `variance` plane.
        """
        compression_options = {}
        if image_compression is not FitsCompressionOptions.DEFAULT:
            compression_options["image"] = image_compression
        if mask_compression is not FitsCompressionOptions.DEFAULT:
            compression_options["mask"] = mask_compression
        if variance_compression is not FitsCompressionOptions.DEFAULT:
            compression_options["variance"] = variance_compression
        with FitsOutputArchive.open(
            filename, opaque_metadata=self._opaque_metadata, compression_options=compression_options
        ) as archive:
            archive.add_tree(self.serialize(archive))

    @classmethod
    def read_fits(cls, url: ResourcePathExpression, *, bbox: Box | None = None) -> MaskedImage:
        """Read an image from a FITS file.

        Parameters
        ----------
        url
            URL of the file to read; may be any type supported by
            `lsst.resources.ResourcePath`.
        bbox
            Bounding box of a subimage to read instead.
        """
        with FitsInputArchive.open(url, partial=(bbox is not None)) as archive:
            model = archive.get_tree(MaskedImageSerializationModel[TableCellReferenceModel])
            result = cls.deserialize(model, archive, bbox=bbox)
            # We only want to save opaque archive metadata on the outermost
            # object, and `deserialize` is designed to be usable if the
            # MaskedImage is nested within some other object, so we set it here
            # instead of passing it to the constructor there.  This might be
            # telling us the opaque metadata archive interfaces ought to be
            # reworked, but I don't see a better approach right now.
            result._opaque_metadata = archive.get_opaque_metadata()
        return result

    @classmethod
    def read_legacy(cls, filename: str, preserve_quantization: bool = False) -> MaskedImage:
        """Read a FITS file written by `lsst.afw.image.MaskedImage.writeFits`.

        Parameters
        ----------
        filename
            Full name of the file.
        preserve_quantization
            If `True`, ensure that writing the masked image back out again will
            exactly preserve quantization-compressed pixel values.  This causes
            the image and variance plane arrays to be marked as read-only and
            stores the original binary table data for those planes in memory.
            If the `MaskedImage` is copied, the precompressed pixel values are
            not transferred to the copy.

        Notes
        -----
        This method does not attach a `Projection` to the `MaskedImage` even
        if the legacy file is actually an `lsst.afw.image.Exposure` with a
        WCS attached.
        """
        opaque_metadata = FitsOpaqueMetadata()
        with astropy.io.fits.open(filename) as hdu_list:
            image_hdu: ExtensionHDU = hdu_list[1]
            image = Image.read_legacy(image_hdu)
            strip_wcs_cards(image_hdu.header)
            image_hdu.header.strip()
            image_hdu.header.remove("EXTNAME", ignore_missing=True)
            image_hdu.header.remove("EXTTYPE", ignore_missing=True)
            image_hdu.header.remove("INHERIT", ignore_missing=True)
            opaque_metadata.headers[ExtensionKey("IMAGE")] = image_hdu.header
            mask_hdu: ExtensionHDU = hdu_list[2]
            mask = Mask.read_legacy(mask_hdu)
            strip_wcs_cards(mask_hdu.header)
            mask_hdu.header.strip()
            mask_hdu.header.remove("EXTNAME", ignore_missing=True)
            mask_hdu.header.remove("EXTTYPE", ignore_missing=True)
            mask_hdu.header.remove("INHERIT", ignore_missing=True)
            # afw set BUNIT on masks because of limitations in how FITS
            # metadata is handled there.
            mask_hdu.header.remove("BUNIT", ignore_missing=True)
            opaque_metadata.headers[ExtensionKey("MASK")] = mask_hdu.header
            variance_hdu: ExtensionHDU = hdu_list[3]
            variance = Image.read_legacy(variance_hdu)
            strip_wcs_cards(variance_hdu.header)
            variance_hdu.header.strip()
            variance_hdu.header.remove("EXTNAME", ignore_missing=True)
            variance_hdu.header.remove("EXTTYPE", ignore_missing=True)
            variance_hdu.header.remove("INHERIT", ignore_missing=True)
            opaque_metadata.headers[ExtensionKey("VARIANCE")] = variance_hdu.header
        if preserve_quantization:
            image._array.flags["WRITEABLE"] = False
            mask._array.flags["WRITEABLE"] = False
            variance._array.flags["WRITEABLE"] = False
            with astropy.io.fits.open(filename, disable_image_compression=True) as hdu_list:
                opaque_metadata.precompressed["IMAGE"] = PrecompressedImage.from_bintable(hdu_list[1])
                opaque_metadata.precompressed["VARIANCE"] = PrecompressedImage.from_bintable(hdu_list[3])
        return cls(image, mask=mask, variance=variance, opaque_metadata=opaque_metadata)


class MaskedImageSerializationModel[P: pydantic.BaseModel](pydantic.BaseModel):
    """A Pydantic model used to represent a serialized `MaskedImage`."""

    image: ImageSerializationModel[P] = pydantic.Field(description="The main data image.")
    mask: MaskSerializationModel[P] = pydantic.Field(
        description="Bitmask that annotates the main image's pixels."
    )
    variance: ImageSerializationModel[P] = pydantic.Field(
        description="Per-pixel variance estimates for the main image."
    )
    projection: ProjectionSerializationModel[P] | None = pydantic.Field(
        default=None,
        exclude_if=is_none,
        description="Projection that maps the pixel grid to the sky.",
    )

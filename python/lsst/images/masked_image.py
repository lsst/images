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

__all__ = ("MaskedImage",)

from typing import Any

import astropy.io.fits
import astropy.units
import numpy as np
import pydantic

from lsst.resources import ResourcePathExpression

from ._geom import Box
from ._image import Image, ImageModel
from ._mask import Mask, MaskModel, MaskSchema
from .archive import InputArchive, OpaqueArchiveMetadata, OutputArchive
from .fits import (
    ExtensionHDU,
    FitsCompressionOptions,
    FitsInputArchive,
    FitsOpaqueMetadata,
    FitsOutputArchive,
    strip_wcs_cards,
)


class MaskedImage:
    """A multi-plane image with data (image), mask, and variance planes.

    Parameters
    ----------
    image
        The main image plane.
    mask, optional
        A bitmask image that annotates the main image plane.  Must have the
        same bounding box as ``image`` if provided.
    variance, optional
        The per-pixel uncertainty of the main image as an image of variance
        values.  Must have the same bounding box as ``image`` if provided, and
        its units must be the square of ``image.unit`` unless both are `None`.
    mask_schema, optional
        Schema for the mask plane.  Must be provided if and only if ``mask`` is
        not provided.
    opaque_metadata, optional
        Opaque metadata obtained from reading this object from storage.  It may
        be provided when writing to storage to propagate that metadata and/or
        preserve file-format-specific options (e.g. compression parameters).
    """

    def __init__(
        self,
        image: Image,
        *,
        mask: Mask | None = None,
        variance: Image | None = None,
        mask_schema: MaskSchema | None = None,
        opaque_metadata: OpaqueArchiveMetadata | None = None,
    ):
        if mask is None:
            if mask_schema is None:
                raise TypeError("'mask_schema' must be provided if 'mask' is not.")
            mask = Mask(schema=mask_schema, bbox=image.bbox)
        elif mask_schema is not None:
            raise TypeError("'mask_schema' may not be provided if 'mask' is.")
        if variance is None:
            variance = Image(
                1.0, dtype=np.float32, bbox=image.bbox, unit=None if image.unit is None else image.unit**2
            )
        assert image.bbox == mask.bbox, "Image and mask bboxes must agree."
        assert image.bbox == variance.bbox, "Image and variance bboxes must agree."
        if image.unit is None:
            assert variance.unit is None, "Image and variance must have units consistently."
        else:
            assert variance.unit is not None and variance.unit == image.unit**2, (
                "Image and variance must have consistent units."
            )
        self._image = image
        self._mask = mask
        self._variance = variance
        self._opaque_metadata = opaque_metadata

    @property
    def image(self) -> Image:
        """The main image plane."""
        return self._image

    @property
    def mask(self) -> Mask:
        """The mask plane."""
        return self._mask

    @property
    def variance(self) -> Image:
        """The variance plane."""
        return self._variance

    @property
    def bbox(self) -> Box:
        """The bounding box shared by all three image planes."""
        return self._image.bbox

    @property
    def unit(self) -> astropy.units.Unit | None:
        """The units of the image plane."""
        return self._image.unit

    def __getitem__(self, bbox: Box) -> MaskedImage:
        return MaskedImage(
            self.image[bbox],
            mask=self.mask[bbox],
            variance=self.variance[bbox],
            opaque_metadata=(
                self._opaque_metadata.subset(bbox) if self._opaque_metadata is not None else None
            ),
        )

    def __str__(self) -> str:
        return f"MaskedImage({self.image!s}, {list(self.mask.schema.names)})"

    def __repr__(self) -> str:
        return f"MaskedImage({self.image!r}, mask_schema={self.mask.schema!r})"

    def copy(self) -> MaskedImage:
        """Deep-copy the masked image."""
        return MaskedImage(
            image=self._image.copy(),
            mask=self._mask.copy(),
            variance=self._variance.copy(),
            opaque_metadata=(self._opaque_metadata.copy() if self._opaque_metadata is not None else None),
        )

    def serialize(self, archive: OutputArchive[Any]) -> MaskedImageModel:
        """Serialize the masked image to an output archive.

        Parameters
        ----------
        archive
            `~archive.OutputArchive` instance to write to.

        Returns
        -------
        model
            A Pydantic model representation of the masked image, holding
            references to data stored in the archive.

        Notes
        -----
        This method has the signature expected by
        `archive.OutputArchive.serialize_direct` and
        `archive.OutputArchive.serialize_pointer`, in order to allow objects
        holding a `MaskedImage` to delegate its serialization.

        This method does not initialize the opaque metadata of the returned
        masked image from the archive, as it does not assume that the masked
        image is the top-level entry in the archive.
        """
        image_model = archive.add_image("image", self.image)
        mask_model = archive.add_mask("mask", self.mask)
        variance_model = archive.add_image("variance", self.variance)
        return MaskedImageModel(image=image_model, mask=mask_model, variance=variance_model)

    @classmethod
    def deserialize(
        cls, model: MaskedImageModel, archive: InputArchive[Any], *, bbox: Box | None = None
    ) -> MaskedImage:
        """Deserialize a masked image from an input archive.

        Parameters
        ----------
        model
            A Pydantic model representation of the masked image, holding
            references to data stored in the archive.
        archive
            `~archive.InputArchive` instance to read from.
        bbox, optional
            Bounding box of a subimage to read instead.

        Returns
        -------
        masked_image
            The deserialized masked image.

        Notes
        -----
        When there is no ``bbox`` argument, this method has the signature
        expected by `archive.InputArchive.deserialize_pointer`, in order to
        allow objects holding a `MaskedImage` to delegate its deserialization.
        A ``lambda`` or `functools.partial` can be used to pass ``bbox`` in
        this case.

        This method does not pass the opaque metadata of the masked image to
        the archive, as it does not assume that the masked image is the
        top-level entry in the archive.
        """
        image = archive.get_image(model.image, bbox=bbox)
        mask = archive.get_mask(model.mask, bbox=bbox)
        variance = archive.get_image(model.variance, bbox=bbox)
        return MaskedImage(image, mask=mask, variance=variance)

    def write_fits(
        self,
        filename: str,
        *,
        image_compression: FitsCompressionOptions | None = FitsCompressionOptions.DEFAULT,
        mask_compression: FitsCompressionOptions | None = FitsCompressionOptions.DEFAULT,
        variance_compression: FitsCompressionOptions | None = FitsCompressionOptions.DEFAULT,
    ) -> None:
        """Write the masked image to a FITS file.

        Parameters
        ----------
        filename
            Name of the file to write to.  Must be a local file.
        compression, optional
            Compression options.
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
        """Read a masked image from a FITS file.

        Parameters
        ----------
        url
            URL of the file to read; may be any type supported by
            `lsst.resources.ResourcePath`.
        bbox, optional
            Bounding box of a subimage to read instead.

        Returns
        -------
        masked_image
            The loaded masked image.
        """
        with FitsInputArchive.open(url, partial=(bbox is not None)) as archive:
            model = archive.get_tree(MaskedImageModel)
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
    def read_legacy(cls, filename: str) -> MaskedImage:
        """Read a FITS file written by `lsst.afw.image.MaskedImage.writeFits`.

        Parameters
        ----------
        filename
            Full name of the file.

        Returns
        -------
        masked_image
            A new `MaskedImage` object.
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
            opaque_metadata.headers["IMAGE"] = image_hdu.header
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
            opaque_metadata.headers["MASK"] = mask_hdu.header
            variance_hdu: ExtensionHDU = hdu_list[3]
            variance = Image.read_legacy(variance_hdu)
            strip_wcs_cards(variance_hdu.header)
            variance_hdu.header.strip()
            variance_hdu.header.remove("EXTNAME", ignore_missing=True)
            variance_hdu.header.remove("EXTTYPE", ignore_missing=True)
            variance_hdu.header.remove("INHERIT", ignore_missing=True)
            opaque_metadata.headers["VARIANCE"] = variance_hdu.header
        return cls(image, mask=mask, variance=variance, opaque_metadata=opaque_metadata)


class MaskedImageModel(pydantic.BaseModel):
    """A Pydantic model used to represent a serialized `MaskedImage`."""

    image: ImageModel
    mask: MaskModel
    variance: ImageModel

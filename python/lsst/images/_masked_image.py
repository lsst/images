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
from collections.abc import Mapping
from contextlib import ExitStack
from typing import Any, Literal, overload

import astropy.io.fits
import astropy.units
import astropy.wcs
import numpy as np
import pydantic
from astro_metadata_translator import ObservationInfo

from lsst.resources import ResourcePath, ResourcePathExpression

from . import fits
from ._geom import Box
from ._image import Image, ImageSerializationModel
from ._mask import Mask, MaskPlane, MaskSchema, MaskSerializationModel
from ._transforms import Frame, Projection, ProjectionAstropyView, ProjectionSerializationModel
from .serialization import (
    ArchiveTree,
    InputArchive,
    OpaqueArchiveMetadata,
    OutputArchive,
)
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
    projection
        Projection that maps the pixel grid to the sky.
    obs_info
        General information about the associated observation in standardized
        form.
    """

    def __init__(
        self,
        image: Image,
        *,
        mask: Mask | None = None,
        variance: Image | None = None,
        mask_schema: MaskSchema | None = None,
        projection: Projection | None = None,
        obs_info: ObservationInfo | None = None,
    ):
        if projection is None:
            projection = image.projection
        else:
            image = image.view(projection=projection)
        if obs_info is None:
            obs_info = image.obs_info
        else:
            image = image.view(obs_info=obs_info)
        if mask is None:
            if mask_schema is None:
                raise TypeError("'mask_schema' must be provided if 'mask' is not.")
            mask = Mask(schema=mask_schema, bbox=image.bbox, projection=projection, obs_info=obs_info)
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
                obs_info=obs_info,
            )
        else:
            if image.bbox != variance.bbox:
                raise ValueError(f"Image ({image.bbox}) and variance ({variance.bbox}) bboxes do not agree.")
            variance = variance.view(projection=projection, obs_info=obs_info)
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
        self._opaque_metadata: OpaqueArchiveMetadata | None = None

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
    def obs_info(self) -> ObservationInfo | None:
        """General information about the associated observation in standard
        form. (`~astro_metadata_translator.ObservationInfo` | `None`).
        """
        return self._image.obs_info

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
        result = MaskedImage(
            # Projection and obs_info propagate from the image.
            self.image[bbox],
            mask=self.mask[bbox],
            variance=self.variance[bbox],
        )
        if self._opaque_metadata is not None:
            result._opaque_metadata = self._opaque_metadata.subset(bbox)
        return result

    def __str__(self) -> str:
        return f"MaskedImage({self.image!s}, {list(self.mask.schema.names)})"

    def __repr__(self) -> str:
        return f"MaskedImage({self.image!r}, mask_schema={self.mask.schema!r})"

    def copy(self) -> MaskedImage:
        """Deep-copy the masked image."""
        result = MaskedImage(image=self._image.copy(), mask=self._mask.copy(), variance=self._variance.copy())
        if self._opaque_metadata is not None:
            result._opaque_metadata = self._opaque_metadata.copy()
        return result

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

    @staticmethod
    def deserialize(
        model: MaskedImageSerializationModel[Any], archive: InputArchive[Any], *, bbox: Box | None = None
    ) -> MaskedImage:
        """Deserialize an image from an input archive.

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
        projection = (
            Projection.deserialize(model.projection, archive) if model.projection is not None else None
        )
        return MaskedImage(image, mask=mask, variance=variance, projection=projection)

    @staticmethod
    def _get_archive_tree_type[P: pydantic.BaseModel](
        pointer_type: type[P],
    ) -> type[MaskedImageSerializationModel[P]]:
        """Return the serialization model type for this object for an archive
        type that uses the given pointer type.
        """
        return MaskedImageSerializationModel[pointer_type]  # type: ignore

    def write_fits(
        self,
        filename: str,
        *,
        image_compression: fits.FitsCompressionOptions | None = fits.FitsCompressionOptions.DEFAULT,
        mask_compression: fits.FitsCompressionOptions | None = fits.FitsCompressionOptions.DEFAULT,
        variance_compression: fits.FitsCompressionOptions | None = fits.FitsCompressionOptions.DEFAULT,
        compression_seed: int | None = None,
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
        compression_seed
            A FITS tile compression seed to use whenever the configured
            compression seed is `None` or (for backwards compatibility) ``0``.
            This value is then incremented every time it is used.
        """
        compression_options = {}
        if image_compression is not fits.FitsCompressionOptions.DEFAULT:
            compression_options["image"] = image_compression
        if mask_compression is not fits.FitsCompressionOptions.DEFAULT:
            compression_options["mask"] = mask_compression
        if variance_compression is not fits.FitsCompressionOptions.DEFAULT:
            compression_options["variance"] = variance_compression
        fits.write(self, filename, compression_options=compression_options, compression_seed=compression_seed)

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
        return fits.read(cls, url, bbox=bbox)

    @staticmethod
    def from_legacy(
        legacy: Any,
        *,
        unit: astropy.units.Unit | None = None,
        plane_map: Mapping[str, MaskPlane] | None = None,
    ) -> MaskedImage:
        """Convert from an `lsst.afw.image.MaskedImage` instance.

        Parameters
        ----------
        legacy
            An `lsst.afw.image.MaskedImage` instance that will share image and
            variance (but not mask) pixel data with the returned object.
        unit
            Units of the image.
        plane_map
            A mapping from legacy mask plane name to the new plane name and
            description.
        """
        return MaskedImage(
            image=Image.from_legacy(legacy.getImage(), unit),
            mask=Mask.from_legacy(legacy.getMask(), plane_map),
            variance=Image.from_legacy(legacy.getVariance()),
        )

    def to_legacy(self, *, copy: bool | None = None, plane_map: Mapping[str, MaskPlane] | None = None) -> Any:
        """Convert to an `lsst.afw.image.MaskedImage` instance.

        Parameters
        ----------
        copy
            If `True`, always copy the image and variance pixel data.
            If `False`, return a view, and raise `TypeError` if the pixel data
            is read-only (this is not supported by afw).  If `None`, onyl if
            the pixel data is read-only.  Mask pixel data is always copied.
        plane_map
            A mapping from legacy mask plane name to the new plane name and
            description.
        """
        import lsst.afw.image

        return lsst.afw.image.MaskedImage(
            self.image.to_legacy(copy=copy),
            mask=self.mask.to_legacy(plane_map),
            variance=self.variance.to_legacy(copy=copy),
            dtype=self.image.array.dtype,
        )

    @overload
    @staticmethod
    def read_legacy(
        uri: ResourcePathExpression,
        *,
        preserve_quantization: bool = False,
        component: Literal["image"],
        fits_wcs_frame: Frame | None = None,
    ) -> Image: ...

    @overload
    @staticmethod
    def read_legacy(
        uri: ResourcePathExpression,
        *,
        plane_map: Mapping[str, MaskPlane] | None = None,
        component: Literal["mask"],
        fits_wcs_frame: Frame | None = None,
    ) -> Mask: ...

    @overload
    @staticmethod
    def read_legacy(
        uri: ResourcePathExpression,
        *,
        preserve_quantization: bool = False,
        component: Literal["variance"],
        fits_wcs_frame: Frame | None = None,
    ) -> Image: ...

    @overload
    @staticmethod
    def read_legacy(
        uri: ResourcePathExpression,
        *,
        preserve_quantization: bool = False,
        plane_map: Mapping[str, MaskPlane] | None = None,
        component: None = None,
        fits_wcs_frame: Frame | None = None,
    ) -> MaskedImage: ...

    @staticmethod
    def read_legacy(
        uri: ResourcePathExpression,
        *,
        preserve_quantization: bool = False,
        plane_map: Mapping[str, MaskPlane] | None = None,
        component: Literal["image", "mask", "variance"] | None = None,
        fits_wcs_frame: Frame | None = None,
    ) -> Any:
        """Read a FITS file written by `lsst.afw.image.MaskedImage.writeFits`.

        Parameters
        ----------
        uri
            URI or file name.
        preserve_quantization
            If `True`, ensure that writing the masked image back out again will
            exactly preserve quantization-compressed pixel values.  This causes
            the image and variance plane arrays to be marked as read-only and
            stores the original binary table data for those planes in memory.
            If the `MaskedImage` is copied, the precompressed pixel values are
            not transferred to the copy.
        plane_map
            A mapping from legacy mask plane name to the new plane name and
            description.
        component
            A component to read instead of the full image.
        fits_wcs_frame
            If not `None` and the HDU containing the image plane has a FITS
            WCS, attach a `Projection` to the returned masked image by
            converting that WCS.  When ``component`` is one of ``"image"``,
            ``"mask"``, or ``"variance"``, a FITS WCS from the component HDU
            is used instead (all three should have the same WCS).
        """
        fs, fspath = ResourcePath(uri).to_fsspec()
        with fs.open(fspath) as stream, astropy.io.fits.open(stream) as hdu_list:
            return MaskedImage._read_legacy_hdus(
                hdu_list,
                uri,
                preserve_quantization=preserve_quantization,
                plane_map=plane_map,
                component=component,
                fits_wcs_frame=fits_wcs_frame,
            )

    @staticmethod
    def _read_legacy_hdus(
        hdu_list: astropy.io.fits.HDUList,
        uri: ResourcePathExpression,
        *,
        preserve_quantization: bool = False,
        plane_map: Mapping[str, MaskPlane] | None = None,
        component: Literal["image", "mask", "variance"] | None,
        fits_wcs_frame: Frame | None = None,
    ) -> Any:
        opaque_metadata = fits.FitsOpaqueMetadata()
        opaque_metadata.extract_legacy_primary_header(hdu_list[0].header)
        image_bintable_hdu: astropy.io.fits.BinTableHDU | None = None
        variance_bintable_hdu: astropy.io.fits.BinTableHDU | None = None
        result: Any
        with ExitStack() as exit_stack:
            if preserve_quantization:
                fs, fspath = ResourcePath(uri).to_fsspec()
                bintable_stream = exit_stack.enter_context(fs.open(fspath))
                bintable_hdu_list = exit_stack.enter_context(
                    astropy.io.fits.open(bintable_stream, disable_image_compression=True)
                )
                image_bintable_hdu = bintable_hdu_list[1]
                variance_bintable_hdu = bintable_hdu_list[3]
            if component is None or component == "image":
                image = Image._read_legacy_hdu(
                    hdu_list[1],
                    opaque_metadata,
                    preserve_bintable=image_bintable_hdu,
                    fits_wcs_frame=fits_wcs_frame,
                )
                if component == "image":
                    result = image
            if component is None or component == "mask":
                mask = Mask._read_legacy_hdu(
                    hdu_list[2],
                    opaque_metadata,
                    plane_map=plane_map,
                    fits_wcs_frame=fits_wcs_frame if component is not None else None,
                )
                if component == "mask":
                    result = mask
            if component is None or component == "variance":
                variance = Image._read_legacy_hdu(
                    hdu_list[3],
                    opaque_metadata,
                    preserve_bintable=variance_bintable_hdu,
                    fits_wcs_frame=fits_wcs_frame if component is not None else None,
                )
                if component == "variance":
                    result = variance
        if component is None:
            result = MaskedImage(image, mask=mask, variance=variance)
        result._opaque_metadata = opaque_metadata
        return result


class MaskedImageSerializationModel[P: pydantic.BaseModel](ArchiveTree):
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

    @property
    def bbox(self) -> Box:
        """The bounding box of the image."""
        return self.image.bbox

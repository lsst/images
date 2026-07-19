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
from collections.abc import Mapping, MutableMapping
from contextlib import ExitStack
from types import EllipsisType
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import astropy.io.fits
import astropy.units
import astropy.wcs
import numpy as np
import pydantic

from lsst.resources import ResourcePath, ResourcePathExpression

from . import fits
from ._generalized_image import GeneralizedImage
from ._geom import Box
from ._image import DEFAULT_PIXEL_FRAME, Image, ImageSerializationModel
from ._mask import Mask, MaskPlane, MaskSchema, MaskSerializationModel
from ._transforms import Frame, SkyProjection, SkyProjectionSerializationModel
from .serialization import (
    ArchiveTree,
    InputArchive,
    InvalidParameterError,
    MetadataValue,
    OutputArchive,
)
from .utils import is_none

if TYPE_CHECKING:
    try:
        from lsst.afw.image import MaskedImage as LegacyMaskedImage
    except ImportError:
        type LegacyMaskedImage = Any  # type: ignore[no-redef]


class MaskedImage(GeneralizedImage):
    """A multi-plane image with data (image), mask, and variance planes.

    Parameters
    ----------
    image
        The main image plane.  If this has a `SkyProjection`, it will be used
        for all planes unless a ``sky_projection`` is passed separately.
    mask
        A bitmask image that annotates the main image plane.  Must have the
        same bounding box as ``image`` if provided.  Any attached
        ``sky_projection`` is replaced (possibly by `None`).
    variance
        The per-pixel uncertainty of the main image as an image of variance
        values.  Must have the same bounding box as ``image`` if provided, and
        its units must be the square of ``image.unit`` or `None`.
        Values default to ``1.0``.  Any attached ``sky_projection`` is replaced
        (possibly by `None`).
    mask_schema
        Schema for the mask plane.  Must be provided if and only if ``mask`` is
        not provided.
    sky_projection
        Projection that maps the pixel grid to the sky.
    metadata
        Arbitrary flexible metadata to associate with the image.
    """

    def __init__(
        self,
        image: Image,
        *,
        mask: Mask | None = None,
        variance: Image | None = None,
        mask_schema: MaskSchema | None = None,
        sky_projection: SkyProjection[Any] | None = None,
        metadata: dict[str, MetadataValue] | None = None,
    ) -> None:
        super().__init__(metadata)
        if sky_projection is None:
            sky_projection = image.sky_projection
        else:
            image = image.view(sky_projection=sky_projection)
        if mask is None:
            if mask_schema is None:
                raise TypeError("'mask_schema' must be provided if 'mask' is not.")
            mask = Mask(schema=mask_schema, bbox=image.bbox, sky_projection=sky_projection)
        elif mask_schema is not None:
            raise TypeError("'mask_schema' may not be provided if 'mask' is.")
        else:
            if image.bbox != mask.bbox:
                raise ValueError(f"Image ({image.bbox}) and mask ({mask.bbox}) bboxes do not agree.")
            mask = mask.view(sky_projection=sky_projection)
        if variance is None:
            variance = Image(
                1.0,
                dtype=np.float32,
                bbox=image.bbox,
                unit=None if image.unit is None else image.unit**2,
                sky_projection=sky_projection,
            )
        else:
            if image.bbox != variance.bbox:
                raise ValueError(f"Image ({image.bbox}) and variance ({variance.bbox}) bboxes do not agree.")
            variance = variance.view(sky_projection=sky_projection)
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

    @property
    def image(self) -> Image:
        """The main image plane (`~lsst.images.Image`)."""
        return self._image

    @property
    def mask(self) -> Mask:
        """The mask plane (`~lsst.images.Mask`).

        Assigning a new `~lsst.images.Mask` (for example one returned by
        `~lsst.images.Mask.add_planes`) replaces the mask plane.  The new mask
        must share this image's bounding box; its sky projection is replaced
        with the image's.
        """
        return self._mask

    @mask.setter
    def mask(self, value: Mask) -> None:
        if value.bbox != self._image.bbox:
            raise ValueError(f"Image ({self._image.bbox}) and mask ({value.bbox}) bboxes do not agree.")
        if self._image.sky_projection != value.sky_projection:
            raise ValueError("Image sky projection and new mask sky projection do not agree")
        # Use a view to ensure that the WCS instances across the masked image
        # are the same projections.
        self._mask = value.view(sky_projection=self._image.sky_projection)

    @property
    def variance(self) -> Image:
        """The variance plane (`~lsst.images.Image`)."""
        return self._variance

    @property
    def bbox(self) -> Box:
        """The bounding box shared by all three image planes
        (`~lsst.images.Box`).
        """
        return self._image.bbox

    @property
    def unit(self) -> astropy.units.UnitBase | None:
        """The units of the image plane (`astropy.units.Unit` | `None`)."""
        return self._image.unit

    @property
    def sky_projection(self) -> SkyProjection[Any] | None:
        """The projection that maps the pixel grid to the sky
        (`~lsst.images.SkyProjection` | `None`).
        """
        return self._image.sky_projection

    def __getitem__(self, bbox: Box | EllipsisType) -> MaskedImage:
        bbox, _ = self._handle_getitem_args(bbox)
        return self._transfer_metadata(
            MaskedImage(
                # Projection and obs_info propagate from the image.
                self.image[bbox],
                mask=self.mask[bbox],
                variance=self.variance[bbox],
            ),
            bbox=bbox,
        )

    def __setitem__(self, bbox: Box | EllipsisType, value: MaskedImage) -> None:
        self._image[bbox] = value.image
        self._mask[bbox] = value.mask
        self._variance[bbox] = value.variance

    def __str__(self) -> str:
        return f"MaskedImage({self.image!s}, {list(self.mask.schema.names)})"

    def __repr__(self) -> str:
        return f"MaskedImage({self.image!r}, mask_schema={self.mask.schema!r})"

    def copy(self) -> MaskedImage:
        """Deep-copy the masked image and metadata."""
        return self._transfer_metadata(
            MaskedImage(image=self._image.copy(), mask=self._mask.copy(), variance=self._variance.copy()),
            copy=True,
        )

    def serialize(self, archive: OutputArchive[Any]) -> MaskedImageSerializationModel[Any]:
        """Serialize the masked image to an output archive.

        Parameters
        ----------
        archive
            Archive to write to.
        """
        return self._serialize_impl(MaskedImageSerializationModel, archive)

    def _serialize_impl[M: MaskedImageSerializationModel[Any]](
        self, model_type: type[M], archive: OutputArchive[Any]
    ) -> M:
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
            archive.serialize_direct("sky_projection", self.sky_projection.serialize)
            if self.sky_projection is not None
            else None
        )
        # When M is a subclass of MaskedImageSerializationModel, it probably
        # has fields that aren't being set here. We're intentionally making use
        # of the fact that model_construct doesn't guard against that so we can
        # instead set them in a subclass implementation later, after calling
        # super() to construct an instance of the right type. MyPy is actually
        # fine with this, but only because it incorrectly thinks model_type is
        # only ever MaskedImageSerializationModel, not a subclass, and that
        # makes it incorrectly unhappy about the return type.
        return model_type.model_construct(  # type: ignore[return-value]
            image=serialized_image,
            mask=serialized_mask,
            variance=serialized_variance,
            sky_projection=serialized_projection,
            metadata=self.metadata,
        )

    @staticmethod
    def _get_archive_tree_type[P: pydantic.BaseModel](
        pointer_type: type[P],
    ) -> type[MaskedImageSerializationModel[P]]:
        """Return the serialization model type for this object for an archive
        type that uses the given pointer type.
        """
        return MaskedImageSerializationModel[pointer_type]  # type: ignore

    @staticmethod
    def from_legacy(
        legacy: LegacyMaskedImage,
        *,
        unit: astropy.units.UnitBase | None = None,
        plane_map: Mapping[str, MaskPlane] | None = None,
        sky_projection: SkyProjection[Any] | None = None,
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
            description.  If not provided, the right legacy mask plane will be
            guessed, but this can depend on which mask planes the legacy
            mask actually has set.
        sky_projection
            Projection from pixels to xky.
        """
        return MaskedImage(
            image=Image.from_legacy(legacy.getImage(), unit, sky_projection=sky_projection),
            mask=Mask.from_legacy(legacy.getMask(), plane_map, sky_projection=sky_projection),
            variance=Image.from_legacy(legacy.getVariance(), sky_projection=sky_projection),
        )

    def to_legacy(
        self, *, copy: bool | None = None, plane_map: Mapping[str, MaskPlane] | None = None
    ) -> LegacyMaskedImage:
        """Convert to an `lsst.afw.image.MaskedImage` instance.

        Parameters
        ----------
        copy
            If `True`, always copy the image and variance pixel data.
            If `False`, return a view, and raise `TypeError` if the pixel data
            is read-only (this is not supported by afw).  If `None`, only copy
            if the pixel data is read-only.  Mask pixel data is always copied.
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

    @classmethod
    def from_hdu_list(
        cls,
        hdu_list: astropy.io.fits.HDUList,
        *,
        fits_wcs_frame: Frame | None = DEFAULT_PIXEL_FRAME,
    ) -> MaskedImage:
        """Reconstruct a `~lsst.images.MaskedImage` from a cut-down
        ``lsst.images`` FITS HDU list.

        This assumes the ``PRIMARY``, ``IMAGE``, ``MASK``, and ``VARIANCE``
        HDUs written for the masked-image cut-outs produced by
        ``dax_images_cutout``: a real ``lsst.images`` file with its JSON-tree,
        index, and any nested-archive HDUs dropped.  The reconstructed object
        can be re-serialized as a normal ``lsst.images`` file (with schema and
        index) so it can be read with the full ``lsst.images`` infrastructure.

        Parameters
        ----------
        hdu_list
            HDU list with ``IMAGE``, ``MASK``, and ``VARIANCE`` extensions and
            a primary HDU.
        fits_wcs_frame
            Pixel-grid `~lsst.images.Frame` for the
            `~lsst.images.SkyProjection` reconstructed from the FITS WCS.
            Defaults to a plain pixel frame; pass `None` to skip attaching a
            projection.

        Returns
        -------
        `~lsst.images.MaskedImage`
            The reconstructed masked image.

        Raises
        ------
        ValueError
            Raised if the ``MASK`` HDU has neither ``MSKN`` nor ``MP_`` mask-
            plane cards, since the mask schema cannot then be reconstructed, or
            if ``hdu_list`` contains more than one ``MASK`` HDU (multiple
            ``MASK`` extensions, distinguished by ``EXTVER``, are not handled
            here and would otherwise be silently dropped).

        Notes
        -----
        Both mask-plane conventions are supported: the self-describing
        ``MSKN``/``MSKM``/``MSKD`` cards written by ``lsst.images``, and the
        legacy `lsst.afw.image` ``MP_*`` cards (as produced by
        ``dax_images_cutout`` from afw-written images).  Legacy masks are
        mapped to a new schema with the same plane-guessing used by
        `read_legacy`.

        Unlike `read_legacy`, the legacy ``MP_*`` mask-plane cards are kept
        (not stripped) for backwards compatibility, since this path
        reconstructs a file that may still be read by legacy tooling.  They are
        re-indexed to the reshuffled schema so each ``MP_`` bit matches the
        plane's position in the written ``MSKN`` layout.

        The headers of the HDUs in ``hdu_list`` are modified in place: the WCS
        and mask-schema cards interpreted here are stripped from the caller's
        headers.
        """
        n_mask_hdus = sum(1 for hdu in hdu_list if hdu.name == "MASK")
        if n_mask_hdus > 1:
            raise ValueError(
                f"Found {n_mask_hdus} MASK HDUs; from_hdu_list supports only a single MASK "
                "extension and would otherwise silently drop mask information from the others."
            )
        mask_hdu = hdu_list["MASK"]
        if not any(card.keyword.startswith(("MSKN", "MP_")) for card in mask_hdu.header.cards):
            raise ValueError("MASK HDU has no MSKN or MP_ cards; cannot reconstruct the mask schema.")
        opaque_metadata = fits.FitsOpaqueMetadata()
        opaque_metadata.add_cutdown_primary_header(hdu_list[0].header)
        image = Image._read_legacy_hdu(
            hdu_list["IMAGE"], opaque_metadata, preserve_bintable=None, fits_wcs_frame=fits_wcs_frame
        )
        mask = Mask._read_legacy_hdu(
            mask_hdu, opaque_metadata, fits_wcs_frame=None, strip_legacy_planes=False
        )
        variance = Image._read_legacy_hdu(
            hdu_list["VARIANCE"], opaque_metadata, preserve_bintable=None, fits_wcs_frame=None
        )
        result = cls(image, mask=mask, variance=variance)
        result._opaque_metadata = opaque_metadata
        return result

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
            If the `~lsst.images.MaskedImage` is copied, the precompressed
            pixel values are not transferred to the copy.
        plane_map
            A mapping from legacy mask plane name to the new plane name and
            description.  If not provided, the right legacy mask plane will be
            guessed, but this can depend on which mask planes the legacy
            mask actually has set.
        component
            A component to read instead of the full image.
        fits_wcs_frame
            If not `None` and the HDU containing the image plane has a FITS
            WCS, attach a `~lsst.images.SkyProjection` to the returned masked
            image by converting that WCS.  When ``component`` is one of
            ``"image"``, ``"mask"``, or ``"variance"``, a FITS WCS from the
            component HDU is used instead (all three should have the same WCS).
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
        opaque_metadata: fits.FitsOpaqueMetadata | None = None,
        preserve_quantization: bool = False,
        plane_map: Mapping[str, MaskPlane] | None = None,
        component: Literal["image", "mask", "variance"] | None,
        fits_wcs_frame: Frame | None = None,
    ) -> Any:
        if opaque_metadata is None:
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

    def _fill_legacy_metadata(self, legacy_metadata: MutableMapping[str, Any]) -> None:
        """Fill a legacy mutable mapping (e.g `lsst.daf.base.PropertySet`)
        with metadata suitable for an `lsst.afw.image.Exposure` representation
        of this object.
        """
        # We just dump all of the FITS headers and non-FITS metadata into the
        # legacy metadata component, to make sure we have everything. We dump
        # the latter into a pair of special cards to be able to full round-trip
        # them (including case preservation).
        if self.unit is not None:
            try:
                legacy_metadata["BUNIT"] = self.unit.to_string(format="fits")
            except ValueError:
                # Write units that astropy doesn't think FITS will accept
                # anyway; FITS standard says "SHOULD" about using its
                # recommended units, and coloring outside the lines is better
                # than lying.
                legacy_metadata["BUNIT"] = self.unit.to_string()
        if isinstance(self._opaque_metadata, fits.FitsOpaqueMetadata):
            legacy_metadata.update(self._opaque_metadata.headers[fits.ExtensionKey()])
        for n, (k, v) in enumerate(self.metadata.items()):
            legacy_metadata[f"LSST IMAGES KEY {n + 1}"] = k
            legacy_metadata[f"LSST IMAGES VALUE {n + 1}"] = v


class MaskedImageSerializationModel[P: pydantic.BaseModel](ArchiveTree):
    """A Pydantic model used to represent a serialized `MaskedImage`."""

    SCHEMA_NAME: ClassVar[str] = "masked_image"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0.dev0"
    MIN_READ_VERSION: ClassVar[int] = 1
    PUBLIC_TYPE: ClassVar[type] = MaskedImage

    image: ImageSerializationModel[P] = pydantic.Field(description="The main data image.")
    mask: MaskSerializationModel[P] = pydantic.Field(
        description="Bitmask that annotates the main image's pixels."
    )
    variance: ImageSerializationModel[P] = pydantic.Field(
        description="Per-pixel variance estimates for the main image."
    )
    sky_projection: SkyProjectionSerializationModel[P] | None = pydantic.Field(
        default=None,
        exclude_if=is_none,
        description="Projection that maps the pixel grid to the sky.",
    )

    @property
    def bbox(self) -> Box:
        """The bounding box of the image."""
        return self.image.bbox

    def deserialize(
        self, archive: InputArchive[Any], *, bbox: Box | None = None, **kwargs: Any
    ) -> MaskedImage:
        """Deserialize an image from an input archive.

        Parameters
        ----------
        archive
            Archive to read from.
        bbox
            Bounding box of a subimage to read instead.
        **kwargs
            Unsupported keyword arguments are accepted only to provide better
            error messages (raising `serialization.InvalidParameterError`).
        """
        if kwargs:
            raise InvalidParameterError(f"Unrecognized parameters for MaskedImage: {set(kwargs.keys())}.")
        image = self.image.deserialize(archive, bbox=bbox)
        mask = self.mask.deserialize(archive, bbox=bbox)
        variance = self.variance.deserialize(archive, bbox=bbox)
        sky_projection = self.sky_projection.deserialize(archive) if self.sky_projection is not None else None
        return MaskedImage(
            image, mask=mask, variance=variance, sky_projection=sky_projection
        )._finish_deserialize(self)

    def deserialize_component(self, component: str, archive: InputArchive[Any], **kwargs: Any) -> Any:
        if component == "bbox" and kwargs:
            raise InvalidParameterError(
                f"Unrecognized parameters for MaskedImage.bbox: {set(kwargs.keys())}."
            )
        return super().deserialize_component(component, archive, **kwargs)

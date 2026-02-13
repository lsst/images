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

__all__ = ("VisitImage", "VisitImageSerializationModel")

import functools
from collections.abc import Mapping, Sequence
from types import EllipsisType
from typing import Any, cast

import astropy.io.fits
import astropy.units
import astropy.wcs
import pydantic

from lsst.resources import ResourcePathExpression

from ._geom import Box
from ._image import Image, ImageSerializationModel
from ._mask import Mask, MaskPlane, MaskSchema, MaskSerializationModel
from ._masked_image import MaskedImage, MaskedImageSerializationModel
from ._transforms import DetectorFrame, Projection, ProjectionAstropyView, ProjectionSerializationModel
from .fits import (
    ExtensionKey,
    FitsInputArchive,
    FitsOpaqueMetadata,
)
from .psfs import (
    PiffSerializationModel,
    PiffWrapper,
    PointSpreadFunction,
    PSFExSerializationModel,
    PSFExWrapper,
)
from .serialization import (
    ArchiveReadError,
    InputArchive,
    OpaqueArchiveMetadata,
    OutputArchive,
    TableCellReferenceModel,
)


class VisitImage(MaskedImage):
    """A calibrated single-visit image.

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
        Projection that maps the pixel grid to the sky.  Can only be `None` if
        a projection is already attached to ``image``.
    psf
        Point-spread function model for this image, or an exception explaining
        why it could not be read (to be raised if the PSF is requested later).
    """

    def __init__(
        self,
        image: Image,
        *,
        mask: Mask | None = None,
        variance: Image | None = None,
        mask_schema: MaskSchema | None = None,
        opaque_metadata: OpaqueArchiveMetadata | None = None,
        projection: Projection[DetectorFrame] | None = None,
        psf: PointSpreadFunction | ArchiveReadError,
    ):
        super().__init__(
            image,
            mask=mask,
            variance=variance,
            mask_schema=mask_schema,
            opaque_metadata=opaque_metadata,
            projection=projection,
        )
        if self.image.unit is None:
            raise TypeError("The image component of a VisitImage must have units.")
        if self.image.projection is None:
            raise TypeError("The projection component of a VisitImage cannot be None.")
        if not isinstance(self.image.projection.pixel_frame, DetectorFrame):
            raise TypeError("The projection's pixel frame must be a DetectorFrame for VisitImage.")
        self._psf = psf

    @property
    def unit(self) -> astropy.units.UnitBase:
        """The units of the image plane (`astropy.units.Unit`)."""
        return cast(astropy.units.UnitBase, super().unit)

    @property
    def projection(self) -> Projection[DetectorFrame]:
        """The projection that maps the pixel grid to the sky
        (`Projection` [`DetectorFrame`]).
        """
        return cast(Projection[DetectorFrame], super().projection)

    @property
    def astropy_wcs(self) -> ProjectionAstropyView:
        """An Astropy WCS for the pixel arrays (`ProjectionAstropyView`).

        Notes
        -----
        As expected for Astropy WCS objects, this defines pixel coordinates
        such that the first row and column in the arrays are ``(0, 0)``, not
        ``bbox.start``, as is the case for `projection`.

        This object satisfies the `astropy.wcs.wcsapi.BaseHighLevelWCS` and
        `astropy.wcs.wcsapi.BaseLowLevelWCS` interfaces, but it is not an
        `astropy.wcs.WCS` (use `fits_wcs` for that).
        """
        return cast(ProjectionAstropyView, super().astropy_wcs)

    @property
    def psf(self) -> PointSpreadFunction:
        """The point-spread function model for this image
        (`.psfs.PointSpreadFunction`).
        """
        if isinstance(self._psf, ArchiveReadError):
            raise self._psf
        return self._psf

    def __getitem__(self, bbox: Box) -> VisitImage:
        return VisitImage(
            self.image[bbox],
            mask=self.mask[bbox],
            variance=self.variance[bbox],
            opaque_metadata=(
                self._opaque_metadata.subset(bbox) if self._opaque_metadata is not None else None
            ),
            projection=self.projection,
            psf=self.psf,
        )

    def __str__(self) -> str:
        return f"VisitImage({self.image!s}, {list(self.mask.schema.names)})"

    def __repr__(self) -> str:
        return f"VisitImage({self.image!r}, mask_schema={self.mask.schema!r})"

    def copy(
        self,
        *,
        unit: astropy.units.UnitBase | None | EllipsisType = ...,
        mask_schema: MaskSchema | EllipsisType = ...,
        projection: Projection | None | EllipsisType = ...,
        psf: PointSpreadFunction | EllipsisType = ...,
        start: Sequence[int] | EllipsisType = ...,
    ) -> VisitImage:
        """Deep-copy the visit image, with optional updates.

        Notes
        -----
        This can also be used to rewrite the mask with a new related schema
        (e.g. adding or dropping mask planes, or changing ``dtype``; all
        planes with names in both schemas will be copied.).
        """
        return VisitImage(
            image=self._image.copy(unit=unit, projection=projection, start=start),
            # We let the constructor take care of propagating projection and
            # unit updates.
            mask=self._mask.copy(schema=mask_schema, projection=None, start=start),
            variance=self._variance.copy(unit=None, projection=None, start=start),
            opaque_metadata=(self._opaque_metadata.copy() if self._opaque_metadata is not None else None),
            psf=psf if psf is not ... else self._psf,
        )

    def view(
        self,
        *,
        unit: astropy.units.UnitBase | None | EllipsisType = ...,
        mask_schema: MaskSchema | EllipsisType = ...,
        projection: Projection | None | EllipsisType = ...,
        psf: PointSpreadFunction | EllipsisType = ...,
        start: Sequence[int] | EllipsisType = ...,
    ) -> VisitImage:
        """Deep-copy the masked image, with optional updates.

        Notes
        -----
        This can only be used to make changes to schema descriptions; plane
        names must remain the same (in the same order).
        """
        return VisitImage(
            image=self._image.view(unit=unit, projection=projection, start=start),
            # We let the constructor take care of propagating projection and
            # unit updates.
            mask=self._mask.view(schema=mask_schema, projection=None, start=start),
            variance=self._variance.view(unit=None, projection=None, start=start),
            opaque_metadata=(self._opaque_metadata.copy() if self._opaque_metadata is not None else None),
            psf=psf if psf is not ... else self._psf,
        )

    def serialize(self, archive: OutputArchive[Any]) -> VisitImageSerializationModel:
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
        serialized_psf: PiffSerializationModel | PSFExSerializationModel
        match self._psf:
            # MyPy is able to figure things out here with this match statement,
            # but not a single isinstance check on both types.
            case PiffWrapper():
                serialized_psf = archive.serialize_direct("psf", self._psf.serialize)
            case PSFExWrapper():
                serialized_psf = archive.serialize_direct("psf", self._psf.serialize)
            case _:
                raise TypeError(
                    f"Cannot serialize VisitImage with unrecognized PSF type {type(self._psf).__name__}."
                )
        return VisitImageSerializationModel(
            image=serialized_image,
            mask=serialized_mask,
            variance=serialized_variance,
            projection=serialized_projection,
            psf=serialized_psf,
        )

    # Type-checkers want the model argument to only require
    # MaskedImageSerializationModel[Any], and they'd be absolutely right if
    # this were a regular instance method. But whether Liskov substitution
    # applies to classmethods and staticmethods is sort of context-dependent,
    # and here we do not want it to.
    @staticmethod
    def deserialize(
        model: VisitImageSerializationModel[Any],  # type: ignore[override]
        archive: InputArchive[Any],
        *,
        bbox: Box | None = None,
    ) -> VisitImage:
        masked_image = MaskedImage.deserialize(model, archive, bbox=bbox)
        psf: PointSpreadFunction | ArchiveReadError
        try:
            match model.psf:
                case PiffSerializationModel():
                    psf = PiffWrapper.deserialize(model.psf, archive)
                case PSFExSerializationModel():
                    psf = PSFExWrapper.deserialize(model.psf, archive)
                case _:
                    raise ArchiveReadError("PSF model type not recognized.")
        except ArchiveReadError as err:
            psf = err
        projection = Projection.deserialize(model.projection, archive)
        result = VisitImage(
            masked_image.image,
            mask=masked_image.mask,
            variance=masked_image.variance,
            opaque_metadata=masked_image._opaque_metadata,
            psf=psf,
            projection=projection,
        )
        return result

    @staticmethod
    def read_fits(url: ResourcePathExpression, *, bbox: Box | None = None) -> VisitImage:
        with FitsInputArchive.open(url, partial=(bbox is not None)) as archive:
            model = archive.get_tree(VisitImageSerializationModel[TableCellReferenceModel])
            result = VisitImage.deserialize(model, archive, bbox=bbox)
            # We only want to save opaque archive metadata on the outermost
            # object, and `deserialize` is designed to be usable if the
            # MaskedImage is nested within some other object, so we set it here
            # instead of passing it to the constructor there.  This might be
            # telling us the opaque metadata archive interfaces ought to be
            # reworked, but I don't see a better approach right now.
            result._opaque_metadata = archive.get_opaque_metadata()
        return result

    @staticmethod
    def read_legacy(
        filename: str,
        *,
        preserve_quantization: bool = False,
        plane_map: Mapping[str, MaskPlane] | None = None,
        instrument: str | None = None,
        visit: int | None = None,
    ) -> VisitImage:
        """Read a FITS file written by `lsst.afw.image.Exposure.writeFits`.

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
        plane_map
            A mapping from legacy mask plane name to the new plane name and
            description.
        """
        from lsst.afw.image import ExposureFitsReader

        masked_image = MaskedImage.read_legacy(
            filename, preserve_quantization=preserve_quantization, plane_map=plane_map
        )
        primary_header = cast(FitsOpaqueMetadata, masked_image._opaque_metadata).headers[ExtensionKey()]
        if instrument is None:
            try:
                instrument = primary_header["LSST BUTLER DATAID INSTRUMENT"]
            except LookupError:
                raise ValueError(
                    "Instrument name could not be found in butler data ID FITS headers and must be provided."
                ) from None
        if visit is None:
            try:
                visit = primary_header["LSST BUTLER DATAID VISIT"]
            except LookupError:
                raise ValueError(
                    "Visit ID could not be found in butler data ID FITS headers and must be provided."
                ) from None
        reader = ExposureFitsReader(filename)
        legacy_wcs = reader.readWcs()
        if legacy_wcs is None:
            raise ValueError(f"Exposure file {filename!r} does not have a SkyWcs.")
        legacy_detector = reader.readDetector()
        if legacy_detector is None:
            raise ValueError("Exposure file {filename!r} does not have a Detector.")
        detector_bbox = Box.from_legacy(legacy_detector.getBBox())
        projection = Projection.from_legacy(
            legacy_wcs,
            DetectorFrame(
                instrument=instrument,
                visit=visit,
                detector=legacy_detector.getId(),
                bbox=detector_bbox,
            ),
        )
        legacy_psf = reader.readPsf()
        if legacy_psf is None:
            raise ValueError("Exposure file {filename!r} does not have a Psf.")
        psf = PointSpreadFunction.from_legacy(legacy_psf, bounds=detector_bbox)
        return VisitImage(
            masked_image.image,
            mask=masked_image.mask,
            variance=masked_image.variance,
            opaque_metadata=masked_image._opaque_metadata,
            projection=projection,
            psf=psf,
        )


class VisitImageSerializationModel[P: pydantic.BaseModel](MaskedImageSerializationModel[P]):
    """A Pydantic model used to represent a serialized `VisitImage`."""

    # Inherited attributes are duplicated because that improves the docs
    # (some limitation in the sphinx/pydantic integration), and these are
    # import docs.

    image: ImageSerializationModel[P] = pydantic.Field(description="The main data image.")
    mask: MaskSerializationModel[P] = pydantic.Field(
        description="Bitmask that annotates the main image's pixels."
    )
    variance: ImageSerializationModel[P] = pydantic.Field(
        description="Per-pixel variance estimates for the main image."
    )
    projection: ProjectionSerializationModel[P] = pydantic.Field(
        description="Projection that maps the pixel grid to the sky.",
    )
    psf: PiffSerializationModel | PSFExSerializationModel | Any = pydantic.Field(
        union_mode="left_to_right", description="PSF model for the image."
    )

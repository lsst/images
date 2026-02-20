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

import warnings

__all__ = ("VisitImage", "VisitImageSerializationModel")

import functools
from collections.abc import Mapping, Sequence
from types import EllipsisType
from typing import Any, Literal, cast, overload

import astropy.io.fits
import astropy.units
import astropy.wcs
import pydantic

from ._geom import Box
from ._image import Image, ImageSerializationModel
from ._mask import Mask, MaskPlane, MaskSchema, MaskSerializationModel, get_legacy_visit_image_mask_planes
from ._masked_image import MaskedImage, MaskedImageSerializationModel
from ._transforms import DetectorFrame, Projection, ProjectionAstropyView, ProjectionSerializationModel
from .fits import FitsOpaqueMetadata
from .psfs import (
    PiffSerializationModel,
    PiffWrapper,
    PointSpreadFunction,
    PSFExSerializationModel,
    PSFExWrapper,
)
from .serialization import ArchiveReadError, InputArchive, OutputArchive


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
        projection: Projection[DetectorFrame] | None = None,
        psf: PointSpreadFunction | ArchiveReadError,
    ):
        super().__init__(
            image,
            mask=mask,
            variance=variance,
            mask_schema=mask_schema,
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
        result = VisitImage(
            self.image[bbox],
            mask=self.mask[bbox],
            variance=self.variance[bbox],
            projection=self.projection,
            psf=self.psf,
        )
        if opaque_metadata := getattr(self, "_opaque_metadata", None):
            result._opaque_metadata = opaque_metadata.subset(bbox)
        return result

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
        result = VisitImage(
            image=self._image.copy(unit=unit, projection=projection, start=start),
            # We let the constructor take care of propagating projection and
            # unit updates.
            mask=self._mask.copy(schema=mask_schema, projection=None, start=start),
            variance=self._variance.copy(unit=None, projection=None, start=start),
            psf=psf if psf is not ... else self._psf,
        )
        if self._opaque_metadata is not None:
            result._opaque_metadata = self._opaque_metadata.copy()
        return result

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
        result = VisitImage(
            image=self._image.view(unit=unit, projection=projection, start=start),
            # We let the constructor take care of propagating projection and
            # unit updates.
            mask=self._mask.view(schema=mask_schema, projection=None, start=start),
            variance=self._variance.view(unit=None, projection=None, start=start),
            psf=psf if psf is not ... else self._psf,
        )
        if self._opaque_metadata is not None:
            result._opaque_metadata = self._opaque_metadata.copy()
        return result

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
        psf = model.deserialize_psf(archive)
        projection = Projection.deserialize(model.projection, archive)
        result = VisitImage(
            masked_image.image,
            mask=masked_image.mask,
            variance=masked_image.variance,
            psf=psf,
            projection=projection,
        )
        return result

    @staticmethod
    def _get_archive_tree_type[P: pydantic.BaseModel](
        pointer_type: type[P],
    ) -> type[VisitImageSerializationModel[P]]:
        """Return the serialization model type for this object for an archive
        type that uses the given pointer type.
        """
        return VisitImageSerializationModel[pointer_type]  # type: ignore

    # write_fits and read_fits inherited from MaskedImage.

    @staticmethod
    def from_legacy(
        legacy: Any,
        *,
        unit: astropy.units.Unit | None = None,
        plane_map: Mapping[str, MaskPlane] | None = None,
        instrument: str | None = None,
        visit: int | None = None,
        component: str | None = None,
    ) -> VisitImage:
        """Convert from an `lsst.afw.image.Exposure` instance.

        Parameters
        ----------
        legacy
            An `lsst.afw.image.Exposure` instance that will share image and
            variance (but not mask) pixel data with the returned object.
        unit
            Units of the image.  If not provided, the ``BUNIT`` metadata
            key will be used, if available.
        plane_map
            A mapping from legacy mask plane name to the new plane name and
            description.  If `None` (default)
            `get_legacy_visit_image_mask_planes` is used.
        instrument
            Name of the instrument.  Extracted from the metadata if not
            provided.
        visit
            ID of the visit.  Extracted from the metadata if not provided.
        """
        if plane_map is None:
            plane_map = get_legacy_visit_image_mask_planes()
        md = legacy.getMetadata()
        if instrument is None:
            try:
                instrument = str(md["LSST BUTLER DATAID INSTRUMENT"])
            except LookupError:
                raise ValueError(
                    "Instrument name could not be found in butler data ID metadata and must be provided."
                ) from None
        if visit is None:
            try:
                visit = int(md["LSST BUTLER DATAID VISIT"])
            except LookupError:
                raise ValueError(
                    "Visit ID could not be found in butler data ID metadata and must be provided."
                ) from None
        if unit is None:
            try:
                unit = astropy.units.Unit(md["BUNIT"], format="fits")
            except LookupError:
                raise ValueError(
                    "BUNIT could not be found in exposure metadata and must be provided."
                ) from None
        legacy_wcs = legacy.getWcs()
        if legacy_wcs is None:
            raise ValueError("Exposure does not have a SkyWcs.")
        legacy_detector = legacy.getDetector()
        if legacy_detector is None:
            raise ValueError("Exposure does not have a Detector.")
        detector_bbox = Box.from_legacy(legacy_detector.getBBox())
        opaque_fits_metadata = FitsOpaqueMetadata()
        primary_header = astropy.io.fits.Header()
        with warnings.catch_warnings():
            # Silence warnings about long keys becoming HIERARCH.
            warnings.simplefilter("ignore", category=astropy.io.fits.verify.VerifyWarning)
            primary_header.update(md.toOrderedDict())
        opaque_fits_metadata.extract_legacy_primary_header(primary_header)
        projection = Projection.from_legacy(
            legacy_wcs,
            DetectorFrame(
                instrument=instrument,
                visit=visit,
                detector=legacy_detector.getId(),
                bbox=detector_bbox,
            ),
        )
        legacy_psf = legacy.getPsf()
        if legacy_psf is None:
            raise ValueError("Exposure file does not have a Psf.")
        psf = PointSpreadFunction.from_legacy(legacy_psf, bounds=detector_bbox)
        masked_image = MaskedImage.from_legacy(legacy.getMaskedImage(), unit=unit, plane_map=plane_map)
        result = VisitImage(
            image=masked_image.image.view(unit=unit),
            mask=masked_image.mask,
            variance=masked_image.variance,
            projection=projection,
            psf=psf,
        )
        result._opaque_metadata = masked_image._opaque_metadata
        return result

    @overload
    @staticmethod
    def read_legacy(
        filename: str,
        *,
        component: Literal["bbox"],
    ) -> Box: ...

    @overload
    @staticmethod
    def read_legacy(
        filename: str,
        *,
        preserve_quantization: bool = False,
        instrument: str | None = None,
        visit: int | None = None,
        component: Literal["image"],
    ) -> Image: ...

    @overload
    @staticmethod
    def read_legacy(
        filename: str,
        *,
        plane_map: Mapping[str, MaskPlane] | None = None,
        instrument: str | None = None,
        visit: int | None = None,
        component: Literal["mask"],
    ) -> Mask: ...

    @overload
    @staticmethod
    def read_legacy(
        filename: str,
        *,
        preserve_quantization: bool = False,
        instrument: str | None = None,
        visit: int | None = None,
        component: Literal["variance"],
    ) -> Image: ...

    @overload
    @staticmethod
    def read_legacy(
        filename: str,
        *,
        instrument: str | None = None,
        visit: int | None = None,
        component: Literal["projection"],
    ) -> Projection[DetectorFrame]: ...

    @overload
    @staticmethod
    def read_legacy(
        filename: str,
        *,
        component: Literal["psf"],
    ) -> PointSpreadFunction: ...

    @overload
    @staticmethod
    def read_legacy(
        filename: str,
        *,
        preserve_quantization: bool = False,
        plane_map: Mapping[str, MaskPlane] | None = None,
        instrument: str | None = None,
        visit: int | None = None,
        component: None = None,
    ) -> VisitImage: ...

    @staticmethod
    def read_legacy(
        filename: str,
        *,
        preserve_quantization: bool = False,
        plane_map: Mapping[str, MaskPlane] | None = None,
        instrument: str | None = None,
        visit: int | None = None,
        component: Literal["bbox", "image", "mask", "variance", "projection", "psf"] | None = None,
    ) -> Any:
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
            description.  If `None` (default)
            `get_legacy_visit_image_mask_planes` is used.
        instrument
            Name of the instrument.  Read from the primary header if not
            provided.
        visit
            ID of the visit.  Read from the primary header if not
            provided.
        component
            A component to read instead of the full image.
        """
        from lsst.afw.image import ExposureFitsReader

        reader = ExposureFitsReader(filename)
        if component == "bbox":
            return Box.from_legacy(reader.readBBox())
        legacy_detector = reader.readDetector()
        if legacy_detector is None:
            raise ValueError(f"Exposure file {filename!r} does not have a Detector.")
        detector_bbox = Box.from_legacy(legacy_detector.getBBox())
        if component in (None, "image", "mask", "variance", "projection"):
            legacy_wcs = reader.readWcs()
            if legacy_wcs is None:
                raise ValueError(f"Exposure file {filename!r} does not have a SkyWcs.")
        if component in (None, "psf"):
            legacy_psf = reader.readPsf()
            if legacy_psf is None:
                raise ValueError(f"Exposure file {filename!r} does not have a Psf.")
            psf = PointSpreadFunction.from_legacy(legacy_psf, bounds=detector_bbox)
            if component == "psf":
                return psf
        assert component in (None, "image", "mask", "variance", "projection"), component  # for MyPy
        with astropy.io.fits.open(filename) as hdu_list:
            primary_header = hdu_list[0].header
            if instrument is None:
                try:
                    instrument = primary_header["LSST BUTLER DATAID INSTRUMENT"]
                except LookupError:
                    raise ValueError(
                        "Instrument could not be found in butler data ID FITS headers and must be provided."
                    ) from None
            if visit is None:
                try:
                    visit = primary_header["LSST BUTLER DATAID VISIT"]
                except LookupError:
                    raise ValueError(
                        "Visit ID could not be found in butler data ID FITS headers and must be provided."
                    ) from None
            projection = Projection.from_legacy(
                legacy_wcs,
                DetectorFrame(
                    instrument=instrument,
                    visit=visit,
                    detector=legacy_detector.getId(),
                    bbox=detector_bbox,
                ),
            )
            if component == "projection":
                return projection
            if plane_map is None:
                plane_map = get_legacy_visit_image_mask_planes()
            assert component != "psf", component  # for MyPy
            from_masked_image = MaskedImage._read_legacy_hdus(
                hdu_list,
                filename,
                preserve_quantization=preserve_quantization,
                plane_map=plane_map,
                component=component,
            )
        if component is not None:
            # This is the image, mask, or variance; attach the projection
            # and return
            return from_masked_image.view(projection=projection)
        result = VisitImage(
            from_masked_image.image,
            mask=from_masked_image.mask,
            variance=from_masked_image.variance,
            projection=projection,
            psf=psf,
        )
        result._opaque_metadata = from_masked_image._opaque_metadata
        return result


class VisitImageSerializationModel[P: pydantic.BaseModel](MaskedImageSerializationModel[P]):
    """A Pydantic model used to represent a serialized `VisitImage`."""

    # Inherited attributes are duplicated because that improves the docs
    # (some limitation in the sphinx/pydantic integration), and these are
    # important docs.

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

    def deserialize_psf(self, archive: InputArchive[Any]) -> PointSpreadFunction | ArchiveReadError:
        """Finish deserializing the PSF model, or *return* any exception
        raised in the attempt.
        """
        try:
            match self.psf:
                case PiffSerializationModel():
                    return PiffWrapper.deserialize(self.psf, archive)
                case PSFExSerializationModel():
                    return PSFExWrapper.deserialize(self.psf, archive)
                case _:
                    raise ArchiveReadError("PSF model type not recognized.")
        except ArchiveReadError as err:
            return err

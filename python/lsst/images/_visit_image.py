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
import warnings
from collections.abc import Callable, Mapping, MutableMapping
from types import EllipsisType
from typing import Any, Literal, cast, overload

import astropy.io.fits
import astropy.units
import astropy.wcs
import pydantic
from astro_metadata_translator import ObservationInfo, VisitInfoTranslator

from ._geom import Box
from ._image import Image, ImageSerializationModel
from ._mask import Mask, MaskPlane, MaskSchema, MaskSerializationModel, get_legacy_visit_image_mask_planes
from ._masked_image import MaskedImage, MaskedImageSerializationModel
from ._transforms import DetectorFrame, Projection, ProjectionAstropyView, ProjectionSerializationModel
from .fits import FitsOpaqueMetadata
from .psfs import (
    GaussianPointSpreadFunction,
    GaussianPSFSerializationModel,
    PiffSerializationModel,
    PiffWrapper,
    PointSpreadFunction,
    PSFExSerializationModel,
    PSFExWrapper,
)
from .serialization import ArchiveReadError, InputArchive, MetadataValue, OutputArchive


def _obs_info_from_md(md: MutableMapping[str, Any], visit_info: Any = None) -> ObservationInfo:
    # Try to get an ObservationInfo from the primary header as if
    # it's a raw header. Else fallback.
    try:
        obs_info = ObservationInfo.from_header(md, quiet=True)
    except ValueError:
        # Not known translator. Must fall back to visit info. If we have
        # an actual VisitInfo, serialize it since we know that it will be
        # complete.
        if visit_info is not None:
            from lsst.afw.image import setVisitInfoMetadata
            from lsst.daf.base import PropertyList

            pl = PropertyList()
            setVisitInfoMetadata(pl, visit_info)
            # Merge so that we still have access to butler provenance.
            md.update(pl)

        # Try the given header looking for VisitInfo hints.
        # We get lots of warnings if nothing can be found. Currently
        # no way to disable those without capturing them.
        obs_info = ObservationInfo.from_header(md, translator_class=VisitInfoTranslator, quiet=True)
    return obs_info


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
    projection
        Projection that maps the pixel grid to the sky.  Can only be `None` if
        a projection is already attached to ``image``.
    psf
        Point-spread function model for this image, or an exception explaining
        why it could not be read (to be raised if the PSF is requested later).
    obs_info
        General information about this visit in standardized form.
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
        projection: Projection[DetectorFrame] | None = None,
        obs_info: ObservationInfo | None = None,
        psf: PointSpreadFunction | ArchiveReadError,
        metadata: dict[str, MetadataValue] | None = None,
    ):
        super().__init__(
            image,
            mask=mask,
            variance=variance,
            mask_schema=mask_schema,
            projection=projection,
            obs_info=obs_info,
            metadata=metadata,
        )
        if self.image.unit is None:
            raise TypeError("The image component of a VisitImage must have units.")
        if self.image.projection is None:
            raise TypeError("The projection component of a VisitImage cannot be None.")
        if self.image.obs_info is None:
            raise TypeError("The observation info component of a VisitImage cannot be None.")
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
    def obs_info(self) -> ObservationInfo:
        """General information about this observation in standard form.
        (`~astro_metadata_translator.ObservationInfo`).
        """
        obs_info = self.image.obs_info
        assert obs_info is not None
        return obs_info

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

    def __getitem__(self, bbox: Box | EllipsisType) -> VisitImage:
        if bbox is ...:
            return self
        super().__getitem__(bbox)
        return self._transfer_metadata(
            VisitImage(
                self.image[bbox],
                mask=self.mask[bbox],
                variance=self.variance[bbox],
                projection=self.projection,
                psf=self.psf,
                obs_info=self.obs_info,
            ),
            bbox=bbox,
        )

    def __str__(self) -> str:
        return f"VisitImage({self.image!s}, {list(self.mask.schema.names)})"

    def __repr__(self) -> str:
        return f"VisitImage({self.image!r}, mask_schema={self.mask.schema!r})"

    def copy(self) -> VisitImage:
        """Deep-copy the visit image."""
        return self._transfer_metadata(
            VisitImage(
                image=self._image.copy(),
                mask=self._mask.copy(),
                variance=self._variance.copy(),
                psf=self._psf,
                obs_info=self.obs_info,
            ),
            copy=True,
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
        serialized_psf: PiffSerializationModel | PSFExSerializationModel | GaussianPSFSerializationModel
        match self._psf:
            # MyPy is able to figure things out here with this match statement,
            # but not a single isinstance check on both types.
            case PiffWrapper():
                serialized_psf = archive.serialize_direct("psf", self._psf.serialize)
            case PSFExWrapper():
                serialized_psf = archive.serialize_direct("psf", self._psf.serialize)
            case GaussianPointSpreadFunction():
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
            obs_info=self.obs_info,
            metadata=self.metadata,
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
        return VisitImage(
            masked_image.image,
            mask=masked_image.mask,
            variance=masked_image.variance,
            psf=psf,
            projection=projection,
            obs_info=model.obs_info,
        )._finish_deserialize(model)

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
        obs_info = _obs_info_from_md(md, visit_info=legacy.info.getVisitInfo())
        instrument = _extract_or_check_header(
            "LSST BUTLER DATAID INSTRUMENT", instrument, md, obs_info.instrument, str
        )
        visit = _extract_or_check_header("LSST BUTLER DATAID VISIT", visit, md, None, int)
        unit = _extract_or_check_header(
            "BUNIT", unit, md, None, lambda x: astropy.units.Unit(x, format="fits")
        )
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
            obs_info=obs_info,
        )

        result._opaque_metadata = opaque_fits_metadata
        return result

    @overload  # type: ignore[override]
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
        component: Literal["obs_info"],
    ) -> ObservationInfo: ...

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
    def read_legacy(  # type: ignore[override]
        filename: str,
        *,
        preserve_quantization: bool = False,
        plane_map: Mapping[str, MaskPlane] | None = None,
        instrument: str | None = None,
        visit: int | None = None,
        component: Literal["bbox", "image", "mask", "variance", "projection", "psf", "obs_info"]
        | None = None,
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
        legacy_wcs = None
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
        assert component in (None, "image", "mask", "variance", "projection", "obs_info"), (
            component
        )  # for MyPy
        with astropy.io.fits.open(filename) as hdu_list:
            primary_header = hdu_list[0].header
            obs_info = _obs_info_from_md(primary_header)
            if component == "obs_info":
                return obs_info
            instrument = _extract_or_check_header(
                "LSST BUTLER DATAID INSTRUMENT", instrument, primary_header, obs_info.instrument, str
            )
            visit = _extract_or_check_header("LSST BUTLER DATAID VISIT", visit, primary_header, None, int)
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
            obs_info=obs_info,
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
    psf: PiffSerializationModel | PSFExSerializationModel | GaussianPSFSerializationModel | Any = (
        pydantic.Field(union_mode="left_to_right", description="PSF model for the image.")
    )
    obs_info: ObservationInfo = pydantic.Field(
        description="Standardized description of visit metadata",
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
                case GaussianPSFSerializationModel():
                    return GaussianPointSpreadFunction.deserialize(self.psf, archive)
                case _:
                    raise ArchiveReadError("PSF model type not recognized.")
        except ArchiveReadError as err:
            return err


def _extract_or_check_value[T](
    key: str,
    given_value: T | None,
    *sources: tuple[str, T | None],
) -> T:
    # Compare given value against multiple sources. If given value is not
    # supplied return the first non-None value in the reference sources.
    if given_value is not None:
        for source_name, source_value in sources:
            if source_value is not None and source_value != given_value:
                raise ValueError(
                    f"Given value {given_value!r} does not match {source_value!r} from {source_name}."
                )
            if source_value is not None:
                # Only check the first non-None source rather than checking
                # all supplied values.
                break
        return given_value

    for _, source_value in sources:
        if source_value is not None:
            return source_value

    raise ValueError(f"No value found for {key}.")


def _extract_or_check_header[T](
    key: str, given_value: T | None, header: Any, obs_info_value: T | None, coerce: Callable[[Any], T]
) -> T:
    hdr_value: T | None = None
    if (hdr_raw_value := header.get(key)) is not None:
        hdr_value = coerce(hdr_raw_value)
    return _extract_or_check_value(
        key, given_value, ("ObservationInfo", obs_info_value), (f"header key {key}", hdr_value)
    )

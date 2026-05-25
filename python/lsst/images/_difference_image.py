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

__all__ = ("DifferenceImage", "DifferenceImageSerializationModel")

from collections.abc import Mapping
from types import EllipsisType
from typing import TYPE_CHECKING, Any, ClassVar, Literal, overload

import astropy.io.fits
import astropy.units
import astropy.wcs
import pydantic
from astro_metadata_translator import ObservationInfo

from ._backgrounds import BackgroundMap
from ._geom import Bounds, Box
from ._image import Image
from ._mask import Mask, MaskPlane, MaskSchema, get_legacy_difference_image_mask_planes
from ._observation_summary_stats import ObservationSummaryStats
from ._transforms import DetectorFrame, Projection
from ._visit_image import VisitImage, VisitImageSerializationModel
from .aperture_corrections import (
    ApertureCorrectionMap,
)
from .cameras import Detector
from .fields import Field
from .psfs import (
    PointSpreadFunction,
)
from .serialization import ArchiveReadError, InputArchive, InvalidParameterError, MetadataValue, OutputArchive

if TYPE_CHECKING:
    try:
        from lsst.afw.image import Exposure as LegacyExposure
    except ImportError:
        type LegacyExposure = Any  # type: ignore[no-redef]


class DifferenceImage(VisitImage):
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
    bounds
        The region where this image's pixels and other properties are valid.
        If not provided, the bounding box of the image is used.  Other
        components (``psf``, ``projection``, ``aperture_corrections``, etc.)
        are assumed to have their own bounds which may or may not be the same
        as the image bounds.  If ``bounds`` extends beyond the image bounding
        box, the intersection between ``bounds`` and the image bounding box
        is used instead.
    obs_info
        General information about this visit in standardized form.
    summary_stats
        Summary statistics associated with this visit.  Initialized to default
        values if not provided.
    photometric_scaling
        Field that can be used to multiply a post-ISR image units to yield
        calibrated image units.  This may be a scaling that was already
        applied (so dividing by it will recover the post-ISR units) or a
        scaling that has not been applied, depending on ``image.unit``.
    psf
        Point-spread function model for this image, or an exception explaining
        why it could not be read (to be raised if the PSF is requested later).
    detector
        Geometry and electronic information for the detector attached to this
        image.
    aperture_corrections : `dict` [`str`, `~fields.BaseField`]
        Mapping from photometry algorithm name to the aperture correction for
        that algorithm.
    backgrounds
        Background models associated with this image.
    band
        Name of the passband the image was observed with (this is a shorter,
        less specific version of ``obs_info.physical_filter``).
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
        bounds: Bounds | None = None,
        obs_info: ObservationInfo | None = None,
        summary_stats: ObservationSummaryStats | None = None,
        photometric_scaling: Field | None = None,
        psf: PointSpreadFunction | ArchiveReadError,
        detector: Detector,
        aperture_corrections: ApertureCorrectionMap | None = None,
        backgrounds: BackgroundMap | None = None,
        band: str,
        metadata: dict[str, MetadataValue] | None = None,
    ):
        super().__init__(
            image,
            mask=mask,
            variance=variance,
            mask_schema=mask_schema,
            projection=projection,
            bounds=bounds,
            obs_info=obs_info,
            summary_stats=summary_stats,
            photometric_scaling=photometric_scaling,
            psf=psf,
            detector=detector,
            aperture_corrections=aperture_corrections,
            backgrounds=backgrounds,
            band=band,
            metadata=metadata,
        )

    @staticmethod
    def _from_visit_image(visit_image: VisitImage) -> DifferenceImage:
        return visit_image._transfer_metadata(
            DifferenceImage(
                visit_image.image,
                mask=visit_image.mask,
                variance=visit_image.variance,
                projection=visit_image.projection,
                bounds=visit_image.bounds,
                obs_info=visit_image.obs_info,
                summary_stats=visit_image.summary_stats,
                photometric_scaling=visit_image.photometric_scaling,
                psf=visit_image._psf,  # get private attr to avoid triggering on ArchiveReadError early.
                detector=visit_image.detector,
                aperture_corrections=visit_image.aperture_corrections,
                backgrounds=visit_image.backgrounds,
                band=visit_image.band,
            ),
        )

    def __getitem__(self, bbox: Box | EllipsisType) -> DifferenceImage:
        if bbox is ...:
            return self
        return self._from_visit_image(super().__getitem__(bbox))

    def __str__(self) -> str:
        return f"DifferenceImage({self.image!s}, {list(self.mask.schema.names)})"

    def __repr__(self) -> str:
        return f"DifferenceImage({self.image!r}, mask_schema={self.mask.schema!r})"

    def copy(self, *, copy_detector: bool = False) -> DifferenceImage:
        """Deep-copy the difference image.

        Parameters
        ----------
        copy_detector
            Whether to deep-copy the `detector` attribute.
        """
        return self._from_visit_image(super().copy(copy_detector=copy_detector))

    def convert_unit(
        self,
        unit: astropy.units.UnitBase = astropy.units.nJy,
        copy: Literal["as-needed"] | bool = True,
        copy_detector: bool = False,
    ) -> DifferenceImage:
        """Return an equivalent image with different pixel units.

        Parameters
        ----------
        unit
            The unit to transform to.  This may be any of the following:

            - any unit directly relatable to the current units via Astropy;
            - any unit relatable to the product of the current units with the
              `photometric_scaling` (i.e. if the current image is in
              instrumental units but we know how to calibrate them)
            - any unit relatable to the quotient of the current units with the
              `photometric_scaling` (i.e. if the current image is in
              calibrated units and we want to revert back to instrumental
              units).
        copy
            Whether to copy the images and other components.  If `True`, all
            components that aren't controlled by some other argument will
            always be deep-copied.  If `False`, the operation will fail if the
            image is not already in the right units.  If ``as-needed``, only
            the image and variance will be copied, and only if they are not
            already in the right units.
        copy_detector
            Whether to deep-copy the `detector` attribute.

        Returns
        -------
        `DifferenceImage`
            An image with the given units.
        """
        return self._from_visit_image(super().convert_unit(unit, copy=copy, copy_detector=copy_detector))

    def serialize(self, archive: OutputArchive[Any]) -> DifferenceImageSerializationModel:
        return self._serialize_impl(DifferenceImageSerializationModel, archive)

    @staticmethod
    def _get_archive_tree_type[P: pydantic.BaseModel](
        pointer_type: type[P],
    ) -> type[DifferenceImageSerializationModel[P]]:
        """Return the serialization model type for this object for an archive
        type that uses the given pointer type.
        """
        return DifferenceImageSerializationModel[pointer_type]  # type: ignore

    # write_fits and read_fits inherited from MaskedImage.

    @staticmethod
    def from_legacy(
        legacy: LegacyExposure,
        *,
        unit: astropy.units.UnitBase | None = None,
        plane_map: Mapping[str, MaskPlane] | None = None,
        instrument: str | None = None,
        visit: int | None = None,
    ) -> DifferenceImage:
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
            plane_map = get_legacy_difference_image_mask_planes()
        return DifferenceImage._from_visit_image(
            VisitImage.from_legacy(legacy, unit=unit, plane_map=plane_map, instrument=instrument, visit=visit)
        )

    def to_legacy(
        self, *, copy: bool | None = None, plane_map: Mapping[str, MaskPlane] | None = None
    ) -> LegacyExposure:
        """Convert to an `lsst.afw.image.Exposure` instance.

        Parameters
        ----------
        copy
            If `True`, always copy the image and variance pixel data.
            If `False`, return a view, and raise `TypeError` if the pixel data
            is read-only (this is not supported by afw).  If `None`, only copy
            if the pixel data is read-only.  Mask pixel data is always copied.
        plane_map
            A mapping from legacy mask plane name to the new plane name and
            description.  If `None` (default),
            `get_legacy_visit_image_mask_planes` is used.
        """
        if plane_map is None:
            plane_map = get_legacy_difference_image_mask_planes()
        return super().to_legacy(copy=copy, plane_map=plane_map)

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
        component: Literal["detector"],
    ) -> Detector: ...

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
        component: Literal["photometric_scaling"],
    ) -> Field | None: ...

    @overload
    @staticmethod
    def read_legacy(
        filename: str,
        *,
        component: Literal["summary_stats"],
    ) -> ObservationSummaryStats: ...

    @overload
    @staticmethod
    def read_legacy(
        filename: str,
        *,
        component: Literal["aperture_corrections"],
    ) -> ApertureCorrectionMap: ...

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
    ) -> DifferenceImage: ...

    @staticmethod
    def read_legacy(  # type: ignore[override]
        filename: str,
        *,
        preserve_quantization: bool = False,
        plane_map: Mapping[str, MaskPlane] | None = None,
        instrument: str | None = None,
        visit: int | None = None,
        component: Literal[
            "bbox",
            "image",
            "mask",
            "variance",
            "projection",
            "psf",
            "detector",
            "photometric_scaling",
            "obs_info",
            "summary_stats",
            "aperture_corrections",
        ]
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
        if plane_map is None:
            plane_map = get_legacy_difference_image_mask_planes()
        result = VisitImage.read_legacy(
            filename,
            preserve_quantization=preserve_quantization,
            plane_map=plane_map,
            instrument=instrument,
            visit=visit,
            component=component,  # type: ignore[arg-type]
        )
        if component is None:
            return DifferenceImage._from_visit_image(result)
        return result


class DifferenceImageSerializationModel[P: pydantic.BaseModel](VisitImageSerializationModel[P]):
    """A Pydantic model used to represent a serialized `DifferenceImage`."""

    SCHEMA_NAME: ClassVar[str] = "difference_image"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"
    MIN_READ_VERSION: ClassVar[int] = 1

    def deserialize(
        self, archive: InputArchive[Any], *, bbox: Box | None = None, **kwargs: Any
    ) -> DifferenceImage:
        if kwargs:
            raise InvalidParameterError(f"Unrecognized parameters for DifferenceImage: {set(kwargs.keys())}.")
        return DifferenceImage._from_visit_image(super().deserialize(archive, bbox=bbox))

    def deserialize_component(self, component: str, archive: InputArchive[Any], **kwargs: Any) -> Any:
        if kwargs and component not in ("image", "mask", "variance"):
            raise InvalidParameterError(
                f"Unsupported parameters for DifferenceImage component {component}: {set(kwargs.keys())}."
            )
        return super().deserialize_component(component, archive)

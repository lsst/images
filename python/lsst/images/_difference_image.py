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

__all__ = ("DifferenceImage", "DifferenceImageSerializationModel", "DifferenceImageTemplateInfo")

import logging
import math
import uuid
from collections.abc import Iterable, Mapping
from types import EllipsisType
from typing import TYPE_CHECKING, Any, ClassVar, Literal, cast

import astropy.units
import pydantic
from astro_metadata_translator import ObservationInfo

from ._backgrounds import BackgroundMap
from ._geom import Bounds, Box
from ._image import Image
from ._mask import Mask, MaskPlane, MaskSchema, get_legacy_difference_image_mask_planes
from ._observation_summary_stats import ObservationSummaryStats
from ._polygon import Polygon
from ._transforms import DetectorFrame, SkyProjection, TractFrame, Transform
from ._visit_image import VisitImage, VisitImageSerializationModel
from .aperture_corrections import (
    ApertureCorrectionMap,
)
from .cameras import Detector
from .convolution_kernels import ConvolutionKernel, ConvolutionKernelSerializationModel
from .fields import Field
from .psfs import (
    PointSpreadFunction,
)
from .serialization import (
    ArchiveReadError,
    InputArchive,
    InvalidParameterError,
    MetadataValue,
    OutputArchive,
)

if TYPE_CHECKING:
    from lsst.daf.butler import DataId

    try:
        from lsst.afw.geom import SkyWcs as LegacySkyWcs
        from lsst.afw.image import Exposure as LegacyExposure
        from lsst.geom import Box2I as LegacyBox2I
        from lsst.meas.algorithms import CoaddPsf as LegacyCoaddPsf
    except ImportError:
        type LegacyBox2I = Any  # type: ignore[no-redef]
        type LegacyExposure = Any  # type: ignore[no-redef]
        type LegacyCoaddPsf = Any  # type: ignore[no-redef]
        type LegacySkyWcs = Any  # type: ignore[no-redef]


class DifferenceImage(VisitImage):
    """An image that is the PSF-matched difference of two other images.

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
        Values default to ``1.0``.  Any attached sky_projection is replaced
        (possibly by `None`).
    mask_schema
        Schema for the mask plane.  Must be provided if and only if ``mask`` is
        not provided.
    sky_projection
        Projection that maps the pixel grid to the sky.  Can only be `None` if
        a ``sky_projection`` is already attached to ``image``.
    bounds
        The region where this image's pixels and other properties are valid.
        If not provided, the bounding box of the image is used.  Other
        components (``psf``, ``sky_projection``, ``aperture_corrections``,
        etc.) are assumed to have their own bounds which may or may not be the
        same as the image bounds.  If ``bounds`` extends beyond the image
        bounding box, the intersection between ``bounds`` and the image
        bounding box is used instead.
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
    kernel
        The convolution kernel used to match the (warped) template to the
        science image.
    templates
        Information about the template coadds that went into this difference
        image.
    metadata
        Arbitrary flexible metadata to associate with the image.

    Notes
    -----
    This class assumes that the difference has been performed on the pixel
    grid of the 'science image' (i.e. a single observation, like `VisitImage`),
    and most of the attributes of `DifferenceImage` correspond to the science
    image.  The 'template image' is assumed to be comprised of one or more
    resampled coadd images stitched together.

    The `DifferenceImage` class can also be used to represent the stitched
    template itself; while this makes the naming a bit confusing, the type has
    the right state to play this role.
    """

    def __init__(
        self,
        image: Image,
        *,
        mask: Mask | None = None,
        variance: Image | None = None,
        mask_schema: MaskSchema | None = None,
        sky_projection: SkyProjection[DetectorFrame] | None = None,
        bounds: Bounds | None = None,
        obs_info: ObservationInfo | None = None,
        summary_stats: ObservationSummaryStats | None = None,
        photometric_scaling: Field | None = None,
        psf: PointSpreadFunction | ArchiveReadError,
        detector: Detector,
        aperture_corrections: ApertureCorrectionMap | None = None,
        backgrounds: BackgroundMap | None = None,
        band: str,
        kernel: ConvolutionKernel | None = None,
        templates: Iterable[DifferenceImageTemplateInfo] | None = None,
        metadata: dict[str, MetadataValue] | None = None,
    ) -> None:
        super().__init__(
            image,
            mask=mask,
            variance=variance,
            mask_schema=mask_schema,
            sky_projection=sky_projection,
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
        self._kernel = kernel
        self._templates = list(templates) if templates is not None else None

    @staticmethod
    def _from_visit_image(
        visit_image: VisitImage,
        kernel: ConvolutionKernel | None,
        templates: Iterable[DifferenceImageTemplateInfo] | None,
    ) -> DifferenceImage:
        return visit_image._transfer_metadata(
            DifferenceImage(
                visit_image.image,
                mask=visit_image.mask,
                variance=visit_image.variance,
                sky_projection=visit_image.sky_projection,
                bounds=visit_image.bounds,
                obs_info=visit_image.obs_info,
                summary_stats=visit_image.summary_stats,
                photometric_scaling=visit_image.photometric_scaling,
                psf=visit_image._psf,  # get private attr to avoid triggering on ArchiveReadError early.
                detector=visit_image.detector,
                aperture_corrections=visit_image.aperture_corrections,
                backgrounds=visit_image.backgrounds,
                kernel=kernel,
                templates=templates,
                band=visit_image.band,
            ),
        )

    @property
    def kernel(self) -> ConvolutionKernel:
        """The convolution kernel used to match the (warped) template
        to the science image (`.convolution_kernels.ConvolutionKernel`).
        """
        if self._kernel is None:
            raise AttributeError("This difference image does not have a kernel attached.")
        return self._kernel

    @kernel.setter
    def kernel(self, kernel: ConvolutionKernel) -> None:
        self._kernel = kernel

    @kernel.deleter
    def kernel(self) -> None:
        self._kernel = None

    @property
    def templates(self) -> list[DifferenceImageTemplateInfo]:
        """Information about the template coadds that went into this
        difference image (`list` [`DifferenceImageTemplateInfo`]).
        """
        if self._templates is None:
            raise AttributeError("This difference image does not have any template information attached.")
        return self._templates

    @templates.setter
    def templates(self, templates: Iterable[DifferenceImageTemplateInfo]) -> None:
        self._templates = list(templates)

    @templates.deleter
    def templates(self) -> None:
        self._templates = None

    def __getitem__(self, bbox: Box | EllipsisType) -> DifferenceImage:
        if bbox is ...:
            return self
        return self._from_visit_image(
            super().__getitem__(bbox), kernel=self._kernel, templates=self._templates
        )

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
        return self._from_visit_image(
            super().copy(copy_detector=copy_detector), kernel=self._kernel, templates=self._templates
        )

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
        return self._from_visit_image(
            super().convert_unit(unit, copy=copy, copy_detector=copy_detector),
            kernel=self._kernel,
            templates=self._templates,
        )

    def serialize(self, archive: OutputArchive[Any]) -> DifferenceImageSerializationModel[Any]:
        result = self._serialize_impl(DifferenceImageSerializationModel, archive)
        if self._kernel is not None:
            result.kernel = archive.serialize_direct("kernel", self._kernel.serialize)
        else:
            result.kernel = None
        result.templates = self._templates
        return result

    @staticmethod
    def _get_archive_tree_type[P: pydantic.BaseModel](
        pointer_type: type[P],
    ) -> type[DifferenceImageSerializationModel[P]]:
        """Return the serialization model type for this object for an archive
        type that uses the given pointer type.
        """
        return DifferenceImageSerializationModel[pointer_type]  # type: ignore

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
            VisitImage.from_legacy(
                legacy, unit=unit, plane_map=plane_map, instrument=instrument, visit=visit
            ),
            kernel=None,
            templates=None,
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
            "sky_projection",
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
            component=component,
        )
        if component is None:
            return DifferenceImage._from_visit_image(result, kernel=None, templates=None)
        return result


class DifferenceImageTemplateInfo(pydantic.BaseModel, ser_json_inf_nan="constants"):
    """Information about how a template image contributed to a difference
    image.
    """

    skymap: str = pydantic.Field(description="Name of the skymap that defines the tract/patch tiling.")
    tract: int = pydantic.Field(description="ID of the tract (each tract is a different projection).")
    patch: int = pydantic.Field(
        description="ID of the patch (all patches within a tract share a projection)."
    )
    dataset_id: uuid.UUID = pydantic.Field(
        description="Universally unique butler identifier for this template.",
    )
    dataset_run: str = pydantic.Field(description="Name of the butler RUN collection for this template.")
    bounds: Polygon = pydantic.Field(
        description=(
            "The approximate intersection of the template and the science image, "
            "in the science image's pixel coordinate system."
        )
    )
    psf_shape_xx: float = pydantic.Field(description="Second moment of the effective PSF of the template.")
    psf_shape_yy: float = pydantic.Field(description="Second moment of the effective PSF of the template.")
    psf_shape_xy: float = pydantic.Field(description="Second moment of the effective PSF of the template.")
    psf_shape_flag: bool = pydantic.Field(
        description="Flag set if the second moments of the effective template PSF could not be computed."
    )

    @staticmethod
    def from_legacy(
        detector_frame: DetectorFrame,
        legacy_template_psf: LegacyCoaddPsf,
        legacy_template_metadata: Mapping[str, Any],
        coadd_data_ids_by_uuid: Mapping[uuid.UUID, DataId],
        coadd_dataset_type: str = "template_coadd",
        log: logging.Logger | None = None,
    ) -> list[DifferenceImageTemplateInfo]:
        """Construct a list of template information structs from information
        stored in a legacy stitched template image.

        Parameters
        ----------
        detector_frame
            Coordinate system and bounding box of the science image.
        legacy_template_psf
            The lazy-evaluation PSF model for the stitched template; used to
            extract the tract and patch IDs of the coadds actually used and
            their PSF models.
        legacy_template_metadata
            The FITS-style metadata of the stitched template; used to extract
            butler UUIDs and RUN collection names for all *potential* input
            coadds.
        coadd_data_ids_by_uuid
            A mapping from butler dataset ID to ``{tract, patch, band}`` data
            ID for all coadds that may have contributed to the template.  May
            be a much larger superset of the needed datasets.
        coadd_dataset_type
            The name of the coadd template dataset type.
        log
            Logger to use for diagnostic messages.
        """
        from lsst.afw.geom import makeWcsPairTransform

        n_inputs = legacy_template_metadata["LSST BUTLER N_INPUTS"]
        butler_info: dict[tuple[int, int], tuple[uuid.UUID, str]] = {}
        skymap: str | None = None
        for n in range(n_inputs):
            if legacy_template_metadata[f"LSST BUTLER INPUT {n} DATASETTYPE"] == coadd_dataset_type:
                input_id = uuid.UUID(legacy_template_metadata[f"LSST BUTLER INPUT {n} ID"])
                input_run = legacy_template_metadata[f"LSST BUTLER INPUT {n} RUN"]
                input_data_id = coadd_data_ids_by_uuid[input_id]
                if skymap is None:
                    skymap = cast(str, input_data_id["skymap"])
                elif skymap != input_data_id["skymap"]:
                    raise RuntimeError("Cannot handle multiple skymaps in the inputs to a single template.")
                butler_info[cast(int, input_data_id["tract"]), cast(int, input_data_id["patch"])] = (
                    input_id,
                    input_run,
                )
        result: list[DifferenceImageTemplateInfo] = []
        # A "component" of this PSF is an input {tract, patch} coadd.
        for n in range(legacy_template_psf.getComponentCount()):
            tract = legacy_template_psf.getTract(n)
            patch = legacy_template_psf.getPatch(n)
            dataset_id, dataset_run = butler_info[tract, patch]
            patch_bbox = Box.from_legacy(legacy_template_psf.getBBox(n))
            coadd_frame = TractFrame(
                skymap=skymap,
                tract=tract,
                # This bbox is supposed to be the full tract bbox, but this
                # frame is just a temporary and we don't have access to that.
                # (If this ever becomes not-a-temporary, we could add a skymap
                # argument).
                bbox=patch_bbox,
            )
            detector_to_coadd = Transform.from_legacy(
                makeWcsPairTransform(
                    # CoaddPsf method names did not anticipate being used for
                    # detector-level templates, so this is confusing:
                    legacy_template_psf.getCoaddWcs(),  # this is the template_detector WCS!
                    legacy_template_psf.getWcs(n),  # this is the template_coadd WCS!
                ),
                detector_frame,
                coadd_frame,
            )
            coadd_to_detector = detector_to_coadd.inverted()
            # We transform the detector bbox to each coadd frame, do the
            # intersection there, and then transform the intersection back to
            # the detector frame, because we do not trust detector WCSs beyond
            # the detector bounding box; they can be polynomials that
            # extrapolate badly. Coadd WCSs in contrast are simple projections.
            tmp_bounds = (
                Polygon.from_box(detector_frame.bbox).transform(detector_to_coadd).intersection(patch_bbox)
            ).transform(coadd_to_detector)
            # Unfortunately doing the intersection in the coadd coordinate
            # system means the transformed intersection might not quite be
            # contained by the detector bounding box, due to floating-point
            # round-off error.  Intersect one more time to tidy it up.
            bounds = tmp_bounds.intersection(detector_frame.bbox)
            assert isinstance(bounds, Polygon), (
                "The operations above should not change the region's fundamental topology."
            )
            try:
                psf_shape = legacy_template_psf.computeShape(bounds.centroid.to_legacy_float_point())
            except Exception:
                if log is not None:
                    log.exception(
                        "Could not compute PSF shape for template coadd with tract=%s, patch=%s", tract, patch
                    )
                else:
                    raise
                psf_shape = None
            result.append(
                DifferenceImageTemplateInfo(
                    skymap=skymap,
                    tract=tract,
                    patch=patch,
                    dataset_id=dataset_id,
                    dataset_run=dataset_run,
                    bounds=bounds,
                    psf_shape_xx=psf_shape.getIxx() if psf_shape is not None else math.nan,
                    psf_shape_yy=psf_shape.getIyy() if psf_shape is not None else math.nan,
                    psf_shape_xy=psf_shape.getIxy() if psf_shape is not None else math.nan,
                    psf_shape_flag=psf_shape is None,
                )
            )
        result.sort(key=lambda item: (item.tract, item.patch))
        return result


class DifferenceImageSerializationModel[P: pydantic.BaseModel](VisitImageSerializationModel[P]):
    """A Pydantic model used to represent a serialized `DifferenceImage`."""

    SCHEMA_NAME: ClassVar[str] = "difference_image"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0.dev0"
    MIN_READ_VERSION: ClassVar[int] = 1
    PUBLIC_TYPE: ClassVar[type] = DifferenceImage

    kernel: ConvolutionKernelSerializationModel | None = pydantic.Field(
        description="The convolution kernel used to match the (warped) template to the science image."
    )
    templates: list[DifferenceImageTemplateInfo] | None = pydantic.Field(
        description="Information about the template coadds that went into this difference image"
    )

    def deserialize(
        self, archive: InputArchive[Any], *, bbox: Box | None = None, **kwargs: Any
    ) -> DifferenceImage:
        if kwargs:
            raise InvalidParameterError(f"Unrecognized parameters for DifferenceImage: {set(kwargs.keys())}.")
        kernel = self.kernel.deserialize(archive) if self.kernel is not None else None
        return DifferenceImage._from_visit_image(
            super().deserialize(archive, bbox=bbox), kernel=kernel, templates=self.templates
        )

    def deserialize_component(self, component: str, archive: InputArchive[Any], **kwargs: Any) -> Any:
        if kwargs and component not in ("image", "mask", "variance", "masked_image"):
            raise InvalidParameterError(
                f"Unsupported parameters for DifferenceImage component {component}: {set(kwargs.keys())}."
            )
        return super().deserialize_component(component, archive, **kwargs)

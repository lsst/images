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

__all__ = (
    "Detector",
    "DetectorAttributes",
    "DetectorSerializationModel",
    "DetectorType",
    "Orientation",
)

import builtins
import enum
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, ClassVar, final

import astropy.units
import numpy as np
import pydantic

from .._geom import Box
from .._transforms import (
    DetectorFrame,
    FieldAngleFrame,
    FocalPlaneFrame,
    Transform,
)
from ..describe import DescribableMixin, DescribeOptions, FieldRole, Report, ReportField
from ..serialization import (
    ArchiveReadError,
    ArchiveTree,
    InputArchive,
    InvalidParameterError,
    OutputArchive,
    Quantity,
)
from ._amplifier import Amplifier
from ._camera_frame_set import CameraFrameSet, CameraFrameSetSerializationModel

if TYPE_CHECKING:
    try:
        from lsst.afw.cameraGeom import Detector as LegacyDetector
        from lsst.afw.cameraGeom import DetectorType as LegacyDetectorType
        from lsst.afw.cameraGeom import Orientation as LegacyOrientation
    except ImportError:
        type LegacyDetector = Any  # type: ignore[no-redef]
        type LegacyDetectorType = Any  # type: ignore[no-redef]
        type LegacyOrientation = Any  # type: ignore[no-redef]


class DetectorType(enum.StrEnum):
    """Enumeration of the types of a detector."""

    SCIENCE = "SCIENCE"
    FOCUS = "FOCUS"
    GUIDER = "GUIDER"
    WAVEFRONT = "WAVEFRONT"

    def to_legacy(self) -> LegacyDetectorType:
        """Convert to `lsst.afw.cameraGeom.DetectorType`."""
        from lsst.afw.cameraGeom import DetectorType as LegacyDetectorType

        return getattr(LegacyDetectorType, self.value)

    @classmethod
    def from_legacy(cls, legacy_detector_type: LegacyDetectorType) -> DetectorType:
        """Convert from `lsst.afw.cameraGeom.DetectorType`.

        Parameters
        ----------
        legacy_detector_type
            Legacy detector type to convert.
        """
        return getattr(cls, legacy_detector_type.name)


@final
class Orientation(pydantic.BaseModel, ser_json_inf_nan="constants"):
    """A struct that represents the nominal position and rotation of a
    detector within a camera focal plane.
    """

    focal_plane_x: float = pydantic.Field(description="Focal plane X coordinate of the reference position.")
    focal_plane_y: float = pydantic.Field(description="Focal plane Y coordinate of the reference position.")
    focal_plane_z: float = pydantic.Field(description="Focal plane Z coordinate of the reference position.")
    pixel_reference_x: float = pydantic.Field(0.5, description="Pixel X coordinate of the reference point.")
    pixel_reference_y: float = pydantic.Field(0.5, description="Pixel Y coordinate of the reference point.")
    yaw: Quantity = pydantic.Field(
        default_factory=lambda: 0.0 * astropy.units.radian,
        description="Rotation about the Z axis.",
    )
    pitch: Quantity = pydantic.Field(
        default_factory=lambda: 0.0 * astropy.units.radian,
        description="Rotation about the Y axis (as defined after applying 'yaw').",
    )
    roll: Quantity = pydantic.Field(
        default_factory=lambda: 0.0 * astropy.units.radian,
        description="Rotation about the X axis (as defined after applying 'yaw' and 'pitch').",
    )

    def to_legacy(self) -> LegacyOrientation:
        """Convert to `lsst.afw.cameraGeom.Orientation`."""
        from lsst.afw.cameraGeom import Orientation as LegacyOrientation
        from lsst.geom import Point2D, Point3D, radians

        return LegacyOrientation(
            Point3D(self.focal_plane_x, self.focal_plane_y, self.focal_plane_z),
            Point2D(self.pixel_reference_x, self.pixel_reference_y),
            self.yaw.to_value(astropy.units.radian) * radians,
            self.pitch.to_value(astropy.units.radian) * radians,
            self.roll.to_value(astropy.units.radian) * radians,
        )

    @staticmethod
    def from_legacy(legacy_orientation: LegacyOrientation) -> Orientation:
        """Convert from `lsst.afw.cameraGeom.Orientation`.

        Parameters
        ----------
        legacy_orientation
            Legacy orientation to convert.
        """
        focal_plane_x, focal_plane_y, focal_plane_z = legacy_orientation.getFpPosition3()
        pixel_reference_x, pixel_reference_y = legacy_orientation.getReferencePoint()
        return Orientation(
            focal_plane_x=focal_plane_x,
            focal_plane_y=focal_plane_y,
            focal_plane_z=focal_plane_z,
            pixel_reference_x=pixel_reference_x,
            pixel_reference_y=pixel_reference_y,
            yaw=legacy_orientation.getYaw().asRadians() * astropy.units.radian,
            pitch=legacy_orientation.getPitch().asRadians() * astropy.units.radian,
            roll=legacy_orientation.getRoll().asRadians() * astropy.units.radian,
        )


@final
class DetectorAttributes(pydantic.BaseModel, ser_json_inf_nan="constants"):
    """Struct holding the plain-old-data attributes of a detector."""

    name: str = pydantic.Field(description="Name of the detector.")
    id: int = pydantic.Field(description="ID of the detector.")
    type: DetectorType = pydantic.Field(description="Enumerated type of the detector.")
    serial: str = pydantic.Field(description="Serial number for the detector.")
    bbox: Box = pydantic.Field(
        description="Bounding box of the detector's science data region after amplifier assembly."
    )
    orientation: Orientation = pydantic.Field(description="Nominal position and rotation of the detector.")
    pixel_size: float = pydantic.Field(
        description="Nominal size of a pixel (assumed square) in focal plane coordinate units."
    )
    physical_type: str = pydantic.Field(
        description=(
            "Vendor name or technology type for this detector "
            "(may have a different interpretation for different cameras)."
        )
    )


@final
class Detector(DescribableMixin):
    """Information about a detector in a camera.

    Parameters
    ----------
    attributes
        Identifying attributes and metadata for the detector.
    amplifiers
        Amplifiers that make up the detector.
    frames
        Coordinate systems and transforms for the camera.
    visit
        Visit number whose geometry to use, or `None` for the nominal
        detector geometry.
    """

    def __init__(
        self,
        attributes: DetectorAttributes,
        amplifiers: Iterable[Amplifier],
        frames: CameraFrameSet,
        visit: int | None = None,
    ) -> None:
        self._attributes = attributes
        self._amplifiers = list(amplifiers)
        self._frames = frames
        self._frame = frames.detector(attributes.id, visit=visit)

    def __eq__(self, other: object) -> bool:
        if type(other) is not Detector:
            return NotImplemented
        return (
            self._attributes == other._attributes
            and self._amplifiers == other._amplifiers
            and self._frames == other._frames
            and self.visit == other.visit
        )

    __hash__ = None  # type: ignore[assignment]

    @property
    def instrument(self) -> str:
        """The name of the instrument this detector belongs to (`str`)."""
        return self._frame.instrument

    @property
    def visit(self) -> int | None:
        """The ID of the visit this detector is associated with (`int` or
        `None`).
        """
        return self._frame.visit

    @property
    def name(self) -> str:
        """Name of the detector (`str`)."""
        return self._attributes.name

    @property
    def id(self) -> int:
        """ID of the detector (`int`)."""
        return self._attributes.id

    @property
    def type(self) -> DetectorType:
        """Enumerated type of the detector (`DetectorType`)."""
        return self._attributes.type

    @property
    def serial(self) -> str:
        """Serial number for the detector (`str`)."""
        return self._attributes.serial

    @property
    def bbox(self) -> Box:
        """Bounding box of the detector's science data region after amplifier
        assembly (`.Box`).
        """
        return self._attributes.bbox

    @property
    def orientation(self) -> Orientation:
        """Nominal position and rotation of the detector
        (`Orientation`).
        """
        return self._attributes.orientation

    @property
    def pixel_size(self) -> float:
        """Nominal size of a pixel (assumed square) in focal plane coordinate
        units (`float`).
        """
        return self._attributes.pixel_size

    @property
    def physical_type(self) -> str:
        """Vendor name or technology type for this detector (`str`).

        This may have a different interpretation for different cameras.
        """
        return self._attributes.physical_type

    @property
    def frame(self) -> DetectorFrame:
        """The coordinate system of this detector's trimmed, assembled pixel
        grid (`.DetectorFrame`).
        """
        return self._frame

    @property
    def to_focal_plane(self) -> Transform[DetectorFrame, FocalPlaneFrame]:
        """The transform from pixels to focal-plane coordinates
        (`.Transform` [`.DetectorFrame`, `.FocalPlaneFrame`]).
        """
        return self._frames[self._frame, self._frames.focal_plane(self.visit)]

    @property
    def to_field_angle(self) -> Transform[DetectorFrame, FieldAngleFrame]:
        """The transform from pixels to field angle coordinates
        (`.Transform` [`.DetectorFrame`, `.FieldAngleFrame`]).
        """
        return self._frames[self._frame, self._frames.field_angle(self.visit)]

    @property
    def amplifiers(self) -> list[Amplifier]:
        """The amplifiers of this detectors (`list` [`Amplifier`])."""
        return self._amplifiers

    def _describe(self, options: DescribeOptions = DescribeOptions(), /) -> Report:
        """Return a `Report` describing this detector.

        Parameters
        ----------
        options : `DescribeOptions`, optional
            Unused; accepted for interface compatibility.
        """
        return Report(
            type_name="Detector",
            summary=f"Detector {self.name!r} ({self.instrument})",
            fields=[
                ReportField(label="instrument", value=self.instrument, role=FieldRole.DERIVED),
                ReportField(label="name", value=self.name, role=FieldRole.DERIVED),
                ReportField(label="id", value=self.id, role=FieldRole.DERIVED),
                ReportField(label="type", value=self.type, role=FieldRole.DERIVED),
                ReportField(label="serial", value=self.serial, role=FieldRole.DERIVED),
                ReportField(label="bbox", value=self.bbox, role=FieldRole.DERIVED),
            ],
        )

    def copy(self) -> Detector:
        """Copy the detector.

        This deep-copies all data fields and amplifiers, but only
        shallow-copies the internal `.CameraFrameSet`, as that's conceptually
        immutable.
        """
        return Detector(
            self._attributes.model_copy(deep=True),
            amplifiers=[a.model_copy(deep=True) for a in self._amplifiers],
            frames=self._frames,
        )

    def serialize(self, archive: OutputArchive[Any], save_frames: bool = True) -> DetectorSerializationModel:
        """Serialize this detector to an archive.

        Parameters
        ----------
        archive
            Archive to save to.
        save_frames
            Whether to save the `.CameraFrameSet` held by this detector.  This
            allows the frame set to be saved once for multiple detectors when
            they are part of a multi-detector object.
        """
        return DetectorSerializationModel(
            attributes=self._attributes,
            amplifiers=self._amplifiers,
            frames=archive.serialize_direct("frames", self._frames.serialize) if save_frames else None,
            visit=self.visit,
        )

    @staticmethod
    def _get_archive_tree_type(
        pointer_type: builtins.type[Any],
    ) -> builtins.type[DetectorSerializationModel]:
        """Return the serialization model type for this object for an archive
        type that uses the given pointer type.
        """
        return DetectorSerializationModel

    def to_legacy(self, *, is_raw_assembled: bool | None = None) -> LegacyDetector:
        """Convert to a legacy `lsst.afw.cameraGeom.Detector` instance.

        Parameters
        ----------
        is_raw_assembled
            Whether to use `Amplifier.assembled_raw_geometry` (`True`) or
            `Amplifier.unassembled_raw_geometry` (`False`).  If `None`, this
            is set to ``self.visit is not None``, since we expect to only add
            a visit ID to detectors that have been assembled.
        """
        from lsst.afw.cameraGeom import FIELD_ANGLE, FOCAL_PLANE, Camera
        from lsst.geom import Extent2D, Point2D

        if is_raw_assembled is None:
            is_raw_assembled = self.visit is not None
        # Legacy Detectors can only be built from scratch as a part of a
        # camera.
        camera_builder = Camera.Builder(self.name)
        fp_to_fa = self._frames[self._frames.focal_plane(), self._frames.field_angle()]
        legacy_fp_to_fa = fp_to_fa.to_legacy()
        camera_builder.setFocalPlaneParity(np.linalg.det(legacy_fp_to_fa.getJacobian(Point2D(0.0, 0.0))) < 0)
        camera_builder.setTransformFromFocalPlaneTo(FIELD_ANGLE, legacy_fp_to_fa)
        detector_builder = camera_builder.add(self.name, self.id)
        detector_builder.setBBox(self.bbox.to_legacy())
        detector_builder.setType(self.type.to_legacy())
        detector_builder.setSerial(self.serial)
        detector_builder.setPhysicalType(self.physical_type)
        detector_builder.setOrientation(self.orientation.to_legacy())
        detector_builder.setPixelSize(Extent2D(self.pixel_size, self.pixel_size))
        detector_builder.setTransformFromPixelsTo(FOCAL_PLANE, self.to_focal_plane.to_legacy())
        for amp in self.amplifiers:
            try:
                detector_builder.append(amp.to_legacy_builder(is_raw_assembled))
            except Exception as err:
                err.add_note(f"On detector {self.id}/{self.name}.")
                raise
        camera = camera_builder.finish()
        return camera[self.id]

    @staticmethod
    def from_legacy(
        legacy_detector: LegacyDetector,
        *,
        instrument: str,
        visit: int | None = None,
        is_raw_assembled: bool | None = None,
    ) -> Detector:
        """Convert from a legacy `lsst.afw.cameraGeom.Detector` instance.

        Parameters
        ----------
        legacy_detector
            Legacy detector to convert.
        instrument
            Name of the instrument this detector belongs to.
        visit
            Visit ID, if this camera geometry can be associated with a
            particular visit.
        is_raw_assembled
            Whether to populate `Amplifier.assembled_raw_geometry` (`True`) or
            `Amplifier.unassembled_raw_geometry` (`False`).  If `None`, this
            is set to ``visit is not None``, since we expect to only add
            a visit ID to detectors that have been assembled.
        """
        if is_raw_assembled is None:
            is_raw_assembled = visit is not None
        attributes = DetectorAttributes(
            name=legacy_detector.getName(),
            id=legacy_detector.getId(),
            type=DetectorType.from_legacy(legacy_detector.getType()),
            bbox=Box.from_legacy(legacy_detector.getBBox()),
            serial=legacy_detector.getSerial(),
            orientation=Orientation.from_legacy(legacy_detector.getOrientation()),
            pixel_size=legacy_detector.getPixelSize().getX(),
            physical_type=legacy_detector.getPhysicalType(),
        )
        amplifiers = [
            Amplifier.from_legacy(legacy_amp, is_raw_assembled=is_raw_assembled)
            for legacy_amp in legacy_detector.getAmplifiers()
        ]
        transform_map = legacy_detector.getTransformMap()
        frames = CameraFrameSet(instrument, transform_map.makeFrameSet([legacy_detector]))
        return Detector(attributes, amplifiers, frames, visit=visit)


class DetectorSerializationModel(ArchiveTree):
    """Serialization model for `Detector`."""

    SCHEMA_NAME: ClassVar[str] = "detector"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0.dev0"
    MIN_READ_VERSION: ClassVar[int] = 1
    PUBLIC_TYPE: ClassVar[type] = Detector

    attributes: DetectorAttributes = pydantic.Field(
        description="The simple plain-old-data attributes of the detector."
    )

    amplifiers: list[Amplifier] = pydantic.Field(
        default_factory=list,
        description="Descriptions of the amplifiers.",
    )

    frames: CameraFrameSetSerializationModel | None = pydantic.Field(
        default=None, description="Mappings to other camera coordinate systems."
    )

    visit: int | None = pydantic.Field(description="ID of the visit this detector is associated with.")

    def deserialize(
        self, archive: InputArchive[Any], frames: CameraFrameSet | None = None, **kwargs: Any
    ) -> Detector:
        """Deserialize this detector from an archive.

        Parameters
        ----------
        archive
            Serialization model instance for this detector.
        frames
            Coordinate systems and transforms to use instead of what is saved
            in ``model``.  Must be provided if ``model.frames`` is `None`.
        **kwargs
            Unsupported keyword arguments are accepted only to provide
            better error messages (raising
            `.serialization.InvalidParameterError`).
        """
        if kwargs:
            raise InvalidParameterError(f"Unrecognized parameters for Detector: {set(kwargs.keys())}.")
        if frames is None:
            if self.frames is None:
                raise ArchiveReadError(
                    "Serialized detector did not include coordinate transforms, "
                    "and 'frames' was not provided."
                )
            frames = self.frames.deserialize(archive)
        return Detector(self.attributes, self.amplifiers, frames, visit=self.visit)

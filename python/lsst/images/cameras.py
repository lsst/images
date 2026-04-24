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
    "Amplifier",
    "AmplifierCalibrations",
    "AmplifierRawGeometry",
    "Detector",
    "DetectorAttributes",
    "DetectorType",
    "Orientation",
    "ReadoutCorner",
)

import builtins
import enum
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, final

import astropy.units
import numpy as np
import pydantic

from ._geom import YX, Box
from ._transforms import (
    CameraFrameSet,
    CameraFrameSetSerializationModel,
    DetectorFrame,
    FieldAngleFrame,
    FocalPlaneFrame,
    Transform,
)
from .serialization import (
    ArchiveReadError,
    ArchiveTree,
    InlineArray,
    InputArchive,
    OutputArchive,
    Quantity,
    Unit,
)

if TYPE_CHECKING:
    try:
        from lsst.afw.cameraGeom import Amplifier as LegacyAmplifier
        from lsst.afw.cameraGeom import Detector as LegacyDetector
        from lsst.afw.cameraGeom import DetectorType as LegacyDetectorType
        from lsst.afw.cameraGeom import Orientation as LegacyOrientation
        from lsst.afw.cameraGeom import ReadoutCorner as LegacyReadoutCorner
    except ImportError:
        type LegacyDetector = Any  # type: ignore[no-redef]
        type LegacyDetectorType = Any  # type: ignore[no-redef]
        type LegacyOrientation = Any  # type: ignore[no-redef]
        type LegacyReadoutCorner = Any  # type: ignore[no-redef]
        type LegacyAmplifier = Any  # type: ignore[no-redef]


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
        """Convert from `lsst.afw.cameraGeom.DetectorType`."""
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
    yaw: Quantity = pydantic.Field(0.0 * astropy.units.radian, description="Rotation about the Z axis.")
    pitch: Quantity = pydantic.Field(
        0.0 * astropy.units.radian, description="Rotation about the Y axis (as defined after applying 'yaw')."
    )
    roll: Quantity = pydantic.Field(
        0.0 * astropy.units.radian,
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
        """Convert from `lsst.afw.cameraGeom.Orientation`."""
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


class ReadoutCorner(enum.StrEnum):
    """Enumeration of the possible readout corners of an amplifier."""

    LL = "LL"
    LR = "LR"
    UR = "UR"
    UL = "UL"

    def to_legacy(self) -> LegacyReadoutCorner:
        """Convert to `lsst.afw.cameraGeom.ReadoutCorner`."""
        from lsst.afw.cameraGeom import ReadoutCorner as LegacyReadoutCorner

        return getattr(LegacyReadoutCorner, self.value)

    @classmethod
    def from_legacy(cls, legacy_readout_corner: LegacyReadoutCorner) -> ReadoutCorner:
        """Convert from `lsst.afw.cameraGeom.ReadoutCorner`."""
        return getattr(cls, legacy_readout_corner.name)

    def as_flips(self) -> YX[bool]:
        """Return a tuple indicating how the image needs to be flipped to
        bring the readout corner to ``LL``.
        """
        return YX(
            y=self is ReadoutCorner.LL or self is ReadoutCorner.LR,
            x=self is ReadoutCorner.UR or self is ReadoutCorner.UR,
        )

    @classmethod
    def from_flips(cls, *, y: bool, x: bool) -> ReadoutCorner:
        """Construct from booleans indicating how the image needs to be
        flipped to bring the readout corner to ``LL``.
        """
        match y, x:
            case False, False:
                return cls.LL
            case False, True:
                return cls.LR
            case True, True:
                return cls.UR
            case True, False:
                return cls.UL
        raise TypeError(f"Invalid arguments: y={y}, x={x} (expected booleans).")

    def apply_flips(self, *, y: bool, x: bool) -> ReadoutCorner:
        """Return the new readout corner after applying the given flips."""
        current = self.as_flips()
        return self.from_flips(y=current.y ^ y, x=current.x ^ x)


@final
class AmplifierRawGeometry(pydantic.BaseModel):
    """A struct that describes the geometry of an amplifire in a raw image."""

    bbox: Box = pydantic.Field(description="Bounding box of the full untrimmed amplifier in the raw image.")
    data_bbox: Box = pydantic.Field(description="Bounding box of the data section in the raw image.")
    flip_x: bool = pydantic.Field(False, description="Whether to flip the X coordinates during assembly.")
    flip_y: bool = pydantic.Field(False, description="Whether to flip the Y coordinates during assembly.")
    x_offset: int = pydantic.Field(
        0,
        description=(
            "X offset between the raw position of this amplifier and the trimmed, "
            "assembled position of the amplifier."
        ),
    )
    y_offset: int = pydantic.Field(
        0,
        description=(
            "Y offset between the raw position of this amplifier and the trimmed, "
            "assembled position of the amplifier."
        ),
    )
    horizontal_overscan_bbox: Box = pydantic.Field(
        description="Bounding box of the horizontal (serial) overscan region in the raw image."
    )
    vertical_overscan_bbox: Box = pydantic.Field(
        description="Bounding box of the vertical (parallel) overscan region in the raw image."
    )
    horizontal_prescan_bbox: Box = pydantic.Field(
        description="Bounding box of the horizontal (serial) pre-scan region in the raw image."
    )
    readout_corner: ReadoutCorner = pydantic.Field(
        description=(
            "Readout corner of the amplifier in the raw image "
            "(with x increasing to the right and y increasing up)."
        )
    )

    @staticmethod
    def from_legacy_amplifier(legacy_amplifier: LegacyAmplifier) -> AmplifierRawGeometry:
        """Convert from a `lsst.afw.cameraGeom.Amplifier`.

        Parameters
        ----------
        legacy_amplifier
            Legacy amplifier to convert.
        """
        x_offset, y_offset = legacy_amplifier.getRawXYOffset()
        return AmplifierRawGeometry(
            bbox=Box.from_legacy(legacy_amplifier.getRawBBox()),
            data_bbox=Box.from_legacy(legacy_amplifier.getRawDataBBox()),
            flip_x=legacy_amplifier.getRawFlipX(),
            flip_y=legacy_amplifier.getRawFlipY(),
            x_offset=x_offset,
            y_offset=y_offset,
            horizontal_overscan_bbox=Box.from_legacy(legacy_amplifier.getRawHorizontalOverscanBBox()),
            vertical_overscan_bbox=Box.from_legacy(legacy_amplifier.getRawVerticalOverscanBBox()),
            horizontal_prescan_bbox=Box.from_legacy(legacy_amplifier.getRawPrescanBBox()),
            readout_corner=ReadoutCorner.from_legacy(legacy_amplifier.getReadoutCorner()),
        )


@final
class AmplifierCalibrations(pydantic.BaseModel, ser_json_inf_nan="constants"):
    """A struct that holds nominal information about an amplifier that is
    often superseded by separate calibration datasets.
    """

    gain: float
    read_noise: float
    saturation: float
    suspect_level: float
    linearity_coefficients: InlineArray
    linearity_type: str
    linearity_threshold: float
    linearity_maximum: float
    linearity_unit: Unit

    @staticmethod
    def from_legacy_amplifier(legacy_amplifier: LegacyAmplifier) -> AmplifierCalibrations:
        """Convert from a `lsst.afw.cameraGeom.Amplifier`.

        Parameters
        ----------
        legacy_amplifier
            Legacy amplifier to convert.
        """
        return AmplifierCalibrations(
            gain=legacy_amplifier.getGain(),
            read_noise=legacy_amplifier.getReadNoise(),
            saturation=legacy_amplifier.getSaturation(),
            suspect_level=legacy_amplifier.getSuspectLevel(),
            linearity_coefficients=legacy_amplifier.getLinearityCoeffs(),
            linearity_type=legacy_amplifier.getLinearityType(),
            linearity_threshold=legacy_amplifier.getLinearityThreshold(),
            linearity_maximum=legacy_amplifier.getLinearityMaximum(),
            linearity_unit=astropy.units.Unit(legacy_amplifier.getLinearityUnits()),
        )


@final
class Amplifier(pydantic.BaseModel, ser_json_inf_nan="constants"):
    """A struct that holds information about an amplifier."""

    name: str = pydantic.Field(description="Name of the amplifier.")
    bbox: Box = pydantic.Field(
        description="Bounding box of the amplifier data region in a trimmed, assembled detector."
    )
    assembled_raw_geometry: AmplifierRawGeometry | None = pydantic.Field(
        None,
        description=(
            "Geometry of this amplifier in an assembled but untrimmed raw image that has all amplifiers."
        ),
    )
    unassembled_raw_geometry: AmplifierRawGeometry | None = pydantic.Field(
        None,
        description=(
            "Geometry of this amplifier in an unassembled, untrimmed raw image that has just this amplifier."
        ),
    )
    nominal_calibrations: AmplifierCalibrations | None = pydantic.Field(
        None,
        description=(
            "Nominal calibration information that may be superseded by separate calibration datasets."
        ),
    )

    @property
    def readout_corner(self) -> ReadoutCorner:
        """The readout corner in the final assembled, trimmed image.

        Notes
        -----
        This definition - i.e. relative to the final assembled, trimmed image
        - is consistent with historical
        `lsst.afw.cameraGeom.Amplifier.getReadoutCorner` documentation, but
        inconsistent with its behavior on unassembled images, which was to
        report the readout corner in the image the legacy amplifier was
        attached to (i.e. the same as `AmplifierRawGeometry.readout_corner`).
        """
        if self.assembled_raw_geometry is not None:
            return self.assembled_raw_geometry.readout_corner.apply_flips(
                y=self.assembled_raw_geometry.flip_y, x=self.assembled_raw_geometry.flip_x
            )
        if self.unassembled_raw_geometry is not None:
            return self.unassembled_raw_geometry.readout_corner.apply_flips(
                y=self.unassembled_raw_geometry.flip_y, x=self.unassembled_raw_geometry.flip_x
            )
        raise AttributeError("Amplifier has no raw geometry.")

    def to_legacy_builder(self, is_raw_assembled: bool) -> LegacyAmplifier.Builder:
        """Convert to a `lsst.afw.cameraGeom.Amplifier.Builder`.

        Parameters
        ----------
        is_raw_assembled
            Whether to use `Amplifier.assembled_raw_geometry` (`True`) or
            `Amplifier.unassembled_raw_geometry` (`False`).  If `None`, this
            is set to ``self.visit is not None``, since we expect to only add
            a visit ID to detectors that have been assembled.
        """
        from lsst.afw.cameraGeom import Amplifier as LegacyAmplifier
        from lsst.geom import Extent2I

        builder = LegacyAmplifier.Builder()
        builder.setName(self.name)
        builder.setBBox(self.bbox.to_legacy())
        if is_raw_assembled:
            if (raw_geom := self.assembled_raw_geometry) is None:
                raise ValueError(
                    f"is_raw_assembled=True but assembled_raw_geometry is None for amp {self.name}."
                )
        else:
            if (raw_geom := self.unassembled_raw_geometry) is None:
                raise ValueError(
                    f"is_raw_assembled=False but unassembled_raw_geometry is None for amp {self.name}."
                )
        builder.setReadoutCorner(raw_geom.readout_corner.to_legacy())
        builder.setRawBBox(raw_geom.bbox.to_legacy())
        builder.setRawDataBBox(raw_geom.data_bbox.to_legacy())
        builder.setRawFlipX(raw_geom.flip_x)
        builder.setRawFlipY(raw_geom.flip_y)
        builder.setRawXYOffset(Extent2I(raw_geom.x_offset, raw_geom.y_offset))
        builder.setRawHorizontalOverscanBBox(raw_geom.horizontal_overscan_bbox.to_legacy())
        builder.setRawVerticalOverscanBBox(raw_geom.vertical_overscan_bbox.to_legacy())
        builder.setRawPrescanBBox(raw_geom.horizontal_prescan_bbox.to_legacy())
        if self.nominal_calibrations is not None:
            builder.setGain(self.nominal_calibrations.gain)
            builder.setReadNoise(self.nominal_calibrations.read_noise)
            builder.setSaturation(self.nominal_calibrations.saturation)
            builder.setSuspectLevel(self.nominal_calibrations.suspect_level)
            builder.setLinearityCoeffs(self.nominal_calibrations.linearity_coefficients)
            builder.setLinearityType(self.nominal_calibrations.linearity_type)
            builder.setLinearityThreshold(self.nominal_calibrations.linearity_threshold)
            builder.setLinearityMaximum(self.nominal_calibrations.linearity_maximum)
            builder.setLinearityUnits(self.nominal_calibrations.linearity_unit.to_string())
        return builder

    @staticmethod
    def from_legacy(legacy_amplifier: LegacyAmplifier, is_raw_assembled: bool) -> Amplifier:
        """Convert from a `lsst.afw.cameraGeom.Amplifier`.

        Parameters
        ----------
        legacy_amplifier
            Legacy amplifier to convert.
        is_raw_assembled
            Whether to populate `Amplifier.assembled_raw_geometry` (`True`) or
            `Amplifier.unassembled_raw_geometry` (`False`).
        """
        raw_geometry = AmplifierRawGeometry.from_legacy_amplifier(legacy_amplifier)
        nominal_calibrations = AmplifierCalibrations.from_legacy_amplifier(legacy_amplifier)
        return Amplifier(
            name=legacy_amplifier.getName(),
            bbox=Box.from_legacy(legacy_amplifier.getBBox()),
            assembled_raw_geometry=raw_geometry if is_raw_assembled else None,
            unassembled_raw_geometry=raw_geometry if not is_raw_assembled else None,
            nominal_calibrations=nominal_calibrations,
        )


@final
class Detector:
    """Information about a detector in a camera."""

    def __init__(
        self,
        attributes: DetectorAttributes,
        amplifiers: Iterable[Amplifier],
        frames: CameraFrameSet,
        visit: int | None = None,
    ):
        self._attributes = attributes
        self._amplifiers = list(amplifiers)
        self._frames = frames
        self._frame = frames.detector(attributes.id, visit=visit)

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
            frames=self._frames.serialize(archive) if save_frames else None,
            visit=self.visit,
        )

    @staticmethod
    def deserialize(
        model: DetectorSerializationModel, archive: InputArchive[Any], frames: CameraFrameSet | None = None
    ) -> Detector:
        """Deserialize this detector from an archive.

        Parameters
        ----------
        model
            Serialization model instance for this detector.
        archive
            Archive to read from.
        frames
            Coordinate systems and transforms to use instead of what is saved
            in ``model``.  Must be provided if ``model.frames`` is `None`.
        """
        if frames is None:
            if model.frames is None:
                raise ArchiveReadError(
                    "Serialized detector did not include coordinate transforms, "
                    "and 'frames' was not provided."
                )
            frames = CameraFrameSet.deserialize(model.frames, archive)
        return Detector(model.attributes, model.amplifiers, frames, visit=model.visit)

    @staticmethod
    def _get_archive_tree_type(
        pointer_type: builtins.type[Any],
    ) -> builtins.type[DetectorSerializationModel]:
        """Return the serialization model type for this object for an archive
        type that uses the given pointer type.
        """
        return DetectorSerializationModel  # type: ignore

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

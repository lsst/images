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
    "ReadoutCorner",
)

import enum
from typing import TYPE_CHECKING, Any, final

import numpy as np
import pydantic

from .._geom import YX, Box
from ..serialization import InlineArray

if TYPE_CHECKING:
    try:
        from lsst.afw.cameraGeom import Amplifier as LegacyAmplifier
        from lsst.afw.cameraGeom import ReadoutCorner as LegacyReadoutCorner
    except ImportError:
        type LegacyReadoutCorner = Any  # type: ignore[no-redef]
        type LegacyAmplifier = Any  # type: ignore[no-redef]


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
        """Convert from `lsst.afw.cameraGeom.ReadoutCorner`.

        Parameters
        ----------
        legacy_readout_corner
            Legacy readout corner to convert.
        """
        return getattr(cls, legacy_readout_corner.name)

    def as_flips(self) -> YX[bool]:
        """Return a tuple indicating how the image needs to be flipped to
        bring the readout corner to ``LL``.
        """
        return YX(
            y=self is ReadoutCorner.UL or self is ReadoutCorner.UR,
            x=self is ReadoutCorner.LR or self is ReadoutCorner.UR,
        )

    @classmethod
    def from_flips(cls, *, y: bool, x: bool) -> ReadoutCorner:
        """Construct from booleans indicating how the image needs to be
        flipped to bring the readout corner to ``LL``.

        Parameters
        ----------
        y
            Whether the image is flipped in the y direction.
        x
            Whether the image is flipped in the x direction.
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
        """Return the new readout corner after applying the given flips.

        Parameters
        ----------
        y
            Whether to flip in the y direction.
        x
            Whether to flip in the x direction.
        """
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
    serial_overscan_bbox: Box = pydantic.Field(
        description="Bounding box of the serial (horizontal) overscan region in the raw image."
    )
    parallel_overscan_bbox: Box = pydantic.Field(
        description="Bounding box of the parallel (vertical) overscan region in the raw image."
    )
    prescan_bbox: Box = pydantic.Field(
        description="Bounding box of the serial (horizontal) pre-scan region in the raw image."
    )
    readout_corner: ReadoutCorner = pydantic.Field(
        description=(
            "Readout corner of the amplifier in the raw image "
            "(with x increasing to the right and y increasing up)."
        )
    )

    @property
    def horizontal_overscan_bbox(self) -> Box:
        """Bounding box of the serial (horizon) overscan region in the raw
        image (`.Box`).
        """
        return self.serial_overscan_bbox

    @horizontal_overscan_bbox.setter
    def horizontal_overscan_bbox(self, value: Box) -> None:
        self.serial_overscan_bbox = value

    @property
    def vertical_overscan_bbox(self) -> Box:
        """Bounding box of the parallel (vertical) overscan region in the raw
        image (`.Box`).
        """
        return self.parallel_overscan_bbox

    @vertical_overscan_bbox.setter
    def vertical_overscan_bbox(self, value: Box) -> None:
        self.parallel_overscan_bbox = value

    @property
    def horizontal_prescan_bbox(self) -> Box:
        """Bounding box of the serial (horizon) prescan region in the raw
        image (`.Box`).
        """
        return self.prescan_bbox

    @horizontal_prescan_bbox.setter
    def horizontal_prescan_bbox(self, value: Box) -> None:
        self.prescan_bbox = value

    @property
    def serial_prescan_bbox(self) -> Box:
        """Bounding box of the serial (horizon) prescan region in the raw
        image (`.Box`).
        """
        return self.prescan_bbox

    @serial_prescan_bbox.setter
    def serial_prescan_bbox(self, value: Box) -> None:
        self.prescan_bbox = value

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
            serial_overscan_bbox=Box.from_legacy(legacy_amplifier.getRawSerialOverscanBBox()),
            parallel_overscan_bbox=Box.from_legacy(legacy_amplifier.getRawParallelOverscanBBox()),
            prescan_bbox=Box.from_legacy(legacy_amplifier.getRawPrescanBBox()),
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

    def __eq__(self, other: object) -> bool:
        if type(other) is not AmplifierCalibrations:
            return NotImplemented
        # ``suspect_level`` is a float whose "unset" sentinel is ``NaN``;
        # treat NaN==NaN as equal here so a round-tripped calibration
        # block does not spuriously compare unequal to its source.
        return (
            self.gain == other.gain
            and self.read_noise == other.read_noise
            and self.saturation == other.saturation
            and (
                self.suspect_level == other.suspect_level
                or (np.isnan(self.suspect_level) and np.isnan(other.suspect_level))
            )
            and np.array_equal(self.linearity_coefficients, other.linearity_coefficients)
            and self.linearity_type == other.linearity_type
        )

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
        )


@final
class Amplifier(pydantic.BaseModel, ser_json_inf_nan="constants"):
    """A struct that holds information about an amplifier."""

    name: str = pydantic.Field(description="Name of the amplifier.")
    bbox: Box = pydantic.Field(
        description="Bounding box of the amplifier data region in a trimmed, assembled detector."
    )
    readout_corner: ReadoutCorner = pydantic.Field(
        description=(
            "Readout corner of the amplifier in the final assembled, trimmed "
            "image (with x increasing to the right and y increasing up). "
        )
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
        # The afw readout corner definition corresponds to the image it is
        # attached to (which might be a raw), not the final trimmed image
        # (despite the docs, until a change on this ticket).
        builder.setReadoutCorner(raw_geom.readout_corner.to_legacy())
        builder.setRawBBox(raw_geom.bbox.to_legacy())
        builder.setRawDataBBox(raw_geom.data_bbox.to_legacy())
        builder.setRawFlipX(raw_geom.flip_x)
        builder.setRawFlipY(raw_geom.flip_y)
        builder.setRawXYOffset(Extent2I(raw_geom.x_offset, raw_geom.y_offset))
        builder.setRawSerialOverscanBBox(raw_geom.serial_overscan_bbox.to_legacy())
        builder.setRawParallelOverscanBBox(raw_geom.parallel_overscan_bbox.to_legacy())
        builder.setRawPrescanBBox(raw_geom.prescan_bbox.to_legacy())
        if self.nominal_calibrations is not None:
            builder.setGain(self.nominal_calibrations.gain)
            builder.setReadNoise(self.nominal_calibrations.read_noise)
            builder.setSaturation(self.nominal_calibrations.saturation)
            builder.setSuspectLevel(self.nominal_calibrations.suspect_level)
            builder.setLinearityCoeffs(self.nominal_calibrations.linearity_coefficients)
            builder.setLinearityType(self.nominal_calibrations.linearity_type)
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
        readout_corner = raw_geometry.readout_corner.apply_flips(y=raw_geometry.flip_y, x=raw_geometry.flip_x)
        return Amplifier(
            name=legacy_amplifier.getName(),
            bbox=Box.from_legacy(legacy_amplifier.getBBox()),
            readout_corner=readout_corner,
            assembled_raw_geometry=raw_geometry if is_raw_assembled else None,
            unassembled_raw_geometry=raw_geometry if not is_raw_assembled else None,
            nominal_calibrations=nominal_calibrations,
        )

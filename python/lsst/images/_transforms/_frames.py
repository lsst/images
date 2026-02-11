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
    "ICRS",
    "DetectorFrame",
    "FieldAngleFrame",
    "FocalPlaneFrame",
    "Frame",
    "SerializableFrame",
    "SkyFrame",
    "TractFrame",
)

import enum
from typing import Annotated, Literal, Protocol, Self, cast, final

import astropy.units as u
import numpy as np
import pydantic

from .._geom import Box
from ..serialization import Unit
from ..utils import is_none


class Frame(Protocol):
    """A description of a coordinate system."""

    @property
    def unit(self) -> u.UnitBase:
        """Units of the coordinates in this frame
        (`astropy.units.UnitBase`).
        """
        ...

    def normalize_x[T: float | np.ndarray](self, x: T) -> T:
        """Normalize ``x`` coordinates into their standard range."""
        ...

    def normalize_y[T: float | np.ndarray](self, y: T) -> T:
        """Normalize ``y`` coordinates into their standard range."""
        ...

    # At present all Frames are members of the same serializable Union,
    # so their serialized form is just the original frame.  But this may not
    # always be the case.

    def serialize(self) -> SerializableFrame:
        """Return a Pydantic-serializable version of this Frame."""
        ...

    @classmethod
    def deserialize(cls, serialized: SerializableFrame) -> Self:
        """Convert a serialized frame to an in-memory one."""
        return cast(Self, serialized)

    @property
    def _ast_ident(self) -> str:
        """String to use as the 'Ident' attribute of an AST Frame."""
        ...


@final
class DetectorFrame(pydantic.BaseModel):
    """A coordinate frame for a particular detector's pixels.

    Notes
    -----
    This frame is only used for post-assembly images (i.e. not those with
    overscan regions still present).
    """

    instrument: str = pydantic.Field(description="Name of the instrument.")
    visit: int | None = pydantic.Field(
        default=None,
        description=(
            "ID of the visit.  May be unset in contexts where there "
            "is no visit or only a relevant single visit."
        ),
        exclude_if=is_none,
    )
    detector: int = pydantic.Field(description="ID of the detector.")
    bbox: Box = pydantic.Field(description="Bounding box of the detector.")
    frame_type: Literal["DETECTOR"] = pydantic.Field(
        default="DETECTOR", description="Descriminator for the frame type."
    )

    model_config = pydantic.ConfigDict(frozen=True)

    @property
    def unit(self) -> u.UnitBase:
        """Units of the coordinates in this frame
        (`astropy.units.UnitBase`).
        """
        return u.pix

    def normalize_x[T: float | np.ndarray](self, x: T) -> T:
        """Normalize ``x`` coordinates into their standard range."""
        return x

    def normalize_y[T: float | np.ndarray](self, y: T) -> T:
        """Normalize ``y`` coordinates into their standard range."""
        return y

    def serialize(self) -> SerializableFrame:
        """Return a Pydantic-serializable version of this Frame."""
        return cast(SerializableFrame, self)

    @classmethod
    def deserialize(cls, serialized: SerializableFrame) -> Self:
        """Convert a serialized frame to an in-memory one."""
        return cast(Self, serialized)

    @property
    def _ast_ident(self) -> str:
        return f"{_camera_ast_ident(self.instrument, self.visit)}/DETECTOR_{self.detector:03d}"


@final
class FocalPlaneFrame(pydantic.BaseModel):
    """A Euclidian coordinate frame for the focal plane of a camera."""

    instrument: str = pydantic.Field(description="Name of the instrument.")
    visit: int | None = pydantic.Field(
        default=None,
        description=(
            "ID of the visit.  May be unset in contexts where there "
            "is no visit or only a relevant single visit."
        ),
        exclude_if=is_none,
    )
    unit: Unit = pydantic.Field(description="Units of the coordinates in this frame.")

    frame_type: Literal["FOCAL_PLANE"] = pydantic.Field(
        default="FOCAL_PLANE", description="Descriminator for the frame type."
    )

    model_config = pydantic.ConfigDict(frozen=True)

    def normalize_x[T: float | np.ndarray](self, x: T) -> T:
        """Normalize ``x`` coordinates into their standard range."""
        return x

    def normalize_y[T: float | np.ndarray](self, y: T) -> T:
        """Normalize ``y`` coordinates into their standard range."""
        return y

    def serialize(self) -> SerializableFrame:
        """Return a Pydantic-serializable version of this Frame."""
        return cast(SerializableFrame, self)

    @classmethod
    def deserialize(cls, serialized: SerializableFrame) -> Self:
        """Convert a serialized frame to an in-memory one."""
        return cast(Self, serialized)

    @property
    def _ast_ident(self) -> str:
        return f"{_camera_ast_ident(self.instrument, self.visit)}/FOCAL_PLANE"


@final
class FieldAngleFrame(pydantic.BaseModel):
    """An angular coordinate frame that maps a camera onto the sky about its
    boresight.

    Notes
    -----
    The transform between a `FocalPlaneFrame` and a `FieldAngleFrame` includes
    optical distortions but no rotation.  It may include a parity flip.
    """

    instrument: str = pydantic.Field(description="Name of the instrument.")
    visit: int | None = pydantic.Field(
        default=None,
        description=(
            "ID of the visit.  May be unset in contexts where there "
            "is no visit or only a relevant single visit."
        ),
        exclude_if=is_none,
    )
    frame_type: Literal["FIELD_ANGLE"] = pydantic.Field(
        default="FIELD_ANGLE", description="Descriminator for the frame type."
    )

    model_config = pydantic.ConfigDict(frozen=True)

    @property
    def unit(self) -> u.UnitBase:
        """Units of the coordinates in this frame
        (`astropy.units.UnitBase`).
        """
        return u.rad

    def normalize_x[T: float | np.ndarray](self, x: T) -> T:
        """Normalize ``x`` coordinates into their standard range."""
        return _wrap_symmetric(x)

    def normalize_y[T: float | np.ndarray](self, y: T) -> T:
        """Normalize ``y`` coordinates into their standard range."""
        return _wrap_symmetric(y)

    def serialize(self) -> SerializableFrame:
        """Return a Pydantic-serializable version of this Frame."""
        return cast(SerializableFrame, self)

    @classmethod
    def deserialize(cls, serialized: SerializableFrame) -> Self:
        """Convert a serialized frame to an in-memory one."""
        return cast(Self, serialized)

    @property
    def _ast_ident(self) -> str:
        return f"{_camera_ast_ident(self.instrument, self.visit)}/FIELD_ANGLE"


@final
class TractFrame(pydantic.BaseModel):
    """The pixel coordinates of a tract: a region on the sky used for
    coaddition, defined by a 'skymap' and split into 'patches' that share
    a common pixel grid.
    """

    skymap: str = pydantic.Field(description="Name of the skymap.")
    tract: int = pydantic.Field(description="ID of the tract within its skymap.")
    bbox: Box = pydantic.Field(description="Bounding box of the full tract.")
    frame_type: Literal["TRACT"] = pydantic.Field(
        default="TRACT", description="Descriminator for the frame type."
    )

    model_config = pydantic.ConfigDict(frozen=True)

    @property
    def unit(self) -> u.UnitBase:
        """Units of the coordinates in this frame
        (`astropy.units.UnitBase`).
        """
        return u.pix

    def normalize_x[T: float | np.ndarray](self, x: T) -> T:
        """Normalize ``x`` coordinates into their standard range."""
        return x

    def normalize_y[T: float | np.ndarray](self, y: T) -> T:
        """Normalize ``y`` coordinates into their standard range."""
        return y

    def serialize(self) -> SerializableFrame:
        """Return a Pydantic-serializable version of this Frame."""
        return cast(SerializableFrame, self)

    @classmethod
    def deserialize(cls, serialized: SerializableFrame) -> Self:
        """Convert a serialized frame to an in-memory one."""
        return cast(Self, serialized)

    @property
    def _ast_ident(self) -> str:
        return f"{self.skymap}@{self.tract}"


class SkyFrame(enum.StrEnum):
    """The special frame that represents the sky, in ICRS coordinates."""

    ICRS = "ICRS"

    @property
    def unit(self) -> u.UnitBase:
        """Units of the coordinates in this frame
        (`astropy.units.UnitBase`).
        """
        return u.rad

    def normalize_x[T: float | np.ndarray](self, x: T) -> T:
        """Normalize ``x`` coordinates into their standard range."""
        return _wrap_positive(x)

    def normalize_y[T: float | np.ndarray](self, y: T) -> T:
        """Normalize ``x`` coordinates into their standard range."""
        return _wrap_symmetric(y)

    def serialize(self) -> SerializableFrame:
        """Return a Pydantic-serializable version of this Frame."""
        return cast(SerializableFrame, self)

    @classmethod
    def deserialize(cls, serialized: SerializableFrame) -> Self:
        """Convert a serialized frame to an in-memory one."""
        return cast(Self, serialized)

    @property
    def _ast_ident(self) -> str:
        return self.value


ICRS = SkyFrame.ICRS


type SerializableFrame = (
    SkyFrame
    | Annotated[
        DetectorFrame | TractFrame | FocalPlaneFrame | FieldAngleFrame,
        pydantic.Field(discriminator="frame_type"),
    ]
)


_TWOPI: float = np.pi * 2


def _camera_ast_ident(instrument: str, visit: int | None) -> str:
    return f"{instrument}@{visit}" if visit is not None else instrument


def _wrap_positive[T: float | np.ndarray](a: T) -> T:
    return a % _TWOPI  # type: ignore[return-value]


def _wrap_symmetric[T: float | np.ndarray](a: T) -> T:
    return (a + np.pi) % _TWOPI - np.pi  # type: ignore[return-value]

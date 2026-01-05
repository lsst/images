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
    "SkyFrame",
    "TractFrame",
)

import enum
from typing import Annotated, Literal, final

import astropy.units as u
import numpy as np
import pydantic

from .._geom import Box
from ..serialization import Unit
from ..utils import is_none


class CameraGeometryFrame(pydantic.BaseModel):
    """A base class for coordinate frames that are associated with a
    particular camera.
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

    @property
    def _ast_ident(self) -> str:
        return f"{self.instrument}@{self.visit}" if self.visit is not None else self.instrument


@final
class DetectorFrame(CameraGeometryFrame):
    """A coordinate frame for a particular detector's pixels.

    Notes
    -----
    This frame is only used for post-assembly images (i.e. not those with
    overscan regions still present).
    """

    detector: int = pydantic.Field(description="ID of the detector.")
    bbox: Box = pydantic.Field(description="Bounding box of the detector.")
    frame_type: Literal["DETECTOR"] = pydantic.Field(
        default="DETECTOR", description="Descriminator for the frame type."
    )

    model_config = pydantic.ConfigDict(frozen=True)

    @property
    def unit(self) -> Unit:
        """Units of the coordinates in this frame (`astropy.unit.UnitBase`)."""
        return u.pix

    def normalize_x[T: float | np.ndarray](self, x: T) -> T:
        """Normalize ``x`` coordinates into their standard range."""
        return x

    def normalize_y[T: float | np.ndarray](self, y: T) -> T:
        """Normalize ``y`` coordinates into their standard range."""
        return y

    @property
    def _ast_ident(self) -> str:
        return f"{super()._ast_ident}/DETECTOR_{self.detector:03d}"


@final
class FocalPlaneFrame(CameraGeometryFrame):
    """A Euclidian coordinate frame for the focal plane of a camera."""

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

    @property
    def _ast_ident(self) -> str:
        return f"{super()._ast_ident}/FOCAL_PLANE"


@final
class FieldAngleFrame(CameraGeometryFrame):
    """An angular coordinate frame that maps a camera onto the sky about its
    boresight.

    Notes
    -----
    The transform between a `FocalPlaneFrame` and a `FieldAngleFrame` includes
    optical distortions but no rotation.  It may include a parity flip.
    """

    frame_type: Literal["FIELD_ANGLE"] = pydantic.Field(
        default="FIELD_ANGLE", description="Descriminator for the frame type."
    )

    model_config = pydantic.ConfigDict(frozen=True)

    @property
    def unit(self) -> Unit:
        """Units of the coordinates in this frame (`astropy.unit.UnitBase`)."""
        return u.rad

    def normalize_x[T: float | np.ndarray](self, x: T) -> T:
        """Normalize ``x`` coordinates into their standard range."""
        return _wrap_symmetric(x)

    def normalize_y[T: float | np.ndarray](self, y: T) -> T:
        """Normalize ``x`` coordinates into their standard range."""
        return _wrap_symmetric(y)

    @property
    def _ast_ident(self) -> str:
        return f"{super()._ast_ident}/FIELD_ANGLE"


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
    def unit(self) -> Unit:
        """Units of the coordinates in this frame (`astropy.unit.UnitBase`)."""
        return u.pix

    def normalize_x[T: float | np.ndarray](self, x: T) -> T:
        """Normalize ``x`` coordinates into their standard range."""
        return x

    def normalize_y[T: float | np.ndarray](self, y: T) -> T:
        """Normalize ``x`` coordinates into their standard range."""
        return y

    @property
    def _ast_ident(self) -> str:
        return f"{self.skymap}@{self.tract}"


class SkyFrame(enum.StrEnum):
    """The special frame that represents the sky, in ICRS coordinates."""

    ICRS = "ICRS"

    @property
    def unit(self) -> Unit:
        """Units of the coordinates in this frame (`astropy.unit.UnitBase`)."""
        return u.rad

    def normalize_x[T: float | np.ndarray](self, x: T) -> T:
        """Normalize ``x`` coordinates into their standard range."""
        return _wrap_positive(x)

    def normalize_y[T: float | np.ndarray](self, y: T) -> T:
        """Normalize ``x`` coordinates into their standard range."""
        return _wrap_symmetric(y)

    @property
    def _ast_ident(self) -> str:
        return self.value


ICRS = SkyFrame.ICRS


type Frame = (
    SkyFrame
    | Annotated[
        DetectorFrame | TractFrame | FocalPlaneFrame | FieldAngleFrame,
        pydantic.Field(discriminator="frame_type"),
    ]
)


_TWOPI: float = np.pi * 2


def _wrap_positive[T: float | np.ndarray](a: T) -> T:
    return a % _TWOPI  # type: ignore[return-value]


def _wrap_symmetric[T: float | np.ndarray](a: T) -> T:
    return (a + np.pi) % _TWOPI - np.pi  # type: ignore[return-value]

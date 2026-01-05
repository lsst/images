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
import pydantic

from .._geom import Box
from ..asdf_utils import Unit


class CameraGeometryFrame(pydantic.BaseModel):
    instrument: str
    visit: int | None = None


@final
class DetectorFrame(CameraGeometryFrame):
    detector: int
    bbox: Box

    frame_type: Literal["DETECTOR"] = "DETECTOR"
    model_config = pydantic.ConfigDict(frozen=True)

    @property
    def unit(self) -> Unit:
        return u.pix


@final
class FocalPlaneFrame(CameraGeometryFrame):
    unit: Unit | None = None

    frame_type: Literal["FOCAL_PLANE"] = "FOCAL_PLANE"
    model_config = pydantic.ConfigDict(frozen=True)


@final
class FieldAngleFrame(CameraGeometryFrame):
    frame_type: Literal["FIELD_ANGLE"] = "FIELD_ANGLE"
    model_config = pydantic.ConfigDict(frozen=True)

    @property
    def unit(self) -> Unit:
        return u.rad


@final
class TractFrame(pydantic.BaseModel):
    skymap: str
    tract: int
    bbox: Box

    frame_type: Literal["TRACT"] = "TRACT"
    model_config = pydantic.ConfigDict(frozen=True)

    @property
    def unit(self) -> Unit:
        return u.pix


class SkyFrame(enum.StrEnum):
    ICRS = "ICRS"

    @property
    def unit(self) -> Unit:
        return u.rad


ICRS = SkyFrame.ICRS


type Frame = (
    SkyFrame
    | Annotated[
        DetectorFrame | TractFrame | FocalPlaneFrame | FieldAngleFrame,
        pydantic.Field(discriminator="frame_type"),
    ]
)

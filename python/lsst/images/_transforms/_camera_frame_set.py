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

__all__ = ("CameraFrameSet", "CameraFrameSetSerializationModel")

from typing import Any

import astropy.units as u
import pydantic

from .._geom import Bounds, Box
from ..serialization import InputArchive, OutputArchive
from . import _frames  # use this import style to facilitate pattern matching
from ._frame_set import FrameLookupError, FrameSet
from ._transform import Transform


class CameraFrameSet(FrameSet):
    """A `FrameSet` that manages the coordinate systems of a camera.

    The `CameraFrameSet` class constructor is considered a private
    implementation detail.  At present, instances can only be obtained by
    loading them from storage (`deserialize`) or converting a legacy
    `lsst.afw.cameraGeom` object (`from_legacy`).
    """

    def __init__(self, instrument: str, ast: Any):
        self._ast = ast
        self._focal_plane_frame_id: int = 0
        self._field_angle_frame_id: int = 0
        self._detector_frame_ids: dict[int, int] = {}
        for frame_id in range(1, self._ast.nFrame + 1):
            ast_frame = self._ast.getFrame(frame_id, copy=False)
            match ast_frame.ident:
                case "FOCAL_PLANE":
                    self._focal_plane_frame_id = frame_id
                case "FIELD_ANGLE":
                    self._field_angle_frame_id = frame_id
                case str(s) if s.startswith("DETECTOR_"):
                    detector_id = int(s.removeprefix("DETECTOR_"))
                    self._detector_frame_ids[detector_id] = frame_id
                case _:
                    raise ValueError(f"Unexpected frame in camera AST FrameSet:\n{ast_frame.show()}.")
        if self._focal_plane_frame_id == 0:
            raise ValueError("No FOCAL_PLANE frame in camera AST FrameSet.")
        self._focal_plane_frame = _frames.FocalPlaneFrame(
            instrument=instrument,
            unit=u.Unit(self._ast.getFrame(self._focal_plane_frame_id, copy=False).getUnit(1)),
        )
        self._field_angle_frame = _frames.FieldAngleFrame(instrument=instrument)
        if self._field_angle_frame_id == 0:
            raise ValueError("No FIELD_ANGLE frame in camera AST FrameSet.")

    @property
    def instrument(self) -> str:
        """Name of the instrument (`str`)."""
        return self._focal_plane_frame.instrument

    def focal_plane(self, visit: int | None = None) -> _frames.FocalPlaneFrame:
        """Return a focal plane frame for this instrument.

        Parameters
        ----------
        visit
            ID for the visit this frame will correspond to.  This only needs
            to be provided in contexts where camera frames will be related to
            the sky via a `Projection`.
        """
        if visit is None:
            return self._focal_plane_frame
        else:
            return self._focal_plane_frame.model_copy(update={"visit": visit})

    def field_angle(self, visit: int | None = None) -> _frames.FieldAngleFrame:
        """Return a field angle frame for this instrument.

        Parameters
        ----------
        visit
            ID for the visit this frame will correspond to.  This only needs
            to be provided in contexts where camera frames will be related to
            the sky via a `Projection`.
        """
        if visit is None:
            return self._field_angle_frame
        else:
            return self._field_angle_frame.model_copy(update={"visit": visit})

    def detector(self, detector: int, *, visit: int | None = None) -> _frames.DetectorFrame:
        """Return a detector pixel-coordinate frame for this instrument.

        Parameters
        ----------
        detector
            ID of the detector.
        visit
            ID for the visit this frame will correspond to.  This only needs
            to be provided in contexts where camera frames will be related to
            the sky via a `Projection`.
        """
        try:
            frame_id = self._detector_frame_ids[detector]
        except KeyError:
            raise FrameLookupError(
                f"No frame for detector {detector!r} in camera for {self.instrument!r}."
            ) from None
        ast_frame = self._ast.getFrame(frame_id, copy=False)
        bbox = Box.factory[
            int(ast_frame.getBottom(2)) : int(ast_frame.getTop(2)),
            int(ast_frame.getBottom(1)) : int(ast_frame.getTop(1)),
        ]
        return _frames.DetectorFrame(instrument=self.instrument, detector=detector, visit=visit, bbox=bbox)

    def __contains__(self, frame: _frames.Frame) -> bool:
        try:
            self._parse_frame_arg(frame)
            return True
        except FrameLookupError:
            return False

    def __getitem__[I: _frames.Frame, O: _frames.Frame](self, key: tuple[I, O]) -> Transform[I, O]:
        in_frame, out_frame = key
        in_frame_id, in_bounds = self._parse_frame_arg(in_frame)
        out_frame_id, out_bounds = self._parse_frame_arg(out_frame)
        return Transform(
            in_frame,
            out_frame,
            self._ast.getMapping(in_frame_id, out_frame_id),
            in_bounds=in_bounds,
            out_bounds=out_bounds,
        )

    def _parse_frame_arg(self, frame: _frames.Frame) -> tuple[int, Bounds | None]:
        bounds: Bounds | None = None
        match frame:
            case _frames.DetectorFrame(instrument=self.instrument, detector=detector_id):
                try:
                    frame_id = self._detector_frame_ids[detector_id]
                except KeyError:
                    raise FrameLookupError(
                        f"No frame for detector {detector_id!r} in camera for {self.instrument!r}."
                    ) from None
                bounds = frame.bbox
            case _frames.FocalPlaneFrame(instrument=self.instrument):
                frame_id = self._focal_plane_frame_id
            case _frames.FieldAngleFrame(instrument=self.instrument):
                frame_id = self._field_angle_frame_id
            case _:
                raise FrameLookupError(f"Invalid frame for camera {self.instrument}: {frame!r}.")
        return frame_id, bounds

    def serialize(self, archive: OutputArchive[Any]) -> CameraFrameSetSerializationModel:
        """Serialize the frame set to an archive.

        Parameters
        ----------
        archive
            Archive to serialize to.

        Returns
        -------
        `CameraFrameSetSerializationModel`
            Serialized form of the frame set.
        """
        return CameraFrameSetSerializationModel(instrument=self.instrument, ast=self._ast.show())

    @staticmethod
    def deserialize(model: CameraFrameSetSerializationModel, archive: InputArchive[Any]) -> CameraFrameSet:
        """Deserialize a frame set from an archive.

        Parameters
        ----------
        model
            Seralized form of the frame set.
        archive
            Archive to read from.
        """
        import astshim

        return CameraFrameSet(model.instrument, astshim.FrameSet.fromString(model.ast))

    @classmethod
    def from_legacy(cls, camera: Any) -> CameraFrameSet:
        """Construct a transform from a legacy `lsst.afw.cameraGeom.Camera`.

        Parameters
        ----------
        sky_wcs
            Legacy WCS object.
        pixel_frame
            Coordinate frame for the pixel grid.
        pixel_bounds
            The region that bounds valid pixels for this transform.
        """
        transform_map = camera.getTransformMap()
        ast_frame_set = transform_map.makeFrameSet(list(camera))
        return CameraFrameSet("HSC", ast_frame_set)


class CameraFrameSetSerializationModel(pydantic.BaseModel):
    """Serialization model for `CameraFrameSet`."""

    instrument: str = pydantic.Field(description="Name of the instrument.")
    ast: str = pydantic.Field(
        description="A serialized Starlink AST FrameSet, using the AST native encoding."
    )

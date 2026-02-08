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
    "ObservationFrameSet",
    "ProjectionFrameSet",
    "SkyFrameSet",
    "TractFrameSet",
)

from abc import abstractmethod
from typing import cast, overload

from . import _frames  # use this import style to facilitate pattern matching
from ._frame_set import FrameLookupError, FrameSet
from ._projection import Projection
from ._transform import Transform


class ProjectionFrameSet(FrameSet):
    def __getitem__[I: _frames.Frame, O: _frames.Frame](self, key: tuple[I, O]) -> Transform[I, O]:
        in_frame, out_frame = key
        if in_frame is _frames.SkyFrame.ICRS:
            if out_frame is _frames.SkyFrame.ICRS:
                return Transform.identity(_frames.SkyFrame.ICRS)
            else:
                # MyPy doesn't see that I must be SkyFrame
                return cast(Transform[I, O], self._get_projection(out_frame).sky_to_pixel_transform)
        else:
            if out_frame is _frames.SkyFrame.ICRS:
                return cast(Transform[I, O], self._get_projection(in_frame).pixel_to_sky_transform)
            else:
                return self._get_projection(in_frame).pixel_to_sky_transform.then(
                    self._get_projection(out_frame).sky_to_pixel_transform,
                    remember_components=False,
                )

    def __contains__(self, frame: _frames.Frame) -> bool:
        try:
            self._get_projection(frame)
            return True
        except FrameLookupError:
            return False

    @abstractmethod
    def _get_projection[F: _frames.Frame](self, frame: F) -> Projection[F]:
        raise NotImplementedError()


class TractFrameSet(ProjectionFrameSet):
    def __init__(self, skymap: str, projections: dict[int, Projection[_frames.TractFrame]]):
        self._skymap = skymap
        self._projections = projections

    @property
    def skymap(self) -> str:
        return self._skymap

    def tract(self, tract: int) -> _frames.TractFrame:
        try:
            return self._projections[tract].pixel_frame
        except KeyError:
            raise FrameLookupError(f"No frame for tract {tract!r} in skymap {self._skymap!r}.") from None

    def projection(self, frame: _frames.TractFrame | int) -> Projection[_frames.TractFrame]:
        match frame:
            case int(tract_id) | _frames.TractFrame(skymap=self._skymap, tract=tract_id):
                try:
                    return self._projections[tract_id]
                except KeyError:
                    raise FrameLookupError(
                        f"No frame for tract {tract_id!r} in skymap {self._skymap!r}."
                    ) from None
            case _:
                raise FrameLookupError(f"Invalid frame for skymap {self._skymap}: {frame!r}.")

    def _get_projection[F: _frames.Frame](self, frame: F) -> Projection[F]:
        return self.projection(frame)  # type: ignore


class ObservationFrameSet(ProjectionFrameSet):
    def __init__(self, instrument: str, projections: dict[int, dict[int, Projection[_frames.DetectorFrame]]]):
        self._instrument = instrument
        self._projections = projections

    @property
    def instrument(self) -> str:
        return self._instrument

    def detector(self, *, detector: int, visit: int) -> _frames.DetectorFrame:
        try:
            return self._projections[visit][detector].pixel_frame
        except KeyError:
            raise FrameLookupError(
                f"No projection for visit={visit!r}, detector={detector!r} in "
                f"observations for instrument {self._instrument!r}."
            ) from None

    @overload
    def projection(self, frame: _frames.DetectorFrame) -> Projection[_frames.DetectorFrame]: ...

    @overload
    def projection(self, *, detector: int, visit: int) -> Projection[_frames.DetectorFrame]: ...

    def projection(
        self,
        frame: _frames.DetectorFrame | None = None,
        *,
        detector: int | None = None,
        visit: int | None = None,
    ) -> Projection:
        match (frame, detector, visit):
            case (None, int(), int()):
                frame = self.detector(detector=detector, visit=visit)
            case (_frames.DetectorFrame(instrument=self._instrument), None, None):
                pass
            case _:
                raise FrameLookupError(
                    f"Invalid arguments for {self._instrument} observation: {frame=}, {detector=}, {visit=}."
                )
        return self._get_projection(frame)

    def _get_projection[F: _frames.Frame](self, frame: F) -> Projection[F]:
        return self.projection(frame)  # type: ignore


class SkyFrameSet(ProjectionFrameSet):
    def __init__(
        self,
        *,
        skymaps: dict[str, TractFrameSet],
        instruments: dict[str, ObservationFrameSet],
    ):
        self.skymaps = skymaps
        self.instruments = instruments

    @property
    def instrument(self) -> str:
        try:
            (instrument,) = self.instruments
        except ValueError:
            raise ValueError(
                f"FrameSet does not have a single unique instrument: {self.instruments}."
            ) from None
        return instrument

    @property
    def skymap(self) -> str:
        try:
            (skymap,) = self.skymaps
        except ValueError:
            raise ValueError(f"FrameSet does not have a single unique skymap: {self.skymaps}.") from None
        return skymap

    def detector(self, *, detector: int, visit: int, instrument: str | None = None) -> _frames.DetectorFrame:
        if instrument is None:
            instrument = self.instrument
        try:
            return self.instruments[instrument].detector(detector=detector, visit=visit)
        except KeyError:
            raise FrameLookupError(f"No camera and no observations for instrument {instrument!r}.") from None

    def tract(self, tract: int, *, skymap: str | None = None) -> _frames.TractFrame:
        if skymap is None:
            skymap = self.skymap
        try:
            return self.skymaps[skymap].tract(tract)
        except KeyError:
            raise FrameLookupError(f"No skymap {skymap!r}.")

    def _get_projection[F: _frames.Frame](self, frame: F) -> Projection[F]:
        match frame:
            case _frames.DetectorFrame(instrument=instrument) if (
                observations := self.instruments.get(instrument)
            ) is not None:
                return observations._get_projection(frame)
            case _frames.TractFrame(skymap=skymap) if (tracts := self.skymaps.get(skymap)) is not None:
                return tracts._get_projection(frame)
            case _:
                raise FrameLookupError(f"Failed to find projection for frame: {frame!r}.")

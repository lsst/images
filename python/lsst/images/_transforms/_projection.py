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

__all__ = ("Projection",)

from typing import Any, Self

import astropy.units as u
import numpy as np
from astropy.coordinates import ICRS, Latitude, Longitude, SkyCoord
from astropy.wcs.wcsapi import BaseLowLevelWCS, HighLevelWCSMixin

from .._geom import XY, YX, Box
from ._frames import Frame, SkyFrame
from ._transform import Transform, _ast_apply


class Projection[P: Frame]:
    """A transform that maps pixel coordinates to sky coordinates."""

    def __init__(self, pixel_to_sky: Transform[P, SkyFrame]):
        self._pixel_to_sky = pixel_to_sky
        if pixel_to_sky.in_frame.unit != u.pix:
            raise ValueError("Transform is not a mapping from pixel coordinates.")
        if pixel_to_sky.out_frame != SkyFrame.ICRS:
            raise ValueError("Transform is not a mapping to ICRS.")

    @property
    def pixel_frame(self) -> P:
        return self._pixel_to_sky.in_frame

    @property
    def sky_frame(self) -> SkyFrame:
        return self._pixel_to_sky.out_frame

    @property
    def pixel_to_sky_transform(self) -> Transform[P, SkyFrame]:
        return self._pixel_to_sky

    @property
    def sky_to_pixel_transform(self) -> Transform[SkyFrame, P]:
        return self._pixel_to_sky.inverted()

    def pixel_to_sky[T: np.ndarray | float](self, *, x: T, y: T) -> SkyCoord:
        sky_rad = self._pixel_to_sky.apply_forward(x=x, y=y)
        return SkyCoord(ra=sky_rad.x, dec=sky_rad.y, unit=u.rad)

    def sky_to_pixel(self, sky: SkyCoord) -> XY[np.ndarray | float]:
        if sky.frame.name != "icrs":
            sky = sky.transform_to("icrs")
        ra: Longitude = sky.ra
        dec: Latitude = sky.dec
        return self._pixel_to_sky.apply_inverse(
            x=ra.to_value(u.rad),
            y=dec.to_value(u.rad),
        )

    def as_astropy(self, bbox: Box | None = None) -> ProjectionAstropyView:
        return ProjectionAstropyView(self._pixel_to_sky, bbox)


class ProjectionAstropyView(BaseLowLevelWCS, HighLevelWCSMixin):
    def __init__(self, ast_pixel_to_sky: Any, bbox: Box | None):
        self._bbox = bbox
        if bbox is not None:
            import astshim

            ast_pixel_to_sky = astshim.ShiftMap(list(bbox.start.xy)).then(ast_pixel_to_sky)
        self._ast_pixel_to_sky = ast_pixel_to_sky

    @property
    def low_level_wcs(self) -> Self:
        return self

    @property
    def array_shape(self) -> YX[int] | None:
        return self._bbox.shape if self._bbox is not None else None

    @property
    def axis_correlation_matrix(self) -> np.ndarray:
        return np.array([[True, True], [True, True]])

    @property
    def pixel_axis_names(self) -> XY[str]:
        return XY("x", "y")

    @property
    def pixel_bounds(self) -> XY[tuple[int, int]] | None:
        if self._bbox is None:
            return None
        return XY((self._bbox.x.min, self._bbox.x.max), (self._bbox.y.min, self._bbox.y.max))

    @property
    def pixel_n_dim(self) -> int:
        return 2

    @property
    def pixel_shape(self) -> XY[int] | None:
        array_shape = self.array_shape
        return array_shape.xy if array_shape is not None else None

    @property
    def serialized_classes(self) -> bool:
        return False

    @property
    def world_axis_names(self) -> tuple[str, str]:
        return ("ra", "dec")

    @property
    def world_axis_object_classes(self) -> dict[str, tuple[type[SkyCoord], tuple[()], dict[str, Any]]]:
        return {"celestial": (SkyCoord, (), {"frame": ICRS, "unit": (u.rad, u.rad)})}

    @property
    def world_axis_object_components(self) -> list[tuple[str, int, str]]:
        return [("celestial", 0, "spherical.lon.radian"), ("celestial", 1, "spherical.lat.radian")]

    @property
    def world_axis_physical_types(self) -> tuple[str, str]:
        return ("pos.eq.ra", "pos.eq.dec")

    @property
    def world_axis_units(self) -> tuple[str, str]:
        return ("rad", "rad")

    @property
    def world_n_dim(self) -> int:
        return 2

    def pixel_to_world_values(self, x: np.ndarray, y: np.ndarray) -> XY[np.ndarray]:
        return _ast_apply(self._ast_pixel_to_sky.applyForward, x=x, y=y)

    def world_to_pixel_values(self, ra: np.ndarray, dec: np.ndarray) -> XY[np.ndarray]:
        return _ast_apply(self._ast_pixel_to_sky.applyInverse, x=ra, y=dec)

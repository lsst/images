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

__all__ = ("PiffWrapper",)

from functools import cached_property
from typing import Any

import numpy as np

from .._geom import Box, Domain
from .._image import Image
from ._base import PointSpreadFunction


class PiffWrapper(PointSpreadFunction):
    """A PSF backed by the Piff library."""

    def __init__(self, impl: Any, domain: Domain, stamp_size: int):
        self._impl = impl
        self._domain = domain
        self._stamp_size = stamp_size

    @property
    def domain(self) -> Domain:
        return self._domain

    @cached_property
    def kernel_bbox(self) -> Box:
        r = self._stamp_size // 2
        return Box.factory[-r : r + 1, -r : r + 1]

    def compute_kernel_image(self, *, x: float, y: float) -> Image:
        if "colorValue" in self._impl.interp_property_names:
            raise NotImplementedError("Chromatic PSFs are not yet supported.")
        gs_image = self._impl.draw(x, y, stamp_size=self._stamp_size, center=True)
        r = self._stamp_size // 2
        result = Image(gs_image.array.copy(), start=(-r, -r))
        result.array /= np.sum(result.array)
        return result

    def compute_stellar_image(self, *, x: float, y: float) -> Image:
        if "colorValue" in self._impl.interp_property_names:
            raise NotImplementedError("Chromatic PSFs are not yet supported.")
        gs_image = self._impl.draw(x, y, stamp_size=self._stamp_size, center=None)
        r = self._stamp_size // 2
        result = Image(gs_image.array.copy(), start=(round(y) - r, round(x) - r))
        result.array /= np.sum(result.array)
        return result

    def compute_stellar_bbox(self, *, x: float, y: float) -> Box:
        r = self._stamp_size // 2
        xi = round(x)
        yi = round(y)
        return Box.factory[yi - r : yi + r + 1, xi - r : xi + r + 1]

    @property
    def piff_psf(self) -> Any:
        """The backing `piff.PSF` object.

        This is an internal object that must not be modified in place.
        """
        return self._impl

    @classmethod
    def from_legacy(cls, legacy_psf: Any, domain: Domain) -> PiffWrapper:
        return cls(impl=legacy_psf._piffResult, domain=domain, stamp_size=legacy_psf.width)

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

__all__ = ("LegacyPointSpreadFunction",)

from functools import cached_property
from typing import Any

import numpy as np

from .._geom import Box, Domain
from .._image import Image
from ._base import PointSpreadFunction


class LegacyPointSpreadFunction(PointSpreadFunction):
    """A PSF backed by legacy `lsst.afw` code."""

    def __init__(self, impl: Any, domain: Domain):
        self._impl = impl
        self._domain = domain

    @property
    def domain(self) -> Domain:
        return self._domain

    @cached_property
    def kernel_bbox(self) -> Box:
        from lsst.geom import Box2I, Point2D

        biggest = Box2I()
        for y, x in self._domain.boundary():
            biggest.include(self._impl.computeKernelBBox(Point2D(x, y)))
        return Box.from_legacy(biggest)

    def compute_kernel_image(self, *, x: float, y: float) -> Image:
        from lsst.geom import Point2D

        result = Image.from_legacy(self._impl.computeKernelImage(Point2D(x, y)))
        if result.bbox != self.kernel_bbox:
            # afw does not guarantee a consistent kernel_bbox, but we do now.
            padded = Image(0.0, bbox=self.kernel_bbox, dtype=np.float64)
            padded[self.kernel_bbox] = result[self.kernel_bbox]
            result = padded
        return result

    def compute_stellar_image(self, *, x: float, y: float) -> Image:
        from lsst.geom import Point2D

        return Image.from_legacy(self._impl.computeImage(Point2D(x, y)))

    def compute_stellar_bbox(self, *, x: float, y: float) -> Box:
        from lsst.geom import Point2D

        return Box.from_legacy(self._impl.computeImageBBox(Point2D(x, y)))

    @property
    def legacy_psf(self) -> Any:
        """The backing `lsst.afw.detection.Psf` object.

        This is an internal object that must not be modified in place.
        """
        return self._impl

    @classmethod
    def from_legacy(cls, legacy_psf: Any, domain: Domain) -> LegacyPointSpreadFunction:
        return cls(impl=legacy_psf, domain=domain)

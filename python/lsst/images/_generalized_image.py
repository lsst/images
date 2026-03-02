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

__all__ = ("GeneralizedImage",)

from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, Self

import astropy.wcs

from ._geom import Box
from ._transforms import Projection, ProjectionAstropyView


class GeneralizedImage(ABC):
    """A base class for types that represent one or more 2-d image-like arrays
    with the same pixel grid and projection.
    """

    @property
    @abstractmethod
    def bbox(self) -> Box:
        """Bounding box for the image (`Box`)."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def projection(self) -> Projection[Any] | None:
        """The projection that maps this image's pixel grid to the sky
        (`Projection` | `None`).

        Notes
        -----
        The pixel coordinates used by this projection account for the bounding
        box ``start``; they are not just array indices.
        """
        raise NotImplementedError()

    @property
    def astropy_wcs(self) -> ProjectionAstropyView | None:
        """An Astropy WCS for this image's pixel array.

        Notes
        -----
        As expected for Astropy WCS objects, this defines pixel coordinates
        such that the first row and column in any associated arrays are
        ``(0, 0)``, not ``bbox.start``, as is the case for `projection`.

        This object satisfies the `astropy.wcs.wcsapi.BaseHighLevelWCS` and
        `astropy.wcs.wcsapi.BaseLowLevelWCS` interfaces, but it is not an
        `astropy.wcs.WCS` (use `fits_wcs` for that).
        """
        return self.projection.as_astropy(self.bbox) if self.projection is not None else None

    @cached_property
    def fits_wcs(self) -> astropy.wcs.WCS | None:
        """An Astropy FITS WCS for this image's pixel array.

        Notes
        -----
        As expected for Astropy WCS objects, this defines pixel coordinates
        such that the first row and column in any associated arrays are
        ``(0, 0)``, not ``bbox.start``, as is the case for `projection`.

        This may be an approximation or absent if `projection` is not
        naturally representable as a FITS WCS.
        """
        return (
            self.projection.as_fits_wcs(self.bbox, allow_approximation=True)
            if self.projection is not None
            else None
        )

    @abstractmethod
    def __getitem__(self, bbox: Box) -> Self:
        raise NotImplementedError()

    @abstractmethod
    def copy(self) -> Self:
        """Deep-copy the image."""
        raise NotImplementedError()

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

__all__ = ("PointSpreadFunction",)

from abc import ABC, abstractmethod
from typing import Any

from .._geom import Box, Domain
from .._image import Image


class PointSpreadFunction(ABC):
    """Base class for point-spread function models."""

    @property
    @abstractmethod
    def domain(self) -> Domain:
        """The region where this PSF model is valid."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def kernel_bbox(self) -> Box:
        """Bounding box of all images returned by `compute_kernel_image`."""
        raise NotImplementedError()

    @abstractmethod
    def compute_kernel_image(self, *, x: float, y: float) -> Image:
        """Evaluate the PSF model into an image suitable for convolution.

        Parameters
        ----------
        x
            Column position coordinate to evaluate at.
        y
            Row position coordinate to evaluate at.

        Returns
        -------
        Image
            An image of the PSF, centered on the center of the center pixel,
            which is defined to be ``(0, 0)`` by the image's origin.
        """
        raise NotImplementedError()

    @abstractmethod
    def compute_stellar_image(self, *, x: float, y: float) -> Image:
        """Evaluate the PSF model into an image suitable for comparison with
        the image of an astrophysical point source.

        Parameters
        ----------
        x
            Column position coordinate to evaluate at.
        y
            Row position coordinate to evaluate at.

        Returns
        -------
        Image
            An image of the PSF, centered on the given coordinates, just like
            the postage stamp of a star would be.
        """
        raise NotImplementedError()

    @abstractmethod
    def compute_stellar_bbox(self, *, x: float, y: float) -> Box:
        """Return the bounding box of the image that would be returned by
        `compute_stellar_image`.

        Parameters
        ----------
        x
            Column position coordinate to evaluate at.
        y
            Row position coordinate to evaluate at.

        Returns
        -------
        Box
            The bounding box of the image that would be returned by
            `compute_stellar_image` at the given point.
        """
        raise NotImplementedError()

    @classmethod
    def from_legacy(cls, legacy_psf: Any, domain: Domain) -> PointSpreadFunction:
        """Make a PSF object from a legacy `lsst.afw.detection.Psf` instance.

        Parameter
        ---------
        legacy_psf
            Legacy PSF object.
        domain
            The region where this PSF model is valid.

        Returns
        -------
        PointSpreadFunction
            A `PointSpreadFunction` instance.

        Notes
        -----
        This base class method is a factory dispatch function that
        automatically selects the right `PointSpreadFunction` subclass to
        use.  When that is already known, a subclass `from_legacy` method can
        be called instead.
        """
        from lsst.afw.detection import Psf
        from lsst.meas.extensions.piff.piffPsf import PiffPsf

        match legacy_psf:
            case PiffPsf():
                from ._piff import PiffWrapper

                return PiffWrapper.from_legacy(legacy_psf, domain)
            case Psf():
                from ._legacy import LegacyPointSpreadFunction

                return LegacyPointSpreadFunction.from_legacy(legacy_psf, domain)
            case _:
                raise TypeError(f"{type(legacy_psf).__name__!r} is not a recognized legacy PSF type.")

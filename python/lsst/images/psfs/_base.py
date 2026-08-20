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

import numpy as np

from .._ellipses import Ellipse
from .._geom import Bounds, Box
from .._image import Image
from ..describe import DescribableMixin, DescribeOptions, FieldRole, Report, ReportField


class PointSpreadFunction(DescribableMixin, ABC):
    """Base class for point-spread function models."""

    @property
    @abstractmethod
    def bounds(self) -> Bounds:
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

    def compute_moments(
        self, *, x: float, y: float, adaptive: bool = True, use_stellar_image: bool = False
    ) -> Ellipse:
        """Compute the moments of the PSF image.

        Parameters
        ----------
        x
            Column position coordinate to measure the PSF at.
        y
            Row position coordinate to measure the PSF at.
        adaptive
            If `True`, use GalSim's HSC adaptive moments algorithm (via
            `.Ellipse.remeasure_adaptive`).  If `False`, use unweighted
            moments, which should be stable for nearly-noiseless PSF models
            but not for the (noisy) stars PSF models are generally compared
            to.
        use_stellar_image
            If `True`, measure the moments of a call to `compute_stellar_image`
            instead of `compute_kernel_image`.  This also causes the center
            of the returned ellipse to be near the given ``(x, y)`` instead of
            near ``(0, 0)``, reflecting the different coordinate conventions
            of stellar vs. kernel images.
        """
        if use_stellar_image:
            psf_image = self.compute_stellar_image(x=x, y=y)
        else:
            psf_image = self.compute_kernel_image(x=x, y=y)
            x = 0.0
            y = 0.0
        result = Ellipse.from_image_unweighted(psf_image, x=x, y=y)
        if adaptive:
            result = result.remeasure_adaptive(psf_image)
        return result

    def compute_effective_area(self, *, x: float, y: float, use_stellar_image: bool = False) -> float:
        """Compute the effective area of the PSF model.

        Parameters
        ----------
        x
            Column position coordinate to measure the PSF at.
        y
            Row position coordinate to measure the PSF at.
        use_stellar_image
            If `True`, measure the moments of a call to `compute_stellar_image`
            instead of `compute_kernel_image`.
        """
        if use_stellar_image:
            psf_image = self.compute_stellar_image(x=x, y=y)
        else:
            psf_image = self.compute_kernel_image(x=x, y=y)
        return float(np.sum(psf_image.array) ** 2.0 / np.sum(psf_image.array**2.0))

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
    def from_legacy(cls, legacy_psf: Any, bounds: Bounds) -> PointSpreadFunction:
        """Make a PSF object from a legacy `lsst.afw.detection.Psf` instance.

        Parameters
        ----------
        legacy_psf
            Legacy PSF object.
        bounds
            The region where this PSF model is valid.

        Returns
        -------
        `~lsst.images.psfs.PointSpreadFunction`
            The converted PSF object.

        Notes
        -----
        This base class method is a factory dispatch function that
        automatically selects the right
        `~lsst.images.psfs.PointSpreadFunction` subclass to use.  When that is
        already known, a subclass `from_legacy` method can be called instead.
        """
        from lsst.afw.detection import Psf
        from lsst.cell_coadds import StitchedPsf
        from lsst.meas.extensions.piff.piffPsf import PiffPsf

        match legacy_psf:
            case PiffPsf():
                from ._piff import PiffWrapper

                return PiffWrapper.from_legacy(legacy_psf, bounds)
            case StitchedPsf():
                from ..cells import CellPointSpreadFunction

                return CellPointSpreadFunction.from_legacy(legacy_psf, bounds)
            case Psf():
                from ._legacy import LegacyPointSpreadFunction

                return LegacyPointSpreadFunction.from_legacy(legacy_psf, bounds)
            case _:
                raise TypeError(f"{type(legacy_psf).__name__!r} is not a recognized legacy PSF type.")

    def _describe(self, options: DescribeOptions = DescribeOptions(), /) -> Report:
        """Return a `Report` describing this PSF.

        Parameters
        ----------
        options : `DescribeOptions`, optional
            Unused; accepted for interface compatibility.
        """
        return Report(
            type_name=type(self).__name__,
            summary=f"{type(self).__name__} over {self.bounds}",
            fields=[
                ReportField(label="bounds", value=self.bounds, role=FieldRole.DERIVED),
                ReportField(label="kernel_bbox", value=self.kernel_bbox, role=FieldRole.DERIVED),
            ],
        )

    def to_legacy(self) -> Any:
        """Convert to a legacy `lsst.afw.detection.Psf`, if possible."""
        raise NotImplementedError("This PSF does not support conversion to lsst.afw.detection.Psf.")

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

__all__ = ("AbsoluteSliceProxy", "GeneralizedImage", "LocalSliceProxy")

from abc import ABC, abstractmethod
from functools import cached_property
from types import EllipsisType
from typing import Any, Self, TypeVar

import astropy.wcs

from ._geom import Box
from ._transforms import Projection, ProjectionAstropyView

T = TypeVar("T", bound="GeneralizedImage")  # for sphinx


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

    @property
    def local(self) -> LocalSliceProxy[Self]:
        """A proxy object for slicing a generalized image using "local" or
        "array" pixel coordinates.

        Notes
        -----
        In this convention, the first row and column of the pixel grid is
        always at ``(0, 0)``.  This is also the convention used by
        `astropy.wcs` objects. When a subimage is created from a parent image,
        its "local" coordinate system is offset from the original.

        Note that most `lsst.images` types (e.g. `~lsst.images.Box`,
        `~lsst.images.Projection`, `~lsst.images.psfs.PointSpreadFunction`)
        operate instead in "absolute" coordinates, which is shared by subimage
        and their parents.

        See Also
        --------
        lsst.images.BoxSliceFactory
        lsst.images.IntervalSliceFactory
        """
        return LocalSliceProxy(self)

    @property
    def absolute(self) -> AbsoluteSliceProxy[Self]:
        """A proxy object for slicing a generalized image using absolute pixel
        coordinates.

        Notes
        -----
        In this convention, the first row and column of the pixel grid is
        ``bbox.start``.  A subimage and its parent image share the same
        absolute pixel coordinate system, and most `lsst.images` types (e.g.
        `~lsst.images.Box`, `~lsst.images.Projection`,
        `~lsst.images.psfs.PointSpreadFunction`) operate exclusively in this
        system.

        Note that `astropy.wcs` and `numpy.ndarray` are not aware of the
        ``bbox.start`` offset that defines tihs coordinates system; use
        `local` slicing for indices obtained from those.

        See Also
        --------
        lsst.images.BoxSliceFactory
        lsst.images.IntervalSliceFactory
        """
        return AbsoluteSliceProxy(self)

    # Subclasses should delegate to super().__getitem__ for some user-friendly
    # argument type-checking before providing their own implementation.
    @abstractmethod
    def __getitem__(self, bbox: Box | EllipsisType) -> Self:
        if not isinstance(bbox, Box):
            raise TypeError(
                "Only Box objects can be used to subset image objects directly; "
                "use .local[y, x] or .absolute[y, x] proxies for slice-based subsets."
            )
        return self

    @abstractmethod
    def copy(self) -> Self:
        """Deep-copy the image."""
        raise NotImplementedError()


class LocalSliceProxy[T: GeneralizedImage]:
    """A proxy object for obtraining a generalized image subset using
    local slicing.

    See `GeneralizedImage.local` for more information.
    """

    def __init__(self, parent: T):
        self._parent = parent

    def __getitem__(self, slices: tuple[slice, slice]) -> T:
        return self._parent[self._parent.bbox.local[slices]]


class AbsoluteSliceProxy[T: GeneralizedImage]:
    """A proxy object for obtraining a generalized image subset using
    absolute slicing.

    See `GeneralizedImage.absolute` for more information.
    """

    def __init__(self, parent: T):
        self._parent = parent

    def __getitem__(self, slices: tuple[slice, slice]) -> T:
        return self._parent[self._parent.bbox.absolute[slices]]

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
from typing import TYPE_CHECKING, Any, Self, TypeVar

import astropy.wcs

from ._geom import Box
from ._transforms import Projection, ProjectionAstropyView
from .serialization import ArchiveTree, ButlerInfo, MetadataValue, OpaqueArchiveMetadata

if TYPE_CHECKING:
    from lsst.daf.butler import DatasetProvenance, SerializedDatasetRef


T = TypeVar("T", bound="GeneralizedImage")  # for sphinx


class GeneralizedImage(ABC):
    """A base class for types that represent one or more 2-d image-like arrays
    with the same pixel grid and projection.

    Parameters
    ----------
    metadata
        Arbitrary flexible metadata to associate with the image.
    """

    def __init__(self, metadata: dict[str, MetadataValue] | None = None):
        self._metadata = metadata if metadata is not None else {}
        self._opaque_metadata: OpaqueArchiveMetadata | None = None
        self._butler_info: ButlerInfo | None = None

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
        its "local" coordinate system is offset from the coordinate systems of
        the parent image.

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

    @property
    def metadata(self) -> dict[str, MetadataValue]:
        """Arbitrary flexible metadata associated with the image (`dict`).

        Notes
        -----
        Metadata is shared with subimages and other views.  It can be
        disconnected by reassigning to a copy explicitly:

            image.metadata = image.metadata.copy()
        """
        return self._metadata

    @metadata.setter
    def metadata(self, value: dict[str, MetadataValue]) -> None:
        self._metadata = value

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
        """Deep-copy the image and metadata.

        Attached immutable objects (like `Projection` instances) are not
        copied.
        """
        raise NotImplementedError()

    @property
    def butler_dataset(self) -> SerializedDatasetRef | None:
        """The butler dataset reference for this image
        (`lsst.daf.butler.SerializedDatasetRef` | `None`).
        """
        if self._butler_info is None:
            return None
        from lsst.daf.butler import SerializedDatasetRef

        # Guard against the unlikely case where the dataset was deserialized as
        # Any because `lsst.daf.butler` couldn't be imported before, but can be
        # imported now (*anything* can happen in Jupyter).
        return SerializedDatasetRef.model_validate(self._butler_info.dataset)

    @property
    def butler_provenance(self) -> DatasetProvenance | None:
        """The butler inputs and ID of the task quantum that produced this
        dataset (`lsst.daf.butler.DatasetProvenance` | `None`)
        """
        if self._butler_info is None:
            return None

        # Guard against the unlikely case where the provenance was deserialized
        # as Any because `lsst.daf.butler` couldn't be imported before, but can
        # be imported now (*anything* can happen in Jupyter).
        from lsst.daf.butler import DatasetProvenance

        return DatasetProvenance.model_validate(self._butler_info.provenance)

    def _transfer_metadata(self, new: Self, copy: bool = False, bbox: Box | None = None) -> Self:
        """Transfer metadata held by this base class to a new instance.

        Parameters
        ----------
        new
            New instance to modify and return.
        copy
            Whether the new instance is a deep-copy of ``self``.
        bbox
            Bounding box used to construct ``new`` as a subset of ``self``.

        Returns
        -------
        GeneralizedImage
            The new object passed in, modified in place.

        Notes
        -----
        This is a utility method for subclasses to use when finishing
        construction of a new one.
        """
        if bbox is not None:
            opaque_metadata = (
                self._opaque_metadata.subset(bbox) if self._opaque_metadata is not None else None
            )
        else:
            opaque_metadata = self._opaque_metadata
        metadata = self._metadata
        if copy:
            metadata = metadata.copy()
            opaque_metadata = opaque_metadata.copy() if opaque_metadata is not None else None
        new._metadata = metadata
        new._opaque_metadata = opaque_metadata
        new._butler_info = self._butler_info
        return new

    def _finish_deserialize(self, model: ArchiveTree) -> Self:
        """Attach generic information from `ArchiveTree` to this instance
        at the end of deserialization.
        """
        self._metadata = model.metadata
        self._butler_info = model.butler_info
        return self


class LocalSliceProxy[T: GeneralizedImage]:
    """A proxy object for obtaining a generalized image subset using local
    slicing.

    See `GeneralizedImage.local` for more information.
    """

    def __init__(self, parent: T):
        self._parent = parent

    def __getitem__(self, slices: tuple[slice, slice]) -> T:
        try:
            return self._parent[self._parent.bbox.local[slices]]
        except TypeError as err:
            if hasattr(self._parent, "array"):
                err.add_note("The .array attribute may provide more slicing flexibility.")
            raise


class AbsoluteSliceProxy[T: GeneralizedImage]:
    """A proxy object for obtaining a generalized image subset using absolute
    slicing.

    See `GeneralizedImage.absolute` for more information.
    """

    def __init__(self, parent: T):
        self._parent = parent

    def __getitem__(self, slices: tuple[slice, slice]) -> T:
        try:
            return self._parent[self._parent.bbox.absolute[slices]]
        except TypeError as err:
            if hasattr(self._parent, "array"):
                err.add_note(
                    "The .array attribute may provide more slicing flexibility "
                    "(but only works in local coordinates)."
                )
            raise

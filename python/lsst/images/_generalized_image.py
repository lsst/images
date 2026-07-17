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

import astropy.units as u
import astropy.wcs

import starlink.Ast as Ast
from lsst.resources import ResourcePathExpression

from ._geom import YX, Box, NoOverlapError
from ._transforms import SkyProjection, SkyProjectionAstropyView
from .serialization import (
    ArchiveTree,
    ButlerInfo,
    MetadataValue,
    OpaqueArchiveMetadata,
    read_archive,
    write_archive,
)

if TYPE_CHECKING:
    from lsst.daf.butler import DatasetProvenance, SerializedDatasetRef


T = TypeVar("T", bound="GeneralizedImage")  # for sphinx


class GeneralizedImage(ABC):
    """A base class for types that represent one or more 2-d image-like arrays
    with the same pixel grid and sky projection.

    Parameters
    ----------
    metadata
        Arbitrary flexible metadata to associate with the image.
    """

    def __init__(self, metadata: dict[str, MetadataValue] | None = None) -> None:
        self._metadata = metadata if metadata is not None else {}
        self._opaque_metadata: OpaqueArchiveMetadata | None = None
        self._butler_info: ButlerInfo | None = None

    @property
    @abstractmethod
    def bbox(self) -> Box:
        """Bounding box for the image (`~lsst.images.Box`)."""
        raise NotImplementedError()

    @property
    def yx0(self) -> YX[int]:
        """The coordinates of the first pixel in the array
        (`~lsst.geom.YX` [`int`]).
        """
        return self.bbox.start

    @property
    @abstractmethod
    def sky_projection(self) -> SkyProjection[Any] | None:
        """The projection that maps this image's pixel grid to the sky
        (`~lsst.images.SkyProjection` | `None`).

        Notes
        -----
        The pixel coordinates used by this projection account for the bounding
        box ``start``; they are not just array indices.
        """
        raise NotImplementedError()

    @property
    def astropy_wcs(self) -> SkyProjectionAstropyView | None:
        """An Astropy WCS for this image's pixel array.

        Notes
        -----
        As expected for Astropy WCS objects, this defines pixel coordinates
        such that the first row and column in any associated arrays are
        ``(0, 0)``, not ``bbox.start``, as is the case for `sky_projection`.

        This object satisfies the `astropy.wcs.wcsapi.BaseHighLevelWCS` and
        `astropy.wcs.wcsapi.BaseLowLevelWCS` interfaces, but it is not an
        `astropy.wcs.WCS` (use `fits_wcs` for that).
        """
        return self.sky_projection.as_astropy(self.bbox) if self.sky_projection is not None else None

    @cached_property
    def fits_wcs(self) -> astropy.wcs.WCS | None:
        """An Astropy FITS WCS for this image's pixel array.

        Notes
        -----
        As expected for Astropy WCS objects, this defines pixel coordinates
        such that the first row and column in any associated arrays are
        ``(0, 0)``, not ``bbox.start``, as is the case for `sky_projection`.

        This may be an approximation or absent if `sky_projection` is not
        naturally representable as a FITS WCS.
        """
        return (
            self.sky_projection.as_fits_wcs(self.bbox, allow_approximation=True)
            if self.sky_projection is not None
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
        `~lsst.images.SkyProjection`, `~lsst.images.psfs.PointSpreadFunction`)
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
        `~lsst.images.Box`, `~lsst.images.SkyProjection`,
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

    # Subclasses should delegate to _handle_getitem_args for some user-friendly
    # argument type-checking before providing their own implementation.
    @abstractmethod
    def __getitem__(self, bbox: Box | EllipsisType) -> Self:
        raise NotImplementedError()

    @abstractmethod
    def copy(self) -> Self:
        """Deep-copy the image and metadata.

        Attached immutable objects (like `~lsst.images.SkyProjection`
        instances) are not copied.
        """
        raise NotImplementedError()

    @classmethod
    def read(cls, path: ResourcePathExpression, **kwargs: Any) -> Self:
        """Read an instance of this class from a file.

        A thin convenience wrapper around
        `lsst.images.serialization.read_archive` that fixes the expected
        in-memory type to this class.  The container format is inferred
        from ``path``'s extension.

        Parameters
        ----------
        path
            File to read; convertible to `lsst.resources.ResourcePath`.
        **kwargs
            Forwarded to `~lsst.images.serialization.read_archive`
            (e.g. ``bbox`` to read a subimage).
        """
        return read_archive(path, cls, **kwargs)

    def write(self, path: str, **kwargs: Any) -> None:
        """Write this object to a file.

        A thin convenience wrapper around
        `lsst.images.serialization.write_archive`.
        The container format is chosen from ``path``'s extension.

        Parameters
        ----------
        path
            Destination file path.  Must not already exist.
        **kwargs
            Forwarded to `~lsst.images.serialization.write_archive` (e.g.
            ``compression_options`` and ``compression_seed`` for FITS).
        """
        write_archive(self, path, **kwargs)

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
        dataset (`lsst.daf.butler.DatasetProvenance` | `None`).
        """
        if self._butler_info is None:
            return None

        # Guard against the unlikely case where the provenance was deserialized
        # as Any because `lsst.daf.butler` couldn't be imported before, but can
        # be imported now (*anything* can happen in Jupyter).
        from lsst.daf.butler import DatasetProvenance

        return DatasetProvenance.model_validate(self._butler_info.provenance)

    def _transfer_metadata[T: GeneralizedImage](
        self, new: T, copy: bool = False, bbox: Box | None = None
    ) -> T:
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

    def _handle_getitem_args(self, bbox: Box | EllipsisType) -> tuple[Box, YX[slice]]:
        """Interpret the standard arguments to __getitem__."""
        if bbox is ...:
            return self.bbox, YX(y=slice(None), x=slice(None))
        elif not isinstance(bbox, Box):
            raise TypeError(
                "Only Box objects can be used to subset image objects directly; "
                "use .local[y, x] or .absolute[y, x] proxies for slice-based subsets."
            )
        return bbox, bbox.slice_within(self.bbox)

    def bbox_from_sky_circle(
        self, center: astropy.coordinates.SkyCoord, radius: astropy.coordinates.Angle, clip: bool = False
    ) -> Box:
        """Calculate the bounding box on this image corresponding to a circular
        region on the sky.

        Parameters
        ----------
        center
            The center of the circle, as a scalar
            `astropy.coordinates.SkyCoord` in any frame.
        radius
            Radius of the circle, as a scalar `astropy.coordinates.Angle`.
        clip
            If `True` (`False` is default), clip pixel bounds when the circle
            extends outside the image. If `False` an exception is raised if
            any part of the circle is off the image.

        Returns
        -------
        Box
            Bounding box enclosing the circle in this image's pixel
            coordinates.

        Raises
        ------
        NoOverlapError
            Raised if the requested region is entirely off the image or if
            any part of the region is off the image and clipping is `False`.
        ValueError
            Raised if ``center`` or ``radius`` is not scalar, or if this
            image has no sky projection.
        """
        if not center.isscalar:
            raise ValueError("The center of the sky circle must be a scalar SkyCoord.")
        if not radius.isscalar:
            raise ValueError("The radius of the sky circle must be a scalar Angle.")
        center = center.transform_to("icrs")

        # Use pyast directly for the region handling.
        sky_region = Ast.Circle(
            Ast.SkyFrame("System=ICRS"),
            1,
            [center.ra.rad, center.dec.rad],
            [radius.to_value(u.rad)],
        )

        # Get the relevant mapping. If it is not already a pyast mapping
        # (e.g., it is implemented with astshim), convert it by round-tripping
        # the AST textual serialization through a pyast Channel.
        sky_projection = self.sky_projection
        if sky_projection is None:
            raise ValueError("A sky projection is required to calculate a bounding box from a sky region.")
        sky_to_pixel: Any = sky_projection.sky_to_pixel_transform._ast_mapping
        if not isinstance(sky_to_pixel, Ast.Mapping):
            # Comments must be disabled for pyast to be able to parse the
            # astshim serialization.
            sky_to_pixel = Ast.Channel(sky_to_pixel.show(False).splitlines()).read()

        # Calculate the Box around the region.
        pixel_region = sky_region.mapregion(sky_to_pixel, Ast.Frame(2))
        lbnd, ubnd = pixel_region.getregionbounds()
        region_box = Box.from_float_bounds(
            x_min=float(lbnd[0]),
            x_max=float(ubnd[0]),
            y_min=float(lbnd[1]),
            y_max=float(ubnd[1]),
        )

        # Determine the box within the image itself, clipping if requested.
        if clip:
            try:
                region_box = region_box.intersection(self.bbox)
            except NoOverlapError as e:
                e.add_note(
                    f"Requested sky circle has pixel bbox {region_box} which does not overlap {self.bbox}"
                )
                raise
        if not self.bbox.contains(region_box):
            raise NoOverlapError(
                f"Requested sky circle has pixel bbox {region_box}, which is not within {self.bbox}"
            )

        return region_box


class LocalSliceProxy[T: GeneralizedImage]:
    """A proxy object for obtaining a generalized image subset using local
    slicing.

    See `~lsst.images.GeneralizedImage.local` for more information.

    Parameters
    ----------
    parent
        Image the slice proxy operates on.
    """

    def __init__(self, parent: T) -> None:
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

    See `~lsst.images.GeneralizedImage.absolute` for more information.

    Parameters
    ----------
    parent
        Image the slice proxy operates on.
    """

    def __init__(self, parent: T) -> None:
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

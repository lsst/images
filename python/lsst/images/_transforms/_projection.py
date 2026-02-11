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

__all__ = ("Projection", "ProjectionAstropyView", "ProjectionSerializationModel")

import functools
from typing import Any, Self, TypeVar, final

import astropy.units as u
import astropy.wcs
import numpy as np
import pydantic
from astropy.coordinates import ICRS, Latitude, Longitude, SkyCoord
from astropy.wcs.wcsapi import BaseLowLevelWCS, HighLevelWCSMixin

from .._geom import XY, YX, Bounds, Box
from ..serialization import ArchiveTree, InputArchive, OutputArchive
from ..utils import is_none
from ._frames import Frame, SkyFrame
from ._transform import Transform, TransformSerializationModel, _ast_apply

# This pre-python-3.12 declaration is needed by Sphinx (probably the
# autodoc-typehints plugin.
F = TypeVar("F", bound=Frame)


@final
class Projection[F: Frame]:
    """A transform from pixel coordinates to sky coordinates.

    Parameters
    ----------
    pixel_to_sky
        A low-level transform that maps pixel coordinates to sky coordinates.
    fits_approximation
        An approximation to ``pixel_to_sky`` that is guaranteed to have a
        `~Transform.as_fits_wcs` method that does not return `None`.  This
        should not be provided if ``pixel_to_sky`` is itself representable
        as a FITS WCS.
    """

    def __init__(
        self, pixel_to_sky: Transform[F, SkyFrame], fits_approximation: Transform[F, SkyFrame] | None = None
    ):
        self._pixel_to_sky = pixel_to_sky
        if pixel_to_sky.in_frame.unit != u.pix:
            raise ValueError("Transform is not a mapping from pixel coordinates.")
        if pixel_to_sky.out_frame != SkyFrame.ICRS:
            raise ValueError("Transform is not a mapping to ICRS.")
        self._fits_approximation = fits_approximation

    @staticmethod
    def from_fits_wcs(
        fits_wcs: astropy.wcs.WCS,
        pixel_frame: F,
        pixel_bounds: Bounds | None = None,
        x0: int = 0,
        y0: int = 0,
    ) -> Projection[F]:
        """Construct a transform from a FITS WCS.

        Parameters
        ----------
        fits_wcs
            FITS WCS to convert.
        pixel_frame
            Coordinate frame for the pixel grid.
        pixel_bounds
            The region that bounds valid pixels for this transform.
        x0
            Logical coordinate of the first column in the array this WCS
            relates to world coordinates.
        y0
            Logical coordinate of the first column in the array this WCS
            relates to world coordinates.

        Notes
        -----
        The ``x0`` and ``y0`` parameters reflect the fact that for FITS, the
        first row and column are always labeled ``(1, 1)``, while in Astropy
        and most other Python libraries, they are ``(0, 0)``.  The `types` in
        this package (e.g. `Image`, `Mask`) allow them to be any pair of
        integers.

        See Also
        --------
        Transform.from_fits_wcs
        """
        return Projection(
            Transform.from_fits_wcs(
                fits_wcs, pixel_frame, SkyFrame.ICRS, in_bounds=pixel_bounds, x0=x0, y0=y0
            )
        )

    @property
    def pixel_frame(self) -> F:
        """Coordinate frame for the pixel grid."""
        return self._pixel_to_sky.in_frame

    @property
    def sky_frame(self) -> SkyFrame:
        """Coordinate frame for the sky."""
        return self._pixel_to_sky.out_frame

    @property
    def pixel_to_sky_transform(self) -> Transform[F, SkyFrame]:
        """Low-level transform from pixel to sky coordinates (`Transform`)."""
        return self._pixel_to_sky

    @property
    def sky_to_pixel_transform(self) -> Transform[SkyFrame, F]:
        """Low-level transform from sky to pixel coordinates (`Transform`)."""
        return self._pixel_to_sky.inverted()

    @property
    def fits_approximation(self) -> Projection[F] | None:
        """An approximation to this projection that is guaranteed to have an
        `as_fits_wcs` method that does not return `None`.
        """
        return Projection(self._fits_approximation) if self._fits_approximation is not None else None

    def pixel_to_sky[T: np.ndarray | float](self, *, x: T, y: T) -> SkyCoord:
        """Transform one or more pixel points to sky coordinates.

        Parameters
        ----------
        x : `numpy.ndarray` | `float`
            ``x`` values of the pixel points to transform.
        y : `numpy.ndarray` | `float`
            ``y`` values of the pixel points to transform.

        Returns
        -------
        astropy.coordinates.SkyCoord
            Transformed sky coordinates.
        """
        sky_rad = self._pixel_to_sky.apply_forward(x=x, y=y)
        return SkyCoord(ra=sky_rad.x, dec=sky_rad.y, unit=u.rad)

    def sky_to_pixel(self, sky: SkyCoord) -> XY[np.ndarray | float]:
        """Transform one or more sky coordinates to pixels

        Parameters
        ----------
        sky
            Sky coordinates to transform.

        Returns
        -------
        `XY` [`numpy.ndarray` | `float`]
            Transformed pixel coordinates.
        """
        if sky.frame.name != "icrs":
            sky = sky.transform_to("icrs")
        ra: Longitude = sky.ra
        dec: Latitude = sky.dec
        return self._pixel_to_sky.apply_inverse(
            x=ra.to_value(u.rad),
            y=dec.to_value(u.rad),
        )

    def as_astropy(self, bbox: Box | None = None) -> ProjectionAstropyView:
        """Return an `astropy.wcs` view of this `Projection`.

        Parameters
        ----------
        bbox
            Bounding box of the array the view will describe.  This
            projection object is assumed to work on the same coordinate system
            in which ``bbox`` is defined, while the Astropy view will consider
            the first row and column in that box to be ``(0, 0)``.

        Notes
        -----
        This returns an object that satisfies the
        `astropy.wcs.wcsapi.BaseHighLevelWCS` and
        `astropy.wcs.wcsapi.BaseLowLevelWCS` interfaces while evaluating the
        underlying `Projection` itself.  It is *not* an `astropy.wcs.WCS`
        instance, which is a type that also satisfies those interfaces but
        only supports FITS WCS representations (see `as_fits_wcs`).
        """
        return ProjectionAstropyView(self._pixel_to_sky._ast_mapping, bbox)

    def as_fits_wcs(self, bbox: Box, allow_approximation: bool = False) -> astropy.wcs.WCS | None:
        """Return a FITS WCS representation of this projection, if possible.

        Parameters
        ----------
        bbox
            Bounding box of the array the FITS WCS will describe.  This
            transform object is assumed to work on the same coordinate system
            in which ``bbox`` is defined, while the FITS WCS will consider the
            first row and column in that box to be ``(0, 0)`` (in Astropy
            interfaces) or ``(1, 1)`` (in the FITS representation itself).
        allow_approximation
            If `True` and this `Projection` holds a FITS approximation to
            itself, return that approximation.
        """
        if allow_approximation and self._fits_approximation:
            return self._fits_approximation.as_fits_wcs(bbox)
        return self._pixel_to_sky.as_fits_wcs(bbox)

    def serialize[P: pydantic.BaseModel](
        self, archive: OutputArchive[P], *, use_frame_sets: bool = False
    ) -> ProjectionSerializationModel[P]:
        """Serialize a projection to an archive.

        Parameters
        ----------
        archive
            Archive to serialize to.
        use_frame_sets
            If `True`, decompose the underlying transform and try to reference
            component mappings that were already serialized into a `FrameSet`
            in the archive.  The FITS approximation transform is never
            decomposed.

        Returns
        -------
        `ProjectionSerializationModel`
            Serialized form of the projection.
        """
        pixel_to_sky = archive.serialize_direct(
            "pixel_to_sky", functools.partial(self._pixel_to_sky.serialize, use_frame_sets=use_frame_sets)
        )
        fits_approximation = (
            archive.serialize_direct("fits_approximation", self._fits_approximation.serialize)
            if self._fits_approximation is not None
            else None
        )
        return ProjectionSerializationModel(pixel_to_sky=pixel_to_sky, fits_approximation=fits_approximation)

    @staticmethod
    def deserialize[P: pydantic.BaseModel](
        model: ProjectionSerializationModel[P], archive: InputArchive[P]
    ) -> Projection[Any]:
        """Deserialize a projection from an archive.

        Parameters
        ----------
        model
            Seralized form of the projection.
        archive
            Archive to read from.
        """
        pixel_to_sky = Transform.deserialize(model.pixel_to_sky, archive)
        fits_approximation = (
            Transform.deserialize(model.fits_approximation, archive)
            if model.fits_approximation is not None
            else None
        )
        return Projection(pixel_to_sky, fits_approximation=fits_approximation)

    @staticmethod
    def from_legacy(sky_wcs: Any, pixel_frame: F, pixel_bounds: Bounds | None = None) -> Projection[F]:
        """Construct a transform from a legacy `lsst.afw.geom.SkyWcs`.

        Parameters
        ----------
        sky_wcs : `lsst.afw.geom.SkyWcs`
            Legacy WCS object.
        pixel_frame
            Coordinate frame for the pixel grid.
        pixel_bounds
            The region that bounds valid pixels for this transform.
        """
        fits_approximation: Transform[F, SkyFrame] | None = None
        if (legacy_fits_approximation := sky_wcs.getFitsApproximation()) is not None:
            fits_approximation = Transform.from_legacy(
                legacy_fits_approximation.getTransform(), pixel_frame, SkyFrame.ICRS, in_bounds=pixel_bounds
            )
        return Projection(
            Transform.from_legacy(sky_wcs.getTransform(), pixel_frame, SkyFrame.ICRS, in_bounds=pixel_bounds),
            fits_approximation=fits_approximation,
        )


class ProjectionAstropyView(BaseLowLevelWCS, HighLevelWCSMixin):
    """An Astropy-interface view of a `Projection`.

    Notes
    -----
    The constructor of this classe is considered a private implementation
    detail; use `Projection.as_astropy` instead.

    This object satisfies the `astropy.wcs.wcsapi.BaseHighLevelWCS` and
    `astropy.wcs.wcsapi.BaseLowLevelWCS` interfaces while evaluating the
    underlying `Projection` itself.  It is *not* an `astropy.wcs.WCS`
    subclass, which is a type that also satisfies those interfaces but
    only supports FITS WCS representations (see `Projection.as_fits_wcs`).
    """

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


class ProjectionSerializationModel[P: pydantic.BaseModel](ArchiveTree):
    """Serialization model for projetions."""

    pixel_to_sky: TransformSerializationModel[P] = pydantic.Field(
        description="The transform that maps pixel coordinates to the sky."
    )
    fits_approximation: TransformSerializationModel[P] | None = pydantic.Field(
        description=(
            "An approximation of the pixel-to-sky transform that is exactly representable as a FITS WCS."
        ),
        exclude_if=is_none,
    )

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

__all__ = ("SkyProjection", "SkyProjectionAstropyView", "SkyProjectionSerializationModel")

import functools
from typing import TYPE_CHECKING, Any, ClassVar, Self, TypeVar, assert_type, cast, final, overload

import astropy.units as u
import astropy.wcs
import numpy as np
import pydantic
from astropy.coordinates import ICRS, Latitude, Longitude, SkyCoord
from astropy.wcs.wcsapi import BaseLowLevelWCS, HighLevelWCSMixin

from .._geom import XY, YX, Bounds, Box
from ..serialization import ArchiveTree, InputArchive, InvalidParameterError, OutputArchive
from ..utils import is_none
from . import _ast as astshim
from ._frames import Frame, SkyFrame
from ._transform import Transform, TransformSerializationModel, _ast_apply

if TYPE_CHECKING:
    import numpy.typing as npt

    try:
        from lsst.afw.geom import SkyWcs as LegacySkyWcs
    except ImportError:
        type LegacySkyWcs = Any  # type: ignore[no-redef]


# This pre-python-3.12 declaration is needed by Sphinx (probably the
# autodoc-typehints plugin.
F = TypeVar("F", bound=Frame)
P = TypeVar("P", bound=pydantic.BaseModel)


def _set_ast_skyframe_system(frame: astshim.SkyFrame, system: str) -> None:
    """Set an AST SkyFrame coordinate system across supported wrappers."""
    if hasattr(frame, "_impl"):
        frame._impl.System = system
    else:
        setattr(frame, "system", system)


@final
class SkyProjection[F: Frame]:
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

    Notes
    -----
    `Transform` is conceptually immutable (the internal AST Mapping should
    never be modified in-place after construction), and hence does not need to
    be copied when any object that holds it is copied.
    """

    def __init__(
        self, pixel_to_sky: Transform[F, SkyFrame], fits_approximation: Transform[F, SkyFrame] | None = None
    ) -> None:
        self._pixel_to_sky = pixel_to_sky
        if pixel_to_sky.in_frame.unit != u.pix:
            raise ValueError("Transform is not a mapping from pixel coordinates.")
        if pixel_to_sky.out_frame != SkyFrame.ICRS:
            raise ValueError("Transform is not a mapping to ICRS.")
        self._fits_approximation = fits_approximation

    def __eq__(self, other: Any) -> bool:
        if self is other:
            return True
        if not isinstance(other, SkyProjection):
            return NotImplemented
        # Even though two approximations could be different and yet consistent
        # with the primary mapping (for example using different tolerances
        # on construction) we require them to be equal to declare that the
        # two objects are equal.
        if self._fits_approximation != other._fits_approximation:
            return False
        return self._pixel_to_sky == other._pixel_to_sky

    @staticmethod
    def from_fits_wcs(
        fits_wcs: astropy.wcs.WCS,
        pixel_frame: F,
        pixel_bounds: Bounds | None = None,
        x0: int = 0,
        y0: int = 0,
    ) -> SkyProjection[F]:
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
        return SkyProjection(
            Transform.from_fits_wcs(
                fits_wcs, pixel_frame, SkyFrame.ICRS, in_bounds=pixel_bounds, x0=x0, y0=y0
            )
        )

    @staticmethod
    def from_ast_frame_set(
        ast_frame_set: astshim.FrameSet,
        pixel_frame: F,
        pixel_bounds: Bounds | None = None,
    ) -> SkyProjection[F]:
        """Construct a sky projection from an AST FrameSet.

        The current frame of the FrameSet must be an AST SkyFrame.  Its
        coordinate system is forced to ICRS (AST adjusts the mapping
        automatically) so the resulting Projection is always in ICRS
        regardless of the original sky system.

        Parameters
        ----------
        ast_frame_set
            An AST FrameSet whose base frame is pixel coordinates and
            whose current frame is a SkyFrame (in any supported sky
            coordinate system).
        pixel_frame
            Coordinate frame for the pixel grid.
        pixel_bounds
            The region that bounds valid pixels for this transform.

        Raises
        ------
        ValueError
            If the current frame of the FrameSet is not a SkyFrame.
        """
        current_frame = ast_frame_set.getFrame(ast_frame_set.current, copy=False)
        if not isinstance(current_frame, astshim.SkyFrame):
            raise ValueError(
                "The current frame of the AST FrameSet is not a SkyFrame "
                f"(got {type(current_frame).__name__})."
            )
        _set_ast_skyframe_system(current_frame, "ICRS")
        return SkyProjection(Transform(pixel_frame, SkyFrame.ICRS, ast_frame_set, in_bounds=pixel_bounds))

    @property
    def pixel_frame(self) -> F:
        """Coordinate frame for the pixel grid."""
        return self._pixel_to_sky.in_frame

    @property
    def sky_frame(self) -> SkyFrame:
        """Coordinate frame for the sky."""
        return self._pixel_to_sky.out_frame

    @property
    def pixel_bounds(self) -> Bounds | None:
        """The region that bounds valid pixel points (`Bounds` | `None`)."""
        return self._pixel_to_sky.in_bounds

    @property
    def pixel_to_sky_transform(self) -> Transform[F, SkyFrame]:
        """Low-level transform from pixel to sky coordinates (`Transform`)."""
        return self._pixel_to_sky

    @property
    def sky_to_pixel_transform(self) -> Transform[SkyFrame, F]:
        """Low-level transform from sky to pixel coordinates (`Transform`)."""
        return self._pixel_to_sky.inverted()

    @property
    def fits_approximation(self) -> SkyProjection[F] | None:
        """An approximation to this projection that is guaranteed to have an
        `as_fits_wcs` method that does not return `None`.
        """
        return SkyProjection(self._fits_approximation) if self._fits_approximation is not None else None

    def show(self, simplified: bool = False, comments: bool = False) -> str:
        """Return the AST native representation of the transform.

        Parameters
        ----------
        simplified
            Whether to ask AST to simplify the mapping before showing it.
            This will make it much more likely that two equivalent transforms
            have the same `show` result.
        comments
            Whether to include descriptive comments.
        """
        return self._pixel_to_sky.show(simplified=simplified, comments=comments)

    @overload
    def pixel_to_sky(self, point: XY[int | float] | YX[int | float], /) -> SkyCoord: ...

    @overload
    def pixel_to_sky(self, point: XY[npt.ArrayLike] | YX[npt.ArrayLike], /) -> SkyCoord: ...

    @overload
    def pixel_to_sky(
        self, /, *, x: int | float | npt.ArrayLike, y: int | float | npt.ArrayLike
    ) -> SkyCoord: ...

    def pixel_to_sky(
        self,
        point: XY[Any] | YX[Any] | None = None,
        /,
        *,
        x: Any = None,
        y: Any = None,
    ) -> SkyCoord:
        """Transform one or more pixel points to sky coordinates.

        Parameters
        ----------
        point
            An `XY` or `YX` coordinate pair of pixel positions to transform.
            Mutually exclusive with ``x`` and ``y``.
        x : `float` | array-like
            ``x`` values of the pixel points to transform, as a scalar or
            any array-like.  Results are broadcast against ``y``.
            Mutually exclusive with ``point``.
        y : `float` | array-like
            ``y`` values of the pixel points to transform, as a scalar or
            any array-like.  Results are broadcast against ``x``.
            Mutually exclusive with ``point``.

        Returns
        -------
        astropy.coordinates.SkyCoord
            Transformed sky coordinates, with the broadcast shape of ``x``
            and ``y``.
        """
        match point:
            case None:
                if x is None or y is None:
                    raise TypeError("Pass either a point or both x= and y= to 'pixel_to_sky'.")
            case XY() | YX():
                if x is not None or y is not None:
                    raise TypeError("'pixel_to_sky' point argument is mutually exclusive with x= and y=.")
                x, y = point.x, point.y
            case _:
                raise TypeError(f"Unexpected positional argument type: {type(point)!r}.")
        sky_rad = self._pixel_to_sky.apply_forward(x=x, y=y)
        return SkyCoord(ra=sky_rad.x, dec=sky_rad.y, unit=u.rad)

    def sky_to_pixel(self, sky: SkyCoord) -> XY[Any]:
        """Transform one or more sky coordinates to pixels.

        Parameters
        ----------
        sky
            Sky coordinates to transform.  Any shape is supported; the
            result has the same shape as ``sky``.

        Returns
        -------
        `XY` [`numpy.ndarray` | `float`]
            Transformed pixel coordinates with the same shape as ``sky``.
        """
        if sky.frame.name != "icrs":
            sky = sky.transform_to("icrs")
        ra: Longitude = sky.ra
        dec: Latitude = sky.dec
        # cast works around a mypy false positive specific to generic
        # NamedTuple classes: returning XY[Any] from an overloaded function
        # when called with Any-typed arguments produces "Incompatible return
        # value type (got XY[Any], expected XY[Any])".
        return cast(
            XY[Any],
            self._pixel_to_sky.apply_inverse(
                x=ra.to_value(u.rad),
                y=dec.to_value(u.rad),
            ),
        )

    def as_astropy(self, bbox: Box | None = None) -> SkyProjectionAstropyView:
        """Return an `astropy.wcs` view of this `SkyProjection`.

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
        underlying `SkyProjection` itself.  It is *not* an `astropy.wcs.WCS`
        instance, which is a type that also satisfies those interfaces but
        only supports FITS WCS representations (see `as_fits_wcs`).
        """
        return SkyProjectionAstropyView(self._pixel_to_sky._ast_mapping, bbox)

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
            If `True` and this `SkyProjection` holds a FITS approximation to
            itself, return that approximation.
        """
        if allow_approximation and self._fits_approximation:
            return self._fits_approximation.as_fits_wcs(bbox)
        return self._pixel_to_sky.as_fits_wcs(bbox)

    def serialize[P: pydantic.BaseModel](
        self, archive: OutputArchive[P], *, use_frame_sets: bool = False
    ) -> SkyProjectionSerializationModel[P]:
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
        `SkyProjectionSerializationModel`
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
        return SkyProjectionSerializationModel(
            pixel_to_sky=pixel_to_sky, fits_approximation=fits_approximation
        )

    @staticmethod
    def _get_archive_tree_type[P: pydantic.BaseModel](
        pointer_type: type[P],
    ) -> type[SkyProjectionSerializationModel[P]]:
        """Return the serialization model type for this object for an archive
        type that uses the given pointer type.
        """
        return SkyProjectionSerializationModel[pointer_type]  # type: ignore

    @staticmethod
    def from_legacy(
        sky_wcs: LegacySkyWcs, pixel_frame: F, pixel_bounds: Bounds | None = None
    ) -> SkyProjection[F]:
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
            fits_approximation = Transform(
                pixel_frame,
                SkyFrame.ICRS,
                legacy_fits_approximation.getFrameDict(),
                pixel_bounds,
            )
        return SkyProjection(
            Transform(pixel_frame, SkyFrame.ICRS, sky_wcs.getFrameDict(), pixel_bounds),
            fits_approximation=fits_approximation,
        )

    def to_legacy(self) -> LegacySkyWcs:
        """Convert to a legacy `lsst.afw.geom.SkyWcs` instance."""
        from lsst.afw.geom import SkyWcs as LegacySkyWcs

        try:
            ast_mapping = astshim.FrameDict(self._pixel_to_sky._ast_mapping)
        except TypeError as err:
            err.add_note(
                "Only Projections created by from_legacy and from_fits_wcs "
                "are guaranteed to be convertible to SkyWcs."
            )
            raise
        legacy_wcs = LegacySkyWcs(ast_mapping)
        if self.fits_approximation is not None:
            legacy_wcs = legacy_wcs.copyWithFitsApproximation(self.fits_approximation.to_legacy())
        return legacy_wcs


class SkyProjectionAstropyView(BaseLowLevelWCS, HighLevelWCSMixin):
    """An Astropy-interface view of a `SkyProjection`.

    Parameters
    ----------
    ast_pixel_to_sky
        AST mapping from pixel coordinates to sky coordinates.
    bbox
        Bounding box of the projection, or `None` if unbounded.

    Notes
    -----
    The constructor of this classe is considered a private implementation
    detail; use `SkyProjection.as_astropy` instead.

    This object satisfies the `astropy.wcs.wcsapi.BaseHighLevelWCS` and
    `astropy.wcs.wcsapi.BaseLowLevelWCS` interfaces while evaluating the
    underlying `SkyProjection` itself.  It is *not* an `astropy.wcs.WCS`
    subclass, which is a type that also satisfies those interfaces but
    only supports FITS WCS representations (see `SkyProjection.as_fits_wcs`).
    """

    def __init__(self, ast_pixel_to_sky: astshim.Mapping, bbox: Box | None) -> None:
        self._bbox = bbox
        if bbox is not None:
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

    def pixel_to_world_values(self, x: np.ndarray, y: np.ndarray) -> XY[Any]:
        return _ast_apply(self._ast_pixel_to_sky.applyForward, x=x, y=y)

    def world_to_pixel_values(self, ra: np.ndarray, dec: np.ndarray) -> XY[Any]:
        return _ast_apply(self._ast_pixel_to_sky.applyInverse, x=ra, y=dec)


class SkyProjectionSerializationModel[P: pydantic.BaseModel](ArchiveTree):
    """Serialization model for projetions."""

    SCHEMA_NAME: ClassVar[str] = "sky_projection"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"
    MIN_READ_VERSION: ClassVar[int] = 1
    PUBLIC_TYPE: ClassVar[type] = SkyProjection

    pixel_to_sky: TransformSerializationModel[P] = pydantic.Field(
        description="The transform that maps pixel coordinates to the sky."
    )
    fits_approximation: TransformSerializationModel[P] | None = pydantic.Field(
        default=None,
        description=(
            "An approximation of the pixel-to-sky transform that is exactly representable as a FITS WCS."
        ),
        exclude_if=is_none,
    )

    def deserialize(self, archive: InputArchive[P], **kwargs: Any) -> SkyProjection[Any]:
        """Deserialize a projection from an archive.

        Parameters
        ----------
        archive
            Archive to read from.
        **kwargs
            Unsupported keyword arguments are accepted only to provide better
            error messages (raising `serialization.InvalidParameterError`).
        """
        if kwargs:
            raise InvalidParameterError(f"Unrecognized parameters for Projection: {set(kwargs.keys())}.")
        pixel_to_sky = self.pixel_to_sky.deserialize(archive)
        fits_approximation = (
            self.fits_approximation.deserialize(archive) if self.fits_approximation is not None else None
        )
        return SkyProjection(pixel_to_sky, fits_approximation=fits_approximation)


if TYPE_CHECKING:

    def _test_types() -> None:
        sp = cast(SkyProjection, None)
        sky = cast(SkyCoord, None)

        # sky_to_pixel returns XY[Any] so that callers need no cast regardless
        # of whether sky is scalar or array-shaped.
        assert_type(sp.sky_to_pixel(sky), XY[Any])

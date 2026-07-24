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

__all__ = (
    "Transform",
    "TransformCompositionError",
    "TransformSerializationModel",
)

import textwrap
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar, assert_type, cast, final, overload

import astropy.io.fits.header
import astropy.units as u
import numpy as np
import numpy.typing as npt
import pydantic

from .._concrete_bounds import BoundsSerializationModel
from .._geom import XY, YX, Bounds, Box
from ..describe import DescribableMixin, FieldRole, Report, ReportField
from ..serialization import ArchiveReadError, ArchiveTree, InputArchive, InvalidParameterError, OutputArchive
from . import _ast as astshim
from ._frames import Frame, SerializableFrame, SkyFrame

if TYPE_CHECKING:
    try:
        from lsst.afw.geom import TransformPoint2ToPoint2 as LegacyTransform
    except ImportError:
        type LegacyTransform = Any  # type: ignore[no-redef]

# These pre-python-3.12 declaration are needed by Sphinx (probably the
# autodoc-typehints plugin.
I = TypeVar("I", bound=Frame)  # noqa: E741
O = TypeVar("O", bound=Frame)  # noqa: E741
P = TypeVar("P", bound=pydantic.BaseModel)


class TransformCompositionError(RuntimeError):
    """Exception raised when two transforms cannot be composed."""


@final
class Transform[I: Frame, O: Frame](DescribableMixin):
    """A transform that maps two coordinate frames.

    Parameters
    ----------
    in_frame
        Input coordinate frame.
    out_frame
        Output coordinate frame.
    ast_mapping
        AST mapping that implements the transform.
    in_bounds
        Bounds of the input frame, defaulting to the input frame's
        bounding box.
    out_bounds
        Bounds of the output frame, defaulting to the output frame's
        bounding box.
    components
        Component transforms that this transform was composed from.

    Notes
    -----
    The `Transform` class constructor is considered a private implementation
    detail.  Instead of using this, various factory methods are available:

    - `from_fits_wcs` constructs a transform from a FITS WCS, as represented
      `astropy.wcs.WCS`;
    - `then` composes two transforms;
    - `identity` constructs a trivial transform that does nothing;
    - `affine` contructs an affine transform from a 2x2 or 3x3 matrix;
    - `inverted` returns the inverse of a transform;
    - `from_legacy` converts an `lsst.afw.geom.Transform` instance.

    When applied to celestial coordinate systems, ``x=ra`` and ``y=dec``.
    `SkyProjection` provides a more natural interface for pixel-to-sky
    transforms.

    `Transform` is conceptually immutable (the internal AST Mapping should
    never be modified in-place after construction), and hence does not need to
    be copied when any object that holds it is copied.
    """

    def __init__(
        self,
        in_frame: I,
        out_frame: O,
        ast_mapping: astshim.Mapping,
        in_bounds: Bounds | None = None,
        out_bounds: Bounds | None = None,
        components: Iterable[Transform[Any, Any]] = (),
    ) -> None:
        self._in_frame = in_frame
        self._out_frame = out_frame
        self._ast_mapping = ast_mapping
        self._in_bounds = in_bounds or getattr(in_frame, "bbox", None)
        self._out_bounds = out_bounds or getattr(out_frame, "bbox", None)
        self._components = list(components)

    def __eq__(self, other: Any) -> bool:
        if self is other:
            # Short circuit for case where you are quickly checking
            # that the image WCS and variance WCS are the same object.
            return True
        if not isinstance(other, Transform):
            return NotImplemented
        if self._ast_mapping != other._ast_mapping:
            return False
        if self._in_bounds != other._in_bounds:
            return False
        if self._out_bounds != other._out_bounds:
            return False
        if self._in_frame != other._in_frame:
            return False
        if self._out_frame != other._out_frame:
            return False
        if self._components != other._components:
            return False
        return True

    @staticmethod
    def from_fits_wcs(
        fits_wcs: astropy.wcs.WCS,
        in_frame: I,
        out_frame: O,
        in_bounds: Bounds | None = None,
        out_bounds: Bounds | None = None,
        x0: int = 0,
        y0: int = 0,
    ) -> Transform[I, O]:
        """Construct a transform from a FITS WCS.

        Parameters
        ----------
        fits_wcs
            FITS WCS to convert.
        in_frame
            Coordinate frame for input points to the forward transform.
        out_frame
            Coordinate frame for output points from the forward transform.
        in_bounds
            The region that bounds valid input points.
        out_bounds
            The region that bounds valid output points.
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
        SkyProjection.from_fits_wcs
        """
        ast_stream = astshim.StringStream(fits_wcs.to_header_string(relax=True))
        ast_fits_chan = astshim.FitsChan(ast_stream, "Encoding=FITS-WCS, SipReplace=0, IWC=1")
        ast_frame_set = ast_fits_chan.read()
        _prepend_ast_shift(ast_frame_set, x=x0 - 1.0, y=y0 - 1.0, ast_domain="PIXEL")
        return Transform(
            in_frame,
            out_frame,
            ast_frame_set,
            in_bounds=in_bounds,
            out_bounds=out_bounds,
        )

    @staticmethod
    def identity(frame: I) -> Transform[I, I]:
        """Construct a trivial transform that maps a frame to itelf.

        Parameters
        ----------
        frame
            Frame used for both input and output points.
        """
        return Transform(frame, frame, astshim.UnitMap(2))

    @staticmethod
    def affine(in_frame: I, out_frame: O, matrix: np.ndarray) -> Transform[I, O]:
        """Construct an affine transform from a matrix.

        Parameters
        ----------
        in_frame
            Coordinate frame for input points to the forward transform.
        out_frame
            Coordinate frame for output points from the forward transform.
        matrix
            Matrix of coefficients, either a 2x2 linear transform or a 3x3
            augmented affine transform, with a shift embedded in the third
            column and ``[0, 0, 1]`` the third row.
        """
        if matrix.shape == (2, 2):
            return Transform(in_frame, out_frame, astshim.MatrixMap(matrix.copy()))
        elif matrix.shape == (3, 3):
            linear = astshim.MatrixMap(matrix[:2, :2].copy())
            shift = astshim.ShiftMap(matrix[:2, 2])
            if not np.array_equal(matrix[2, :], np.array([0.0, 0.0, 1.0])):
                raise ValueError("3x3 affine transform array must have [0, 0, 1] in its last row.")
            return Transform(in_frame, out_frame, linear.then(shift))
        else:
            raise ValueError("Affine transform array must be 2x2 or 3x3.")

    @property
    def in_frame(self) -> I:
        """Coordinate frame for input points."""
        return self._in_frame

    @property
    def out_frame(self) -> O:
        """Coordinate frame for output points."""
        return self._out_frame

    @property
    def in_bounds(self) -> Bounds | None:
        """The region that bounds valid input points (`Bounds` | `None`)."""
        return self._in_bounds

    @property
    def out_bounds(self) -> Bounds | None:
        """The region that bounds valid output points (`Bounds` | `None`)."""
        return self._out_bounds

    def show(self, simplified: bool = False, comments: bool = False) -> str:
        """Return the AST native representation of the transform.

        Parameters
        ----------
        simplified
            Whether to ask AST to simplify the mapping before showing it.
            This will make it much more likely that two equivalent transforms
            have the same `show` result.  If the internal mapping is actually
            a frame set (as needed to round-trip legacy
            `lsst.afw.geom.SkyWcs` objects), this will also just show the
            mapping with no frame set information.
        comments
            Whether to include descriptive comments.
        """
        ast_mapping = self._ast_mapping
        if simplified:
            if isinstance(ast_mapping, astshim.FrameSet):
                ast_mapping = ast_mapping.getMapping()
            ast_mapping = ast_mapping.simplified()
        return ast_mapping.show(comments)

    def _describe(self, **kwargs: Any) -> Report:
        """Return a `Report` describing this transform.

        Parameters
        ----------
        **kwargs
            Unused; accepted for interface compatibility.
        """
        return Report(
            type_name="Transform",
            summary=f"{self.in_frame!s} → {self.out_frame!s}",
            fields=[
                ReportField(label="in_frame", value=self.in_frame, role=FieldRole.DERIVED),
                ReportField(label="out_frame", value=self.out_frame, role=FieldRole.DERIVED),
                ReportField(label="in_bounds", value=self.in_bounds, role=FieldRole.DERIVED),
                ReportField(label="out_bounds", value=self.out_bounds, role=FieldRole.DERIVED),
                ReportField(
                    label="mapping",
                    value=self.show(simplified=True),
                    role=FieldRole.DERIVED,
                ),
            ],
        )

    @overload
    def apply_forward(self, point: XY[int | float] | YX[int | float], /) -> XY[float]: ...

    @overload
    def apply_forward(self, point: XY[npt.ArrayLike] | YX[npt.ArrayLike], /) -> XY[np.ndarray]: ...

    @overload
    def apply_forward(self, /, *, x: int | float, y: int | float) -> XY[float]: ...

    @overload
    def apply_forward(self, /, *, x: npt.ArrayLike, y: npt.ArrayLike) -> XY[np.ndarray]: ...

    def apply_forward(
        self, point: XY[Any] | YX[Any] | None = None, /, *, x: Any = None, y: Any = None
    ) -> XY[float] | XY[np.ndarray]:
        """Apply the forward transform to one or more points.

        Parameters
        ----------
        point
            An `XY` or `YX` coordinate pair to transform.  Mutually exclusive
            with ``x`` and ``y``.
        x : `float` | array-like
            ``x`` values of the points to transform, as a scalar or any
            array-like.  Results are broadcast against ``y``.
            Mutually exclusive with ``point``.
        y : `float` | array-like
            ``y`` values of the points to transform, as a scalar or any
            array-like.  Results are broadcast against ``x``.
            Mutually exclusive with ``point``.

        Returns
        -------
        `XY` [`float` | `numpy.ndarray`]
            The transformed point or points.  A scalar input pair returns
            `XY` of `float`; array-like inputs return `XY` of
            `numpy.ndarray` with the broadcast shape of ``x`` and ``y``.
        """
        match point:
            case None:
                if x is None or y is None:
                    raise TypeError("Pass either a point or both x= and y= to 'apply_forward'.")
            case XY() | YX():
                if x is not None or y is not None:
                    raise TypeError("'apply_forward' point argument is mutually exclusive with x= and y=.")
                x, y = point.x, point.y
            case _:
                raise TypeError(f"Unexpected positional argument type: {type(point)!r}.")
        return _standardize_xy(
            _ast_apply(
                self._ast_mapping.applyForward,
                x=self._in_frame.standardize_x(x),
                y=self._in_frame.standardize_y(y),
            ),
            self._out_frame,
        )

    @overload
    def apply_inverse(self, point: XY[int | float] | YX[int | float], /) -> XY[float]: ...

    @overload
    def apply_inverse(self, point: XY[npt.ArrayLike] | YX[npt.ArrayLike], /) -> XY[np.ndarray]: ...

    @overload
    def apply_inverse(self, /, *, x: int | float, y: int | float) -> XY[float]: ...

    @overload
    def apply_inverse(self, /, *, x: npt.ArrayLike, y: npt.ArrayLike) -> XY[np.ndarray]: ...

    def apply_inverse(
        self, point: XY[Any] | YX[Any] | None = None, /, *, x: Any = None, y: Any = None
    ) -> XY[float] | XY[np.ndarray]:
        """Apply the inverse transform to one or more points.

        Parameters
        ----------
        point
            An `XY` or `YX` coordinate pair to transform.  Mutually exclusive
            with ``x`` and ``y``.
        x : `float` | array-like
            ``x`` values of the points to transform, as a scalar or any
            array-like.  Results are broadcast against ``y``.
            Mutually exclusive with ``point``.
        y : `float` | array-like
            ``y`` values of the points to transform, as a scalar or any
            array-like.  Results are broadcast against ``x``.
            Mutually exclusive with ``point``.

        Returns
        -------
        `XY` [`float` | `numpy.ndarray`]
            The transformed point or points.  A scalar input pair returns
            `XY` of `float`; array-like inputs return `XY` of
            `numpy.ndarray` with the broadcast shape of ``x`` and ``y``.
        """
        match point:
            case None:
                if x is None or y is None:
                    raise TypeError("Pass either a point or both x= and y= to 'apply_inverse'.")
            case XY() | YX():
                if x is not None or y is not None:
                    raise TypeError("'apply_inverse' point argument is mutually exclusive with x= and y=.")
                x, y = point.x, point.y
            case _:
                raise TypeError(f"Unexpected positional argument type: {type(point)!r}.")
        return _standardize_xy(
            _ast_apply(
                self._ast_mapping.applyInverse,
                x=self._out_frame.standardize_x(x),
                y=self._out_frame.standardize_y(y),
            ),
            self._in_frame,
        )

    @overload
    def apply_forward_q(self, point: XY[u.Quantity] | YX[u.Quantity], /) -> XY[u.Quantity]: ...

    @overload
    def apply_forward_q(self, /, *, x: u.Quantity, y: u.Quantity) -> XY[u.Quantity]: ...

    def apply_forward_q(
        self, point: XY[u.Quantity] | YX[u.Quantity] | None = None, /, *, x: Any = None, y: Any = None
    ) -> XY[u.Quantity]:
        """Apply the forward transform to one or more unit-aware points.

        Parameters
        ----------
        point
            An `XY` or `YX` coordinate pair of `~astropy.units.Quantity` to
            transform.  Mutually exclusive with ``x`` and ``y``.
        x
            ``x`` values of the points to transform.
            Mutually exclusive with ``point``.
        y
            ``y`` values of the points to transform.
            Mutually exclusive with ``point``.

        Returns
        -------
        `XY` [`astropy.units.Quantity`]
            The transformed point or points.
        """
        match point:
            case None:
                if x is None or y is None:
                    raise TypeError("Pass either a point or both x= and y= to 'apply_forward_q'.")
            case XY() | YX():
                if x is not None or y is not None:
                    raise TypeError("'apply_forward_q' point argument is mutually exclusive with x= and y=.")
                x, y = point.x, point.y
            case _:
                raise TypeError(f"Unexpected positional argument type: {type(point)!r}.")
        xy = self.apply_forward(x=x.to_value(self._in_frame.unit), y=y.to_value(self._in_frame.unit))
        return XY(xy.x * self._out_frame.unit, xy.y * self._out_frame.unit)

    @overload
    def apply_inverse_q(self, point: XY[u.Quantity] | YX[u.Quantity], /) -> XY[u.Quantity]: ...

    @overload
    def apply_inverse_q(self, /, *, x: u.Quantity, y: u.Quantity) -> XY[u.Quantity]: ...

    def apply_inverse_q(
        self, point: XY[u.Quantity] | YX[u.Quantity] | None = None, /, *, x: Any = None, y: Any = None
    ) -> XY[u.Quantity]:
        """Apply the inverse transform to one or more unit-aware points.

        Parameters
        ----------
        point
            An `XY` or `YX` coordinate pair of `~astropy.units.Quantity` to
            transform.  Mutually exclusive with ``x`` and ``y``.
        x
            ``x`` values of the points to transform.
            Mutually exclusive with ``point``.
        y
            ``y`` values of the points to transform.
            Mutually exclusive with ``point``.

        Returns
        -------
        `XY` [`astropy.units.Quantity`]
            The transformed point or points.
        """
        match point:
            case None:
                if x is None or y is None:
                    raise TypeError("Pass either a point or both x= and y= to 'apply_inverse_q'.")
            case XY() | YX():
                if x is not None or y is not None:
                    raise TypeError("'apply_inverse_q' point argument is mutually exclusive with x= and y=.")
                x, y = point.x, point.y
            case _:
                raise TypeError(f"Unexpected positional argument type: {type(point)!r}.")
        xy = self.apply_inverse(x=x.to_value(self._out_frame.unit), y=y.to_value(self._out_frame.unit))
        return XY(xy.x * self._in_frame.unit, xy.y * self._in_frame.unit)

    def decompose(self) -> list[Transform[Any, Any]]:
        """Deconstruct a composed transform into its constituent parts.

        Notes
        -----
        Most transforms will just return a single-element list holding
        ``self``.  Identity transform will return an empty list, and
        transforms composed with `then` will return the original transforms.
        Transforms constructed by `FrameSet` may or may not be decomposable.
        """
        if not self._components:
            if self.in_frame == self._out_frame:
                return []
            else:
                return [self]
        else:
            return list(self._components)

    def inverted(self) -> Transform[O, I]:
        """Return the inverse of this transform."""
        return Transform[O, I](
            self._out_frame,
            self._in_frame,
            self._ast_mapping.inverted(),
            in_bounds=self.out_bounds,
            out_bounds=self.in_bounds,
            components=[t.inverted() for t in reversed(self._components)],
        )

    def then[F: Frame](self, next: Transform[O, F], remember_components: bool = True) -> Transform[I, F]:
        """Compose two transforms into another.

        Parameters
        ----------
        next
            Another transform to apply after ``self``.
        remember_components
            If `True`, the returned composed transform will remember ``self``
            and ``other`` so they can be returned by `decompose`.
        """
        if self._out_frame != next._in_frame:
            raise TransformCompositionError(
                "Cannot compose transforms that do not share a common intermediate frame: "
                f"{self._out_frame} != {next._in_frame}."
            )
        components = self.decompose() + next.decompose() if remember_components else ()
        return Transform(
            self._in_frame,
            next._out_frame,
            self._ast_mapping.then(next._ast_mapping),
            in_bounds=self.in_bounds,
            out_bounds=next.out_bounds,
            components=components,
        )

    def as_fits_wcs(self, bbox: Box) -> astropy.wcs.WCS | None:
        """Return a FITS WCS representation of this transform, if possible.

        Parameters
        ----------
        bbox
            Bounding box of the array the FITS WCS will describe.  This
            transform object is assumed to work on the same coordinate system
            in which ``bbox`` is defined, while the FITS WCS will consider the
            first row and column in that box to be ``(0, 0)`` (in Astropy
            interfaces) or ``(1, 1)`` (in the FITS representation itself).

        Notes
        -----
        This method assumes the transform maps pixel coordinates to world
        coordinates.

        Not all transforms can be represented exactly; when a FITS
        represention is not possible, `None` is returned.  When the returned
        WCS is not `None`, it will have the same functional form, but it may
        not evaluate identically due to small implementation differences in
        the order of floating-point operations.
        """
        ast_frame_set = self._get_ast_frame_set()
        _prepend_ast_shift(ast_frame_set, x=1.0 - bbox.x.start, y=1.0 - bbox.y.start, ast_domain="GRID")
        ast_stream = astshim.StringStream()
        ast_fits_chan = astshim.FitsChan(
            ast_stream, "Encoding=FITS-WCS, CDMatrix=1, FitsAxisOrder=<copy>, FitsTol=0.0001"
        )
        ast_fits_chan.setFitsI("NAXIS1", bbox.x.size)
        ast_fits_chan.setFitsI("NAXIS2", bbox.y.size)
        n_writes = ast_fits_chan.write(ast_frame_set)
        if not n_writes:
            return None
        header = astropy.io.fits.Header(astropy.io.fits.Card.fromstring(c) for c in ast_fits_chan)
        return astropy.wcs.WCS(header)

    def serialize[P: pydantic.BaseModel](
        self, archive: OutputArchive[P], *, use_frame_sets: bool = False
    ) -> TransformSerializationModel[P]:
        """Serialize a transform to an archive.

        Parameters
        ----------
        archive
            Archive to serialize to.
        use_frame_sets
            If `True`, decompose the transform and try to reference component
            mappings that were already serialized into a `FrameSet` in the
            archive.  Note that if multiple transforms exist between a pair of
            frames (e.g. a `SkyProjection` and its FITS approximation), this
            may cause the wrong one to be saved.  When this option is used, the
            frame set must be saved before the transform, and it must be
            deserialized before the transform as well.

        Returns
        -------
        `TransformSerializationModel`
            Serialized form of the transform.
        """
        model = TransformSerializationModel[P]()
        if use_frame_sets:
            for link in self.decompose():
                model.frames.append(link.in_frame.serialize())
                model.bounds.append(link.in_bounds.serialize() if link.in_bounds is not None else None)
                for frame_set, pointer in archive.iter_frame_sets():
                    if link.in_frame in frame_set and link.out_frame in frame_set:
                        model.mappings.append(pointer)
                        break
                else:
                    model.mappings.append(MappingSerializationModel(ast=link._ast_mapping.show()))
        else:
            model.frames.append(self.in_frame.serialize())
            model.bounds.append(self.in_bounds.serialize() if self.in_bounds is not None else None)
            model.mappings.append(MappingSerializationModel(ast=self._ast_mapping.show()))
        model.frames.append(self.out_frame.serialize())
        model.bounds.append(self.out_bounds.serialize() if self.out_bounds is not None else None)
        return model

    @staticmethod
    def _get_archive_tree_type[P: pydantic.BaseModel](
        pointer_type: type[P],
    ) -> type[TransformSerializationModel[P]]:
        """Return the serialization model type for this object for an archive
        type that uses the given pointer type.
        """
        return TransformSerializationModel[pointer_type]  # type: ignore

    @staticmethod
    def from_legacy(
        legacy: LegacyTransform,
        in_frame: I,
        out_frame: O,
        in_bounds: Bounds | None = None,
        out_bounds: Bounds | None = None,
    ) -> Transform[I, O]:
        """Construct a transform from a legacy `lsst.afw.geom.Transform`.

        Parameters
        ----------
        legacy : `lsst.afw.geom.Transform`
            Legacy transform object.
        in_frame
            Coordinate frame for input points to the forward transform.
        out_frame
            Coordinate frame for output points from the forward transform.
        in_bounds
            The region that bounds valid input points.
        out_bounds
            The region that bounds valid output points.
        """
        return Transform(
            in_frame,
            out_frame,
            legacy.getMapping(),
            in_bounds=in_bounds,
            out_bounds=out_bounds,
        )

    def to_legacy(self) -> LegacyTransform:
        """Convert to a legacy `lsst.afw.geom.TransformPoint2ToPoint2`
        instance.
        """
        from lsst.afw.geom import TransformPoint2ToPoint2 as LegacyTransform

        return LegacyTransform(self._ast_mapping, False)

    def _get_ast_frame_set(self) -> Any:
        ast_frame_set = astshim.FrameSet(_make_ast_frame(self._in_frame))
        ast_frame_set.addFrame(astshim.FrameSet.BASE, self._ast_mapping, _make_ast_frame(self._out_frame))
        return ast_frame_set


def _ast_apply(method: Any, *, x: Any, y: Any) -> XY[float] | XY[np.ndarray]:
    # TODO: add bounds argument and check inputs
    xa = np.asarray(x)
    ya = np.asarray(y)
    broadcast_shape = np.broadcast(xa, ya).shape
    scalar = not broadcast_shape
    xb, yb = np.broadcast_arrays(xa, ya)
    xy_in = np.vstack([xb.ravel(), yb.ravel()]).astype(np.float64)
    xy_out = method(xy_in)
    if scalar:
        return XY(float(xy_out[0, 0]), float(xy_out[1, 0]))
    return XY(xy_out[0].reshape(broadcast_shape), xy_out[1].reshape(broadcast_shape))


def _prepend_ast_shift(ast_frame_set: Any, x: float, y: float, ast_domain: str) -> None:
    ast_output_frame_id = ast_frame_set.current
    ast_frame_set.addFrame(
        astshim.FrameSet.BASE,
        astshim.ShiftMap([x, y]),
        astshim.Frame(2, f"Domain={ast_domain}"),
    )
    ast_frame_set.base = ast_frame_set.current
    ast_frame_set.current = ast_output_frame_id


def _make_ast_frame(frame: Frame) -> Any:
    if frame is SkyFrame.ICRS:
        return astshim.SkyFrame("")
    ast_frame = astshim.Frame(2, f"Ident={frame._ast_ident}")
    if frame.unit is not None:
        fits_unit = frame.unit.to_string(format="fits")
        ast_frame.setUnit(1, fits_unit)
        ast_frame.setUnit(2, fits_unit)
    ast_frame.setLabel(1, "x")
    ast_frame.setLabel(2, "y")
    return ast_frame


def _standardize_xy(xy: XY[Any], frame: Frame) -> XY[Any]:
    return XY(x=frame.standardize_x(xy.x), y=frame.standardize_y(xy.y))


class MappingSerializationModel(pydantic.BaseModel):
    """Serialization model for an AST Mapping."""

    ast: str = pydantic.Field(description="A serialized Starlink AST Mapping, using the AST native encoding.")


class TransformSerializationModel[P: pydantic.BaseModel](ArchiveTree):
    """Serialization model for coordinate transforms."""

    SCHEMA_NAME: ClassVar[str] = "transform"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"
    MIN_READ_VERSION: ClassVar[int] = 1
    PUBLIC_TYPE: ClassVar[type] = Transform

    frames: list[SerializableFrame] = pydantic.Field(
        default_factory=list,
        description=textwrap.dedent(
            """
            List of frames that this transform passes through.

            All transforms include at least two frames (the endpoints).  Others
            intermediate frames may be included to facilitate data-sharing
            between transforms.
            """
        ),
    )

    bounds: list[BoundsSerializationModel | None] = pydantic.Field(
        default_factory=list,
        description=textwrap.dedent(
            """
            List of the bounds of the ``frames`` for this transform.

            This always has the same number of elements as ``frames``.
            """
        ),
    )

    mappings: list[P | MappingSerializationModel] = pydantic.Field(
        default_factory=list,
        description=textwrap.dedent(
            """
            The actual mappings between frames, or archive pointers to
            serialized FrameSet objects from which they can be obtained.

            This always has one fewer element than ``frames``.
            """
        ),
    )

    def deserialize(self, archive: InputArchive[P], **kwargs: Any) -> Transform[Any, Any]:
        """Deserialize a transform from an archive.

        Parameters
        ----------
        archive
            Archive to read from.
        **kwargs
            Unsupported keyword arguments are accepted only to provide better
            error messages (raising `serialization.InvalidParameterError`).
        """
        if kwargs:
            raise InvalidParameterError(f"Unrecognized parameters for Transform: {set(kwargs.keys())}.")
        if len(self.frames) != len(self.bounds):
            raise ArchiveReadError(
                f"Inconsistent lengths for 'frames' ({len(self.frames)}) and 'bounds' ({len(self.bounds)})."
            )
        if len(self.frames) != len(self.mappings) + 1:
            raise ArchiveReadError(
                f"Inconsistent lengths for 'frames' ({len(self.frames)}) and "
                f"'mappings' ({len(self.mappings)}; should be one less)."
            )
        # We can't just compose onto an identity Transform if we want to
        # preserve the FrameSet-ness of any of these mappings.
        transform: Transform | None = None
        for n, mapping in enumerate(self.mappings):
            match mapping:
                case MappingSerializationModel(ast=serialized_mapping):
                    ast_mapping = astshim.Mapping.fromString(serialized_mapping)
                    in_bounds = self.bounds[n]
                    out_bounds = self.bounds[n + 1]
                    new_transform = Transform(
                        self.frames[n].deserialize(),
                        self.frames[n + 1].deserialize(),
                        ast_mapping,
                        in_bounds.deserialize() if in_bounds is not None else None,
                        out_bounds.deserialize() if out_bounds is not None else None,
                    )
                case reference:
                    frame_set = archive.get_frame_set(reference)
                    new_transform = frame_set[self.frames[n].deserialize(), self.frames[n + 1].deserialize()]
            if transform is None:
                transform = new_transform
            else:
                transform = transform.then(new_transform)
        if transform is None:
            transform = Transform.identity(self.frames[0].deserialize())
        return transform


if TYPE_CHECKING:

    def _test_types() -> None:
        t = cast(Transform, None)
        arr = np.zeros(3)

        # Scalar inputs → XY[float]
        assert_type(t.apply_forward(x=1.0, y=2.0), XY[float])
        assert_type(t.apply_inverse(x=1.0, y=2.0), XY[float])

        # Array inputs → XY[np.ndarray]
        assert_type(t.apply_forward(x=arr, y=arr), XY[np.ndarray])
        assert_type(t.apply_inverse(x=arr, y=arr), XY[np.ndarray])

        # Array-like (list) inputs → XY[np.ndarray]
        assert_type(t.apply_forward(x=[1.0, 2.0], y=[3.0, 4.0]), XY[np.ndarray])
        assert_type(t.apply_inverse(x=[1.0, 2.0], y=[3.0, 4.0]), XY[np.ndarray])

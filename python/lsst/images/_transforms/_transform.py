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
from typing import TYPE_CHECKING, Any, final

import astropy.io.fits.header
import astropy.units as u
import numpy as np
import pydantic

from .._geom import XY, Bounds, Box, SerializableBounds
from ..serialization import ArchiveReadError, InputArchive, OutputArchive
from ._frames import Frame, SkyFrame

if TYPE_CHECKING:
    import astshim
    import lsst.afw.geom

    from ._projection import Projection


class TransformCompositionError(RuntimeError):
    """Exception raised when two transforms cannot be composed."""


@final
class Transform[I: Frame, O: Frame]:
    """A transform that maps two coordinate frames.

    Notes
    -----
    The `Transform` class constructor is considered a private implementation
    detail.  Instead, various factory methods are available:

    - `from_fits_wcs` constructs a transform from a FITS WCS, as represented
      `astropy.wcs.WCS`;
    - `then` composes two transforms;
    - `identity` constructs a trivial transform that does nothing;
    - `inverted` returns the inverse of a transform;
    - `from_legacy` converts an `lsst.afw.geom.Transform` instance.

    When applied to celestial coordinate systems, ``x=ra`` and ``y=dec``.
    `Projection` provides a more natural interface for pixel-to-sky transforms.
    """

    def __init__(
        self,
        in_frame: I,
        out_frame: O,
        ast_mapping: astshim.Mapping,
        in_bounds: Bounds | None = None,
        out_bounds: Bounds | None = None,
        components: Iterable[Transform[Any, Any]] = (),
    ):
        self._in_frame = in_frame
        self._out_frame = out_frame
        self._ast_mapping = ast_mapping
        self._in_bounds = in_bounds or getattr(in_frame, "bbox", None)
        self._out_bounds = out_bounds or getattr(out_frame, "bbox", None)
        self._components = list(components)

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
        The `x0` and `y0` parameters reflect the fact that for FITS, the first
        row and column are always labeled ``(1, 1)``, while in Astropy and
        most other Python libraries, they are ``(0, 0)``.  The `types` in this
        package (e.g. `Image`, `Mask`) allow them to be any pair of integers.

        See Also
        --------
        Projection.from_fits_wcs
        """
        import astshim

        ast_stream = astshim.StringStream(fits_wcs.to_header_string(relax=True))
        ast_fits_chan = astshim.FitsChan(ast_stream, "Encoding=FITS-WCS, SipReplace=0")
        ast_frame_set = ast_fits_chan.read()
        _prepend_ast_shift(ast_frame_set, x=x0 - 1.0, y=y0 - 1.0, ast_bounds="PIXEL")
        return Transform(
            in_frame,
            out_frame,
            ast_frame_set.getMapping(),
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
        import astshim

        return Transform(frame, frame, astshim.UnitMap(2))

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

    def apply_forward[T: np.ndarray | float](self, *, x: T, y: T) -> XY[T]:
        """Apply the forward transform to one or more points.

        Parameters
        ----------
        x : `numpy.ndarray` | `float`
            ``x`` values of the points to transform.
        y : `numpy.ndarray` | `float`
            ``y`` values of the points to transform.

        Returns
        -------
        `XY` [`numpy.ndarray` | `float`]
            The transformed point or points.
        """
        return _normalize_xy(
            _ast_apply(
                self._ast_mapping.applyForward,
                x=self._in_frame.normalize_x(x),
                y=self._in_frame.normalize_y(y),
            ),
            self._out_frame,
        )

    def apply_inverse[T: np.ndarray | float](self, *, x: T, y: T) -> XY[T]:
        """Apply the inverse transform to one or more points.

        Parameters
        ----------
        x : `numpy.ndarray` | `float`
            ``x`` values of the points to transform.
        y : `numpy.ndarray` | `float`
            ``y`` values of the points to transform.

        Returns
        -------
        `XY` [`numpy.ndarray` | `float`]
            The transformed point or points.
        """
        return _normalize_xy(
            _ast_apply(
                self._ast_mapping.applyInverse,
                x=self._out_frame.normalize_x(x),
                y=self._out_frame.normalize_y(y),
            ),
            self._in_frame,
        )

    def apply_forward_q(self, *, x: u.Quantity, y: u.Quantity) -> XY[u.Quantity]:
        """Apply the forward transform to one or more unit-aware points.

        Parameters
        ----------
        x
            ``x`` values of the points to transform.
        y
            ``y`` values of the points to transform.

        Returns
        -------
        `XY` [`astropy.units.Quantity`]
            The transformed point or points.
        """
        xy = self.apply_forward(x=x.to_value(self._in_frame.unit), y=y.to_value(self._in_frame.unit))
        return XY(xy.x * self._out_frame.unit, xy.y * self._out_frame.unit)

    def apply_inverse_q(self, *, x: u.Quantity, y: u.Quantity) -> XY[u.Quantity]:
        """Apply the inverse transform to one or more unit-aware points.

        Parameters
        ----------
        x
            ``x`` values of the points to transform.
        y
            ``y`` values of the points to transform.

        Returns
        -------
        `XY` [`astropy.units.Quantity`]
            The transformed point or points.
        """
        xy = self.apply_inverse(x=x.to_value(self._out_frame.unit), y=y.to_value(self._out_frame.unit))
        return XY(xy.x * self._in_frame.unit, xy.y * self._in_frame.unit)

    def decompose(self) -> list[Transform[Any, Any]]:
        """Deconstruct a composed transform into its constituent parts.

        Notes
        -----
        Most transforms will just return a single-element list holding
        ``self``.  Identity transform willi return an empty list, and
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

    def as_projection(self: Transform[I, SkyFrame]) -> Projection[I]:
        """Return a `Projection` view of this transform.

        This is only valid when `out_frame` is `~SkyFrame.ICRS`.
        """
        from ._projection import Projection

        return Projection(self)

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
        import astshim

        ast_frame_set = self._get_ast_frame_set()
        _prepend_ast_shift(ast_frame_set, x=1.0 - bbox.x.start, y=1.0 - bbox.y.start, ast_bounds="GRID")
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
            frames (e.g. a `Projection` and its FITS approximation), this may
            cause the wrong one to be saved.  When this option is used, the
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
                model.frames.append(link.in_frame)
                model.bounds.append(link.in_bounds.serialize() if link.in_bounds is not None else None)
                for frame_set, pointer in archive.iter_frame_sets():
                    if link.in_frame in frame_set and link.out_frame in frame_set:
                        model.mappings.append(pointer)
                        break
                else:
                    model.mappings.append(MappingSerializationModel(ast=link._ast_mapping.show()))
        else:
            model.frames.append(self.in_frame)
            model.bounds.append(self.in_bounds.serialize() if self.in_bounds is not None else None)
            model.mappings.append(MappingSerializationModel(ast=self._ast_mapping.show()))
        model.frames.append(self.out_frame)
        model.bounds.append(self.out_bounds.serialize() if self.out_bounds is not None else None)
        return model

    @staticmethod
    def deserialize[P: pydantic.BaseModel](
        model: TransformSerializationModel[P], archive: InputArchive[P]
    ) -> Transform[Any, Any]:
        """Deserialize a transform from an archive.

        Parameters
        ----------
        model
            Seralized form of the transform.
        archive
            Archive to read from.
        """
        import astshim

        if len(model.frames) != len(model.bounds):
            raise ArchiveReadError(
                f"Inconsistent lengths for 'frames' ({len(model.frames)}) and "
                f"'boundss' ({len(model.bounds)})."
            )
        if len(model.frames) != len(model.mappings) + 1:
            raise ArchiveReadError(
                f"Inconsistent lengths for 'frames' ({len(model.frames)}) and "
                f"'mappings' ({len(model.mappings)}; should be one less)."
            )
        transform = Transform.identity(model.frames[0])
        for n, mapping in enumerate(model.mappings):
            match mapping:
                case MappingSerializationModel(ast=serialized_mapping):
                    ast_mapping = astshim.Mapping.fromString(serialized_mapping)
                    in_bounds = model.bounds[n]
                    out_bounds = model.bounds[n + 1]
                    transform = transform.then(
                        Transform(
                            model.frames[n],
                            model.frames[n + 1],
                            ast_mapping,
                            Bounds.deserialize(in_bounds) if in_bounds is not None else None,
                            Bounds.deserialize(out_bounds) if out_bounds is not None else None,
                        )
                    )
                case reference:
                    frame_set = archive.get_frame_set(reference)
                    transform = transform.then(frame_set[model.frames[n], model.frames[n + 1]])
        return transform

    @staticmethod
    def from_legacy(
        legacy: lsst.afw.geom.Transform,
        in_frame: I,
        out_frame: O,
        in_bounds: Bounds | None = None,
        out_bounds: Bounds | None = None,
    ) -> Transform[I, O]:
        """Construct a transform from a legacy `lsst.afw.geom.Transform`.

        Parameters
        ----------
        legacy
            Legacy transform object.
        in_frame
            Coordinate frame for input points to the forward transform.
        out_frame
            Coordinate frame for output points from the forward transform.
        in_domain
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

    def _get_ast_frame_set(self) -> Any:
        import astshim

        ast_frame_set = astshim.FrameSet(_make_ast_frame(self._in_frame))
        ast_frame_set.addFrame(astshim.FrameSet.BASE, self._ast_mapping, _make_ast_frame(self._out_frame))
        return ast_frame_set


def _ast_apply[T: np.ndarray | float](method: Any, *, x: T, y: T) -> XY[T]:
    # TODO: add bounds argument and check inputs
    # TODO: broadcast arrays with different shapes.
    xy_in = np.vstack([x, y]).astype(np.float64)
    xy_out = method(xy_in)
    return XY(xy_out[0, :], xy_out[1, :])


def _prepend_ast_shift(ast_frame_set: Any, x: float, y: float, ast_bounds: str) -> None:
    import astshim

    ast_output_frame_id = ast_frame_set.current
    ast_frame_set.addFrame(
        astshim.FrameSet.BASE,
        astshim.ShiftMap([x, y]),
        astshim.Frame(2, f"Bounds={ast_bounds}"),
    )
    ast_frame_set.base = ast_frame_set.current
    ast_frame_set.current = ast_output_frame_id


def _make_ast_frame(frame: Frame) -> Any:
    import astshim

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


def _normalize_xy[T: np.ndarray | float](xy: XY[T], frame: Frame) -> XY[T]:
    return XY(x=frame.normalize_x(xy.x), y=frame.normalize_y(xy.y))


class MappingSerializationModel(pydantic.BaseModel):
    """Serialization model for an AST Mapping."""

    ast: str = pydantic.Field(description="A serialized Starlink AST Mapping, using the AST native encoding.")


class TransformSerializationModel[P: pydantic.BaseModel](pydantic.BaseModel):
    """Serialization model for coordinate transforms."""

    frames: list[Frame] = pydantic.Field(
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

    bounds: list[SerializableBounds | None] = pydantic.Field(
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

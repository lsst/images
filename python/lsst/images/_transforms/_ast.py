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

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, ClassVar, Self

import numpy as np

__all__ = (
    "USING_STARLINK_PYAST",
    "Channel",
    "CmpFrame",
    "CmpMap",
    "FitsChan",
    "Frame",
    "FrameSet",
    "Mapping",
    "MatrixMap",
    "PolyMap",
    "ShiftMap",
    "SkyFrame",
    "StringStream",
    "UnitMap",
    "ZoomMap",
)

if TYPE_CHECKING:
    import starlink.Ast

    USING_STARLINK_PYAST = True
else:
    try:
        from astshim import (
            Channel,
            CmpFrame,
            CmpMap,
            FitsChan,
            Frame,
            FrameDict,
            FrameSet,
            Mapping,
            MatrixMap,
            Object,
            PolyMap,
            ShiftMap,
            SkyFrame,
            StringStream,
            UnitMap,
            ZoomMap,
        )
    except ImportError:
        import starlink.Ast

        USING_STARLINK_PYAST = True
    else:
        USING_STARLINK_PYAST = False


if USING_STARLINK_PYAST:

    class StringStream:
        """A bridge object that mimics both astshim.StringStream and the
        interface expected by starlink.Ast.Channel.

        Notes
        -----
        This object can be constructed like an `astshim.StringStream`, but its
        `astsink` and `astsource` methods actually correspond to the
        `starlink.Ast` interface, so we can use `starlink.Ast` objects to
        implement the `FitsChan` classes in this module
        """

        def __init__(self, text: str = "") -> None:
            if "\n" in text or "\r" in text:
                self._lines = text.splitlines()
            elif text and len(text) % 80 == 0:
                # Astropy WCS.to_header_string() yields a single concatenated
                # FITS header block; FitsChan expects one card per source line.
                self._lines = [text[n : n + 80] for n in range(0, len(text), 80)]
            else:
                self._lines = text.splitlines()

        def astsource(self) -> str | None:
            if not self._lines:
                return None
            return self._lines.pop(0)

        def astsink(self, line: str) -> None:
            self._lines.append(line)

        def to_string(self) -> str:
            if not self._lines:
                return ""
            return "\n".join(self._lines) + "\n"

        def getSinkData(self) -> str:
            return self.to_string()

    class Object:
        """Bridge class that exposes the `astshim.Object` interface while
        being backed by an `astshim.Ast.Object`.
        """

        def __init__(self, impl: starlink.Ast.Object) -> None:
            if not isinstance(impl, self._IMPL_TYPE):
                raise TypeError(f"{type(self).__name__} cannot wrap {type(impl).__name__}.")
            self._impl = impl

        _IMPL_TYPE: ClassVar[type[starlink.Ast.Object]] = starlink.Ast.Object

        def show(self, showComments: bool = True) -> str:
            sink = StringStream()
            options = "" if showComments else "Comment=0"
            chan = starlink.Ast.Channel(None, sink, options=options)
            chan.write(self._impl)
            return sink.to_string()

        def __eq__(self, other: object) -> bool:
            if not isinstance(other, Object) or type(self) is not type(other):
                return NotImplemented
            if self is other:
                # Bypass stringification if they are the same object.
                return True
            # ``astshim.Object`` ships a structural ``__eq__``; mirror that on
            # the starlink-pyast wrapper by comparing the AST channel
            # serialisation, which is the canonical content-faithful
            # representation for AST objects.  Strip comments so cosmetic
            # changes between equivalent objects do not break equality.
            return self.show(showComments=False) == other.show(showComments=False)

        __hash__ = None  # type: ignore[assignment]

        @classmethod
        def fromString(cls, serialized: str) -> Self:
            source = StringStream(serialized)
            channel = starlink.Ast.Channel(source)
            return cls._wrap(channel.read())

        @classmethod
        def _wrap(cls, impl: starlink.Ast.Object) -> Self:
            subcls = cls._most_derived_type(impl)
            result = object.__new__(subcls)
            Object.__init__(result, impl)
            return result

        @classmethod
        def _most_derived_type(cls, impl: starlink.Ast.Object) -> type[Self]:
            for subcls in cls.__subclasses__():
                if isinstance(impl, subcls._IMPL_TYPE):
                    return subcls._most_derived_type(impl)
            return cls

    class Mapping(Object):
        _IMPL_TYPE: ClassVar[type[starlink.Ast.Mapping]] = starlink.Ast.Mapping

        def simplified(self) -> Mapping:
            return Mapping._wrap(self._impl.simplify())

        def applyForward(self, xy: Any) -> Any:
            return self._impl.tran(xy, True)

        def applyInverse(self, xy: Any) -> Any:
            return self._impl.tran(xy, False)

        def then(self, other: Mapping) -> Mapping:
            return Mapping._wrap(starlink.Ast.CmpMap(self._impl, other._impl, True))

        def inverted(self) -> Mapping:
            copy = self._impl.copy()
            copy.invert()
            return Mapping._wrap(copy)

        def linearApprox(self, lbnd: Any, ubnd: Any, tol: float) -> np.ndarray:
            """Best linear approximation to this mapping over a hyper-box.

            Parameters
            ----------
            lbnd, ubnd
                Per-axis lower / upper input-coordinate bounds of the
                box over which the approximation is required.
            tol
                Maximum permitted deviation from linearity, expressed
                as a positive Cartesian displacement in the output
                coordinate system.

            Returns
            -------
            fit : `numpy.ndarray`
                A ``(1 + Nout, Nin)`` array whose first row holds the
                per-output constant offsets and whose remaining rows hold
                the Jacobian (``J[i][j] = ∂out_i/∂in_j``).

            Raises
            ------
            RuntimeError
                Raised if no linear approximation within ``tol`` exists
                over the requested box.

            Notes
            -----
            This matches ``astshim.Mapping.linearApprox``. starlink-pyast
            instead returns ``(success_flag, flat_coeffs)`` with the
            coefficients in the same row-major-by-output ordering as the
            astshim flat buffer, so reshaping recovers astshim's layout.
            """
            success, coeffs = self._impl.linearapprox(lbnd, ubnd, tol)
            if not success:
                raise RuntimeError("Mapping not sufficiently linear")
            nin = self._impl.Nin
            nout = self._impl.Nout
            return np.asarray(coeffs, dtype=float).reshape(1 + nout, nin)

    class UnitMap(Mapping):
        def __init__(self, n_coord: int) -> None:
            super().__init__(starlink.Ast.UnitMap(n_coord))

        _IMPL_TYPE: ClassVar[type[starlink.Ast.UnitMap]] = starlink.Ast.UnitMap

    class ShiftMap(Mapping):
        def __init__(self, shift: Iterable[float]) -> None:
            super().__init__(starlink.Ast.ShiftMap(list(shift)))

        _IMPL_TYPE: ClassVar[type[starlink.Ast.ShiftMap]] = starlink.Ast.ShiftMap

    class CmpMap(Mapping):
        def __init__(self, map_a: Mapping, map_b: Mapping, series: bool) -> None:
            super().__init__(starlink.Ast.CmpMap(map_a._impl, map_b._impl, series))

        _IMPL_TYPE: ClassVar[type[starlink.Ast.CmpMap]] = starlink.Ast.CmpMap

    class ZoomMap(Mapping):
        def __init__(self, n_coord: int, zoom: float) -> None:
            super().__init__(starlink.Ast.ZoomMap(n_coord, zoom))

        _IMPL_TYPE: ClassVar[type[starlink.Ast.ZoomMap]] = starlink.Ast.ZoomMap

    class PolyMap(Mapping):
        def __init__(self, coeff_f: Any, coeff_i_or_nout: Any, options: str = "") -> None:
            # astshim's PolyMap takes ``nout`` as the second positional;
            # starlink.Ast.PolyMap requires an explicit inverse-coefficient
            # array. Adapt to both by synthesizing an empty inverse when
            # an integer ``nout`` is supplied.
            coeff_f_arr = np.asarray(coeff_f, dtype=float)
            if isinstance(coeff_i_or_nout, int):
                nin = coeff_f_arr.shape[1] - 2
                coeff_i = np.zeros((0, 2 + nin), dtype=float)
            else:
                coeff_i = np.asarray(coeff_i_or_nout, dtype=float)
            super().__init__(starlink.Ast.PolyMap(coeff_f_arr, coeff_i, options))

        _IMPL_TYPE: ClassVar[type[starlink.Ast.PolyMap]] = starlink.Ast.PolyMap

    class MatrixMap(Mapping):
        def __init__(self, matrix: np.ndarray, options: str = ""):
            super().__init__(starlink.Ast.MatrixMap(matrix, options))

        _IMPL_TYPE: ClassVar[type[starlink.Ast.MatrixMap]] = starlink.Ast.MatrixMap

    class Frame(Mapping):
        def __init__(self, n_axes: int, options: str = "") -> None:
            super().__init__(starlink.Ast.Frame(n_axes, options))

        _IMPL_TYPE: ClassVar[type[starlink.Ast.Frame]] = starlink.Ast.Frame

        @property
        def ident(self) -> str:
            return self._impl.Ident

        @property
        def domain(self) -> str:
            return self._impl.Domain

        @domain.setter
        def domain(self, value: str) -> None:
            self._impl.Domain = value

        def setUnit(self, axis: int, unit: str) -> None:
            setattr(self._impl, f"Unit_{axis}", unit)

        def getUnit(self, axis: int) -> str:
            return getattr(self._impl, f"Unit_{axis}")

        def setLabel(self, axis: int, label: str) -> None:
            setattr(self._impl, f"Label_{axis}", label)

        def getBottom(self, axis: int) -> float:
            return getattr(self._impl, f"Bottom_{axis}")

        def getTop(self, axis: int) -> float:
            return getattr(self._impl, f"Top_{axis}")

    class SkyFrame(Frame):
        def __init__(self, options: str = "") -> None:
            Object.__init__(self, starlink.Ast.SkyFrame(options))

        _IMPL_TYPE: ClassVar[type[starlink.Ast.SkyFrame]] = starlink.Ast.SkyFrame

    class CmpFrame(Frame):
        def __init__(self, frame_a: Frame, frame_b: Frame) -> None:
            Object.__init__(self, starlink.Ast.CmpFrame(frame_a._impl, frame_b._impl, ""))

        _IMPL_TYPE: ClassVar[type[starlink.Ast.CmpFrame]] = starlink.Ast.CmpFrame

    class FrameSet(Frame):
        def __init__(self, base_frame: Frame) -> None:
            Object.__init__(self, starlink.Ast.FrameSet(base_frame._impl))

        BASE: ClassVar[int] = 1
        _IMPL_TYPE: ClassVar[type[starlink.Ast.FrameSet]] = starlink.Ast.FrameSet

        @property
        def nFrame(self) -> int:
            return self._impl.Nframe

        @property
        def base(self) -> int:
            return self._impl.Base

        @base.setter
        def base(self, value: int) -> None:
            self._impl.Base = value

        @property
        def current(self) -> int:
            return self._impl.Current

        @current.setter
        def current(self, value: int) -> None:
            self._impl.Current = value

        def addFrame(self, iframe: int, mapping: Mapping, frame: Frame) -> None:
            self._impl.addframe(iframe, mapping._impl, frame._impl)

        def getFrame(self, iframe: int, copy: bool = True) -> Frame:
            result = self._impl.getframe(iframe)
            if copy:
                result = result.copy()
            return Frame._wrap(result)

        def getMapping(self, iframe1: int | None = None, iframe2: int | None = None) -> Mapping:
            if iframe1 is None:
                iframe1 = self.base
            if iframe2 is None:
                iframe2 = self.current
            return Mapping._wrap(self._impl.getmapping(iframe1, iframe2))

    class FrameDict(FrameSet):
        def __init__(self, obj: Object) -> None:
            Object.__init__(self, obj._impl)

    class FitsChan(Object):
        def __init__(self, stream: StringStream | None = None, options: str = "") -> None:
            source = stream if stream is not None else None
            sink = stream if stream is not None else None
            super().__init__(starlink.Ast.FitsChan(source, sink, options))

        _IMPL_TYPE: ClassVar[type[starlink.Ast.FitsChan]] = starlink.Ast.FitsChan

        def read(self) -> Any:
            return Object._wrap(self._impl.read())

        def write(self, obj: Any) -> int:
            return self._impl.write(obj._impl)

        def setFitsI(self, keyword: str, value: int) -> None:
            self._impl.setfitsI(keyword, value, "", 1)

        def __iter__(self) -> Any:
            return iter(self._impl)

    class Channel(Object):
        def __init__(self, stream: StringStream, options: str = "") -> None:
            super().__init__(starlink.Ast.Channel(None, stream, options))

        _IMPL_TYPE: ClassVar[type[starlink.Ast.Channel]] = starlink.Ast.Channel

        def write(self, obj: Object) -> int:
            return self._impl.write(obj._impl)

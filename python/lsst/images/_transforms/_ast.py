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

__all__ = (
    "USING_STARLINK_PYAST",
    "FitsChan",
    "Frame",
    "FrameSet",
    "Mapping",
    "ShiftMap",
    "SkyFrame",
    "StringStream",
    "UnitMap",
)

if TYPE_CHECKING:
    import starlink.Ast

    USING_STARLINK_PYAST = True
else:
    try:
        from astshim import (
            FitsChan,
            Frame,
            FrameDict,
            FrameSet,
            Mapping,
            Object,
            ShiftMap,
            SkyFrame,
            StringStream,
            UnitMap,
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

        def __init__(self, text: str = ""):
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

    class Object:
        """Bridge class that exposes the `astshim.Object` interface while
        being backed by an `astshim.Ast.Object`.
        """

        def __init__(self, impl: starlink.Ast.Object):
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

    class UnitMap(Mapping):
        def __init__(self, n_coord: int):
            super().__init__(starlink.Ast.UnitMap(n_coord))

        _IMPL_TYPE: ClassVar[type[starlink.Ast.UnitMap]] = starlink.Ast.UnitMap

    class ShiftMap(Mapping):
        def __init__(self, shift: Iterable[float]):
            super().__init__(starlink.Ast.ShiftMap(list(shift)))

        _IMPL_TYPE: ClassVar[type[starlink.Ast.ShiftMap]] = starlink.Ast.ShiftMap

    class Frame(Mapping):
        def __init__(self, n_axes: int, options: str = ""):
            super().__init__(starlink.Ast.Frame(n_axes, options))

        _IMPL_TYPE: ClassVar[type[starlink.Ast.Frame]] = starlink.Ast.Frame

        @property
        def ident(self) -> str:
            return self._impl.Ident

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
        def __init__(self, options: str = ""):
            Object.__init__(self, starlink.Ast.SkyFrame(options))

        _IMPL_TYPE: ClassVar[type[starlink.Ast.SkyFrame]] = starlink.Ast.SkyFrame

    class FrameSet(Frame):
        def __init__(self, base_frame: Frame):
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
        def __init__(self, obj: Object):
            Object.__init__(self, obj._impl)

    class FitsChan(Object):
        def __init__(self, stream: StringStream | None = None, options: str = ""):
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

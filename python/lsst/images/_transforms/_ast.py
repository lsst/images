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
from typing import Any

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
    "wrap_frame_set",
    "wrap_mapping",
)

try:
    import starlink.Ast as StarAst
except ImportError:
    StarAst = None
    import astshim as AstShim

    USING_STARLINK_PYAST = False
else:
    AstShim = None
    USING_STARLINK_PYAST = True

# Public backend-dependent bindings.
StringStream: Any
FitsChan: Any
Mapping: Any
UnitMap: Any
ShiftMap: Any
Frame: Any
SkyFrame: Any
FrameSet: Any


if USING_STARLINK_PYAST:
    assert StarAst is not None

    class _StringStream:
        """A source/sink object compatible with starlink.Ast channel APIs."""

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

    class _AstWrapper:
        def __init__(self, obj: Any):
            self._obj = obj

        @classmethod
        def _from_obj(cls, obj: Any) -> Any:
            self = cls.__new__(cls)
            _AstWrapper.__init__(self, obj)
            return self

        def __getattr__(self, name: str) -> Any:
            return _wrap_ast_object(getattr(self._obj, name))

    def _serialize_ast_object(obj: Any, showComments: bool = True) -> str:
        sink = _StringStream()
        options = "" if showComments else "Comment=0"
        chan = StarAst.Channel(None, sink, options=options)
        chan.write(_unwrap_ast_object(obj))
        return sink.to_string()

    def _deserialize_ast_object(serialized: str) -> Any:
        source = _StringStream(serialized)
        chan = StarAst.Channel(source)
        return chan.read()

    def _coerce_to_star_ast_object(obj: Any) -> Any:
        match obj:
            case _AstWrapper():
                return obj._obj
            case StarAst.FrameSet() | StarAst.Mapping() | StarAst.Frame():
                return obj
            case _:
                if hasattr(obj, "show"):
                    try:
                        return _deserialize_ast_object(obj.show())
                    except Exception:
                        return obj
                return obj

    def _unwrap_ast_object(obj: Any) -> Any:
        return _coerce_to_star_ast_object(obj)

    def _wrap_ast_object(obj: Any) -> Any:
        match obj:
            case None:
                return obj
            case _AstWrapper():
                return obj
            case StarAst.FrameSet():
                return _FrameSet._from_obj(obj)
            case StarAst.SkyFrame():
                return _SkyFrame._from_obj(obj)
            case StarAst.Frame():
                return _Frame._from_obj(obj)
            case StarAst.Mapping():
                return _Mapping._from_obj(obj)
            case _:
                coerced = _coerce_to_star_ast_object(obj)
                if coerced is not obj:
                    return _wrap_ast_object(coerced)
                return obj

    class _Mapping(_AstWrapper):
        @staticmethod
        def fromString(serialized: str) -> _Mapping:
            obj = _deserialize_ast_object(serialized)
            if not isinstance(obj, StarAst.Mapping):
                raise TypeError(f"Serialized object is not a Mapping: {type(obj)}")
            return _Mapping._from_obj(obj)

        def show(self, showComments: bool = True) -> str:
            return _serialize_ast_object(self._obj, showComments=showComments)

        def simplified(self) -> _Mapping:
            return _Mapping._from_obj(self._obj.simplify())

        def applyForward(self, xy: Any) -> Any:
            return self._obj.tran(xy, True)

        def applyInverse(self, xy: Any) -> Any:
            return self._obj.tran(xy, False)

        def then(self, other: _Mapping) -> _Mapping:
            return _Mapping._from_obj(StarAst.CmpMap(self._obj, _unwrap_ast_object(other), True))

        def inverted(self) -> _Mapping:
            copy = self._obj.copy()
            copy.invert()
            return _Mapping._from_obj(copy)

    class _UnitMap(_Mapping):
        def __init__(self, n_coord: int):
            super().__init__(StarAst.UnitMap(n_coord))

    class _ShiftMap(_Mapping):
        def __init__(self, shift: Iterable[float]):
            super().__init__(StarAst.ShiftMap(list(shift)))

    class _Frame(_AstWrapper):
        def __init__(self, n_axes: int, options: str = ""):
            super().__init__(StarAst.Frame(n_axes, options))

        @property
        def ident(self) -> str:
            return self._obj.Ident

        def setUnit(self, axis: int, unit: str) -> None:
            setattr(self._obj, f"Unit_{axis}", unit)

        def getUnit(self, axis: int) -> str:
            return getattr(self._obj, f"Unit_{axis}")

        def setLabel(self, axis: int, label: str) -> None:
            setattr(self._obj, f"Label_{axis}", label)

        def getBottom(self, axis: int) -> float:
            return getattr(self._obj, f"Bottom_{axis}")

        def getTop(self, axis: int) -> float:
            return getattr(self._obj, f"Top_{axis}")

        def show(self) -> str:
            return _serialize_ast_object(self._obj)

    class _SkyFrame(_Frame):
        def __init__(self, options: str = ""):
            _AstWrapper.__init__(self, StarAst.SkyFrame(options))

    class _FrameSet(_AstWrapper):
        BASE = 1

        def __init__(self, base_frame: _Frame):
            super().__init__(StarAst.FrameSet(_unwrap_ast_object(base_frame)))

        @property
        def nFrame(self) -> int:
            return self._obj.Nframe

        @property
        def base(self) -> int:
            return self._obj.Base

        @base.setter
        def base(self, value: int) -> None:
            self._obj.Base = value

        @property
        def current(self) -> int:
            return self._obj.Current

        @current.setter
        def current(self, value: int) -> None:
            self._obj.Current = value

        def addFrame(self, iframe: int, mapping: _Mapping, frame: _Frame) -> None:
            self._obj.addframe(iframe, _unwrap_ast_object(mapping), _unwrap_ast_object(frame))

        def getFrame(self, iframe: int, copy: bool = True) -> _Frame:
            del copy
            return _wrap_ast_object(self._obj.getframe(iframe))

        def getMapping(self, iframe1: int | None = None, iframe2: int | None = None) -> _Mapping:
            if iframe1 is None:
                iframe1 = self.base
            if iframe2 is None:
                iframe2 = self.current
            return _wrap_ast_object(self._obj.getmapping(iframe1, iframe2))

        def show(self) -> str:
            return _serialize_ast_object(self._obj)

        @staticmethod
        def fromString(serialized: str) -> _FrameSet:
            obj = _deserialize_ast_object(serialized)
            if not isinstance(obj, StarAst.FrameSet):
                raise TypeError(f"Serialized object is not a FrameSet: {type(obj)}")
            return _FrameSet._from_obj(obj)

    class _FitsChan:
        def __init__(self, stream: _StringStream | None = None, options: str = ""):
            source = stream if stream is not None else None
            sink = stream if stream is not None else None
            self._obj = StarAst.FitsChan(source, sink, options)

        def read(self) -> Any:
            return _wrap_ast_object(self._obj.read())

        def write(self, obj: Any) -> int:
            return self._obj.write(_unwrap_ast_object(obj))

        def setFitsI(self, keyword: str, value: int) -> None:
            self._obj.setfitsI(keyword, value, "", 1)

        def __iter__(self) -> Any:
            return iter(self._obj)

    StringStream = _StringStream
    FitsChan = _FitsChan
    Mapping = _Mapping
    UnitMap = _UnitMap
    ShiftMap = _ShiftMap
    Frame = _Frame
    SkyFrame = _SkyFrame
    FrameSet = _FrameSet

    def wrap_mapping(mapping: Any) -> Any:
        return _wrap_ast_object(mapping)

    def wrap_frame_set(frame_set: Any) -> Any:
        return _wrap_ast_object(frame_set)

else:
    assert AstShim is not None

    StringStream = AstShim.StringStream
    FitsChan = AstShim.FitsChan
    Mapping = AstShim.Mapping
    UnitMap = AstShim.UnitMap
    ShiftMap = AstShim.ShiftMap
    Frame = AstShim.Frame
    SkyFrame = AstShim.SkyFrame
    FrameSet = AstShim.FrameSet

    def wrap_mapping(mapping: Any) -> Any:
        return mapping

    def wrap_frame_set(frame_set: Any) -> Any:
        return frame_set

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
    "XY",
    "YX",
    "Box",
    "Domain",
    "Interval",
    "SerializableDomain",
)

from collections.abc import Iterator, Sequence
from typing import Any, ClassVar, NamedTuple, Protocol, Self, TypedDict, final, overload

import numpy as np
import pydantic
import pydantic_core.core_schema as pcs

# Interval and Box are defined as regular Python classes rather than
# dataclasses or Pydantic models because we might want to implement them as
# compiled-extension types in the future, and we want that to be transparent.

# In a similar vein, we avoid declaring specific types for multidimensional
# points or extents (other than ``tuple[int, ...]`` for numpy-compatible
# shapes) in order to leave room for more fully-featured types to be added
# upstream of this package in the future.


class YX[T](NamedTuple):
    """A pair of per-dimension objects, ordered ``(y, x)``.

    Notes
    -----
    `YX` is used for slices, shapes, and other 2-d pairs when the most
    natural ordering is ``(y, x)``.  Because it is a `tuple`, however,
    arithmetic operations behave as they would on a
    `collections.abc.Sequence`, not a mathematical vector (e.g. adding
    concatenates).

    See Also
    --------
    XY
    """

    y: T
    """The y / row object."""

    x: T
    """The x / column object."""

    @property
    def xy(self) -> XY:
        """A tuple of the same objects in the opposite order."""
        return XY(x=self.x, y=self.y)


class XY[T](NamedTuple):
    """A pair of per-dimension objects, ordered ``(x, y)``.

    Notes
    -----
    `XY` is used for points and other 2-d pairs when the most natural ordering
    is ``(x, y)``.  Because it is a `tuple`, however, arithmetic operations
    behave as they would on a `collections.abc.Sequence`, not a mathematical
    vector (e.g. adding concatenates).

    See Also
    --------
    YX
    """

    x: T
    """The x / column object."""

    y: T
    """The y / row object."""

    @property
    def yx(self) -> YX:
        """A tuple of the same objects in the opposite order."""
        return YX(y=self.y, x=self.x)


class _SerializedInterval(TypedDict):
    start: int
    stop: int


@final
class Interval:
    """A 1-d integer interval with positive size.

    Parameters
    ----------
    start
        Inclusive minimum point in the interval.
    stop
        One past the maximum point in the interval.

    Notes
    -----
    Adding or subtracting an `int` from an interval returns a shifted interval.

    `Interval` implements the necessary hooks to be included directly in a
    `pydantic.BaseModel`, even though it is neither a built-in type nor a
    Pydantic model itself.
    """

    def __init__(self, start: int, stop: int):
        # Coerce to be defensive against numpy int scalars.
        self._start = int(start)
        self._stop = int(stop)
        if not (self._stop > self._start):
            raise ValueError(f"Interval must have positive size; got [{self._start}, {self._stop})")

    __slots__ = ("_start", "_stop")

    factory: ClassVar[IntervalSliceFactory]
    """A factory for creating intervals using slice syntax.

    For example::

        interval = Interval.factory[2:5]
    """

    @classmethod
    def hull(cls, first: int | Interval, *args: int | Interval) -> Interval:
        """Construct an interval that includes all of the given points and/or
        intervals.
        """
        if type(first) is Interval:
            rmin = first.min
            rmax = first.max
        else:
            rmin = rmax = first
        for arg in args:
            if type(arg) is Interval:
                rmin = min(rmin, arg.min)
                rmax = max(rmax, arg.max)
            else:
                rmin = min(rmin, arg)
                rmax = max(rmax, arg)
        return Interval(start=rmin, stop=rmax + 1)

    @classmethod
    def from_size(cls, size: int, start: int = 0) -> Interval:
        """Construct an interval from its size and optional start."""
        return cls(start=start, stop=start + size)

    @property
    def start(self) -> int:
        """Inclusive minimum point in the interval (`int`)."""
        return self._start

    @property
    def stop(self) -> int:
        """One past the maximum point in the interval (`int`)."""
        return self._stop

    @property
    def min(self) -> int:
        """Inclusive minimum point in the interval (`int`)."""
        return self.start

    @property
    def max(self) -> int:
        """Inclusive maximum point in the interval (`int`)."""
        return self.stop - 1

    @property
    def size(self) -> int:
        """Size of the interval (`int`)."""
        return self.stop - self.start

    @property
    def range(self) -> __builtins__.range:
        """An iterable over all values in the interval
        (`__builtins__.range`).
        """
        return range(self.start, self.stop)

    @property
    def arange(self) -> np.ndarray:
        """An array of all the values in the interval (`numpy.ndarray`)."""
        return np.arange(self.start, self.stop)

    @property
    def center(self) -> float:
        """The center of the interval (`float`)."""
        return 0.5 * (self.min + self.max)

    def __str__(self) -> str:
        return f"{self.start}:{self.stop}"

    def __repr__(self) -> str:
        return f"Interval(start={self.start}, stop={self.stop})"

    def __eq__(self, other: object) -> bool:
        if type(other) is Interval:
            return self._start == other._start and self._stop == other._stop
        return False

    def __add__(self, other: int) -> Interval:
        return Interval(start=self.start + other, stop=self.stop + other)

    def __sub__(self, other: int) -> Interval:
        return Interval(start=self.start - other, stop=self.stop - other)

    def __contains__(self, x: int) -> bool:
        return x >= self.start and x < self.stop

    @overload
    def contains(self, other: Interval | int) -> bool: ...

    @overload
    def contains(self, other: np.ndarray) -> np.ndarray: ...

    def contains(self, other: Interval | int | float | np.ndarray) -> bool | np.ndarray:
        """Test whether this interval fully contains another or one or more
        points.

        Parameters
        ----------
        other
            Another interval to compare to, or one or more position values.

        Returns
        -------
        `bool` | `numpy.ndarray`
            If a single interval or value was passed, a single `bool`.  If an
            array was passed, an array with the same shape.

        Notes
        -----
        In order to yield the desired behavior for floating-point arguments,
        points are actually tested against an interval that is 0.5 larger on
        both sides: this makes positions within the outer boundary of pixels
        (but beyond the centers of those pixels, which have integer positions)
        appear "on the image".
        """
        if isinstance(other, Interval):
            return self.start <= other.start and self.stop >= other.stop
        else:
            result = np.logical_and(self.start - 0.5 <= other, other < self.stop + 0.5)
            if not result.shape:
                return bool(result)
            return result

    def intersection(self, other: Interval) -> Interval | None:
        """Return an interval that is contained by both ``self`` and ``other``.

        When there is no overlap between the intervals, `None` is returned.
        """
        new_start = max(self.start, other.start)
        new_stop = min(self.stop, other.stop)
        if new_start < new_stop:
            return Interval(start=new_start, stop=new_stop)
        return None

    def dilated_by(self, padding: int) -> Interval:
        """Return a new interval padded by the given amount on both sides."""
        return Interval(start=self._start - padding, stop=self._stop + padding)

    def slice_within(self, other: Interval) -> slice:
        """Return the `slice` that corresponds to the values in this interval
        when the items of the container being sliced correspond to ``other``.

        This assumes ``other.contains(self)``.
        """
        return slice(self.start - other.start, self.stop - other.start)

    @classmethod
    def from_legacy(cls, legacy: Any) -> Interval:
        """Convert from an `lsst.geom.IntervalI` instance."""
        return cls(legacy.begin, legacy.end)

    def to_legacy(self) -> Any:
        """Convert to an `lsst.geom.IntervalI` instance."""
        from lsst.geom import IntervalI

        return IntervalI(min=self.min, max=self.max)

    def __reduce__(self) -> tuple[type[Interval], tuple[int, int]]:
        return (
            Interval,
            (
                self._start,
                self._stop,
            ),
        )

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: pydantic.GetCoreSchemaHandler
    ) -> pcs.CoreSchema:
        from_typed_dict = pcs.chain_schema(
            [
                handler(_SerializedInterval),
                pcs.no_info_plain_validator_function(cls._validate),
            ]
        )
        return pcs.json_or_python_schema(
            json_schema=from_typed_dict,
            python_schema=pcs.union_schema([pcs.is_instance_schema(Interval), from_typed_dict]),
            serialization=pcs.plain_serializer_function_ser_schema(cls._serialize, info_arg=False),
        )

    @classmethod
    def _validate(cls, data: _SerializedInterval) -> Interval:
        return cls(**data)

    def _serialize(self) -> _SerializedInterval:
        return {"start": self._start, "stop": self._stop}


class IntervalSliceFactory:
    """A factory for `Interval` objects using array-slice syntax.

    Notes
    -----
    When indexed with a single slice, this returns an `Interval`::

        assert Interval.factory[3:6] == Interval(start=3, stop=6)

    """

    def __getitem__(self, s: slice) -> Interval:
        if s.step is not None and s.step != 1:
            raise ValueError(f"Slice {s} has non-unit step.")
        return Interval(start=s.start, stop=s.stop)


Interval.factory = IntervalSliceFactory()


class _SerializedBox(TypedDict):
    y: _SerializedInterval
    x: _SerializedInterval


class Box:
    """An axis-aligned 2-d rectangular region.

    Parameters
    ----------
    y
        Interval for the y dimension.
    x
        Interval for the x dimension.

    Notes
    -----
    `Box` implements the necessary hooks to be included directly in a
    `pydantic.BaseModel`, even though it is neither a built-in type nor a
    Pydantic model itself.
    """

    def __init__(self, y: Interval, x: Interval):
        self._intervals = YX(y, x)

    __slots__ = ("_intervals",)

    factory: ClassVar[BoxSliceFactory]
    """A factory for creating boxes using slice syntax.

    For example::

        box = Box.factory[2:5, 3:9]
    """

    @classmethod
    def from_shape(cls, shape: Sequence[int], start: Sequence[int] | None = None) -> Box:
        """Construct a box from its shape and optional start.

        Parameters
        ----------
        shape
            Sequence of sizes, ordered ``(y, x)`` (except for `XY` instances).
        start
            Sequence of starts, ordered ``(y, x)`` (except for `XY` instances).
        """
        if start is None:
            start = (0,) * len(shape)
        match shape:
            case XY(x=x_size, y=y_size):
                pass
            case [y_size, x_size]:
                pass
            case _:
                raise ValueError(f"Invalid sequence for shape: {shape!r}.")
        match start:
            case XY(x=x_start, y=y_start):
                pass
            case [y_start, x_start]:
                pass
            case _:
                raise ValueError(f"Invalid sequence for shape: {shape!r}.")
        return Box(y=Interval.from_size(y_size, start=y_start), x=Interval.from_size(x_size, start=x_start))

    @property
    def start(self) -> YX[int]:
        """Tuple holding the starts of the intervals ordered ``(y, x)``
        (`YX` [`int`]).
        """
        return YX(self.y.start, self.x.start)

    @property
    def shape(self) -> YX[int]:
        """Tuple holding the sizes of the intervals ordered ``(y, x)``
        (`YX` [`int`]).
        """
        return YX(self.y.size, self.x.size)

    @property
    def x(self) -> Interval:
        """Shortcut for the last dimension's interval (`int`)."""
        return self._intervals[-1]

    @property
    def y(self) -> Interval:
        """Shortcut for the second-to-last dimension's interval (`int`)."""
        return self._intervals[-2]

    def __eq__(self, other: object) -> bool:
        if type(other) is Box:
            return self._intervals == other._intervals
        return False

    def __str__(self) -> str:
        return f"[y={self.y}, x={self.x}]"

    def __repr__(self) -> str:
        return f"Box(y={self.y!r}, x={self.x!r})"

    @overload
    def contains(self, other: Box, /) -> bool: ...

    @overload
    def contains(self, *, y: int, x: int) -> bool: ...

    @overload
    def contains(self, *, y: np.ndarray, x: np.ndarray) -> np.ndarray: ...

    def contains(
        self,
        other: Box | None = None,
        *,
        y: int | np.ndarray | None = None,
        x: int | np.ndarray | None = None,
    ) -> bool | np.ndarray:
        """Test whether this box fully contains another or one or more points.

        Parameters
        ----------
        other
            Another box to compare to.  Not compatible with the ``y`` and ``x``
            arguments.
        y
            One or more integer Y coordinates to test for containment.
            If an array, an array of results will be returned.
        x
            One or more integer X coordinates to test for containment.
            If an array, an array of results will be returned.

        Returns
        -------
        `bool` | `numpy.ndarray`
            If ``other`` was passed or ``x`` and ``y`` are both scalars, a
            single `bool` value.  If ``x`` and ``y`` are arrays, a boolean
            array with their broadcasted shape.

        Notes
        -----
        In order to yield the desired behavior for floating-point arguments,
        points are actually tested against an interval that is 0.5 larger on
        both sides: this makes positions within the outer boundary of pixels
        (but beyond the centers of those pixels, which have integer positions)
        appear "on the image".
        """
        if other is not None:
            if x is not None or y is not None:
                raise TypeError("Too many arguments to 'Box.contain'.")
            return all(a.contains(b) for a, b in zip(self._intervals, other._intervals, strict=True))
        elif x is None or y is None:
            raise TypeError("Not enough arguments to 'Box.contain'.")
        else:
            result = np.logical_and(self.x.contains(x), self.y.contains(y))
            if not result.shape:
                return bool(result)
            return result

    def intersection(self, other: Box) -> Box | None:
        """Return a box that is contained by both ``self`` and ``other``.

        When there is no overlap between the boxes, `None` is returned.
        """
        intervals = []
        for a, b in zip(self._intervals, other._intervals, strict=True):
            if (r := a.intersection(b)) is None:
                return None
            intervals.append(r)
        return Box(*intervals)

    def dilated_by(self, padding: int) -> Box:
        """Return a new box padded by the given amount on all sides."""
        return Box(*[i.dilated_by(padding) for i in self._intervals])

    def slice_within(self, other: Box) -> YX[slice]:
        """Return a `tuple` of `slice` objects that correspond to the
        positions in this box this box when the items of the container being
        sliced correspond to ``other``.

        This assumes ``other.contains(self)``.
        """
        return YX(self.y.slice_within(other.y), self.x.slice_within(other.x))

    def boundary(self) -> Iterator[YX[int]]:
        """Iterate over the corners of the box as ``(y, x)`` tuples."""
        if len(self._intervals) != 2:
            raise TypeError("Box is not 2-d.")
        yield YX(self.y.min, self.x.min)
        yield YX(self.y.min, self.x.max)
        yield YX(self.y.max, self.x.max)
        yield YX(self.y.max, self.x.min)

    def __reduce__(self) -> tuple[type[Box], tuple[Interval, ...]]:
        return (Box, self._intervals)

    @classmethod
    def from_legacy(cls, legacy: Any) -> Box:
        """Convert from an `lsst.geom.Box2I` instance."""
        return cls(y=Interval.from_legacy(legacy.y), x=Interval.from_legacy(legacy.x))

    def to_legacy(self) -> Any:
        """Convert to an `lsst.geom.BoxI` instance."""
        from lsst.geom import Box2I

        return Box2I(x=self.x.to_legacy(), y=self.y.to_legacy())

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: pydantic.GetCoreSchemaHandler
    ) -> pcs.CoreSchema:
        from_typed_dict = pcs.chain_schema(
            [
                handler(_SerializedBox),
                pcs.no_info_plain_validator_function(cls._validate),
            ]
        )
        return pcs.json_or_python_schema(
            json_schema=from_typed_dict,
            python_schema=pcs.union_schema([pcs.is_instance_schema(Box), from_typed_dict]),
            serialization=pcs.plain_serializer_function_ser_schema(cls._serialize, info_arg=False),
        )

    @classmethod
    def _validate(cls, data: _SerializedBox) -> Box:
        return cls(y=Interval._validate(data["y"]), x=Interval._validate(data["x"]))

    def _serialize(self) -> _SerializedBox:
        return {"y": self.y._serialize(), "x": self.x._serialize()}

    def serialize(self) -> Box:
        """Return a Pydantic-friendly representation of this object.

        This method just returns the `Box` itself, since that already provides
        Pydantic serialization hooks.  It exists for compatibility with the
        `Domain` protocol.
        """
        return self

    @classmethod
    def deserialize(cls, serialized: SerializableDomain) -> Box:
        """Deserialize a domain object on the assumption it is a `Box`.

        This method just returns the `Box` itself, since that already provides
        Pydantic serialization hooks.  It exists for compatibility with the
        `Domain` protocol.
        """
        assert isinstance(serialized, Box)
        return serialized


class BoxSliceFactory:
    """A factory for `Box` objects using array-slice syntax.

    Notes
    -----
    When indexed with one or more slices, this returns a `Box`:

        assert (
            Box.factory[3:6, -1:2]
            == Box(x=Interval(start=-1, stop=2), y=Interval(start=3, stop=6)
        )
    """

    def __getitem__(self, key: tuple[slice, slice]) -> Box:
        match key:
            case XY(x=x, y=y):
                return Box(Interval.factory[y], Interval.factory[x])
            case (y, x):
                return Box(Interval.factory[y], Interval.factory[x])
            case _:
                raise TypeError("Expected exactly two slices.")


Box.factory = BoxSliceFactory()


# This is expected to become a union of concrete Domain types that we can
# serialize via pydantic.  Right now that's only Box.
type SerializableDomain = Box


class Domain(Protocol):
    """A protocol for objects that represent the validity region for a function
    defined in 2-d pixel coordinates.

    Notes
    -----
    Most objects natively have a simple 2-d bounding box as their domain
    (typically the boundary of a sensor), and the `Box` class is hence the
    most common domain implementation.  But sometimes a large chunk of that
    box may be missing due to vignetting or bad amplifiers, and we may want to
    transform from one coordinate system to another.  The Domain interface is
    intended to handle both of these cases as well.
    """

    def boundary(self) -> Iterator[YX[int]]:
        """Iterate over points on the boundary as ``(y, x)`` tuples."""
        ...

    @overload
    def contains(self, *, x: int, y: int) -> bool: ...

    @overload
    def contains(self, *, x: np.ndarray, y: np.ndarray) -> np.ndarray: ...

    def contains(self, *, x: int | np.ndarray, y: int | np.ndarray) -> bool | np.ndarray:
        """Test whether this box fully contains another or one or more points.

        Parameters
        ----------
        x
            One or more integer X coordinates to test for containment.
            If an array, an array of results will be returned.
        y
            One or more integer Y coordinates to test for containment.
            If an array, an array of results will be returned.

        Returns
        -------
        `bool` | `numpy.ndarray`
            If ``x`` and ``y`` are both scalars, a single `bool` value.  If
            ``x`` and ``y`` are arrays, a boolean array with their broadcasted
            shape.
        """
        ...

    def serialize(self) -> SerializableDomain:
        """Convert a domain instance into a serializable object."""
        ...

    @classmethod
    def deserialize(cls, serialized: SerializableDomain) -> Self:
        """Convert a serialized domain object into its in-memory form."""
        match serialized:
            case Box():
                return serialized  # type: ignore[return-value]
        raise RuntimeError(f"Cannot deserialize {serialized!r}.")

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
    "Bounds",
    "BoundsError",
    "Box",
    "BoxSliceFactory",
    "Interval",
    "IntervalSliceFactory",
    "NoOverlapError",
)

import math
from collections.abc import Callable, Iterator, Sequence
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    NamedTuple,
    Protocol,
    Self,
    TypedDict,
    TypeVar,
    cast,
    final,
    overload,
)

import numpy as np
import pydantic
import pydantic_core.core_schema as pcs

if TYPE_CHECKING:
    from ._concrete_bounds import SerializableBounds

# This pre-python-3.12 declaration is needed by Sphinx (probably the
# autodoc-typehints plugin.
T = TypeVar("T")

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

    def map[U](self, func: Callable[[T], U]) -> YX[U]:
        """Apply a function to both objects."""
        return YX(y=func(self.y), x=func(self.x))


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

    def map[U](self, func: Callable[[T], U]) -> XY[U]:
        """Apply a function to both objects."""
        return XY(x=func(self.x), y=func(self.y))


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
            raise IndexError(f"Interval must have positive size; got [{self._start}, {self._stop})")

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
        """An array of all the values in the interval (`numpy.ndarray`).

        Array values are integers.
        """
        return np.arange(self.start, self.stop)

    @property
    def absolute(self) -> IntervalSliceFactory:
        """A factory for constructing a contained `Interval` using slice
        syntax and absolute coordinates.

        Notes
        -----
        Slice bounds that are absent are replaced with the bounds of ``self``.
        """
        return IntervalSliceFactory(self, is_local=False)

    @property
    def local(self) -> IntervalSliceFactory:
        """A factory for constructing a contained `Interval` using a slice
        relative to the start of this one (`IntervalSliceFactory`).

        Notes
        -----
        This factory interprets slices as "local" coordinates, in which ``0``
        corresponds to ``self.start``.  Negative bounds are relative to
        ``self.stop``, as is usually the case for Python sequences.
        """
        return IntervalSliceFactory(self, is_local=True)

    def linspace(self, n: int | None = None, *, step: float | None = None) -> np.ndarray:
        """Return an array of values that spans the interval.

        Parameters
        ----------
        n
            How many values to return.  The default (if ``step`` is also not
            provided) is the size of the interval, i.e. equivalent to the
            `arange` property (but converted to `float`).
        step
            Set ``n`` such that the distance between points is equal to or
            just less than this.  Mutually exclusive with ``n``.

        Returns
        -------
        numpy.ndarray
            Array of `float` values.

        See Also
        --------
        numpy.linspace
        """
        if n is None:
            if step is None:
                return self.arange.astype(np.float64)
            n = math.ceil(self.size / step)
        elif step is not None:
            raise TypeError("'n' and 'step' cannot both be provided.")
        return np.linspace(self.min, self.max, n, dtype=np.float64)

    @property
    def center(self) -> float:
        """The center of the interval (`float`)."""
        return 0.5 * (self.min + self.max)

    def padded(self, padding: int) -> Interval:
        """Return a new interval expanded by the given padding on
        either side.
        """
        return Interval(self.start - padding, self.stop + padding)

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
    def contains(self, other: Interval | int | float) -> bool: ...

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

    def intersection(self, other: Interval) -> Interval:
        """Return an interval that is contained by both ``self`` and ``other``.

        When there is no overlap between the intervals, `NoOverlapError` is
        raised.
        """
        new_start = max(self.start, other.start)
        new_stop = min(self.stop, other.stop)
        if new_start < new_stop:
            return Interval(start=new_start, stop=new_stop)
        raise NoOverlapError(f"No overlap between {self} and {other}.")

    def dilated_by(self, padding: int) -> Interval:
        """Return a new interval padded by the given amount on both sides."""
        return Interval(start=self._start - padding, stop=self._stop + padding)

    def slice_within(self, other: Interval) -> slice:
        """Return the `slice` that corresponds to the values in this interval
        when the items of the container being sliced correspond to ``other``.

        This assumes ``other.contains(self)``.
        """
        if not other.contains(self):
            raise IndexError(
                f"Can not calculate a slice of {other} within {self} "
                "since the given interval does not contain this one."
            )
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
    When indexed with a single slice on the `Interval.factory` attribute, this
    returns an `Interval` with exactly the given bounds::

        assert Interval.factory[3:6] == Interval(start=3, stop=6)

    A missing start bound is replaced by ``0``, but a missing stop bound is
    not allowed.

    When obtained from the `Interval.absolute` property, indices are absolute
    coordinate values, but any omitted bounds are replaced with the parent
    interval's bounds::

        parent = Interval.factory[3:6]
        assert Interval.factory[4:5] == parent.absolute[:5]

    The final interval is also checked to be contained by the parent interval.

    When obtained from the `Interval.local` property, indices are interpreted
    as relative to the parent interval, and negative indices are relative to
    the end (like `~collections.abc.Sequence` indexing)::

        parent = Interval.factory[3:6]
        assert Interval.factory[4:5] == parent.local[1:-1]

    When the stop bound is greater than the size of the parent interval, the
    returned interval is clipped to be contained by the parent (as in
    `~collections.abc.Sequence` indexing).
    """

    def __init__(self, parent: Interval | None = None, is_local: bool = False):
        self._parent = parent
        self._is_local = is_local

    def __getitem__(self, s: slice) -> Interval:
        if s.step is not None and s.step != 1:
            raise ValueError(f"Slice {s} has non-unit step.")
        if self._is_local:
            assert self._parent is not None, "is_local=True requires a parent interval"
            start, stop, _ = s.indices(self._parent.size)
            start += self._parent.start
            stop += self._parent.start
        else:
            start = s.start
            stop = s.stop
            if start is None:
                if self._parent is None:
                    start = 0
                else:
                    start = self._parent.start
            if stop is None:
                if self._parent is None:
                    raise IndexError("An Interval cannot have an empty upper bound.")
                stop = self._parent.stop
        if self._parent is not None:
            if start < self._parent.start:
                raise IndexError(f"Absolute start {start} (passed as {s.start}) is < {self._parent.start}.")
            if stop > self._parent.stop:
                raise IndexError(f"Absolute stop {stop} (passed as {s.stop}) is > {self._parent.stop}.")
        return Interval(start=start, stop=stop)


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
                raise ValueError(f"Invalid sequence for start: {start!r}.")
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
        """The x-dimension interval (`int`)."""
        return self._intervals[-1]

    @property
    def y(self) -> Interval:
        """The y-dimension interval (`int`)."""
        return self._intervals[-2]

    @property
    def absolute(self) -> BoxSliceFactory:
        """A factory for constructing a contained `Box` using slice
        syntax and absolute coordinates.

        Notes
        -----
        Slice bounds that are absent are replaced with the bounds of ``self``.
        """
        return BoxSliceFactory(y=self.y.absolute, x=self.x.absolute)

    @property
    def local(self) -> BoxSliceFactory:
        """A factory for constructing a contained `Interval` using a slice
        relative to the start of this one (`BoxSliceFactory`).

        Notes
        -----
        This factory interprets slices as "local" coordinates, in which ``0``
        corresponds to ``self.start``.  Negative bounds are relative to
        ``self.stop``, as is usually the case for Python sequences.
        """
        return BoxSliceFactory(y=self.y.local, x=self.x.local)

    def meshgrid(self, n: int | Sequence[int] | None = None, *, step: float | None = None) -> XY[np.ndarray]:
        """Return a pair of 2-d arrays of the coordinate values of the box.

        Parameters
        ----------
        n
            Number of points in each dimension.  If a sequence, points are
            assumed to be ordered ``(x, y)`` unless a `YX` instance is
            provided.
        step
            Set ``n`` such that the distance between points is equal to or
            just less than this in each dimension.  Mutually exclusive with
            ``n``.

        Returns
        -------
        `XY` [`numpy.ndarray`]
            A pair of arrays, each of which is 2-d with floating-point values.

        See Also
        --------
        numpy.meshgrid
        """
        if n is not None and step is not None:
            raise TypeError("'n' and 'step' cannot both be provided.")
        match n:
            case int():
                ax = self.x.linspace(n)
                ay = self.y.linspace(n)
            case YX(y=ny, x=nx):
                ax = self.x.linspace(nx)
                ay = self.y.linspace(ny)
            case [nx, ny]:
                ax = self.x.linspace(nx)
                ay = self.y.linspace(ny)
            case None:
                ax = self.x.linspace(step=step)
                ay = self.y.linspace(step=step)
            case _:
                raise ValueError(f"Unexpected values for n ({n})")
        return XY(*np.meshgrid(ax, ay))

    def padded(self, padding: int) -> Box:
        """Return a new box expanded by the given padding on
        all sides.
        """
        return Box(y=self.y.padded(padding), x=self.x.padded(padding))

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
                raise TypeError("Too many arguments to 'Box.contains'.")
            return all(a.contains(b) for a, b in zip(self._intervals, other._intervals, strict=True))
        elif x is None or y is None:
            raise TypeError("Not enough arguments to 'Box.contains'.")
        else:
            result = np.logical_and(self.x.contains(x), self.y.contains(y))
            if not result.shape:
                return bool(result)
            return result

    @overload
    def intersection(self, other: Box) -> Box: ...

    @overload
    def intersection(self, other: Bounds) -> Bounds: ...

    def intersection(self, other: Bounds) -> Bounds:
        """Return a bounds object that is contained by both ``self`` and
        ``other``.

        When there is no overlap, `NoOverlapError` is raised.
        """
        from ._concrete_bounds import _intersect_box

        return _intersect_box(self, other)

    def dilated_by(self, padding: int) -> Box:
        """Return a new box padded by the given amount on all sides."""
        return Box(*[i.dilated_by(padding) for i in self._intervals])

    def slice_within(self, other: Box) -> YX[slice]:
        """Return a `tuple` of `slice` objects that correspond to the
        positions in this box when the items of the container being sliced
        correspond to ``other``.

        This assumes ``other.contains(self)``.
        """
        return YX(self.y.slice_within(other.y), self.x.slice_within(other.x))

    @property
    def bbox(self) -> Box:
        """The box itself (`Box`).

        This is provided for compatibility with the `Bounds` interface.
        """
        return self

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
        `Bounds` protocol.
        """
        return self

    @classmethod
    def deserialize(cls, serialized: SerializableBounds) -> Box:
        """Deserialize a bounds object on the assumption it is a `Box`.

        This method just returns the `Box` itself, since that already provides
        Pydantic serialization hooks.  It exists for compatibility with the
        `Bounds` protocol.
        """
        assert isinstance(serialized, Box)
        return serialized


class BoxSliceFactory:
    """A factory for `Box` objects using array-slice syntax.

    Notes
    -----
    When `Box.factory` is indexed with a pair of slices, this returns a
    `Box` with exactly those bounds::

        assert (
            Box.factory[3:6, -1:2]
            == Box(x=Interval(start=-1, stop=2), y=Interval(start=3, stop=6)
        )

    A missing start bound is replaced by ``0``, but a missing stop bound is
    not allowed.

    When obtained from the `Box.absolute` property, indices are absolute
    coordinate values, but any omitted bounds are replaced with the parent
    box's bounds::

        parent = Box.factory[3:6, -1:2]
        assert Box.factory[4:5, 0:2] == parent.absolute[:5, 0:]

    The final box is also checked to be contained by the parent box.

    When obtained from the `Box.local` property, indices are interpreted
    as relative to the parent box, and negative indices are relative to
    the end (like `~collections.abc.Sequence` indexing)::

        parent = Box.factory[3:6, -1:2]
        assert Box.factory[4:5, 0:2] == parent.local[1:-1, 1:]
    """

    def __init__(
        self, y: IntervalSliceFactory = Interval.factory, x: IntervalSliceFactory = Interval.factory
    ):
        self._y = y
        self._x = x

    def __getitem__(self, key: tuple[slice, slice]) -> Box:
        match key:
            case XY(x=x, y=y):
                return Box(y=self._y[y], x=self._x[x])
            case (y, x):
                return Box(y=self._y[y], x=self._x[x])
            case _:
                raise TypeError("Expected exactly two slices.")


Box.factory = BoxSliceFactory()


class Bounds(Protocol):
    """A protocol for objects that represent the validity region for a function
    defined in 2-d pixel coordinates.

    Notes
    -----
    Most objects natively have a simple 2-d bounding box as their bounds
    (typically the boundary of a sensor), and the `Box` class is hence the
    most common bounds implementation.  But sometimes a large chunk of that
    box may be missing due to vignetting or bad amplifiers, and we may want to
    transform from one coordinate system to another.  The Bounds interface is
    intended to handle both of these cases as well.
    """

    @property
    def bbox(self) -> Box: ...

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

    def intersection(self, bounds: Bounds) -> Bounds:
        """Compute the intersection of this bounds object with another."""
        ...

    def serialize(self) -> SerializableBounds:
        """Convert a bounds instance into a serializable object."""
        ...

    @classmethod
    def deserialize(cls, serialized: SerializableBounds) -> Self:
        """Convert a serialized bounds object into its in-memory form."""
        from ._concrete_bounds import deserialize_bounds

        return cast(Self, deserialize_bounds(serialized))


class BoundsError(ValueError):
    """Exception raised when an object is evaluated outside its bounds."""


class NoOverlapError(ValueError):
    """Exception raised when intervals or bounds do not overlap."""

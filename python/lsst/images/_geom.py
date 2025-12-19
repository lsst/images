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
    "Box",
    "Interval",
)

from collections.abc import Iterator, Sequence
from typing import Any, ClassVar, TypedDict, final, overload

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
    """

    def __init__(self, start: int, stop: int):
        # Coerce to be defensive against numpy int scalars.
        self._start = int(start)
        self._stop = int(stop)
        if not (self._stop > self._start):
            raise ValueError(f"Interval must have positive size; got [{self._start}, {self._stop})")

    __slots__ = ("_start", "_stop")

    factory: ClassVar[IntervalSliceFactory]

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
        """Inclusive minimum point in the interval."""
        return self._start

    @property
    def stop(self) -> int:
        """One past the maximum point in the interval."""
        return self._stop

    @property
    def min(self) -> int:
        """Inclusive minimum point in the interval."""
        return self.start

    @property
    def max(self) -> int:
        """Inclusive maximum point in the interval."""
        return self.stop - 1

    @property
    def size(self) -> int:
        """Size of the interval."""
        return self.stop - self.start

    @property
    def range(self) -> range:
        """A `range` object that iterates over all values in the interval."""
        return range(self.start, self.stop)

    @property
    def arange(self) -> np.ndarray:
        """A `numpy.ndarray` of all the values in the interval."""
        return np.arange(self.start, self.stop)

    @property
    def center(self) -> float:
        """The center of the interval."""
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

    def contains(self, other: Interval) -> bool:
        """Test whether this interval fully contains another."""
        return self.start <= other.start and self.stop >= other.stop

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


class Box(Sequence[Interval]):
    """An axis-aligned [hyper]rectangular region.

    Parameters
    ----------
    *args
        Intervals for each dimension.
    """

    def __init__(self, *args: Interval):
        self._intervals = tuple(args)

    __slots__ = ("_intervals",)

    factory: ClassVar[BoxSliceFactory]

    @classmethod
    def from_shape(cls, shape: Sequence[int], start: Sequence[int] | None = None) -> Box:
        """Construct a box from its shape and optional start."""
        if start is None:
            start = (0,) * len(shape)
        return Box(
            *[Interval.from_size(size, start=i_start) for size, i_start in zip(shape, start, strict=True)]
        )

    @property
    def shape(self) -> tuple[int, ...]:
        """Tuple holding the sizes of the intervals in all dimension."""
        return tuple([i.size for i in self._intervals])

    @property
    def x(self) -> Interval:
        """Shortcut for the last dimension's interval."""
        return self._intervals[-1]

    @property
    def y(self) -> Interval:
        """Shortcut for the second-to-last dimension's interval."""
        return self._intervals[-2]

    @property
    def z(self) -> Interval:
        """Shortcut for the third-to-last dimension's interval."""
        return self._intervals[-3]

    def __eq__(self, other: object) -> bool:
        if type(other) is Box:
            return self._intervals == other._intervals
        return False

    def __len__(self) -> int:
        return len(self._intervals)

    @overload
    def __getitem__(self, key: int) -> Interval: ...

    @overload
    def __getitem__(self, key: slice) -> Box: ...

    def __getitem__(self, key: object) -> Box | Interval:
        match key:
            case slice():
                return Box(*self._intervals[key])
            case int():
                return self._intervals[key]
            case _:
                raise TypeError("Box can only be indexed with slices or integers.")

    def __iter__(self) -> Iterator[Interval]:
        return iter(self._intervals)

    def __str__(self) -> str:
        return f"[{', '.join([str(i) for i in self._intervals])}]"

    def __repr__(self) -> str:
        return f"Box({', '.join([repr(i) for i in self._intervals])})"

    def contains(self, other: Box) -> bool:
        """Test whether this box fully contains another."""
        return all(a.contains(b) for a, b in zip(self, other, strict=True))

    def intersection(self, other: Box) -> Box | None:
        """Return a box that is contained by both ``self`` and ``other``.

        When there is no overlap between the box, `None` is returned.
        """
        intervals = []
        for a, b in zip(self._intervals, other._intervals, strict=True):
            if (r := a.intersection(b)) is None:
                return None
            intervals.append(r)
        return Box(*intervals)

    def dilated_by(self, padding: int) -> Box:
        """Return a new box padded by the given amount on all sides."""
        return Box(*[i.dilated_by(padding) for i in self])

    def slice_within(self, other: Box) -> tuple[slice, ...]:
        """Return a tuple of `slice` objects that corresponds to the values of
        this box when the items of the container being sliced correspond to
        ``other``.

        This assumes ``other.contains(self)``.
        """
        return tuple([a.slice_within(b) for a, b in zip(self, other, strict=True)])

    def __reduce__(self) -> tuple[type[Box], tuple[Interval, ...]]:
        return (Box, self._intervals)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: pydantic.GetCoreSchemaHandler
    ) -> pcs.CoreSchema:
        from_list_schema = pcs.chain_schema(
            [
                pcs.list_schema(handler(_SerializedInterval)),
                pcs.no_info_plain_validator_function(cls._validate),
            ]
        )
        return pcs.json_or_python_schema(
            json_schema=from_list_schema,
            python_schema=pcs.union_schema([pcs.is_instance_schema(Box), from_list_schema]),
            serialization=pcs.plain_serializer_function_ser_schema(cls._serialize, info_arg=False),
        )

    @classmethod
    def _validate(cls, data: list[_SerializedInterval]) -> Box:
        return cls(*[Interval._validate(i) for i in data])

    def _serialize(self) -> list[_SerializedInterval]:
        return [i._serialize() for i in self]


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

    def __getitem__(self, key: slice | tuple[slice, ...]) -> Box:
        match key:
            case slice():
                return Box(Interval.factory[key])
            case tuple():
                return Box(*[Interval.factory[s] for s in key])
            case _:
                raise TypeError("Expected slice or tuple of slices.")


Box.factory = BoxSliceFactory()

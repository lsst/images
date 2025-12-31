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
    "FloatType",
    "IntegerType",
    "NumberType",
    "SignedIntegerType",
    "UnsignedIntegerType",
    "is_unsigned",
)

import enum
from typing import Literal, TypeGuard

import numpy as np
import numpy.typing as npt


class NumberType(enum.StrEnum):
    """Enumeration of array values types supported by the library."""

    bool = enum.auto()
    uint8 = enum.auto()
    uint16 = enum.auto()
    uint32 = enum.auto()
    uint64 = enum.auto()
    int8 = enum.auto()
    int16 = enum.auto()
    int32 = enum.auto()
    int64 = enum.auto()
    float32 = enum.auto()
    float64 = enum.auto()

    def to_numpy(self) -> type:
        """Convert an enumeration member to the corresponding numpy scalar
        type object.

        Returns
        -------
        scalar_type
            Numpy scalar type, e.g. `numpy.int16`.  Note that this inherits
            from `type`, not `numpy.dtype` (though a `numpy.dtype` instance
            can always be constructed from it).
        """
        return getattr(np, self.value)

    @classmethod
    def from_numpy(cls, dtype: npt.DTypeLike) -> NumberType:
        """Construct an enumeration member from anything that can be coerced
        to `numpy.dtype`.

        Parameters
        ----------
        dtype
            Object convertible to `numpy.dtype`.

        Returns
        -------
        member
            Enumeration member.
        """
        return cls(np.dtype(dtype).name)

    def require_unsigned(self) -> UnsignedIntegerType:
        """Raise `TypeError` if this enumeration does not represent an
        unsigned integer type, and return it if it does.
        """
        if is_unsigned(self):
            return self
        raise TypeError(f"{self} is not an unsigned integer type.")


type UnsignedIntegerType = (
    Literal[NumberType.bool]
    | Literal[NumberType.uint8]
    | Literal[NumberType.uint16]
    | Literal[NumberType.uint32]
    | Literal[NumberType.uint64]
)

type SignedIntegerType = (
    Literal[NumberType.int8]
    | Literal[NumberType.int16]
    | Literal[NumberType.int32]
    | Literal[NumberType.int64]
)


type IntegerType = SignedIntegerType | UnsignedIntegerType

type FloatType = Literal[NumberType.float32] | Literal[NumberType.float64]


def is_unsigned(t: NumberType) -> TypeGuard[UnsignedIntegerType]:
    """Test whether a `NumberType` corresponds to an unsigned integer type."""
    return np.dtype(t.to_numpy()).kind == "u"

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

__all__ = ("SerializableBounds", "deserialize_bounds")


from ._cell_grid import CellGridBounds
from ._geom import Bounds, Box

type SerializableBounds = Box | CellGridBounds


def deserialize_bounds(serialized: SerializableBounds) -> Bounds:
    """Convert a serialized bounds object into its in-memory form."""
    match serialized:
        case Box() | CellGridBounds():
            return serialized  # type: ignore[return-value]
    raise RuntimeError(f"Cannot deserialize {serialized!r}.")

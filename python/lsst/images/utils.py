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

__all__ = ("is_none", "round_half_away_from_zero", "round_half_up")

import math
import operator
import sys

if sys.version_info >= (3, 14, 0):
    is_none = operator.is_none  # type: ignore[attr-defined]
else:

    def is_none(x: object) -> bool:
        """Test whether an object is None."""
        return x is None


def round_half_up(x: float) -> int:
    """Round a `float` to an `int`, always rounding half up.

    Note that Python's built-in `round` implements the "round half to even"
    strategy.  This function implements the strategy used in `lsst.geom.Point`
    conversions.
    """
    return math.floor(x + 0.5)


def round_half_away_from_zero(x: float) -> int:
    """Round a `float` to an `int`, always rounding away from zero.

    Note that Python's built-in `round` implements the "round half to even"
    strategy.  This function implements the C/C++ standard strategy.
    """
    if x > 0:
        return math.floor(x + 0.5)
    else:
        return math.ceil(x - 0.5)

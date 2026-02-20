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

__all__ = ("is_none",)

import operator
import sys

if sys.version_info >= (3, 14, 0):
    is_none = operator.is_none  # type: ignore[attr-defined]
else:

    def is_none(x: object) -> bool:
        """Test whether an object is None."""
        return x is None

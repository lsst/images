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

__all__ = ("MaskModel",)


import pydantic

from .._geom import Box, Interval
from .._mask import MaskPlane
from ._asdf_utils import ArrayReferenceModel


class MaskModel(pydantic.BaseModel):
    """Pydantic model used to represent the serialized form of a `.Mask`."""

    data: ArrayReferenceModel
    start: list[int]
    planes: list[MaskPlane | None]

    @property
    def bbox(self) -> Box:
        """The 2-d bounding box of the mask."""
        return Box(
            *[Interval.factory[begin : begin + size] for begin, size in zip(self.start, self.data.shape[:-1])]
        )

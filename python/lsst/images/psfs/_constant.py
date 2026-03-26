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

__all__ = ("ConstantPSFSerializationModel", "ConstantPointSpreadFunction")

from functools import cached_property
from typing import Any

import pydantic

from lsst.images._image import Image

from .. import serialization
from .._geom import Bounds, Box, SerializableBounds
from ._base import PointSpreadFunction


class ConstantPointSpreadFunction(PointSpreadFunction):
    """A simple PSF suitable for tests that do not require a real PSF."""

    def __init__(self, constant: float, bounds: Bounds, stamp_size: int) -> None:
        self.constant = constant
        self._stamp_size = stamp_size
        self._bounds = bounds

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ConstantPointSpreadFunction):
            return NotImplemented
        if self.constant != other.constant:
            return False
        if self._stamp_size != other._stamp_size:
            return False
        if self._bounds != other._bounds:
            return False
        return True

    def __repr__(self) -> str:
        return (
            f"ConstantPointSpreadFunction({self.constant}, "
            f"stamp_size={self._stamp_size}, bounds={self._bounds!r})"
        )

    @property
    def bounds(self) -> Bounds:
        return self._bounds

    @cached_property
    def kernel_bbox(self) -> Box:
        r = self._stamp_size // 2
        return Box.factory[-r : r + 1, -r : r + 1]

    def compute_kernel_image(self, *, x: float, y: float) -> Image:
        return Image(self.constant, bbox=self.kernel_bbox)

    def compute_stellar_image(self, *, x: float, y: float) -> Image:
        r = self._stamp_size // 2
        return Image(self.constant, start=(round(y) - r, round(x) - r), shape=self.kernel_bbox.shape)

    def compute_stellar_bbox(self, *, x: float, y: float) -> Box:
        r = self._stamp_size // 2
        xi = round(x)
        yi = round(y)
        return Box.factory[yi - r : yi + r + 1, xi - r : xi + r + 1]

    def serialize(self, archive: serialization.OutputArchive[Any]) -> ConstantPSFSerializationModel:
        return ConstantPSFSerializationModel(
            constant=self.constant, stamp_size=self._stamp_size, bounds=self._bounds.serialize()
        )

    @classmethod
    def deserialize(
        cls, model: ConstantPSFSerializationModel, archive: serialization.InputArchive[Any]
    ) -> ConstantPointSpreadFunction:
        return cls(
            constant=model.constant, bounds=Bounds.deserialize(model.bounds), stamp_size=model.stamp_size
        )

    @staticmethod
    def _get_archive_tree_type(
        pointer_type: type[pydantic.BaseModel],
    ) -> type[ConstantPSFSerializationModel]:
        """Return the serialization model type for this object for an archive
        type that uses the given pointer type.
        """
        return ConstantPSFSerializationModel


class ConstantPSFSerializationModel(serialization.ArchiveTree):
    constant: float = pydantic.Field(description="Constant value for PSF.")
    stamp_size: int = pydantic.Field(
        description="Width of the (square) images returned by this PSF's methods."
    )

    bounds: SerializableBounds = pydantic.Field(
        description="The bounds object that represents the PSF's validity region."
    )

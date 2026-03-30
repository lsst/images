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
    "GaussianPSFSerializationModel",
    "GaussianPointSpreadFunction",
)

from functools import cached_property
from typing import Any

import numpy as np
import pydantic

from lsst.images._image import Image

from .. import serialization
from .._geom import Bounds, Box, SerializableBounds
from ._base import PointSpreadFunction


class GaussianPointSpreadFunction(PointSpreadFunction):
    """A PSF with a spatially-invariant circular Gaussian profile."""

    def __init__(self, sigma: float, bounds: Bounds, stamp_size: int) -> None:
        if sigma <= 0:
            raise ValueError(f"sigma must be positive; got {sigma}.")
        if stamp_size <= 0:
            raise ValueError(f"stamp_size must be positive; got {stamp_size}.")
        if stamp_size % 2 != 1:
            raise ValueError(f"stamp_size must be odd number, got {stamp_size}")
        self.sigma = float(sigma)
        self._stamp_size = stamp_size
        self._bounds = bounds
        self._sigma2 = self.sigma * self.sigma

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, GaussianPointSpreadFunction):
            return NotImplemented
        if self.sigma != other.sigma:
            return False
        if self._stamp_size != other._stamp_size:
            return False
        if self._bounds != other._bounds:
            return False
        return True

    def __repr__(self) -> str:
        return (
            f"GaussianPointSpreadFunction({self.sigma}, "
            f"stamp_size={self._stamp_size}, bounds={self._bounds!r})"
        )

    @property
    def bounds(self) -> Bounds:
        return self._bounds

    @cached_property
    def kernel_bbox(self) -> Box:
        r = self._stamp_size // 2
        return Box.factory[-r : r + 1, -r : r + 1]

    @cached_property
    def _centered_coordinates(self) -> np.ndarray:
        r = self._stamp_size // 2
        return np.arange(-r, r + 1, dtype=np.float64)

    @cached_property
    def _kernel_array(self) -> np.ndarray:
        profile = np.exp(-0.5 * np.square(self._centered_coordinates) / self._sigma2)
        kernel = np.multiply.outer(profile, profile)
        kernel /= kernel.sum()
        return kernel

    def compute_kernel_image(self, *, x: float, y: float) -> Image:
        return Image(self._kernel_array.copy(), bbox=self.kernel_bbox)

    def compute_stellar_image(self, *, x: float, y: float) -> Image:
        x0 = round(x)
        y0 = round(y)
        # Calculate the profiles accounting for subpixel shifts.
        x_profile = np.exp(-0.5 * np.square(self._centered_coordinates - (x - x0)) / self._sigma2)
        y_profile = np.exp(-0.5 * np.square(self._centered_coordinates - (y - y0)) / self._sigma2)
        image = np.multiply.outer(y_profile, x_profile)
        image /= image.sum()
        r = self._stamp_size // 2
        return Image(image, start=(y0 - r, x0 - r))

    def compute_stellar_bbox(self, *, x: float, y: float) -> Box:
        r = self._stamp_size // 2
        xi = round(x)
        yi = round(y)
        return Box.factory[yi - r : yi + r + 1, xi - r : xi + r + 1]

    def serialize(self, archive: serialization.OutputArchive[Any]) -> GaussianPSFSerializationModel:
        return GaussianPSFSerializationModel(
            sigma=self.sigma, stamp_size=self._stamp_size, bounds=self._bounds.serialize()
        )

    @classmethod
    def deserialize(
        cls, model: GaussianPSFSerializationModel, archive: serialization.InputArchive[Any]
    ) -> GaussianPointSpreadFunction:
        return cls(sigma=model.sigma, bounds=Bounds.deserialize(model.bounds), stamp_size=model.stamp_size)

    @staticmethod
    def _get_archive_tree_type(
        pointer_type: type[pydantic.BaseModel],
    ) -> type[GaussianPSFSerializationModel]:
        """Return the serialization model type for this object for an archive
        type that uses the given pointer type.
        """
        return GaussianPSFSerializationModel


class GaussianPSFSerializationModel(serialization.ArchiveTree):
    sigma: float = pydantic.Field(
        description="Gaussian sigma for the PSF.",
    )
    stamp_size: int = pydantic.Field(
        description="Width of the (square) images returned by this PSF's methods."
    )
    bounds: SerializableBounds = pydantic.Field(
        description="The bounds object that represents the PSF's validity region."
    )

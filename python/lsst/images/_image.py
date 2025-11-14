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

__all__ = ("Image",)

from collections.abc import Sequence
from typing import final

import astropy.units
import numpy as np
import numpy.typing as npt

from ._geom import Box


@final
class Image:
    """A 2-d array that may be augmented with units and a nonzero origin.

    Parameters
    ----------
    array_or_fill, optional
        Array or fill value for the image.  If a fill value, ``bbox`` or
        ``shape`` must be provided.
    bbox, optional
        Bounding box for the image.
    start, optional
        Logical coordinates of the first pixel in the array.  Ignored if
        ``bbox`` is provided.  Defaults to zeros.
    shape, optional
        Dimensions of the array.  Only needed if ``array_or_fill` is not an
        array and ``bbox`` is not provided.
    unit, optional
        Units for the image's pixel values.
    dtype, optional
        Pixel data type override.

    Notes
    -----
    Indexing the `array` attribute of an `Image` does not take into account its
    `start` offset, but accessing a subimage by indexing an `Image` with a
    `Box` does, and the `bbox` of the subimage is set to match its location
    within the original image.
    """

    def __init__(
        self,
        array_or_fill: np.ndarray | int | float = 0,
        /,
        *,
        bbox: Box | None = None,
        start: Sequence[int] | None = None,
        shape: Sequence[int] | None = None,
        unit: astropy.units.Unit | None = None,
        dtype: npt.DTypeLike | None = None,
    ):
        if isinstance(array_or_fill, np.ndarray):
            if dtype is not None:
                array = np.array(array_or_fill, dtype=dtype)
            else:
                array = array_or_fill
            if bbox is None:
                bbox = Box.from_shape(array.shape, start=start)
            elif bbox.shape != array.shape:
                raise ValueError(
                    f"Explicit bbox shape {bbox.shape} does not match array with shape {array.shape}."
                )
            if shape is not None and shape != array.shape:
                raise ValueError(f"Explicit shape {shape} does not match array with shape {array.shape}.")

        else:
            if bbox is None:
                if shape is None:
                    raise TypeError("No bbox, shape, or array provided.")
                bbox = Box.from_shape(shape, start=start)
            elif shape is not None and shape != bbox.shape:
                raise ValueError(f"Explicit shape {shape} does not match bbox shape {bbox.shape}.")
            array = np.full(bbox.shape, array_or_fill, dtype=dtype)
        self._array = array
        self._bbox = bbox
        self._unit = unit

    @property
    def array(self) -> np.ndarray:
        """The low-level array.

        Assigning to this attribute modifies the existing array in place; the
        bounding box and underlying data pointer are never changed.
        """
        return self._array

    @array.setter
    def array(self, value: np.ndarray | int | float) -> None:
        self._array[...] = value

    @property
    def unit(self) -> astropy.units.Unit | None:
        """Units for the image's pixel values."""
        return self._unit

    @property
    def bbox(self) -> Box:
        """Bounding box for the image."""
        return self._bbox

    def __getitem__(self, bbox: Box) -> Image:
        return Image(self.array[bbox.slice_within(self._bbox)], bbox=bbox)

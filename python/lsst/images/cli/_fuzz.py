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

__all__ = ("shuffle_blocks",)

import numpy as np


def shuffle_blocks(
    image: np.ndarray,
    mask: np.ndarray,
    variance: np.ndarray,
    block_shape: tuple[int, int],
    rng: np.random.Generator,
) -> None:
    """Shuffle image, mask, and variance pixels within each block in place.

    A single permutation is drawn for each block and applied to all three
    planes, so a pixel's values move together and stay mutually consistent.
    Blocks at the edges may be smaller than ``block_shape`` when its dimensions
    do not divide the image evenly.

    Parameters
    ----------
    image : `numpy.ndarray`
        The 2-d image plane, modified in place.
    mask : `numpy.ndarray`
        The mask plane with a leading ``(ny, nx)`` shape and any trailing axes
        (for example a per-pixel byte axis), modified in place.
    variance : `numpy.ndarray`
        The 2-d variance plane, modified in place.
    block_shape : `tuple` [`int`, `int`]
        The ``(y, x)`` size of a single block.
    rng : `numpy.random.Generator`
        Random number generator used to permute pixels.
    """
    block_y, block_x = block_shape
    n_y, n_x = image.shape
    for y0 in range(0, n_y, block_y):
        y1 = min(y0 + block_y, n_y)
        for x0 in range(0, n_x, block_x):
            x1 = min(x0 + block_x, n_x)
            count = (y1 - y0) * (x1 - x0)
            permutation = rng.permutation(count)
            for plane in (image, mask, variance):
                block = plane[y0:y1, x0:x1]
                flat = block.reshape(count, *block.shape[2:])
                block[...] = flat[permutation].reshape(block.shape)

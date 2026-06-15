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

import unittest

import numpy as np

from lsst.images.cli._fuzz import shuffle_blocks


class ShuffleBlocksTestCase(unittest.TestCase):
    """Unit tests for the pure pixel-shuffling helper."""

    def test_consistency_even_blocks(self) -> None:
        # Each pixel carries the same code in all three planes, so a shared
        # per-block permutation must keep the planes aligned afterwards.
        ny, nx = 4, 4
        codes = np.arange(ny * nx, dtype=np.float64).reshape(ny, nx)
        image = codes.copy()
        variance = codes.copy()
        mask = codes.astype(np.uint8).reshape(ny, nx, 1).copy()

        shuffle_blocks(image, mask, variance, (2, 2), np.random.default_rng(7))

        np.testing.assert_array_equal(image, variance)
        np.testing.assert_array_equal(image.astype(np.uint8), mask[..., 0])
        # Every 2x2 block keeps its original multiset of values.
        for y0 in (0, 2):
            for x0 in (0, 2):
                np.testing.assert_array_equal(
                    np.sort(image[y0 : y0 + 2, x0 : x0 + 2], axis=None),
                    np.sort(codes[y0 : y0 + 2, x0 : x0 + 2], axis=None),
                )

    def test_partial_edge_blocks(self) -> None:
        # A 2x2 block does not divide a 5x5 image evenly; the edge blocks are
        # smaller and must still shuffle without error or value loss.
        ny, nx = 5, 5
        codes = np.arange(ny * nx, dtype=np.float64).reshape(ny, nx)
        image = codes.copy()
        variance = codes.copy()
        mask = codes.astype(np.uint8).reshape(ny, nx, 1).copy()

        shuffle_blocks(image, mask, variance, (2, 2), np.random.default_rng(3))

        np.testing.assert_array_equal(image, variance)
        np.testing.assert_array_equal(image.astype(np.uint8), mask[..., 0])
        np.testing.assert_array_equal(np.sort(image, axis=None), np.sort(codes, axis=None))


if __name__ == "__main__":
    unittest.main()

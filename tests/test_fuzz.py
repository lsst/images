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

import os
import tempfile
import unittest

import astropy.units as u
import numpy as np
from click.testing import CliRunner

from lsst.images import Image, MaskedImage, MaskPlane, MaskSchema
from lsst.images.cli import main
from lsst.images.cli._fuzz import shuffle_blocks
from lsst.images.serialization import read


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


def _make_masked_image() -> MaskedImage:
    """Build a small MaskedImage with noisy pixels and some mask bits set."""
    rng = np.random.default_rng(500)
    mi = MaskedImage(
        Image(rng.normal(100.0, 8.0, size=(64, 64)), dtype=np.float32, unit=u.nJy),
        mask_schema=MaskSchema([MaskPlane("BAD", "Bad pixel.")]),
        metadata={"hello": "world"},
    )
    mi.mask.array |= np.multiply.outer(mi.image.array < 100.0, mi.mask.schema.bitmask("BAD"))
    mi.variance.array = rng.normal(64.0, 0.5, size=mi.bbox.shape).astype(np.float32)
    return mi


class FuzzMaskedImageCommandTestCase(unittest.TestCase):
    """Tests for the fuzz-masked-image CLI command."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name

    def test_help(self) -> None:
        result = CliRunner().invoke(main, ["fuzz-masked-image", "--help"])
        self.assertEqual(result.exit_code, 0, result.output)

    def test_no_files_errors(self) -> None:
        result = CliRunner().invoke(main, ["fuzz-masked-image"])
        self.assertNotEqual(result.exit_code, 0)

    def test_round_trip_fits(self) -> None:
        src = os.path.join(self.tmp, "mi.fits")
        mi = _make_masked_image()
        mi.write(src)
        original_image = mi.image.array.copy()
        original_mask = mi.mask.array.copy()

        result = CliRunner().invoke(main, ["fuzz-masked-image", src])
        self.assertEqual(result.exit_code, 0, result.output)

        out = os.path.join(self.tmp, "mi.fuzzed.fits")
        self.assertTrue(os.path.exists(out))
        check = read(out)
        finite = np.isfinite(original_image)
        changed = float(np.mean(check.image.array[finite] != original_image[finite]))
        self.assertGreaterEqual(changed, 0.5)
        self.assertFalse(np.array_equal(check.mask.array, original_mask))
        # An untouched part of the object survives the round trip.
        self.assertEqual(check.metadata.get("hello"), "world")

    def test_skips_existing_without_overwrite(self) -> None:
        src = os.path.join(self.tmp, "mi.fits")
        _make_masked_image().write(src)
        out = os.path.join(self.tmp, "mi.fuzzed.fits")
        with open(out, "w") as stream:
            stream.write("EXISTING")
        result = CliRunner().invoke(main, ["fuzz-masked-image", src])
        self.assertEqual(result.exit_code, 0, result.output)
        with open(out) as stream:
            self.assertEqual(stream.read(), "EXISTING")


if __name__ == "__main__":
    unittest.main()

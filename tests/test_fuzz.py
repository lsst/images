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
from pathlib import Path

import astropy.units as u
import numpy as np
from click.testing import CliRunner

from lsst.images import Image, MaskedImage, MaskPlane, MaskSchema
from lsst.images.cli import main
from lsst.images.cli._fuzz import shuffle_blocks
from lsst.images.serialization import read_archive


def test_consistency_even_blocks() -> None:
    """Verify a shared per-block permutation keeps image/mask/variance planes
    aligned.
    """
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


def test_partial_edge_blocks() -> None:
    """Verify edge blocks smaller than block_size shuffle without error or
    value loss.
    """
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
    """Return a small MaskedImage with noisy pixels and some mask bits set."""
    rng = np.random.default_rng(500)
    mi = MaskedImage(
        Image(rng.normal(100.0, 8.0, size=(64, 64)), dtype=np.float32, unit=u.nJy),
        mask_schema=MaskSchema([MaskPlane("BAD", "Bad pixel.")]),
        metadata={"hello": "world"},
    )
    mi.mask.array |= np.multiply.outer(mi.image.array < 100.0, mi.mask.schema.bitmask("BAD"))
    mi.variance.array = rng.normal(64.0, 0.5, size=mi.bbox.shape).astype(np.float32)
    return mi


def test_fuzz_help() -> None:
    """Verify the fuzz-masked-image CLI command exposes help text."""
    result = CliRunner().invoke(main, ["fuzz-masked-image", "--help"])
    assert result.exit_code == 0, result.output


def test_fuzz_no_files_errors() -> None:
    """Verify fuzz-masked-image fails when no files are given."""
    result = CliRunner().invoke(main, ["fuzz-masked-image"])
    assert result.exit_code != 0


def test_fuzz_round_trip_fits(tmp_path: Path) -> None:
    """Verify fuzz-masked-image produces a modified but structurally valid
    FITS file.
    """
    src = tmp_path / "mi.fits"
    mi = _make_masked_image()
    mi.write(src)
    original_image = mi.image.array.copy()
    original_mask = mi.mask.array.copy()

    result = CliRunner().invoke(main, ["fuzz-masked-image", str(src)])
    assert result.exit_code == 0, result.output

    out = tmp_path / "mi.fuzzed.fits"
    assert os.path.exists(out)
    check = read_archive(out)
    finite = np.isfinite(original_image)
    changed = float(np.mean(check.image.array[finite] != original_image[finite]))
    assert changed >= 0.5
    assert not np.array_equal(check.mask.array, original_mask)
    # An untouched part of the object survives the round trip.
    assert check.metadata.get("hello") == "world"


def test_fuzz_skips_existing_without_overwrite(tmp_path: Path) -> None:
    """Verify fuzz-masked-image leaves an existing output file untouched when
    not overwriting.
    """
    src = tmp_path / "mi.fits"
    _make_masked_image().write(src)
    out = tmp_path / "mi.fuzzed.fits"
    with open(out, "w") as stream:
        stream.write("EXISTING")
    result = CliRunner().invoke(main, ["fuzz-masked-image", str(src)])
    assert result.exit_code == 0, result.output
    with open(out) as stream:
        assert stream.read() == "EXISTING"

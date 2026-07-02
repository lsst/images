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

import numpy as np
import pytest

from lsst.images import Box, Image, Interval
from lsst.images.fits import FitsCompressionOptions

try:
    import lsst.afw.image
    import lsst.geom

    HAVE_LEGACY = True
except ImportError:
    HAVE_LEGACY = False

skip_no_legacy = pytest.mark.skipif(not HAVE_LEGACY, reason="lsst legacy packages could not be imported.")


@pytest.fixture
def rng() -> np.random.Generator:
    """Return a seeded random number generator."""
    return np.random.default_rng(500)


@skip_no_legacy
def test_interval(rng: np.random.Generator) -> None:
    """Test Interval to/from legacy lsst.geom.IntervalI conversion."""
    i = Interval.factory[3:6]
    j = i.to_legacy()
    assert isinstance(j, lsst.geom.IntervalI)
    assert j.min == 3
    assert j.max == 5
    k = Interval.from_legacy(j)
    assert i == k


@skip_no_legacy
def test_box(rng: np.random.Generator) -> None:
    """Test Box to/from legacy lsst.geom.Box2I conversion."""
    b = Box.factory[3:6, -2:1]
    c = b.to_legacy()
    assert isinstance(c, lsst.geom.Box2I)
    assert c.y.min == 3
    assert c.y.max == 5
    assert c.x.min == -2
    assert c.x.max == 0
    d = Box.from_legacy(c)
    assert b == d


@skip_no_legacy
def test_image(rng: np.random.Generator) -> None:
    """Test Image to/from legacy lsst.afw.image.ImageD conversion."""
    i = Image(rng.normal(100.0, 8.0, size=(200, 251)), dtype=np.float64, yx0=(5, 8))
    j = i.to_legacy()
    assert isinstance(j, lsst.afw.image.ImageD)
    assert Box.from_legacy(j.getBBox()) == i.bbox
    np.testing.assert_array_equal(i.array, j.array)
    k = Image.from_legacy(j)
    assert i == k


@skip_no_legacy
def test_fits_compression_from_recipe(rng: np.random.Generator) -> None:
    """Test that we can convert butler configuration for a compression
    write recipe into a FitsCompressionOptions dict.
    """
    config = {
        "image": {
            "algorithm": "RICE_1",
            "quantization": {
                "dither": "SUBTRACTIVE_DITHER_2",
                "scaling": "STDEV_MASKED",
                "mask_planes": ["NO_DATA", "INTRP"],
                "level": 16.0,
            },
        },
        "mask": {
            "algorithm": "GZIP_2",
        },
    }
    assert FitsCompressionOptions.model_validate(config["image"]) == FitsCompressionOptions.LOSSY
    assert FitsCompressionOptions.model_validate(config["mask"]) == FitsCompressionOptions.DEFAULT

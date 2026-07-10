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

from pathlib import Path

import astropy.io.fits
import numpy as np
import pytest

from lsst.images import Box, Image
from lsst.images.fits import write as fits_write
from lsst.images.serialization import read_archive

try:
    import zarr  # noqa: F401

    from lsst.images.zarr import write as zarr_write

    HAVE_ZARR = True
except ImportError:
    HAVE_ZARR = False

skip_no_zarr = pytest.mark.skipif(not HAVE_ZARR, reason="zarr is not installed")


@skip_no_zarr
def test_fits_to_zarr_to_fits_preserves_primary_header(tmp_path: Path) -> None:
    """End-to-end FITS -> Zarr -> FITS preserves the primary header."""
    original = Image(
        np.arange(20, dtype=np.float32).reshape(4, 5),
        bbox=Box.factory[10:14, 20:25],
    )
    fits_a = tmp_path / "a.fits"
    zarr_path = tmp_path / "b.zarr"
    fits_b = tmp_path / "c.fits"

    def update_header(header: astropy.io.fits.Header) -> None:
        header["ORIGIN"] = "RUBIN"
        header["EXPTIME"] = 30.0

    fits_write(original, fits_a, update_header=update_header)
    from_fits = read_archive(fits_a, Image)
    zarr_write(from_fits, zarr_path)
    from_zarr = read_archive(zarr_path, Image)
    fits_write(from_zarr, fits_b)

    with astropy.io.fits.open(fits_b) as hdul:
        assert hdul[0].header["ORIGIN"] == "RUBIN"
        assert hdul[0].header["EXPTIME"] == 30.0

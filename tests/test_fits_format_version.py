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
import pytest

from lsst.images import Image
from lsst.images.fits import FitsInputArchive
from lsst.images.serialization import ArchiveReadError, write


def _write_simple_image_fits(path: Path | str) -> None:
    """Write a tiny Image to ``path`` via the high-level API."""
    image = Image(0.0, shape=(4, 4), dtype="float32")
    write(image, path)


def test_write_emits_fmtver_and_datamodl(tmp_path: Path) -> None:
    """Verify a freshly-written FITS carries FMTVER=1 and the root DATAMODL."""
    path = tmp_path / "x.fits"
    _write_simple_image_fits(path)
    with astropy.io.fits.open(path) as hdul:
        assert hdul[0].header["FMTVER"] == 1
        assert hdul[0].header["DATAMODL"] == "https://images.lsst.io/schemas/image-1.0.0"


def test_read_succeeds_when_fmtver_matches(tmp_path: Path) -> None:
    """Verify that a round-trip read of a freshly-written file succeeds."""
    path = tmp_path / "x.fits"
    _write_simple_image_fits(path)
    with FitsInputArchive.open(path):
        pass


def test_read_fails_when_fmtver_too_high(tmp_path: Path) -> None:
    """Verify that a file whose FMTVER is newer than this release raises."""
    path = tmp_path / "x.fits"
    _write_simple_image_fits(path)
    with astropy.io.fits.open(path, mode="update") as hdul:
        hdul[0].header["FMTVER"] = 2
        hdul.flush()
    with pytest.raises(ArchiveReadError):
        with FitsInputArchive.open(path):
            pass


def test_read_succeeds_when_fmtver_absent(tmp_path: Path) -> None:
    """Verify a legacy file lacking FMTVER reads successfully.

    The reader should default to format version 1 when FMTVER is absent.
    """
    path = tmp_path / "x.fits"
    _write_simple_image_fits(path)
    with astropy.io.fits.open(path, mode="update") as hdul:
        if "FMTVER" in hdul[0].header:
            del hdul[0].header["FMTVER"]
        if "DATAMODL" in hdul[0].header:
            del hdul[0].header["DATAMODL"]
        hdul.flush()
    with FitsInputArchive.open(path):
        pass

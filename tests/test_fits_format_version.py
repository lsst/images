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

from lsst.images import Image
from lsst.images.fits import FitsInputArchive
from lsst.images.serialization import ArchiveReadError, write_archive


def _write_simple_image_fits(path: Path | str) -> None:
    """Write a tiny Image to ``path`` via the high-level API."""
    image = Image(0.0, shape=(4, 4), dtype="float32")
    write_archive(image, path)


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


def test_read_fails_when_fmtver_absent(tmp_path: Path) -> None:
    """Verify a file lacking FMTVER is rejected rather than assumed to be v1.

    FitsOutputArchive writes FMTVER before anything else, so every file this
    reader can open has one; absence means a damaged file, and guessing 1
    would guess at the layout.
    """
    path = tmp_path / "x.fits"
    _write_simple_image_fits(path)
    with astropy.io.fits.open(path, mode="update") as hdul:
        del hdul[0].header["FMTVER"]
        hdul.flush()
    with pytest.raises(ArchiveReadError, match="FMTVER"):
        with FitsInputArchive.open(path):
            pass


def test_get_basic_info_fails_when_fmtver_absent(tmp_path: Path) -> None:
    """Verify the info-only read requires FMTVER as well."""
    path = tmp_path / "x.fits"
    _write_simple_image_fits(path)
    with astropy.io.fits.open(path, mode="update") as hdul:
        del hdul[0].header["FMTVER"]
        hdul.flush()
    with pytest.raises(ArchiveReadError, match="FMTVER"):
        FitsInputArchive.get_basic_info(path)


def test_read_succeeds_when_datamodl_absent(tmp_path: Path) -> None:
    """Verify a file with the layout but no schema card still opens.

    DATAMODL is informational on read, so only callers that ask for the
    schema through ``info`` need it; requiring the layout stamp must not
    start requiring this one too.
    """
    path = tmp_path / "x.fits"
    _write_simple_image_fits(path)
    with astropy.io.fits.open(path, mode="update") as hdul:
        del hdul[0].header["DATAMODL"]
        hdul.flush()
    with FitsInputArchive.open(path) as archive:
        with pytest.raises(ArchiveReadError, match="DATAMODL"):
            archive.info


def test_foreign_fits_is_rejected_cleanly(tmp_path: Path) -> None:
    """Verify a FITS file this package did not write raises ArchiveReadError.

    Such a file has never been readable -- it carries none of the container
    cards -- but it used to surface as a bare KeyError from the first card
    popped without a default.
    """
    path = tmp_path / "plain.fits"
    astropy.io.fits.PrimaryHDU(np.zeros((4, 4), dtype="float32")).writeto(path)
    with pytest.raises(ArchiveReadError):
        with FitsInputArchive.open(path):
            pass

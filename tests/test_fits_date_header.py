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
import astropy.time

from lsst.images import Image
from lsst.images.fits import ExtensionKey, FitsInputArchive


def _simple_image() -> Image:
    return Image(0.0, shape=(4, 4), dtype="float32")


def test_every_hdu_has_a_fits_compliant_date(tmp_path: Path) -> None:
    """Test that each HDU carries a DATE card recording approximately when
    the file was written.
    """
    before = astropy.time.Time.now()
    path = tmp_path / "x.fits"
    _simple_image().write(path)
    after = astropy.time.Time.now()
    with astropy.io.fits.open(path) as hdul:
        assert len(hdul) > 1
        for index, hdu in enumerate(hdul):
            extname = hdu.header.get("EXTNAME", "PRIMARY")
            assert "DATE" in hdu.header, f"hdu={index} ({extname}): no DATE header"
            # The value must parse in the FITS time format.
            date = astropy.time.Time(hdu.header["DATE"], format="fits")
            # And it must reflect this write, not a stale value.
            assert date.jd >= before.jd, f"hdu={index} ({extname}): DATE before write started"
            assert date.jd <= after.jd, f"hdu={index} ({extname}): DATE after write finished"


def test_update_header_cannot_set_a_stale_primary_date(tmp_path: Path) -> None:
    """Test that an ``update_header`` callback cannot leave a stale DATE in
    the primary header.
    """
    before = astropy.time.Time.now()
    path = tmp_path / "x.fits"
    _simple_image().write(path, update_header=lambda h: h.set("DATE", "1999-01-01T00:00:00"))
    after = astropy.time.Time.now()
    with astropy.io.fits.open(path) as hdul:
        date = astropy.time.Time(hdul[0].header["DATE"], format="fits")
        assert date.jd >= before.jd
        assert date.jd <= after.jd


def test_date_is_not_captured_as_opaque_metadata(tmp_path: Path) -> None:
    """Test that DATE is not stored in opaque metadata.

    DATE is regenerated on every write, so the value read back must not
    be carried in opaque metadata (which would propagate a stale date to
    the next write).
    """
    path = tmp_path / "x.fits"
    _simple_image().write(path)
    with FitsInputArchive.open(path) as archive:
        opaque = archive.get_opaque_metadata()
        primary_header = opaque.headers[ExtensionKey()]
        assert "DATE" not in primary_header

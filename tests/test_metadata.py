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

import warnings
from collections.abc import Mapping

import astropy.io.fits
import pytest

from lsst.images.fits import ExtensionKey, FitsExternalMetadata, FitsOpaqueMetadata
from lsst.images.serialization import EmptyExternalMetadata

EXTERNAL_KEYS = ["EXPTIME", "BGMEAN", "LSST ISR UNITS", "LOWER KEY", "NOVAL", "CPLX"]


def _make_header() -> astropy.io.fits.Header:
    """Return a primary header exercising repeated, HIERARCH, commentary,
    valueless, and complex cards.
    """
    header = astropy.io.fits.Header()
    header.append(("EXPTIME", 30.0), end=True)
    header.append(("BGMEAN", 1.5), end=True)
    header.append(("BGMEAN", 2.5), end=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", astropy.io.fits.verify.VerifyWarning)
        header.append(("HIERARCH LSST ISR UNITS", "adu"), end=True)
        # astropy preserves the case of HIERARCH keywords.
        header.append(("HIERARCH lower key", 3), end=True)
    header.append(("COMMENT", "a comment"), end=True)
    header.append(("HISTORY", "a history"), end=True)
    header.append(("", "blank"), end=True)
    header.append(astropy.io.fits.Card.fromstring("NOVAL   =".ljust(80)), end=True)
    header.append(("CPLX", 1 + 2j), end=True)
    return header


def test_empty_external_metadata() -> None:
    """Test that EmptyExternalMetadata behaves as an empty read-only
    mapping.
    """
    external = EmptyExternalMetadata()
    assert isinstance(external, Mapping)
    assert len(external) == 0
    assert list(external) == []
    assert "EXPTIME" not in external
    assert external.get("EXPTIME") is None
    with pytest.raises(KeyError):
        external["EXPTIME"]
    with pytest.raises(KeyError):
        external.get_all("EXPTIME")
    assert repr(external) == "EmptyExternalMetadata({})"


def test_fits_external_metadata_lookup() -> None:
    """Test case-insensitive lookup, repeated keywords, HIERARCH keywords,
    and value conversion.
    """
    external = FitsExternalMetadata(_make_header())
    assert external["EXPTIME"] == 30.0
    assert external["exptime"] == 30.0
    assert external["lsst isr units"] == "adu"
    assert external["HIERARCH LSST ISR UNITS"] == "adu"
    assert external["LOWER KEY"] == 3
    assert external["lower key"] == 3
    assert external["NOVAL"] is None
    assert external["CPLX"] == 1 + 2j
    assert external["BGMEAN"] in (1.5, 2.5)
    assert external.get_all("bgmean") == (1.5, 2.5)
    assert external.get_all("exptime") == (30.0,)
    assert external.get_all("NOVAL") == (None,)
    assert external.get_all("Lower Key") == (3,)
    assert "Exptime" in external
    assert "lower key" in external


def test_fits_external_metadata_hides_commentary_cards() -> None:
    """Test that COMMENT, HISTORY, and blank cards are not visible."""
    external = FitsExternalMetadata(_make_header())
    for key in ("COMMENT", "comment", "HISTORY", ""):
        assert key not in external
        with pytest.raises(KeyError):
            external[key]
        with pytest.raises(KeyError):
            external.get_all(key)


def test_fits_external_metadata_odd_keys() -> None:
    """Test that keys that cannot be FITS keywords are simply absent."""
    external = FitsExternalMetadata(_make_header())
    for key in ("roundtrip_test_1", "a=b", "é", "x" * 100):
        assert key not in external
        with pytest.raises(KeyError):
            external[key]
    assert 1 not in external


def test_fits_external_metadata_iteration() -> None:
    """Test that iteration yields each upper-case keyword once."""
    external = FitsExternalMetadata(_make_header())
    assert list(external) == EXTERNAL_KEYS
    assert len(external) == len(EXTERNAL_KEYS)
    assert repr(external).startswith("FitsExternalMetadata({'EXPTIME': 30.0")


def test_fits_external_metadata_lowercase_dotted_hierarch() -> None:
    """Test that a lower-case HIERARCH keyword that looks like a
    record-valued keyword is reported and found consistently.
    """
    header = astropy.io.fits.Header.fromstring("HIERARCH x.y = 3".ljust(80))
    header.append(("EXPTIME", 30.0), end=True)
    external = FitsExternalMetadata(header)
    assert list(external) == ["X.Y", "EXPTIME"]
    assert external["X.Y"] == 3
    assert external["x.y"] == 3
    assert external.get_all("X.Y") == (3,)
    assert dict(external) == {"X.Y": 3, "EXPTIME": 30.0}


def test_fits_opaque_metadata_external_metadata_tracks_primary_header() -> None:
    """Test that external metadata reflects a primary header added after it
    was first requested.
    """
    opaque_metadata = FitsOpaqueMetadata()
    assert len(opaque_metadata.external_metadata()) == 0
    opaque_metadata.add_header(_make_header(), name="", ver=1)
    assert list(opaque_metadata.external_metadata()) == EXTERNAL_KEYS


def test_fits_external_metadata_empty() -> None:
    """Test that a missing header behaves as an empty one."""
    external = FitsExternalMetadata(None)
    assert len(external) == 0
    assert "EXPTIME" not in external


def test_fits_opaque_metadata_external_metadata() -> None:
    """Test that only the primary header is exposed."""
    opaque_metadata = FitsOpaqueMetadata()
    assert len(opaque_metadata.external_metadata()) == 0
    opaque_metadata.add_header(_make_header(), name="", ver=1)
    extension_header = astropy.io.fits.Header()
    extension_header["EXTRA"] = 1
    opaque_metadata.add_header(extension_header, name="IMAGE", ver=1)
    external = opaque_metadata.external_metadata()
    assert list(external) == EXTERNAL_KEYS
    assert "EXTRA" not in external
    assert opaque_metadata.headers[ExtensionKey("IMAGE")]["EXTRA"] == 1

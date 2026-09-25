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

import astropy.time
import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from lsst.images import Box, Mask, MaskPlane, MaskSchema
from lsst.images.tests import annotate_errors, assert_masks_equal, assert_values_equal


def test_assert_values_equal_passes():
    """Assert equal arrays pass."""
    assert_values_equal(np.array([1, 2, 3]), np.array([1, 2, 3]))


def test_assert_values_equal_nan_equal_by_default():
    """Assert NaN compares equal by default."""
    a = np.array([1.0, np.nan])
    assert_values_equal(a, a.copy())


def test_assert_values_equal_exact_by_default():
    """Assert a small difference fails by default (exact equality)."""
    with pytest.raises(AssertionError, match="max abs diff"):
        assert_values_equal(np.array([1.0]), np.array([1.0 + 1e-6]))


def test_assert_values_equal_rtol_passes():
    """Assert an explicit relative tolerance passes."""
    assert_values_equal(np.array([1.0]), np.array([1.0 + 1e-6]), rtol=1e-5)


def test_assert_values_equal_fails_with_report():
    """Assert a mismatch raises a report message."""
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 4.0])
    with pytest.raises(AssertionError, match="1/3 values differ"):
        assert_values_equal(a, b)


def test_assert_values_equal_label_prefix():
    """Assert the label prefixes the failure message."""
    with pytest.raises(AssertionError, match=r"^mask\[BAD\]: "):
        assert_values_equal(np.array([True]), np.array([False]), label="mask[BAD]")


def test_assert_values_equal_shape_mismatch():
    """Assert a shape mismatch raises."""
    with pytest.raises(AssertionError, match="shape"):
        assert_values_equal(np.zeros((2, 2)), np.zeros((3, 3)))


def test_assert_values_equal_non_numeric_reports_indices():
    """Assert non-numeric arrays report differing indices."""
    a = np.array(["a", "b", "c"])
    b = np.array(["a", "b", "z"])
    with pytest.raises(AssertionError, match="values differ"):
        assert_values_equal(a, b)


def test_assert_values_equal_nan_not_equal_to_number():
    """Assert NaN still does not equal a finite number."""
    with pytest.raises(AssertionError):
        assert_values_equal(np.array([np.nan]), np.array([1.0]))


def test_assert_values_equal_atol_quantity_unit_aware():
    """Assert a Quantity atol is converted to the unit of a."""
    a = np.array([1.0, 2.0]) * u.deg
    b = a.to(u.arcsec)
    assert_values_equal(a, b, rtol=0.0, atol=1e-7 * u.arcsec)


def test_assert_values_equal_unit_aware():
    """Assert unit-aware comparison converts b to the unit of a."""
    a = np.array([1.0, 2.0]) * u.deg
    b = a.to(u.arcsec)
    assert_values_equal(a, b, rtol=1e-5, atol=0.0)


def test_assert_values_equal_incompatible_units_raise():
    """Assert incompatible units raise."""
    with pytest.raises(u.UnitConversionError):
        assert_values_equal(np.array([1.0]) * u.deg, np.array([1.0]) * u.s)


def test_assert_values_equal_time_within_atol():
    """Assert times close enough (with atol in seconds) pass."""
    t0 = astropy.time.Time("2026-01-01T00:00:00", scale="utc")
    t1 = t0 + astropy.time.TimeDelta(1e-6, format="sec")
    assert_values_equal(t1, t0, atol=1e-5)


def test_assert_values_equal_time_exact_by_default():
    """Assert times differing by any amount fail with zero tolerance."""
    t0 = astropy.time.Time("2026-01-01T00:00:00", scale="utc")
    t1 = t0 + astropy.time.TimeDelta(1e-6, format="sec")
    with pytest.raises(AssertionError, match="!= 0.0"):
        assert_values_equal(t1, t0)


def test_assert_values_equal_time_quantity_atol():
    """Assert a time-unit Quantity atol bounds the time difference."""
    t0 = astropy.time.Time("2026-01-01T00:00:00", scale="utc")
    t1 = t0 + astropy.time.TimeDelta(0.002, format="sec")
    assert_values_equal(t1, t0, atol=u.Quantity(3, "ms"))
    with pytest.raises(AssertionError, match="!= 0.0"):
        assert_values_equal(t1, t0, atol=u.Quantity(1, "ms"))


def test_assert_values_equal_time_incompatible_atol_raises():
    """Assert a non-time Quantity atol for times raises."""
    t0 = astropy.time.Time("2026-01-01T00:00:00", scale="utc")
    with pytest.raises(u.UnitConversionError):
        assert_values_equal(t0, t0, atol=u.Quantity(1.0, "m"))


def test_assert_values_equal_timedelta():
    """Assert timedeltas compare by their difference in seconds."""
    d0 = astropy.time.TimeDelta(30.0, format="sec")
    assert_values_equal(d0, astropy.time.TimeDelta(30.0, format="sec"))
    assert_values_equal(d0, astropy.time.TimeDelta(30.000001, format="sec"), atol=1e-5)
    with pytest.raises(AssertionError, match="!= 0.0"):
        assert_values_equal(d0, astropy.time.TimeDelta(30.5, format="sec"))
    with pytest.raises(ValueError, match="rtol"):
        assert_values_equal(d0, d0, rtol=1e-6)


def test_assert_values_equal_sky_coords_within_atol():
    """Assert sky coordinates close enough (with an angular atol) pass."""
    a = SkyCoord(10.0, 45.0, unit="deg")
    b = SkyCoord(10.0 + 1e-4, 45.0, unit="deg")
    assert_values_equal(a, b, atol=1 * u.arcsec)
    with pytest.raises(AssertionError, match=r"!= <Angle 0\. deg>"):
        assert_values_equal(a, b, atol=0.1 * u.arcsec)


def test_assert_values_equal_sky_coords_wrap():
    """Assert separations across the RA = 0 meridian compare correctly."""
    a = SkyCoord(359.9999, 30.0, unit="deg")
    b = SkyCoord(0.0001, 30.0, unit="deg")
    assert_values_equal(a, b, atol=1 * u.arcsec)


def test_assert_values_equal_sky_coords_exact_atol_zero():
    """Assert bare atol=0.0 demands exact coincidence."""
    a = SkyCoord(10.0, 45.0, unit="deg")
    assert_values_equal(a, SkyCoord(10.0, 45.0, unit="deg"), atol=0.0)
    with pytest.raises(AssertionError, match=r"!= <Angle 0\. deg>"):
        assert_values_equal(a, SkyCoord(10.0 + 1e-9, 45.0, unit="deg"), atol=0.0)


def test_assert_values_equal_sky_coords_unitless_atol_raises():
    """Assert a nonzero bare-float atol for sky coords raises."""
    a = SkyCoord(10.0, 45.0, unit="deg")
    with pytest.raises(TypeError, match="angular Quantity"):
        assert_values_equal(a, a, atol=1e-3)
    with pytest.raises(ValueError, match="rtol"):
        assert_values_equal(a, a, rtol=1e-6)


def test_assert_values_equal_wrap_angles():
    """Assert angles differing by whole turns compare equal."""
    a = 359.9999 * u.deg
    b = 0.0001 * u.deg
    assert_values_equal(a, b, atol=1 * u.arcsec, wrap_angles=True)
    assert_values_equal(1 * u.deg, -359.0 * u.deg, atol=0.0, wrap_angles=True)
    with pytest.raises(AssertionError, match="!="):
        assert_values_equal(a, b, atol=0.1 * u.arcsec, wrap_angles=True)


def test_assert_values_equal_wrap_angles_requires_units():
    """Assert wrap_angles demands angular units on operands and atol."""
    with pytest.raises(TypeError, match="angular units"):
        assert_values_equal(1.0, 1.0, wrap_angles=True)
    with pytest.raises(TypeError, match="angular Quantity"):
        assert_values_equal(1 * u.deg, 1 * u.deg, atol=1e-3, wrap_angles=True)
    with pytest.raises(ValueError, match="rtol"):
        assert_values_equal(1 * u.deg, 1 * u.deg, rtol=1e-6, wrap_angles=True)
    with pytest.raises(u.UnitConversionError):
        assert_values_equal(1 * u.s, 1 * u.s, wrap_angles=True)


def test_label_assertions_notes():
    """Assert a failing check is re-raised with the label as a note."""
    with pytest.raises(AssertionError) as exc_info:
        with annotate_errors("psf"):
            raise AssertionError("inner")
    assert exc_info.value.args == ("inner",)
    assert exc_info.value.__notes__ == ["psf"]


def test_label_assertions_passthrough_on_success():
    """Assert a passing block is not re-raised."""
    with annotate_errors("psf"):
        pass  # must not raise


def test_assert_masks_equal_notes_plane_diff() -> None:
    """Assert that a mask mismatch adds a per-plane +/- note."""
    schema = MaskSchema([MaskPlane("A", "dA")], dtype=np.uint8)
    m1 = Mask(0, schema=schema, bbox=Box.factory[0:2, 0:1])
    m1.set("A", np.array([[True], [False]]))
    m2 = Mask(0, schema=schema, bbox=Box.factory[0:2, 0:1])
    m2.set("A", np.array([[True], [True]]))
    with pytest.raises(AssertionError) as excinfo:
        assert_masks_equal(m1, m2)
    notes = excinfo.value.__notes__ or []
    assert any("mask[A]: +0 -1" in n for n in notes)

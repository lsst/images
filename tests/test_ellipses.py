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

import astropy.units as u
import numpy as np
import pytest
from scipy.stats import multivariate_normal

from lsst.images import XY, Box, Ellipse, EllipseArray, Image, InvalidEllipseError
from lsst.images.tests import assert_close

try:
    import lsst.afw.geom.ellipses

    HAVE_LEGACY = True
except ImportError:
    HAVE_LEGACY = False

skip_no_legacy = pytest.mark.skipif(not HAVE_LEGACY, reason="lsst legacy packages could not be imported.")

try:
    import galsim

    HAVE_GALSIM = True
except ImportError:
    HAVE_GALSIM = False

skip_no_galsim = pytest.mark.skipif(not HAVE_GALSIM, reason="galsim is not installed")


def test_ellipse_parameterizations_round_trip() -> None:
    """Test that moments->axes->moments and axes->moments->axes round trip."""
    ellipse = Ellipse(xx=4.0, yy=2.0, xy=0.5)
    from_axes = Ellipse(a=ellipse.a, b=ellipse.b, theta=ellipse.theta)
    assert_close(from_axes.xx, ellipse.xx)
    assert_close(from_axes.yy, ellipse.yy)
    assert_close(from_axes.xy, ellipse.xy)

    ellipse = Ellipse(a=3.0, b=2.0, theta=0.4 * u.rad)
    from_moments = Ellipse(xx=ellipse.xx, yy=ellipse.yy, xy=ellipse.xy)
    assert_close(from_moments.a, ellipse.a)
    assert_close(from_moments.b, ellipse.b)
    assert_close(from_moments.theta.to_value(u.rad), ellipse.theta.to_value(u.rad))


def test_ellipse_properties() -> None:
    """Verify the derived ellipse properties against hand-computed values."""
    ellipse = Ellipse(xx=4.0, yy=2.0, xy=0.5)
    np.testing.assert_array_equal(ellipse.matrix, [[4.0, 0.5], [0.5, 2.0]])
    det = 4.0 * 2.0 - 0.5**2
    assert_close(ellipse.det, det)
    assert_close(ellipse.tr, 6.0)
    assert_close(ellipse.area, np.pi * np.sqrt(det))
    assert_close(ellipse.r_det, det**0.25)
    assert_close(ellipse.r_tr, np.sqrt(0.5 * 6.0))


def test_ellipse_reduced_shear_consistent() -> None:
    """Test that reduced shear from moments agrees with reduced shear from
    axes.
    """
    from_moments = Ellipse(xx=4.0, yy=2.0, xy=0.5)
    from_axes = Ellipse(a=from_moments.a, b=from_moments.b, theta=from_moments.theta)
    assert_close(from_axes.reduced_shear, from_moments.reduced_shear)


def test_ellipse_swap() -> None:
    """Test that axes with a < b are swapped and theta offset by 90 degrees"""
    ellipse = Ellipse(a=2.0, b=3.0, theta=0.0 * u.rad)
    assert ellipse.a == 3.0
    assert ellipse.b == 2.0
    assert_close(ellipse.theta.to_value(u.rad), -0.5 * np.pi)

    ellipse = Ellipse(a=3.0, b=2.0, theta=(np.pi + 0.1) * u.rad)
    theta = ellipse.theta.to_value(u.rad)
    assert theta >= -0.5 * np.pi
    assert theta < 0.5 * np.pi


_THETA_BOUND_CASES = [
    (2.0, 1.0, -0.5 * np.pi),  # at -pi/2
    (2.0, 1.0, -0.5 * np.pi - 0.1),  # just beyond -pi/2
    (2.0, 1.0, 0.5 * np.pi),  # at +pi/2
    (2.0, 1.0, 0.5 * np.pi + 0.1),  # just beyond +pi/2
    (2.0, 1.0, -np.pi),  # at -pi
    (2.0, 1.0, np.pi),  # at +pi
    (2.0, 1.0, -np.pi - 0.1),  # just beyond -pi
    (2.0, 1.0, np.pi + 0.1),  # just beyond +pi
    (1.0, 2.0, 0.0),  # a < b swap, theta 0 -> -pi/2
    (1.0, 2.0, 0.5 * np.pi),  # a < b swap, theta +pi/2 -> 0
    (1.0, 2.0, 1.2),  # a < b swap, arbitrary theta
]


@pytest.mark.parametrize("a, b, theta_in", _THETA_BOUND_CASES)
def test_ellipse_theta_bounded(a: float, b: float, theta_in: float) -> None:
    """Test that scalar Ellipse theta lies in [-pi/2, pi/2)."""
    ellipse = Ellipse(a=a, b=b, theta=theta_in * u.rad)
    theta = ellipse.theta.to_value(u.rad)
    assert theta >= -0.5 * np.pi
    assert theta < 0.5 * np.pi


@pytest.mark.parametrize(
    "kwargs",
    [
        {"xx": np.nan, "yy": 2.0, "xy": 0.5},
        {"xx": 4.0, "yy": np.nan, "xy": 0.5},
        {"xx": 4.0, "yy": 2.0, "xy": np.nan},
        {"xx": np.inf, "yy": 2.0, "xy": 0.5},
        {"xx": -1.0, "yy": 2.0, "xy": 0.5},
        {"xx": 4.0, "yy": -1.0, "xy": 0.5},
        {"xx": 1.0, "yy": 1.0, "xy": 5.0},
    ],
)
def test_ellipse_invalid_moments(kwargs: dict) -> None:
    """Test that tnvalid moment combinations raise InvalidEllipseError."""
    with pytest.raises(InvalidEllipseError):
        Ellipse(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"a": np.nan, "b": 2.0, "theta": 0.0 * u.rad},
        {"a": 2.0, "b": np.nan, "theta": 0.0 * u.rad},
        {"a": 2.0, "b": 2.0, "theta": np.nan * u.rad},
        {"a": -1.0, "b": 2.0, "theta": 0.0 * u.rad},
        {"a": 2.0, "b": -1.0, "theta": 0.0 * u.rad},
    ],
)
def test_ellipse_invalid_axes(kwargs: dict) -> None:
    """Test that invalid axis combinations raise InvalidEllipseError."""
    with pytest.raises(InvalidEllipseError):
        Ellipse(**kwargs)


def test_ellipse_constructor_exclusivity() -> None:
    """Test that the constructor requires all or none of each
    parameterization.
    """
    with pytest.raises(TypeError):
        Ellipse()
    with pytest.raises(TypeError):
        Ellipse(xx=1.0, yy=2.0)
    with pytest.raises(TypeError):
        Ellipse(a=1.0, theta=2.0)


def test_ellipse_center() -> None:
    """Test ellipse center handling."""
    ellipse = Ellipse(xx=4.0, yy=2.0, xy=0.5, x=1.0, y=2.0)
    assert ellipse.center == XY(x=1.0, y=2.0)
    ellipse = Ellipse(xx=4.0, yy=2.0, xy=0.5, center=XY(x=1.0, y=2.0))
    assert ellipse.center == XY(x=1.0, y=2.0)
    with pytest.raises(TypeError):
        Ellipse(xx=4.0, yy=2.0, xy=0.5, center=XY(x=1.0, y=2.0), x=1.0, y=2.0)
    ellipse = Ellipse(xx=4.0, yy=2.0, xy=0.5)
    assert not hasattr(ellipse, "center")
    with pytest.raises(AttributeError):
        ellipse.center


def test_from_reduced_shear_round_trip() -> None:
    """Test Ellipse.from_reduced_shear against the reduced_shear property."""
    g = 0.3 + 0.1j
    ellipse = Ellipse.from_reduced_shear(g=g)
    assert_close(ellipse.reduced_shear, g)


def test_from_reduced_shear_args() -> None:
    """Test different argument combinations for from_reduced_shear."""
    with pytest.raises(TypeError):
        Ellipse.from_reduced_shear(g=0.1 + 0.2j, g1=0.1, g2=0.2)
    with pytest.raises(TypeError):
        Ellipse.from_reduced_shear(g1=0.1, g2=0.2, r_tr=1.0, r_det=1.0)
    with pytest.raises(TypeError):
        Ellipse.from_reduced_shear(g1=0.1, g2=0.2, tr=1.0, r_tr=1.0)
    with pytest.raises(TypeError):
        Ellipse.from_reduced_shear(g1=0.1, g2=0.2, det=1.0, r_det=1.0)
    with pytest.raises(TypeError):
        Ellipse.from_reduced_shear(g1=0.1, g2=0.2, r_tr=1.0, det=1.0)

    ellipse = Ellipse.from_reduced_shear(g=0.2 + 0.0j)
    assert ellipse.a > 0
    assert ellipse.b > 0
    assert ellipse.a >= ellipse.b
    assert_close(ellipse.r_det, 1.0)
    assert_close(ellipse.theta.to_value(u.rad), 0.0)

    ellipse = Ellipse.from_reduced_shear(g1=0.3, g2=0.1)
    assert ellipse.a >= ellipse.b
    assert_close(ellipse.theta.to_value(u.rad), 0.5 * np.atan2(0.1, 0.3))


def test_ellipse_array_basic() -> None:
    """Test EllipseArray construction, shape, length, and indexing."""
    array = EllipseArray(xx=np.array([4.0, 9.0]), yy=np.array([2.0, 4.0]), xy=np.array([0.5, 1.0]))
    assert array.shape == (2,)
    assert len(array) == 2
    assert isinstance(array[0], Ellipse)
    subset = array[:1]
    assert isinstance(subset, EllipseArray)
    assert subset.shape == (1,)


def test_ellipse_array_properties() -> None:
    """Test EllipseArray properties against scalar Ellipse."""
    xx = np.array([4.0, 9.0])
    yy = np.array([2.0, 4.0])
    xy = np.array([0.5, 1.0])
    array = EllipseArray(xx=xx, yy=yy, xy=xy)
    np.testing.assert_array_equal(array.xx, xx)
    np.testing.assert_array_equal(array.yy, yy)
    np.testing.assert_array_equal(array.xy, xy)
    assert array.matrix.shape == (2, 2, 2)
    for i in range(2):
        ellipse = Ellipse(xx=xx[i], yy=yy[i], xy=xy[i])
        assert_close(array.a[i], ellipse.a)
        assert_close(array.b[i], ellipse.b)
        assert_close(array.theta[i].to_value(u.rad), ellipse.theta.to_value(u.rad))
        assert_close(array.det[i], ellipse.det)
        assert_close(array.tr[i], ellipse.tr)
        assert_close(array.area[i], ellipse.area)
        assert_close(array.r_det[i], ellipse.r_det)
        assert_close(array.r_tr[i], ellipse.r_tr)


def test_ellipse_array_standardize() -> None:
    """Test that standardize marks invalid entries and swaps a<b in place."""
    a = np.array([3.0, 2.0, 1.0, 2.0, 5.0, np.nan])
    b = np.array([2.0, 3.0, 1.0, 5.0, -2.0, 1.0])
    theta = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) * u.rad
    array = EllipseArray(a=a, b=b, theta=theta)
    mask = array.standardize()
    np.testing.assert_array_equal(mask, [True, True, True, True, False, False])
    assert array.a[0] == 3.0 and array.b[0] == 2.0
    assert array.a[1] == 3.0 and array.b[1] == 2.0
    assert array.a[2] == 1.0 and array.b[2] == 1.0
    assert array.a[3] == 5.0 and array.b[3] == 2.0
    assert_close(array.theta[1].to_value(u.rad), -0.5 * np.pi)
    assert_close(array.theta[3].to_value(u.rad), -0.5 * np.pi)


@pytest.mark.parametrize("a, b, theta_in", _THETA_BOUND_CASES)
def test_ellipse_array_standardize_theta_bounded(a: float, b: float, theta_in: float) -> None:
    """Test that EllipseArray.standardize leaves theta in [-pi/2, pi/2)."""
    array = EllipseArray(a=np.array([a]), b=np.array([b]), theta=np.array([theta_in]) * u.rad)
    mask = array.standardize()
    assert mask[0]
    theta = array.theta[0].to_value(u.rad)
    assert theta >= -0.5 * np.pi
    assert theta < 0.5 * np.pi


def test_ellipse_array_reduced_shear() -> None:
    """Test that array reduced shear matches elementwise computation."""
    xx = np.array([4.0, 9.0])
    yy = np.array([2.0, 4.0])
    xy = np.array([0.5, 1.0])
    array = EllipseArray(xx=xx, yy=yy, xy=xy)
    from_axes = EllipseArray(a=array.a, b=array.b, theta=array.theta)
    np.testing.assert_allclose(array.reduced_shear, from_axes.reduced_shear)
    for i in range(2):
        ellipse = Ellipse(xx=xx[i], yy=yy[i], xy=xy[i])
        assert_close(array.reduced_shear[i], ellipse.reduced_shear)


def test_ellipse_array_reduced_shear_round_trip() -> None:
    """Test Ellipse.from_reduced_shear against the reduced_shear property."""
    g = np.array([0.3 + 0.1j, 0.2 + 0.0j])
    array = EllipseArray.from_reduced_shear(g=g)
    assert_close(array.reduced_shear, g)


_ARRAY_BROADCAST_PARAMS = [
    # Three moments arrays with broadcastable shapes.
    pytest.param(
        {
            "xx": np.arange(6.0).reshape(2, 3) + 4.0,  # (2, 3)
            "yy": np.array([2.0, 4.0, 6.0]),  # (3,)
            "xy": np.array([0.5, 1.0, 1.5]),  # (3,)
        },
        (2, 3),
        id="moments",
    ),
    # Three axes arrays with broadcastable shapes.
    pytest.param(
        {
            "a": np.array([[3.0, 4.0, 5.0]]),  # (1, 3)
            "b": np.array([2.0, 3.0, 1.0]),  # (3,)
            "theta": np.array([0.0, 0.5, -0.5]) * u.rad,  # (3,)
        },
        (1, 3),
        id="axes",
    ),
    # A scalar moment broadcast against equal-length arrays.
    pytest.param(
        {"xx": 4.0, "yy": np.array([2.0, 4.0, 6.0]), "xy": np.array([0.5, 1.0, 1.5])},
        (3,),
        id="moments-scalar",
    ),
    # A scalar theta broadcast against array a/b in the axes parameterization.
    pytest.param(
        {"a": np.array([3.0, 4.0, 5.0]), "b": np.array([2.0, 3.0, 1.0]), "theta": 0.4 * u.rad},
        (3,),
        id="axes-scalar-theta",
    ),
]


def _reference_array(kwargs: dict) -> EllipseArray:
    """Build an identical-shape array from explicitly-broadcast inputs."""
    if "xx" in kwargs:
        bxx, byy, bxy = np.broadcast_arrays(kwargs["xx"], kwargs["yy"], kwargs["xy"])
        return EllipseArray(xx=bxx, yy=byy, xy=bxy)
    else:
        ba, bb = np.broadcast_arrays(kwargs["a"], kwargs["b"])
        bth = np.broadcast_to(kwargs["theta"].to_value(u.rad), ba.shape)
        return EllipseArray(a=ba, b=bb, theta=bth * u.rad)


@pytest.mark.parametrize("kwargs, shape", _ARRAY_BROADCAST_PARAMS)
def test_ellipse_array_broadcast_shape_and_properties(kwargs: dict, shape: tuple) -> None:
    """Test that broadcastable differing shapes give the broadcast shape and
    matching derived properties.
    """
    array = EllipseArray(**kwargs)
    ref = _reference_array(kwargs)
    assert array.shape == shape
    assert len(array) == shape[0]
    for prop in ("a", "b", "det", "tr", "area", "r_det", "r_tr"):
        np.testing.assert_allclose(getattr(array, prop), getattr(ref, prop))
    np.testing.assert_allclose(array.theta.to_value(u.rad), ref.theta.to_value(u.rad))
    np.testing.assert_allclose(array.reduced_shear, ref.reduced_shear)


@pytest.mark.parametrize("kwargs, shape", _ARRAY_BROADCAST_PARAMS)
def test_ellipse_array_broadcast_matrix(kwargs: dict, shape: tuple) -> None:
    """Test that the matrix of a broadcast-built array has shape (2, 2, ...)
    with no error.
    """
    array = EllipseArray(**kwargs)
    ref = _reference_array(kwargs)
    np.testing.assert_allclose(array.matrix, ref.matrix)
    assert array.matrix.shape == (2, 2) + shape


def test_ellipse_array_broadcast_standardize_moments() -> None:
    """Test standardize on broadcasted shapes (moments)."""
    array = EllipseArray(
        xx=np.array([[4.0, 9.0, -1.0]]),  # (1, 3); last entry invalid (xx < 0)
        yy=np.array([2.0, 4.0, 6.0]),
        xy=np.array([0.5, 1.0, 1.5]),
    )
    mask = array.standardize()
    np.testing.assert_array_equal(mask.shape, (1, 3))
    assert not mask[0, 2]


def test_ellipse_array_broadcast_standardize_axes_swap() -> None:
    """Test standardize on broadcasted shapes (axes, with a < b swap)."""
    a = np.array([[3.0, 2.0, 1.0]])  # (1, 3)
    b = np.array([2.0, 3.0, 5.0])  # (3,)
    theta = np.array([0.0, 0.0, 0.0]) * u.rad
    array = EllipseArray(a=a, b=b, theta=theta)
    mask = array.standardize()
    np.testing.assert_array_equal(mask.shape, (1, 3))
    assert mask.all()
    # a < b at each column must have been swapped so a >= b after standardize.
    assert (array.a >= array.b).all()


@pytest.mark.parametrize("kwargs, shape", _ARRAY_BROADCAST_PARAMS)
def test_ellipse_array_broadcast_full_tuple_indexing(kwargs: dict, shape: tuple) -> None:
    """Test that full-tuple indexing into a broadcast-built array returns an
    Ellipse.
    """
    array = EllipseArray(**kwargs)
    last = tuple(s - 1 for s in shape)
    result = array[last]
    assert isinstance(result, Ellipse)
    ref = _reference_array(kwargs)[last]
    assert_close(result.a, ref.a)
    assert_close(result.b, ref.b)
    assert_close(result.theta.to_value(u.rad), ref.theta.to_value(u.rad))


def test_ellipse_array_ambiguous_broadcast_raises() -> None:
    """Test that non-broadcastable shapes are rejected at construction."""
    with pytest.raises(ValueError):
        EllipseArray(xx=np.ones((2, 3)), yy=np.ones(2), xy=np.ones(2))


def test_ellipse_array_copy_default_aliases_and_standardize_round_trips() -> None:
    """Test that with equal-shape inputs, arrays are not copied and
    standardize modifies them in-place.
    """
    a = np.array([3.0, 2.0])
    b = np.array([2.0, 3.0])
    theta = np.array([0.0, 0.0]) * u.rad
    array = EllipseArray(a=a, b=b, theta=theta)
    assert array.a is a
    assert array.b is b
    array.standardize()  # a[1] < b[1] -> swap in place
    np.testing.assert_array_equal(a, [3.0, 3.0])
    np.testing.assert_array_equal(b, [2.0, 2.0])
    # A broadcast-stored array is never aliased even under the default copy.
    yy = np.array([2.0, 4.0, 6.0])
    broadcast = EllipseArray(xx=np.ones((2, 3)) * 4.0, yy=yy, xy=np.ones(3) * 0.5)
    assert broadcast.yy is not yy
    assert broadcast.yy.shape == (2, 3)


def test_ellipse_array_copy_true_materializes_independent_arrays() -> None:
    """Test that with copy=True, stored arrays are copied and standardize
    leaves the original arrays unchanged.
    """
    a = np.array([3.0, 2.0])
    b = np.array([2.0, 3.0])
    theta = np.array([0.0, 0.0]) * u.rad
    array = EllipseArray(a=a, b=b, theta=theta, copy=True)
    assert array.a is not a
    assert array.b is not b
    np.testing.assert_array_equal(array.a, a)
    array.standardize()
    np.testing.assert_array_equal(a, [3.0, 2.0])
    np.testing.assert_array_equal(b, [2.0, 3.0])


def test_ellipse_array_copy_false_aliases_but_rejects_broadcast() -> None:
    """Test that copy=False references the input arrays, and fails if
    broadcasting is necessary.
    """
    a = np.array([3.0, 2.0])
    b = np.array([2.0, 3.0])
    theta = np.array([0.0, 0.0]) * u.rad
    array = EllipseArray(a=a, b=b, theta=theta, copy=False)
    assert array.a is a
    with pytest.raises(ValueError):
        EllipseArray(xx=np.ones((2, 3)), yy=np.ones(3), xy=np.ones(3), copy=False)
    with pytest.raises(TypeError):
        EllipseArray(xx=np.ones(2), yy=np.ones(2), xy=np.ones(2), copy="if_needed")


@skip_no_legacy
def test_cross_validate_ellipse_to_afw() -> None:
    """Test that our a/b/theta/area/r_det/r_tr match lsst.afw.geom.ellipses."""
    rng = np.random.default_rng(1234)
    xx = rng.uniform(1.0, 10.0, size=50)
    yy = rng.uniform(1.0, 10.0, size=50)
    xy = rng.uniform(-2.0, 2.0, size=50)
    for xxv, yyv, xyv in zip(xx, yy, xy):
        if xxv * yyv < xyv**2:
            continue
        ellipse = Ellipse(xx=xxv, yy=yyv, xy=xyv)
        legacy = lsst.afw.geom.ellipses.Axes(lsst.afw.geom.ellipses.Quadrupole(xxv, yyv, xyv))
        assert_close(ellipse.a, legacy.getA())
        assert_close(ellipse.b, legacy.getB())
        assert_close(ellipse.theta.to_value(u.rad), legacy.getTheta())
        assert_close(ellipse.area, legacy.getArea())
        assert_close(ellipse.r_det, legacy.getDeterminantRadius())
        assert_close(ellipse.r_tr, legacy.getTraceRadius())


@skip_no_legacy
def test_cross_validate_ellipse_from_afw() -> None:
    """Test that xx/yy/xy/area/r_det/r_tr match lsst.afw.geom.ellipses."""
    rng = np.random.default_rng(4321)
    a = rng.uniform(1.0, 5.0, size=50)
    b = rng.uniform(0.1, 5.0, size=50)
    theta = rng.uniform(-np.pi, np.pi, size=50)
    for av, bv, thetav in zip(a, b, theta):
        if av < bv:
            continue
        ellipse = Ellipse(a=av, b=bv, theta=thetav * u.rad)
        legacy = lsst.afw.geom.ellipses.Quadrupole(lsst.afw.geom.ellipses.Axes(av, bv, thetav))
        assert_close(ellipse.xx, legacy.getIxx())
        assert_close(ellipse.yy, legacy.getIyy())
        assert_close(ellipse.xy, legacy.getIxy())
        assert_close(ellipse.area, legacy.getArea())
        assert_close(ellipse.r_det, legacy.getDeterminantRadius())
        assert_close(ellipse.r_tr, legacy.getTraceRadius())


@skip_no_legacy
def test_cross_validate_reduced_shear_to_afw() -> None:
    """Test that our reduced_shear, theta, and r_tr match
    lsst.afw.geom.ellipses.
    """
    rng = np.random.default_rng(5678)
    xx = rng.uniform(1.0, 10.0, size=50)
    yy = rng.uniform(1.0, 10.0, size=50)
    xy = rng.uniform(-2.0, 2.0, size=50)
    for xxv, yyv, xyv in zip(xx, yy, xy):
        if xxv * yyv < xyv**2:
            continue
        ellipse = Ellipse(xx=xxv, yy=yyv, xy=xyv)
        legacy = lsst.afw.geom.ellipses.SeparableReducedShearTraceRadius(
            lsst.afw.geom.ellipses.Quadrupole(xxv, yyv, xyv)
        )
        assert_close(ellipse.reduced_shear.real, legacy.getE1())
        assert_close(ellipse.reduced_shear.imag, legacy.getE2())
        assert_close(ellipse.theta.to_value(u.rad), legacy.getEllipticity().getTheta())
        assert_close(ellipse.r_tr, legacy.getTraceRadius())


@skip_no_galsim
@pytest.mark.parametrize("q", [0.5, 0.7, 0.9])
@pytest.mark.parametrize("beta", [-1.2, -0.7, 0.0, 0.4, 1.1])
def test_cross_validate_reduced_shear_vs_galsim(q: float, beta: float) -> None:
    """Test that our from_reduced_shear matches a galsim.Shear"""
    galsim_shear = galsim.Shear(q=q, beta=beta * galsim.radians)
    ellipse = Ellipse.from_reduced_shear(g1=galsim_shear.g1, g2=galsim_shear.g2, r_det=1.0)
    assert_close(ellipse.b / ellipse.a, galsim_shear.q)
    assert_close(ellipse.theta.to_value(u.rad), galsim_shear.beta.rad)
    assert_close(ellipse.reduced_shear.real, galsim_shear.g1)
    assert_close(ellipse.reduced_shear.imag, galsim_shear.g2)


def test_from_image_unweighted_nonzero_origin() -> None:
    """Test unweighted moments from an asymmetric image with trivial
    moments.
    """
    arr = np.zeros((4, 3))
    arr[0, 0] = 1.0
    arr[1, 2] = 1.0
    arr[3, 1] = 1.0
    img = Image(arr, yx0=(100, 200))
    ell = Ellipse.from_image_unweighted(img)
    assert_close(ell.center.x, 201.0, atol=1e-12)
    assert_close(ell.center.y, 304.0 / 3.0, atol=1e-12)
    assert_close(ell.xx, 2.0 / 3.0, atol=1e-12)
    assert_close(ell.yy, 14.0 / 9.0, atol=1e-12)
    assert_close(ell.xy, 1.0 / 3.0, atol=1e-12)


def test_from_image_unweighted_center_exclusivity() -> None:
    """Test that center is mutually exclusive with x and y in
    from_unweighted_moments.
    """
    img = Image(np.zeros((3, 3)))
    with pytest.raises(TypeError):
        Ellipse.from_image_unweighted(img, center=XY(x=1.0, y=1.0), x=1.0, y=1.0)


_MOMENTS_TEST_ELLIPSES = [
    Ellipse(a=4.0, b=2.0, theta=0.3 * u.rad, center=XY(x=30.0, y=31.0)),
    Ellipse(a=5.0, b=3.0, theta=-0.7 * u.rad, center=XY(x=30.0, y=31.0)),
]


def _gaussian_image(
    ellipse: Ellipse, *, bbox: Box = Box.factory[2:59, -1:58], center: XY[float] = XY(x=30.0, y=31.0)
) -> Image:
    """Render a noiseless Gaussian image whose covariance is ellipse.matrix."""
    x, y = bbox.meshgrid()
    r = np.dstack((x, y))
    pdf = multivariate_normal(mean=[center.x, center.y], cov=ellipse.matrix).pdf(r)
    return Image(pdf, bbox=bbox)


@pytest.mark.parametrize("ellipse", _MOMENTS_TEST_ELLIPSES, ids=lambda e: e.theta)
def test_from_image_unweighted_gaussian(ellipse: Ellipse) -> None:
    """Test that unweighted moments of a Gaussian image recover the input
    ellipse.
    """
    img = _gaussian_image(ellipse)
    measured = Ellipse.from_image_unweighted(img)
    assert_close(measured.xx, ellipse.xx, rtol=1e-4, atol=1e-4)
    assert_close(measured.yy, ellipse.yy, rtol=1e-4, atol=1e-4)
    assert_close(measured.xy, ellipse.xy, rtol=1e-4, atol=1e-4)
    assert_close(measured.center.x, ellipse.center.x, rtol=1e-4, atol=1e-4)
    assert_close(measured.center.y, ellipse.center.y, rtol=1e-4, atol=1e-4)


@skip_no_galsim
@pytest.mark.parametrize("ellipse", _MOMENTS_TEST_ELLIPSES, ids=lambda e: e.theta)
def test_remeasure_adaptive_gaussian(ellipse: Ellipse) -> None:
    """Test that adaptive moments recover the input ellipse from a Gaussian
    image.
    """
    img = _gaussian_image(ellipse)
    measured = ellipse.remeasure_adaptive(img)
    assert_close(measured.xx, ellipse.xx, rtol=1e-4, atol=1e-4)
    assert_close(measured.yy, ellipse.yy, rtol=1e-4, atol=1e-4)
    assert_close(measured.xy, ellipse.xy, rtol=1e-4, atol=1e-4)
    assert_close(measured.center.x, ellipse.center.x, rtol=1e-4, atol=1e-4)
    assert_close(measured.center.y, ellipse.center.y, rtol=1e-4, atol=1e-4)


@skip_no_galsim
@pytest.mark.parametrize("ellipse", _MOMENTS_TEST_ELLIPSES, ids=lambda e: e.theta)
def test_remeasure_adaptive_no_center(ellipse: Ellipse) -> None:
    """Test that center-less adaptive guess still recovers center and
    covariance.
    """
    ellipse = Ellipse(a=ellipse.a, b=ellipse.b, theta=ellipse.theta)
    img = _gaussian_image(ellipse)
    measured = ellipse.remeasure_adaptive(img)
    assert_close(measured.center.x, 30.0, rtol=1e-4, atol=1e-4)
    assert_close(measured.center.y, 31.0, rtol=1e-4, atol=1e-4)
    assert_close(measured.xx, ellipse.xx, rtol=1e-4, atol=1e-4)
    assert_close(measured.yy, ellipse.yy, rtol=1e-4, atol=1e-4)
    assert_close(measured.xy, ellipse.xy, rtol=1e-4, atol=1e-4)

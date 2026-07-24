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
from collections.abc import Callable
from typing import Any

import astropy.units as u
import numpy as np
import pytest

from lsst.images import XY, YX, Box, Image
from lsst.images.describe import DescribableMixin, Report
from lsst.images.fields import (
    BaseField,
    ChebyshevField,
    ProductField,
    SplineField,
    SumField,
    field_from_legacy,
    field_from_legacy_background,
)
from lsst.images.tests import (
    RoundtripFits,
    assert_close,
    assert_images_equal,
    compare_field_to_legacy,
)

try:
    from lsst.afw.image import MaskedImageF
    from lsst.afw.math import BackgroundList as LegacyBackgroundList
    from lsst.afw.math import BackgroundMI
    from lsst.afw.math import Chebyshev1Function2D as LegacyChebyshev1Function2D
    from lsst.afw.math import ChebyshevBoundedField as LegacyChebyshevBoundedField
    from lsst.afw.math import ProductBoundedField as LegacyProductBoundedField
    from lsst.geom import Box2D as LegacyBox2D

    HAVE_LEGACY = True
except ImportError:
    HAVE_LEGACY = False
    type LegacyBackgroundList = Any  # type: ignore[no-redef]
    type LegacyBox2D = Any  # type: ignore[no-redef]
    type LegacyChebyshev1Function2D = Any  # type: ignore[no-redef]
    type LegacyChebyshevBoundedField = Any  # type: ignore[no-redef]
    type LegacyProductBoundedField = Any  # type: ignore[no-redef]


EXTERNAL_DATA_DIR = os.environ.get("TESTDATA_IMAGES_DIR", None)
TEST_BOX = Box.factory[6:32, -7:26]


@pytest.fixture(scope="session")
def legacy_visit_background() -> LegacyBackgroundList:
    """Load and return an `lsst.afw.math.BackgroundList`.

    Skips if TESTDATA_IMAGES_DIR is unset or lsst.afw.math is unavailable.
    """
    if EXTERNAL_DATA_DIR is None:
        pytest.skip("TESTDATA_IMAGES_DIR is not in the environment.")
    if not HAVE_LEGACY:
        pytest.skip("This test requires lsst.afw.math to be importable.")
    filename = os.path.join(EXTERNAL_DATA_DIR, "dp2", "legacy", "visit_image_background.fits")
    return LegacyBackgroundList.readFits(filename)


def _make_test_chebyshev_field() -> ChebyshevField:
    return ChebyshevField(TEST_BOX, np.array([[0.5, -0.25], [0.40, 0.0]]))


def _make_test_spline_field() -> SplineField:
    rng = np.random.default_rng(10)
    return SplineField(
        TEST_BOX,
        rng.standard_normal(size=(6, 7)),
        y=TEST_BOX.y.linspace(6),
        x=TEST_BOX.x.linspace(7),
    )


def _make_test_product_field() -> ProductField:
    return ProductField([_make_test_spline_field(), _make_test_chebyshev_field()])


def _make_test_sum_field() -> SumField:
    return SumField([_make_test_spline_field(), _make_test_chebyshev_field()])


def _make_legacy_chebyshev_field() -> LegacyChebyshevBoundedField:
    if not HAVE_LEGACY:
        pytest.skip("Legacy LSST packages could not be imported.")
    rng = np.random.default_rng(10)
    cheby_coeffs = rng.random((6, 3))
    return LegacyChebyshevBoundedField(TEST_BOX.to_legacy(), cheby_coeffs)


def _make_legacy_product_field() -> LegacyProductBoundedField:
    if not HAVE_LEGACY:
        pytest.skip("Legacy LSST packages could not be imported.")
    rng = np.random.default_rng(11)
    cheby2 = LegacyChebyshevBoundedField(TEST_BOX.to_legacy(), rng.standard_normal(size=(2, 2)))
    return LegacyProductBoundedField([_make_legacy_chebyshev_field(), cheby2])


def check_evaluation_consistency(field: BaseField) -> None:
    """Check that __call__ and render agree."""
    image_1 = field.render()
    p = field.bounds.bbox.meshgrid()
    image_2 = Image(field(x=p.x, y=p.y), bbox=field.bounds.bbox, unit=field.unit)
    assert_images_equal(image_1, image_2)
    scaled_field = field * 2.0
    image_3 = scaled_field.render()
    image_3.array *= 0.5
    assert_images_equal(image_1, image_3)
    image_4 = field.render(Box.factory[9:11, -3:1])
    assert_images_equal(image_4, image_1[image_4.bbox])


@pytest.mark.parametrize(
    "factory",
    [
        _make_test_chebyshev_field,
        _make_test_spline_field,
        _make_test_product_field,
        _make_test_sum_field,
    ],
)
def test_evaluation_consistency(factory: Callable[[], BaseField]) -> None:
    """Test that __call__ and render agree."""
    check_evaluation_consistency(factory())


@pytest.mark.parametrize(
    "factory",
    [
        _make_test_chebyshev_field,
        _make_test_spline_field,
        _make_test_product_field,
        _make_test_sum_field,
    ],
)
def test_units(factory: Callable[[], BaseField]) -> None:
    """Test that fields correctly propagats and check units."""
    field = factory()
    assert field.unit is None
    with_units_1 = field * u.nJy
    assert with_units_1.unit == u.nJy
    assert with_units_1(x=np.array([0.0]), y=np.array([10.0]), quantity=True).unit == u.nJy
    image_1 = with_units_1.render(bbox=Box.factory[10:12, 0:3])
    assert image_1.unit == u.nJy
    assert (with_units_1 * 2.0).unit == u.nJy
    assert (with_units_1 / u.arcsec**2).unit == u.nJy / u.arcsec**2


def test_chebyshev_call_limits() -> None:
    """Test that ChebyshevField evaluates correctly at first order at the
    corners of its box.
    """
    cheby = _make_test_chebyshev_field()
    result = cheby(x=np.array([-7.5, 25.5, 25.5, -7.5]), y=np.array([5.5, 5.5, 31.5, 31.5]))
    assert result[0] == 0.5 + 0.25 - 0.4  # [x=-1, y=-1] after remap
    assert result[1] == 0.5 - 0.25 - 0.4  # [x=1, y=-1] after remap
    assert result[2] == 0.5 - 0.25 + 0.4  # [x=1, y=1] after remap
    assert result[3] == 0.5 + 0.25 + 0.4  # [x=-1, y=1] after remap


def test_chebyshev_attributes() -> None:
    """Test the basic properties of a ChebyshevField."""
    cheby = _make_test_chebyshev_field()
    assert cheby.bounds == TEST_BOX
    assert cheby.unit is None
    assert cheby.x_order == 1
    assert cheby.y_order == 1
    assert cheby.order == 1
    np.testing.assert_array_equal(cheby.coefficients, np.array([[0.5, -0.25], [0.40, 0.0]]))


def test_chebyshev_fit() -> None:
    """Test that ChebyshevField.fit recovers the original coefficients with
    zero residuals.
    """
    cheby = _make_test_chebyshev_field()
    rng = np.random.default_rng(22)
    data_image = cheby.render()
    cheby2 = ChebyshevField.fit(TEST_BOX, data_image.array, order=1, y=TEST_BOX.y.arange, x=TEST_BOX.x.arange)
    assert cheby2.bounds == TEST_BOX
    np.testing.assert_array_almost_equal(cheby2.coefficients, cheby.coefficients)
    # Fit to order 2 in x (will get us extra zero-valued coefficients):
    cheby3 = ChebyshevField.fit(
        TEST_BOX, data_image.array, x_order=2, y_order=1, y=TEST_BOX.y.arange, x=TEST_BOX.x.arange
    )
    assert cheby3.bounds == TEST_BOX
    np.testing.assert_array_almost_equal(
        cheby3.coefficients,
        np.array([[0.5, -0.25, 0.0], [0.40, 0.0, 0.0]], dtype=np.float64),
    )
    # Fit with triangular=False:
    cheby4 = ChebyshevField.fit(
        TEST_BOX, data_image.array, order=1, y=TEST_BOX.y.arange, x=TEST_BOX.x.arange, triangular=False
    )
    assert cheby4.bounds == TEST_BOX
    np.testing.assert_array_almost_equal(cheby4.coefficients, cheby.coefficients)
    # Fit with weights.
    cheby5 = ChebyshevField.fit(
        TEST_BOX,
        data_image.array,
        order=1,
        y=TEST_BOX.y.arange,
        x=TEST_BOX.x.arange,
        weight=rng.uniform(0.8, 1.2, size=data_image.array.shape),
    )
    assert cheby5.bounds == TEST_BOX
    np.testing.assert_array_almost_equal(cheby5.coefficients, cheby.coefficients)
    # Fit to a Quantity.
    cheby6 = ChebyshevField.fit(
        TEST_BOX, data_image.array * u.nJy, order=1, y=TEST_BOX.y.arange, x=TEST_BOX.x.arange
    )
    assert cheby6.bounds == TEST_BOX
    assert cheby6.unit == u.nJy
    np.testing.assert_array_almost_equal(cheby6.coefficients, cheby.coefficients)
    # Fit with units provided separately.
    cheby7 = ChebyshevField.fit(
        TEST_BOX, data_image.array, order=1, y=TEST_BOX.y.arange, x=TEST_BOX.x.arange, unit=u.nJy
    )
    assert cheby7.bounds == TEST_BOX
    assert cheby7.unit == u.nJy
    np.testing.assert_array_almost_equal(cheby7.coefficients, cheby.coefficients)
    # Fit with x and y labeling every data point.
    m = TEST_BOX.meshgrid()
    cheby8 = ChebyshevField.fit(TEST_BOX, data_image.array, order=1, y=m.y, x=m.x)
    assert cheby8.bounds == TEST_BOX
    np.testing.assert_array_almost_equal(cheby8.coefficients, cheby.coefficients)
    # Fit with x and y labeling every data point plus weights.
    cheby9 = ChebyshevField.fit(
        TEST_BOX,
        data_image.array,
        order=1,
        y=m.y,
        x=m.x,
        weight=rng.uniform(0.8, 1.2, size=data_image.array.shape),
    )
    assert cheby9.bounds == TEST_BOX
    np.testing.assert_array_almost_equal(cheby9.coefficients, cheby.coefficients)
    # Fit with one data point replaced by NaN (should be ignored).
    new_data = data_image.array.copy()
    new_data[5, 7] = np.nan
    cheby10 = ChebyshevField.fit(TEST_BOX, new_data, order=1, y=TEST_BOX.y.arange, x=TEST_BOX.x.arange)
    assert cheby10.bounds == TEST_BOX
    np.testing.assert_array_almost_equal(cheby10.coefficients, cheby.coefficients)


def test_spline_knot_evaluation() -> None:
    """Test that SplineField evaluates to its input data at the knot
    positions.
    """
    spline = _make_test_spline_field()
    xv, yv = np.meshgrid(spline.x, spline.y)
    result = spline(x=xv, y=yv)
    np.testing.assert_array_almost_equal(result, spline.data)


def test_product_evaluation() -> None:
    """Test that ProductField.__call__ against direct calls to its operands."""
    product = _make_test_product_field()
    cheby = _make_test_chebyshev_field()
    spline = _make_test_spline_field()
    xv, yv = TEST_BOX.meshgrid(n=3)
    z = product(x=xv, y=yv)
    np.testing.assert_array_equal(z, cheby(x=xv, y=yv) * spline(x=xv, y=yv))


def test_product_units() -> None:
    """Test that ProductField correctly propagates and combines units."""
    cheby = _make_test_chebyshev_field()
    spline = _make_test_spline_field()
    assert ProductField([cheby, spline * u.nJy]).unit == u.nJy
    assert ProductField([cheby * u.nJy, spline]).unit == u.nJy
    assert ProductField([cheby * u.nJy, spline / u.arcsec**2]).unit == u.nJy / u.arcsec**2


def test_sum_evaluation() -> None:
    """Test that SumField.__call__ against direct calls to its operands."""
    sum_ = _make_test_sum_field()
    cheby = _make_test_chebyshev_field()
    spline = _make_test_spline_field()
    xv, yv = TEST_BOX.meshgrid(n=3)
    z = sum_(x=xv, y=yv)
    np.testing.assert_array_equal(z, cheby(x=xv, y=yv) + spline(x=xv, y=yv))


def test_sum_units() -> None:
    """Test that SumField correctly propagates units and raises on incompatible
    units.
    """
    cheby = _make_test_chebyshev_field()
    spline = _make_test_spline_field()
    with pytest.raises(u.UnitConversionError):
        SumField([cheby, spline * u.nJy])
    with pytest.raises(u.UnitConversionError):
        SumField([cheby * u.nJy, spline])
    # Test a SumField where the operands have different but compatible units.
    mixed = SumField([cheby * u.rad, spline * u.deg])
    assert mixed.unit == u.rad
    small_box = Box.factory[10:12, 2:5]
    cheby_render = cheby.render(small_box)
    spline_render = spline.render(small_box)
    mixed_render = mixed.render(small_box)
    assert mixed_render.unit == u.rad
    np.testing.assert_array_almost_equal(
        mixed_render.array, cheby_render.array + (spline_render.array * np.pi / 180.0)
    )
    check_evaluation_consistency(mixed)


def test_chebyshev_roundtrip() -> None:
    """Test converting ChebyshevField from and to legacy, and serializing it
    in between.
    """
    legacy_cheby = _make_legacy_chebyshev_field()
    cheby = field_from_legacy(legacy_cheby)
    assert isinstance(cheby, ChebyshevField)
    compare_field_to_legacy(cheby, legacy_cheby, subimage_bbox=Box.factory[8:12, -3:2])
    with RoundtripFits(cheby) as roundtrip:
        pass
    compare_field_to_legacy(roundtrip.result, legacy_cheby, subimage_bbox=Box.factory[8:12, -3:2])
    compare_field_to_legacy(roundtrip.result, cheby.to_legacy(), subimage_bbox=Box.factory[8:12, -3:2])


def test_product_roundtrip() -> None:
    """Test converting ProductField from and to legacy, and serializing it
    in between.
    """
    legacy_product = _make_legacy_product_field()
    product = field_from_legacy(legacy_product)
    assert isinstance(product, ProductField)
    compare_field_to_legacy(product, legacy_product, subimage_bbox=Box.factory[8:12, -3:2])
    with RoundtripFits(product) as roundtrip:
        pass
    compare_field_to_legacy(roundtrip.result, legacy_product, subimage_bbox=Box.factory[8:12, -3:2])
    compare_field_to_legacy(roundtrip.result, product.to_legacy(), subimage_bbox=Box.factory[8:12, -3:2])


def test_spline_simple() -> None:
    """Test SplineField against BackgroundMI with no missing data."""
    if not HAVE_LEGACY:
        pytest.skip("Legacy LSST packages could not be imported.")
    rng = np.random.default_rng(23)
    bins = MaskedImageF(Box.factory[0:5, 0:6].to_legacy())
    bins.image.array[:, :] = rng.standard_normal(bins.image.array.shape)
    bins.variance.array[::] = 1.0
    legacy_bg = BackgroundMI(TEST_BOX.to_legacy(), bins)
    spline = field_from_legacy_background(legacy_bg)
    render_bbox = TEST_BOX.padded(-3)
    assert_images_equal(
        spline.render(render_bbox),
        Image.from_legacy(
            legacy_bg.getImageF(render_bbox.to_legacy(), legacy_bg.getBackgroundControl().getInterpStyle())
        ),
        rtol=1e-7,
    )


def test_spline_one_nan() -> None:
    """Test SplineField against BackgroundMI with one missing data point."""
    if not HAVE_LEGACY:
        pytest.skip("Legacy LSST packages could not be imported.")
    rng = np.random.default_rng(24)
    bins = MaskedImageF(Box.factory[0:7, 0:6].to_legacy())
    bins.image.array[:, :] = rng.standard_normal(bins.image.array.shape)
    bins.image.array[3, 2] = np.nan
    bins.variance.array[::] = 1.0
    legacy_bg = BackgroundMI(TEST_BOX.to_legacy(), bins)
    spline = field_from_legacy_background(legacy_bg)
    render_bbox = TEST_BOX.padded(-3)
    assert_images_equal(
        spline.render(render_bbox),
        Image.from_legacy(
            legacy_bg.getImageF(
                render_bbox.to_legacy(),
                legacy_bg.getBackgroundControl().getInterpStyle(),
            )
        ),
        rtol=1e-7,
    )


def test_chebyshev1_function2() -> None:
    """Verify ChebyshevField.from_legacy_function2 and to_legacy_function2
    round-trip.
    """
    if not HAVE_LEGACY:
        pytest.skip("Legacy LSST packages could not be imported.")
    rng = np.random.default_rng(25)
    legacy_func2a = LegacyChebyshev1Function2D(4, LegacyBox2D(TEST_BOX.to_legacy()))
    legacy_func2a.setParameters(rng.standard_normal(legacy_func2a.getNParameters()))
    field = ChebyshevField.from_legacy_function2(legacy_func2a)
    legacy_func2b = field.to_legacy_function2()
    assert field.bounds == TEST_BOX
    xy_array = TEST_BOX.meshgrid(4)
    z_array = field(x=xy_array.x, y=xy_array.y)
    for z, x, y in zip(z_array.flat, xy_array.x.flat, xy_array.y.flat):
        assert_close(legacy_func2a(x, y), z)
        assert_close(legacy_func2b(x, y), z)


def test_visit_background(legacy_visit_background: LegacyBackgroundList) -> None:
    """Test field_from_legacy_background against a real visit image
    background.
    """
    bg_field = field_from_legacy_background(legacy_visit_background)
    assert_images_equal(bg_field.render(), Image.from_legacy(legacy_visit_background.getImage()), rtol=1e-6)


# Scalar x/y inside TEST_BOX used by the broadcasting/scalar tests below.
_SCALAR_X = 9.0
_SCALAR_Y = 18.5
_SCALAR_X_INT = 9
_SCALAR_Y_INT = 18


@pytest.mark.parametrize(
    "factory",
    [
        _make_test_chebyshev_field,
        _make_test_spline_field,
        _make_test_product_field,
        _make_test_sum_field,
    ],
)
def test_call_scalar(factory: Callable[[], BaseField]) -> None:
    """Verify that __call__ accepts scalar x/y and returns a scalar float,
    and returns a 0-d Quantity when quantity=True.
    """
    field = factory()
    # quantity=False: Python int and float scalars should return float.
    result_float = field(x=_SCALAR_X, y=_SCALAR_Y)
    assert type(result_float) is float
    result_int = field(x=_SCALAR_X_INT, y=_SCALAR_Y_INT)
    assert type(result_int) is float
    # Values must match the corresponding single-element array call.
    ref_float = field(x=np.array([_SCALAR_X]), y=np.array([_SCALAR_Y]))[0]
    assert result_float == ref_float
    ref_int = field(x=np.array([float(_SCALAR_X_INT)]), y=np.array([float(_SCALAR_Y_INT)]))[0]
    assert result_int == ref_int
    # quantity=True: scalar input should yield a 0-d Quantity.
    field_with_units = field * u.nJy
    result_q = field_with_units(x=_SCALAR_X, y=_SCALAR_Y, quantity=True)
    assert isinstance(result_q, u.Quantity)
    assert result_q.shape == ()
    assert result_q.unit == u.nJy
    ref_q = field_with_units(x=np.array([_SCALAR_X]), y=np.array([_SCALAR_Y]), quantity=True)
    assert result_q.value == ref_q[0].value


@pytest.mark.parametrize(
    "factory",
    [
        _make_test_chebyshev_field,
        _make_test_spline_field,
        _make_test_product_field,
        _make_test_sum_field,
    ],
)
def test_call_array_like_and_integer_input(factory: Callable[[], BaseField]) -> None:
    """Verify that __call__ accepts Python lists and integer arrays, returning
    float64 ndarray results consistent with float64 array input.
    """
    field = factory()
    xv = [_SCALAR_X, _SCALAR_X + 2.0, _SCALAR_X + 4.0]
    yv = [_SCALAR_Y, _SCALAR_Y + 1.0, _SCALAR_Y + 2.0]
    # Python list input should return an ndarray.
    result_list = field(x=xv, y=yv)
    assert isinstance(result_list, np.ndarray)
    ref = field(x=np.array(xv), y=np.array(yv))
    np.testing.assert_array_equal(result_list, ref)
    # Integer array input should not raise and should return float64.
    xi = np.array([_SCALAR_X_INT, _SCALAR_X_INT + 2, _SCALAR_X_INT + 4], dtype=np.int32)
    yi = np.array([_SCALAR_Y_INT, _SCALAR_Y_INT + 1, _SCALAR_Y_INT + 2], dtype=np.int32)
    result_int = field(x=xi, y=yi)
    assert isinstance(result_int, np.ndarray)
    assert result_int.dtype == np.float64
    ref_int = field(x=xi.astype(np.float64), y=yi.astype(np.float64))
    np.testing.assert_array_equal(result_int, ref_int)


@pytest.mark.parametrize(
    "factory",
    [
        _make_test_chebyshev_field,
        _make_test_spline_field,
        _make_test_product_field,
        _make_test_sum_field,
    ],
)
def test_call_broadcast(factory: Callable[[], BaseField]) -> None:
    """Verify that __call__ broadcasts x and y like a NumPy ufunc, in both
    1-D and 2-D cases.
    """
    field = factory()
    xv = np.linspace(_SCALAR_X, _SCALAR_X + 4.0, 5)
    yv = np.linspace(_SCALAR_Y, _SCALAR_Y + 3.0, 4)
    # 1-D broadcast: array x, scalar y.
    result_1d = field(x=xv, y=_SCALAR_Y)
    assert isinstance(result_1d, np.ndarray)
    assert result_1d.shape == xv.shape
    ref_1d = field(x=xv, y=np.full_like(xv, _SCALAR_Y))
    np.testing.assert_array_equal(result_1d, ref_1d)
    # 2-D broadcast: column x (M,1), row y (1,N) -> (M,N).
    x2d = xv[:, np.newaxis]  # shape (5, 1)
    y2d = yv[np.newaxis, :]  # shape (1, 4)
    result_2d = field(x=x2d, y=y2d)
    assert isinstance(result_2d, np.ndarray)
    assert result_2d.shape == (xv.size, yv.size)
    # Values must match the fully expanded meshgrid call.
    xmesh, ymesh = np.meshgrid(xv, yv, indexing="ij")
    ref_2d = field(x=xmesh, y=ymesh)
    np.testing.assert_array_equal(result_2d, ref_2d)


@pytest.mark.parametrize(
    "factory",
    [
        _make_test_chebyshev_field,
        _make_test_spline_field,
        _make_test_product_field,
        _make_test_sum_field,
    ],
)
def test_call_float32_input(factory: Callable[[], BaseField]) -> None:
    """Verify that float32 inputs produce float64 output with correct values,
    without silent precision loss.
    """
    field = factory()
    x32 = np.array([_SCALAR_X, _SCALAR_X + 2.0, _SCALAR_X + 4.0], dtype=np.float32)
    y32 = np.array([_SCALAR_Y, _SCALAR_Y + 1.0, _SCALAR_Y + 2.0], dtype=np.float32)
    result = field(x=x32, y=y32)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    ref = field(x=x32.astype(np.float64), y=y32.astype(np.float64))
    np.testing.assert_array_equal(result, ref)


@pytest.mark.parametrize(
    "factory",
    [
        _make_test_chebyshev_field,
        _make_test_spline_field,
        _make_test_product_field,
        _make_test_sum_field,
    ],
)
def test_call_xy_yx(factory: Callable[[], BaseField]) -> None:
    """Verify that __call__ accepts XY and YX positional arguments, producing
    results identical to the equivalent x=/y= keyword call.
    """
    field = factory()
    # Scalar XY and YX — return type must be float and values must match.
    ref_scalar = field(x=_SCALAR_X, y=_SCALAR_Y)
    assert field(XY(x=_SCALAR_X, y=_SCALAR_Y)) == ref_scalar
    assert field(YX(y=_SCALAR_Y, x=_SCALAR_X)) == ref_scalar
    # Array XY and YX — return type must be ndarray and values must match.
    xv = np.array([_SCALAR_X, _SCALAR_X + 2.0, _SCALAR_X + 4.0])
    yv = np.array([_SCALAR_Y, _SCALAR_Y + 1.0, _SCALAR_Y + 2.0])
    ref_array = field(x=xv, y=yv)
    np.testing.assert_array_equal(field(XY(xv, yv)), ref_array)
    np.testing.assert_array_equal(field(YX(yv, xv)), ref_array)
    # quantity=True still works alongside an XY positional argument.
    field_with_units = field * u.nJy
    ref_q = field_with_units(x=_SCALAR_X, y=_SCALAR_Y, quantity=True)
    result_q = field_with_units(XY(x=_SCALAR_X, y=_SCALAR_Y), quantity=True)
    assert isinstance(result_q, u.Quantity)
    assert result_q.value == ref_q.value
    # Mixing a positional point with explicit x= or y= must raise TypeError.
    with pytest.raises(TypeError):
        field(XY(x=_SCALAR_X, y=_SCALAR_Y), x=_SCALAR_X)
    with pytest.raises(TypeError):
        field(YX(y=_SCALAR_Y, x=_SCALAR_X), y=_SCALAR_Y)


def test_chebyshev_field_describe() -> None:
    """ChebyshevField._describe returns a Report with order and bounds."""
    field = _make_test_chebyshev_field()
    assert isinstance(field, DescribableMixin)
    report = field._describe()
    assert isinstance(report, Report)
    assert report.type_name == "ChebyshevField"
    labels = {f.label for f in report.fields}
    assert "order" in labels
    assert "x_order" in labels
    assert "y_order" in labels
    assert "bounds" in labels


def test_spline_field_describe() -> None:
    """SplineField._describe returns a Report with grid_shape and bounds."""
    field = _make_test_spline_field()
    assert isinstance(field, DescribableMixin)
    report = field._describe()
    assert isinstance(report, Report)
    assert report.type_name == "SplineField"
    labels = {f.label for f in report.fields}
    assert "grid_shape" in labels
    assert "bounds" in labels


def test_product_field_describe() -> None:
    """ProductField._describe returns a Report with operand children."""
    field = _make_test_product_field()
    assert isinstance(field, DescribableMixin)
    report = field._describe()
    assert isinstance(report, Report)
    assert report.type_name == "ProductField"
    assert len(report.children) == 2
    labels = {f.label for f in report.fields}
    assert "n_operands" in labels


def test_sum_field_describe() -> None:
    """SumField._describe returns a Report with operand children."""
    field = _make_test_sum_field()
    assert isinstance(field, DescribableMixin)
    report = field._describe()
    assert isinstance(report, Report)
    assert report.type_name == "SumField"
    assert len(report.children) == 2
    labels = {f.label for f in report.fields}
    assert "n_operands" in labels

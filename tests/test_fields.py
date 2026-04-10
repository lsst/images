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
import unittest

import astropy.units as u
import numpy as np

from lsst.images import Box, Image
from lsst.images.fields import (
    BaseField,
    ChebyshevField,
    ProductField,
    SplineField,
    field_from_legacy,
    field_from_legacy_background,
)
from lsst.images.tests import (
    RoundtripFits,
    assert_images_equal,
    compare_field_to_legacy,
)

try:
    from lsst.afw.math import BackgroundList as LegacyBackgroundList
    from lsst.afw.math import ChebyshevBoundedField as LegacyChebyshevBoundedField
    from lsst.afw.math import ProductBoundedField as LegacyProductBoundedField
except ImportError:
    HAVE_LEGACY = False
else:
    HAVE_LEGACY = True

DATA_DIR = os.environ.get("TESTDATA_IMAGES_DIR", None)


class FieldTestCase(unittest.TestCase):
    """Tests for the Field classes that do not require legacy code."""

    def setUp(self) -> None:
        self.rng = np.random.default_rng(10)
        self.box = Box.factory[6:32, -7:26]
        self.cheby = ChebyshevField(self.box, np.array([[0.5, -0.25], [0.40, 0.0]]))
        self.spline = SplineField(
            self.box,
            self.rng.standard_normal(size=(6, 7)),
            y=self.box.y.linspace(6),
            x=self.box.x.linspace(7),
        )
        self.product = ProductField([self.spline, self.cheby])

    def test_chebyshev_call_limits(self) -> None:
        """Test that ChebyshevField.__call__ evaluates correctly at low order
        at the corners of its box.

        Testing at first order is enough to catch (y, x) swap errors, and is
        easy because ``T_0(x) = 1`` and ``T_1(x) = x``.  Higher-order
        evaluation is covered in the legacy comparison tests.
        """
        result = self.cheby(x=np.array([-7.5, 25.5, 25.5, -7.5]), y=np.array([5.5, 5.5, 31.5, 31.5]))
        self.assertEqual(result[0], 0.5 + 0.25 - 0.4)  # [x=-1, y=-1] after remap
        self.assertEqual(result[1], 0.5 - 0.25 - 0.4)  # [x=1, y=-1] after remap
        self.assertEqual(result[2], 0.5 - 0.25 + 0.4)  # [x=1, y=1] after remap
        self.assertEqual(result[3], 0.5 + 0.25 + 0.4)  # [x=-1, y=1] after remap

    def test_chebyshev_attributes(self) -> None:
        """Test the basic properties of a Chebyshev field."""
        self.assertEqual(self.cheby.bounds, self.box)
        self.assertIsNone(self.cheby.unit)
        self.assertEqual(self.cheby.x_order, 1)
        self.assertEqual(self.cheby.y_order, 1)
        self.assertEqual(self.cheby.order, 1)
        np.testing.assert_array_equal(self.cheby.coefficients, np.array([[0.5, -0.25], [0.40, 0.0]]))

    def test_chebyshev_fit(self) -> None:
        """Test that we can fit a ChebyshevField to gridded data that should
        have zero residuals.
        """
        data_image = self.cheby.render()
        cheby2 = ChebyshevField.fit(
            self.box, data_image.array, order=1, y=self.box.y.arange, x=self.box.x.arange
        )
        self.assertEqual(cheby2.bounds, self.box)
        np.testing.assert_array_almost_equal(cheby2.coefficients, self.cheby.coefficients)
        # Repeat the experiment in a few different ways that don't affect the
        # result (at least in this scenario where we have an exact fit).
        # Fit to order 2 in x (will get us extra zero-valued coefficients):
        cheby3 = ChebyshevField.fit(
            self.box, data_image.array, x_order=2, y_order=1, y=self.box.y.arange, x=self.box.x.arange
        )
        self.assertEqual(cheby3.bounds, self.box)
        np.testing.assert_array_almost_equal(
            cheby3.coefficients,
            np.array([[0.5, -0.25, 0.0], [0.40, 0.0, 0.0]], dtype=np.float64),
        )
        # Fit with triangular=False (would allow the (1, 1) term to be nonzero,
        # but it will still be zero here):
        cheby4 = ChebyshevField.fit(
            self.box, data_image.array, order=1, y=self.box.y.arange, x=self.box.x.arange, triangular=False
        )
        self.assertEqual(cheby4.bounds, self.box)
        np.testing.assert_array_almost_equal(cheby4.coefficients, self.cheby.coefficients)
        # Fit with weights.
        cheby5 = ChebyshevField.fit(
            self.box,
            data_image.array,
            order=1,
            y=self.box.y.arange,
            x=self.box.x.arange,
            weight=self.rng.uniform(0.8, 1.2, size=data_image.array.shape),
        )
        self.assertEqual(cheby5.bounds, self.box)
        np.testing.assert_array_almost_equal(cheby5.coefficients, self.cheby.coefficients)
        # Fit to a Quantity.
        cheby6 = ChebyshevField.fit(
            self.box, data_image.array * u.nJy, order=1, y=self.box.y.arange, x=self.box.x.arange
        )
        self.assertEqual(cheby6.bounds, self.box)
        self.assertEqual(cheby6.unit, u.nJy)
        np.testing.assert_array_almost_equal(cheby6.coefficients, self.cheby.coefficients)
        # Fit with units provided separately.
        cheby7 = ChebyshevField.fit(
            self.box, data_image.array, order=1, y=self.box.y.arange, x=self.box.x.arange, unit=u.nJy
        )
        self.assertEqual(cheby7.bounds, self.box)
        self.assertEqual(cheby7.unit, u.nJy)
        np.testing.assert_array_almost_equal(cheby7.coefficients, self.cheby.coefficients)
        # Fit with x and y labeling every data point, not the grid.
        m = self.box.meshgrid()
        cheby8 = ChebyshevField.fit(self.box, data_image.array, order=1, y=m.y, x=m.x)
        self.assertEqual(cheby8.bounds, self.box)
        np.testing.assert_array_almost_equal(cheby8.coefficients, self.cheby.coefficients)
        # Fit with x and y labeling every data point, not the grid, as well as
        # weights.
        cheby9 = ChebyshevField.fit(
            self.box,
            data_image.array,
            order=1,
            y=m.y,
            x=m.x,
            weight=self.rng.uniform(0.8, 1.2, size=data_image.array.shape),
        )
        self.assertEqual(cheby9.bounds, self.box)
        np.testing.assert_array_almost_equal(cheby9.coefficients, self.cheby.coefficients)
        # Fit with one data points replaced by NaN, which should be ignored
        # (the fit is still overconstrained).
        new_data = data_image.array.copy()
        new_data[5, 7] = np.nan
        cheby10 = ChebyshevField.fit(self.box, new_data, order=1, y=self.box.y.arange, x=self.box.x.arange)
        self.assertEqual(cheby10.bounds, self.box)
        np.testing.assert_array_almost_equal(cheby10.coefficients, self.cheby.coefficients)

    def test_chebyshev_evaluation_consistency(self) -> None:
        self.check_evaluation_consistency(self.cheby)

    def test_chebyshev_units(self) -> None:
        self.check_units(self.cheby)

    def test_spline_knot_evaluation(self) -> None:
        """Test that SplineField.__call__ evaluates to the input data points
        at their positions.
        """
        xv, yv = np.meshgrid(self.spline.x, self.spline.y)
        result = self.spline(x=xv, y=yv)
        np.testing.assert_array_almost_equal(result, self.spline.data)

    def test_spline_evaluation_consistency(self) -> None:
        self.check_evaluation_consistency(self.spline)

    def test_spline_units(self) -> None:
        self.check_units(self.spline)

    def test_product_evaluation(self) -> None:
        """Test ProductField.__call__ against direct calls to its operands."""
        xv, yv = self.box.meshgrid(n=3)
        z = self.product(x=xv, y=yv)
        np.testing.assert_array_equal(z, self.cheby(x=xv, y=yv) * self.spline(x=xv, y=yv))

    def test_product_evaluation_consistency(self) -> None:
        self.check_evaluation_consistency(self.product)

    def test_product_units(self) -> None:
        self.check_units(self.product)
        self.assertEqual(ProductField([self.cheby, self.spline * u.nJy]).unit, u.nJy)
        self.assertEqual(ProductField([self.cheby * u.nJy, self.spline]).unit, u.nJy)
        self.assertEqual(
            ProductField([self.cheby * u.nJy, self.spline / u.arcsec**2]).unit, u.nJy / u.arcsec**2
        )

    def check_evaluation_consistency(self, field: BaseField) -> None:
        """Test that `BaseField.__call__` and `BaseField.render` agree on a
        concrete field.
        """
        image_1 = field.render()
        p = field.bounds.bbox.meshgrid()
        image_2 = Image(field(x=p.x, y=p.y), bbox=field.bounds.bbox, unit=field.unit)
        assert_images_equal(self, image_1, image_2)
        scaled_field = field * 2.0
        image_3 = scaled_field.render()
        image_3.array *= 0.5
        assert_images_equal(self, image_1, image_3)
        image_4 = field.render(Box.factory[9:11, -3:1])
        assert_images_equal(self, image_4, image_1[image_4.bbox])

    def check_units(self, field: BaseField) -> None:
        """Test that the methods of a `BaseField` implementation correctly
        propogate and check units.
        """
        self.assertIsNone(field.unit)
        with_units_1 = field * u.nJy
        self.assertEqual(with_units_1.unit, u.nJy)
        self.assertEqual(with_units_1(x=np.array([0.0]), y=np.array([10.0]), quantity=True).unit, u.nJy)
        image_1 = with_units_1.render(bbox=Box.factory[10:12, 0:3])
        self.assertEqual(image_1.unit, u.nJy)
        self.assertEqual((with_units_1 * 2.0).unit, u.nJy)
        self.assertEqual((with_units_1 / u.arcsec**2).unit, u.nJy / u.arcsec**2)


@unittest.skipUnless(HAVE_LEGACY, "This test requires lsst.afw.math to be importable.")
class FieldLegacyTestCase(unittest.TestCase):
    """Test the Field classes against legacy implementations.

    This includes many tests for correct evaluation, since the legacy types
    serve as our reference implementation.
    """

    def setUp(self) -> None:
        self.rng = np.random.default_rng(10)
        self.box = Box.factory[6:32, -7:26]
        # This Chebyshev coefficient matrix is unusual in that it has nonzero
        # entries for the whole matrix, not just the p + q < order triangle,
        # and it has different orders for y and x.
        # But we want to make sure those round-trip, too.
        self.cheby_coeffs = self.rng.random((6, 3))
        self.legacy_cheby = LegacyChebyshevBoundedField(self.box.to_legacy(), self.cheby_coeffs)
        cheby2 = LegacyChebyshevBoundedField(self.box.to_legacy(), self.rng.standard_normal(size=(2, 2)))
        self.legacy_product = LegacyProductBoundedField([self.legacy_cheby, cheby2])

    def test_chebyshev_roundtrip(self) -> None:
        """Test converting ChebyshevField from and to legacy, and serializing
        it in between.
        """
        cheby = field_from_legacy(self.legacy_cheby)
        assert isinstance(cheby, ChebyshevField)
        compare_field_to_legacy(self, cheby, self.legacy_cheby, subimage_bbox=Box.factory[8:12, -3:2])
        with RoundtripFits(self, cheby) as roundtrip:
            pass
        compare_field_to_legacy(
            self, roundtrip.result, self.legacy_cheby, subimage_bbox=Box.factory[8:12, -3:2]
        )
        compare_field_to_legacy(
            self, roundtrip.result, cheby.to_legacy(), subimage_bbox=Box.factory[8:12, -3:2]
        )

    def test_product_roundtrip(self) -> None:
        """Test converting ProductField from and to legacy, and serializing
        it in between.
        """
        product = field_from_legacy(self.legacy_product)
        assert isinstance(product, ProductField)
        compare_field_to_legacy(self, product, self.legacy_product, subimage_bbox=Box.factory[8:12, -3:2])
        with RoundtripFits(self, product) as roundtrip:
            pass
        compare_field_to_legacy(
            self, roundtrip.result, self.legacy_product, subimage_bbox=Box.factory[8:12, -3:2]
        )
        compare_field_to_legacy(
            self, roundtrip.result, product.to_legacy(), subimage_bbox=Box.factory[8:12, -3:2]
        )

    def test_spline_simple(self) -> None:
        """Test SplineField against `lsst.afw.math.BackgroundMI`, when there
        is no missing data.
        """
        from lsst.afw.image import MaskedImageF
        from lsst.afw.math import BackgroundMI

        bins = MaskedImageF(Box.factory[0:5, 0:6].to_legacy())
        bins.image.array[:, :] = self.rng.standard_normal(bins.image.array.shape)
        bins.variance.array[::] = 1.0
        legacy_bg = BackgroundMI(self.box.to_legacy(), bins)
        spline = field_from_legacy_background(legacy_bg)
        render_bbox = self.box.padded(-3)
        assert_images_equal(
            self,
            spline.render(render_bbox),
            Image.from_legacy(
                legacy_bg.getImageF(
                    render_bbox.to_legacy(), legacy_bg.getBackgroundControl().getInterpStyle()
                )
            ),
            rtol=1e-7,
        )

    def test_spline_one_nan(self) -> None:
        """Test SplineField against `lsst.afw.math.BackgroundMI`, when there
        is missing data.
        """
        from lsst.afw.image import MaskedImageF
        from lsst.afw.math import BackgroundMI

        bins = MaskedImageF(Box.factory[0:7, 0:6].to_legacy())
        bins.image.array[:, :] = self.rng.standard_normal(bins.image.array.shape)
        bins.image.array[3, 2] = np.nan
        bins.variance.array[::] = 1.0
        legacy_bg = BackgroundMI(self.box.to_legacy(), bins)
        spline = field_from_legacy_background(legacy_bg)
        render_bbox = self.box.padded(-3)
        assert_images_equal(
            self,
            spline.render(render_bbox),
            Image.from_legacy(
                legacy_bg.getImageF(
                    render_bbox.to_legacy(),
                    legacy_bg.getBackgroundControl().getInterpStyle(),
                )
            ),
            rtol=1e-7,
        )


@unittest.skipIf(DATA_DIR is None, "This test requires the TESTDATA_IMAGES_DIR envvar to be set.")
@unittest.skipUnless(HAVE_LEGACY, "This test requires lsst.afw.math to be importable.")
class FieldLegacyDataTestCase(unittest.TestCase):
    """Tests for using Field classes using legacy datasets."""

    def test_chebyshev_background(self) -> None:
        assert DATA_DIR is not None, "Guaranteed by decorator."
        filename = os.path.join(DATA_DIR, "dp2", "legacy", "visit_image_background.fits")
        legacy_bg_list = LegacyBackgroundList.readFits(filename)
        legacy_bg_0 = legacy_bg_list[0][0]
        cheby_fit = ChebyshevField.from_legacy_background(legacy_bg_0)
        assert_images_equal(self, cheby_fit.render(), Image.from_legacy(legacy_bg_0.getImageF()), rtol=1e-7)


if __name__ == "__main__":
    unittest.main()

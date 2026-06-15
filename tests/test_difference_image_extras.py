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

from lsst.images.convolution_kernels import ConvolutionKernel, ImageBasisConvolutionKernel
from lsst.images.tests import assert_close

try:
    from lsst.afw.image import ImageD as LegacyImageD
    from lsst.afw.math import Kernel as LegacyKernel
except ImportError:
    pass

EXTERNAL_DATA_DIR = os.environ.get("TESTDATA_IMAGES_DIR", None)


@unittest.skipUnless(EXTERNAL_DATA_DIR is not None, "TESTDATA_IMAGES_DIR is not in the environment.")
class DifferenceImageExtraLegacyTestCase(unittest.TestCase):
    """Tests for DifferenceImage components that were stored externally in
    separate files in the legacy system.

    Requires legacy code.  Mosts tests for DifferenceImage are in
    test_visit_image.py, since most of DifferenceImage is inherited from
    VisitImage.
    """

    @classmethod
    def setUpClass(cls) -> None:
        assert EXTERNAL_DATA_DIR is not None, "Guaranteed by decorator."
        cls.kernel_filename = os.path.join(EXTERNAL_DATA_DIR, "dp2", "legacy", "difference_kernel.fits")
        try:
            from lsst.afw.math import Kernel

            cls.legacy_kernel = Kernel.readFits(cls.kernel_filename)
        except ImportError:
            raise unittest.SkipTest(
                "afw not available; cannot read legacy difference image components"
            ) from None

    def test_difference_kernel(self) -> None:
        """Test that we can convert to and from legacy difference kernels."""
        kernel = ImageBasisConvolutionKernel.from_legacy(self.legacy_kernel)
        self.compare_kernel_to_legacy(kernel, self.legacy_kernel)
        legacy_kernel_2 = kernel.to_legacy()
        self.compare_kernel_to_legacy(kernel, legacy_kernel_2)

    def compare_kernel_to_legacy(self, kernel: ConvolutionKernel, legacy_kernel: LegacyKernel) -> None:
        xy_array = kernel.bounds.bbox.meshgrid(3)
        legacy_im = LegacyImageD(kernel.kernel_bbox.to_legacy())
        for x, y in zip(xy_array.x.flat, xy_array.y.flat):
            x = round(x)
            y = round(y)
            im = kernel.compute_kernel_image(x=x, y=y)
            legacy_im.array[...] = 0.0
            legacy_kernel.computeImage(legacy_im, doNormalize=False, x=x, y=y)
            assert_close(self, im.array, legacy_im.array, rtol=1e-15, atol=1e-15)


if __name__ == "__main__":
    unittest.main()

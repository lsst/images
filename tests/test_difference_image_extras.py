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

import math
import os
import unittest

from lsst.images import Box, DetectorFrame, DifferenceImage, DifferenceImageTemplateInfo
from lsst.images.convolution_kernels import ConvolutionKernel, ImageBasisConvolutionKernel
from lsst.images.tests import (
    DP2_TEMPLATE_COADD_DATASETS,
    DP2_VISIT_DETECTOR_DATA_ID,
    RoundtripFits,
    assert_close,
)

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
        cls.template_filename = os.path.join(EXTERNAL_DATA_DIR, "dp2", "legacy", "template_detector.fits")
        cls.exposure_filename = os.path.join(EXTERNAL_DATA_DIR, "dp2", "legacy", "difference_image.fits")
        try:
            from lsst.afw.image import ExposureFitsReader
            from lsst.afw.math import Kernel

            cls.legacy_kernel = Kernel.readFits(cls.kernel_filename)
            template_reader = ExposureFitsReader(cls.template_filename)
            cls.legacy_template_metadata = template_reader.readMetadata()
            cls.legacy_template_psf = template_reader.readPsf()
            cls.legacy_exposure = ExposureFitsReader(cls.exposure_filename).read()
        except ImportError:
            raise unittest.SkipTest(
                "afw not available; cannot read legacy difference image or components"
            ) from None
        cls.detector_frame = DetectorFrame(
            **DP2_VISIT_DETECTOR_DATA_ID, bbox=Box.from_legacy(cls.legacy_exposure.getDetector().getBBox())
        )

    def test_roundtrip(self) -> None:
        """Test round-tripping the difference image through FITs with the
        extra components attached.
        """
        difference_image = DifferenceImage.from_legacy(self.legacy_exposure)
        difference_image.kernel = ImageBasisConvolutionKernel.from_legacy(self.legacy_kernel)
        difference_image.templates = DifferenceImageTemplateInfo.from_legacy(
            self.detector_frame,
            self.legacy_template_psf,
            self.legacy_template_metadata,
            DP2_TEMPLATE_COADD_DATASETS,
        )
        with RoundtripFits(difference_image, storage_class="DifferenceImage") as roundtrip:
            self.compare_kernel_to_legacy(roundtrip.get("kernel"), self.legacy_kernel)
        self.compare_kernel_to_legacy(roundtrip.result.kernel, self.legacy_kernel)
        self.sanity_check_template_info(roundtrip.result.templates)

    def test_difference_kernel(self) -> None:
        """Test that we can convert to and from legacy difference kernels."""
        kernel = ImageBasisConvolutionKernel.from_legacy(self.legacy_kernel)
        self.compare_kernel_to_legacy(kernel, self.legacy_kernel)
        legacy_kernel_2 = kernel.to_legacy()
        self.compare_kernel_to_legacy(kernel, legacy_kernel_2)

    @staticmethod
    def compare_kernel_to_legacy(kernel: ConvolutionKernel, legacy_kernel: LegacyKernel) -> None:
        xy_array = kernel.bounds.bbox.meshgrid(3)
        legacy_im = LegacyImageD(kernel.kernel_bbox.to_legacy())
        for x, y in zip(xy_array.x.flat, xy_array.y.flat):
            x = round(x)
            y = round(y)
            im = kernel.compute_kernel_image(x=x, y=y)
            legacy_im.array[...] = 0.0
            legacy_kernel.computeImage(legacy_im, doNormalize=False, x=x, y=y)
            assert_close(im.array, legacy_im.array, rtol=1e-15, atol=1e-15)

    def test_template_info(self) -> None:
        """Test extracting template information from various legacy
        template_detector components.
        """
        template_info = DifferenceImageTemplateInfo.from_legacy(
            self.detector_frame,
            self.legacy_template_psf,
            self.legacy_template_metadata,
            DP2_TEMPLATE_COADD_DATASETS,
        )
        self.sanity_check_template_info(template_info)

    def sanity_check_template_info(self, template_info: list[DifferenceImageTemplateInfo]) -> None:
        self.assertEqual(len(template_info), 9)
        self.assertCountEqual([info.dataset_id for info in template_info], DP2_TEMPLATE_COADD_DATASETS.keys())
        self.assertCountEqual(
            [
                {"skymap": info.skymap, "tract": info.tract, "patch": info.patch, "band": "r"}
                for info in template_info
            ],
            DP2_TEMPLATE_COADD_DATASETS.values(),
        )
        self.assertFalse(any(info.psf_shape_flag for info in template_info))
        self.assertFalse(any(math.isnan(info.psf_shape_xx) for info in template_info))
        self.assertFalse(any(math.isnan(info.psf_shape_yy) for info in template_info))
        self.assertFalse(any(math.isnan(info.psf_shape_xy) for info in template_info))
        self.assertTrue(all(self.detector_frame.bbox.contains(info.bounds.bbox) for info in template_info))
        # The template bounds will overlap somewhat because patches overlap,
        # so their total area should be a bit more than the total area of the
        # detector.
        self.assertLess(sum(info.bounds.area for info in template_info), 1.5 * self.detector_frame.bbox.area)


if __name__ == "__main__":
    unittest.main()

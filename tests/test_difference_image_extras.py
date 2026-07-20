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

import dataclasses
import math
import os
from typing import Any

import pytest

from lsst.images import Box, DetectorFrame, DifferenceImage, DifferenceImageTemplateInfo
from lsst.images.convolution_kernels import ConvolutionKernel, ImageBasisConvolutionKernel
from lsst.images.tests import (
    DP2_TEMPLATE_COADD_DATASETS,
    DP2_VISIT_DETECTOR_DATA_ID,
    RoundtripFits,
    assert_close,
)

try:
    from lsst.afw.image import Exposure as LegacyExposure
    from lsst.afw.image import ImageD as LegacyImageD
    from lsst.afw.math import Kernel as LegacyKernel
    from lsst.daf.base import PropertyList as LegacyPropertyList
    from lsst.meas.algorithms import CoaddPsf as LegacyCoaddPsf
except ImportError:
    type LegacyExposure = Any  # type: ignore[no-redef]
    type LegacyKernel = Any  # type: ignore[no-redef]
    type LegacyPropertyList = Any  # type: ignore[no-redef]
    type LegacyCoaddPsf = Any  # type: ignore[no-redef]


EXTERNAL_DATA_DIR = os.environ.get("TESTDATA_IMAGES_DIR", None)


@dataclasses.dataclass
class _LegacyTestData:
    kernel: LegacyKernel
    template_metadata: LegacyPropertyList
    template_psf: LegacyCoaddPsf
    exposure: LegacyExposure
    detector_frame: DetectorFrame


@pytest.fixture(scope="session")
def legacy_test_data() -> _LegacyTestData:
    """Return a struct of legacy test objects loaded from EXTERNAL_DATA_DIR.

    Skips if TESTDATA_IMAGES_DIR is unset or afw is unavailable.
    """
    if EXTERNAL_DATA_DIR is None:
        pytest.skip("TESTDATA_IMAGES_DIR is not in the environment.")
    kernel_filename = os.path.join(EXTERNAL_DATA_DIR, "dp2", "legacy", "difference_kernel.fits")
    template_filename = os.path.join(EXTERNAL_DATA_DIR, "dp2", "legacy", "template_detector.fits")
    exposure_filename = os.path.join(EXTERNAL_DATA_DIR, "dp2", "legacy", "difference_image.fits")
    try:
        from lsst.afw.image import ExposureFitsReader
    except ImportError:
        pytest.skip("afw not available; cannot read legacy difference image or components")
    kernel = LegacyKernel.readFits(kernel_filename)
    template_reader = ExposureFitsReader(template_filename)
    template_metadata = template_reader.readMetadata()
    template_psf = template_reader.readPsf()
    exposure = ExposureFitsReader(exposure_filename).read()
    detector_frame = DetectorFrame(
        **DP2_VISIT_DETECTOR_DATA_ID, bbox=Box.from_legacy(exposure.getDetector().getBBox())
    )
    return _LegacyTestData(
        kernel=kernel,
        template_metadata=template_metadata,
        template_psf=template_psf,
        exposure=exposure,
        detector_frame=detector_frame,
    )


def compare_kernel_to_legacy(kernel: ConvolutionKernel, legacy_kernel: LegacyKernel) -> None:
    """Assert that a ConvolutionKernel matches a legacy Kernel at sampled
    points.
    """
    xy_array = kernel.bounds.bbox.meshgrid(3)
    legacy_im = LegacyImageD(kernel.kernel_bbox.to_legacy())
    for x, y in zip(xy_array.x.flat, xy_array.y.flat):
        x = round(x)
        y = round(y)
        im = kernel.compute_kernel_image(x=x, y=y)
        legacy_im.array[...] = 0.0
        legacy_kernel.computeImage(legacy_im, doNormalize=False, x=x, y=y)
        assert_close(im.array, legacy_im.array, rtol=1e-15, atol=1e-15)


def _sanity_check_template_info(
    template_info: list[DifferenceImageTemplateInfo], detector_frame: DetectorFrame
) -> None:
    """Check that a list of DifferenceImageTemplateInfo looks plausible."""
    assert len(template_info) == 9
    assert {info.dataset_id for info in template_info} == set(DP2_TEMPLATE_COADD_DATASETS.keys())
    assert {
        frozenset({"skymap": info.skymap, "tract": info.tract, "patch": info.patch, "band": "r"}.items())
        for info in template_info
    } == {frozenset(v.items()) for v in DP2_TEMPLATE_COADD_DATASETS.values()}
    assert not any(info.psf_shape_flag for info in template_info)
    assert not any(math.isnan(info.psf_shape_xx) for info in template_info)
    assert not any(math.isnan(info.psf_shape_yy) for info in template_info)
    assert not any(math.isnan(info.psf_shape_xy) for info in template_info)
    assert all(detector_frame.bbox.contains(info.bounds.bbox) for info in template_info)
    # Patches overlap, so total area is a bit more than detector area.
    assert sum(info.bounds.area for info in template_info) < 1.5 * detector_frame.bbox.area


def _make_difference_image(legacy_test_data: _LegacyTestData) -> DifferenceImage:
    """Return a DifferenceImage with kernel and template components
    attached.
    """
    difference_image = DifferenceImage.from_legacy(legacy_test_data.exposure)
    difference_image.kernel = ImageBasisConvolutionKernel.from_legacy(legacy_test_data.kernel)
    difference_image.templates = DifferenceImageTemplateInfo.from_legacy(
        legacy_test_data.detector_frame,
        legacy_test_data.template_psf,
        legacy_test_data.template_metadata,
        DP2_TEMPLATE_COADD_DATASETS,
    )
    return difference_image


def test_roundtrip(legacy_test_data: _LegacyTestData) -> None:
    """Test round-tripping a DifferenceImage with extra components through
    FITS.
    """
    difference_image = _make_difference_image(legacy_test_data)
    with RoundtripFits(difference_image, storage_class="DifferenceImage") as roundtrip:
        pass
    compare_kernel_to_legacy(roundtrip.result.kernel, legacy_test_data.kernel)
    _sanity_check_template_info(roundtrip.result.templates, legacy_test_data.detector_frame)


def test_kernel_component_read(legacy_test_data: _LegacyTestData) -> None:
    """Verify the kernel component of a DifferenceImage can be read on its
    own.

    Requires a butler; skips when `lsst.daf.butler` is absent.  Butler-free
    assertions live in `test_roundtrip`.
    """
    difference_image = _make_difference_image(legacy_test_data)
    with RoundtripFits(difference_image, storage_class="DifferenceImage") as roundtrip:
        compare_kernel_to_legacy(roundtrip.get("kernel"), legacy_test_data.kernel)


def test_difference_kernel(legacy_test_data: _LegacyTestData) -> None:
    """Test converting a legacy difference kernel to and from the new type."""
    kernel = ImageBasisConvolutionKernel.from_legacy(legacy_test_data.kernel)
    compare_kernel_to_legacy(kernel, legacy_test_data.kernel)
    legacy_kernel_2 = kernel.to_legacy()
    compare_kernel_to_legacy(kernel, legacy_kernel_2)


def test_template_info(legacy_test_data: _LegacyTestData) -> None:
    """Test extracting template information from legacy template_detector
    components.
    """
    template_info = DifferenceImageTemplateInfo.from_legacy(
        legacy_test_data.detector_frame,
        legacy_test_data.template_psf,
        legacy_test_data.template_metadata,
        DP2_TEMPLATE_COADD_DATASETS,
    )
    _sanity_check_template_info(template_info, legacy_test_data.detector_frame)

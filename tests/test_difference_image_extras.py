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
import logging
import math
import os
from typing import Any

import astropy.units as u
import numpy as np
import pytest
from astro_metadata_translator import ObservationInfo

from lsst.images import (
    Box,
    DetectorFrame,
    DifferenceImage,
    DifferenceImageTemplateInfo,
    Image,
    MaskPlane,
    MaskSchema,
)
from lsst.images.cameras import Detector
from lsst.images.convolution_kernels import ConvolutionKernel, ImageBasisConvolutionKernel
from lsst.images.psfs import GaussianPointSpreadFunction
from lsst.images.serialization import read_archive
from lsst.images.tests import (
    DP2_TEMPLATE_COADD_DATASETS,
    DP2_VISIT_DETECTOR_DATA_ID,
    RoundtripFits,
    assert_values_equal,
    make_random_sky_projection,
    reset_afw_mask_planes,  # noqa: F401
)

try:
    from lsst.afw.image import Exposure as LegacyExposure
    from lsst.afw.image import ImageD as LegacyImageD
    from lsst.afw.math import Kernel as LegacyKernel
    from lsst.afw.table import ExposureCatalog as LegacyExposureCatalog
    from lsst.daf.base import PropertyList as LegacyPropertyList
    from lsst.geom import Extent2I as LegacyExtent2I
    from lsst.meas.algorithms import CoaddPsf as LegacyCoaddPsf
except ImportError:
    type LegacyExposure = Any  # type: ignore[no-redef]
    type LegacyImageD = Any  # type: ignore[no-redef]
    type LegacyKernel = Any  # type: ignore[no-redef]
    type LegacyPropertyList = Any  # type: ignore[no-redef]
    type LegacyCoaddPsf = Any  # type: ignore[no-redef]
    type LegacyExposureCatalog = Any  # type: ignore[no-redef]
    type LegacyExtent2I = Any  # type: ignore[no-redef]


EXTERNAL_DATA_DIR = os.environ.get("TESTDATA_IMAGES_DIR", None)


@dataclasses.dataclass
class _LegacyTestData:
    kernel: LegacyKernel
    template_metadata: LegacyPropertyList
    template_psf: LegacyCoaddPsf
    exposure: LegacyExposure
    detector_frame: DetectorFrame


@pytest.fixture
def legacy_test_data(reset_afw_mask_planes: None) -> _LegacyTestData:  # noqa: F811
    """Return a struct of legacy test objects loaded from EXTERNAL_DATA_DIR.

    Skips if TESTDATA_IMAGES_DIR is unset or afw is unavailable.
    """
    # reset_afw_mask_planes will have already skipped if afw is not available.
    from lsst.afw.image import ExposureFitsReader

    if EXTERNAL_DATA_DIR is None:
        pytest.skip("TESTDATA_IMAGES_DIR is not in the environment.")
    kernel_filename = os.path.join(EXTERNAL_DATA_DIR, "dp2", "legacy", "difference_kernel.fits")
    template_filename = os.path.join(EXTERNAL_DATA_DIR, "dp2", "legacy", "template_detector.fits")
    exposure_filename = os.path.join(EXTERNAL_DATA_DIR, "dp2", "legacy", "difference_image.fits")
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
        assert_values_equal(im.array, legacy_im.array, rtol=1e-15, atol=1e-15)


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


def test_template_info_no_overlap_is_skipped(
    legacy_test_data: _LegacyTestData, caplog: pytest.LogCaptureFixture
) -> None:
    """Test that template coadd whose patch does not overlap the detector is
    skipped in DifferenceImageTemplateInfo.from_legacy, instead of raising
    NoOverlapError.
    """
    psf = legacy_test_data.template_psf
    n_components = psf.getComponentCount()
    assert n_components > 1

    # Rebuild the CoaddPsf from its per-component getters, shifting component
    # 0's patch bbox far from the detector so the bbox intersection in
    # from_legacy is empty (NoOverlapError). Keep the tract/patch columns so
    # from_legacy can still resolve its butler info.
    schema = LegacyExposureCatalog.Table.makeMinimalSchema()
    schema.addField("weight", type="D")
    schema.addField("tract", type="I")
    schema.addField("patch", type="I")
    catalog = LegacyExposureCatalog(schema)
    shift = LegacyExtent2I(50000, 50000)
    for n in range(n_components):
        record = catalog.addNew()
        record.setId(psf.getId(n))
        record.setWcs(psf.getWcs(n))
        record.setPsf(psf.getPsf(n))
        record.setValidPolygon(psf.getValidPolygon(n))
        bbox = psf.getBBox(n)
        if n == 0:
            bbox = bbox.shiftedBy(shift)
        record.setBBox(bbox)
        record.set("weight", psf.getWeight(n))
        record.set("tract", psf.getTract(n))
        record.set("patch", psf.getPatch(n))
    shifted_psf = LegacyCoaddPsf(catalog, psf.getCoaddWcs(), psf.getAveragePosition())

    skipped_tract = psf.getTract(0)
    skipped_patch = psf.getPatch(0)
    with caplog.at_level(logging.ERROR):
        template_info = DifferenceImageTemplateInfo.from_legacy(
            legacy_test_data.detector_frame,
            shifted_psf,
            legacy_test_data.template_metadata,
            DP2_TEMPLATE_COADD_DATASETS,
            log=logging.getLogger("test_template_info_no_overlap"),
        )
    assert len(template_info) == n_components - 1
    assert all((info.tract, info.patch) != (skipped_tract, skipped_patch) for info in template_info)
    assert not any(info.psf_shape_flag for info in template_info)
    assert all(legacy_test_data.detector_frame.bbox.contains(info.bounds.bbox) for info in template_info)
    assert any(
        f"No overlap with tract={skipped_tract}, patch={skipped_patch}" in record.message
        for record in caplog.records
    )


LOCAL_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def test_difference_image_repr_str_pinned() -> None:
    """Pin the exact str and repr output of a DifferenceImage."""
    rng = np.random.default_rng(500)
    det_frame = DetectorFrame(instrument="Inst", visit=1234, detector=1, bbox=Box.factory[1:4096, 1:4096])
    mask_schema = MaskSchema([MaskPlane("M1", "D1")])
    obs_info = ObservationInfo(instrument="LSSTCam", detector_num=4, physical_filter="r1")
    detector = read_archive(os.path.join(LOCAL_DATA_DIR, "detector.json"), Detector)
    image = Image(42, shape=(1024, 1024), unit=u.nJy)
    sky_projection = make_random_sky_projection(rng, det_frame, det_frame.bbox)
    di = DifferenceImage(
        image,
        psf=GaussianPointSpreadFunction(2.5, stamp_size=33, bounds=Box.factory[-10:10, -12:13]),
        mask_schema=mask_schema,
        sky_projection=sky_projection,
        detector=detector,
        obs_info=obs_info,
        band="r",
    )
    assert str(di) == "DifferenceImage(Image([y=0:1024, x=0:1024], int64), ['M1'])"
    assert repr(di) == (
        "DifferenceImage(Image(..., bbox=Box(y=Interval(start=0, stop=1024), x=Interval(start=0, stop=1024)),"
        " dtype=dtype('int64')), mask_schema=MaskSchema([MaskPlane(name='M1', description='D1')],"
        " dtype=dtype('uint8')))"
    )

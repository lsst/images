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
import os
import pickle
from typing import Any

import astropy.units as u
import numpy as np
import pytest

from lsst.images import YX, Box, get_legacy_non_cell_coadd_mask_planes
from lsst.images.cells import CellCoadd
from lsst.images.tests import (
    DP2_COADD_DATA_ID,
    RoundtripFits,
    assert_cell_coadds_equal,
    assert_masked_images_equal,
    compare_cell_coadd_to_legacy_lazy,
    compare_field_to_legacy,
    compare_masked_image_to_legacy,
    compare_psf_to_legacy,
    compare_sky_projection_to_legacy_wcs,
    reset_afw_mask_planes,  # noqa: F401
)

EXTERNAL_DATA_DIR = os.environ.get("TESTDATA_IMAGES_DIR", None)


@dataclasses.dataclass
class _LazyCoaddTestData:
    """A struct holding test data loaded from EXTERNAL_DATA_DIR.

    The legacy fixture is a DP2 ``template_coadd``: a patch-level coadd
    ``lsst.afw.image.Exposure`` with a spatially varying PSF and aperture
    corrections, but no cell structure.
    """

    filename: str
    tract_info: Any
    legacy_exposure: Any
    cell_coadd: CellCoadd


@pytest.fixture
def legacy_test_data(reset_afw_mask_planes: None) -> _LazyCoaddTestData:  # noqa: F811
    """Return a struct with a legacy lazy coadd and its CellCoadd conversion.

    The legacy mask must be loaded after the afw mask-plane registry is
    reset (so its header's plane definitions, such as CLIPPED, get
    registered in the reset registry), which means the fixture can not be
    shared between tests via a module scope.  The conversion itself is
    cheap, since it only covers the cells overlapping the legacy coadd.

    Skips if ``TESTDATA_IMAGES_DIR`` is not set or if ``lsst.afw`` cannot be
    imported.
    """
    if EXTERNAL_DATA_DIR is None:
        pytest.skip("TESTDATA_IMAGES_DIR is not in the environment.")
    filename = os.path.join(EXTERNAL_DATA_DIR, "dp2", "legacy", "template_coadd.fits")
    try:
        from lsst.afw.image import ExposureFitsReader
    except ImportError:
        pytest.skip("lsst.afw.image is not available; cannot read legacy exposures")
    legacy_exposure = ExposureFitsReader(filename).read()
    with open(os.path.join(EXTERNAL_DATA_DIR, "dp2", "legacy", "skyMap.pickle"), "rb") as stream:
        skymap = pickle.load(stream)
    tract_info = skymap[DP2_COADD_DATA_ID["tract"]]
    cell_coadd = CellCoadd.from_legacy_lazy_coadd(
        legacy_exposure,
        instrument="LSSTCam",
        tract_info=tract_info,
        patch=DP2_COADD_DATA_ID["patch"],
    )
    return _LazyCoaddTestData(
        filename=filename,
        tract_info=tract_info,
        legacy_exposure=legacy_exposure,
        cell_coadd=cell_coadd,
    )


def test_from_legacy_lazy_coadd(legacy_test_data: _LazyCoaddTestData) -> None:
    """Test constructing a CellCoadd by converting a legacy lazy
    (non-cell) coadd.
    """
    compare_cell_coadd_to_legacy_lazy(
        legacy_test_data.cell_coadd,
        legacy_test_data.legacy_exposure,
        tract_bbox=Box.from_legacy(legacy_test_data.tract_info.getBBox()),
        **DP2_COADD_DATA_ID,
    )
    assert legacy_test_data.cell_coadd.bbox == Box.from_legacy(legacy_test_data.legacy_exposure.getBBox())
    assert legacy_test_data.cell_coadd.unit == u.nJy
    assert legacy_test_data.cell_coadd.skymap == DP2_COADD_DATA_ID["skymap"]
    assert legacy_test_data.cell_coadd.tract == DP2_COADD_DATA_ID["tract"]
    assert legacy_test_data.cell_coadd.band == DP2_COADD_DATA_ID["band"]
    patch_info = legacy_test_data.tract_info[DP2_COADD_DATA_ID["patch"]]
    assert legacy_test_data.cell_coadd.patch.id == DP2_COADD_DATA_ID["patch"]
    assert legacy_test_data.cell_coadd.patch.index == YX(y=patch_info.getIndex().y, x=patch_info.getIndex().x)
    assert legacy_test_data.cell_coadd.patch.inner_bbox == Box.from_legacy(patch_info.getInnerBBox())
    assert legacy_test_data.cell_coadd.mask_fractions == {}
    assert list(legacy_test_data.cell_coadd.noise_realizations) == []
    assert not legacy_test_data.cell_coadd.backgrounds.keys()
    assert legacy_test_data.cell_coadd.backgrounds.subtracted is None


def test_roundtrip(legacy_test_data: _LazyCoaddTestData) -> None:
    """Test that a CellCoadd converted from a legacy lazy coadd round-trips
    through FITS.
    """
    cell_coadd = legacy_test_data.cell_coadd
    with RoundtripFits(cell_coadd, "CellCoadd") as roundtrip:
        subbox = Box.factory[
            cell_coadd.bbox.y.start + 3 : cell_coadd.bbox.y.stop - 4,
            cell_coadd.bbox.x.start + 5 : cell_coadd.bbox.x.stop - 2,
        ]
        subimage = roundtrip.get(bbox=subbox)
        assert_masked_images_equal(subimage, cell_coadd[subbox], expect_view=False)
        alternates = {
            k: roundtrip.get(k) for k in ["sky_projection", "psf", "aperture_corrections", "provenance"]
        }
        compare_cell_coadd_to_legacy_lazy(
            roundtrip.result,
            legacy_test_data.legacy_exposure,
            tract_bbox=Box.from_legacy(legacy_test_data.tract_info.getBBox()),
            alternates=alternates,
            **DP2_COADD_DATA_ID,
        )
    assert_cell_coadds_equal(roundtrip.result, cell_coadd, expect_view=False)


def test_to_legacy(legacy_test_data: _LazyCoaddTestData) -> None:
    """Test converting a coadd that was itself converted from a legacy lazy
    coadd back into a legacy Exposure.
    """
    cell_coadd = legacy_test_data.cell_coadd
    plane_map = get_legacy_non_cell_coadd_mask_planes()
    legacy_exposure = cell_coadd.to_legacy(plane_map=plane_map)
    assert legacy_exposure.getFilter().bandLabel == cell_coadd.band
    assert Box.from_legacy(legacy_exposure.getBBox()) == cell_coadd.bbox
    compare_masked_image_to_legacy(
        cell_coadd, legacy_exposure.maskedImage, plane_map=plane_map, expect_view=True
    )
    # The legacy PSF here is a StitchedPsf made from the very same per-cell
    # stamps, so this comparison is exact at any point again.
    compare_psf_to_legacy(cell_coadd.psf, legacy_exposure.getPsf())
    compare_sky_projection_to_legacy_wcs(
        cell_coadd.sky_projection,
        legacy_exposure.getWcs(),
        cell_coadd.sky_projection.pixel_frame,
        subimage_bbox=cell_coadd.bbox,
        is_fits=True,
    )


def test_compare_lazy_coadd_image_atol(legacy_test_data: _LazyCoaddTestData) -> None:
    """Test that image_atol lets compare_cell_coadd_to_legacy_lazy accept a
    small offset in the image pixels (while image_atol=0.0 rejects it).
    """
    perturbed = legacy_test_data.legacy_exposure.clone()
    perturbed.getMaskedImage().getImage().array += 0.5
    kwargs = _lazy_coadd_kwargs(legacy_test_data)
    with pytest.raises(AssertionError):
        compare_cell_coadd_to_legacy_lazy(legacy_test_data.cell_coadd, perturbed, **kwargs)
    compare_cell_coadd_to_legacy_lazy(legacy_test_data.cell_coadd, perturbed, image_atol=1.0, **kwargs)


def test_compare_lazy_coadd_variance_atol(legacy_test_data: _LazyCoaddTestData) -> None:
    """Test that variance_atol lets compare_cell_coadd_to_legacy_lazy accept
    a small offset in the variance pixels (while variance_atol=0.0 rejects
    it), with the image still compared exactly.
    """
    perturbed = legacy_test_data.legacy_exposure.clone()
    perturbed.getMaskedImage().getVariance().array += 0.5
    kwargs = _lazy_coadd_kwargs(legacy_test_data)
    with pytest.raises(AssertionError):
        compare_cell_coadd_to_legacy_lazy(legacy_test_data.cell_coadd, perturbed, **kwargs)
    compare_cell_coadd_to_legacy_lazy(legacy_test_data.cell_coadd, perturbed, variance_atol=1.0, **kwargs)


def test_compare_field_to_legacy_rtol(legacy_test_data: _LazyCoaddTestData) -> None:
    """Test that compare_field_to_legacy's default rtol catches a
    1e-4-scale eval-point mismatch, while a looser rtol accepts it.
    """
    ap_corr_map = legacy_test_data.legacy_exposure.info.getApCorrMap()
    name = sorted(ap_corr_map.keys())[0]
    field = legacy_test_data.cell_coadd.aperture_corrections[name]
    legacy_field = ap_corr_map[name]
    perturbed = field * (1.0 + 1e-4)
    eval_points = field.bounds.bbox.meshgrid(n=3).map(np.ravel)
    with pytest.raises(AssertionError):
        compare_field_to_legacy(
            perturbed,
            legacy_field,
            field.bounds.bbox,
            eval_points=eval_points,
            expect_subset_bounds=True,
        )
    compare_field_to_legacy(
        perturbed,
        legacy_field,
        field.bounds.bbox,
        eval_points=eval_points,
        expect_subset_bounds=True,
        rtol=1e-2,
    )

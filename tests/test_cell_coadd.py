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

import numpy as np
import pytest

from lsst.images import YX, Box, Interval, MaskPlane, get_legacy_deep_coadd_mask_planes
from lsst.images.cells import CellCoadd, CellIJ
from lsst.images.fits import FitsCompressionOptions
from lsst.images.serialization import read_archive
from lsst.images.tests import (
    DP2_COADD_DATA_ID,
    DP2_COADD_MISSING_CELL,
    RoundtripFits,
    RoundtripJson,
    RoundtripNdf,
    assert_cell_coadds_equal,
    assert_images_equal,
    assert_masked_images_equal,
    assert_psfs_equal,
    check_bounds_contains_broadcasting,
    compare_cell_coadd_to_legacy,
    compare_masked_image_to_legacy,
    compare_psf_to_legacy,
    compare_sky_projection_to_legacy_wcs,
)

try:
    import h5py  # noqa: F401

    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False

try:
    import lsst.afw.image  # noqa: F401
    from lsst.cell_coadds import MultipleCellCoadd as LegacyMultipleCellCoadd

    HAVE_LEGACY = True
except ImportError:
    HAVE_LEGACY = False
    type LegacyMultipleCellCoadd = Any  # type: ignore[no-redef]

EXTERNAL_DATA_DIR = os.environ.get("TESTDATA_IMAGES_DIR", None)
LOCAL_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

skip_no_h5py = pytest.mark.skipif(not HAVE_H5PY, reason="h5py is not installed")
skip_no_legacy = pytest.mark.skipif(not HAVE_LEGACY, reason="lsst.afw (etc) could not be imported.")


@dataclasses.dataclass
class _LegacyTestData:
    """A struct holding test data loaded from EXTERNAL_DATA_DIR."""

    filename: str
    tract_bbox: Box
    legacy_cell_coadd: LegacyMultipleCellCoadd
    cell_coadd: CellCoadd
    plane_map: dict[str, MaskPlane] = dataclasses.field(default_factory=get_legacy_deep_coadd_mask_planes)

    def make_psf_points(self, bbox: Box | None = None) -> YX[np.ndarray]:
        """Create random PSF sample points within the given bbox."""
        if bbox is None:
            bbox = self.cell_coadd.bbox
        rng = np.random.default_rng(44)
        xc, yc = np.meshgrid(
            np.arange(
                bbox.x.start + self.cell_coadd.grid.cell_shape.x * 0.5,
                bbox.x.stop,
                self.cell_coadd.grid.cell_shape.x,
            ),
            np.arange(
                bbox.y.start + self.cell_coadd.grid.cell_shape.y * 0.5,
                bbox.y.stop,
                self.cell_coadd.grid.cell_shape.y,
            ),
        )
        return YX(
            y=yc.ravel() + rng.uniform(-0.4, 0.4, size=yc.size),
            x=xc.ravel() + rng.uniform(-0.4, 0.4, size=xc.size),
        )


@pytest.fixture(scope="session")
def legacy_test_data() -> _LegacyTestData:
    """Return a struct of CellCoadd loaded from legacy test data.

    Skips if ``TESTDATA_IMAGES_DIR`` is not set or if ``lsst.cell_coadds``
    cannot be imported.
    """
    if EXTERNAL_DATA_DIR is None:
        pytest.skip("TESTDATA_IMAGES_DIR is not in the environment.")
    try:
        from lsst.cell_coadds import MultipleCellCoadd
    except ImportError:
        pytest.skip("lsst.cell_coadds could not be imported.")
    filename = os.path.join(EXTERNAL_DATA_DIR, "dp2", "legacy", "deep_coadd_cell_predetection.fits")
    plane_map = get_legacy_deep_coadd_mask_planes()
    legacy_cell_coadd = MultipleCellCoadd.read_fits(filename)
    with open(os.path.join(EXTERNAL_DATA_DIR, "dp2", "legacy", "skyMap.pickle"), "rb") as stream:
        skymap = pickle.load(stream)
    cell_coadd = CellCoadd.from_legacy(
        legacy_cell_coadd,
        plane_map=plane_map,
        tract_info=skymap[DP2_COADD_DATA_ID["tract"]],
    )
    return _LegacyTestData(
        filename=filename,
        tract_bbox=Box.from_legacy(skymap[DP2_COADD_DATA_ID["tract"]].getBBox()),
        legacy_cell_coadd=legacy_cell_coadd,
        cell_coadd=cell_coadd,
    )


@pytest.fixture
def minified_cell_coadd() -> CellCoadd:
    """Return a tiny CellCoadd from JSON data stored in this package."""
    path = os.path.join(LOCAL_DATA_DIR, "schema_v1", "legacy", "cell_coadd.json")
    return read_archive(path, CellCoadd)


def make_subbox(full_bbox: Box) -> Box:
    """Make a box that's useful for nontrivial subimage tests.

    This box only overlaps (but does not fully cover) the middle 2 (of 4)
    cells in y, while covering exactly the last column of cells in x. It does
    not cover the missing cell.
    """
    return Box.factory[
        full_bbox.y.start + 252 : full_bbox.y.stop - 175,
        full_bbox.x.stop - 150 : full_bbox.x.stop,
    ]


def test_from_legacy(legacy_test_data: _LegacyTestData) -> None:
    """Test constructing a CellCoadd by converting a legacy
    ``MultipleCellCoadd``.
    """
    assert legacy_test_data.cell_coadd.bounds.missing == {CellIJ(**DP2_COADD_MISSING_CELL)}
    assert legacy_test_data.cell_coadd.bbox == Box.factory[12900:13500, 9600:10050]
    compare_cell_coadd_to_legacy(
        legacy_test_data.cell_coadd,
        legacy_test_data.legacy_cell_coadd,
        tract_bbox=legacy_test_data.tract_bbox,
        plane_map=legacy_test_data.plane_map,
        psf_points=legacy_test_data.make_psf_points(),
    )


def test_roundtrip(legacy_test_data: _LegacyTestData) -> None:
    """Test that a CellCoadd roundtrips through FITS."""
    with RoundtripFits(legacy_test_data.cell_coadd, "CellCoadd") as roundtrip:
        # Check a subimage read (no component arg — does not trigger a skip).
        subbox = Box.factory[
            legacy_test_data.cell_coadd.bbox.y.start + 252 : legacy_test_data.cell_coadd.bbox.y.stop - 175,
            legacy_test_data.cell_coadd.bbox.x.stop - 150 : legacy_test_data.cell_coadd.bbox.x.stop,
        ]
        subimage = roundtrip.get(bbox=subbox)
        assert_masked_images_equal(subimage, legacy_test_data.cell_coadd[subbox], expect_view=False)
        with roundtrip.inspect() as fits:
            for extname in ["IMAGE", "MASK", "VARIANCE", "MASK_FRACTIONS/REJECTED"] + [
                f"NOISE_REALIZATIONS/{n}" for n in range(len(legacy_test_data.cell_coadd.noise_realizations))
            ]:
                assert fits[extname].header["ZTILE1"] == legacy_test_data.cell_coadd.grid.cell_shape.x
                assert fits[extname].header["ZTILE2"] == legacy_test_data.cell_coadd.grid.cell_shape.y
    # Fixture self-consistency: bbox and missing-cell set are as expected.
    assert legacy_test_data.cell_coadd.bounds.missing == {CellIJ(**DP2_COADD_MISSING_CELL)}
    assert legacy_test_data.cell_coadd.bbox == Box.factory[12900:13500, 9600:10050]
    # Full round-trip fidelity.
    assert_cell_coadds_equal(roundtrip.result, legacy_test_data.cell_coadd, expect_view=False)
    compare_cell_coadd_to_legacy(
        roundtrip.result,
        legacy_test_data.legacy_cell_coadd,
        tract_bbox=legacy_test_data.tract_bbox,
        plane_map=legacy_test_data.plane_map,
        psf_points=legacy_test_data.make_psf_points(),
    )


def test_roundtrip_components(legacy_test_data: _LegacyTestData) -> None:
    """Test component and subimage reads.

    This test will be skipped if `lsst.daf.butler` is not available instead of
    falling back to non-butler I/O, which is why we don't want to merge it
    with `test_roundtrip`.
    """
    with RoundtripFits(legacy_test_data.cell_coadd, "CellCoadd") as roundtrip:
        subbox = make_subbox(legacy_test_data.cell_coadd.bbox)
        subpsf = roundtrip.get("psf", bbox=subbox)
        assert subpsf.bounds.bbox == Box(
            y=Interval.factory[
                legacy_test_data.cell_coadd.bbox.y.start + 150 : legacy_test_data.cell_coadd.bbox.y.stop - 150
            ],
            x=subbox.x,
        )
        assert_psfs_equal(
            subpsf,
            legacy_test_data.cell_coadd.psf,
            points=legacy_test_data.make_psf_points(subbox),
        )
        assert roundtrip.get("bbox") == legacy_test_data.cell_coadd.bbox
        alternates = {
            k: roundtrip.get(k)
            for k in [
                "sky_projection",
                "image",
                "mask",
                "variance",
                "masked_image",
                "psf",
                "aperture_corrections",
                "provenance",
                "backgrounds",
                "bbox",
            ]
        }
        # Read all the components at once.
        all_components = roundtrip.get("components")
        assert set(all_components) == set(alternates) - {"masked_image"}
        assert all_components["bbox"] == alternates["bbox"]
        assert_psfs_equal(all_components["psf"], alternates["psf"])
        assert_images_equal(all_components["image"], alternates["image"])

        backgrounds = roundtrip.get("backgrounds")
        assert backgrounds.keys() == set()
        assert backgrounds.subtracted is None

        compare_cell_coadd_to_legacy(
            roundtrip.result,
            legacy_test_data.legacy_cell_coadd,
            tract_bbox=legacy_test_data.tract_bbox,
            plane_map=legacy_test_data.plane_map,
            alternates=alternates,
            psf_points=legacy_test_data.make_psf_points(),
        )


def test_fits_compression(legacy_test_data: _LegacyTestData) -> None:
    """Test lossy FITS compression produces the expected headers."""
    with RoundtripFits(
        legacy_test_data.cell_coadd,
        storage_class="CellCoadd",
        recipe="lossy16",
        compression_options={
            "image": FitsCompressionOptions.LOSSY,
            "variance": FitsCompressionOptions.LOSSY,
        },
    ) as roundtrip:
        with roundtrip.inspect() as fits:
            for extname in ["IMAGE", "MASK", "VARIANCE", "MASK_FRACTIONS/REJECTED"] + [
                f"NOISE_REALIZATIONS/{n}" for n in range(len(legacy_test_data.cell_coadd.noise_realizations))
            ]:
                assert fits[extname].header["ZTILE1"] == legacy_test_data.cell_coadd.grid.cell_shape.x
                assert fits[extname].header["ZTILE2"] == legacy_test_data.cell_coadd.grid.cell_shape.y
                if extname == "MASK" or extname.startswith("MASK_FRACTIONS"):
                    assert fits[extname].header["ZCMPTYPE"] == "GZIP_2"
                else:
                    assert fits[extname].header["ZCMPTYPE"] == "RICE_1"
                    assert fits[extname].header["ZQUANTIZ"] == "SUBTRACTIVE_DITHER_2"


def test_json_roundtrip(legacy_test_data: _LegacyTestData) -> None:
    """Verify a CellCoadd round-trips correctly through the JSON archive."""
    with RoundtripJson(legacy_test_data.cell_coadd) as roundtrip:
        pass
    assert_cell_coadds_equal(roundtrip.result, legacy_test_data.cell_coadd, expect_view=False)


def test_to_legacy_cell_coadd(legacy_test_data: _LegacyTestData) -> None:
    """Verify converting a CellCoadd back into a legacy MultipleCellCoadd."""
    legacy_cell_coadd = legacy_test_data.cell_coadd.to_legacy_cell_coadd()
    compare_cell_coadd_to_legacy(
        legacy_test_data.cell_coadd,
        legacy_cell_coadd,
        tract_bbox=legacy_test_data.tract_bbox,
        plane_map=legacy_test_data.plane_map,
        psf_points=legacy_test_data.make_psf_points(),
    )
    with pytest.raises(
        ValueError, match="MultipleCellCoadd requires its bounding box to lie on the cell grid."
    ):
        legacy_test_data.cell_coadd[make_subbox(legacy_test_data.cell_coadd.bbox)].to_legacy_cell_coadd()


@skip_no_legacy
def test_to_legacy(legacy_test_data: _LegacyTestData) -> None:
    """Test converting a CellCoadd back into a legacy Exposure."""
    legacy_exposure = legacy_test_data.cell_coadd.to_legacy()
    assert legacy_exposure.getFilter().bandLabel == legacy_test_data.cell_coadd.band
    assert Box.from_legacy(legacy_exposure.getBBox()) == legacy_test_data.cell_coadd.bbox
    compare_masked_image_to_legacy(
        legacy_test_data.cell_coadd,
        legacy_exposure.maskedImage,
        plane_map=legacy_test_data.plane_map,
        expect_view=True,
    )
    compare_psf_to_legacy(
        legacy_test_data.cell_coadd.psf,
        legacy_exposure.getPsf(),
        points=legacy_test_data.make_psf_points(),
        expect_legacy_raise_on_out_of_bounds=True,
    )
    compare_sky_projection_to_legacy_wcs(
        legacy_test_data.cell_coadd.sky_projection,
        legacy_exposure.getWcs(),
        legacy_test_data.cell_coadd.sky_projection.pixel_frame,
        subimage_bbox=legacy_test_data.cell_coadd.bbox,
        is_fits=True,
    )
    subbox = make_subbox(legacy_test_data.cell_coadd.bbox)
    compare_masked_image_to_legacy(
        legacy_test_data.cell_coadd[subbox],
        legacy_test_data.cell_coadd[subbox].to_legacy().maskedImage,
        plane_map=legacy_test_data.plane_map,
        expect_view=True,
    )


@skip_no_h5py
def test_ndf_roundtrip(legacy_test_data: _LegacyTestData) -> None:
    """Test that CellCoadd round-trips through NDF."""
    with RoundtripNdf(legacy_test_data.cell_coadd, "CellCoadd") as roundtrip:
        assert_cell_coadds_equal(roundtrip.result, legacy_test_data.cell_coadd, expect_view=False)


def test_cell_grid_bounds_contains_broadcasting(minified_cell_coadd: CellCoadd) -> None:
    """Test that CellGridBounds.contains broadcasts like a numpy ufunc."""
    assert minified_cell_coadd.bounds.missing, "fixture should retain a missing cell"
    check_bounds_contains_broadcasting(minified_cell_coadd.bounds)


def test_intersection_bounds_contains_broadcasting(minified_cell_coadd: CellCoadd) -> None:
    """Test that IntersectionBounds.contains broadcasts like a numpy ufunc."""
    # Clip the CellGridBounds with a Box offset by 1 pixel on each side so it
    # does not snap to any cell boundary, forcing a lazy IntersectionBounds.
    bounds = minified_cell_coadd.bounds
    clip = Box.factory[
        bounds.bbox.y.start + 1 : bounds.bbox.y.stop - 1,
        bounds.bbox.x.start + 1 : bounds.bbox.x.stop - 1,
    ]
    check_bounds_contains_broadcasting(bounds.intersection(clip))

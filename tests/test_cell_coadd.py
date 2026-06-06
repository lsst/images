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
import pickle
import unittest
from typing import Any

import numpy as np

from lsst.images import YX, Box, Interval, get_legacy_deep_coadd_mask_planes
from lsst.images.cells import CellCoadd, CellIJ
from lsst.images.fits import FitsCompressionOptions
from lsst.images.tests import (
    DP2_COADD_DATA_ID,
    DP2_COADD_MISSING_CELL,
    RoundtripFits,
    RoundtripJson,
    RoundtripNdf,
    RoundtripZarr,
    assert_cell_coadds_equal,
    assert_masked_images_equal,
    assert_psfs_equal,
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
    import zarr

    from lsst.images.zarr._store import open_store_for_read

    HAVE_ZARR = True
except ImportError:
    HAVE_ZARR = False

DATA_DIR = os.environ.get("TESTDATA_IMAGES_DIR", None)


@unittest.skipUnless(DATA_DIR is not None, "TESTDATA_IMAGES_DIR is not in the environment.")
class CellCoaddTestCase(unittest.TestCase):
    """Tests for the CellCoadd class and its many component classes."""

    @classmethod
    def setUpClass(cls) -> None:
        assert DATA_DIR is not None, "Guaranteed by decorator."
        cls.filename = os.path.join(DATA_DIR, "dp2", "legacy", "deep_coadd_cell_predetection.fits")
        cls.plane_map = get_legacy_deep_coadd_mask_planes()
        cls.missing_cell = CellIJ(**DP2_COADD_MISSING_CELL)
        try:
            from lsst.cell_coadds import MultipleCellCoadd

            cls.legacy_cell_coadd = MultipleCellCoadd.read_fits(cls.filename)
        except ImportError:
            raise unittest.SkipTest("lsst.cell_coadds could not be imported.") from None
        with open(os.path.join(DATA_DIR, "dp2", "legacy", "skyMap.pickle"), "rb") as stream:
            cls.skymap = pickle.load(stream)
        cls.cell_coadd = CellCoadd.from_legacy(
            cls.legacy_cell_coadd,
            plane_map=cls.plane_map,
            tract_info=cls.skymap[DP2_COADD_DATA_ID["tract"]],
        )

    def make_psf_points(self, bbox: Box) -> YX[np.ndarray]:
        """Make arrays of points to test PSFs at, given a bbox that is assumed
        to be snapped to the cell_coadd grid.
        """
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
            y=yc.ravel() + self.rng.uniform(-0.4, 0.4, size=yc.size),
            x=xc.ravel() + self.rng.uniform(-0.4, 0.4, size=xc.size),
        )

    def setUp(self) -> None:
        self.rng = np.random.default_rng(44)
        self.psf_points = self.make_psf_points(self.cell_coadd.bbox)

    def test_from_legacy(self) -> None:
        """Test constructing a CellCoadd by converting a legacy
        lsst.cell_coadds.MultipleCellCoadd.
        """
        self.assertEqual(self.cell_coadd.bounds.missing, {self.missing_cell})
        self.assertEqual(self.cell_coadd.bbox, Box.factory[12900:13500, 9600:10050])
        compare_cell_coadd_to_legacy(
            self,
            self.cell_coadd,
            self.legacy_cell_coadd,
            tract_bbox=Box.from_legacy(self.skymap[DP2_COADD_DATA_ID["tract"]].getBBox()),
            plane_map=self.plane_map,
            psf_points=self.psf_points,
        )

    def test_roundtrip(self) -> None:
        """Test serializing a CellCoadd and reading it back in, including
        subimage and component reads.
        """
        with RoundtripFits(self, self.cell_coadd, "CellCoadd") as roundtrip:
            # Check a subimage read.  The subbox only overlaps (but does not
            # fully cover) the middle 2 (of 4) cells in y, while covering
            # exactly the last column of cells in x.  It does not cover the
            # missing cell.
            subbox = Box.factory[
                self.cell_coadd.bbox.y.start + 252 : self.cell_coadd.bbox.y.stop - 175,
                self.cell_coadd.bbox.x.stop - 150 : self.cell_coadd.bbox.x.stop,
            ]
            subimage = roundtrip.get(bbox=subbox)
            assert_masked_images_equal(self, subimage, self.cell_coadd[subbox], expect_view=False)
            alternates: dict[str, Any] = {}
            with self.subTest():
                subpsf = roundtrip.get("psf", bbox=subbox)
                self.assertEqual(
                    subpsf.bounds.bbox,
                    Box(
                        y=Interval.factory[
                            self.cell_coadd.bbox.y.start + 150 : self.cell_coadd.bbox.y.stop - 150
                        ],
                        x=subbox.x,
                    ),
                )
                assert_psfs_equal(self, subpsf, self.cell_coadd.psf, points=self.make_psf_points(subbox))
                self.assertEqual(roundtrip.get("bbox"), self.cell_coadd.bbox)
                alternates = {
                    k: roundtrip.get(k)
                    for k in [
                        "sky_projection",
                        "image",
                        "mask",
                        "variance",
                        "psf",
                        "aperture_corrections",
                        "provenance",
                    ]
                }
                with self.subTest():
                    backgrounds = roundtrip.get("backgrounds")
                    self.assertEqual(backgrounds.keys(), set())
                    self.assertIsNone(backgrounds.subtracted)
            with roundtrip.inspect() as fits:
                for extname in ["IMAGE", "MASK", "VARIANCE", "MASK_FRACTIONS/REJECTED"] + [
                    f"NOISE_REALIZATIONS/{n}" for n in range(len(self.cell_coadd.noise_realizations))
                ]:
                    self.assertEqual(fits[extname].header["ZTILE1"], self.cell_coadd.grid.cell_shape.x)
                    self.assertEqual(fits[extname].header["ZTILE2"], self.cell_coadd.grid.cell_shape.y)
        # Fixture self-consistency: bbox and missing-cell set are what setUp
        # claims they are.
        self.assertEqual(self.cell_coadd.bounds.missing, {self.missing_cell})
        self.assertEqual(self.cell_coadd.bbox, Box.factory[12900:13500, 9600:10050])
        # Full round-trip fidelity, including background contents.
        assert_cell_coadds_equal(self, roundtrip.result, self.cell_coadd, expect_view=False)
        compare_cell_coadd_to_legacy(
            self,
            roundtrip.result,
            self.legacy_cell_coadd,
            tract_bbox=Box.from_legacy(self.skymap[DP2_COADD_DATA_ID["tract"]].getBBox()),
            plane_map=self.plane_map,
            alternates=alternates,
            psf_points=self.psf_points,
        )

    def test_fits_compression(self) -> None:
        """Test writing with quantized FITS compression."""
        with RoundtripFits(
            self,
            self.cell_coadd,
            storage_class="CellCoadd",
            recipe="lossy16",
            compression_options={
                "image": FitsCompressionOptions.LOSSY,
                "variance": FitsCompressionOptions.LOSSY,
            },
        ) as roundtrip:
            with roundtrip.inspect() as fits:
                for extname in ["IMAGE", "MASK", "VARIANCE", "MASK_FRACTIONS/REJECTED"] + [
                    f"NOISE_REALIZATIONS/{n}" for n in range(len(self.cell_coadd.noise_realizations))
                ]:
                    with self.subTest(extname=extname):
                        self.assertEqual(fits[extname].header["ZTILE1"], self.cell_coadd.grid.cell_shape.x)
                        self.assertEqual(fits[extname].header["ZTILE2"], self.cell_coadd.grid.cell_shape.y)
                        if extname == "MASK" or extname.startswith("MASK_FRACTIONS"):
                            self.assertEqual(fits[extname].header["ZCMPTYPE"], "GZIP_2")
                        else:
                            self.assertEqual(fits[extname].header["ZCMPTYPE"], "RICE_1")
                            self.assertEqual(fits[extname].header["ZQUANTIZ"], "SUBTRACTIVE_DITHER_2")

    def test_fits_json_consistency(self) -> None:
        """FITS and JSON backends produce equal CellCoadds on round-trip."""
        with (
            RoundtripFits(self, self.cell_coadd) as fits_rt,
            RoundtripJson(self, self.cell_coadd) as json_rt,
        ):
            assert_cell_coadds_equal(self, self.cell_coadd, fits_rt.result, expect_view=False)
            assert_cell_coadds_equal(self, self.cell_coadd, json_rt.result, expect_view=False)
            assert_cell_coadds_equal(self, fits_rt.result, json_rt.result, expect_view=False)

    def test_to_legacy(self) -> None:
        """Test converting a CellCoadd back into a legacy MultipleCellCoadd."""
        legacy_cell_coadd = self.cell_coadd.to_legacy()
        compare_cell_coadd_to_legacy(
            self,
            self.cell_coadd,
            legacy_cell_coadd,
            tract_bbox=Box.from_legacy(self.skymap[DP2_COADD_DATA_ID["tract"]].getBBox()),
            plane_map=self.plane_map,
            psf_points=self.psf_points,
        )

    def test_to_legacy_exposure(self) -> None:
        """Test converting a CellCoadd back into a legacy Exposure."""
        legacy_exposure = self.cell_coadd.to_legacy_exposure()

        self.assertEqual(legacy_exposure.getFilter().bandLabel, self.cell_coadd.band)
        self.assertEqual(Box.from_legacy(legacy_exposure.getBBox()), self.cell_coadd.bbox)
        compare_masked_image_to_legacy(
            self, self.cell_coadd, legacy_exposure.maskedImage, plane_map=self.plane_map, expect_view=True
        )
        compare_psf_to_legacy(
            self,
            self.cell_coadd.psf,
            legacy_exposure.getPsf(),
            points=self.psf_points,
            expect_legacy_raise_on_out_of_bounds=True,
        )
        compare_sky_projection_to_legacy_wcs(
            self,
            self.cell_coadd.sky_projection,
            legacy_exposure.getWcs(),
            self.cell_coadd.sky_projection.pixel_frame,
            subimage_bbox=self.cell_coadd.bbox,
            is_fits=True,
        )

    @unittest.skipUnless(HAVE_H5PY, "h5py is not installed")
    def test_round_trip_ndf(self) -> None:
        """NDF round-trip for CellCoadd, exercising hoisted long-named arrays.

        This test covers the HDS name-shrinker fix for noise_realizations.
        """
        with RoundtripNdf(self, self.cell_coadd, "CellCoadd") as roundtrip:
            assert_cell_coadds_equal(self, roundtrip.result, self.cell_coadd, expect_view=False)

    @unittest.skipUnless(HAVE_H5PY, "h5py is not installed")
    def test_fits_ndf_consistency(self) -> None:
        """FITS and NDF backends produce equal CellCoadds on round-trip."""
        with (
            RoundtripFits(self, self.cell_coadd) as fits_rt,
            RoundtripNdf(self, self.cell_coadd) as ndf_rt,
        ):
            assert_cell_coadds_equal(self, self.cell_coadd, fits_rt.result, expect_view=False)
            assert_cell_coadds_equal(self, self.cell_coadd, ndf_rt.result, expect_view=False)
            assert_cell_coadds_equal(self, fits_rt.result, ndf_rt.result, expect_view=False)

    @unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
    def test_zarr_roundtrip_uses_cell_aligned_chunks(self) -> None:
        """Writing a CellCoadd to zarr aligns chunks to the cell shape.

        The bug fixed in DM-55041 was that ``write()`` probed
        ``obj.cell_shape`` / ``obj.cell_grid`` but `CellCoadd` exposes
        the cell shape under ``obj.grid.cell_shape``. Without the fix,
        real CellCoadd writes fall back to generic 256-pixel chunks
        instead of cell-aligned chunks.
        """
        cell_shape = self.cell_coadd.grid.cell_shape
        with RoundtripZarr(self, self.cell_coadd, "CellCoadd") as roundtrip:
            with open_store_for_read(roundtrip.filename) as store:
                root = zarr.open_group(store=store, mode="r", zarr_format=3)
                self.assertEqual(tuple(root["image"].chunks), (cell_shape.y, cell_shape.x))

    @unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
    def test_fits_zarr_consistency(self) -> None:
        """FITS and zarr backends produce equal CellCoadds on round-trip."""
        with (
            RoundtripFits(self, self.cell_coadd) as fits_rt,
            RoundtripZarr(self, self.cell_coadd) as zarr_rt,
        ):
            assert_cell_coadds_equal(self, self.cell_coadd, fits_rt.result, expect_view=False)
            assert_cell_coadds_equal(self, self.cell_coadd, zarr_rt.result, expect_view=False)
            assert_cell_coadds_equal(self, fits_rt.result, zarr_rt.result, expect_view=False)


if __name__ == "__main__":
    unittest.main()

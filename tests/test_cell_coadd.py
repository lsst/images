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
from lsst.images.tests import (
    DP2_COADD_DATA_ID,
    DP2_COADD_MISSING_CELL,
    RoundtripFits,
    assert_masked_images_equal,
    assert_psfs_equal,
    compare_cell_coadd_to_legacy,
)

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
                    for k in ["projection", "image", "mask", "variance", "psf", "provenance"]
                }
        self.assertEqual(self.cell_coadd.bounds.missing, {self.missing_cell})
        self.assertEqual(self.cell_coadd.bbox, Box.factory[12900:13500, 9600:10050])
        compare_cell_coadd_to_legacy(
            self,
            roundtrip.result,
            self.legacy_cell_coadd,
            tract_bbox=Box.from_legacy(self.skymap[DP2_COADD_DATA_ID["tract"]].getBBox()),
            plane_map=self.plane_map,
            alternates=alternates,
            psf_points=self.psf_points,
        )


if __name__ == "__main__":
    unittest.main()

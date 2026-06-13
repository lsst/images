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

"""Read-back tests for the legacy-derived v1 fixtures.

The fixtures under ``tests/data/schema_v1/legacy/`` are derived from real
files (converted from legacy formats) by ``_minify_for_fixtures``.  Their
whole point is to prove that we can still read real-data serializations back
in -- so these tests just read each fixture and check its structure.

These tests do not require ``lsst.cell_coadds`` (the ``CellCoadd`` fixture is
read purely through ``lsst.images``).  They also do not require ``piff``: a
``VisitImage`` whose PSF cannot be deserialized still reads successfully, with
the error deferred until the PSF is actually accessed.  We assert that
deferred behaviour directly, and only dereference the PSF when ``piff`` is
importable.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from lsst.images import VisitImage
from lsst.images.cells import CellCoadd
from lsst.images.serialization import ArchiveReadError, read

SCHEMA_DIR = Path(__file__).parent / "data" / "schema_v1"
LEGACY_DIR = SCHEMA_DIR / "legacy"

try:
    import piff  # noqa: F401

    HAVE_PIFF = True
except ImportError:
    HAVE_PIFF = False


class LegacyFixtureReadBackTestCase(unittest.TestCase):
    """Read each legacy-derived fixture back in and check its structure."""

    def test_cell_coadd(self) -> None:
        """The CellCoadd fixture reads back as a multi-cell coadd and can be
        subset down to a single cell.
        """
        path = LEGACY_DIR / "cell_coadd.json"
        if not path.exists():
            self.skipTest(f"{path} not present")

        coadd = read(str(path), CellCoadd)

        self.assertIsInstance(coadd.band, str)
        self.assertIsNotNone(coadd.image.unit)
        # The fixture deliberately keeps more than one cell so it is a useful
        # test, and includes a missing cell to exercise the sparse-grid path.
        present = list(coadd.bounds.cell_indices())
        self.assertGreaterEqual(len(present), 2)
        self.assertTrue(coadd.bounds.missing, "fixture should retain a missing cell")
        # Provenance survived the minify and has the expected two-table shape.
        self.assertGreater(len(coadd.provenance.inputs), 0)
        self.assertGreater(len(coadd.provenance.contributions), 0)
        self.assertIn("polygon", coadd.provenance.inputs.colnames)

        # Subsetting to a single present cell still works on the morphed grid.
        cell_bbox = coadd.grid.bbox_of(present[0])
        single = coadd[cell_bbox]
        self.assertEqual(list(single.bounds.cell_indices()), [present[0]])

    def test_visit_images(self) -> None:
        """Each VisitImage fixture reads back with its real detector and a
        deferred PSF (raised only on access when piff is unavailable).
        """
        for name in ("visit_image_dp1.json", "visit_image_dp2.json"):
            with self.subTest(name=name):
                path = LEGACY_DIR / name
                if not path.exists():
                    self.skipTest(f"{path} not present")

                visit_image = read(str(path), VisitImage)

                # Pixel planes were cropped to a small corner.
                self.assertEqual(visit_image.image.array.ndim, 2)
                self.assertLessEqual(max(visit_image.image.array.shape), 16)
                self.assertIsInstance(visit_image.band, str)
                self.assertIsNotNone(visit_image.image.unit)
                # The real detector (with trimmed amplifiers) round-tripped.
                self.assertGreaterEqual(len(visit_image.detector.amplifiers), 1)
                self.assertGreater(len(visit_image.aperture_corrections), 0)

                if HAVE_PIFF:
                    self.assertIsNotNone(visit_image.psf)
                else:
                    # Reading succeeded above; the PSF error is deferred to
                    # the point of access.
                    with self.assertRaises(ArchiveReadError):
                        visit_image.psf


if __name__ == "__main__":
    unittest.main()

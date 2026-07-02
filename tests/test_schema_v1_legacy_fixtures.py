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

from pathlib import Path

import pytest

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


def test_cell_coadd() -> None:
    """Verify the CellCoadd fixture reads back as a multi-cell coadd.

    Also verifies it can be subset down to a single cell.
    """
    path = LEGACY_DIR / "cell_coadd.json"
    if not path.exists():
        pytest.skip(f"{path} not present")

    coadd = read(path, CellCoadd)

    assert isinstance(coadd.band, str)
    assert coadd.image.unit is not None
    # The fixture deliberately keeps more than one cell so it is a useful
    # test, and includes a missing cell to exercise the sparse-grid path.
    present = list(coadd.bounds.cell_indices())
    assert len(present) >= 2
    assert coadd.bounds.missing, "fixture should retain a missing cell"
    # Provenance survived the minify and has the expected two-table shape.
    assert len(coadd.provenance.inputs) > 0
    assert len(coadd.provenance.contributions) > 0
    assert "polygon" in coadd.provenance.inputs.colnames

    # Subsetting to a single present cell still works on the morphed grid.
    cell_bbox = coadd.grid.bbox_of(present[0])
    single = coadd[cell_bbox]
    assert list(single.bounds.cell_indices()) == [present[0]]


@pytest.mark.parametrize("name", ["visit_image_dp1.json", "visit_image_dp2.json"])
def test_visit_images(name: str) -> None:
    """Verify each VisitImage legacy fixture reads back with its real detector.

    Also verifies deferred PSF behaviour when piff is unavailable.
    """
    path = LEGACY_DIR / name
    if not path.exists():
        pytest.skip(f"{path} not present")

    visit_image = read(path, VisitImage)

    # Pixel planes were cropped to a small corner.
    assert visit_image.image.array.ndim == 2
    assert max(visit_image.image.array.shape) <= 16
    assert isinstance(visit_image.band, str)
    assert visit_image.image.unit is not None
    # The real detector (with trimmed amplifiers) round-tripped.
    assert len(visit_image.detector.amplifiers) >= 1
    assert len(visit_image.aperture_corrections) > 0

    if HAVE_PIFF:
        assert visit_image.psf is not None
    else:
        # Reading succeeded above; the PSF error is deferred to
        # the point of access.
        with pytest.raises(ArchiveReadError):
            visit_image.psf

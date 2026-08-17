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

import copy
import dataclasses
import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from lsst.images import YX, BoundsError, Box, Interval, MaskPlane, get_legacy_deep_coadd_mask_planes
from lsst.images.cells import (
    CellCoadd,
    CellGrid,
    CellGridBounds,
    CellIJ,
    CellPointSpreadFunctionSerializationModel,
    CoaddProvenance,
    PatchDefinition,
)
from lsst.images.describe import DescribeOptions, Report
from lsst.images.fields import ChebyshevField
from lsst.images.fits import FitsCompressionOptions
from lsst.images.serialization import JsonRef, class_for_schema, parameterize_tree, read_archive
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
    current_fixture_path,
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
FIXTURE_DIR = Path(__file__).parent / "data" / "schemas"

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
    cell_coadd = CellCoadd.from_legacy_cell_coadd(
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
    path = current_fixture_path(FIXTURE_DIR, "cell_coadd", variant="as_shipped")
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


def test_cell_coadd_repr_str_pinned(minified_cell_coadd: CellCoadd) -> None:
    """Pin the exact str and repr output of a CellCoadd.

    A coadd takes component types no string can express, so repr is the
    descriptive form rather than a constructor call.  Both come from the
    report, as they do for the sibling `VisitImage`.
    """
    assert str(minified_cell_coadd) == "CellCoadd([y=48:60, x=36:48], tract=9813)"
    assert repr(minified_cell_coadd) == "<CellCoadd([y=48:60, x=36:48], tract=9813)>"


def report_fields(report: Report) -> dict[str, Any]:
    """Return a mapping from report field label to field value."""
    return {field.label: field.value for field in report.fields}


def make_test_provenance() -> CoaddProvenance:
    """Return a provenance with three input images taken over two nights,
    whose contributions cover three cells unevenly.
    """
    inputs = CoaddProvenance.make_empty_input_table(3)
    inputs["instrument"] = "LSSTCam"
    inputs["physical_filter"] = "r_57"
    inputs["visit"] = [101, 102, 103]
    inputs["detector"] = [1, 2, 3]
    inputs["day_obs"] = [20250520, 20250520, 20250521]
    contributions = CoaddProvenance.make_empty_contribution_table(6)
    contributions["cell_i"] = [0, 0, 0, 1, 1, 2]
    contributions["cell_j"] = 0
    return CoaddProvenance(inputs=inputs, contributions=contributions)


def test_coadd_provenance_repr_str_pinned(minified_cell_coadd: CellCoadd) -> None:
    """Pin the str and repr of a CoaddProvenance.

    Neither table can be expressed in an eval-able string, so repr is the
    descriptive form, and both report only what len() can answer.
    """
    provenance = minified_cell_coadd.provenance
    assert str(provenance) == "CoaddProvenance(6 input images)"
    assert repr(provenance) == "<CoaddProvenance(6 input images)>"


def test_coadd_provenance_brief_report_skips_column_scans(minified_cell_coadd: CellCoadd) -> None:
    """The brief report carries no fields, so repr and str never scan a
    column of either table.
    """
    report = minified_cell_coadd.provenance._describe(DescribeOptions(brief=True))
    assert not report.fields
    assert report.to_repr() == "<CoaddProvenance(6 input images)>"


def test_coadd_provenance_report_summarizes_its_tables(minified_cell_coadd: CellCoadd) -> None:
    """The expanded report summarizes both tables instead of rendering them."""
    report = minified_cell_coadd.provenance.describe()
    assert report.type_name == "CoaddProvenance"
    assert report_fields(report) == {
        "instrument": "LSSTCam",
        "physical_filter": "r_57",
        "input images": "6 from 2 visits",
        "day_obs": "20250520",
        "cells": "3 with contributions",
        "per cell": "2 input images",
    }
    assert list(report_fields(report)) == [
        "instrument",
        "physical_filter",
        "input images",
        "day_obs",
        "cells",
        "per cell",
    ]
    # Rich renders an astropy table as plain text and never consults its
    # _repr_html_, so the report tabulates nothing.
    assert not report.tables
    assert "input images" in report._repr_html_()


def test_coadd_provenance_report_counts_cells_against_bounds(minified_cell_coadd: CellCoadd) -> None:
    """Given the image's cells, the report states coverage as a fraction."""
    report = minified_cell_coadd.provenance._describe(bounds=minified_cell_coadd.bounds)
    assert report_fields(report)["cells"] == "3 of 3 with contributions"


def test_coadd_provenance_report_shows_partial_cell_coverage() -> None:
    """Cells with image data but no contribution rows show up in the ratio."""
    grid = CellGrid(bbox=Box.from_shape((30, 30)), cell_shape=YX(10, 10))
    bounds = CellGridBounds(grid=grid, bbox=Box.factory[0:30, 0:30])
    report = make_test_provenance()._describe(bounds=bounds)
    assert report_fields(report)["cells"] == "3 of 9 with contributions"


def test_coadd_provenance_report_ranges_over_nights_and_cells() -> None:
    """Several visits, several nights and uneven cell coverage give the
    ranged forms of each field.
    """
    fields = report_fields(make_test_provenance().describe())
    assert fields["input images"] == "3 from 3 visits"
    assert fields["day_obs"] == "20250520 - 20250521"
    assert fields["cells"] == "3 with contributions"
    assert fields["per cell"] == "1 - 3 input images (median 2)"


def test_coadd_provenance_report_handles_empty_tables() -> None:
    """A provenance with no rows describes without raising."""
    provenance = CoaddProvenance(
        inputs=CoaddProvenance.make_empty_input_table(0),
        contributions=CoaddProvenance.make_empty_contribution_table(0),
    )
    assert str(provenance) == "CoaddProvenance(no input images)"
    report = provenance.describe()
    assert report_fields(report) == {"input images": "none", "cells": "none"}
    report.__rich__()


def test_coadd_provenance_report_counts_one_input_image_in_the_singular() -> None:
    """Pin the singular forms of the counted phrases.

    One input image is what the standalone ``coadd_provenance`` file holds,
    and what a single-cell subset of a coadd holds, so this is the case the
    ``describe`` command line subcommand shows most often.
    """
    inputs = CoaddProvenance.make_empty_input_table(1)
    inputs["instrument"] = "LSSTCam"
    inputs["physical_filter"] = "r_57"
    inputs["visit"] = [101]
    inputs["detector"] = [1]
    inputs["day_obs"] = [20250520]
    provenance = CoaddProvenance(
        inputs=inputs, contributions=CoaddProvenance.make_empty_contribution_table(1)
    )
    assert str(provenance) == "CoaddProvenance(1 input image)"
    assert repr(provenance) == "<CoaddProvenance(1 input image)>"
    fields = report_fields(provenance.describe())
    assert fields["input images"] == "1 from 1 visit"
    assert fields["per cell"] == "1 input image"


def test_cell_coadd_report_accounts_for_every_component(minified_cell_coadd: CellCoadd) -> None:
    """Nothing the coadd carries is absent from its report."""
    report = minified_cell_coadd.describe()
    fields = report_fields(report)
    assert fields["mask_fractions"] == "rejected"
    assert fields["noise_realizations"] == "1 image (dtype float32)"
    assert fields["aperture_corrections"] == "3 fields"
    # Provenance is a child, so it needs no field line of its own.
    assert "provenance" not in fields
    assert list(report.children) == [
        "image",
        "mask",
        "variance",
        "sky_projection",
        "psf",
        "provenance",
        "backgrounds",
    ]
    provenance = report.children["provenance"]
    assert provenance.type_name == "CoaddProvenance"
    # The coadd passed its cells down, so coverage is stated as a fraction.
    assert report_fields(provenance)["cells"] == "3 of 3 with contributions"


def test_cell_coadd_report_counts_aperture_corrections_only(minified_cell_coadd: CellCoadd) -> None:
    """A coadd can carry dozens of aperture corrections with long names, so
    the report counts them and never names them, detail or not.
    """
    plain = report_fields(minified_cell_coadd.describe())["aperture_corrections"]
    assert plain == "3 fields"
    detailed = report_fields(minified_cell_coadd.describe(detail=True))["aperture_corrections"]
    assert detailed == "3 fields"  # No additional detail


def test_cell_coadd_report_states_absent_provenance(minified_cell_coadd: CellCoadd) -> None:
    """A coadd with no provenance says so, while components that are merely
    empty stay out of the report.
    """
    bare = CellCoadd(
        minified_cell_coadd.image,
        mask=minified_cell_coadd.mask,
        variance=minified_cell_coadd.variance,
        sky_projection=minified_cell_coadd.sky_projection,
        band=minified_cell_coadd.band,
        psf=minified_cell_coadd.psf,
        patch=minified_cell_coadd.patch,
    )
    report = bare.describe()
    fields = report_fields(report)
    assert fields["provenance"] == "none"
    assert "provenance" not in report.children
    assert "mask_fractions" not in fields
    assert "noise_realizations" not in fields
    assert "aperture_corrections" not in fields


def test_cell_grid_patch_str_uses_clean_geometry() -> None:
    """CellGrid, PatchDefinition and CellGridBounds str drop the
    Interval/YX/Box/CellIJ wrappers.

    The report renders field values with str, so these must use the compact
    geometry forms rather than pydantic's default field-by-field repr.
    """
    grid = CellGrid(bbox=Box.from_shape((100, 200)), cell_shape=YX(10, 20))
    assert str(grid) == "[y=0:100, x=0:200], cell_shape=(y=10, x=20)"

    patch = PatchDefinition(id=73, index=YX(7, 3), inner_bbox=Box.factory[1:3, 2:4], cells=grid)
    assert str(patch) == (
        "id=73, index=(y=7, x=3), inner_bbox=[y=1:3, x=2:4], "
        "cells=([y=0:100, x=0:200], cell_shape=(y=10, x=20))"
    )
    # repr stays as the pydantic default so it remains eval-ish and distinct.
    assert "Interval(" in repr(patch)
    assert "YX(" in repr(patch)

    bounds = CellGridBounds(grid=grid, bbox=Box.factory[0:40, 0:60])
    assert str(bounds) == "[y=0:40, x=0:60] in grid ([y=0:100, x=0:200], cell_shape=(y=10, x=20))"
    bounds_missing = CellGridBounds(
        grid=grid, bbox=Box.factory[0:40, 0:60], missing=frozenset({CellIJ(i=1, j=1), CellIJ(i=0, j=2)})
    )
    assert str(bounds_missing) == (
        "[y=0:40, x=0:60] in grid ([y=0:100, x=0:200], cell_shape=(y=10, x=20)), "
        "missing={(i=0, j=2), (i=1, j=1)}"
    )
    assert "Interval(" in repr(bounds_missing)


def test_cell_shape_accepts_both_spellings() -> None:
    """Verify both on-disk spellings of an XY pair still validate.

    Shipped cell_coadd 1.0.0 files carry the array spelling, so this stays
    readable until a major bump retires the as_shipped fixture.  A focused
    check here gives a two-line failure instead of a 52 KB fixture failing
    to read.
    """
    tree_cls = class_for_schema("cell_psf")
    assert tree_cls is not None
    model = parameterize_tree(tree_cls, JsonRef)
    fixture = json.loads(current_fixture_path(FIXTURE_DIR, "cell_psf").read_text())
    assert fixture["bounds"]["grid"]["cell_shape"] == {"y": 4, "x": 4}

    legacy = copy.deepcopy(fixture)
    legacy["bounds"]["grid"]["cell_shape"] = [4, 4]
    tree = model.model_validate(legacy)
    assert isinstance(tree, CellPointSpreadFunctionSerializationModel)
    assert tree.bounds.grid.cell_shape.y == 4
    assert tree.bounds.grid.cell_shape.x == 4


def test_cell_psf_rejects_index_below_bounds(minified_cell_coadd: CellCoadd) -> None:
    """An index below the PSF bounds must not wrap around its array."""
    start = minified_cell_coadd.psf.bounds.subgrid_start
    with pytest.raises(BoundsError, match="out of bounds"):
        minified_cell_coadd.psf[CellIJ(i=start.i - 1, j=start.j)]


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


# The float32 image pixels are of order 10 nJy, as is the test background
# below, so background arithmetic is only reproducible to a few ULPs at that
# scale.
BACKGROUND_ATOL = 1e-5


def _add_gradient_background(coadd: CellCoadd, name: str = "pretty") -> ChebyshevField:
    """Add a non-constant background over the coadd's full bbox and return the
    field that was added.
    """
    field = ChebyshevField(
        coadd.bbox,
        np.array([[10.0, 2.0, 0.5], [1.0, 0.75, 0.0], [0.25, 0.0, 0.0]]),
        unit=coadd.unit,
    )
    coadd.backgrounds.add(name, field, description="Gradient background for tests.")
    return field


def _make_background_subbox(coadd: CellCoadd) -> Box:
    """Make a box that trims a different number of pixels from each side of the
    coadd, so a background rendered over the wrong box cannot match by chance.
    """
    return Box.factory[
        coadd.bbox.y.start + 2 : coadd.bbox.y.stop - 3,
        coadd.bbox.x.start + 1 : coadd.bbox.x.stop - 2,
    ]


def test_apply_background_subimage(minified_cell_coadd: CellCoadd) -> None:
    """Test that applying a background to a subimage subtracts only the
    portion of the background model that overlaps the subimage.
    """
    field = _add_gradient_background(minified_cell_coadd)
    subbox = _make_background_subbox(minified_cell_coadd)
    subimage = minified_cell_coadd[subbox]
    original = subimage.image.array.copy()

    subimage.apply_background("pretty")

    assert subimage.backgrounds.subtracted is not None
    assert subimage.backgrounds.subtracted.name == "pretty"
    assert subimage.image.bbox == subbox
    expected = field.render(subbox, dtype=subimage.image.array.dtype).array
    np.testing.assert_allclose(original - subimage.image.array, expected, atol=BACKGROUND_ATOL)


def test_restore_background_subimage(minified_cell_coadd: CellCoadd) -> None:
    """Test that restoring the original background on a subimage adds back
    only the portion of the background model that overlaps the subimage.
    """
    _add_gradient_background(minified_cell_coadd)
    subbox = _make_background_subbox(minified_cell_coadd)
    subimage = minified_cell_coadd[subbox]
    original = subimage.image.array.copy()
    subimage.apply_background("pretty")

    subimage.apply_background(None)

    assert subimage.backgrounds.subtracted is None
    np.testing.assert_allclose(subimage.image.array, original, atol=BACKGROUND_ATOL)


def test_failed_background_switch_preserves_pixels_and_state(
    minified_cell_coadd: CellCoadd,
) -> None:
    """An invalid replacement must not restore the current background."""
    _add_gradient_background(minified_cell_coadd)
    minified_cell_coadd.apply_background("pretty")
    before = minified_cell_coadd.image.array.copy()

    with pytest.raises(KeyError):
        minified_cell_coadd.apply_background("missing")

    np.testing.assert_array_equal(minified_cell_coadd.image.array, before)
    assert minified_cell_coadd.backgrounds.subtracted is not None
    assert minified_cell_coadd.backgrounds.subtracted.name == "pretty"


def test_switch_background_applies_replacement(minified_cell_coadd: CellCoadd) -> None:
    """Switching models restores the old background and subtracts the new."""
    field = _add_gradient_background(minified_cell_coadd)
    minified_cell_coadd.backgrounds.add("other", field * 2.0)
    original = minified_cell_coadd.image.array.copy()
    minified_cell_coadd.apply_background("pretty")

    minified_cell_coadd.apply_background("other")

    expected = (
        original
        - (field * 2.0).render(minified_cell_coadd.bbox, dtype=minified_cell_coadd.image.array.dtype).array
    )
    np.testing.assert_allclose(minified_cell_coadd.image.array, expected, atol=BACKGROUND_ATOL)
    assert minified_cell_coadd.backgrounds.subtracted is not None
    assert minified_cell_coadd.backgrounds.subtracted.name == "other"


def test_apply_background_after_bounded_read(minified_cell_coadd: CellCoadd) -> None:
    """Test that a background can be applied to a coadd read with a bbox
    parameter.
    """
    field = _add_gradient_background(minified_cell_coadd)
    subbox = _make_background_subbox(minified_cell_coadd)
    with RoundtripJson(minified_cell_coadd, "CellCoadd") as roundtrip:
        subimage = roundtrip.get(bbox=subbox)
        original = subimage.image.array.copy()

        subimage.apply_background("pretty")

        assert subimage.image.bbox == subbox
        expected = field.render(subbox, dtype=subimage.image.array.dtype).array
        np.testing.assert_allclose(original - subimage.image.array, expected, atol=BACKGROUND_ATOL)


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

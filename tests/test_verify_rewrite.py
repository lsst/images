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
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from click.testing import CliRunner

from lsst.images import YX, Background, BackgroundMap, Box, DifferenceImage
from lsst.images.cli._main import main
from lsst.images.fields import ChebyshevField
from lsst.images.tests import (
    DP2_COADD_DATA_ID,
    DP2_VISIT_DETECTOR_DATA_ID,
    get_dp2_exposure_record,
    reset_afw_mask_planes,  # noqa: F401
)
from lsst.images.tests.verify_rewrite import (
    RewriteVerifier,
    _check_backgrounds,
    _check_kernel,
    _check_templates,
    _spot_check_points,
)


def _bbox() -> Box:
    return Box.factory[0:10, 0:10]


def _constant_background(name: str = "subtracted") -> Background:
    field = ChebyshevField(_bbox(), np.array([[5.0]]))
    return Background(name=name, field=field, description="test")


def test_check_backgrounds_present_and_finite() -> None:
    """Check that present and finite backgrounds pass."""
    backgrounds = BackgroundMap(
        [_constant_background("subtracted"), _constant_background("skyCorr")],
        subtracted="subtracted",
    )
    _check_backgrounds(backgrounds, _bbox(), expected=("subtracted", "skyCorr"))


def test_check_backgrounds_missing_expected_fails() -> None:
    """Check that a missing expected background name fails."""
    backgrounds = BackgroundMap([_constant_background("subtracted")], subtracted="subtracted")
    with pytest.raises(AssertionError, match="expected background 'skyCorr' not attached"):
        _check_backgrounds(backgrounds, _bbox(), expected=("subtracted", "skyCorr"))


def test_check_backgrounds_non_finite_fails() -> None:
    """Check that a non-finite background field fails."""
    field = ChebyshevField(_bbox(), np.array([[np.nan]]))
    backgrounds = BackgroundMap([Background(name="subtracted", field=field)], subtracted="subtracted")
    with pytest.raises(AssertionError, match="non-finite"):
        _check_backgrounds(backgrounds, _bbox())


def test_check_backgrounds_no_background_ok() -> None:
    """Check that an empty background map with no expectations passes."""
    _check_backgrounds(BackgroundMap(), _bbox(), expected=())


def test_check_kernel_none_fails() -> None:
    """Check that a None kernel fails."""
    with pytest.raises(AssertionError):
        _check_kernel(None)


def test_check_templates_none_fails() -> None:
    """Check that a None templates list fails."""
    with pytest.raises(AssertionError):
        _check_templates(None, _bbox())


def test_check_templates_empty_fails() -> None:
    """Check that an empty templates list fails."""
    with pytest.raises(AssertionError):
        _check_templates([], _bbox())


def test_rewrite_verifier_reports_problems() -> None:
    """Assert that print_error increments the problem count."""
    verifier = RewriteVerifier(None, "difference_image", old_prefix="legacy_", new_prefix="")
    assert verifier.n_problems == 0
    verifier.print_error("data_id", AssertionError("boom"))
    assert verifier.n_problems == 1


def test_print_error_no_note(capsys: pytest.CaptureFixture[str]) -> None:
    """An error without a component note is printed on one line."""
    verifier = RewriteVerifier(None, "difference_image", old_prefix="legacy_", new_prefix="")
    verifier.print_error("data_id", AssertionError("boom"))
    assert capsys.readouterr().out == "data_id: boom (AssertionError)\n"


def test_print_error_component_note(capsys: pytest.CaptureFixture[str]) -> None:
    """A component note is printed as a prefix on its own indented line."""
    verifier = RewriteVerifier(None, "difference_image", old_prefix="legacy_", new_prefix="")
    err = AssertionError("81/6642 values differ; max abs diff 1e-9 at index (np.int64(0),)")
    err.add_note("sky_projection")
    verifier.print_error("data_id", err)
    assert capsys.readouterr().out == (
        "data_id\n   sky_projection: 81/6642 values differ; max abs diff 1e-9 at index"
        " (np.int64(0),) (AssertionError)\n"
    )


def test_print_error_nested_notes(capsys: pytest.CaptureFixture[str]) -> None:
    """Nested component notes are rendered as a path, not repeated messages."""
    verifier = RewriteVerifier(None, "difference_image", old_prefix="legacy_", new_prefix="")
    err = AssertionError("boom")
    err.add_note("visit_image")
    err.add_note("sky_projection")
    verifier.print_error("data_id", err)
    assert capsys.readouterr().out == "data_id\n   visit_image -> sky_projection: boom (AssertionError)\n"


def test_spot_check_points_subset() -> None:
    """Check that the sample has the requested size, no repeats, and stays
    within the input points.
    """
    points = YX(y=np.arange(10.0), x=100.0 * np.arange(10.0))
    result = _spot_check_points(points, 4, random.Random("test"))
    assert len(result.y) == 4
    assert len(set(result.y.tolist())) == 4
    assert set(result.y.tolist()) <= set(points.y.tolist())
    assert set(result.x.tolist()) <= set(points.x.tolist())
    # x/y pairs must stay together.
    np.testing.assert_array_equal(result.x, 100.0 * result.y)


def test_spot_check_points_deterministic() -> None:
    """Check that a seeded random.Random gives reproducible samples."""
    points = YX(y=np.arange(20.0), x=np.arange(20.0) ** 2)
    first = _spot_check_points(points, 5, random.Random("seed"))
    second = _spot_check_points(points, 5, random.Random("seed"))
    np.testing.assert_array_equal(first.y, second.y)
    np.testing.assert_array_equal(first.x, second.x)


def test_spot_check_points_more_requested_than_available() -> None:
    """Check that requesting at least as many points as available returns
    all of them.
    """
    points = YX(y=np.arange(3.0), x=2.0 * np.arange(3.0))
    for n in (3, 10):
        result = _spot_check_points(points, n, random.Random("all"))
        assert len(result.y) == 3
        assert set(result.y.tolist()) == set(points.y.tolist())
        np.testing.assert_array_equal(result.x, 2.0 * result.y)


def test_rewrite_verifier_stores_coadd_options() -> None:
    """Check that the constructor stores the CellCoadd-check options."""
    verifier = RewriteVerifier(
        None,
        "template_coadd",
        old_prefix="legacy_",
        new_prefix="",
        pixel_atol_factor=0.25,
        coadd_check_points=7,
    )
    assert verifier.pixel_atol_factor == 0.25
    assert verifier.coadd_check_points == 7
    default = RewriteVerifier(None, "template_coadd", old_prefix="legacy_", new_prefix="")
    assert default.pixel_atol_factor == 0.5
    assert default.coadd_check_points == 4


@pytest.fixture(scope="module")
def testdata_dir() -> str:
    """Return the external test-data directory, skipping if unset."""
    if (result := os.environ.get("TESTDATA_IMAGES_DIR")) is None:
        pytest.skip("TESTDATA_IMAGES_DIR is not set.")
    return result


def _make_dp2_rewrite_repo(repo: str, src: str) -> None:
    """Build a repo holding a legacy DP2 difference image and its rewritten
    counterpart, ready for ``verify-rewrite``.

    The rewritten dataset is converted from the source file with the given
    (or the true DP2) exposure record; passing a tampered record produces a
    product that ``verify-rewrite``'s observation-info check should reject.
    """
    from lsst.afw.image import ExposureF
    from lsst.daf.butler import Butler, DataCoordinate, DatasetType

    Butler.makeRepo(repo)
    with Butler.from_config(repo, run="run1") as butler:
        exposure_record = get_dp2_exposure_record(butler.dimensions)
        reg = butler.registry
        reg.insertDimensionData("instrument", {"instrument": exposure_record.instrument})
        reg.insertDimensionData(
            "day_obs", {"instrument": exposure_record.instrument, "day_obs": exposure_record.day_obs}
        )
        reg.insertDimensionData(
            "physical_filter",
            {
                "physical_filter": exposure_record.physical_filter,
                "band": DP2_VISIT_DETECTOR_DATA_ID["band"],
                "instrument": exposure_record.instrument,
            },
        )
        reg.insertDimensionData(
            "detector",
            {
                "instrument": exposure_record.instrument,
                "id": DP2_VISIT_DETECTOR_DATA_ID["detector"],
                "full_name": "R21_S11",
            },
        )
        reg.insertDimensionData(
            "visit",
            {
                "instrument": exposure_record.instrument,
                "id": exposure_record.id,
                "physical_filter": exposure_record.physical_filter,
                "name": str(exposure_record.id),
                "day_obs": exposure_record.day_obs,
            },
        )
        reg.insertDimensionData(
            "visit_detector_region",
            {
                "instrument": exposure_record.instrument,
                "visit": exposure_record.id,
                "detector": DP2_VISIT_DETECTOR_DATA_ID["detector"],
                "region": None,
            },
        )
        reg.insertDimensionData(
            "group", {"instrument": exposure_record.instrument, "name": exposure_record.group}
        )
        reg.insertDimensionData("exposure", exposure_record)
        reg.insertDimensionData(
            "visit_definition",
            {
                "instrument": exposure_record.instrument,
                "visit": exposure_record.id,
                "exposure": exposure_record.id,
            },
        )

        dims = ("instrument", "visit", "detector")
        data_id = DataCoordinate.standardize(
            {
                "instrument": exposure_record.instrument,
                "visit": exposure_record.id,
                "detector": DP2_VISIT_DETECTOR_DATA_ID["detector"],
            },
            universe=butler.dimensions,
        )
        legacy_dt = DatasetType("legacy_difference_image", dims, "ExposureF", universe=butler.dimensions)
        new_dt = DatasetType("difference_image", dims, "DifferenceImage", universe=butler.dimensions)
        reg.registerDatasetType(legacy_dt)
        reg.registerDatasetType(new_dt)

        butler.put(ExposureF(src), legacy_dt, data_id)
        # Convert with a second read of the file, not from_legacy() on the
        # exposure above: the conversion shares pixel data with, and mutates
        # the metadata of, the exposure it is given.
        new = DifferenceImage.read_legacy(src, exposure_record=exposure_record)
        butler.put(new, new_dt, data_id)


def _invoke_verify_rewrite(repo: str) -> Any:
    return CliRunner().invoke(
        main,
        [
            "verify-rewrite",
            repo,
            "difference_image",
            "run1",
            "--no-check-kernel",
            "--no-check-templates",
            "--no-require-compressed",
        ],
    )


def test_verify_rewrite_end_to_end(tmp_path: Path, testdata_dir: str, reset_afw_mask_planes) -> None:  # noqa: F811
    """Run verify-rewrite on a real difference image."""
    try:
        import lsst.afw.image
        import lsst.daf.butler  # noqa: F401
    except ImportError:
        pytest.skip("lsst.daf.butler and lsst.afw could not be imported.")

    src = os.path.join(testdata_dir, "dp2", "legacy", "difference_image.fits")
    repo = str(tmp_path / "repo")
    _make_dp2_rewrite_repo(repo, src)
    result = _invoke_verify_rewrite(repo)
    assert result.exit_code == 0, result.output


def _make_dp2_coadd_rewrite_repo(repo: str, testdata_dir: str, *, tamper: bool = False) -> None:
    """Build a repo holding a legacy DP2 ``template_coadd`` (a patch-level
    lazy-PSF ``ExposureF``) and its CellCoadd conversion, ready for
    ``verify-rewrite``.

    With ``tamper``, the converted coadd's image gets a large constant
    offset over a sub-box, which ``verify-rewrite``'s coadd comparison
    should reject.
    """
    import pickle

    from lsst.afw.image import ExposureFitsReader
    from lsst.daf.butler import Butler, DataCoordinate, DatasetType
    from lsst.images.cells import CellCoadd

    Butler.makeRepo(repo)
    with Butler.from_config(repo, run="run1") as butler:
        with open(os.path.join(testdata_dir, "dp2", "legacy", "skyMap.pickle"), "rb") as stream:
            skymap = pickle.load(stream)
        tract_info = skymap[DP2_COADD_DATA_ID["tract"]]
        patch_index = tract_info[DP2_COADD_DATA_ID["patch"]].getIndex()
        reg = butler.registry
        # band has no records of its own; it is populated by physical_filter.
        reg.insertDimensionData("instrument", {"instrument": "LSSTCam"})
        reg.insertDimensionData(
            "physical_filter",
            {
                "instrument": "LSSTCam",
                "physical_filter": "r_57",
                "band": DP2_COADD_DATA_ID["band"],
            },
        )
        # The skymap hash and metadata and the tract/patch regions are never
        # consulted by verify-rewrite, so insert placeholders for them (the
        # same region=None trick _make_dp2_rewrite_repo uses for
        # visit_detector_region).
        reg.insertDimensionData(
            "skymap",
            {"name": DP2_COADD_DATA_ID["skymap"], "hash": (50).to_bytes(8, byteorder="little")},
        )
        reg.insertDimensionData(
            "tract",
            {"skymap": DP2_COADD_DATA_ID["skymap"], "id": DP2_COADD_DATA_ID["tract"], "region": None},
        )
        reg.insertDimensionData(
            "patch",
            {
                "skymap": DP2_COADD_DATA_ID["skymap"],
                "tract": DP2_COADD_DATA_ID["tract"],
                "id": DP2_COADD_DATA_ID["patch"],
                "cell_x": patch_index.x,
                "cell_y": patch_index.y,
                "region": None,
            },
        )

        dims = ("band", "skymap", "tract", "patch")
        data_id = DataCoordinate.standardize(DP2_COADD_DATA_ID, universe=butler.dimensions)
        legacy_dt = DatasetType("legacy_template_coadd", dims, "ExposureF", universe=butler.dimensions)
        new_dt = DatasetType("template_coadd", dims, "CellCoadd", universe=butler.dimensions)
        reg.registerDatasetType(legacy_dt)
        reg.registerDatasetType(new_dt)

        src = os.path.join(testdata_dir, "dp2", "legacy", "template_coadd.fits")
        butler.put(ExposureFitsReader(src).read(), legacy_dt, data_id)
        # Convert with a second read of the file, not from the exposure
        # above: the conversion shares pixel data with, and mutates the
        # metadata of, the exposure it is given.
        new = CellCoadd.from_legacy_lazy_coadd(
            ExposureFitsReader(src).read(),
            instrument="LSSTCam",
            tract_info=tract_info,
            skymap=DP2_COADD_DATA_ID["skymap"],
            patch=DP2_COADD_DATA_ID["patch"],
            band=DP2_COADD_DATA_ID["band"],
        )
        if tamper:
            # Offset far larger than verify-rewrite's pixel atol
            # (pixel_atol_factor times the legacy image's standard deviation).
            new.image.array[64:128, 64:128] += 1e4
        butler.put(new, new_dt, data_id)


def _invoke_verify_coadd_rewrite(repo: str) -> Any:
    return CliRunner().invoke(
        main,
        [
            "verify-rewrite",
            repo,
            "template_coadd",
            "run1",
            "--coadd-check-points",
            "2",
            # The test datastore writes plain GZIP-compressed (rather than
            # lossily RICE-compressed) FITS, so skip that leg, as in the
            # difference-image e2e test.
            "--no-require-compressed",
        ],
    )


def test_verify_rewrite_coadd_end_to_end(
    tmp_path: Path,
    testdata_dir: str,
    reset_afw_mask_planes,  # noqa: F811
) -> None:
    """Run verify-rewrite on a real legacy lazy-PSF coadd and its CellCoadd
    conversion.
    """
    try:
        import lsst.afw.image
        import lsst.daf.butler
        import lsst.skymap  # noqa: F401
    except ImportError:
        pytest.skip("lsst.daf.butler, lsst.skymap, and lsst.afw could not be imported.")

    repo = str(tmp_path / "repo")
    _make_dp2_coadd_rewrite_repo(repo, testdata_dir)
    result = _invoke_verify_coadd_rewrite(repo)
    assert result.exit_code == 0, result.output


def test_verify_rewrite_coadd_detects_tampering(
    tmp_path: Path,
    testdata_dir: str,
    reset_afw_mask_planes,  # noqa: F811
) -> None:
    """Check that verify-rewrite rejects a CellCoadd whose image was
    corrupted in the datastore.
    """
    try:
        import lsst.afw.image
        import lsst.daf.butler
        import lsst.skymap  # noqa: F401
    except ImportError:
        pytest.skip("lsst.daf.butler, lsst.skymap, and lsst.afw could not be imported.")

    repo = str(tmp_path / "repo")
    _make_dp2_coadd_rewrite_repo(repo, testdata_dir, tamper=True)
    result = _invoke_verify_coadd_rewrite(repo)
    assert result.exit_code == 1, result.output

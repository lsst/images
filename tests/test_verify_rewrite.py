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
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from lsst.images import Background, BackgroundMap, Box
from lsst.images.cli._main import main
from lsst.images.fields import ChebyshevField
from lsst.images.tests import (
    DP2_VISIT_DETECTOR_DATA_ID,
    reset_afw_mask_planes,  # noqa: F401
)
from lsst.images.tests.verify_rewrite import (
    RewriteVerifier,
    _check_backgrounds,
    _check_kernel,
    _check_templates,
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
    assert capsys.readouterr().out == "data_id: boom\n"


def test_print_error_component_note(capsys: pytest.CaptureFixture[str]) -> None:
    """A component note is printed as a prefix on its own indented line."""
    verifier = RewriteVerifier(None, "difference_image", old_prefix="legacy_", new_prefix="")
    err = AssertionError("81/6642 values differ; max abs diff 1e-9 at index (np.int64(0),)")
    err.add_note("sky_projection")
    verifier.print_error("data_id", err)
    assert capsys.readouterr().out == (
        "data_id\n   sky_projection: 81/6642 values differ; max abs diff 1e-9 at index (np.int64(0),)\n"
    )


def test_print_error_nested_notes(capsys: pytest.CaptureFixture[str]) -> None:
    """Nested component notes are rendered as a path, not repeated messages."""
    verifier = RewriteVerifier(None, "difference_image", old_prefix="legacy_", new_prefix="")
    err = AssertionError("boom")
    err.add_note("visit_image")
    err.add_note("sky_projection")
    verifier.print_error("data_id", err)
    assert capsys.readouterr().out == "data_id\n   visit_image -> sky_projection: boom\n"


@pytest.fixture(scope="module")
def testdata_dir() -> str:
    """Return the external test-data directory, skipping if unset."""
    if (result := os.environ.get("TESTDATA_IMAGES_DIR")) is None:
        pytest.skip("TESTDATA_IMAGES_DIR is not set.")
    return result


def test_verify_rewrite_end_to_end(tmp_path: Path, testdata_dir: str, reset_afw_mask_planes) -> None:  # noqa: F811
    """Run convert then verify-rewrite on a real difference image.

    Happy path: asserts the whole flow exits successfully.
    """
    try:
        from lsst.daf.butler import Butler, DataCoordinate, DatasetRef, DatasetType, FileDataset
    except ImportError:
        pytest.skip("lsst.daf.butler could not be imported.")

    src = os.path.join(testdata_dir, "dp2", "legacy", "difference_image.fits")
    converted = str(tmp_path / "difference_image.fits")
    result = CliRunner().invoke(main, ["convert", src, converted])
    assert result.exit_code == 0, result.output

    repo = str(tmp_path / "repo")
    Butler.makeRepo(repo)
    butler = Butler.from_config(repo, run="run1")
    reg = butler.registry
    reg.insertDimensionData(
        "instrument",
        {
            "instrument": DP2_VISIT_DETECTOR_DATA_ID["instrument"],
            "name": DP2_VISIT_DETECTOR_DATA_ID["instrument"],
        },
    )
    reg.insertDimensionData(
        "day_obs",
        {
            "instrument": DP2_VISIT_DETECTOR_DATA_ID["instrument"],
            "day_obs": DP2_VISIT_DETECTOR_DATA_ID["day_obs"],
        },
    )
    reg.insertDimensionData(
        "physical_filter",
        {
            "physical_filter": DP2_VISIT_DETECTOR_DATA_ID["physical_filter"],
            "band": DP2_VISIT_DETECTOR_DATA_ID["band"],
            "instrument": DP2_VISIT_DETECTOR_DATA_ID["instrument"],
        },
    )
    reg.insertDimensionData(
        "detector",
        {
            "instrument": DP2_VISIT_DETECTOR_DATA_ID["instrument"],
            "id": DP2_VISIT_DETECTOR_DATA_ID["detector"],
            "full_name": "R21_S11",
        },
    )
    reg.insertDimensionData(
        "visit",
        {
            "instrument": DP2_VISIT_DETECTOR_DATA_ID["instrument"],
            "id": DP2_VISIT_DETECTOR_DATA_ID["visit"],
            "physical_filter": DP2_VISIT_DETECTOR_DATA_ID["physical_filter"],
            "name": str(DP2_VISIT_DETECTOR_DATA_ID["visit"]),
            "day_obs": DP2_VISIT_DETECTOR_DATA_ID["day_obs"],
        },
    )
    reg.insertDimensionData(
        "visit_detector_region",
        {
            "instrument": DP2_VISIT_DETECTOR_DATA_ID["instrument"],
            "visit": DP2_VISIT_DETECTOR_DATA_ID["visit"],
            "detector": DP2_VISIT_DETECTOR_DATA_ID["detector"],
            "region": None,
        },
    )

    dims = ("instrument", "visit", "detector")
    data_id = DataCoordinate.standardize(
        {
            "instrument": DP2_VISIT_DETECTOR_DATA_ID["instrument"],
            "visit": DP2_VISIT_DETECTOR_DATA_ID["visit"],
            "detector": DP2_VISIT_DETECTOR_DATA_ID["detector"],
        },
        universe=butler.dimensions,
    )
    legacy_dt = DatasetType("legacy_difference_image", dims, "ExposureF", universe=butler.dimensions)
    new_dt = DatasetType("difference_image", dims, "DifferenceImage", universe=butler.dimensions)
    reg.registerDatasetType(legacy_dt)
    reg.registerDatasetType(new_dt)

    butler.ingest(
        FileDataset(src, DatasetRef(legacy_dt, data_id, "run1")),
        FileDataset(converted, DatasetRef(new_dt, data_id, "run1")),
    )

    result = CliRunner().invoke(
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
    assert result.exit_code == 0, result.output

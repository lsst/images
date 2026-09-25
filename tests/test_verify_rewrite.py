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
from typing import Any

import numpy as np
import pytest
from click.testing import CliRunner

from lsst.images import Background, BackgroundMap, Box, DifferenceImage
from lsst.images.cli._main import main
from lsst.images.fields import ChebyshevField
from lsst.images.tests import (
    DP2_VISIT_DETECTOR_DATA_ID,
    get_dp2_exposure_record,
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

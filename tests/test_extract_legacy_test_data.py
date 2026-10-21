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

from pathlib import Path

from click.testing import CliRunner

from lsst.images.cli._main import main


def test_extract_dp2_requires_a_selection(tmp_path: Path) -> None:
    """Test that extract-dp2 refuses to run with every dataset option off,
    since that would extract nothing.
    """
    result = CliRunner().invoke(main, ["extract-test-data", "dp2", "-b", str(tmp_path), "-d", str(tmp_path)])
    assert result.exit_code == 2, result.output
    assert "No datasets selected" in result.output

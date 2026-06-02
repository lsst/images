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

import unittest

from click.testing import CliRunner

from lsst.images.cli import main


class CliSkeletonTestCase(unittest.TestCase):
    """The root group loads and shows help with core deps only."""

    def test_group_help(self) -> None:
        result = CliRunner().invoke(main, ["--help"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("convert", result.output)
        self.assertIn("inspect", result.output)

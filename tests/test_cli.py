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
import tempfile
import unittest

import numpy as np
from click.testing import CliRunner

from lsst.images import Box, Image
from lsst.images import fits as images_fits
from lsst.images import json as images_json
from lsst.images.cli import main


class CliSkeletonTestCase(unittest.TestCase):
    """The root group loads and shows help with core deps only."""

    def test_group_help(self) -> None:
        result = CliRunner().invoke(main, ["--help"])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("convert", result.output)
        self.assertIn("inspect", result.output)


class InspectTestCase(unittest.TestCase):
    """inspect prints schema URL and format version."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.image = Image(np.zeros((4, 4), dtype=np.float32), bbox=Box.factory[0:4, 0:4])

    def test_inspect_fits(self) -> None:
        path = os.path.join(self.tmp, "x.fits")
        images_fits.write(self.image, path)
        result = CliRunner().invoke(main, ["inspect", path])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("https://images.lsst.io/schemas/image-1.0.0", result.output)
        self.assertIn("1", result.output)  # format version

    def test_inspect_json(self) -> None:
        path = os.path.join(self.tmp, "x.json")
        images_json.write(self.image, path)
        result = CliRunner().invoke(main, ["inspect", path])
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIn("image-1.0.0", result.output)
        self.assertIn("n/a", result.output)  # no container format version for JSON

    def test_inspect_unknown_extension(self) -> None:
        path = os.path.join(self.tmp, "x.txt")
        with open(path, "w") as stream:
            stream.write("nope")
        result = CliRunner().invoke(main, ["inspect", path])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn(".fits", result.output)

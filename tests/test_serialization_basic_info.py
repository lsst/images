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
import pydantic

from lsst.images import Box, Image
from lsst.images import json as images_json
from lsst.images.json import JsonInputArchive
from lsst.images.serialization import ArchiveInfo, InputArchive


class ArchiveInfoTestCase(unittest.TestCase):
    """Tests for the ArchiveInfo model and its schema_url parsing."""

    def test_from_schema_url(self) -> None:
        info = ArchiveInfo.from_schema_url(
            "https://images.lsst.io/schemas/visit_image-1.2.3", format_version=1
        )
        self.assertEqual(info.schema_name, "visit_image")
        self.assertEqual(info.schema_version, "1.2.3")
        self.assertEqual(info.schema_url, "https://images.lsst.io/schemas/visit_image-1.2.3")
        self.assertEqual(info.format_version, 1)

    def test_from_schema_url_none_format(self) -> None:
        info = ArchiveInfo.from_schema_url("https://images.lsst.io/schemas/image-1.0.0", format_version=None)
        self.assertEqual(info.schema_name, "image")
        self.assertIsNone(info.format_version)

    def test_frozen(self) -> None:
        info = ArchiveInfo.from_schema_url("https://x/schemas/image-1.0.0", format_version=None)
        with self.assertRaises(pydantic.ValidationError):
            info.schema_name = "other"  # type: ignore[misc]

    def test_from_schema_url_invalid(self) -> None:
        with self.assertRaises(ValueError):
            ArchiveInfo.from_schema_url("https://images.lsst.io/schemas/noversion", format_version=None)

    def test_get_basic_info_base_raises(self) -> None:
        with self.assertRaises(NotImplementedError):
            InputArchive.get_basic_info("x.fits")


class JsonBasicInfoTestCase(unittest.TestCase):
    """get_basic_info for the JSON backend."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.image = Image(np.zeros((4, 4), dtype=np.float32), bbox=Box.factory[0:4, 0:4])

    def test_json_basic_info(self) -> None:
        path = os.path.join(self.tmp, "x.json")
        images_json.write(self.image, path)
        info = JsonInputArchive.get_basic_info(path)
        self.assertEqual(info.schema_name, "image")
        self.assertEqual(info.schema_version, "1.0.0")
        self.assertEqual(info.schema_url, "https://images.lsst.io/schemas/image-1.0.0")
        self.assertIsNone(info.format_version)

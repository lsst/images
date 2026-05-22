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
import warnings

from lsst.images import VisitImageSerializationModel
from lsst.images.cells import CellCoaddSerializationModel


class JsonSchemaTestCase(unittest.TestCase):
    """Test that pydantic JSON Schema generation succeeds without warnings
    for the complex serialization models that compose many helpers from
    `_asdf_utils` (Quantity, Unit, Time, InlineArray) and `_geom` (Box,
    Interval).
    """

    def _check(self, model: type, mode: str) -> None:
        with warnings.catch_warnings():
            # Any warning emitted during schema generation (e.g.
            # PydanticJsonSchemaWarning for non-serializable defaults) is
            # treated as a test failure.
            warnings.simplefilter("error")
            schema = model.model_json_schema(mode=mode)
        self.assertIsInstance(schema, dict)
        self.assertEqual(schema.get("type"), "object")
        self.assertIn("properties", schema)

    def test_visit_image(self) -> None:
        for mode in ("validation", "serialization"):
            with self.subTest(mode=mode):
                self._check(VisitImageSerializationModel, mode)

    def test_cell_coadd(self) -> None:
        for mode in ("validation", "serialization"):
            with self.subTest(mode=mode):
                self._check(CellCoaddSerializationModel, mode)


if __name__ == "__main__":
    unittest.main()

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

from lsst.images.serialization import class_for_schema


class ClassForSchemaTestCase(unittest.TestCase):
    """class_for_schema returns None for unknown (name, version)."""

    def test_unknown_returns_none(self) -> None:
        self.assertIsNone(class_for_schema("does-not-exist", "1.0.0"))


if __name__ == "__main__":
    unittest.main()

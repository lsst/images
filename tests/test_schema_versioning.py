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

from lsst.images.serialization import ArchiveReadError
from lsst.images.serialization._common import _check_compat, _check_format_version


class CheckCompatTestCase(unittest.TestCase):
    """Tests for the _check_compat and _check_format_version helpers."""

    def test_silent_when_min_read_satisfied(self):
        # min_read_version equals reader major: silent.
        _check_compat("foo", "1.0.0", 1, "1.0.0")

    def test_silent_when_on_disk_major_is_lower(self):
        # 1.0.0 file with min_read_version=1 read by 2.0.0 code: silent.
        _check_compat("foo", "1.0.0", 1, "2.0.0")

    def test_silent_when_on_disk_major_is_higher_but_min_read_low(self):
        # 2.0.0 file declares it is safe for major-1 readers: silent.
        _check_compat("foo", "2.0.0", 1, "1.0.0")

    def test_raises_when_min_read_exceeds_reader_major(self):
        with self.assertRaises(ArchiveReadError) as ctx:
            _check_compat("foo", "2.0.0", 2, "1.0.0")
        self.assertIn("foo", str(ctx.exception))
        self.assertIn(">= 2", str(ctx.exception))

    def test_format_version_silent_when_equal(self):
        _check_format_version("fits", 1, 1)

    def test_format_version_silent_when_on_disk_lower(self):
        _check_format_version("fits", 1, 2)

    def test_format_version_raises_when_on_disk_higher(self):
        with self.assertRaises(ArchiveReadError):
            _check_format_version("fits", 2, 1)


if __name__ == "__main__":
    unittest.main()

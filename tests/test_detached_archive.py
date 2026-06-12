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

from lsst.images.serialization import ArchiveAccessRequiredError, ArchiveReadError


class ArchiveAccessRequiredErrorTestCase(unittest.TestCase):
    """Tests for the ArchiveAccessRequiredError hierarchy."""

    def test_exception_hierarchy(self) -> None:
        self.assertTrue(issubclass(ArchiveAccessRequiredError, RuntimeError))
        # This is a control-flow signal, not a corrupt-file diagnosis: it
        # must never be swallowed by 'except ArchiveReadError' handlers
        # (e.g. the deferred-PSF handling in VisitImage full reads).
        self.assertFalse(issubclass(ArchiveAccessRequiredError, ArchiveReadError))


if __name__ == "__main__":
    unittest.main()

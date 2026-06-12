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

from lsst.images.serialization import (
    ArchiveAccessRequiredError,
    ArchiveReadError,
    ArchiveTree,
    DetachedArchive,
)


class ArchiveAccessRequiredErrorTestCase(unittest.TestCase):
    """Tests for the ArchiveAccessRequiredError hierarchy."""

    def test_exception_hierarchy(self) -> None:
        self.assertTrue(issubclass(ArchiveAccessRequiredError, RuntimeError))
        # This is a control-flow signal, not a corrupt-file diagnosis: it
        # must never be swallowed by 'except ArchiveReadError' handlers
        # (e.g. the deferred-PSF handling in VisitImage full reads).
        self.assertFalse(issubclass(ArchiveAccessRequiredError, ArchiveReadError))


class DetachedArchiveTestCase(unittest.TestCase):
    """Every data-access method of DetachedArchive raises."""

    def setUp(self) -> None:
        self.archive = DetachedArchive()

    def test_deserialize_pointer_raises(self) -> None:
        with self.assertRaises(ArchiveAccessRequiredError):
            self.archive.deserialize_pointer(None, ArchiveTree, lambda model, archive: None)

    def test_get_frame_set_raises(self) -> None:
        with self.assertRaises(ArchiveAccessRequiredError):
            self.archive.get_frame_set(None)

    def test_get_array_raises(self) -> None:
        with self.assertRaises(ArchiveAccessRequiredError):
            self.archive.get_array(None)

    def test_get_table_raises(self) -> None:
        with self.assertRaises(ArchiveAccessRequiredError):
            self.archive.get_table(None)

    def test_get_structured_array_raises(self) -> None:
        with self.assertRaises(ArchiveAccessRequiredError):
            self.archive.get_structured_array(None)

    def test_get_opaque_metadata_is_none(self) -> None:
        # A detached probe has no file to take opaque metadata from.
        self.assertIsNone(self.archive.get_opaque_metadata())


if __name__ == "__main__":
    unittest.main()

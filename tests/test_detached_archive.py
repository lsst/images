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

from lsst.images.serialization import (
    ArchiveAccessRequiredError,
    ArchiveReadError,
    ArchiveTree,
    DetachedArchive,
    InvalidComponentError,
    read,
    write,
)
from lsst.images.serialization import open as ser_open

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "schema_v1")

FREE_COMPONENTS = (
    "sky_projection",
    "psf",
    "obs_info",
    "summary_stats",
    "detector",
    "aperture_corrections",
    "backgrounds",
    "band",
    "bbox",
)

PIXEL_COMPONENTS = ("image", "mask", "variance")


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


class ComponentProbeTestCase(unittest.TestCase):
    """Which VisitImage components deserialize without file access.

    The fixture has a Gaussian PSF and inline sky-projection mappings, so
    both probe as free; PSF models and transforms that use archive pointers
    (e.g. Piff PSFs, frame-set references) take the deserialize_pointer /
    get_frame_set paths instead and would raise.
    """

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmpdir = tmp.name
        self.visit_image = read(os.path.join(DATA_DIR, "visit_image.json"))
        self.archive = DetachedArchive()

    def _get_tree(self, extension: str) -> ArchiveTree:
        """Write the fixture in the given format and return its tree.

        The tree is deliberately used after the reader is closed, mirroring
        how the formatter cache holds a tree with no open file.
        """
        path = os.path.join(self.tmpdir, "visit_image" + extension)
        write(self.visit_image, path)
        with ser_open(path) as reader:
            return reader.get_tree()

    def test_free_components(self) -> None:
        tree = self._get_tree(".fits")
        for component in FREE_COMPONENTS:
            with self.subTest(component=component):
                value = tree.deserialize_component(component, self.archive)
                self.assertIsNotNone(value)

    def test_pixel_components_need_file(self) -> None:
        tree = self._get_tree(".fits")
        for component in PIXEL_COMPONENTS:
            with self.subTest(component=component):
                with self.assertRaises(ArchiveAccessRequiredError):
                    tree.deserialize_component(component, self.archive)

    def test_json_pixel_components_need_file(self) -> None:
        # Inline arrays in a JSON tree still go through archive.get_array,
        # so pixel components fall back to the file even for .json.
        tree = self._get_tree(".json")
        with self.assertRaises(ArchiveAccessRequiredError):
            tree.deserialize_component("image", self.archive)

    def test_invalid_component_propagates(self) -> None:
        tree = self._get_tree(".fits")
        with self.assertRaises(InvalidComponentError):
            tree.deserialize_component("not_a_component", self.archive)


if __name__ == "__main__":
    unittest.main()

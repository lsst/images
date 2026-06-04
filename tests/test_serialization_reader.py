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

from lsst.images import fits as images_fits
from lsst.images.fits import FitsInputArchive
from lsst.images.serialization import ArchiveTree, backend_for_path, class_for_schema, read

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "schema_v1")


def _visit_image():
    """Load a VisitImage from the committed JSON fixture."""
    return read(os.path.join(DATA_DIR, "visit_image.json")).deserialized


class FitsOpenTreeTestCase(unittest.TestCase):
    """InputArchive.open_tree yields a live (archive, tree) pair."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = os.path.join(tmp.name, "v.fits")
        images_fits.write(_visit_image(), self.path)

    def test_open_tree_yields_archive_and_tree(self) -> None:
        info = backend_for_path(self.path).input_archive.get_basic_info(self.path)
        tree_cls = class_for_schema(info.schema_name)
        assert tree_cls is not None
        with FitsInputArchive.open_tree(self.path, tree_cls) as (archive, tree):
            self.assertIsInstance(tree, ArchiveTree)
            proj = tree.deserialize_component("projection", archive)
            self.assertIsNotNone(proj)

    def test_read_still_works(self) -> None:
        # read() routes through read_tree, which now sits on open_tree.
        result = read(self.path)
        self.assertEqual(type(result.deserialized).__name__, "VisitImage")


try:
    import h5py  # noqa: F401

    from lsst.images import ndf as images_ndf
    from lsst.images.ndf import NdfInputArchive

    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False


@unittest.skipUnless(HAVE_H5PY, "h5py is not available.")
class NdfOpenTreeTestCase(unittest.TestCase):
    """open_tree works for the NDF backend."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = os.path.join(tmp.name, "v.sdf")
        images_ndf.write(_visit_image(), self.path)

    def test_open_tree_yields_archive_and_tree(self) -> None:
        info = backend_for_path(self.path).input_archive.get_basic_info(self.path)
        tree_cls = class_for_schema(info.schema_name)
        assert tree_cls is not None
        with NdfInputArchive.open_tree(self.path, tree_cls) as (archive, tree):
            self.assertIsInstance(tree, ArchiveTree)
            self.assertIsNotNone(tree.deserialize_component("obs_info", archive))

    def test_read_still_works(self) -> None:
        self.assertEqual(type(read(self.path).deserialized).__name__, "VisitImage")


if __name__ == "__main__":
    unittest.main()

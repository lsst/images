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

import builtins
import contextlib
import os
import tempfile
import unittest

from lsst.images import fits as images_fits
from lsst.images import json as images_json
from lsst.images.fits import FitsInputArchive
from lsst.images.json import JsonInputArchive
from lsst.images.serialization import ArchiveTree, read_archive


@contextlib.contextmanager
def count_opens(path: str):
    """Count how many times ``path`` is physically opened for reading.

    Yields a one-element list whose single entry is the running open count;
    read it after the ``with`` block.
    """
    count = [0]
    real_open = builtins.open

    def counting_open(file, *args, **kwargs):
        if isinstance(file, (str, bytes, os.PathLike)) and os.fspath(file) == path:
            count[0] += 1
        return real_open(file, *args, **kwargs)

    builtins.open = counting_open
    try:
        yield count
    finally:
        builtins.open = real_open


DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "schema_v1")


def _visit_image():
    """Load a VisitImage from the committed JSON fixture."""
    return read_archive(os.path.join(DATA_DIR, "visit_image.json"))


class FitsOpenTreeTestCase(unittest.TestCase):
    """InputArchive.open_tree yields a live (archive, tree) pair."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = os.path.join(tmp.name, "v.fits")
        images_fits.write(_visit_image(), self.path)

    def test_open_tree_yields_archive_tree_and_info(self) -> None:
        with FitsInputArchive.open_tree(self.path) as (archive, tree, info):
            self.assertIsInstance(tree, ArchiveTree)
            self.assertEqual(info.schema_name, "visit_image")
            proj = tree.deserialize_component("sky_projection", archive)
            self.assertIsNotNone(proj)

    def test_read_still_works(self) -> None:
        # read_archive() returns the deserialized object directly, via
        # open_archive().
        result = read_archive(self.path)
        self.assertEqual(type(result).__name__, "VisitImage")


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

    def test_open_tree_yields_archive_tree_and_info(self) -> None:
        with NdfInputArchive.open_tree(self.path) as (archive, tree, info):
            self.assertIsInstance(tree, ArchiveTree)
            self.assertEqual(info.schema_name, "visit_image")
            self.assertIsNotNone(tree.deserialize_component("obs_info", archive))

    def test_read_still_works(self) -> None:
        self.assertEqual(type(read_archive(self.path)).__name__, "VisitImage")


class JsonOpenTreeTestCase(unittest.TestCase):
    """open_tree works for the JSON backend."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = os.path.join(tmp.name, "v.json")
        images_json.write(_visit_image(), self.path)

    def test_open_tree_yields_archive_tree_and_info(self) -> None:
        with JsonInputArchive.open_tree(self.path) as (archive, tree, info):
            self.assertIsInstance(tree, ArchiveTree)
            self.assertEqual(info.schema_name, "visit_image")
            self.assertIsNotNone(tree.deserialize_component("sky_projection", archive))

    def test_read_still_works(self) -> None:
        self.assertEqual(type(read_archive(self.path)).__name__, "VisitImage")


class ReaderApiTestCase(unittest.TestCase):
    """The user-facing serialization.open_archive() / Reader interface."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = tmp.name
        self.vi = _visit_image()
        self.fits = os.path.join(self.tmp, "v.fits")
        images_fits.write(self.vi, self.fits)

    def _check_components_and_read(self, path: str) -> None:
        import lsst.images.serialization as ser

        with ser.open_archive(path) as reader:
            self.assertIsNotNone(reader.get_component("sky_projection"))
            self.assertIsNotNone(reader.get_component("obs_info"))
            full = reader.read()
            self.assertEqual(type(full).__name__, "VisitImage")

    def test_components_and_read_fits(self) -> None:
        self._check_components_and_read(self.fits)

    def test_components_and_read_json(self) -> None:
        path = os.path.join(self.tmp, "v.json")
        images_json.write(self.vi, path)
        self._check_components_and_read(path)

    @unittest.skipUnless(HAVE_H5PY, "h5py is not available.")
    def test_components_and_read_ndf(self) -> None:
        path = os.path.join(self.tmp, "v.sdf")
        images_ndf.write(self.vi, path)
        self._check_components_and_read(path)

    def test_info(self) -> None:
        import lsst.images.serialization as ser

        with ser.open_archive(self.fits) as reader:
            self.assertEqual(reader.info.schema_name, "visit_image")
            self.assertEqual(reader.info.schema_version, "1.0.0")
            self.assertIsInstance(reader.metadata, dict)

    def test_cls_match(self) -> None:
        import lsst.images.serialization as ser
        from lsst.images import VisitImage

        with ser.open_archive(self.fits, cls=VisitImage) as reader:
            self.assertIsInstance(reader.read(), VisitImage)

    def test_cls_mismatch_raises(self) -> None:
        import lsst.images.serialization as ser
        from lsst.images import Mask

        with self.assertRaises(TypeError):
            with ser.open_archive(self.fits, cls=Mask):
                pass

    def test_unknown_component(self) -> None:
        import lsst.images.serialization as ser
        from lsst.images.serialization import InvalidComponentError

        with ser.open_archive(self.fits) as reader:
            with self.assertRaises(InvalidComponentError):
                reader.get_component("does_not_exist")

    def test_use_after_close_raises(self) -> None:
        import lsst.images.serialization as ser

        with ser.open_archive(self.fits) as reader:
            pass
        with self.assertRaises(RuntimeError):
            reader.get_component("sky_projection")

    def test_fits_open_reads_file_once(self) -> None:
        # open() must not open the file a second time just to read the schema
        # from the primary header: the archive it opens already parses that
        # header, so the schema is identified from a single open.
        import lsst.images.serialization as ser

        with count_opens(self.fits) as count:
            with ser.open_archive(self.fits) as reader:
                reader.get_component("sky_projection")
                reader.get_component("obs_info")
        self.assertEqual(count[0], 1)


if __name__ == "__main__":
    unittest.main()

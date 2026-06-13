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

from lsst.images import Image
from lsst.images.serialization import ArchiveReadError

try:
    import h5py

    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False


def _write_simple_image_ndf(path: str) -> None:
    """Write a tiny Image to ``path`` as an NDF."""
    # Imported here rather than at module scope because importing
    # ``lsst.images.ndf`` hard-requires the optional ``h5py``; this helper is
    # only reached from tests that already skip when h5py is missing.
    from lsst.images.ndf import write as ndf_write

    image = Image(0.0, shape=(4, 4), dtype="float32")
    ndf_write(image, path)


@unittest.skipUnless(HAVE_H5PY, "NDF backend requires h5py")
class NdfFormatVersionTestCase(unittest.TestCase):
    """Tests for the NDF DATA_MODEL and FORMAT_VERSION components."""

    def test_write_emits_data_model_and_format_version(self) -> None:
        """A freshly-written NDF carries DATA_MODEL and FORMAT_VERSION."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.sdf")
            _write_simple_image_ndf(path)
            with h5py.File(path, "r") as f:
                self.assertIn("FORMAT_VERSION", f["/MORE/LSST"])
                self.assertIn("DATA_MODEL", f["/MORE/LSST"])

    def test_read_succeeds_when_format_version_matches(self) -> None:
        """A freshly-written NDF reads successfully."""
        from lsst.images.ndf import NdfInputArchive

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.sdf")
            _write_simple_image_ndf(path)
            with NdfInputArchive.open(path):
                pass

    def test_read_fails_when_format_version_too_high(self) -> None:
        """A file with a newer FORMAT_VERSION raises ArchiveReadError."""
        from lsst.images.ndf import NdfInputArchive

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.sdf")
            _write_simple_image_ndf(path)
            with h5py.File(path, "r+") as f:
                if "FORMAT_VERSION" in f["/MORE/LSST"]:
                    del f["/MORE/LSST/FORMAT_VERSION"]
                f["/MORE/LSST"].create_dataset("FORMAT_VERSION", data=np.int32(2))
            with self.assertRaises(ArchiveReadError):
                with NdfInputArchive.open(path):
                    pass

    def test_read_succeeds_when_format_version_absent(self) -> None:
        """A legacy file lacking FORMAT_VERSION reads (defaults to 1)."""
        from lsst.images.ndf import NdfInputArchive

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.sdf")
            _write_simple_image_ndf(path)
            with h5py.File(path, "r+") as f:
                if "FORMAT_VERSION" in f["/MORE/LSST"]:
                    del f["/MORE/LSST/FORMAT_VERSION"]
                if "DATA_MODEL" in f["/MORE/LSST"]:
                    del f["/MORE/LSST/DATA_MODEL"]
            with NdfInputArchive.open(path):
                pass


if __name__ == "__main__":
    unittest.main()

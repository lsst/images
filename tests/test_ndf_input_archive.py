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

import tempfile
import unittest

import numpy as np

from lsst.images import Box, Image
from lsst.images.ndf._input_archive import NdfInputArchive
from lsst.images.ndf._output_archive import write


class NdfInputArchiveOpenTestCase(unittest.TestCase):
    """Tests for `NdfInputArchive.open` and `get_tree`."""

    def test_open_round_trips_image_tree(self):
        image = Image(
            np.arange(20, dtype=np.float32).reshape(4, 5),
            bbox=Box.factory[10:14, 20:25],
        )
        with tempfile.NamedTemporaryFile(suffix=".sdf", delete_on_close=False) as tmp:
            tmp.close()
            written_tree = write(image, tmp.name)
            with NdfInputArchive.open(tmp.name) as archive:
                tree = archive.get_tree(type(written_tree))
                self.assertEqual(tree.model_dump_json(), written_tree.model_dump_json())

    def test_get_tree_raises_when_main_json_missing(self):
        # A file with no /MORE/LSST/JSON should raise ArchiveReadError.
        import h5py

        from lsst.images.serialization import ArchiveReadError

        with tempfile.NamedTemporaryFile(suffix=".sdf", delete_on_close=False) as tmp:
            tmp.close()
            with h5py.File(tmp.name, "w") as f:
                f["/"].attrs["CLASS"] = "NDF"
            with NdfInputArchive.open(tmp.name) as archive:
                # type(written_tree) doesn't exist in scope here, but get_tree
                # raises before the type matters.
                from lsst.images._image import ImageSerializationModel
                from lsst.images.ndf._common import NdfPointerModel

                model_type = ImageSerializationModel[NdfPointerModel]
                with self.assertRaises(ArchiveReadError):
                    archive.get_tree(model_type)

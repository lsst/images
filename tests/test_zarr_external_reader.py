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

from lsst.images import Box, Image

try:
    import zarr  # noqa: F401

    from lsst.images.zarr import write

    HAVE_ZARR = True
except ImportError:
    HAVE_ZARR = False

try:
    import ome_zarr
    import ome_zarr.io
    import ome_zarr.reader  # noqa: F401

    HAVE_OME_ZARR = True
except ImportError:
    HAVE_OME_ZARR = False


@unittest.skipUnless(HAVE_ZARR and HAVE_OME_ZARR, "ome-zarr is not installed")
class OmeZarrReaderTestCase(unittest.TestCase):
    """``ome-zarr-py`` can open archives written by ``lsst.images.zarr``."""

    def test_ome_zarr_can_open_image(self) -> None:
        from ome_zarr.io import parse_url
        from ome_zarr.reader import Reader

        original = Image(
            np.arange(20, dtype=np.float32).reshape(4, 5),
            bbox=Box.factory[10:14, 20:25],
        )
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "out.zarr")
            write(original, target)
            location = parse_url(target)
            self.assertIsNotNone(location)
            reader = Reader(location)
            nodes = list(reader())
            self.assertGreaterEqual(len(nodes), 1)
            data = nodes[0].data[0]  # level 0
            self.assertEqual(tuple(data.shape), (4, 5))


if __name__ == "__main__":
    unittest.main()

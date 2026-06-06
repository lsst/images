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

try:
    import zarr

    from lsst.images.zarr._store import open_store_for_read, open_store_for_write

    HAVE_ZARR = True
except ImportError:
    HAVE_ZARR = False


@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class StoreDispatchTestCase(unittest.TestCase):
    """URI-based dispatch for zarr stores."""

    def test_local_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "out.zarr")
            with open_store_for_write(target) as store:
                self.assertIsInstance(store, zarr.storage.LocalStore)
                zarr.create_group(store=store, zarr_format=3)
            with open_store_for_read(target) as store:
                self.assertIsInstance(store, zarr.storage.LocalStore)
                root = zarr.open_group(store=store, mode="r")
                self.assertEqual(list(root.keys()), [])

    def test_zip_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "out.zarr.zip")
            with open_store_for_write(target) as store:
                self.assertIsInstance(store, zarr.storage.ZipStore)
                zarr.create_group(store=store, zarr_format=3)
            with open_store_for_read(target) as store:
                self.assertIsInstance(store, zarr.storage.ZipStore)

    def test_create_only_refuses_existing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "out.zarr")
            with open_store_for_write(target) as store:
                zarr.create_group(store=store, zarr_format=3)
            with self.assertRaisesRegex(OSError, "already exists"):
                with open_store_for_write(target):
                    pass

    def test_zip_store_round_trips_sharded_array(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "out.zarr.zip")
            data = np.arange(300 * 300, dtype=np.float32).reshape(300, 300)
            with open_store_for_write(target) as store:
                group = zarr.create_group(store=store, zarr_format=3)
                arr = group.create_array(
                    name="image",
                    shape=data.shape,
                    chunks=(256, 256),
                    shards=(512, 512),
                    dtype=data.dtype,
                )
                arr[:] = data
            with open_store_for_read(target) as store:
                group = zarr.open_group(store=store, mode="r", zarr_format=3)
                image = group["image"]
                self.assertEqual(tuple(image.chunks), (256, 256))
                self.assertEqual(tuple(image.shards), (512, 512))
                np.testing.assert_array_equal(image[...], data)


if __name__ == "__main__":
    unittest.main()

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

from pathlib import Path

import numpy as np
import pytest

try:
    import zarr

    from lsst.images.zarr._store import open_store_for_read, open_store_for_write

    HAVE_ZARR = True
except ImportError:
    HAVE_ZARR = False

skip_no_zarr = pytest.mark.skipif(not HAVE_ZARR, reason="zarr is not installed")


@skip_no_zarr
def test_local_directory(tmp_path: Path) -> None:
    """Verify a local directory path dispatches to LocalStore."""
    target = tmp_path / "out.zarr"
    with open_store_for_write(target) as store:
        assert isinstance(store, zarr.storage.LocalStore)
        zarr.create_group(store=store, zarr_format=3)
    with open_store_for_read(target) as store:
        assert isinstance(store, zarr.storage.LocalStore)
        root = zarr.open_group(store=store, mode="r")
        assert list(root.keys()) == []


@skip_no_zarr
def test_zip_store(tmp_path: Path) -> None:
    """Verify a .zarr.zip path dispatches to ZipStore."""
    target = tmp_path / "out.zarr.zip"
    with open_store_for_write(target) as store:
        assert isinstance(store, zarr.storage.ZipStore)
        zarr.create_group(store=store, zarr_format=3)
    with open_store_for_read(target) as store:
        assert isinstance(store, zarr.storage.ZipStore)


@skip_no_zarr
def test_create_only_refuses_existing(tmp_path: Path) -> None:
    """Verify opening for write refuses to overwrite an existing store."""
    target = tmp_path / "out.zarr"
    with open_store_for_write(target) as store:
        zarr.create_group(store=store, zarr_format=3)
    with pytest.raises(OSError, match="already exists"):
        with open_store_for_write(target):
            pass


@skip_no_zarr
def test_zip_store_round_trips_sharded_array(tmp_path: Path) -> None:
    """Verify a sharded array round-trips through a ZipStore."""
    target = tmp_path / "out.zarr.zip"
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
        assert isinstance(image, zarr.Array)
        assert image.chunks == (256, 256)
        assert image.shards == (512, 512)
        np.testing.assert_array_equal(image[...], data)

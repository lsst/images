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

__all__ = ("open_store_for_read", "open_store_for_write")

import os
from collections.abc import Iterator
from contextlib import contextmanager

import zarr
from zarr.abc.store import Store

from lsst.resources import ResourcePath, ResourcePathExpression


def _is_zip(rp: ResourcePath) -> bool:
    return rp.path.endswith(".zarr.zip") or rp.path.endswith(".zip")


def _is_remote(rp: ResourcePath) -> bool:
    return rp.scheme not in ("", "file")


@contextmanager
def open_store_for_write(path: ResourcePathExpression) -> Iterator[Store]:
    """Open a zarr store for writing.

    Refuses to overwrite a non-empty existing store. The returned
    context manager closes the store on exit; for ``ZipStore`` this
    finalizes the central directory.
    """
    rp = ResourcePath(path)
    store: Store
    if _is_zip(rp):
        if _is_remote(rp):
            raise NotImplementedError("Remote ZipStore writes are a follow-up.")
        local = rp.ospath
        if os.path.exists(local) and os.path.getsize(local) > 0:
            raise OSError(f"File {local!r} already exists.")
        zip_store = zarr.storage.ZipStore(local, mode="w")
        try:
            yield zip_store
        finally:
            if getattr(zip_store, "_is_open", False):
                zip_store.close()
        return
    if _is_remote(rp):
        import fsspec

        fs, fs_path = fsspec.url_to_fs(str(rp))
        if fs.exists(fs_path) and fs.ls(fs_path):
            raise OSError(f"Store {rp!s} already exists.")
        store = zarr.storage.FsspecStore(fs=fs, path=fs_path, read_only=False)
        yield store
        return
    local = rp.ospath
    if os.path.exists(local) and os.listdir(local):
        raise OSError(f"Directory {local!r} already exists and is non-empty.")
    os.makedirs(local, exist_ok=True)
    store = zarr.storage.LocalStore(local, read_only=False)
    yield store


@contextmanager
def open_store_for_read(path: ResourcePathExpression) -> Iterator[Store]:
    """Open a zarr store for reading."""
    rp = ResourcePath(path)
    store: Store
    if _is_zip(rp):
        if _is_remote(rp):
            with rp.as_local() as local:
                zip_store = zarr.storage.ZipStore(local.ospath, mode="r")
                try:
                    yield zip_store
                finally:
                    if getattr(zip_store, "_is_open", False):
                        zip_store.close()
            return
        zip_store = zarr.storage.ZipStore(rp.ospath, mode="r")
        try:
            yield zip_store
        finally:
            if getattr(zip_store, "_is_open", False):
                zip_store.close()
        return
    if _is_remote(rp):
        import fsspec

        fs, fs_path = fsspec.url_to_fs(str(rp))
        store = zarr.storage.FsspecStore(fs=fs, path=fs_path, read_only=True)
        yield store
        return
    store = zarr.storage.LocalStore(rp.ospath, read_only=True)
    yield store

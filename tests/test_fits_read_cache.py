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
"""The FITS partial-read path should configure fsspec's block cache so remote
reads coalesce into a small number of larger range requests.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from contextlib import contextmanager
from typing import Any
from unittest import mock

from fsspec.spec import AbstractBufferedFile, AbstractFileSystem

from lsst.images import Image
from lsst.images.fits import FitsInputArchive
from lsst.images.fits._input_archive import _DEFAULT_PAGE_SIZE
from lsst.images.serialization import write


def _write_simple_image_fits(path: str) -> None:
    """Write a tiny Image to ``path`` via the high-level API."""
    write(Image(0.0, shape=(4, 4), dtype="float32"), path)


class _RecordingFile(AbstractBufferedFile):
    """A buffered file backed by an in-memory blob that records the cache it
    was constructed with.
    """

    def _fetch_range(self, start: int, end: int) -> bytes:
        return self.fs.blob[start:end]


class _RecordingFS(AbstractFileSystem):
    """A minimal buffered filesystem serving one in-memory blob.

    Unlike `LocalFileSystem`, this produces a real `AbstractBufferedFile`, so
    the ``cache_type`` / ``block_size`` passed to ``open`` take effect and can
    be inspected -- which is what matters for remote stores like GCS.
    """

    protocol = "recording"
    cachable = False  # avoid fsspec's instance cache leaking state between opens

    def __init__(self, blob: bytes, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.blob = blob
        # (cache class name, blocksize) recorded at open time, one per open.
        self.opened: list[tuple[str, int]] = []

    def info(self, path: str, **kwargs: Any) -> dict[str, Any]:
        return {"name": path, "size": len(self.blob), "type": "file"}

    def _open(
        self,
        path: str,
        mode: str = "rb",
        block_size: int | None = None,
        cache_options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> _RecordingFile:
        handle = _RecordingFile(
            self, path, mode=mode, block_size=block_size, cache_options=cache_options, **kwargs
        )
        # Record the cache configuration now: the archive closes the file at
        # context exit, which clears ``.cache``.
        self.opened.append((type(handle.cache).__name__, handle.blocksize))
        return handle


@contextmanager
def _route_through(fs: _RecordingFS):
    """Make ``ResourcePath.to_fsspec`` return ``fs`` so the *real*
    `FitsInputArchive.open` opens our recording filesystem with whatever
    ``block_size`` / ``cache_type`` it chooses.

    A local path is a `FileResourcePath`, which overrides ``to_fsspec``; patch
    it there so the archive's ``path.to_fsspec()`` call is intercepted.
    """
    from lsst.resources.file import FileResourcePath

    with mock.patch.object(FileResourcePath, "to_fsspec", return_value=(fs, "blob")):
        yield


class FitsReadCacheTestCase(unittest.TestCase):
    """The partial FITS read path uses a bounded block cache with a tunable,
    documented block size.
    """

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = os.path.join(tmp.name, "x.fits")
        _write_simple_image_fits(self.path)
        with open(self.path, "rb") as handle:
            self.blob = handle.read()

    def test_default_page_size_is_documented_constant(self) -> None:
        # The block size lives in one place and is a multiple of the FITS
        # block (2880 bytes).
        self.assertEqual(_DEFAULT_PAGE_SIZE % 2880, 0)

    def test_partial_open_uses_block_cache(self) -> None:
        fs = _RecordingFS(self.blob)
        with _route_through(fs):
            with FitsInputArchive.open(self.path, partial=True):
                pass
        self.assertEqual(len(fs.opened), 1)
        cache_name, blocksize = fs.opened[0]
        self.assertEqual(cache_name, "BlockCache")
        self.assertEqual(blocksize, _DEFAULT_PAGE_SIZE)


if __name__ == "__main__":
    unittest.main()

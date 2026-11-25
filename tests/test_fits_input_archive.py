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
"""Tests for reading arrays from a FITS archive in several threads."""

from __future__ import annotations

import io
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import EllipsisType
from typing import Any, ClassVar

import astropy.io.fits
import numpy as np
import pytest

from lsst.images.fits import FitsInputArchive, FitsOutputArchive
from lsst.images.serialization import ArchiveTree, ArrayReferenceModel, InputArchive


class _TinyTree(ArchiveTree):
    """Minimal concrete ArchiveTree for low-level archive writes."""

    SCHEMA_NAME: ClassVar[str] = "test_fits_input_archive"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"
    MIN_READ_VERSION: ClassVar[int] = 1
    PUBLIC_TYPE: ClassVar[type] = object

    def deserialize(
        self, archive: InputArchive[Any], **kwargs: Any
    ) -> _TinyTree:  # pragma: no cover - never invoked
        raise NotImplementedError()


class _YieldingStream(io.BytesIO):
    """An in-memory stream that lets other threads run after every seek.

    A read from a real file or a remote object releases the GIL, so another
    thread can move a shared stream between one thread's seek and the read
    that follows it.  Sleeping here makes that interleaving happen every
    time instead of occasionally.
    """

    def seek(self, offset: int, whence: int = os.SEEK_SET, /) -> int:
        position = super().seek(offset, whence)
        time.sleep(0.001)
        return position


def _write_arrays(
    arrays: list[np.ndarray], tmp_path: Path, *, compressed: bool = True
) -> tuple[Path, list[ArrayReferenceModel]]:
    """Write each array to its own HDU of a new archive."""
    filename = tmp_path / "arrays.fits"
    names = [f"array{n}" for n in range(len(arrays))]
    compression_options = None if compressed else dict.fromkeys(names)
    with FitsOutputArchive.open(filename, compression_options=compression_options) as archive:
        models = [archive.add_array(array, name=name) for name, array in zip(names, arrays, strict=True)]
        archive.add_tree(_TinyTree())
    return filename, models


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("dtype", [np.uint16, np.uint32, np.uint64])
def test_full_read_of_unsigned_array(
    tmp_path: Path, dtype: type[np.unsignedinteger], compressed: bool
) -> None:
    """Verify that reading all of an unsigned integer array, which FITS
    stores as signed integers with an offset, returns the written values.
    """
    largest = np.iinfo(dtype).max
    array = np.array([[0, 1], [largest - 1, largest]], dtype=dtype)
    filename, (model,) = _write_arrays([array], tmp_path, compressed=compressed)
    with FitsInputArchive.open(filename) as archive:
        result = archive.get_array(model)
    assert result.dtype == array.dtype
    np.testing.assert_array_equal(result, array)


@pytest.mark.parametrize("slices", [..., (slice(4, 12), slice(8, 24))])
def test_arrays_read_concurrently_from_shared_stream(
    tmp_path: Path, slices: tuple[slice, ...] | EllipsisType
) -> None:
    """Verify that threads reading different HDUs get the right pixels even
    though every HDU is read through the same underlying stream.
    """
    rng = np.random.default_rng(57102)
    arrays = [rng.integers(0, 1000, size=(32, 32), dtype=np.int32) + 1000 * n for n in range(4)]
    filename, models = _write_arrays(arrays, tmp_path)
    start = threading.Barrier(len(models), timeout=10)

    with FitsInputArchive.open(_YieldingStream(filename.read_bytes())) as archive:

        def read(model: ArrayReferenceModel) -> np.ndarray:
            start.wait()
            return archive.get_array(model, slices=slices)

        with ThreadPoolExecutor(max_workers=len(models)) as executor:
            results = list(executor.map(read, models))

    for result, array in zip(results, arrays, strict=True):
        np.testing.assert_array_equal(result, array[slices])


def test_compressed_arrays_decompress_concurrently(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that reading all of one compressed HDU does not have to wait
    for another thread to finish reading all of a different one.
    """
    arrays = [np.arange(64, dtype=np.float32).reshape(8, 8) + n for n in range(2)]
    filename, models = _write_arrays(arrays, tmp_path)
    # Passed only when both threads are decompressing at the same time.
    both_decompressing = threading.Barrier(len(models), timeout=10)
    decompress = astropy.io.fits.CompImageSection.__getitem__

    def decompress_together(section: astropy.io.fits.CompImageSection, index: Any) -> np.ndarray:
        both_decompressing.wait()
        return decompress(section, index)

    monkeypatch.setattr(astropy.io.fits.CompImageSection, "__getitem__", decompress_together)

    with FitsInputArchive.open(filename) as archive:
        with ThreadPoolExecutor(max_workers=len(models)) as executor:
            results = list(executor.map(archive.get_array, models))

    for result, array in zip(results, arrays, strict=True):
        np.testing.assert_array_equal(result, array)


def test_executor_runs_tasks_in_calling_thread_by_default(tmp_path: Path) -> None:
    """Verify that an archive opened without extra workers runs the tasks
    given to its executor in the calling thread.
    """
    filename, _ = _write_arrays([np.zeros((2, 2), dtype=np.float32)], tmp_path)
    with FitsInputArchive.open(filename) as archive, archive.make_executor() as executor:
        future = executor.submit(threading.get_ident)
    assert future.result() == threading.get_ident()


def test_default_executor_reports_task_failure_from_result(tmp_path: Path) -> None:
    """Verify that a task that fails in the calling thread raises when its
    result is requested, as it would had it run in another thread.
    """
    filename, _ = _write_arrays([np.zeros((2, 2), dtype=np.float32)], tmp_path)
    with FitsInputArchive.open(filename) as archive, archive.make_executor() as executor:
        future = executor.submit(int, "not a number")
    with pytest.raises(ValueError, match="not a number"):
        future.result()


def test_executor_runs_tasks_at_the_same_time_when_given_workers(tmp_path: Path) -> None:
    """Verify that an archive opened with several workers runs the tasks
    given to its executor at the same time.
    """
    filename, _ = _write_arrays([np.zeros((2, 2), dtype=np.float32)], tmp_path)
    # Each task returns only once both are running.
    both_running = threading.Barrier(2, timeout=10)
    with FitsInputArchive.open(filename, num_workers=2) as archive, archive.make_executor() as executor:
        futures = [executor.submit(both_running.wait) for _ in range(2)]
    assert sorted(future.result() for future in futures) == [0, 1]

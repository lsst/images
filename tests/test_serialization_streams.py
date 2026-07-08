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
"""Tests for reading archives from in-memory bytes and binary streams."""

from __future__ import annotations

import gzip
import io
import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from lsst.images import Box, Image, Mask
from lsst.images.fits import FitsInputArchive
from lsst.images.json import JsonInputArchive
from lsst.images.serialization import open_archive, read_archive, write_archive

try:
    import h5py  # noqa: F401  -- detect availability for NDF stream skips

    from lsst.images.ndf import NdfInputArchive

    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

try:
    from compression import zstd as _stdlib_zstd  # noqa: F401  -- detect zstd availability

    ZSTD_AVAILABLE = True
except ImportError:
    try:
        import zstandard  # noqa: F401

        ZSTD_AVAILABLE = True
    except ImportError:
        ZSTD_AVAILABLE = False

skip_no_h5py = pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py is not installed")
skip_no_zstd = pytest.mark.skipif(not ZSTD_AVAILABLE, reason="no zstd decompressor available")

LOCAL_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "schema_v1")


def _zstd_compress(data: bytes) -> bytes:
    """Compress with whichever zstd library is available."""
    try:
        from compression import zstd
    except ImportError:
        import zstandard

        return zstandard.ZstdCompressor().compress(data)
    return zstd.compress(data)


def _make_image() -> Image:
    """Return a small float32 Image for stream round-trip tests."""
    return Image(np.arange(16, dtype=np.float32).reshape(4, 4), bbox=Box.factory[0:4, 0:4])


def _serialized_bytes(obj: object, extension: str, tmp_path: Path) -> bytes:
    """Write ``obj`` to a file under ``tmp_path`` and return its content."""
    path = tmp_path / f"x{extension}"
    write_archive(obj, path)
    return path.read_bytes()


def test_fits_open_tree_stream(tmp_path: Path) -> None:
    """Verify FitsInputArchive.open_tree accepts a seekable binary stream."""
    image = _make_image()
    stream = io.BytesIO(_serialized_bytes(image, ".fits", tmp_path))
    with FitsInputArchive.open_tree(stream) as (archive, tree, info):
        assert info.schema_name == "image"
        result = tree.deserialize(archive)
    assert isinstance(result, Image)
    np.testing.assert_array_equal(result.array, image.array)


def test_json_open_tree_stream(tmp_path: Path) -> None:
    """Verify JsonInputArchive.open_tree accepts a seekable binary stream."""
    image = _make_image()
    stream = io.BytesIO(_serialized_bytes(image, ".json", tmp_path))
    with JsonInputArchive.open_tree(stream) as (archive, tree, info):
        result = tree.deserialize(archive)
    assert isinstance(result, Image)
    np.testing.assert_array_equal(result.array, image.array)


@skip_no_h5py
def test_ndf_open_tree_stream(tmp_path: Path) -> None:
    """Verify NdfInputArchive.open_tree accepts a seekable binary stream."""
    image = _make_image()
    stream = io.BytesIO(_serialized_bytes(image, ".sdf", tmp_path))
    with NdfInputArchive.open_tree(stream) as (archive, tree, info):
        result = tree.deserialize(archive)
    assert isinstance(result, Image)
    np.testing.assert_array_equal(result.array, image.array)


def test_read_stream_fits(tmp_path: Path) -> None:
    """Verify read_archive() turns in-memory FITS bytes into an Image."""
    image = _make_image()
    result = read_archive(io.BytesIO(_serialized_bytes(image, ".fits", tmp_path)))
    assert isinstance(result, Image)
    np.testing.assert_array_equal(result.array, image.array)


def test_read_stream_json(tmp_path: Path) -> None:
    """Verify read_archive() turns in-memory JSON bytes into an Image."""
    image = _make_image()
    result = read_archive(io.BytesIO(_serialized_bytes(image, ".json", tmp_path)))
    assert isinstance(result, Image)
    np.testing.assert_array_equal(result.array, image.array)


@skip_no_h5py
def test_read_stream_ndf(tmp_path: Path) -> None:
    """Verify read_archive() turns in-memory NDF bytes into an Image."""
    image = _make_image()
    result = read_archive(io.BytesIO(_serialized_bytes(image, ".sdf", tmp_path)))
    assert isinstance(result, Image)
    np.testing.assert_array_equal(result.array, image.array)


def test_read_stream_cls_match_and_mismatch(tmp_path: Path) -> None:
    """Verify stream reads honor cls= for both matching and mismatched
    types.
    """
    data = _serialized_bytes(_make_image(), ".fits", tmp_path)
    result = read_archive(io.BytesIO(data), cls=Image)
    assert isinstance(result, Image)
    with pytest.raises(TypeError):
        read_archive(io.BytesIO(data), cls=Mask)


def test_read_stream_kwargs_forwarded(tmp_path: Path) -> None:
    """Verify stream reads forward deserialize kwargs like bbox."""
    big = Image(np.arange(64, dtype=np.float32).reshape(8, 8), bbox=Box.factory[0:8, 0:8])
    sub = read_archive(io.BytesIO(_serialized_bytes(big, ".fits", tmp_path)), bbox=Box.factory[2:6, 2:6])
    assert sub.array.shape == (4, 4)
    np.testing.assert_array_equal(sub.array, big.array[2:6, 2:6])


def test_read_stream_format_override(tmp_path: Path) -> None:
    """Verify format= forces the named backend for a stream read."""
    image = _make_image()
    result = read_archive(io.BytesIO(_serialized_bytes(image, ".json", tmp_path)), format="json")
    assert isinstance(result, Image)


def test_read_stream_wrong_format_override(tmp_path: Path) -> None:
    """Verify FITS bytes forced through the JSON backend fail in that
    backend.
    """
    with pytest.raises(ValueError):
        read_archive(io.BytesIO(_serialized_bytes(_make_image(), ".fits", tmp_path)), format="json")


def test_read_stream_unrecognized_bytes() -> None:
    """Verify unrecognized stream content raises ValueError naming known
    formats.
    """
    with pytest.raises(ValueError) as exc_info:
        read_archive(io.BytesIO(b"certainly not a supported format"))
    assert "FITS" in str(exc_info.value)


def test_read_stream_bad_format_name(tmp_path: Path) -> None:
    """Verify an unknown format= name raises ValueError."""
    with pytest.raises(ValueError):
        read_archive(io.BytesIO(_serialized_bytes(_make_image(), ".json", tmp_path)), format="asdf")


def test_read_compressed_stream_raises(tmp_path: Path) -> None:
    """Verify a compressed stream raises ValueError telling the caller to
    decompress.
    """
    # Compressed streams are the caller's responsibility to decompress;
    # the sniff error says what to do.
    data = gzip.compress(_serialized_bytes(_make_image(), ".fits", tmp_path))
    with pytest.raises(ValueError) as exc_info:
        read_archive(io.BytesIO(data))
    assert "gzip" in str(exc_info.value)


def test_open_stream_components(tmp_path: Path) -> None:
    """Verify open_archive() accepts a stream and supports component
    reads.
    """
    visit_image = read_archive(os.path.join(LOCAL_DATA_DIR, "visit_image.json"))
    stream = io.BytesIO(_serialized_bytes(visit_image, ".fits", tmp_path))
    with open_archive(stream) as reader:
        assert reader.info.schema_name == "visit_image"
        assert isinstance(reader.metadata, dict)
        assert reader.get_component("sky_projection") is not None
        full = reader.read()
    assert type(full).__name__ == "VisitImage"


def _write_compressed(
    extension: str, compress: Callable[[bytes], bytes], tmp_path: Path
) -> tuple[Path, Image]:
    """Write a compressed serialized Image under ``tmp_path``.

    Returns the compressed file's path and the image it contains.
    """
    image = _make_image()
    data = compress(_serialized_bytes(image, extension, tmp_path))
    suffix = ".gz" if compress is gzip.compress else ".zst"
    path = tmp_path / f"c{extension}{suffix}"
    path.write_bytes(data)
    return path, image


def test_read_compressed_path_fits_gz(tmp_path: Path) -> None:
    """Verify a .fits.gz path reads transparently through read_archive()."""
    # Regression: .fits.gz was dispatched to the FITS backend but
    # handed to it still compressed.
    path, image = _write_compressed(".fits", gzip.compress, tmp_path)
    result = read_archive(path)
    assert isinstance(result, Image)
    np.testing.assert_array_equal(result.array, image.array)


def test_read_compressed_path_json_gz(tmp_path: Path) -> None:
    """Verify a .json.gz path reads transparently through read_archive()."""
    path, _ = _write_compressed(".json", gzip.compress, tmp_path)
    assert isinstance(read_archive(path), Image)


@skip_no_zstd
def test_read_compressed_path_fits_zst(tmp_path: Path) -> None:
    """Verify a .fits.zst path reads transparently through read_archive()."""
    path, _ = _write_compressed(".fits", _zstd_compress, tmp_path)
    assert isinstance(read_archive(path), Image)


def test_open_compressed_path_fits_gz(tmp_path: Path) -> None:
    """Verify open_archive() reads a .fits.gz path transparently."""
    path, _ = _write_compressed(".fits", gzip.compress, tmp_path)
    with open_archive(path) as reader:
        assert isinstance(reader.read(), Image)

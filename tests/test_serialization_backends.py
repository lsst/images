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

import gzip
import io
import os
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from lsst.images import Box, Image
from lsst.images import fits as images_fits
from lsst.images.serialization import Backend, backend_for_name, backend_for_path, backend_for_stream
from lsst.images.serialization._backends import (
    _decompress_path_to_temp_file,
    _is_binary_stream,
    _path_is_compressed,
)
from lsst.resources import ResourcePath

try:
    from compression import zstd as _stdlib_zstd  # noqa: F401  -- detect zstd availability

    ZSTD_AVAILABLE = True
except ImportError:
    try:
        import zstandard  # noqa: F401

        ZSTD_AVAILABLE = True
    except ImportError:
        ZSTD_AVAILABLE = False


def _zstd_compress(data: bytes) -> bytes:
    """Compress with whichever zstd library is available."""
    try:
        from compression import zstd
    except ImportError:
        import zstandard

        return zstandard.ZstdCompressor().compress(data)
    return zstd.compress(data)


def test_backend_for_path_fits() -> None:
    """Verify that a .fits path resolves to the FITS backend."""
    from lsst.images.fits import FitsInputArchive

    b = backend_for_path("a/b/c.fits")
    assert isinstance(b, Backend)
    assert b.name == "fits"
    assert b.input_archive is FitsInputArchive
    assert callable(b.write)


def test_backend_for_path_fits_gz() -> None:
    """Verify that .fits.gz and URL-form .fits.gz both resolve to the
    FITS backend.
    """
    assert backend_for_path("c.fits.gz").name == "fits"
    assert backend_for_path("file://a/b/c.fits.gz?param=2").name == "fits"


def test_backend_for_path_json() -> None:
    """Verify that a .json path resolves to the JSON backend."""
    from lsst.images.json import JsonInputArchive

    b = backend_for_path("c.json")
    assert b.name == "json"
    assert b.input_archive is JsonInputArchive


def test_backend_for_path_ndf() -> None:
    """Verify that .sdf and .h5 paths both resolve to the NDF backend."""
    assert backend_for_path("c.sdf").name == "ndf"
    assert backend_for_path("c.h5").name == "ndf"


def test_backend_for_path_unknown_raises() -> None:
    """Verify that an unrecognised suffix raises ValueError naming known
    formats.
    """
    with pytest.raises(ValueError) as exc_info:
        backend_for_path("c.txt")
    assert ".fits" in str(exc_info.value)


def test_minify_unsupported_schema_uses_shared_dispatch(tmp_path: Path) -> None:
    """Verify minify resolves backend and schema via the shared APIs.

    Reaching the 'no subsetter' error proves backend_for_path and
    get_basic_info ran and detected schema_name 'image'.
    """
    from lsst.images.tests._minify_for_fixtures import minify

    src = os.path.join(tmp_path, "plain.fits")
    out = os.path.join(tmp_path, "plain.json")
    images_fits.write(Image(np.zeros((4, 4), dtype=np.float32), bbox=Box.factory[0:4, 0:4]), src)
    with pytest.raises(NotImplementedError) as exc_info:
        minify(src, out)
    assert "image" in str(exc_info.value)


def test_backend_for_name() -> None:
    """Verify explicit format names resolve to their backends."""
    assert backend_for_name("fits").name == "fits"
    assert backend_for_name("ndf").name == "ndf"
    assert backend_for_name("json").name == "json"


def test_backend_for_name_unknown_raises() -> None:
    """Verify an unknown format name raises ValueError naming it."""
    with pytest.raises(ValueError) as exc_info:
        backend_for_name("hdf")
    assert "hdf" in str(exc_info.value)


def test_backend_for_stream_fits_magic() -> None:
    """Verify FITS magic bytes resolve to the FITS backend without consuming
    the stream.
    """
    stream = io.BytesIO(b"SIMPLE  =                    T / conforms")
    assert backend_for_stream(stream).name == "fits"
    # Sniffing must not consume the stream.
    assert stream.tell() == 0


def test_backend_for_stream_hdf5_magic() -> None:
    """Verify the HDF5 signature resolves to the NDF backend."""
    stream = io.BytesIO(b"\x89HDF\r\n\x1a\n" + b"\x00" * 8)
    assert backend_for_stream(stream).name == "ndf"


def test_backend_for_stream_json() -> None:
    """Verify JSON content resolves to the JSON backend."""
    assert backend_for_stream(io.BytesIO(b'{"schema_url": "x"}')).name == "json"


def test_backend_for_stream_json_leading_whitespace() -> None:
    """Verify JSON detection tolerates leading whitespace."""
    assert backend_for_stream(io.BytesIO(b'  \n\t {"a": 1}')).name == "json"


def test_backend_for_stream_unknown_raises() -> None:
    """Verify unrecognized content raises ValueError naming known formats."""
    with pytest.raises(ValueError) as exc_info:
        backend_for_stream(io.BytesIO(b"not any known format"))
    msg = str(exc_info.value)
    assert "FITS" in msg
    assert "format" in msg


def test_backend_for_stream_gzip_compressed_raises() -> None:
    """Verify a gzip-compressed stream raises ValueError naming gzip."""
    with pytest.raises(ValueError) as exc_info:
        backend_for_stream(io.BytesIO(gzip.compress(b"SIMPLE  = whatever")))
    assert "gzip" in str(exc_info.value)


def test_backend_for_stream_zstd_compressed_raises() -> None:
    """Verify a zstd-compressed stream raises ValueError naming zstd."""
    with pytest.raises(ValueError) as exc_info:
        backend_for_stream(io.BytesIO(b"\x28\xb5\x2f\xfd" + b"frame"))
    assert "zstd" in str(exc_info.value)


def test_backend_for_path_strips_gz() -> None:
    """Verify .gz compression suffixes are stripped before extension
    dispatch.
    """
    assert backend_for_path("c.json.gz").name == "json"
    assert backend_for_path("c.h5.gz").name == "ndf"


def test_backend_for_path_strips_zst() -> None:
    """Verify .zst compression suffixes are stripped before extension
    dispatch.
    """
    assert backend_for_path("c.fits.zst").name == "fits"
    assert backend_for_path("c.sdf.zst").name == "ndf"


def test_backend_for_path_bare_compression_suffix_raises() -> None:
    """Verify a bare compression suffix with no format extension raises
    ValueError.
    """
    with pytest.raises(ValueError):
        backend_for_path("c.gz")


def test_is_binary_stream() -> None:
    """Verify _is_binary_stream accepts streams and rejects path-like
    inputs.
    """
    assert _is_binary_stream(io.BytesIO(b""))
    assert not _is_binary_stream("a.fits")
    # ResourcePath has read() but no seek(); it must not look like a
    # stream.
    assert not _is_binary_stream(ResourcePath("a.fits", forceAbsolute=False))


def test_path_is_compressed() -> None:
    """Verify _path_is_compressed recognizes .gz and .zst suffixes only."""
    assert _path_is_compressed("a/b.fits.gz")
    assert _path_is_compressed("a/b.json.zst")
    assert not _path_is_compressed("a/b.fits")


_DECOMPRESS_PAYLOAD = b"SIMPLE  = fake fits payload " * 1000


def test_decompress_path_gzip(tmp_path: Path) -> None:
    """Verify a .gz path stream-decompresses into a real temporary file."""
    path = tmp_path / "x.fits.gz"
    path.write_bytes(gzip.compress(_DECOMPRESS_PAYLOAD))
    with _decompress_path_to_temp_file(str(path)) as handle:
        # The decompressed data lives in a real file, not in memory.
        assert not isinstance(handle, io.BytesIO)
        assert hasattr(handle, "fileno")
        assert handle.read() == _DECOMPRESS_PAYLOAD


@pytest.mark.skipif(not ZSTD_AVAILABLE, reason="no zstd decompressor available")
def test_decompress_path_zstd(tmp_path: Path) -> None:
    """Verify a .zst path decompresses through the available zstd library."""
    path = tmp_path / "x.fits.zst"
    path.write_bytes(_zstd_compress(_DECOMPRESS_PAYLOAD))
    with _decompress_path_to_temp_file(str(path)) as handle:
        assert handle.read() == _DECOMPRESS_PAYLOAD


def test_decompress_suffix_without_compression_raises(tmp_path: Path) -> None:
    """Verify a .gz name whose content is not gzip fails honestly."""
    # The suffix selects the decompressor, so a .gz name whose content
    # is not gzip fails honestly.
    path = tmp_path / "x.fits.gz"
    path.write_bytes(_DECOMPRESS_PAYLOAD)
    with pytest.raises(gzip.BadGzipFile):
        _decompress_path_to_temp_file(str(path))


def test_decompress_zstd_no_decompressor(tmp_path: Path) -> None:
    """Verify a helpful ValueError when no zstd decompressor can be
    imported.
    """
    path = tmp_path / "x.fits.zst"
    path.write_bytes(b"\x28\xb5\x2f\xfd" + b"pretend zstd frame")
    # None entries make both import routes raise ImportError.
    blocked = {"compression": None, "compression.zstd": None, "zstandard": None}
    with mock.patch.dict(sys.modules, blocked):
        with pytest.raises(ValueError) as exc_info:
            _decompress_path_to_temp_file(str(path))
    assert "zstd" in str(exc_info.value)

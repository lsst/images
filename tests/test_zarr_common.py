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
import subprocess
import sys

import numpy as np
import pytest

try:
    from lsst.images.zarr._common import (
        LSST_NS,
        LSST_VERSION,
        OME_NS,
        OME_VERSION,
        ZarrCompressionOptions,
        ZarrPointerModel,
        archive_path_to_zarr_path,
        mask_dtype_for_plane_count,
    )

    HAVE_ZARR = True
except ImportError:
    HAVE_ZARR = False

skip_no_zarr = pytest.mark.skipif(not HAVE_ZARR, reason="zarr is not installed")


@skip_no_zarr
def test_pointer_round_trips() -> None:
    """Verify ZarrPointerModel round-trips through JSON serialization."""
    original = ZarrPointerModel(path="/lsst/psf/lsst_json")
    recovered = ZarrPointerModel.model_validate_json(original.model_dump_json())
    assert recovered == original


@skip_no_zarr
def test_constants() -> None:
    """Verify the zarr namespace and version constants."""
    assert LSST_NS == "lsst"
    assert OME_NS == "ome"
    assert OME_VERSION == "0.5"
    assert LSST_VERSION >= 1


@skip_no_zarr
def test_archive_path_translation() -> None:
    """Verify archive_path_to_zarr_path maps archive paths to zarr paths."""
    # Empty archive path -> the canonical root-level JSON tree.
    assert archive_path_to_zarr_path("") == "/lsst_json"
    # Non-empty archive paths are kept verbatim.
    assert archive_path_to_zarr_path("/image") == "/image"
    assert archive_path_to_zarr_path("image") == "/image"
    assert archive_path_to_zarr_path("/red/image") == "/red/image"
    assert archive_path_to_zarr_path("/psf") == "/psf"


@skip_no_zarr
def test_compression_defaults() -> None:
    """Verify default compression options depend on the dtype."""
    floats = ZarrCompressionOptions.default_for_dtype("float32")
    assert floats.codec == "blosc"
    assert floats.shuffle == "shuffle"
    ints = ZarrCompressionOptions.default_for_dtype("uint8")
    assert ints.shuffle == "bitshuffle"


@skip_no_zarr
def test_mask_dtype_picks_smallest_fit() -> None:
    """Verify the mask dtype is the smallest unsigned type that fits."""
    assert mask_dtype_for_plane_count(1) == np.dtype("uint8")
    assert mask_dtype_for_plane_count(8) == np.dtype("uint8")
    assert mask_dtype_for_plane_count(9) == np.dtype("uint16")
    assert mask_dtype_for_plane_count(16) == np.dtype("uint16")
    assert mask_dtype_for_plane_count(17) == np.dtype("uint32")
    assert mask_dtype_for_plane_count(32) == np.dtype("uint32")
    assert mask_dtype_for_plane_count(33) == np.dtype("uint64")
    assert mask_dtype_for_plane_count(64) == np.dtype("uint64")


@skip_no_zarr
def test_mask_dtype_refuses_more_than_64_planes() -> None:
    """Verify more than 64 mask planes is rejected."""
    with pytest.raises(ValueError, match="supports up to 64"):
        mask_dtype_for_plane_count(65)


def _import_in_subprocess(env_value: str | None) -> subprocess.CompletedProcess[str]:
    """Import `DEFAULT_TARGET_SHARD_BYTES` in a subprocess, since it reads
    from an environment variable at import time.
    """
    env = dict(os.environ)
    env.pop("LSST_IMAGES_ZARR_TARGET_SHARD_BYTES", None)
    if env_value is not None:
        env["LSST_IMAGES_ZARR_TARGET_SHARD_BYTES"] = env_value
    code = "from lsst.images.zarr._common import DEFAULT_TARGET_SHARD_BYTES;print(DEFAULT_TARGET_SHARD_BYTES)"
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


@skip_no_zarr
def test_unset_uses_default() -> None:
    """Verify the default shard size is used when the env var is unset."""
    result = _import_in_subprocess(None)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(16 * 1024 * 1024)


@skip_no_zarr
def test_set_value_overrides() -> None:
    """Verify the env var overrides the default shard size."""
    result = _import_in_subprocess("1234567")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "1234567"


@skip_no_zarr
def test_garbage_value_fails_at_import() -> None:
    """Verify a non-integer env var value fails at import time."""
    result = _import_in_subprocess("not-a-number")
    assert result.returncode != 0
    assert "LSST_IMAGES_ZARR_TARGET_SHARD_BYTES" in result.stderr
    assert "is not a base-10 integer" in result.stderr

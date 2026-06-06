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
import unittest

import numpy as np

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


@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class CommonTestCase(unittest.TestCase):
    """Tests for the zarr ``_common`` module."""

    def test_pointer_round_trips(self) -> None:
        original = ZarrPointerModel(path="/lsst/psf/lsst_json")
        recovered = ZarrPointerModel.model_validate_json(original.model_dump_json())
        self.assertEqual(recovered, original)

    def test_constants(self) -> None:
        self.assertEqual(LSST_NS, "lsst")
        self.assertEqual(OME_NS, "ome")
        self.assertEqual(OME_VERSION, "0.5")
        self.assertGreaterEqual(LSST_VERSION, 1)

    def test_archive_path_translation(self) -> None:
        # Empty archive path -> the canonical root-level JSON tree.
        self.assertEqual(archive_path_to_zarr_path(""), "/lsst_json")
        # Non-empty archive paths are kept verbatim.
        self.assertEqual(archive_path_to_zarr_path("/image"), "/image")
        self.assertEqual(archive_path_to_zarr_path("image"), "/image")
        self.assertEqual(archive_path_to_zarr_path("/red/image"), "/red/image")
        self.assertEqual(archive_path_to_zarr_path("/psf"), "/psf")

    def test_compression_defaults(self) -> None:
        floats = ZarrCompressionOptions.default_for_dtype("float32")
        self.assertEqual(floats.codec, "blosc")
        self.assertEqual(floats.shuffle, "shuffle")
        ints = ZarrCompressionOptions.default_for_dtype("uint8")
        self.assertEqual(ints.shuffle, "bitshuffle")

    def test_mask_dtype_picks_smallest_fit(self) -> None:
        self.assertEqual(mask_dtype_for_plane_count(1), np.dtype("uint8"))
        self.assertEqual(mask_dtype_for_plane_count(8), np.dtype("uint8"))
        self.assertEqual(mask_dtype_for_plane_count(9), np.dtype("uint16"))
        self.assertEqual(mask_dtype_for_plane_count(16), np.dtype("uint16"))
        self.assertEqual(mask_dtype_for_plane_count(17), np.dtype("uint32"))
        self.assertEqual(mask_dtype_for_plane_count(32), np.dtype("uint32"))
        self.assertEqual(mask_dtype_for_plane_count(33), np.dtype("uint64"))
        self.assertEqual(mask_dtype_for_plane_count(64), np.dtype("uint64"))

    def test_mask_dtype_refuses_more_than_64_planes(self) -> None:
        with self.assertRaisesRegex(ValueError, "supports up to 64"):
            mask_dtype_for_plane_count(65)


@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class TargetShardBytesEnvVarTestCase(unittest.TestCase):
    """`DEFAULT_TARGET_SHARD_BYTES` reads from env var at import time."""

    def _import_in_subprocess(self, env_value: str | None) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env.pop("LSST_IMAGES_ZARR_TARGET_SHARD_BYTES", None)
        if env_value is not None:
            env["LSST_IMAGES_ZARR_TARGET_SHARD_BYTES"] = env_value
        code = (
            "from lsst.images.zarr._common import DEFAULT_TARGET_SHARD_BYTES;"
            "print(DEFAULT_TARGET_SHARD_BYTES)"
        )
        return subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_unset_uses_default(self) -> None:
        result = self._import_in_subprocess(None)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), str(16 * 1024 * 1024))

    def test_set_value_overrides(self) -> None:
        result = self._import_in_subprocess("1234567")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "1234567")

    def test_garbage_value_fails_at_import(self) -> None:
        result = self._import_in_subprocess("not-a-number")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("LSST_IMAGES_ZARR_TARGET_SHARD_BYTES", result.stderr)
        self.assertIn("is not a base-10 integer", result.stderr)


if __name__ == "__main__":
    unittest.main()

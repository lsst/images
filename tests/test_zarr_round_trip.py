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

import numpy as np
import pytest

from lsst.images import Box, ColorImage, Image, Mask, MaskedImage, MaskPlane, MaskSchema

try:
    import zarr

    from lsst.images.tests import RoundtripZarr
    from lsst.images.zarr._store import open_store_for_read

    HAVE_ZARR = True
except ImportError:
    HAVE_ZARR = False

skip_no_zarr = pytest.mark.skipif(not HAVE_ZARR, reason="zarr is not installed")


@skip_no_zarr
def test_image_round_trip() -> None:
    """Verify an Image round-trips through the zarr backend."""
    original = Image(
        np.arange(20, dtype=np.float32).reshape(4, 5),
        bbox=Box.factory[10:14, 20:25],
    )
    with RoundtripZarr(original) as roundtrip:
        recovered = roundtrip.result
        np.testing.assert_array_equal(recovered.array, original.array)
        assert recovered.bbox == original.bbox


@skip_no_zarr
def test_image_round_trip_writes_shards() -> None:
    """Verify a large Image is written with sharded chunks."""
    # 300x300 float32: chunks (256, 256) -> shard (512, 512) by the
    # byte-budget rule (target 16 MiB, ratio ~64, k ~ 8 capped at the
    # 2-chunk-per-axis ceiling of 256 * 2 = 512).
    original = Image(
        np.zeros((300, 300), dtype=np.float32),
        bbox=Box.factory[0:300, 0:300],
    )
    with RoundtripZarr(original) as roundtrip:
        with open_store_for_read(roundtrip.filename) as store:
            root = zarr.open_group(store=store, mode="r", zarr_format=3)
            image_arr = root["image"]
            assert tuple(image_arr.chunks) == (256, 256)
            assert tuple(image_arr.shards) == (512, 512)
            # Single-chunk metadata arrays must NOT be sharded.
            lsst_json_arr = root["lsst_json"]
            assert lsst_json_arr.shards is None
        # Data round-trip is preserved.
        np.testing.assert_array_equal(roundtrip.result.array, original.array)


@skip_no_zarr
def test_masked_image_round_trip() -> None:
    """Verify a MaskedImage round-trips through the zarr backend."""
    schema = MaskSchema(
        [
            MaskPlane("BAD", "Bad pixel."),
            MaskPlane("SAT", "Saturated."),
            MaskPlane("CR", "Cosmic ray."),
        ]
    )
    image = Image(
        np.arange(20, dtype=np.float32).reshape(4, 5),
        bbox=Box.factory[10:14, 20:25],
    )
    original = MaskedImage(image, mask_schema=schema)
    original.mask.set("BAD", image.array % 2 == 0)
    original.mask.set("SAT", image.array > 10)

    with RoundtripZarr(original) as roundtrip:
        recovered = roundtrip.result
        np.testing.assert_array_equal(recovered.image.array, original.image.array)
        np.testing.assert_array_equal(recovered.mask.array, original.mask.array)


@skip_no_zarr
def test_mask_round_trip() -> None:
    """Verify a top-level Mask round-trips through the zarr backend."""
    # Top-level Mask: schema is on the object itself, not on an
    # inner ``mask`` attribute. write() must reach it.
    schema = MaskSchema(
        [
            MaskPlane("BAD", "Bad pixel."),
            MaskPlane("SAT", "Saturated."),
            MaskPlane("CR", "Cosmic ray."),
        ]
    )
    original = Mask(
        np.zeros((4, 5, schema.mask_size), dtype=schema.dtype),
        bbox=Box.factory[10:14, 20:25],
        schema=schema,
    )
    original.set("BAD", np.array([[i % 2 == 0 for i in range(5)] for _ in range(4)]))
    original.set("SAT", np.array([[i > 2 for i in range(5)] for _ in range(4)]))
    with RoundtripZarr(original) as roundtrip:
        recovered = roundtrip.result
        np.testing.assert_array_equal(recovered.array, original.array)
        assert recovered.bbox == original.bbox
        assert list(recovered.schema.names) == list(original.schema.names)


@skip_no_zarr
def test_uint16_mask_packs_with_element_stride() -> None:
    """Verify multi-element uint16 masks pack with element stride."""
    # 20-plane uint16 schema (mask_size = 2 elements per pixel).
    # Setting plane 16 must produce an on-disk packed value whose
    # CF flag_masks[16] bit is set — the bit position must match
    # the schema's element-stride layout, not the byte-stride
    # layout. Without the fix, plane 16 lands at packed bit 8.
    schema = MaskSchema(
        [MaskPlane(f"P{i}", f"Plane {i}.") for i in range(20)],
        dtype=np.uint16,
    )
    image = Image(
        np.zeros((4, 5), dtype=np.float32),
        bbox=Box.factory[10:14, 20:25],
    )
    original = MaskedImage(image, mask_schema=schema)
    target_pixel = np.zeros((4, 5), dtype=bool)
    target_pixel[0, 0] = True
    original.mask.set("P16", target_pixel)
    with RoundtripZarr(original) as roundtrip:
        with open_store_for_read(roundtrip.filename) as store:
            root = zarr.open_group(store=store, mode="r", zarr_format=3)
            mask_arr = root["mask"]
            flag_masks = list(mask_arr.attrs["flag_masks"])
            on_disk = int(mask_arr[0, 0])
            assert flag_masks[16] == 1 << 16
            assert on_disk & flag_masks[16] != 0
        recovered = roundtrip.result
        np.testing.assert_array_equal(recovered.mask.array, original.mask.array)


@skip_no_zarr
def test_masked_image_with_40_planes_round_trip() -> None:
    """Verify a MaskedImage with 40 mask planes round-trips."""
    schema = MaskSchema([MaskPlane(f"P{i}", f"Plane {i}.") for i in range(40)])
    image = Image(
        np.arange(20, dtype=np.float32).reshape(4, 5),
        bbox=Box.factory[10:14, 20:25],
    )
    original = MaskedImage(image, mask_schema=schema)
    original.mask.set("P0", image.array % 2 == 0)
    original.mask.set("P39", image.array > 10)

    with RoundtripZarr(original) as roundtrip:
        recovered = roundtrip.result
        # 40 planes packed into uint64 on disk; recovered as 5 bytes/pixel.
        np.testing.assert_array_equal(recovered.mask.array, original.mask.array)


@skip_no_zarr
def test_color_image_round_trip() -> None:
    """ColorImage round-trips through the zarr backend."""
    arr = np.zeros((4, 5, 3), dtype=np.uint8)
    arr[..., 0] = 1
    arr[..., 1] = 2
    arr[..., 2] = 3
    original = ColorImage(arr, bbox=Box.factory[10:14, 20:25])

    with RoundtripZarr(original) as roundtrip:
        recovered = roundtrip.result
        np.testing.assert_array_equal(recovered.array, original.array)
        assert recovered.bbox == original.bbox

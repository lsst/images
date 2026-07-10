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
import warnings

import numpy as np
import pytest

from lsst.images import Box, Image, Mask, MaskedImage, MaskPlane, MaskSchema

try:
    import zarr  # noqa: F401

    from lsst.images.zarr import write
    from lsst.images.zarr._model import ZarrArray, ZarrDocument, ZarrGroup
    from lsst.images.zarr._store import open_store_for_write

    HAVE_ZARR = True
except ImportError:
    HAVE_ZARR = False

try:
    import xarray as xr

    HAVE_XARRAY = True
except ImportError:
    HAVE_XARRAY = False

skip_no_zarr = pytest.mark.skipif(not HAVE_ZARR, reason="zarr is not installed")
skip_no_xarray = pytest.mark.skipif(not HAVE_XARRAY, reason="xarray is not installed")


def make_masked_image() -> MaskedImage:
    """Return a freshly-constructed MaskedImage with the BAD mask plane
    set.
    """
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
    masked = MaskedImage(image, mask_schema=schema)
    masked.mask.set("BAD", image.array % 2 == 0)
    return masked


@skip_no_zarr
@skip_no_xarray
def test_open_zarr_returns_dataset_with_masked_image_components() -> None:
    """``xr.open_zarr`` returns a Dataset with the masked-image siblings."""
    masked = make_masked_image()

    with tempfile.TemporaryDirectory() as tmp:
        target = os.path.join(tmp, "masked.zarr")
        write(masked, target)
        ds = xr.open_zarr(target, consolidated=False)
        # Three data variables sharing the (y, x) dims.
        assert "image" in ds.data_vars
        assert "variance" in ds.data_vars
        assert "mask" in ds.data_vars
        assert ds["image"].dims == ("y", "x")
        assert ds["mask"].dims == ("y", "x")
        assert ds["image"].shape == (4, 5)
        # CF flag attrs survive on the mask variable.
        assert ds["mask"].attrs["flag_meanings"] == "BAD SAT CR"
        assert list(ds["mask"].attrs["flag_masks"]) == [1, 2, 4]


@skip_no_zarr
@skip_no_xarray
def test_open_zarr_uses_consolidated_metadata() -> None:
    """``write()`` consolidates metadata so xr.open_zarr uses one fetch."""
    masked = make_masked_image()
    with tempfile.TemporaryDirectory() as tmp:
        target = os.path.join(tmp, "masked.zarr")
        write(masked, target)
        # Default ``consolidated=None`` means "use it if available";
        # if it isn't present xarray emits a ``RuntimeWarning`` and
        # falls back to walking every array. Promote that warning to
        # an error to confirm the consolidated path is taken.
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            xr.open_zarr(target)


@skip_no_zarr
@skip_no_xarray
def test_open_zarr_data_values_match_in_memory() -> None:
    """The bytes xarray reads are the same bytes the archive wrote."""
    masked = make_masked_image()

    with tempfile.TemporaryDirectory() as tmp:
        target = os.path.join(tmp, "masked.zarr")
        write(masked, target)
        ds = xr.open_zarr(target, consolidated=False)
        np.testing.assert_array_equal(ds["image"].values, masked.image.array)
        np.testing.assert_array_equal(ds["variance"].values, masked.variance.array)
        # Mask on disk is a 2-D packed wide-int; compare against the
        # equivalent packing of the in-memory (y, x, mask_size) array.
        packed = np.zeros(masked.mask.array.shape[:2], dtype=ds["mask"].dtype)
        for i in range(masked.mask.array.shape[2]):
            packed |= masked.mask.array[..., i].astype(ds["mask"].dtype) << (8 * i)
        np.testing.assert_array_equal(ds["mask"].values, packed)


def make_document() -> ZarrDocument:
    """Return a ZarrDocument with a (y, x) image alongside a 4-D psf and
    a 1-D JSON blob.
    """
    root = ZarrGroup()
    image = ZarrArray(data=np.zeros((4, 5), dtype=np.float32))
    image.attributes.extra["_ARRAY_DIMENSIONS"] = ["y", "x"]
    root.arrays["image"] = image
    # 4-D PSF and the JSON blob have no explicit dimension names and
    # differing axis sizes (3 vs 128) — the colliding case.
    root.arrays["psf"] = ZarrArray(data=np.zeros((2, 3, 6, 6), dtype=np.float64))
    root.arrays["lsst_json"] = ZarrArray(data=np.zeros((128,), dtype=np.uint8))
    return ZarrDocument(root=root)


@skip_no_zarr
@skip_no_xarray
def test_open_zarr_with_psf_and_json_blob() -> None:
    """A group with several non-``(y, x)`` arrays still opens in xarray.

    The ``CellCoadd`` layout puts the ``(y, x)`` ``image`` in the root
    group alongside a 4-D ``psf`` and the 1-D ``lsst_json`` blob. Every
    array a group writes must carry distinct dimension names; otherwise
    xarray collapses the unnamed ones onto a single anonymous dimension
    and rejects the group when their sizes differ.
    """
    with tempfile.TemporaryDirectory() as tmp:
        target = os.path.join(tmp, "cell.zarr")
        with open_store_for_write(target) as store:
            make_document().to_zarr(store)
        ds = xr.open_zarr(target, consolidated=False)
        assert ds["image"].dims == ("y", "x")
        # psf / lsst_json each get their own distinct, non-anonymous
        # dims rather than colliding on a single ``None`` dimension.
        assert None not in ds["psf"].dims
        assert None not in ds["lsst_json"].dims
        assert len(set(ds["psf"].dims)) == ds["psf"].ndim
        assert ds["psf"].shape == (2, 3, 6, 6)


@skip_no_zarr
@skip_no_xarray
def test_uint16_schema_decodes_under_cf_rules() -> None:
    """A standalone CF-aware reader can decode plane membership.

    Uses ``xarray.open_zarr`` to read the archive without any LSST
    code on the read side, then applies the standard CF flag-decoding
    rule ``(value & flag_masks[i]) != 0`` to recover the plane
    membership of every pixel. The recovered membership must match
    what was written. Catches regressions in the on-disk packing
    layout (e.g. element-stride vs byte-stride bugs) that would
    otherwise be invisible to an internal round-trip.
    """
    # 20-plane uint16 schema (mask_size = 2) — exercises the
    # multi-element packing path.
    plane_names = [f"P{i}" for i in range(20)]
    schema = MaskSchema(
        [MaskPlane(name, f"Plane {name}.") for name in plane_names],
        dtype=np.uint16,
    )
    # Set distinct planes in distinct pixels so a single pass can
    # cover the whole bit range, including planes that only the
    # high element holds (P16..P19).
    original = Mask(
        np.zeros((4, 5, schema.mask_size), dtype=schema.dtype),
        bbox=Box.factory[0:4, 0:5],
        schema=schema,
    )
    plane_for_pixel = {(0, 0): "P0", (0, 1): "P7", (1, 2): "P8", (2, 3): "P15", (3, 4): "P16"}
    for (y, x), plane_name in plane_for_pixel.items():
        sel = np.zeros((4, 5), dtype=bool)
        sel[y, x] = True
        original.set(plane_name, sel)

    with tempfile.TemporaryDirectory() as tmp:
        target = os.path.join(tmp, "mask.zarr")
        write(original, target)
        ds = xr.open_zarr(target, consolidated=False)
        mask_da = ds["mask"]
        flag_masks = list(mask_da.attrs["flag_masks"])
        flag_meanings = mask_da.attrs["flag_meanings"].split()
        assert flag_meanings == plane_names
        assert flag_masks == [1 << i for i in range(20)]
        mask_values = mask_da.values
        # CF decode: plane i is set at (y, x) iff
        # (mask_values[y, x] & flag_masks[i]) != 0.
        for (y, x), plane_name in plane_for_pixel.items():
            plane_idx = flag_meanings.index(plane_name)
            bit = flag_masks[plane_idx]
            assert int(mask_values[y, x]) & bit != 0, (
                f"plane {plane_name} (bit {bit:#x}) not set at ({y}, {x}); "
                f"on-disk value = {int(mask_values[y, x]):#x}"
            )
            # All other planes must be unset at this pixel.
            for other_idx in range(len(flag_meanings)):
                if other_idx == plane_idx:
                    continue
                assert int(mask_values[y, x]) & flag_masks[other_idx] == 0, (
                    f"plane {flag_meanings[other_idx]} unexpectedly set at ({y}, {x})"
                )

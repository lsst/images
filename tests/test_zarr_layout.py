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

from lsst.images._transforms._ast import (
    CmpMap,
    Frame,
    FrameSet,
    PolyMap,
    ZoomMap,
)

try:
    from lsst.images.zarr._layout import (
        affine_check,
        axes_for_archive_class,
        chunks_aligned_to,
        chunks_for,
        decorate_sub_archives,
        default_shards,
    )
    from lsst.images.zarr._model import ZarrArray, ZarrDocument, ZarrGroup

    HAVE_ZARR = True
except ImportError:
    HAVE_ZARR = False

skip_no_zarr = pytest.mark.skipif(not HAVE_ZARR, reason="zarr is not installed")

# Byte budget used by the `default_shards` tests.
TARGET = 16 * 1024 * 1024  # 16 MiB


def _make_linear_frame_set(*, scale: float = 0.2) -> FrameSet:
    base = Frame(2, "Domain=PIXEL")
    sky = Frame(2, "Domain=SKY")
    fs = FrameSet(base)
    fs.addFrame(FrameSet.BASE, ZoomMap(2, scale), sky)
    return fs


def _make_distorted_frame_set() -> FrameSet:
    base = Frame(2, "Domain=PIXEL")
    sky = Frame(2, "Domain=SKY")
    # astshim's pybind11 bindings require an ndarray for the
    # coefficients; a nested Python list is no longer auto-converted.
    forward_coeffs = np.array(
        [
            [1.0, 1, 1, 0],
            [0.001, 1, 0, 2],
            [1.0, 2, 0, 1],
            [0.001, 2, 2, 0],
        ],
        dtype=float,
    )
    poly = PolyMap(forward_coeffs, 2, "IterInverse=1, NIterInverse=20")
    cmp = CmpMap(poly, ZoomMap(2, 0.2), True)
    fs = FrameSet(base)
    fs.addFrame(FrameSet.BASE, cmp, sky)
    return fs


@skip_no_zarr
def test_axes_for_archive_class() -> None:
    """Per-archive-class axes derivation rules."""
    # Standard 2-D images use (y, x).
    assert axes_for_archive_class("Image") == ("y", "x")
    assert axes_for_archive_class("MaskedImage") == ("y", "x")
    assert axes_for_archive_class("VisitImage") == ("y", "x")
    assert axes_for_archive_class("Mask") == ("y", "x")
    assert axes_for_archive_class("CellCoadd") == ("y", "x")
    # ColorImage's root has no top-level multiscale; this returns
    # an empty tuple to signal "no OME multiscale at this level".
    assert axes_for_archive_class("ColorImage") == ()


@skip_no_zarr
def test_chunks_for_default() -> None:
    """Chunk derivation without an explicit override."""
    # Plain images clamp to the per-axis chunk limit (256 by default).
    assert chunks_for((4096, 4096), None) == (256, 256)
    # Smaller than the limit -> use full dim.
    assert chunks_for((200, 100), None) == (200, 100)


@skip_no_zarr
def test_chunks_for_override() -> None:
    """Chunk derivation with an explicit override."""
    assert chunks_for((4096, 4096), (256, 256)) == (256, 256)


@skip_no_zarr
def test_chunks_aligned_to_matches_image() -> None:
    """Sibling arrays align their chunks to the image's chunks."""
    # variance / mask follow image's chunks when not overridden.
    assert chunks_aligned_to(image_chunks=(256, 256), shape=(4096, 4096)) == (256, 256)
    # If the sibling shape is smaller than image's chunks, clamp.
    assert chunks_aligned_to(image_chunks=(1024, 1024), shape=(300, 600)) == (300, 600)


@skip_no_zarr
def test_pure_linear_passes() -> None:
    """Affine-residual validator accepts a purely linear frame set."""
    # NGFF v0.5 composes ``coordinateTransformations`` in list
    # order: ``scale`` is applied first, then ``affine``. For a
    # pure 0.2 pixel→sky scale, the composed effect on a unit
    # pixel must be 0.2 — not 0.04 (which would result from
    # leaving the scale embedded in both the explicit scale block
    # AND the affine's Jacobian).
    fs = _make_linear_frame_set(scale=0.2)
    result = affine_check(
        frame_set=fs,
        image_shape=(64, 64),
        max_residual_pixels=1.0,
    )
    assert not result.dropped
    assert result.coordinate_transformations is not None
    ct = result.coordinate_transformations
    scale_block, affine_block = ct[0], ct[1]
    assert scale_block["type"] == "scale"
    assert affine_block["type"] == "affine"
    # The scale block carries the per-axis pixel size; the affine
    # has unit-norm columns (a pure rotation/translation here).
    assert scale_block["scale"][0] == pytest.approx(0.2)
    assert scale_block["scale"][1] == pytest.approx(0.2)
    assert affine_block["affine"][0][0] == pytest.approx(1.0)
    assert affine_block["affine"][1][1] == pytest.approx(1.0)
    # Compose scale ∘ affine and apply to a unit pixel vector.
    scale = scale_block["scale"]
    affine = np.array(affine_block["affine"])
    scaled = np.array([scale[0] * 1.0, scale[1] * 1.0, 1.0])
    composed = affine @ scaled
    assert composed[0] == pytest.approx(0.2)
    assert composed[1] == pytest.approx(0.2)


@skip_no_zarr
def test_high_distortion_drops_block() -> None:
    """Affine-residual validator drops the OME affine block for a
    highly distorted frame set.
    """
    fs = _make_distorted_frame_set()
    result = affine_check(
        frame_set=fs,
        image_shape=(4096, 4096),
        max_residual_pixels=1.0,
    )
    assert result.dropped
    assert result.coordinate_transformations is None


@skip_no_zarr
def test_sub_group_with_image_gets_lsst_and_ome_attrs() -> None:
    """`decorate_sub_archives` walks the IR and adds OME / lsst attrs."""
    doc = ZarrDocument(root=ZarrGroup())
    doc.root.attributes.lsst["archive_class"] = "ColorImage"
    red = doc.root.ensure_group("/red")
    red.arrays["image"] = ZarrArray(data=np.ones((4, 5), dtype="float32"))

    decorate_sub_archives(doc)

    assert red.attributes.lsst["archive_class"] == "Image"
    assert "multiscales" in red.attributes.ome
    assert red.attributes.ome["multiscales"][0]["datasets"][0]["path"] == "image"


@skip_no_zarr
def test_root_archive_class_is_unchanged() -> None:
    """`decorate_sub_archives` leaves the root archive class alone."""
    doc = ZarrDocument(root=ZarrGroup())
    doc.root.attributes.lsst["archive_class"] = "ColorImage"
    red = doc.root.ensure_group("/red")
    red.arrays["image"] = ZarrArray(data=np.ones((4, 5), dtype="float32"))

    decorate_sub_archives(doc)

    # Root keeps ColorImage; only sub-groups are decorated.
    assert doc.root.attributes.lsst["archive_class"] == "ColorImage"


@skip_no_zarr
def test_4k_float32_image_uses_byte_budget() -> None:
    """The `default_shards` byte-budget rule for a 4k float32 image."""
    result = default_shards(
        chunks=(256, 256),
        shape=(4096, 4096),
        dtype=np.dtype("float32"),
        target_bytes=TARGET,
    )
    assert result == (2048, 2048)


@skip_no_zarr
def test_3d_mask_plane_axis_untouched() -> None:
    """`default_shards` grows only the spatial axes of a 3-D mask."""
    # chunks already cover the plane axis; growable axes are y, x only.
    result = default_shards(
        chunks=(8, 256, 256),
        shape=(8, 4096, 4096),
        dtype=np.dtype("uint8"),
        target_bytes=TARGET,
    )
    assert result == (8, 1536, 1536)


@skip_no_zarr
def test_tiny_single_chunk_returns_none() -> None:
    """`default_shards` returns None for a tiny single-chunk array."""
    result = default_shards(
        chunks=(40,),
        shape=(40,),
        dtype=np.dtype("uint8"),
        target_bytes=TARGET,
    )
    assert result is None


@skip_no_zarr
def test_chunks_equal_shape_returns_none() -> None:
    """`default_shards` returns None when chunks already cover the array."""
    result = default_shards(
        chunks=(1024, 1024),
        shape=(1024, 1024),
        dtype=np.dtype("float32"),
        target_bytes=TARGET,
    )
    assert result is None


@skip_no_zarr
def test_already_big_chunk_returns_none() -> None:
    """`default_shards` returns None when a chunk already exceeds the
    byte budget.
    """
    # 4096*4096*4 = 64 MiB > 16 MiB target.
    result = default_shards(
        chunks=(4096, 4096),
        shape=(8192, 8192),
        dtype=np.dtype("float32"),
        target_bytes=TARGET,
    )
    assert result is None


@skip_no_zarr
def test_k_le_one_returns_none() -> None:
    """`default_shards` returns None when the growth factor rounds to 1."""
    # chunk=256x256 float32 = 256 KiB; ratio=1.25 -> k=round(1.25)=1
    # -> returns None.
    chunk_bytes = 256 * 256 * 4
    result = default_shards(
        chunks=(256, 256),
        shape=(4096, 4096),
        dtype=np.dtype("float32"),
        target_bytes=int(chunk_bytes * 1.25),
    )
    assert result is None


@skip_no_zarr
def test_cap_at_array_bounds() -> None:
    """`default_shards` caps shard sizes at the array bounds."""
    # 600x600 float32; chunk_bytes = 256 KiB; ratio = 64; k = 8.
    # Uncapped shard would be (2048, 2048) but the array only has
    # 3 chunks per axis (ceil(600/256) = 3), so the cap is (768, 768).
    result = default_shards(
        chunks=(256, 256),
        shape=(600, 600),
        dtype=np.dtype("float32"),
        target_bytes=TARGET,
    )
    assert result == (768, 768)


@skip_no_zarr
def test_cell_coadd_psf() -> None:
    """`default_shards` handles the 4-D cell-coadd PSF layout."""
    # (25, 25, 150, 150) float32 with (1, 1, 150, 150) chunks.
    # chunk_bytes = 90 KiB; ratio ~= 186; growable axes are 0 and 1.
    # k = round(sqrt(186)) = 14.
    result = default_shards(
        chunks=(1, 1, 150, 150),
        shape=(25, 25, 150, 150),
        dtype=np.dtype("float32"),
        target_bytes=TARGET,
    )
    assert result == (14, 14, 150, 150)


@skip_no_zarr
def test_mismatched_ndim_raises() -> None:
    """`default_shards` rejects chunks and shape of different rank."""
    with pytest.raises(ValueError, match="rank"):
        default_shards(
            chunks=(256, 256),
            shape=(4096, 4096, 4096),
            dtype=np.dtype("float32"),
            target_bytes=TARGET,
        )


@skip_no_zarr
def test_zero_itemsize_returns_none() -> None:
    """`default_shards` returns None for a zero-itemsize dtype."""
    # void(0) has itemsize 0; defensive guard against degenerate dtypes.
    result = default_shards(
        chunks=(256, 256),
        shape=(4096, 4096),
        dtype=np.dtype("V0"),
        target_bytes=TARGET,
    )
    assert result is None

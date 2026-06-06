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

import unittest

import numpy as np

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


@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class LayoutTestCase(unittest.TestCase):
    """Per-archive-class axes and chunk derivation rules."""

    def test_axes_for_archive_class(self) -> None:
        # Standard 2-D images use (y, x).
        self.assertEqual(axes_for_archive_class("Image"), ("y", "x"))
        self.assertEqual(axes_for_archive_class("MaskedImage"), ("y", "x"))
        self.assertEqual(axes_for_archive_class("VisitImage"), ("y", "x"))
        self.assertEqual(axes_for_archive_class("Mask"), ("y", "x"))
        self.assertEqual(axes_for_archive_class("CellCoadd"), ("y", "x"))
        # ColorImage's root has no top-level multiscale; this returns
        # an empty tuple to signal "no OME multiscale at this level".
        self.assertEqual(axes_for_archive_class("ColorImage"), ())

    def test_chunks_for_default(self) -> None:
        # Plain images clamp to the per-axis chunk limit (256 by default).
        self.assertEqual(chunks_for("Image", (4096, 4096), None), (256, 256))
        # Smaller than the limit -> use full dim.
        self.assertEqual(chunks_for("Image", (200, 100), None), (200, 100))

    def test_chunks_for_override(self) -> None:
        self.assertEqual(chunks_for("Image", (4096, 4096), (256, 256)), (256, 256))

    def test_chunks_for_cell_coadd_uses_cell_shape(self) -> None:
        result = chunks_for(
            "CellCoadd",
            (4096, 4096),
            None,
            archive_metadata={"cell_shape": (256, 256)},
        )
        self.assertEqual(result, (256, 256))

    def test_chunks_for_cell_coadd_without_metadata_falls_back(self) -> None:
        self.assertEqual(chunks_for("CellCoadd", (4096, 4096), None), (256, 256))

    def test_chunks_aligned_to_matches_image(self) -> None:
        # variance / mask follow image's chunks when not overridden.
        self.assertEqual(
            chunks_aligned_to(image_chunks=(256, 256), shape=(4096, 4096)),
            (256, 256),
        )
        # If the sibling shape is smaller than image's chunks, clamp.
        self.assertEqual(
            chunks_aligned_to(image_chunks=(1024, 1024), shape=(300, 600)),
            (300, 600),
        )


@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class AffineValidatorTestCase(unittest.TestCase):
    """Affine-residual validator gating the OME affine block."""

    def _make_linear_frame_set(self, *, scale: float = 0.2) -> FrameSet:
        base = Frame(2, "Domain=PIXEL")
        sky = Frame(2, "Domain=SKY")
        fs = FrameSet(base)
        fs.addFrame(FrameSet.BASE, ZoomMap(2, scale), sky)
        return fs

    def _make_distorted_frame_set(self) -> FrameSet:
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

    def test_pure_linear_passes(self) -> None:
        # NGFF v0.5 composes ``coordinateTransformations`` in list
        # order: ``scale`` is applied first, then ``affine``. For a
        # pure 0.2 pixel→sky scale, the composed effect on a unit
        # pixel must be 0.2 — not 0.04 (which would result from
        # leaving the scale embedded in both the explicit scale block
        # AND the affine's Jacobian).
        fs = self._make_linear_frame_set(scale=0.2)
        result = affine_check(
            frame_set=fs,
            image_shape=(64, 64),
            max_residual_pixels=1.0,
        )
        self.assertFalse(result.dropped)
        self.assertIsNotNone(result.coordinate_transformations)
        ct = result.coordinate_transformations
        assert ct is not None  # for type checkers
        scale_block, affine_block = ct[0], ct[1]
        self.assertEqual(scale_block["type"], "scale")
        self.assertEqual(affine_block["type"], "affine")
        # The scale block carries the per-axis pixel size; the affine
        # has unit-norm columns (a pure rotation/translation here).
        self.assertAlmostEqual(scale_block["scale"][0], 0.2)
        self.assertAlmostEqual(scale_block["scale"][1], 0.2)
        self.assertAlmostEqual(affine_block["affine"][0][0], 1.0)
        self.assertAlmostEqual(affine_block["affine"][1][1], 1.0)
        # Compose scale ∘ affine and apply to a unit pixel vector.
        scale = scale_block["scale"]
        affine = np.array(affine_block["affine"])
        scaled = np.array([scale[0] * 1.0, scale[1] * 1.0, 1.0])
        composed = affine @ scaled
        self.assertAlmostEqual(composed[0], 0.2)
        self.assertAlmostEqual(composed[1], 0.2)

    def test_high_distortion_drops_block(self) -> None:
        fs = self._make_distorted_frame_set()
        result = affine_check(
            frame_set=fs,
            image_shape=(4096, 4096),
            max_residual_pixels=1.0,
        )
        self.assertTrue(result.dropped)
        self.assertIsNone(result.coordinate_transformations)


@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class DecorateSubArchivesTestCase(unittest.TestCase):
    """`decorate_sub_archives` walks the IR and adds OME / lsst attrs."""

    def test_sub_group_with_image_gets_lsst_and_ome_attrs(self) -> None:
        doc = ZarrDocument(root=ZarrGroup())
        doc.root.attributes.lsst["archive_class"] = "ColorImage"
        red = doc.root.ensure_group("/red")
        red.arrays["image"] = ZarrArray(data=np.ones((4, 5), dtype="float32"))

        decorate_sub_archives(doc)

        self.assertEqual(red.attributes.lsst["archive_class"], "Image")
        self.assertIn("multiscales", red.attributes.ome)
        self.assertEqual(red.attributes.ome["multiscales"][0]["datasets"][0]["path"], "image")

    def test_root_archive_class_is_unchanged(self) -> None:
        doc = ZarrDocument(root=ZarrGroup())
        doc.root.attributes.lsst["archive_class"] = "ColorImage"
        red = doc.root.ensure_group("/red")
        red.arrays["image"] = ZarrArray(data=np.ones((4, 5), dtype="float32"))

        decorate_sub_archives(doc)

        # Root keeps ColorImage; only sub-groups are decorated.
        self.assertEqual(doc.root.attributes.lsst["archive_class"], "ColorImage")


@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class DefaultShardsTestCase(unittest.TestCase):
    """The `default_shards` byte-budget rule."""

    TARGET = 16 * 1024 * 1024  # 16 MiB

    def test_4k_float32_image_uses_byte_budget(self) -> None:
        result = default_shards(
            chunks=(256, 256),
            shape=(4096, 4096),
            dtype=np.dtype("float32"),
            target_bytes=self.TARGET,
        )
        self.assertEqual(result, (2048, 2048))

    def test_3d_mask_plane_axis_untouched(self) -> None:
        # chunks already cover the plane axis; growable axes are y, x only.
        result = default_shards(
            chunks=(8, 256, 256),
            shape=(8, 4096, 4096),
            dtype=np.dtype("uint8"),
            target_bytes=self.TARGET,
        )
        self.assertEqual(result, (8, 1536, 1536))

    def test_tiny_single_chunk_returns_none(self) -> None:
        result = default_shards(
            chunks=(40,),
            shape=(40,),
            dtype=np.dtype("uint8"),
            target_bytes=self.TARGET,
        )
        self.assertIsNone(result)

    def test_chunks_equal_shape_returns_none(self) -> None:
        result = default_shards(
            chunks=(1024, 1024),
            shape=(1024, 1024),
            dtype=np.dtype("float32"),
            target_bytes=self.TARGET,
        )
        self.assertIsNone(result)

    def test_already_big_chunk_returns_none(self) -> None:
        # 4096*4096*4 = 64 MiB > 16 MiB target.
        result = default_shards(
            chunks=(4096, 4096),
            shape=(8192, 8192),
            dtype=np.dtype("float32"),
            target_bytes=self.TARGET,
        )
        self.assertIsNone(result)

    def test_k_le_one_returns_none(self) -> None:
        # chunk=256x256 float32 = 256 KiB; ratio=1.25 -> k=round(1.25)=1
        # -> returns None.
        chunk_bytes = 256 * 256 * 4
        result = default_shards(
            chunks=(256, 256),
            shape=(4096, 4096),
            dtype=np.dtype("float32"),
            target_bytes=int(chunk_bytes * 1.25),
        )
        self.assertIsNone(result)

    def test_cap_at_array_bounds(self) -> None:
        # 600x600 float32; chunk_bytes = 256 KiB; ratio = 64; k = 8.
        # Uncapped shard would be (2048, 2048) but the array only has
        # 3 chunks per axis (ceil(600/256) = 3), so the cap is (768, 768).
        result = default_shards(
            chunks=(256, 256),
            shape=(600, 600),
            dtype=np.dtype("float32"),
            target_bytes=self.TARGET,
        )
        self.assertEqual(result, (768, 768))

    def test_cell_coadd_psf(self) -> None:
        # (25, 25, 150, 150) float32 with (1, 1, 150, 150) chunks.
        # chunk_bytes = 90 KiB; ratio ~= 186; growable axes are 0 and 1.
        # k = round(sqrt(186)) = 14.
        result = default_shards(
            chunks=(1, 1, 150, 150),
            shape=(25, 25, 150, 150),
            dtype=np.dtype("float32"),
            target_bytes=self.TARGET,
        )
        self.assertEqual(result, (14, 14, 150, 150))

    def test_mismatched_ndim_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "rank"):
            default_shards(
                chunks=(256, 256),
                shape=(4096, 4096, 4096),
                dtype=np.dtype("float32"),
                target_bytes=self.TARGET,
            )

    def test_zero_itemsize_returns_none(self) -> None:
        # void(0) has itemsize 0; defensive guard against degenerate dtypes.
        result = default_shards(
            chunks=(256, 256),
            shape=(4096, 4096),
            dtype=np.dtype("V0"),
            target_bytes=self.TARGET,
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()

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
import unittest

import astropy.io.fits
import astropy.table
import astropy.units as u
import numpy as np
import pydantic

from lsst.images import Box, ColorImage, Image, Mask, MaskedImage, MaskPlane, MaskSchema
from lsst.images.fits._common import ExtensionKey, FitsOpaqueMetadata
from lsst.images.serialization import ArrayReferenceModel

try:
    import zarr

    from lsst.images.zarr import ZarrPointerModel, write
    from lsst.images.zarr._model import ZarrDocument
    from lsst.images.zarr._output_archive import ZarrOutputArchive, build_archive_metadata

    HAVE_ZARR = True
except ImportError:
    HAVE_ZARR = False


class _Sub(pydantic.BaseModel):
    label: str = "sub"


@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class ZarrOutputArchiveSkeletonTestCase(unittest.TestCase):
    """Constructor + serialize_direct / serialize_pointer plumbing."""

    def test_serialize_direct_returns_nested_result(self) -> None:
        archive = ZarrOutputArchive()

        def serializer(arch):
            return _Sub(label="ok")

        result = archive.serialize_direct("red", serializer)
        self.assertEqual(result.label, "ok")

    def test_serialize_pointer_writes_json_subtree(self) -> None:
        archive = ZarrOutputArchive()

        def serializer(arch):
            return _Sub(label="psf")

        pointer = archive.serialize_pointer("psf", serializer, key=12345)
        self.assertIsInstance(pointer, ZarrPointerModel)
        self.assertEqual(pointer.path, "/psf/lsst_json")
        # Cached on second call.
        again = archive.serialize_pointer("psf", serializer, key=12345)
        self.assertEqual(again, pointer)
        # IR holds the JSON bytes as a 1-D uint8 array.
        node = archive.document.root.get("/psf/lsst_json")
        self.assertEqual(str(node.dtype), "uint8")


@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class ZarrOutputArchiveAddArrayTestCase(unittest.TestCase):
    """`add_array` handling for image / variance / mask plus nested arrays."""

    def test_add_image(self) -> None:
        archive = ZarrOutputArchive()
        ref = archive.add_array(np.ones((4, 5), dtype=np.float32), name="image")
        self.assertEqual(ref.source, "zarr:/image")
        self.assertEqual(list(ref.shape), [4, 5])
        node = archive.document.root.get("/image")
        self.assertEqual(node.shape, (4, 5))
        self.assertEqual(node.attributes.extra["_ARRAY_DIMENSIONS"], ["y", "x"])

    def test_add_variance_aligns_to_image_chunks(self) -> None:
        archive = ZarrOutputArchive(chunks={"image": (2, 2)})
        archive.add_array(np.ones((4, 5), dtype=np.float32), name="image")
        archive.add_array(np.ones((4, 5), dtype=np.float64), name="variance")
        var_node = archive.document.root.get("/variance")
        self.assertEqual(tuple(var_node.chunks), (2, 2))

    def test_add_mask_packs_to_2d_with_cf_flag_attrs(self) -> None:
        schema = MaskSchema(
            [
                MaskPlane("BAD", "Bad pixel."),
                MaskPlane("SAT", "Saturated."),
                MaskPlane("CR", "Cosmic ray."),
            ]
        )
        # ``Mask.serialize`` emits the byte axis first when the archive opts
        # into native-mask arrays — shape ``(mask_size, y, x)``.
        in_memory = np.zeros((1, 4, 5), dtype=np.uint8)
        in_memory[0, 0, 0] = 0b1  # BAD
        in_memory[0, 1, 1] = 0b110  # SAT | CR

        archive = ZarrOutputArchive(archive_metadata={"mask_schema": schema})
        archive.add_array(np.ones((4, 5), dtype=np.float32), name="image")
        ref = archive.add_array(in_memory, name="mask")
        self.assertEqual(ref.source, "zarr:/mask")
        node = archive.document.root.get("/mask")
        # 2-D packed integer.
        self.assertEqual(node.shape, (4, 5))
        self.assertEqual(str(node.dtype), "uint8")  # 3 planes -> uint8
        # Bytes packed correctly.
        np.testing.assert_array_equal(node.data[0, 0], 0b1)
        np.testing.assert_array_equal(node.data[1, 1], 0b110)
        # CF flag attrs.
        attrs = node.attributes.extra
        self.assertEqual(attrs["flag_masks"], [1, 2, 4])
        self.assertEqual(attrs["flag_meanings"], "BAD SAT CR")
        self.assertEqual(
            attrs["flag_descriptions"],
            ["Bad pixel.", "Saturated.", "Cosmic ray."],
        )
        self.assertEqual(attrs["_ARRAY_DIMENSIONS"], ["y", "x"])

    def test_add_mask_picks_widest_dtype_for_40_planes(self) -> None:
        planes = [MaskPlane(f"P{i}", f"Plane {i}.") for i in range(40)]
        schema = MaskSchema(planes)
        # 40 planes -> mask_size=5 -> (5, y, x).
        in_memory = np.zeros((5, 4, 5), dtype=np.uint8)

        archive = ZarrOutputArchive(archive_metadata={"mask_schema": schema})
        archive.add_array(np.ones((4, 5), dtype=np.float32), name="image")
        archive.add_array(in_memory, name="mask")
        node = archive.document.root.get("/mask")
        self.assertEqual(node.shape, (4, 5))
        self.assertEqual(str(node.dtype), "uint64")

    def test_add_mask_refuses_more_than_64_planes(self) -> None:
        planes = [MaskPlane(f"P{i}", f"Plane {i}.") for i in range(65)]
        schema = MaskSchema(planes)
        # 65 planes -> mask_size=9 -> (9, y, x).
        in_memory = np.zeros((9, 4, 5), dtype=np.uint8)

        archive = ZarrOutputArchive(archive_metadata={"mask_schema": schema})
        archive.add_array(np.ones((4, 5), dtype=np.float32), name="image")
        with self.assertRaisesRegex(ValueError, "supports up to 64"):
            archive.add_array(in_memory, name="mask")

    def test_add_anonymous_nested_array(self) -> None:
        archive = ZarrOutputArchive()
        ref = archive.add_array(np.ones((3,), dtype=np.float32), name="psf/centroids")
        self.assertEqual(ref.source, "zarr:/psf/centroids")
        self.assertEqual(archive.document.root.get("/psf/centroids").shape, (3,))

    def test_tile_shape_drives_chunks(self) -> None:
        # The caller's per-array tile hint (e.g. a CellCoadd cell shape)
        # becomes the chunk shape, clamped per axis to the array extent.
        archive = ZarrOutputArchive()
        archive.add_array(np.ones((400, 600), dtype=np.float32), name="image", tile_shape=(150, 200))
        self.assertEqual(tuple(archive.document.root.get("/image").chunks), (150, 200))

    def test_tile_shape_clamped_to_array_extent(self) -> None:
        archive = ZarrOutputArchive()
        archive.add_array(np.ones((100, 80), dtype=np.float32), name="image", tile_shape=(150, 200))
        self.assertEqual(tuple(archive.document.root.get("/image").chunks), (100, 80))

    def test_explicit_chunk_override_beats_tile_shape(self) -> None:
        archive = ZarrOutputArchive(chunks={"image": (32, 32)})
        archive.add_array(np.ones((400, 600), dtype=np.float32), name="image", tile_shape=(150, 200))
        self.assertEqual(tuple(archive.document.root.get("/image").chunks), (32, 32))

    def test_options_name_borrows_chunk_override(self) -> None:
        # ``options_name`` lets one array reuse another's overrides (e.g.
        # noise realizations following the image).
        archive = ZarrOutputArchive(chunks={"image": (64, 64)})
        archive.add_array(np.ones((400, 600), dtype=np.float32), name="noise", options_name="image")
        self.assertEqual(tuple(archive.document.root.get("/noise").chunks), (64, 64))


@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class ZarrOutputArchiveAddTableTestCase(unittest.TestCase):
    """`add_table` / `add_structured_array` plumbing."""

    def test_add_table_creates_one_array_per_column(self) -> None:
        archive = ZarrOutputArchive()
        original = astropy.table.Table(
            {
                "x": np.arange(4, dtype=np.int32),
                "y": np.arange(4, dtype=np.float32),
            },
            meta={"comment": "small catalog"},
        )
        model = archive.add_table(original, name="cat")
        self.assertEqual(len(model.columns), 2)
        sources = {c.name: c.data.source for c in model.columns}
        self.assertEqual(sources["x"], "zarr:/lsst/tables/cat/x")
        self.assertEqual(sources["y"], "zarr:/lsst/tables/cat/y")
        # Each column is its own zarr array under the parent group.
        x_node = archive.document.root.get("/lsst/tables/cat/x")
        self.assertEqual(x_node.shape, (4,))

    def test_add_structured_array_writes_column_arrays_with_units(self) -> None:
        rec = np.zeros(3, dtype=[("x", np.float64), ("y", np.int32)])
        rec["x"] = [1.0, 2.0, 3.0]
        rec["y"] = [10, 20, 30]
        archive = ZarrOutputArchive()
        model = archive.add_structured_array(
            rec,
            name="rec",
            units={"x": u.m},
            descriptions={"y": "the y values"},
        )
        self.assertEqual(len(model.columns), 2)
        col_x = next(c for c in model.columns if c.name == "x")
        col_y = next(c for c in model.columns if c.name == "y")
        self.assertIsInstance(col_x.data, ArrayReferenceModel)
        self.assertIsInstance(col_y.data, ArrayReferenceModel)
        self.assertEqual(col_x.unit, u.m)
        self.assertIsNone(col_y.unit)
        self.assertFalse(col_x.description)
        self.assertEqual(col_y.description, "the y values")
        self.assertEqual(col_x.data.source, "zarr:/lsst/tables/rec/x")
        self.assertEqual(col_y.data.source, "zarr:/lsst/tables/rec/y")
        np.testing.assert_array_equal(archive.document.root.get("/lsst/tables/rec/x").data, rec["x"])
        np.testing.assert_array_equal(archive.document.root.get("/lsst/tables/rec/y").data, rec["y"])

    def test_add_structured_array_supports_nested_table_name(self) -> None:
        rec = np.zeros(1, dtype=[("solution", np.float64, (4,))])
        rec["solution"] = [[1.0, 2.0, 3.0, 4.0]]
        archive = ZarrOutputArchive()
        model = archive.add_structured_array(rec, name="psf/piff/interp/solution")
        self.assertEqual(len(model.columns), 1)
        column = model.columns[0]
        self.assertIsInstance(column.data, ArrayReferenceModel)
        self.assertEqual(column.data.source, "zarr:/lsst/tables/psf/piff/interp/solution/solution")
        self.assertEqual(column.data.shape, [4])
        node = archive.document.root.get("/lsst/tables/psf/piff/interp/solution/solution")
        np.testing.assert_array_equal(node.data, rec["solution"])

    def test_add_structured_array_rejects_anonymous(self) -> None:
        rec = np.zeros(2, dtype=[("x", np.float64)])
        archive = ZarrOutputArchive()
        with self.assertRaisesRegex(ValueError, "Anonymous structured arrays"):
            archive.add_structured_array(rec)


@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class ZarrWriteHelperTestCase(unittest.TestCase):
    """Public ``write()`` end-to-end for a plain `Image`."""

    def test_write_image_to_local_directory(self) -> None:
        original = Image(
            np.arange(20, dtype=np.float32).reshape(4, 5),
            bbox=Box.factory[10:14, 20:25],
        )
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "out.zarr")
            tree = write(original, target)
            self.assertIsNotNone(tree)
            with zarr.storage.LocalStore(target, read_only=True) as store:
                doc = ZarrDocument.from_zarr(store)
                # Top-level image and tree are present.
                self.assertIn("image", doc.root.arrays)
                self.assertIn("lsst_json", doc.root.arrays)
                self.assertEqual(doc.root.arrays["image"].shape, (4, 5))
                # LSST root attrs.
                lsst_attrs = doc.root.attributes.lsst
                self.assertEqual(lsst_attrs["archive_class"], "Image")
                self.assertEqual(lsst_attrs["json"], "lsst_json")
                # OME multiscales points at /image; no projection means
                # the unit scale is emitted.
                ome = doc.root.attributes.ome
                self.assertIn("multiscales", ome)
                self.assertEqual(ome["multiscales"][0]["datasets"][0]["path"], "image")
                # Data-model schema URL on the lsst namespace; the container
                # (file-format) version travels as lsst.version (stashed
                # under a private sentinel by ZarrAttributes.load).
                self.assertEqual(lsst_attrs["data_model"], "https://images.lsst.io/schemas/image-1.0.0")
                self.assertEqual(lsst_attrs["__version_remembered_at_load__"], 1)
                self.assertNotIn("data_model", doc.root.attributes.extra)
                self.assertNotIn("version", doc.root.attributes.extra)


@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class ZarrWriteOnDiskShapeTestCase(unittest.TestCase):
    """Pin the on-disk layout for harder archive classes."""

    def _round_trip_doc(self, obj):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "out.zarr")
            write(obj, target)
            with zarr.storage.LocalStore(target, read_only=True) as store:
                return ZarrDocument.from_zarr(store)

    def test_masked_image_layout(self) -> None:
        schema = MaskSchema([MaskPlane("BAD", "Bad pixel.")])
        image = Image(
            np.arange(20, dtype=np.float32).reshape(4, 5),
            bbox=Box.factory[10:14, 20:25],
        )
        masked = MaskedImage(image, mask_schema=schema)
        masked.mask.set("BAD", image.array % 2 == 0)

        doc = self._round_trip_doc(masked)
        self.assertEqual(doc.root.attributes.lsst["archive_class"], "MaskedImage")
        # image / variance / mask are sibling root arrays.
        self.assertIn("image", doc.root.arrays)
        self.assertIn("variance", doc.root.arrays)
        self.assertIn("mask", doc.root.arrays)
        # Mask is 2-D packed integer with CF flag attrs.
        mask = doc.root.arrays["mask"]
        self.assertEqual(mask.shape, (4, 5))
        self.assertEqual(mask.attributes.extra["flag_meanings"], "BAD")
        # CF / xarray dims on every 2-D array.
        for name in ("image", "variance", "mask"):
            self.assertEqual(
                doc.root.arrays[name].attributes.extra["_ARRAY_DIMENSIONS"],
                ["y", "x"],
            )


@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class ZarrColorImageWriteTestCase(unittest.TestCase):
    """ColorImage emits decorated red/green/blue sub-archives."""

    def test_color_image_emits_per_channel_arrays(self) -> None:
        arr = np.zeros((4, 5, 3), dtype=np.uint8)
        arr[..., 0] = 1
        arr[..., 1] = 2
        arr[..., 2] = 3
        color = ColorImage(arr, bbox=Box.factory[10:14, 20:25])

        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "out.zarr")
            write(color, target)
            with zarr.storage.LocalStore(target, read_only=True) as store:
                doc = ZarrDocument.from_zarr(store)
                # Root: ColorImage, no ome.multiscales
                # (axes_for_archive_class returns () for ColorImage).
                self.assertEqual(doc.root.attributes.lsst["archive_class"], "ColorImage")
                self.assertNotIn("multiscales", doc.root.attributes.ome)
                # Each channel is a top-level 2-D array.
                for channel in ("red", "green", "blue"):
                    self.assertIn(channel, doc.root.arrays)
                    self.assertEqual(doc.root.arrays[channel].shape, (4, 5))


@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class ZarrPsfChunkingTestCase(unittest.TestCase):
    """`add_array` defaults a 4-D ``psf`` array to single-cell chunks."""

    def test_psf_array_uses_single_cell_chunks(self) -> None:
        psf = np.zeros((2, 3, 21, 21), dtype=np.float32)
        archive = ZarrOutputArchive(archive_class="CellCoadd")
        ref = archive.add_array(psf, name="psf")
        self.assertEqual(ref.source, "zarr:/psf")
        node = archive.document.root.get("/psf")
        # Single-cell chunks: leading axes are 1; spatial axes match shape.
        self.assertEqual(tuple(node.chunks), (1, 1, 21, 21))

    def test_psf_user_override_wins(self) -> None:
        psf = np.zeros((2, 3, 21, 21), dtype=np.float32)
        archive = ZarrOutputArchive(
            archive_class="CellCoadd",
            chunks={"psf": (2, 3, 21, 21)},
        )
        archive.add_array(psf, name="psf")
        node = archive.document.root.get("/psf")
        self.assertEqual(tuple(node.chunks), (2, 3, 21, 21))

    def test_psf_array_gets_default_shards(self) -> None:
        # 25x25 cells of 150x150 float32: chunk_bytes = 90 KiB,
        # ratio ~ 186, k = round(sqrt(186)) = 14 -> shard (14, 14, 150, 150).
        psf = np.zeros((25, 25, 150, 150), dtype=np.float32)
        archive = ZarrOutputArchive(archive_class="CellCoadd")
        archive.add_array(psf, name="psf")
        node = archive.document.root.get("/psf")
        self.assertEqual(tuple(node.shards), (14, 14, 150, 150))

    def test_psf_user_shard_override_wins(self) -> None:
        psf = np.zeros((25, 25, 150, 150), dtype=np.float32)
        archive = ZarrOutputArchive(
            archive_class="CellCoadd",
            shards={"psf": (5, 5, 150, 150)},
        )
        archive.add_array(psf, name="psf")
        node = archive.document.root.get("/psf")
        self.assertEqual(tuple(node.shards), (5, 5, 150, 150))

    def test_small_psf_shard_caps_at_array_bounds(self) -> None:
        # 2x3 cells of 21x21 float32: chunk_bytes = 1764 B, ratio ~9511,
        # 2 growable axes, k = round(sqrt(9511)) = 98. The cap clamps
        # each growable axis to chunks[i] * ceil(shape[i]/chunks[i]) =
        # 1 * shape[i] = shape[i], yielding shard (2, 3, 21, 21) — the
        # whole 6-cell PSF goes into one shard. Inner axes (21, 21) are
        # not growable since chunks already cover them.
        psf = np.zeros((2, 3, 21, 21), dtype=np.float32)
        archive = ZarrOutputArchive(archive_class="CellCoadd")
        archive.add_array(psf, name="psf")
        node = archive.document.root.get("/psf")
        self.assertEqual(tuple(node.shards), (2, 3, 21, 21))


@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class ZarrOpaqueMetadataWriteTestCase(unittest.TestCase):
    """FITS opaque metadata persists at /lsst/opaque_metadata/fits/primary."""

    def test_fits_opaque_metadata_persists(self) -> None:
        image = Image(
            np.arange(20, dtype=np.float32).reshape(4, 5),
            bbox=Box.factory[10:14, 20:25],
        )
        header = astropy.io.fits.Header()
        header["ORIGIN"] = "RUBIN"
        header["EXPTIME"] = 30.0
        opaque = FitsOpaqueMetadata()
        opaque.headers[ExtensionKey()] = header
        image._opaque_metadata = opaque

        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "out.zarr")
            write(image, target)
            with zarr.storage.LocalStore(target, read_only=True) as store:
                doc = ZarrDocument.from_zarr(store)
                self.assertEqual(
                    doc.root.attributes.lsst.get("opaque_metadata_format"),
                    "fits",
                )
                opaque_node = doc.root.get("/lsst/opaque_metadata/fits/primary")
                # ``(N, 80)`` byte array with explicit dim names.
                self.assertEqual(len(opaque_node.shape), 2)
                self.assertEqual(opaque_node.shape[1], 80)
                self.assertEqual(
                    opaque_node.attributes.extra["_ARRAY_DIMENSIONS"],
                    ["card", "char"],
                )
                # Recover the original header from the raw bytes.
                text = bytes(opaque_node.read()).decode("ascii")
                recovered = astropy.io.fits.Header.fromstring(text)
                self.assertEqual(recovered["ORIGIN"], "RUBIN")
                self.assertEqual(recovered["EXPTIME"], 30.0)


@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class BuildArchiveMetadataTestCase(unittest.TestCase):
    """`build_archive_metadata` resolves the mask schema."""

    def test_mask_schema_from_inner_mask(self) -> None:
        schema = MaskSchema([MaskPlane("BAD", "Bad pixel.")])
        image = Image(np.zeros((4, 5), dtype=np.float32), bbox=Box.factory[0:4, 0:5])
        masked = MaskedImage(image, mask_schema=schema)
        metadata = build_archive_metadata(masked)
        self.assertIs(metadata["mask_schema"], masked.mask.schema)

    def test_mask_schema_for_top_level_mask(self) -> None:
        schema = MaskSchema([MaskPlane("BAD", "Bad pixel.")])
        mask = Mask(
            np.zeros((4, 5, schema.mask_size), dtype=schema.dtype),
            bbox=Box.factory[0:4, 0:5],
            schema=schema,
        )
        metadata = build_archive_metadata(mask)
        self.assertIs(metadata["mask_schema"], schema)


if __name__ == "__main__":
    unittest.main()

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
import numpy as np
import pydantic

from lsst.images import Box, Image
from lsst.images._image import ImageSerializationModel
from lsst.images.fits._common import ExtensionKey, FitsOpaqueMetadata
from lsst.images.serialization import ArchiveReadError
from lsst.images.serialization import read as read_archive

try:
    import zarr

    from lsst.images.serialization import ArrayReferenceModel, NumberType
    from lsst.images.zarr import ZarrPointerModel, write
    from lsst.images.zarr._common import LSST_NS, LSST_VERSION
    from lsst.images.zarr._input_archive import ZarrInputArchive
    from lsst.images.zarr._model import ZarrArray, ZarrDocument
    from lsst.images.zarr._output_archive import ZarrOutputArchive

    HAVE_ZARR = True
except ImportError:
    HAVE_ZARR = False


class _CountingStore(zarr.storage.MemoryStore if HAVE_ZARR else object):
    """A `zarr.storage.MemoryStore` that counts ``get`` calls.

    The counter is shared across instances created by zarr's
    ``with_read_only`` so the test sees every read regardless of which
    store wrapper handles it.
    """

    _shared_counter: list[int] = [0]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @property
    def reads(self) -> int:
        return self._shared_counter[0]

    @reads.setter
    def reads(self, value: int) -> None:
        self._shared_counter[0] = value

    async def get(self, key, prototype, byte_range=None):
        self._shared_counter[0] += 1
        return await super().get(key, prototype, byte_range)


@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class ZarrInputArchiveSkeletonTestCase(unittest.TestCase):
    """Open + version validation + ``get_tree``."""

    def test_open_reads_tree(self) -> None:
        original = Image(
            np.arange(20, dtype=np.float32).reshape(4, 5),
            bbox=Box.factory[10:14, 20:25],
        )
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "out.zarr")
            write(original, target)
            with ZarrInputArchive.open(target) as archive:
                tree = archive.get_tree(ImageSerializationModel)
                self.assertIsNotNone(tree)

    def test_missing_archive_class_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "bare.zarr")
            os.makedirs(target)
            store = zarr.storage.LocalStore(target, read_only=False)
            zarr.create_group(store=store, zarr_format=3)  # no lsst attrs
            with self.assertRaisesRegex(ArchiveReadError, "not an LSST zarr archive"):
                with ZarrInputArchive.open(target):
                    pass

    def test_future_version_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "future.zarr")
            os.makedirs(target)
            store = zarr.storage.LocalStore(target, read_only=False)
            root = zarr.create_group(store=store, zarr_format=3)
            root.update_attributes(
                {
                    LSST_NS: {
                        "version": LSST_VERSION + 1,
                        "archive_class": "Image",
                        "json": "lsst_json",
                    }
                }
            )
            with self.assertRaisesRegex(ArchiveReadError, "container format version"):
                with ZarrInputArchive.open(target):
                    pass

    def test_data_model_is_informational(self) -> None:
        """A bogus ``lsst.data_model`` does not block opening the file.

        It is informational only on read, mirroring FITS DATAMODL / NDF
        DATA_MODEL; the JSON tree's own version fields drive compatibility.
        """
        original = Image(
            np.arange(20, dtype=np.float32).reshape(4, 5),
            bbox=Box.factory[10:14, 20:25],
        )
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "out.zarr")
            write(original, target)
            store = zarr.storage.LocalStore(target, read_only=False)
            root = zarr.open_group(store=store, zarr_format=3)
            attrs = dict(root.attrs)
            attrs[LSST_NS] = {**attrs[LSST_NS], "data_model": "not-a-real-schema-url"}
            root.update_attributes(attrs)
            with ZarrInputArchive.open(target) as archive:
                tree = archive.get_tree(ImageSerializationModel)
                self.assertEqual(tree.schema_url, "https://images.lsst.io/schemas/image-1.0.0")


@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class ZarrInputArchiveLazySubsetTestCase(unittest.TestCase):
    """Subset reads only fetch chunks intersecting the slice."""

    def test_subset_read_touches_only_intersecting_chunks(self) -> None:
        store = _CountingStore()
        root = zarr.create_group(store=store, zarr_format=3)
        root.update_attributes(
            {
                LSST_NS: {
                    "version": LSST_VERSION,
                    "archive_class": "Image",
                    "json": "lsst_json",
                }
            }
        )
        zarr_array = root.create_array(name="image", shape=(16, 16), chunks=(4, 4), dtype="float32")
        zarr_array[:] = np.arange(256, dtype=np.float32).reshape(16, 16)
        # Stub /lsst_json so the input archive's constructor accepts the file.
        tree_arr = root.create_array(name="lsst_json", shape=(2,), chunks=(2,), dtype="uint8")
        tree_arr[:] = np.frombuffer(b"{}", dtype=np.uint8)

        doc = ZarrDocument.from_zarr(store)
        archive = ZarrInputArchive(doc)

        store.reads = 0
        full_ref = ArrayReferenceModel(
            source="zarr:/image",
            shape=[16, 16],
            datatype=NumberType.from_numpy(np.dtype("float32")),
        )
        full = archive.get_array(full_ref)
        full_reads = store.reads
        self.assertEqual(full.shape, (16, 16))

        store.reads = 0
        subset = archive.get_array(full_ref, slices=(slice(0, 4), slice(0, 4)))
        subset_reads = store.reads
        self.assertEqual(subset.shape, (4, 4))
        np.testing.assert_array_equal(subset, np.arange(256).reshape(16, 16)[:4, :4])
        self.assertLess(subset_reads, full_reads)


@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class ZarrInputArchiveMaskUnpackTestCase(unittest.TestCase):
    """Round-trip a packed 2-D mask through ``get_array``'s unpack path."""

    def test_unpack_2d_packed_back_to_3d(self) -> None:
        store = zarr.storage.MemoryStore()
        root = zarr.create_group(store=store, zarr_format=3)
        root.update_attributes(
            {
                LSST_NS: {
                    "version": LSST_VERSION,
                    "archive_class": "Mask",
                    "json": "lsst_json",
                }
            }
        )
        # 4x5 mask, 3 planes -> packed in uint8.
        on_disk = np.zeros((4, 5), dtype=np.uint8)
        on_disk[0, 0] = 0b001
        on_disk[1, 1] = 0b110
        mask_array = root.create_array(name="mask", shape=(4, 5), chunks=(4, 5), dtype="uint8")
        mask_array[:] = on_disk
        mask_array.update_attributes(
            {
                "_ARRAY_DIMENSIONS": ["y", "x"],
                "flag_masks": [1, 2, 4],
                "flag_meanings": "BAD SAT CR",
                "flag_descriptions": ["Bad pixel.", "Saturated.", "Cosmic ray."],
            }
        )
        tree_arr = root.create_array(name="lsst_json", shape=(2,), chunks=(2,), dtype="uint8")
        tree_arr[:] = np.frombuffer(b"{}", dtype=np.uint8)

        doc = ZarrDocument.from_zarr(store)
        archive = ZarrInputArchive(doc)

        # The model records (y, x, mask_size) but the storage layout is the
        # transposed (mask_size, y, x) — Mask.deserialize does the final
        # moveaxis to recover (y, x, mask_size).
        model = ArrayReferenceModel(
            source="zarr:/mask",
            shape=[4, 5, 1],
            datatype=NumberType.from_numpy(np.dtype("uint8")),
        )
        result = archive.get_array(model)
        self.assertEqual(result.shape, (1, 4, 5))
        self.assertEqual(result[0, 0, 0], 0b001)
        self.assertEqual(result[0, 1, 1], 0b110)

    def test_unpack_uint64_with_5_bytes(self) -> None:
        # 40 planes packed into uint64 -> mask_size = 5.
        store = zarr.storage.MemoryStore()
        root = zarr.create_group(store=store, zarr_format=3)
        root.update_attributes(
            {
                LSST_NS: {
                    "version": LSST_VERSION,
                    "archive_class": "Mask",
                    "json": "lsst_json",
                }
            }
        )
        on_disk = np.zeros((4, 5), dtype=np.uint64)
        on_disk[0, 0] = 0x01_02_03_04_05  # arbitrary bit pattern
        mask_array = root.create_array(name="mask", shape=(4, 5), chunks=(4, 5), dtype="uint64")
        mask_array[:] = on_disk
        mask_array.update_attributes(
            {
                "_ARRAY_DIMENSIONS": ["y", "x"],
                "flag_masks": [1 << i for i in range(40)],
                "flag_meanings": " ".join(f"P{i}" for i in range(40)),
                "flag_descriptions": [f"Plane {i}." for i in range(40)],
            }
        )
        tree_arr = root.create_array(name="lsst_json", shape=(2,), chunks=(2,), dtype="uint8")
        tree_arr[:] = np.frombuffer(b"{}", dtype=np.uint8)

        doc = ZarrDocument.from_zarr(store)
        archive = ZarrInputArchive(doc)

        model = ArrayReferenceModel(
            source="zarr:/mask",
            shape=[4, 5, 5],
            datatype=NumberType.from_numpy(np.dtype("uint8")),
        )
        result = archive.get_array(model)
        self.assertEqual(result.shape, (5, 4, 5))
        # Bytes recovered from the packed uint64 (mask_size, y, x order).
        self.assertEqual(result[0, 0, 0], 0x05)  # low byte
        self.assertEqual(result[1, 0, 0], 0x04)
        self.assertEqual(result[2, 0, 0], 0x03)
        self.assertEqual(result[3, 0, 0], 0x02)
        self.assertEqual(result[4, 0, 0], 0x01)


@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class ZarrInputArchivePointerTestCase(unittest.TestCase):
    """``deserialize_pointer`` cache + JSON sub-tree handling."""

    def test_deserialize_pointer_caches_results(self) -> None:
        class _Sub(pydantic.BaseModel):
            label: str

        store = zarr.storage.MemoryStore()
        root = zarr.create_group(store=store, zarr_format=3)
        root.update_attributes(
            {LSST_NS: {"version": LSST_VERSION, "archive_class": "Image", "json": "lsst_json"}}
        )
        # Stub /lsst_json.
        tree_arr = root.create_array(name="lsst_json", shape=(2,), chunks=(2,), dtype="uint8")
        tree_arr[:] = np.frombuffer(b"{}", dtype=np.uint8)
        # Sub-archive with its own /lsst_json at /psf/lsst_json.
        json_bytes = b'{"label": "psf"}'
        psf = root.create_group("psf")
        arr = psf.create_array(
            name="lsst_json", shape=(len(json_bytes),), chunks=(len(json_bytes),), dtype="uint8"
        )
        arr[:] = np.frombuffer(json_bytes, dtype=np.uint8)

        doc = ZarrDocument.from_zarr(store)
        archive = ZarrInputArchive(doc)

        deserialize_calls: list[int] = []

        def deserializer(model, arch):
            deserialize_calls.append(1)
            return model

        pointer = ZarrPointerModel(path="/psf/lsst_json")
        first = archive.deserialize_pointer(pointer, _Sub, deserializer)
        second = archive.deserialize_pointer(pointer, _Sub, deserializer)
        self.assertEqual(first.label, "psf")
        self.assertIs(first, second)
        self.assertEqual(len(deserialize_calls), 1)


@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class ZarrInputArchiveTableTestCase(unittest.TestCase):
    """``get_table`` reconstructs columns via ``get_array``."""

    def test_get_table_reconstructs_columns(self) -> None:
        out = ZarrOutputArchive()
        out.document.root.attributes.lsst["archive_class"] = "Image"
        out.document.root.attributes.lsst["json"] = "lsst_json"
        out.document.root.arrays["lsst_json"] = ZarrArray(data=np.frombuffer(b"{}", dtype=np.uint8))
        original = astropy.table.Table(
            {
                "x": np.arange(4, dtype=np.int32),
                "y": np.arange(4, dtype=np.float32),
            }
        )
        model = out.add_table(original, name="cat")

        store = zarr.storage.MemoryStore()
        out.document.to_zarr(store)
        doc = ZarrDocument.from_zarr(store)
        inp = ZarrInputArchive(doc)

        recovered = inp.get_table(model)
        self.assertEqual(recovered.colnames, ["x", "y"])
        np.testing.assert_array_equal(recovered["x"], original["x"])
        np.testing.assert_array_equal(recovered["y"], original["y"])


@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class ZarrReadHelperTestCase(unittest.TestCase):
    """End-to-end public ``read()`` round-trip."""

    def test_round_trip_image(self) -> None:
        original = Image(
            np.arange(20, dtype=np.float32).reshape(4, 5),
            bbox=Box.factory[10:14, 20:25],
        )
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "out.zarr")
            write(original, target)
            result = read_archive(target, Image)
            self.assertEqual(result.array.shape, (4, 5))
            np.testing.assert_array_equal(result.array, original.array)
            self.assertEqual(result.bbox, original.bbox)


@unittest.skipUnless(HAVE_ZARR, "zarr is not installed")
class ZarrOpaqueMetadataReadTestCase(unittest.TestCase):
    """FITS opaque metadata is restored on read."""

    def test_fits_opaque_metadata_round_trips(self) -> None:
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
            recovered = read_archive(target, Image)
            recovered_opaque = recovered._opaque_metadata
            self.assertIsInstance(recovered_opaque, FitsOpaqueMetadata)
            recovered_header = recovered_opaque.headers[ExtensionKey()]
            self.assertEqual(recovered_header["ORIGIN"], "RUBIN")
            self.assertEqual(recovered_header["EXPTIME"], 30.0)

    def test_fits_opaque_metadata_preserves_full_card_fidelity(self) -> None:
        """Comments, HISTORY, COMMENT, and HIERARCH all survive round-trip."""
        image = Image(
            np.arange(20, dtype=np.float32).reshape(4, 5),
            bbox=Box.factory[10:14, 20:25],
        )
        header = astropy.io.fits.Header()
        header["ORIGIN"] = ("RUBIN", "Source observatory")
        header["EXPTIME"] = (30.0, "[s] Total exposure time")
        header["HIERARCH LSST INSTRUMENT"] = "LSSTCam"
        header.add_history("Bias-subtracted on 2026-05-21")
        header.add_history("ISR completed 2026-05-22")
        header.add_comment("This file was generated for testing.")
        opaque = FitsOpaqueMetadata()
        opaque.headers[ExtensionKey()] = header
        image._opaque_metadata = opaque

        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "out.zarr")
            write(image, target)
            recovered = read_archive(target, Image)
            recovered_header = recovered._opaque_metadata.headers[ExtensionKey()]
            # Byte-exact equality of the serialized card stream.
            self.assertEqual(recovered_header.tostring(), header.tostring())
            # Spot-check the round-tripped values + comments.
            self.assertEqual(recovered_header.comments["ORIGIN"], "Source observatory")
            self.assertEqual(recovered_header.comments["EXPTIME"], "[s] Total exposure time")
            self.assertEqual(recovered_header["HIERARCH LSST INSTRUMENT"], "LSSTCam")
            self.assertEqual(
                list(recovered_header["HISTORY"]),
                ["Bias-subtracted on 2026-05-21", "ISR completed 2026-05-22"],
            )
            self.assertEqual(
                list(recovered_header["COMMENT"]),
                ["This file was generated for testing."],
            )


if __name__ == "__main__":
    unittest.main()

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
from pathlib import Path

import astropy.io.fits
import astropy.table
import numpy as np
import pydantic
import pytest

from lsst.images import Box, Image
from lsst.images._image import ImageSerializationModel
from lsst.images.fits._common import ExtensionKey, FitsOpaqueMetadata
from lsst.images.serialization import ArchiveReadError, read_archive

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

skip_no_zarr = pytest.mark.skipif(not HAVE_ZARR, reason="zarr is not installed")


class _CountingStore(zarr.storage.MemoryStore if HAVE_ZARR else object):  # type: ignore[misc]
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


def make_image() -> Image:
    """Return a freshly-constructed 4x5 float32 Image with a non-trivial
    bounding box.
    """
    return Image(
        np.arange(20, dtype=np.float32).reshape(4, 5),
        bbox=Box.factory[10:14, 20:25],
    )


@skip_no_zarr
def test_open_reads_tree(tmp_path: Path) -> None:
    """Verify opening a written archive yields a serialization tree."""
    original = make_image()
    target = str(tmp_path / "out.zarr")
    write(original, target)
    with ZarrInputArchive.open(target) as archive:
        tree = archive.get_tree(ImageSerializationModel)
        assert tree is not None


@skip_no_zarr
def test_missing_archive_class_raises(tmp_path: Path) -> None:
    """Verify a zarr group without LSST attributes is rejected on open."""
    target = str(tmp_path / "bare.zarr")
    os.makedirs(target)
    store = zarr.storage.LocalStore(target, read_only=False)
    zarr.create_group(store=store, zarr_format=3)  # no lsst attrs
    with pytest.raises(ArchiveReadError, match="not an LSST zarr archive"):
        with ZarrInputArchive.open(target):
            pass


@skip_no_zarr
def test_future_version_refused(tmp_path: Path) -> None:
    """Verify an archive with a future container format version is
    refused.
    """
    target = str(tmp_path / "future.zarr")
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
    with pytest.raises(ArchiveReadError, match="container format version"):
        with ZarrInputArchive.open(target):
            pass


@skip_no_zarr
def test_data_model_is_informational(tmp_path: Path) -> None:
    """A bogus ``lsst.data_model`` does not block opening the file.

    It is informational only on read, mirroring FITS DATAMODL / NDF
    DATA_MODEL; the JSON tree's own version fields drive compatibility.
    """
    original = make_image()
    target = str(tmp_path / "out.zarr")
    write(original, target)
    store = zarr.storage.LocalStore(target, read_only=False)
    root = zarr.open_group(store=store, zarr_format=3)
    attrs = dict(root.attrs)
    attrs[LSST_NS] = {**attrs[LSST_NS], "data_model": "not-a-real-schema-url"}  # type: ignore[dict-item]
    root.update_attributes(attrs)
    with ZarrInputArchive.open(target) as archive:
        tree = archive.get_tree(ImageSerializationModel)
        assert tree.schema_url == "https://images.lsst.io/schemas/image-1.0.0"


@skip_no_zarr
def test_subset_read_touches_only_intersecting_chunks() -> None:
    """Subset reads only fetch chunks intersecting the slice."""
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
    assert full.shape == (16, 16)

    store.reads = 0
    subset = archive.get_array(full_ref, slices=(slice(0, 4), slice(0, 4)))
    subset_reads = store.reads
    assert subset.shape == (4, 4)
    np.testing.assert_array_equal(subset, np.arange(256).reshape(16, 16)[:4, :4])
    assert subset_reads < full_reads


@skip_no_zarr
def test_unpack_2d_packed_back_to_3d() -> None:
    """Round-trip a packed 2-D mask through ``get_array``'s unpack path."""
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
    assert result.shape == (1, 4, 5)
    assert result[0, 0, 0] == 0b001
    assert result[0, 1, 1] == 0b110


@skip_no_zarr
def test_unpack_uint64_with_5_bytes() -> None:
    """Unpack a mask whose 40 planes were packed into uint64 storage."""
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
    assert result.shape == (5, 4, 5)
    # Bytes recovered from the packed uint64 (mask_size, y, x order).
    assert result[0, 0, 0] == 0x05  # low byte
    assert result[1, 0, 0] == 0x04
    assert result[2, 0, 0] == 0x03
    assert result[3, 0, 0] == 0x02
    assert result[4, 0, 0] == 0x01


@skip_no_zarr
def test_deserialize_pointer_caches_results() -> None:
    """``deserialize_pointer`` caches results and handles JSON sub-trees."""

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
    first = archive.deserialize_pointer(pointer, _Sub, deserializer)  # type: ignore[type-var]
    second = archive.deserialize_pointer(pointer, _Sub, deserializer)  # type: ignore[type-var]
    assert first.label == "psf"
    assert first is second
    assert len(deserialize_calls) == 1


@skip_no_zarr
def test_get_table_reconstructs_columns() -> None:
    """``get_table`` reconstructs columns via ``get_array``."""
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
    assert recovered.colnames == ["x", "y"]
    np.testing.assert_array_equal(recovered["x"], original["x"])
    np.testing.assert_array_equal(recovered["y"], original["y"])


@skip_no_zarr
def test_round_trip_image(tmp_path: Path) -> None:
    """End-to-end public ``read()`` round-trip."""
    original = make_image()
    target = str(tmp_path / "out.zarr")
    write(original, target)
    result = read_archive(target, Image)
    assert result.array.shape == (4, 5)
    np.testing.assert_array_equal(result.array, original.array)
    assert result.bbox == original.bbox


@skip_no_zarr
def test_fits_opaque_metadata_round_trips(tmp_path: Path) -> None:
    """FITS opaque metadata is restored on read."""
    image = make_image()
    header = astropy.io.fits.Header()
    header["ORIGIN"] = "RUBIN"
    header["EXPTIME"] = 30.0
    opaque = FitsOpaqueMetadata()
    opaque.headers[ExtensionKey()] = header
    image._opaque_metadata = opaque

    target = str(tmp_path / "out.zarr")
    write(image, target)
    recovered = read_archive(target, Image)
    recovered_opaque = recovered._opaque_metadata
    assert isinstance(recovered_opaque, FitsOpaqueMetadata)
    recovered_header = recovered_opaque.headers[ExtensionKey()]
    assert recovered_header["ORIGIN"] == "RUBIN"
    assert recovered_header["EXPTIME"] == 30.0


@skip_no_zarr
def test_fits_opaque_metadata_preserves_full_card_fidelity(tmp_path: Path) -> None:
    """Comments, HISTORY, COMMENT, and HIERARCH all survive round-trip."""
    image = make_image()
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

    target = str(tmp_path / "out.zarr")
    write(image, target)
    recovered = read_archive(target, Image)
    recovered_opaque = recovered._opaque_metadata
    assert isinstance(recovered_opaque, FitsOpaqueMetadata)
    recovered_header = recovered_opaque.headers[ExtensionKey()]
    # Byte-exact equality of the serialized card stream.
    assert recovered_header.tostring() == header.tostring()
    # Spot-check the round-tripped values + comments.
    assert recovered_header.comments["ORIGIN"] == "Source observatory"
    assert recovered_header.comments["EXPTIME"] == "[s] Total exposure time"
    assert recovered_header["HIERARCH LSST INSTRUMENT"] == "LSSTCam"
    assert list(recovered_header["HISTORY"]) == [
        "Bias-subtracted on 2026-05-21",
        "ISR completed 2026-05-22",
    ]
    assert list(recovered_header["COMMENT"]) == ["This file was generated for testing."]

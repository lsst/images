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

from pathlib import Path

import astropy.io.fits
import astropy.table
import astropy.units as u
import numpy as np
import pydantic
import pytest

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

skip_no_zarr = pytest.mark.skipif(not HAVE_ZARR, reason="zarr is not installed")


class _Sub(pydantic.BaseModel):
    label: str = "sub"


@skip_no_zarr
def test_serialize_direct_returns_nested_result() -> None:
    """Verify serialize_direct invokes the serializer and returns its
    result.
    """
    archive = ZarrOutputArchive()
    result = archive.serialize_direct("red", lambda nested: _Sub(label="ok"))
    assert result.label == "ok"


@skip_no_zarr
def test_serialize_pointer_writes_json_subtree() -> None:
    """Verify serialize_pointer writes a JSON subtree and caches by key."""
    archive = ZarrOutputArchive()
    pointer = archive.serialize_pointer("psf", lambda nested: _Sub(label="psf"), key=12345)
    assert isinstance(pointer, ZarrPointerModel)
    assert pointer.path == "/psf/lsst_json"
    # Cached on second call.
    again = archive.serialize_pointer("psf", lambda nested: _Sub(label="psf"), key=12345)
    assert again == pointer
    # IR holds the JSON bytes as a 1-D uint8 array.
    node = archive.document.root.get("/psf/lsst_json")
    assert str(node.dtype) == "uint8"


@skip_no_zarr
def test_add_image() -> None:
    """Verify add_array routes a top-level image array to /image."""
    archive = ZarrOutputArchive()
    ref = archive.add_array(np.ones((4, 5), dtype=np.float32), name="image")
    assert ref.source == "zarr:/image"
    assert list(ref.shape) == [4, 5]
    node = archive.document.root.get("/image")
    assert node.shape == (4, 5)
    assert node.attributes.extra["_ARRAY_DIMENSIONS"] == ["y", "x"]


@skip_no_zarr
def test_add_variance_aligns_to_image_chunks() -> None:
    """Verify the variance array inherits the image's chunk shape."""
    archive = ZarrOutputArchive(chunks={"image": (2, 2)})
    archive.add_array(np.ones((4, 5), dtype=np.float32), name="image")
    archive.add_array(np.ones((4, 5), dtype=np.float64), name="variance")
    var_node = archive.document.root.get("/variance")
    assert tuple(var_node.chunks) == (2, 2)


@skip_no_zarr
def test_add_mask_packs_to_2d_with_cf_flag_attrs() -> None:
    """Verify add_array packs a mask to a 2-D integer array with CF flag
    attributes.
    """
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
    assert ref.source == "zarr:/mask"
    node = archive.document.root.get("/mask")
    # 2-D packed integer.
    assert node.shape == (4, 5)
    assert str(node.dtype) == "uint8"  # 3 planes -> uint8
    # Bytes packed correctly.
    np.testing.assert_array_equal(node.data[0, 0], 0b1)
    np.testing.assert_array_equal(node.data[1, 1], 0b110)
    # CF flag attrs.
    attrs = node.attributes.extra
    assert attrs["flag_masks"] == [1, 2, 4]
    assert attrs["flag_meanings"] == "BAD SAT CR"
    assert attrs["flag_descriptions"] == ["Bad pixel.", "Saturated.", "Cosmic ray."]
    assert attrs["_ARRAY_DIMENSIONS"] == ["y", "x"]


@skip_no_zarr
def test_add_mask_picks_widest_dtype_for_40_planes() -> None:
    """Verify a 40-plane mask packs to a uint64 array."""
    planes = [MaskPlane(f"P{i}", f"Plane {i}.") for i in range(40)]
    schema = MaskSchema(planes)
    # 40 planes -> mask_size=5 -> (5, y, x).
    in_memory = np.zeros((5, 4, 5), dtype=np.uint8)

    archive = ZarrOutputArchive(archive_metadata={"mask_schema": schema})
    archive.add_array(np.ones((4, 5), dtype=np.float32), name="image")
    archive.add_array(in_memory, name="mask")
    node = archive.document.root.get("/mask")
    assert node.shape == (4, 5)
    assert str(node.dtype) == "uint64"


@skip_no_zarr
def test_add_mask_refuses_more_than_64_planes() -> None:
    """Verify add_array raises ValueError for a mask with more than 64
    planes.
    """
    planes = [MaskPlane(f"P{i}", f"Plane {i}.") for i in range(65)]
    schema = MaskSchema(planes)
    # 65 planes -> mask_size=9 -> (9, y, x).
    in_memory = np.zeros((9, 4, 5), dtype=np.uint8)

    archive = ZarrOutputArchive(archive_metadata={"mask_schema": schema})
    archive.add_array(np.ones((4, 5), dtype=np.float32), name="image")
    with pytest.raises(ValueError, match="supports up to 64"):
        archive.add_array(in_memory, name="mask")


@skip_no_zarr
def test_add_anonymous_nested_array() -> None:
    """Verify a nested array name produces a nested zarr array path."""
    archive = ZarrOutputArchive()
    ref = archive.add_array(np.ones((3,), dtype=np.float32), name="psf/centroids")
    assert ref.source == "zarr:/psf/centroids"
    assert archive.document.root.get("/psf/centroids").shape == (3,)


@skip_no_zarr
def test_tile_shape_drives_chunks() -> None:
    """Verify the per-array tile hint becomes the chunk shape."""
    # The caller's per-array tile hint (e.g. a CellCoadd cell shape)
    # becomes the chunk shape, clamped per axis to the array extent.
    archive = ZarrOutputArchive()
    archive.add_array(np.ones((400, 600), dtype=np.float32), name="image", tile_shape=(150, 200))
    assert tuple(archive.document.root.get("/image").chunks) == (150, 200)


@skip_no_zarr
def test_tile_shape_clamped_to_array_extent() -> None:
    """Verify tile-derived chunks are clamped per axis to the array extent."""
    archive = ZarrOutputArchive()
    archive.add_array(np.ones((100, 80), dtype=np.float32), name="image", tile_shape=(150, 200))
    assert tuple(archive.document.root.get("/image").chunks) == (100, 80)


@skip_no_zarr
def test_explicit_chunk_override_beats_tile_shape() -> None:
    """Verify an explicit chunk override takes precedence over the tile
    hint.
    """
    archive = ZarrOutputArchive(chunks={"image": (32, 32)})
    archive.add_array(np.ones((400, 600), dtype=np.float32), name="image", tile_shape=(150, 200))
    assert tuple(archive.document.root.get("/image").chunks) == (32, 32)


@skip_no_zarr
def test_options_name_borrows_chunk_override() -> None:
    """Verify options_name lets one array reuse another's chunk overrides."""
    # ``options_name`` lets one array reuse another's overrides (e.g.
    # noise realizations following the image).
    archive = ZarrOutputArchive(chunks={"image": (64, 64)})
    archive.add_array(np.ones((400, 600), dtype=np.float32), name="noise", options_name="image")
    assert tuple(archive.document.root.get("/noise").chunks) == (64, 64)


@skip_no_zarr
def test_add_table_creates_one_array_per_column() -> None:
    """Verify add_table writes one zarr array per column under
    /lsst/tables.
    """
    archive = ZarrOutputArchive()
    original = astropy.table.Table(
        {
            "x": np.arange(4, dtype=np.int32),
            "y": np.arange(4, dtype=np.float32),
        },
        meta={"comment": "small catalog"},
    )
    model = archive.add_table(original, name="cat")
    assert len(model.columns) == 2
    sources = {c.name: c.data.source for c in model.columns}
    assert sources["x"] == "zarr:/lsst/tables/cat/x"
    assert sources["y"] == "zarr:/lsst/tables/cat/y"
    # Each column is its own zarr array under the parent group.
    x_node = archive.document.root.get("/lsst/tables/cat/x")
    assert x_node.shape == (4,)


@skip_no_zarr
def test_add_structured_array_writes_column_arrays_with_units() -> None:
    """Verify add_structured_array stores each column as a zarr array with
    correct units and descriptions.
    """
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
    assert len(model.columns) == 2
    col_x = next(c for c in model.columns if c.name == "x")
    col_y = next(c for c in model.columns if c.name == "y")
    assert isinstance(col_x.data, ArrayReferenceModel)
    assert isinstance(col_y.data, ArrayReferenceModel)
    assert col_x.unit == u.m
    assert col_y.unit is None
    assert not col_x.description
    assert col_y.description == "the y values"
    assert col_x.data.source == "zarr:/lsst/tables/rec/x"
    assert col_y.data.source == "zarr:/lsst/tables/rec/y"
    np.testing.assert_array_equal(archive.document.root.get("/lsst/tables/rec/x").data, rec["x"])
    np.testing.assert_array_equal(archive.document.root.get("/lsst/tables/rec/y").data, rec["y"])


@skip_no_zarr
def test_add_structured_array_supports_nested_table_name() -> None:
    """Verify add_structured_array supports nested table names."""
    rec = np.zeros(1, dtype=[("solution", np.float64, (4,))])
    rec["solution"] = [[1.0, 2.0, 3.0, 4.0]]
    archive = ZarrOutputArchive()
    model = archive.add_structured_array(rec, name="psf/piff/interp/solution")
    assert len(model.columns) == 1
    column = model.columns[0]
    assert isinstance(column.data, ArrayReferenceModel)
    assert column.data.source == "zarr:/lsst/tables/psf/piff/interp/solution/solution"
    assert column.data.shape == [4]
    node = archive.document.root.get("/lsst/tables/psf/piff/interp/solution/solution")
    np.testing.assert_array_equal(node.data, rec["solution"])


@skip_no_zarr
def test_add_structured_array_rejects_anonymous() -> None:
    """Verify add_structured_array raises ValueError without a name."""
    rec = np.zeros(2, dtype=[("x", np.float64)])
    archive = ZarrOutputArchive()
    with pytest.raises(ValueError, match="Anonymous structured arrays"):
        archive.add_structured_array(rec)


@skip_no_zarr
def test_write_image_to_local_directory(tmp_path: Path) -> None:
    """Verify the public write() end-to-end for a plain Image."""
    original = Image(
        np.arange(20, dtype=np.float32).reshape(4, 5),
        bbox=Box.factory[10:14, 20:25],
    )
    target = str(tmp_path / "out.zarr")
    tree = write(original, target)
    assert tree is not None
    with zarr.storage.LocalStore(target, read_only=True) as store:
        doc = ZarrDocument.from_zarr(store)
        # Top-level image and tree are present.
        assert "image" in doc.root.arrays
        assert "lsst_json" in doc.root.arrays
        assert doc.root.arrays["image"].shape == (4, 5)
        # LSST root attrs.
        lsst_attrs = doc.root.attributes.lsst
        assert lsst_attrs["archive_class"] == "Image"
        assert lsst_attrs["json"] == "lsst_json"
        # OME multiscales points at /image; no projection means
        # the unit scale is emitted.
        ome = doc.root.attributes.ome
        assert "multiscales" in ome
        assert ome["multiscales"][0]["datasets"][0]["path"] == "image"
        # Data-model schema URL on the lsst namespace; the container
        # (file-format) version travels as lsst.version (stashed
        # under a private sentinel by ZarrAttributes.load).
        assert lsst_attrs["data_model"] == "https://images.lsst.io/schemas/image-1.0.0"
        assert lsst_attrs["__version_remembered_at_load__"] == 1
        assert "data_model" not in doc.root.attributes.extra
        assert "version" not in doc.root.attributes.extra


@skip_no_zarr
def test_masked_image_layout(tmp_path: Path) -> None:
    """Pin the on-disk layout for a MaskedImage."""
    schema = MaskSchema([MaskPlane("BAD", "Bad pixel.")])
    image = Image(
        np.arange(20, dtype=np.float32).reshape(4, 5),
        bbox=Box.factory[10:14, 20:25],
    )
    masked = MaskedImage(image, mask_schema=schema)
    masked.mask.set("BAD", image.array % 2 == 0)

    target = str(tmp_path / "out.zarr")
    write(masked, target)
    with zarr.storage.LocalStore(target, read_only=True) as store:
        doc = ZarrDocument.from_zarr(store)
    assert doc.root.attributes.lsst["archive_class"] == "MaskedImage"
    # image / variance / mask are sibling root arrays.
    assert "image" in doc.root.arrays
    assert "variance" in doc.root.arrays
    assert "mask" in doc.root.arrays
    # Mask is 2-D packed integer with CF flag attrs.
    mask = doc.root.arrays["mask"]
    assert mask.shape == (4, 5)
    assert mask.attributes.extra["flag_meanings"] == "BAD"
    # CF / xarray dims on every 2-D array.
    for name in ("image", "variance", "mask"):
        assert doc.root.arrays[name].attributes.extra["_ARRAY_DIMENSIONS"] == ["y", "x"]


@skip_no_zarr
def test_color_image_emits_per_channel_arrays(tmp_path: Path) -> None:
    """Verify ColorImage emits decorated red/green/blue sub-archives."""
    arr = np.zeros((4, 5, 3), dtype=np.uint8)
    arr[..., 0] = 1
    arr[..., 1] = 2
    arr[..., 2] = 3
    color = ColorImage(arr, bbox=Box.factory[10:14, 20:25])

    target = str(tmp_path / "out.zarr")
    write(color, target)
    with zarr.storage.LocalStore(target, read_only=True) as store:
        doc = ZarrDocument.from_zarr(store)
        # Root: ColorImage, no ome.multiscales
        # (axes_for_archive_class returns () for ColorImage).
        assert doc.root.attributes.lsst["archive_class"] == "ColorImage"
        assert "multiscales" not in doc.root.attributes.ome
        # Each channel is a top-level 2-D array.
        for channel in ("red", "green", "blue"):
            assert channel in doc.root.arrays
            assert doc.root.arrays[channel].shape == (4, 5)


@skip_no_zarr
def test_psf_array_uses_single_cell_chunks() -> None:
    """Verify add_array defaults a 4-D psf array to single-cell chunks."""
    psf = np.zeros((2, 3, 21, 21), dtype=np.float32)
    archive = ZarrOutputArchive(archive_class="CellCoadd")
    ref = archive.add_array(psf, name="psf")
    assert ref.source == "zarr:/psf"
    node = archive.document.root.get("/psf")
    # Single-cell chunks: leading axes are 1; spatial axes match shape.
    assert tuple(node.chunks) == (1, 1, 21, 21)


@skip_no_zarr
def test_psf_user_override_wins() -> None:
    """Verify a user chunk override beats the default psf chunking."""
    psf = np.zeros((2, 3, 21, 21), dtype=np.float32)
    archive = ZarrOutputArchive(
        archive_class="CellCoadd",
        chunks={"psf": (2, 3, 21, 21)},
    )
    archive.add_array(psf, name="psf")
    node = archive.document.root.get("/psf")
    assert tuple(node.chunks) == (2, 3, 21, 21)


@skip_no_zarr
def test_psf_array_gets_default_shards() -> None:
    """Verify a large psf array receives default shards."""
    # 25x25 cells of 150x150 float32: chunk_bytes = 90 KiB,
    # ratio ~ 186, k = round(sqrt(186)) = 14 -> shard (14, 14, 150, 150).
    psf = np.zeros((25, 25, 150, 150), dtype=np.float32)
    archive = ZarrOutputArchive(archive_class="CellCoadd")
    archive.add_array(psf, name="psf")
    node = archive.document.root.get("/psf")
    assert tuple(node.shards) == (14, 14, 150, 150)


@skip_no_zarr
def test_psf_user_shard_override_wins() -> None:
    """Verify a user shard override beats the default psf sharding."""
    psf = np.zeros((25, 25, 150, 150), dtype=np.float32)
    archive = ZarrOutputArchive(
        archive_class="CellCoadd",
        shards={"psf": (5, 5, 150, 150)},
    )
    archive.add_array(psf, name="psf")
    node = archive.document.root.get("/psf")
    assert tuple(node.shards) == (5, 5, 150, 150)


@skip_no_zarr
def test_small_psf_shard_caps_at_array_bounds() -> None:
    """Verify shard growth is capped at the array bounds for a small psf."""
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
    assert tuple(node.shards) == (2, 3, 21, 21)


@skip_no_zarr
def test_fits_opaque_metadata_persists(tmp_path: Path) -> None:
    """Verify FITS opaque metadata persists at
    /lsst/opaque_metadata/fits/primary.
    """
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

    target = str(tmp_path / "out.zarr")
    write(image, target)
    with zarr.storage.LocalStore(target, read_only=True) as store:
        doc = ZarrDocument.from_zarr(store)
        assert doc.root.attributes.lsst.get("opaque_metadata_format") == "fits"
        opaque_node = doc.root.get("/lsst/opaque_metadata/fits/primary")
        # ``(N, 80)`` byte array with explicit dim names.
        assert len(opaque_node.shape) == 2
        assert opaque_node.shape[1] == 80
        assert opaque_node.attributes.extra["_ARRAY_DIMENSIONS"] == ["card", "char"]
        # Recover the original header from the raw bytes.
        text = bytes(opaque_node.read()).decode("ascii")
        recovered = astropy.io.fits.Header.fromstring(text)
        assert recovered["ORIGIN"] == "RUBIN"
        assert recovered["EXPTIME"] == 30.0


@skip_no_zarr
def test_mask_schema_from_inner_mask() -> None:
    """Verify build_archive_metadata resolves the mask schema from a
    MaskedImage's inner mask.
    """
    schema = MaskSchema([MaskPlane("BAD", "Bad pixel.")])
    image = Image(np.zeros((4, 5), dtype=np.float32), bbox=Box.factory[0:4, 0:5])
    masked = MaskedImage(image, mask_schema=schema)
    metadata = build_archive_metadata(masked)
    assert metadata["mask_schema"] is masked.mask.schema


@skip_no_zarr
def test_mask_schema_for_top_level_mask() -> None:
    """Verify build_archive_metadata resolves the mask schema for a top-level
    Mask.
    """
    schema = MaskSchema([MaskPlane("BAD", "Bad pixel.")])
    mask = Mask(
        np.zeros((4, 5, schema.mask_size), dtype=schema.dtype),
        bbox=Box.factory[0:4, 0:5],
        schema=schema,
    )
    metadata = build_archive_metadata(mask)
    assert metadata["mask_schema"] is schema

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
import astropy.units as u
import numpy as np
import pydantic
import pytest

from lsst.images import Box, Image, ImageSerializationModel, Mask, MaskedImage
from lsst.images._transforms import FrameSet
from lsst.images.fits import ExtensionKey, FitsOpaqueMetadata
from lsst.images.serialization import (
    ArchiveReadError,
    ArrayReferenceModel,
    InlineArrayModel,
    NumberType,
    read_archive,
)

try:
    import h5py

    from lsst.images.ndf import (
        NdfInputArchive,
        NdfOutputArchive,
        NdfPointerModel,
        _hds,
        read_starlink,
        write,
    )

    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False

skip_no_h5py = pytest.mark.skipif(not HAVE_H5PY, reason="h5py is not installed")


@skip_no_h5py
def test_open_round_trips_image_tree(tmp_path: Path) -> None:
    """Verify NdfInputArchive.open round-trips an Image's ArchiveTree."""
    image = Image(
        np.arange(20, dtype=np.float32).reshape(4, 5),
        bbox=Box.factory[10:14, 20:25],
    )
    path = tmp_path / "test.sdf"
    written_tree = write(image, path)
    with NdfInputArchive.open(path) as archive:
        tree = archive.get_tree(type(written_tree))
        assert tree.model_dump_json() == written_tree.model_dump_json()


@skip_no_h5py
def test_get_tree_raises_when_main_json_missing(tmp_path: Path) -> None:
    """Verify get_tree raises ArchiveReadError when /MORE/LSST/JSON is
    absent.
    """
    path = tmp_path / "test.sdf"
    with h5py.File(path, "w") as f:
        f["/"].attrs["CLASS"] = "NDF"
    with NdfInputArchive.open(path) as archive:
        model_type = ImageSerializationModel[NdfPointerModel]
        with pytest.raises(ArchiveReadError):
            archive.get_tree(model_type)


@skip_no_h5py
def test_get_array_reads_image_array(tmp_path: Path) -> None:
    """Verify get_array returns the full image array from the NDF file."""
    image = Image(np.arange(20, dtype=np.float32).reshape(4, 5))
    path = tmp_path / "test.sdf"
    tree = write(image, path)
    with NdfInputArchive.open(path) as archive:
        arr = archive.get_array(tree.data)
        np.testing.assert_array_equal(arr, image.array)


@skip_no_h5py
def test_get_array_supports_slicing(tmp_path: Path) -> None:
    """Verify get_array returns the correct sub-array when slices are given."""
    image = Image(np.arange(20, dtype=np.float32).reshape(4, 5))
    path = tmp_path / "test.sdf"
    tree = write(image, path)
    with NdfInputArchive.open(path) as archive:
        arr = archive.get_array(tree.data, slices=(slice(0, 2), slice(1, 4)))
        np.testing.assert_array_equal(arr, image.array[:2, 1:4])


@skip_no_h5py
def test_get_array_handles_inline_array(tmp_path: Path) -> None:
    """Verify get_array converts an InlineArrayModel to a numpy array."""
    inline = InlineArrayModel(data=[1.0, 2.0, 3.0], datatype=NumberType.float64)
    image = Image(np.zeros((2, 2), dtype=np.float32))
    path = tmp_path / "test.sdf"
    write(image, path)
    with NdfInputArchive.open(path) as archive:
        arr = archive.get_array(inline)
        np.testing.assert_array_equal(arr, np.array([1.0, 2.0, 3.0]))


@skip_no_h5py
def test_get_array_unrecognised_source_raises(tmp_path: Path) -> None:
    """Verify get_array raises ArchiveReadError for an unrecognised source
    scheme.
    """
    image = Image(np.zeros((2, 2), dtype=np.float32))
    bogus = ArrayReferenceModel(source="fits:NOTUS", shape=[2, 2], datatype=NumberType.float32)
    path = tmp_path / "test.sdf"
    write(image, path)
    with NdfInputArchive.open(path) as archive:
        with pytest.raises(ArchiveReadError):
            archive.get_array(bogus)


@skip_no_h5py
def test_deserialize_pointer_round_trips_subtree(tmp_path: Path) -> None:
    """Verify deserialize_pointer reconstructs a hoisted sub-tree correctly."""

    class TinyTree(pydantic.BaseModel):
        name: str

    path = tmp_path / "test.sdf"
    with h5py.File(path, "w") as f:
        arch = NdfOutputArchive(f)
        ptr = arch.serialize_pointer("psf", lambda nested: TinyTree(name="hello"), key=("psf", 1))
    with NdfInputArchive.open(path) as archive:
        result = archive.deserialize_pointer(ptr, TinyTree, lambda m, _a: m)
        assert result.name == "hello"


@skip_no_h5py
def test_deserialize_pointer_caches_by_ref(tmp_path: Path) -> None:
    """Verify deserialize_pointer calls the deserializer only once for the same
    ref.
    """

    class TinyTree(pydantic.BaseModel):
        name: str

    calls: list[TinyTree] = []

    def deserializer(model: TinyTree, _archive: NdfInputArchive) -> TinyTree:
        calls.append(model)
        return model

    path = tmp_path / "test.sdf"
    with h5py.File(path, "w") as f:
        arch = NdfOutputArchive(f)
        ptr = arch.serialize_pointer("psf", lambda nested: TinyTree(name="x"), key=("psf", 1))
    with NdfInputArchive.open(path) as archive:
        first = archive.deserialize_pointer(ptr, TinyTree, deserializer)
        second = archive.deserialize_pointer(ptr, TinyTree, deserializer)
        assert first is second
        assert len(calls) == 1


@skip_no_h5py
def test_deserialize_pointer_caches_frame_set_for_get_frame_set(tmp_path: Path) -> None:
    """Verify a deserialized FrameSet is returned by get_frame_set via the
    shared cache.
    """

    class TinyTree(pydantic.BaseModel):
        name: str

    class DummyFrameSet(FrameSet):
        def __contains__(self, frame: object) -> bool:
            return False

        def __getitem__(self, key: object) -> object:
            raise AssertionError("DummyFrameSet should not be indexed in this test.")

    sentinel = DummyFrameSet()

    path = tmp_path / "test.sdf"
    with h5py.File(path, "w") as f:
        arch = NdfOutputArchive(f)
        ptr = arch.serialize_frame_set(
            "frames",
            sentinel,
            lambda nested: TinyTree(name="frames"),
            key=("frames", 1),
        )
    with NdfInputArchive.open(path) as archive:
        result = archive.deserialize_pointer(ptr, TinyTree, lambda _m, _a: sentinel)
        assert result is sentinel
        assert archive.get_frame_set(ptr) is sentinel


@skip_no_h5py
def test_get_frame_set_returns_cached_value(tmp_path: Path) -> None:
    """Verify get_frame_set returns a sentinel previously placed in the
    cache.
    """
    sentinel = object()
    path = tmp_path / "test.sdf"
    write(Image(np.zeros((2, 2), dtype=np.float32)), path)
    with NdfInputArchive.open(path) as archive:
        archive._frame_set_cache["/MORE/LSST/PIXEL_TO_SKY"] = sentinel
        pointer = NdfPointerModel(path="/MORE/LSST/PIXEL_TO_SKY")
        assert archive.get_frame_set(pointer) is sentinel


@skip_no_h5py
def test_get_frame_set_raises_if_not_cached(tmp_path: Path) -> None:
    """Verify get_frame_set raises AssertionError when the pointer is not in
    the cache.
    """
    path = tmp_path / "test.sdf"
    write(Image(np.zeros((2, 2), dtype=np.float32)), path)
    with NdfInputArchive.open(path) as archive:
        pointer = NdfPointerModel(path="/MORE/LSST/UNKNOWN")
        with pytest.raises(AssertionError):
            archive.get_frame_set(pointer)


@skip_no_h5py
def test_more_fits_round_trips_via_opaque_metadata(tmp_path: Path) -> None:
    """Verify /MORE/FITS headers are recovered by get_opaque_metadata."""
    image = Image(np.zeros((2, 2), dtype=np.float32))
    primary = astropy.io.fits.Header()
    primary["FOO"] = ("bar", "test card")
    opaque = FitsOpaqueMetadata()
    opaque.add_header(primary, name="", ver=1)
    image._opaque_metadata = opaque
    path = tmp_path / "test.sdf"
    write(image, path)
    with NdfInputArchive.open(path) as archive:
        recovered = archive.get_opaque_metadata()
        assert ExtensionKey() in recovered.headers
        assert recovered.headers[ExtensionKey()]["FOO"] == "bar"


@skip_no_h5py
def test_get_opaque_metadata_empty_when_no_more_fits(tmp_path: Path) -> None:
    """Verify get_opaque_metadata returns an empty FitsOpaqueMetadata when
    /MORE/FITS is absent.
    """
    image = Image(np.zeros((2, 2), dtype=np.float32))
    path = tmp_path / "test.sdf"
    write(image, path)
    with NdfInputArchive.open(path) as archive:
        recovered = archive.get_opaque_metadata()
        assert isinstance(recovered, FitsOpaqueMetadata)
        assert not recovered.headers


@skip_no_h5py
def test_read_round_trips_image(tmp_path: Path) -> None:
    """Verify read_archive() round-trips an Image through an NDF file."""
    image = Image(
        np.arange(20, dtype=np.float32).reshape(4, 5),
        bbox=Box.factory[10:14, 20:25],
    )
    path = tmp_path / "test.sdf"
    write(image, path)
    result = read_archive(path, Image)
    assert isinstance(result, Image)
    np.testing.assert_array_equal(result.array, image.array)
    assert result.bbox == image.bbox


@skip_no_h5py
def test_read_starlink_file_auto_detects_image(tmp_path: Path) -> None:
    """Verify read_starlink auto-detects an Image from a schema-less NDF
    fixture.
    """
    example_path = os.path.join(os.path.dirname(__file__), "data", "example-ndf.sdf")
    result = read_starlink(Image, example_path)
    assert isinstance(result, Image)
    assert result.array.shape == (611, 609)
    assert result.array.dtype == np.int16
    assert result.sky_projection is not None


@skip_no_h5py
def test_read_starlink_file_recovers_opaque_fits_metadata(tmp_path: Path) -> None:
    """Verify read_starlink recovers /MORE/FITS opaque metadata from a Starlink
    NDF.
    """
    example_path = os.path.join(os.path.dirname(__file__), "data", "example-ndf.sdf")
    result = read_starlink(Image, example_path)
    opaque = result._opaque_metadata
    assert ExtensionKey() in opaque.headers
    primary = opaque.headers[ExtensionKey()]
    assert "NAXIS" in primary


@skip_no_h5py
def test_read_auto_detects_nested_quality_array(tmp_path: Path) -> None:
    """Verify read_starlink maps a QUALITY array to a MaskedImage mask."""
    image_array = np.arange(6, dtype=np.float32).reshape(2, 3)
    quality_array = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.uint8)

    path = tmp_path / "test.sdf"
    with h5py.File(path, "w") as f:
        _hds.set_root_name(f, "TEST", "NDF")
        data_array = _hds.create_structure(f, "DATA_ARRAY", "ARRAY")
        _hds.write_array(data_array, "DATA", image_array)
        quality = _hds.create_structure(f, "QUALITY", "QUALITY")
        quality_array_struct = _hds.create_structure(quality, "QUALITY", "ARRAY")
        _hds.write_array(quality_array_struct, "DATA", quality_array)
        _hds.write_array(quality_array_struct, "ORIGIN", np.array([0, 0], dtype=np.int32))
        _hds.write_array(quality_array_struct, "BAD_PIXEL", np.array(False, dtype=np.bool_))
        _hds.write_array(quality, "BADBITS", np.array(1, dtype=np.uint8))
    result = read_starlink(MaskedImage, path)
    assert isinstance(result, MaskedImage)
    np.testing.assert_array_equal(result.mask.array[:, :, 0], quality_array)
    assert set(result.mask.schema.names) == {f"MASK{i}" for i in range(8)}
    image_result = read_starlink(Image, path)
    assert isinstance(image_result, Image)
    np.testing.assert_array_equal(image_result.array, image_array)


@skip_no_h5py
def test_read_auto_detect_preserves_quality_bits(tmp_path: Path) -> None:
    """Verify read_starlink correctly maps BADBITS-selected quality planes."""
    image_array = np.arange(6, dtype=np.float32).reshape(2, 3)
    quality_array = np.array([[0, 2, 4], [2, 0, 6]], dtype=np.uint8)
    expected_mask1 = np.array([[0, 1, 0], [1, 0, 1]], dtype=bool)
    expected_mask2 = np.array([[0, 0, 1], [0, 0, 1]], dtype=bool)

    path = tmp_path / "test.sdf"
    with h5py.File(path, "w") as f:
        _hds.set_root_name(f, "TEST", "NDF")
        data_array = _hds.create_structure(f, "DATA_ARRAY", "ARRAY")
        _hds.write_array(data_array, "DATA", image_array)
        quality = _hds.create_structure(f, "QUALITY", "QUALITY")
        quality_array_struct = _hds.create_structure(quality, "QUALITY", "ARRAY")
        _hds.write_array(quality_array_struct, "DATA", quality_array)
        _hds.write_array(quality_array_struct, "ORIGIN", np.array([0, 0], dtype=np.int32))
        _hds.write_array(quality_array_struct, "BAD_PIXEL", np.array(False, dtype=np.bool_))
        _hds.write_array(quality, "BADBITS", np.array(2, dtype=np.uint8))
    result = read_starlink(MaskedImage, path)
    assert isinstance(result, MaskedImage)
    mask = result.mask
    np.testing.assert_array_equal(mask.array[:, :, 0], quality_array)
    np.testing.assert_array_equal(mask.get("MASK1"), expected_mask1)
    np.testing.assert_array_equal(mask.get("MASK2"), expected_mask2)
    assert "Selected by BADBITS" in mask.schema.descriptions["MASK1"]
    assert "Selected by BADBITS" not in mask.schema.descriptions["MASK2"]


@skip_no_h5py
def test_read_auto_detected_data_only_as_masked_image_uses_defaults(tmp_path: Path) -> None:
    """Verify read_starlink fills in default mask and variance when only
    DATA_ARRAY is present.
    """
    image_array = np.arange(6, dtype=np.float32).reshape(2, 3)

    path = tmp_path / "test.sdf"
    with h5py.File(path, "w") as f:
        _hds.set_root_name(f, "TEST", "NDF")
        data_array = _hds.create_structure(f, "DATA_ARRAY", "ARRAY")
        _hds.write_array(data_array, "DATA", image_array)
        _hds.write_array(data_array, "ORIGIN", np.array([5, 4], dtype=np.int32))
    result = read_starlink(MaskedImage, path)
    assert isinstance(result, MaskedImage)
    assert result.bbox == Box.factory[4:6, 5:8]
    np.testing.assert_array_equal(result.image.array, image_array)
    np.testing.assert_array_equal(result.mask.array, np.zeros((2, 3, 1), dtype=np.uint8))
    np.testing.assert_array_equal(result.variance.array, np.ones((2, 3), dtype=np.float32))


@skip_no_h5py
def test_read_auto_detected_variance_as_masked_image_keeps_variance(tmp_path: Path) -> None:
    """Verify read_starlink preserves a VARIANCE component in the
    MaskedImage.
    """
    image_array = np.arange(6, dtype=np.float32).reshape(2, 3)
    variance_array = np.full((2, 3), 2.5, dtype=np.float32)

    path = tmp_path / "test.sdf"
    with h5py.File(path, "w") as f:
        _hds.set_root_name(f, "TEST", "NDF")
        data_array = _hds.create_structure(f, "DATA_ARRAY", "ARRAY")
        _hds.write_array(data_array, "DATA", image_array)
        _hds.write_array(data_array, "ORIGIN", np.array([5, 4], dtype=np.int32))
        variance = _hds.create_structure(f, "VARIANCE", "ARRAY")
        _hds.write_array(variance, "DATA", variance_array)
        _hds.write_array(variance, "ORIGIN", np.array([5, 4], dtype=np.int32))
    result = read_starlink(MaskedImage, path)
    assert isinstance(result, MaskedImage)
    np.testing.assert_array_equal(result.variance.array, variance_array)
    np.testing.assert_array_equal(result.mask.array, np.zeros((2, 3, 1), dtype=np.uint8))


@skip_no_h5py
def test_read_auto_detected_units_component(tmp_path: Path) -> None:
    """Verify read_starlink maps a UNITS NDF component to the Image unit."""
    image_array = np.arange(6, dtype=np.float32).reshape(2, 3)

    path = tmp_path / "test.sdf"
    with h5py.File(path, "w") as f:
        _hds.set_root_name(f, "TEST", "NDF")
        data_array = _hds.create_structure(f, "DATA_ARRAY", "ARRAY")
        _hds.write_array(data_array, "DATA", image_array)
        f.create_dataset("UNITS", data=np.bytes_("count"))
    result = read_starlink(Image, path)
    assert result.unit == u.ct


@skip_no_h5py
def test_read_missing_data_array_raises(tmp_path: Path) -> None:
    """Verify read_starlink raises ArchiveReadError for an NDF with no
    DATA_ARRAY or JSON.
    """
    path = tmp_path / "test.sdf"
    with h5py.File(path, "w") as f:
        f["/"].attrs["CLASS"] = "NDF"
    with pytest.raises(ArchiveReadError):
        read_starlink(Image, path)


@skip_no_h5py
def test_read_auto_detect_wrong_target_type_raises(tmp_path: Path) -> None:
    """Verify read_starlink raises ArchiveReadError when the target type is
    unsupported.
    """
    example_path = os.path.join(os.path.dirname(__file__), "data", "example-ndf.sdf")
    with pytest.raises(ArchiveReadError):
        read_starlink(Mask, example_path)

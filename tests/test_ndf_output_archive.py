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

import json
from pathlib import Path
from unittest import mock

import astropy.io.fits
import astropy.table
import astropy.units as u
import numpy as np
import pydantic
import pytest

from lsst.images import Box, Image, MaskedImage, MaskPlane, MaskSchema
from lsst.images._transforms import FrameLookupError, FrameSet, Transform
from lsst.images._transforms._frames import DetectorFrame, Frame
from lsst.images.fits import ExtensionKey, FitsOpaqueMetadata
from lsst.images.serialization import ArrayReferenceModel, InlineArrayModel, read
from lsst.images.serialization import open as open_archive
from lsst.images.tests import make_random_sky_projection

try:
    import h5py

    from lsst.images.ndf import (
        NdfInputArchive,
        NdfOutputArchive,
        _hds,
        write,
    )
    from lsst.images.ndf._hds import DAT__SZNAM

    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False

skip_no_h5py = pytest.mark.skipif(not HAVE_H5PY, reason="h5py is not installed")


class TinyFrameSet(FrameSet):
    """Minimal concrete frame-set for archive bookkeeping tests."""

    def __contains__(self, frame: Frame) -> bool:
        return False

    def __getitem__[I: Frame, O: Frame](self, key: tuple[I, O]) -> Transform[I, O]:
        raise FrameLookupError(key)


class TinyTree(pydantic.BaseModel):
    """A trivial Pydantic model used as a serialization stand-in."""

    name: str


@skip_no_h5py
def test_serialize_direct_calls_serializer_with_nested_archive(tmp_path: Path) -> None:
    """Verify serialize_direct invokes the serializer and returns its
    result.
    """
    path = str(tmp_path / "test.sdf")
    with h5py.File(path, "w") as f:
        arch = NdfOutputArchive(f)
        tree = arch.serialize_direct("top", lambda nested: TinyTree(name="hello"))
        assert tree.name == "hello"


@skip_no_h5py
def test_constructor_marks_root_as_ndf(tmp_path: Path) -> None:
    """Verify the NdfOutputArchive constructor sets CLASS=NDF on the root
    group.
    """
    path = str(tmp_path / "test.sdf")
    with h5py.File(path, "w") as f:
        NdfOutputArchive(f)
    with h5py.File(path, "r") as f:
        assert f["/"].attrs["CLASS"] == b"NDF"


@skip_no_h5py
def test_top_level_image_routes_to_data_array(tmp_path: Path) -> None:
    """Verify add_array routes a top-level image array to /DATA_ARRAY/DATA."""
    data = np.arange(20, dtype=np.float32).reshape(4, 5)
    path = str(tmp_path / "test.sdf")
    with h5py.File(path, "w") as f:
        arch = NdfOutputArchive(f)
        ref = arch.add_array(data, name="image")
        assert ref.source == "ndf:/DATA_ARRAY/DATA"
    with h5py.File(path, "r") as f:
        ds = f["/DATA_ARRAY/DATA"]
        assert ds.dtype == np.float32
        np.testing.assert_array_equal(ds[()], data)
        assert f["/DATA_ARRAY"].attrs["CLASS"] == b"ARRAY"
        origin = f["/DATA_ARRAY/ORIGIN"]
        assert origin.dtype == np.int64
        assert origin.shape == (2,)


@skip_no_h5py
def test_top_level_variance_routes_to_variance(tmp_path: Path) -> None:
    """Verify add_array routes a top-level variance array to /VARIANCE/DATA."""
    data = np.full((3, 3), 0.5, dtype=np.float64)
    path = str(tmp_path / "test.sdf")
    with h5py.File(path, "w") as f:
        arch = NdfOutputArchive(f)
        ref = arch.add_array(data, name="variance")
        assert ref.source == "ndf:/VARIANCE/DATA"
    with h5py.File(path, "r") as f:
        assert f["/VARIANCE"].attrs["CLASS"] == b"ARRAY"
        assert f["/VARIANCE/DATA"].dtype == np.float64


@skip_no_h5py
def test_top_level_compatible_mask_routes_to_quality(tmp_path: Path) -> None:
    """Verify add_array routes a 2D uint8 mask to /QUALITY/QUALITY/DATA."""
    data = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint8)
    path = str(tmp_path / "test.sdf")
    with h5py.File(path, "w") as f:
        arch = NdfOutputArchive(f)
        ref = arch.add_array(data, name="mask")
        assert ref.source == "ndf:/QUALITY/QUALITY/DATA"
    with h5py.File(path, "r") as f:
        assert f["/QUALITY"].attrs["CLASS"] == b"QUALITY"
        assert f["/QUALITY/QUALITY"].attrs["CLASS"] == b"ARRAY"
        assert f["/QUALITY/QUALITY/DATA"].dtype == np.uint8
        np.testing.assert_array_equal(f["/QUALITY/QUALITY/DATA"][()], data)
        assert f["/QUALITY/QUALITY/ORIGIN"].dtype == np.int32
        assert f["/QUALITY/QUALITY/ORIGIN"].shape == (2,)
        assert f["/QUALITY/QUALITY/BAD_PIXEL"].id.get_type().get_class() == h5py.h5t.BITFIELD
        assert not _hds.read_array(f["/QUALITY/QUALITY/BAD_PIXEL"])
        assert f["/QUALITY/BADBITS"][()] == 255


@skip_no_h5py
def test_top_level_incompatible_mask_routes_to_more_lsst(tmp_path: Path) -> None:
    """Verify add_array hoists a 3D uint8 mask to /MORE/LSST/MASK as a sub-
    NDF.
    """
    data = np.zeros((2, 3, 4), dtype=np.uint8)
    data[0, 1, 2] = 4
    data[1, 2, 3] = 8
    expected_quality = np.any(data != 0, axis=0).astype(np.uint8)
    path = str(tmp_path / "test.sdf")
    with h5py.File(path, "w") as f:
        arch = NdfOutputArchive(f)
        ref = arch.add_array(data, name="mask")
        assert ref.source == "ndf:/MORE/LSST/MASK/DATA_ARRAY/DATA"
    with h5py.File(path, "r") as f:
        assert f["/MORE/LSST/MASK"].attrs["CLASS"] == b"NDF"
        assert f["/MORE/LSST/MASK/DATA_ARRAY"].attrs["CLASS"] == b"ARRAY"
        assert f["/MORE/LSST/MASK/DATA_ARRAY/DATA"].shape == data.shape
        assert f["/QUALITY/QUALITY"].attrs["CLASS"] == b"ARRAY"
        np.testing.assert_array_equal(f["/QUALITY/QUALITY/DATA"][()], expected_quality)
        assert f["/QUALITY/BADBITS"][()] == 255
        origin = f["/MORE/LSST/MASK/DATA_ARRAY/ORIGIN"]
        assert origin.dtype == np.int64
        assert origin.shape == (3,)


@skip_no_h5py
def test_long_hoisted_component_is_shrunk(tmp_path: Path) -> None:
    """Verify HDS component names exceeding DAT__SZNAM are shrunk in the stored
    path.
    """
    data = np.array([[1.0, 2.0]], dtype=np.float32)
    path = str(tmp_path / "test.sdf")
    with h5py.File(path, "w") as f:
        arch = NdfOutputArchive(f)
        ref = arch.add_array(data, name="noise_realizations/0")
        assert ref.source.startswith("ndf:/MORE/LSST/")
        assert ref.source.endswith("/DATA_ARRAY/DATA")
    with h5py.File(path, "r") as f:
        hdf5_path = ref.source[len("ndf:") :]
        for component in hdf5_path.strip("/").split("/"):
            assert len(component) <= DAT__SZNAM
        assert hdf5_path in f


@skip_no_h5py
def test_long_name_round_trips_through_input_archive(tmp_path: Path) -> None:
    """Verify a long-named array can be read back via NdfInputArchive."""
    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    path = str(tmp_path / "test.sdf")
    with h5py.File(path, "w") as f:
        arch = NdfOutputArchive(f)
        ref = arch.add_array(data, name="noise_realizations/0")
    with NdfInputArchive.open(path) as inp:
        read_back = inp.get_array(ref)
    np.testing.assert_array_equal(read_back, data)


@skip_no_h5py
def test_repeated_long_name_gets_distinct_versioned_paths(tmp_path: Path) -> None:
    """Verify two identically-named long arrays receive distinct versioned
    paths.
    """
    data = np.array([[1.0]], dtype=np.float32)
    path = str(tmp_path / "test.sdf")
    with h5py.File(path, "w") as f:
        arch = NdfOutputArchive(f)
        first = arch.add_array(data, name="noise_realizations_value")
        second = arch.add_array(data, name="noise_realizations_value")
        assert first.source != second.source
        second_leaf = second.source[len("ndf:") :].split("/")[-3]
        assert second_leaf.endswith("_2")
    with h5py.File(path, "r") as f:
        assert first.source[len("ndf:") :] in f
        assert second.source[len("ndf:") :] in f


@skip_no_h5py
def test_nested_array_hoists_as_sub_ndf(tmp_path: Path) -> None:
    """Verify nested array names produce a CLASS=NDF sub-structure under
    /MORE/LSST.
    """
    data = np.array([[1.0, 2.0]], dtype=np.float32)
    path = str(tmp_path / "test.sdf")
    with h5py.File(path, "w") as f:
        arch = NdfOutputArchive(f)
        ref = arch.add_array(data, name="psf/coefficients")
        assert ref.source == "ndf:/MORE/LSST/PSF/COEFFICIENTS/DATA_ARRAY/DATA"
    with h5py.File(path, "r") as f:
        assert "MORE" in f
        assert "LSST" in f["/MORE"]
        assert "PSF" in f["/MORE/LSST"]
        assert "COEFFICIENTS" in f["/MORE/LSST/PSF"]
        sub = f["/MORE/LSST/PSF/COEFFICIENTS"]
        assert sub.attrs["CLASS"] == b"NDF"
        assert sub["DATA_ARRAY"].attrs["CLASS"] == b"ARRAY"
        np.testing.assert_array_equal(sub["DATA_ARRAY/DATA"][()], data)
        origin = sub["DATA_ARRAY/ORIGIN"]
        assert origin.dtype == np.int64
        assert origin.shape == (data.ndim,)


@skip_no_h5py
def test_colliding_shrunk_names_raise(tmp_path: Path) -> None:
    """Verify add_array raises ValueError when two long names shrink to the
    same token.
    """
    data = np.array([[1.0]], dtype=np.float32)
    path = str(tmp_path / "test.sdf")
    with h5py.File(path, "w") as f:
        arch = NdfOutputArchive(f)
        with mock.patch.object(
            arch._name_shrinker,
            "shrink",
            side_effect=lambda name, *a, **k: name.upper() if len(name) <= DAT__SZNAM else "CLASH",
        ):
            arch.add_array(data, name="long_component_name_one")
            with pytest.raises(ValueError, match="name collision"):
                arch.add_array(data, name="long_component_name_two")


@skip_no_h5py
def test_serialize_pointer_writes_subtree_and_returns_pointer(tmp_path: Path) -> None:
    """Verify serialize_pointer stores the sub-tree JSON and returns the
    correct pointer.
    """
    path = str(tmp_path / "test.sdf")
    with h5py.File(path, "w") as f:
        arch = NdfOutputArchive(f)
        ptr = arch.serialize_pointer(
            "psf",
            lambda nested: TinyTree(name="gaussian"),
            key=("psf", 1),
        )
        assert ptr.path == "/MORE/LSST/PSF/JSON"
    with h5py.File(path, "r") as f:
        raw = f["/MORE/LSST/PSF/JSON"][()]
        joined = b"".join(raw).decode("ascii").rstrip(" ")
        assert '"name":"gaussian"' in joined.replace(" ", "")


@skip_no_h5py
def test_serialize_pointer_caches_by_key(tmp_path: Path) -> None:
    """Verify serialize_pointer returns the cached pointer and does not re-run
    the serializer.
    """
    path = str(tmp_path / "test.sdf")
    with h5py.File(path, "w") as f:
        arch = NdfOutputArchive(f)
        ptr1 = arch.serialize_pointer(
            "psf",
            lambda nested: TinyTree(name="first"),
            key=("psf", 1),
        )
        ptr2 = arch.serialize_pointer(
            "psf",
            lambda nested: TinyTree(name="second"),
            key=("psf", 1),
        )
        assert ptr1 == ptr2
    with h5py.File(path, "r") as f:
        raw = f["/MORE/LSST/PSF/JSON"][()]
        joined = b"".join(raw).decode("ascii").rstrip(" ")
        assert "first" in joined
        assert "second" not in joined


@skip_no_h5py
def test_serialize_pointer_preserves_nested_arrays(tmp_path: Path) -> None:
    """Verify serialize_pointer does not clobber nested arrays written by the
    serializer.
    """

    class TreeWithArray(pydantic.BaseModel):
        name: str
        data: ArrayReferenceModel

    payload = np.arange(6, dtype=np.float32).reshape(2, 3)
    path = str(tmp_path / "test.sdf")
    with h5py.File(path, "w") as f:
        arch = NdfOutputArchive(f)
        ptr = arch.serialize_pointer(
            "psf",
            lambda nested: TreeWithArray(
                name="gaussian",
                data=nested.add_array(payload, name="parameters"),
            ),
            key=("psf", 1),
        )
        assert ptr.path == "/MORE/LSST/PSF/JSON"
    with h5py.File(path, "r") as f:
        assert "/MORE/LSST/PSF/JSON" in f
        assert "/MORE/LSST/PSF/PARAMETERS/DATA_ARRAY/DATA" in f
        np.testing.assert_array_equal(f["/MORE/LSST/PSF/PARAMETERS/DATA_ARRAY/DATA"][()], payload)
        assert f["/MORE/LSST/PSF"].attrs["CLASS"] == b"PSF"
        assert f["/MORE/LSST"].attrs["CLASS"] == b"LSST"


@skip_no_h5py
def test_serialize_frame_set_records_for_iter(tmp_path: Path) -> None:
    """Verify serialize_frame_set records the (FrameSet, pointer) pair for
    iter_frame_sets.
    """
    frame_set = TinyFrameSet()
    path = str(tmp_path / "test.sdf")
    with h5py.File(path, "w") as f:
        arch = NdfOutputArchive(f)
        ptr = arch.serialize_frame_set(
            "wcs/pixel_to_sky",
            frame_set,
            lambda nested: TinyTree(name="proj"),
            key=("frame_set", 1),
        )
        assert ptr.path == "/MORE/LSST/WCS/PIXEL_TO_SKY/JSON"
        recorded = list(arch.iter_frame_sets())
        assert len(recorded) == 1
        assert recorded[0][0] is frame_set
        assert recorded[0][1].path == "/MORE/LSST/WCS/PIXEL_TO_SKY/JSON"


@skip_no_h5py
def test_add_table_returns_inline_table_model(tmp_path: Path) -> None:
    """Verify add_table returns an inline table model for a simple astropy
    Table.
    """
    t = astropy.table.Table({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    path = str(tmp_path / "test.sdf")
    with h5py.File(path, "w") as f:
        arch = NdfOutputArchive(f)
        model = arch.add_table(t, name="some_table")
        assert len(model.columns) == 2
        assert isinstance(model.columns[0].data, InlineArrayModel)


@skip_no_h5py
def test_add_structured_array_writes_column_ndfs_with_units(tmp_path: Path) -> None:
    """Verify add_structured_array stores each column as a sub-NDF with correct
    units.
    """
    rec = np.zeros(3, dtype=[("x", np.float64), ("y", np.int32)])
    rec["x"] = [1.0, 2.0, 3.0]
    rec["y"] = [10, 20, 30]
    path = str(tmp_path / "test.sdf")
    with h5py.File(path, "w") as f:
        arch = NdfOutputArchive(f)
        model = arch.add_structured_array(
            rec,
            name="rec",
            units={"x": u.m},
            descriptions={"y": "the y values"},
        )
        assert len(model.columns) == 2
        assert isinstance(model.columns[0].data, ArrayReferenceModel)
        col_x = next(c for c in model.columns if c.name == "x")
        col_y = next(c for c in model.columns if c.name == "y")
        assert col_x.unit == u.m
        assert col_y.description == "the y values"
        assert col_x.data.source == "ndf:/MORE/LSST/REC/X/DATA_ARRAY/DATA"
        assert col_y.data.source == "ndf:/MORE/LSST/REC/Y/DATA_ARRAY/DATA"
    with h5py.File(path, "r") as f:
        assert f["/MORE/LSST/REC/X"].attrs["CLASS"] == b"NDF"
        np.testing.assert_array_equal(f["/MORE/LSST/REC/X/DATA_ARRAY/DATA"][()], rec["x"])
        assert f["/MORE/LSST/REC/Y"].attrs["CLASS"] == b"NDF"
        np.testing.assert_array_equal(f["/MORE/LSST/REC/Y/DATA_ARRAY/DATA"][()], rec["y"])
    with NdfInputArchive.open(path) as archive:
        recovered = archive.get_structured_array(model)
        np.testing.assert_array_equal(recovered, rec)


@skip_no_h5py
def test_add_single_column_structured_array_uses_table_name(tmp_path: Path) -> None:
    """Verify a single-column structured array uses the table path as its NDF
    component name.
    """
    rec = np.zeros(1, dtype=[("solution", np.float64, (4,))])
    rec["solution"] = [[1.0, 2.0, 3.0, 4.0]]
    path = str(tmp_path / "test.sdf")
    with h5py.File(path, "w") as f:
        arch = NdfOutputArchive(f)
        model = arch.add_structured_array(rec, name="psf/piff/interp/solution")
        assert len(model.columns) == 1
        column = model.columns[0]
        assert isinstance(column.data, ArrayReferenceModel)
        assert column.data.source == "ndf:/MORE/LSST/PSF/PIFF/INTERP/SOLUTION/DATA_ARRAY/DATA"
        assert column.data.shape == [4]
    with h5py.File(path, "r") as f:
        assert "PSF" in f["/MORE/LSST"]
        assert "PIFF" in f["/MORE/LSST/PSF"]
        assert "INTERP" in f["/MORE/LSST/PSF/PIFF"]
        assert "SOLUTION" in f["/MORE/LSST/PSF/PIFF/INTERP"]
        np.testing.assert_array_equal(
            f["/MORE/LSST/PSF/PIFF/INTERP/SOLUTION/DATA_ARRAY/DATA"][()],
            rec["solution"],
        )


@skip_no_h5py
def test_structured_array_long_name_is_shrunk_and_versioned(tmp_path: Path) -> None:
    """Verify long structured-array names are shrunk and repeated names get
    versioned paths.
    """
    dtype = np.dtype([("alpha", "f8"), ("beta", "i4")])
    arr = np.zeros(3, dtype=dtype)
    path = str(tmp_path / "test.sdf")
    with h5py.File(path, "w") as f:
        arch = NdfOutputArchive(f)
        first = arch.add_structured_array(arr, name="catalog_of_long_named_sources")
        second = arch.add_structured_array(arr, name="catalog_of_long_named_sources")
        for model in (first, second):
            for column in model.columns:
                token = column.data.source[len("ndf:") :]
                for component in token.strip("/").split("/"):
                    assert len(component) <= DAT__SZNAM
        assert first.columns[0].data.source != second.columns[0].data.source
        second_parent = second.columns[0].data.source[len("ndf:") :].strip("/").split("/")[-4]
        assert second_parent.endswith("_2")
    with h5py.File(path, "r") as f:
        for model in (first, second):
            for column in model.columns:
                assert column.data.source[len("ndf:") :] in f


@skip_no_h5py
def test_write_with_projection_creates_wcs_component(tmp_path: Path) -> None:
    """Verify write() creates a /WCS/DATA component when the image has a
    sky_projection.
    """
    rng = np.random.default_rng(42)
    det_frame = DetectorFrame(instrument="TestInst", detector=4, bbox=Box.factory[1:4096, 1:4096])
    bbox = Box.factory[10:14, 20:25]
    sky_projection = make_random_sky_projection(rng, det_frame, Box.factory[1:4096, 1:4096])
    image = Image(
        np.arange(20, dtype=np.float32).reshape(4, 5),
        bbox=bbox,
        sky_projection=sky_projection,
    )
    path = str(tmp_path / "test.sdf")
    write(image, path)
    with h5py.File(path, "r") as f:
        assert "WCS" in f
        assert f["/WCS"].attrs["CLASS"] == b"WCS"
        wcs_data = f["/WCS/DATA"]
        assert wcs_data.dtype == np.dtype("|S32")
        records = [s.decode("ascii").rstrip(" ") for s in wcs_data[()]]
        assert all(record[0] in {" ", "+"} for record in records)
        assert not any(record.startswith("#") for record in records)
        text = _hds.decode_ndf_ast_data(records)
        stripped = [line.lstrip() for line in text.splitlines()]
        assert any(s.startswith("Begin FrameSet") for s in stripped)
        assert any(s.startswith("End FrameSet") for s in stripped)
        assert 'Domain = "GRID"' in stripped
        assert 'Domain = "PIXEL"' in stripped
        assert "Sft1 = -19" in stripped
        assert "Sft2 = -9" in stripped


@skip_no_h5py
def test_write_without_projection_omits_wcs_component(tmp_path: Path) -> None:
    """Verify write() omits /WCS when the image has no sky_projection."""
    image = Image(np.zeros((2, 2), dtype=np.float32))
    path = str(tmp_path / "test.sdf")
    write(image, path)
    with h5py.File(path, "r") as f:
        assert "WCS" not in f


@skip_no_h5py
def test_mask_sub_ndf_gets_3d_wcs(tmp_path: Path) -> None:
    """Verify an incompatible mask hoisted to /MORE/LSST/MASK carries a 3D
    /WCS.
    """
    rng = np.random.default_rng(42)
    det_frame = DetectorFrame(instrument="TestInst", detector=4, bbox=Box.factory[1:4096, 1:4096])
    bbox = Box.factory[10:14, 20:25]
    sky_projection = make_random_sky_projection(rng, det_frame, Box.factory[1:4096, 1:4096])
    planes = [MaskPlane(f"P{i}", f"Plane {i}") for i in range(12)]
    image = Image(
        np.arange(20, dtype=np.float32).reshape(4, 5),
        bbox=bbox,
        sky_projection=sky_projection,
    )
    masked = MaskedImage(image, mask_schema=MaskSchema(planes))
    path = str(tmp_path / "test.sdf")
    write(masked, path)
    with h5py.File(path, "r") as f:
        assert "WCS" in f
        top_lines = [s.decode("ascii") for s in f["/WCS/DATA"][()]]
        assert "MASK" in f["/MORE/LSST"]
        assert "WCS" in f["/MORE/LSST/MASK"]
        assert f["/MORE/LSST/MASK/WCS"].attrs["CLASS"] == b"WCS"
        mask_lines = [s.decode("ascii") for s in f["/MORE/LSST/MASK/WCS/DATA"][()]]
        assert top_lines != mask_lines
        mask_text = _hds.decode_ndf_ast_data(mask_lines)
        stripped = [line.lstrip() for line in mask_text.splitlines()]
        assert "Naxes = 3" in stripped
        assert 'Domain = "GRID"' in stripped
        assert 'Domain = "PIXEL"' in stripped
        assert "Sft1 = -19" in stripped
        assert "Sft2 = -9" in stripped
        assert "Sft3 = 1" in stripped
        assert "Begin CmpFrame" in stripped
        assert "Begin SkyFrame" in stripped
        assert 'Domain = "MASK"' in stripped
        assert "Begin CmpMap" in stripped
        assert "Series = 0" in stripped


@skip_no_h5py
def test_mask_sub_ndf_no_wcs_when_image_has_no_projection(tmp_path: Path) -> None:
    """Verify /MORE/LSST/MASK does not carry /WCS when the image has no
    sky_projection.
    """
    planes = [MaskPlane(f"P{i}", f"Plane {i}") for i in range(12)]
    masked = MaskedImage(
        Image(np.zeros((4, 5), dtype=np.float32)),
        mask_schema=MaskSchema(planes),
    )
    path = str(tmp_path / "test.sdf")
    write(masked, path)
    with h5py.File(path, "r") as f:
        assert "WCS" not in f
        assert "MASK" in f["/MORE/LSST"]
        assert "WCS" not in f["/MORE/LSST/MASK"]


@skip_no_h5py
def test_write_image_produces_valid_layout(tmp_path: Path) -> None:
    """Verify write() produces a valid NDF layout with DATA_ARRAY, ORIGIN, and
    /MORE/LSST/JSON.
    """
    image = Image(
        np.arange(20, dtype=np.float32).reshape(4, 5),
        bbox=Box.factory[10:14, 20:25],
    )
    path = str(tmp_path / "test.sdf")
    tree = write(image, path)
    assert tree is not None
    with h5py.File(path, "r") as f:
        assert f["/"].attrs["CLASS"] == b"NDF"
        assert "HDS_ROOT_NAME" in f["/"].attrs
        assert f["/DATA_ARRAY"].attrs["CLASS"] == b"ARRAY"
        np.testing.assert_array_equal(f["/DATA_ARRAY/DATA"][()], image.array)
        origin = f["/DATA_ARRAY/ORIGIN"][()]
        assert origin.dtype == np.int64
        assert len(origin) == 2
        assert not (origin == 0).all()
        assert "MORE" in f
        assert "LSST" in f["/MORE"]
        assert "JSON" in f["/MORE/LSST"]


@skip_no_h5py
def test_write_image_preserves_opaque_fits_metadata(tmp_path: Path) -> None:
    """Verify write() stores opaque FITS headers in /MORE/FITS and they survive
    a round-trip.
    """
    image = Image(np.zeros((2, 2), dtype=np.float32))
    primary = astropy.io.fits.Header()
    primary["FOO"] = ("bar", "test card")
    long_value = "x" * 100
    primary["LONGSTR"] = (long_value, "long string value")
    opaque = FitsOpaqueMetadata()
    opaque.add_header(primary, name="", ver=1)
    image._opaque_metadata = opaque
    path = str(tmp_path / "test.sdf")
    write(image, path)
    with h5py.File(path, "r") as f:
        assert "FITS" in f["/MORE"]
        cards = [c.decode("ascii").rstrip(" ") for c in f["/MORE/FITS"][()]]
        assert any(c.startswith("FOO") for c in cards)
        assert any(c.startswith("CONTINUE") for c in cards)
        assert all(len(c.encode("ascii")) <= 80 for c in cards)
    result = read(path, Image)
    recovered = result._opaque_metadata.headers[ExtensionKey()]
    assert recovered["LONGSTR"] == long_value


@skip_no_h5py
def test_write_image_main_json_round_trips_back(tmp_path: Path) -> None:
    """Verify the main JSON tree at /MORE/LSST/JSON matches write()'s returned
    ArchiveTree.
    """
    image = Image(np.arange(6, dtype=np.float32).reshape(2, 3))
    path = str(tmp_path / "test.sdf")
    tree = write(image, path)
    with h5py.File(path, "r") as f:
        raw = f["/MORE/LSST/JSON"][()]
    joined = b"".join(raw).decode("ascii").rstrip(" ")
    recovered = json.loads(joined)
    assert json.loads(tree.model_dump_json()) == recovered


@skip_no_h5py
def test_write_image_with_unit_creates_units_component(tmp_path: Path) -> None:
    """Verify write() creates a /UNITS component and it round-trips back as the
    correct unit.
    """
    image = Image(np.arange(6, dtype=np.float32).reshape(2, 3), unit=u.ct)
    path = str(tmp_path / "test.sdf")
    write(image, path)
    with h5py.File(path, "r") as f:
        assert "UNITS" in f
        assert f["/UNITS"].shape == ()
        assert f["/UNITS"][()].decode("ascii").rstrip(" ") == "count"
    result = read(path, Image)
    assert result.unit == u.ct


@skip_no_h5py
def test_write_propagates_metadata(tmp_path: Path) -> None:
    """Verify write() stores caller-supplied metadata and it is readable via
    open_archive.
    """
    image = Image(np.arange(6, dtype=np.float32).reshape(2, 3))
    extra = {"test_key": 42, "another": "hello"}
    path = str(tmp_path / "test.sdf")
    tree = write(image, path, metadata=extra)
    assert tree.metadata["test_key"] == 42
    assert tree.metadata["another"] == "hello"
    with open_archive(path, Image) as reader:
        assert reader.metadata["test_key"] == 42
        assert reader.metadata["another"] == "hello"

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

import numpy as np
import pytest

try:
    import h5py

    from lsst.images.ndf import _hds

    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False

skip_no_h5py = pytest.mark.skipif(not HAVE_H5PY, reason="h5py is not installed")

EXAMPLE = os.path.join(os.path.dirname(__file__), "data", "example-ndf.sdf")


def _attr_str(value: object) -> str | None:
    """Return an h5py attribute decoded to str, or None if not string-like."""
    if isinstance(value, bytes):
        return value.decode("ascii")
    if isinstance(value, str):
        return value
    return None


@skip_no_h5py
def test_real_array_round_trip(tmp_path: Path) -> None:
    """Verify float32 arrays round-trip through _hds write_array /
    read_array.
    """
    path = tmp_path / "test.sdf"
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    with h5py.File(path, "w") as f:
        _hds.write_array(f, "DATA", data)
    with h5py.File(path, "r") as f:
        ds = f["DATA"]
        assert ds.dtype == np.float32
        assert ds.shape == (3, 4)
        assert dict(ds.attrs) == {}
        np.testing.assert_array_equal(_hds.read_array(ds), data)


@skip_no_h5py
def test_double_array_round_trip(tmp_path: Path) -> None:
    """Verify float64 arrays round-trip through _hds write_array /
    read_array.
    """
    path = tmp_path / "test.sdf"
    data = np.linspace(0, 1, 6, dtype=np.float64).reshape(2, 3)
    with h5py.File(path, "w") as f:
        _hds.write_array(f, "DATA", data)
    with h5py.File(path, "r") as f:
        assert f["DATA"].dtype == np.float64
        np.testing.assert_array_equal(_hds.read_array(f["DATA"]), data)


@skip_no_h5py
def test_ubyte_and_integer(tmp_path: Path) -> None:
    """Verify uint8 and int32 arrays preserve dtype through _hds round-trip."""
    path = tmp_path / "test.sdf"
    data_u = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    data_i = np.array([10, 20, 30], dtype=np.int32)
    with h5py.File(path, "w") as f:
        _hds.write_array(f, "Q", data_u)
        _hds.write_array(f, "I", data_i)
    with h5py.File(path, "r") as f:
        assert f["Q"].dtype == np.uint8
        assert f["I"].dtype == np.int32
        np.testing.assert_array_equal(_hds.read_array(f["Q"]), data_u)
        np.testing.assert_array_equal(_hds.read_array(f["I"]), data_i)


@skip_no_h5py
def test_logical_uses_hdf5_bitfield(tmp_path: Path) -> None:
    """Verify boolean arrays are stored as HDF5 BITFIELD, not integer."""
    path = tmp_path / "test.sdf"
    with h5py.File(path, "w") as f:
        _hds.write_array(f, "SCALAR", np.array(False, dtype=np.bool_))
        _hds.write_array(f, "ARRAY", np.array([True, False], dtype=np.bool_))
    with h5py.File(path, "r") as f:
        assert f["SCALAR"].id.get_type().get_class() == h5py.h5t.BITFIELD
        assert f["SCALAR"].id.get_type().get_size() == 1
        assert not _hds.read_array(f["SCALAR"])
        assert f["ARRAY"].id.get_type().get_class() == h5py.h5t.BITFIELD
        assert f["ARRAY"].id.get_type().get_size() == 1
        np.testing.assert_array_equal(_hds.read_array(f["ARRAY"]), np.array([True, False]))


@skip_no_h5py
def test_unsupported_dtype_raises_on_write(tmp_path: Path) -> None:
    """Verify write_array raises NotImplementedError for complex128."""
    path = tmp_path / "test.sdf"
    data = np.array([1.0], dtype=np.complex128)
    with h5py.File(path, "w") as f:
        with pytest.raises(NotImplementedError):
            _hds.write_array(f, "X", data)


@skip_no_h5py
def test_unsupported_dtype_raises_on_read(tmp_path: Path) -> None:
    """Verify read_array raises NotImplementedError when reading complex128."""
    path = tmp_path / "test.sdf"
    with h5py.File(path, "w") as f:
        # Write directly with h5py, bypassing write_array's check.
        f.create_dataset("X", data=np.array([1.0], dtype=np.complex128))
    with h5py.File(path, "r") as f:
        with pytest.raises(NotImplementedError):
            _hds.read_array(f["X"])


@skip_no_h5py
def test_read_array_rejects_char_dataset(tmp_path: Path) -> None:
    """Verify read_array raises ValueError when given a char (string)
    dataset.
    """
    path = tmp_path / "test.sdf"
    with h5py.File(path, "w") as f:
        _hds.write_char_array(f, "WCS", ["hello", "world"], width=16)
    with h5py.File(path, "r") as f:
        with pytest.raises(ValueError):
            _hds.read_array(f["WCS"])


@skip_no_h5py
def test_char_array_round_trip(tmp_path: Path) -> None:
    """Verify string lines round-trip through write_char_array /
    read_char_array.
    """
    path = tmp_path / "test.sdf"
    lines = ["Begin FrameSet", "Nframe = 5", "End FrameSet"]
    with h5py.File(path, "w") as f:
        _hds.write_char_array(f, "DATA", lines, width=80)
    with h5py.File(path, "r") as f:
        ds = f["DATA"]
        assert ds.dtype == np.dtype("|S80")
        assert ds.shape == (3,)
        assert dict(ds.attrs) == {}
        assert _hds.read_char_array(ds) == lines


@skip_no_h5py
def test_char_array_pads_and_strips(tmp_path: Path) -> None:
    """Verify write_char_array space-pads to width and read_char_array strips
    trailing spaces.
    """
    path = tmp_path / "test.sdf"
    with h5py.File(path, "w") as f:
        _hds.write_char_array(f, "X", ["short"], width=80)
    with h5py.File(path, "r") as f:
        # Raw data should be space-padded to 80 characters.
        assert f["X"][0] == b"short" + b" " * 75
        # read_char_array strips trailing spaces.
        assert _hds.read_char_array(f["X"]) == ["short"]


@skip_no_h5py
def test_char_array_rejects_long_lines(tmp_path: Path) -> None:
    """Verify write_char_array raises ValueError when a line exceeds the
    width.
    """
    path = tmp_path / "test.sdf"
    with h5py.File(path, "w") as f:
        with pytest.raises(ValueError):
            _hds.write_char_array(f, "X", ["too long"], width=3)


@skip_no_h5py
def test_char_array_rejects_non_ascii(tmp_path: Path) -> None:
    """Verify write_char_array raises ValueError for non-ASCII content."""
    path = tmp_path / "test.sdf"
    with h5py.File(path, "w") as f:
        with pytest.raises(ValueError):
            _hds.write_char_array(f, "X", ["not ascii: \N{LATIN SMALL LETTER E WITH ACUTE}"], width=80)


@skip_no_h5py
def test_ndf_ast_data_encoding_uses_flagged_fixed_width_records() -> None:
    """Verify encode_ndf_ast_data produces fixed-width flagged records that
    decode correctly.
    """
    text = (
        ' Begin FrameSet\n#   Title = "demo"\n    VeryLongAttribute = 12345678901234567890\n End FrameSet\n'
    )
    expected = 'Begin FrameSet\n#   Title = "demo"\nVeryLongAttribute = 12345678901234567890\nEnd FrameSet\n'

    records = _hds.encode_ndf_ast_data(text)

    assert all(len(record) <= _hds.NDF_AST_DATA_WIDTH for record in records)
    assert all(record[0] in {" ", "+"} for record in records)
    assert ' #   Title = "demo"' in records
    assert any(record.startswith("+") for record in records)
    assert _hds.decode_ndf_ast_data(records) == expected


@skip_no_h5py
def test_read_char_array_rejects_numeric_dataset(tmp_path: Path) -> None:
    """Verify read_char_array raises ValueError when given a numeric
    dataset.
    """
    path = tmp_path / "test.sdf"
    with h5py.File(path, "w") as f:
        _hds.write_array(f, "DATA", np.zeros((2,), dtype=np.float32))
    with h5py.File(path, "r") as f:
        with pytest.raises(ValueError):
            _hds.read_char_array(f["DATA"])


@skip_no_h5py
def test_hds_type_for_dtype() -> None:
    """Verify hds_type_for_dtype maps numpy dtypes to correct HDS type
    strings.
    """
    assert _hds.hds_type_for_dtype(np.dtype(np.bool_)) == "_LOGICAL"
    assert _hds.hds_type_for_dtype(np.dtype(np.float32)) == "_REAL"
    assert _hds.hds_type_for_dtype(np.dtype(np.float64)) == "_DOUBLE"
    assert _hds.hds_type_for_dtype(np.dtype(np.uint8)) == "_UBYTE"
    assert _hds.hds_type_for_dtype(np.dtype(np.int32)) == "_INTEGER"
    assert _hds.hds_type_for_dtype(np.dtype("|S80")) == "_CHAR*80"
    with pytest.raises(NotImplementedError):
        _hds.hds_type_for_dtype(np.dtype(np.complex128))


@skip_no_h5py
def test_create_open_structure(tmp_path: Path) -> None:
    """Verify create_structure and open_structure round-trip CLASS and child
    names.
    """
    path = tmp_path / "test.sdf"
    with h5py.File(path, "w") as f:
        ndf = _hds.create_structure(f, "ROOT", "NDF")
        _hds.create_structure(ndf, "DATA_ARRAY", "ARRAY")
    with h5py.File(path, "r") as f:
        root_obj = f["ROOT"]
        assert _attr_str(root_obj.attrs["CLASS"]) == "NDF"
        root, root_type = _hds.open_structure(f, "ROOT")
        assert root_type == "NDF"
        child_names = sorted(name for name, _ in _hds.iter_children(root))
        assert child_names == ["DATA_ARRAY"]
        _, child_type = _hds.open_structure(root, "DATA_ARRAY")
        assert child_type == "ARRAY"


@skip_no_h5py
def test_open_structure_missing_class_raises(tmp_path: Path) -> None:
    """Verify open_structure raises ValueError when the CLASS attribute is
    absent.
    """
    path = tmp_path / "test.sdf"
    with h5py.File(path, "w") as f:
        f.create_group("BAD")
    with h5py.File(path, "r") as f:
        with pytest.raises(ValueError):
            _hds.open_structure(f, "BAD")


@skip_no_h5py
def test_open_structure_accepts_legacy_hdstype(tmp_path: Path) -> None:
    """Verify open_structure accepts the legacy HDSTYPE attribute in place
    of CLASS.
    """
    path = tmp_path / "test.sdf"
    with h5py.File(path, "w") as f:
        g = f.create_group("LEGACY")
        g.attrs["HDSTYPE"] = b"NDF"
    with h5py.File(path, "r") as f:
        _, t = _hds.open_structure(f, "LEGACY")
        assert t == "NDF"


@skip_no_h5py
def test_set_root_name(tmp_path: Path) -> None:
    """Verify set_root_name writes HDS_ROOT_NAME and CLASS to the root
    group.
    """
    path = tmp_path / "test.sdf"
    with h5py.File(path, "w") as f:
        _hds.set_root_name(f, "MYNDF", "NDF")
    with h5py.File(path, "r") as f:
        assert _attr_str(f["/"].attrs["HDS_ROOT_NAME"]) == "MYNDF"
        assert _attr_str(f["/"].attrs["CLASS"]) == "NDF"


@skip_no_h5py
def test_root_is_ndf_with_root_name() -> None:
    """Verify the canonical example NDF has CLASS=NDF and the expected root
    name.
    """
    with h5py.File(EXAMPLE, "r") as f:
        assert _attr_str(f["/"].attrs["CLASS"]) == "NDF"
        assert _attr_str(f["/"].attrs["HDS_ROOT_NAME"]) == "M57"


@skip_no_h5py
def test_data_array_is_array_structure() -> None:
    """Verify the example NDF's DATA_ARRAY is an ARRAY structure with
    correct dtype.
    """
    with h5py.File(EXAMPLE, "r") as f:
        data_array, hds_type = _hds.open_structure(f, "DATA_ARRAY")
        assert hds_type == "ARRAY"
        data = data_array["DATA"]
        assert data.dtype == np.int16
        assert data.shape == (611, 609)
        assert _hds.hds_type_for_dtype(data.dtype) == "_WORD"
        arr = _hds.read_array(data)
        assert arr.shape == (611, 609)
        origin = _hds.read_array(data_array["ORIGIN"])
        assert origin.dtype == np.int64
        assert origin.shape == (2,)


@skip_no_h5py
def test_wcs_is_structure_with_ast_text() -> None:
    """Verify the example NDF's WCS structure contains a valid AST FrameSet
    text dump.
    """
    with h5py.File(EXAMPLE, "r") as f:
        wcs, hds_type = _hds.open_structure(f, "WCS")
        assert hds_type == "WCS"
        lines = _hds.read_char_array(wcs["DATA"])
        text = _hds.decode_ndf_ast_data(lines)
        stripped = [line.lstrip() for line in text.splitlines()]
        assert any(s.startswith("Begin FrameSet") for s in stripped)
        assert any(s.startswith("End FrameSet") for s in stripped)


@skip_no_h5py
def test_more_fits_present() -> None:
    """Verify the example NDF's MORE/FITS extension contains FITS cards
    including NAXIS.
    """
    with h5py.File(EXAMPLE, "r") as f:
        more, hds_type = _hds.open_structure(f, "MORE")
        assert hds_type == "EXT"
        cards = _hds.read_char_array(more["FITS"])
        assert len(cards) > 0
        assert any(c.startswith("NAXIS") for c in cards)

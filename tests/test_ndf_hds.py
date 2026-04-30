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

import h5py
import numpy as np

from lsst.images.ndf import _hds


def _attr_str(value: object) -> str | None:
    """Decode an h5py attribute value (bytes or str) to a Python str."""
    if isinstance(value, bytes):
        return value.decode("ascii")
    if isinstance(value, str):
        return value
    return None


class HdsPrimitiveTestCase(unittest.TestCase):
    """Primitives are bare HDF5 datasets with no HDS-specific attributes."""

    def test_real_array_round_trip(self):
        data = np.arange(12, dtype=np.float32).reshape(3, 4)
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                _hds.write_array(f, "DATA", data)
            with h5py.File(tmp.name, "r") as f:
                ds = f["DATA"]
                self.assertEqual(ds.dtype, np.float32)
                self.assertEqual(ds.shape, (3, 4))
                self.assertEqual(dict(ds.attrs), {})
                np.testing.assert_array_equal(_hds.read_array(ds), data)

    def test_double_array_round_trip(self):
        data = np.linspace(0, 1, 6, dtype=np.float64).reshape(2, 3)
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                _hds.write_array(f, "DATA", data)
            with h5py.File(tmp.name, "r") as f:
                self.assertEqual(f["DATA"].dtype, np.float64)
                np.testing.assert_array_equal(_hds.read_array(f["DATA"]), data)

    def test_ubyte_and_integer(self):
        data_u = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
        data_i = np.array([10, 20, 30], dtype=np.int32)
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                _hds.write_array(f, "Q", data_u)
                _hds.write_array(f, "I", data_i)
            with h5py.File(tmp.name, "r") as f:
                self.assertEqual(f["Q"].dtype, np.uint8)
                self.assertEqual(f["I"].dtype, np.int32)
                np.testing.assert_array_equal(_hds.read_array(f["Q"]), data_u)
                np.testing.assert_array_equal(_hds.read_array(f["I"]), data_i)

    def test_unsupported_dtype_raises_on_write(self):
        data = np.array([1.0], dtype=np.complex128)
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp, h5py.File(tmp.name, "w") as f:
            with self.assertRaises(NotImplementedError):
                _hds.write_array(f, "X", data)

    def test_unsupported_dtype_raises_on_read(self):
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                # Write directly with h5py, bypassing write_array's check.
                f.create_dataset("X", data=np.array([1.0], dtype=np.complex128))
            with h5py.File(tmp.name, "r") as f:
                with self.assertRaises(NotImplementedError):
                    _hds.read_array(f["X"])

    def test_read_array_rejects_char_dataset(self):
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                _hds.write_char_array(f, "WCS", ["hello", "world"], width=16)
            with h5py.File(tmp.name, "r") as f:
                with self.assertRaises(ValueError):
                    _hds.read_array(f["WCS"])

    def test_char_array_round_trip(self):
        lines = ["Begin FrameSet", "Nframe = 5", "End FrameSet"]
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                _hds.write_char_array(f, "DATA", lines, width=80)
            with h5py.File(tmp.name, "r") as f:
                ds = f["DATA"]
                self.assertEqual(ds.dtype, np.dtype("|S80"))
                self.assertEqual(ds.shape, (3,))
                self.assertEqual(dict(ds.attrs), {})
                self.assertEqual(_hds.read_char_array(ds), lines)

    def test_char_array_pads_and_strips(self):
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                _hds.write_char_array(f, "X", ["short"], width=80)
            with h5py.File(tmp.name, "r") as f:
                # Raw data should be space-padded to 80 characters.
                self.assertEqual(f["X"][0], b"short" + b" " * 75)
                # read_char_array strips trailing spaces.
                self.assertEqual(_hds.read_char_array(f["X"]), ["short"])

    def test_read_char_array_rejects_numeric_dataset(self):
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                _hds.write_array(f, "DATA", np.zeros((2,), dtype=np.float32))
            with h5py.File(tmp.name, "r") as f:
                with self.assertRaises(ValueError):
                    _hds.read_char_array(f["DATA"])

    def test_hds_type_for_dtype(self):
        self.assertEqual(_hds.hds_type_for_dtype(np.dtype(np.float32)), "_REAL")
        self.assertEqual(_hds.hds_type_for_dtype(np.dtype(np.float64)), "_DOUBLE")
        self.assertEqual(_hds.hds_type_for_dtype(np.dtype(np.uint8)), "_UBYTE")
        self.assertEqual(_hds.hds_type_for_dtype(np.dtype(np.int32)), "_INTEGER")
        self.assertEqual(_hds.hds_type_for_dtype(np.dtype("|S80")), "_CHAR*80")
        with self.assertRaises(NotImplementedError):
            _hds.hds_type_for_dtype(np.dtype(np.complex128))


class HdsStructureTestCase(unittest.TestCase):
    """Structures are HDF5 groups with a CLASS attribute."""

    def test_create_open_structure(self):
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                ndf = _hds.create_structure(f, "ROOT", "NDF")
                _hds.create_structure(ndf, "DATA_ARRAY", "ARRAY")
            with h5py.File(tmp.name, "r") as f:
                root_obj = f["ROOT"]
                self.assertEqual(_attr_str(root_obj.attrs["CLASS"]), "NDF")
                root, root_type = _hds.open_structure(f, "ROOT")
                self.assertEqual(root_type, "NDF")
                child_names = sorted(name for name, _ in _hds.iter_children(root))
                self.assertEqual(child_names, ["DATA_ARRAY"])
                _, child_type = _hds.open_structure(root, "DATA_ARRAY")
                self.assertEqual(child_type, "ARRAY")

    def test_open_structure_missing_class_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                f.create_group("BAD")
            with h5py.File(tmp.name, "r") as f:
                with self.assertRaises(ValueError):
                    _hds.open_structure(f, "BAD")

    def test_open_structure_accepts_legacy_hdstype(self):
        """Files from older HDS variants used HDSTYPE rather than CLASS."""
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                g = f.create_group("LEGACY")
                g.attrs["HDSTYPE"] = b"NDF"
            with h5py.File(tmp.name, "r") as f:
                _, t = _hds.open_structure(f, "LEGACY")
                self.assertEqual(t, "NDF")

    def test_set_root_name(self):
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                _hds.set_root_name(f, "MYNDF", "NDF")
            with h5py.File(tmp.name, "r") as f:
                self.assertEqual(_attr_str(f["/"].attrs["HDS_ROOT_NAME"]), "MYNDF")
                self.assertEqual(_attr_str(f["/"].attrs["CLASS"]), "NDF")


class HdsCanonicalExampleTestCase(unittest.TestCase):
    """Validate _hds against a canonical-format Starlink-generated NDF.

    The example file is an M57 image with the modern hds-v5 layout:
    root group with CLASS="NDF" and HDS_ROOT_NAME, DATA_ARRAY as an
    ARRAY structure containing DATA (int16) and ORIGIN (int64), WCS as
    a structure with an AST text-dump DATA primitive, and MORE.FITS as
    an 80-character card array.
    """

    EXAMPLE = os.path.join(os.path.dirname(__file__), "data", "example-ndf.sdf")

    def test_root_is_ndf_with_root_name(self):
        with h5py.File(self.EXAMPLE, "r") as f:
            self.assertEqual(_attr_str(f["/"].attrs["CLASS"]), "NDF")
            self.assertEqual(_attr_str(f["/"].attrs["HDS_ROOT_NAME"]), "M57")

    def test_data_array_is_array_structure(self):
        with h5py.File(self.EXAMPLE, "r") as f:
            data_array, hds_type = _hds.open_structure(f, "DATA_ARRAY")
            self.assertEqual(hds_type, "ARRAY")
            data = data_array["DATA"]
            self.assertEqual(data.dtype, np.int16)
            self.assertEqual(data.shape, (611, 609))
            self.assertEqual(_hds.hds_type_for_dtype(data.dtype), "_WORD")
            arr = _hds.read_array(data)
            self.assertEqual(arr.shape, (611, 609))
            origin = _hds.read_array(data_array["ORIGIN"])
            self.assertEqual(origin.dtype, np.int64)
            self.assertEqual(origin.shape, (2,))

    def test_wcs_is_structure_with_ast_text(self):
        with h5py.File(self.EXAMPLE, "r") as f:
            wcs, hds_type = _hds.open_structure(f, "WCS")
            self.assertEqual(hds_type, "WCS")
            lines = _hds.read_char_array(wcs["DATA"])
            # AST channel text uses leading whitespace for nesting; strip it
            # for the structural-marker checks here.
            stripped = [line.lstrip() for line in lines]
            self.assertTrue(any(s.startswith("Begin FrameSet") for s in stripped))
            self.assertTrue(any(s.startswith("End FrameSet") for s in stripped))

    def test_more_fits_present(self):
        with h5py.File(self.EXAMPLE, "r") as f:
            more, hds_type = _hds.open_structure(f, "MORE")
            self.assertEqual(hds_type, "EXT")
            cards = _hds.read_char_array(more["FITS"])
            self.assertGreater(len(cards), 0)
            self.assertTrue(any(c.startswith("NAXIS") for c in cards))

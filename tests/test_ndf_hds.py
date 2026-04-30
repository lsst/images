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

import tempfile
import unittest

import h5py
import numpy as np

from lsst.images.ndf import _hds


class HdsPrimitiveTestCase(unittest.TestCase):
    def test_real_array_round_trip(self):
        data = np.arange(12, dtype=np.float32).reshape(3, 4)
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                _hds.write_array(f, "DATA", data)
            with h5py.File(tmp.name, "r") as f:
                ds = f["DATA"]
                self.assertEqual(ds.attrs["HDSTYPE"], "_REAL")
                self.assertEqual(ds.attrs["HDSNDIMS"], 2)
                self.assertEqual(ds.attrs["HDS_DATASET_IS_DEFINED"], True)
                np.testing.assert_array_equal(_hds.read_array(ds), data)

    def test_double_array_round_trip(self):
        data = np.linspace(0, 1, 6, dtype=np.float64).reshape(2, 3)
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                _hds.write_array(f, "DATA", data)
            with h5py.File(tmp.name, "r") as f:
                self.assertEqual(f["DATA"].attrs["HDSTYPE"], "_DOUBLE")
                np.testing.assert_array_equal(_hds.read_array(f["DATA"]), data)

    def test_ubyte_and_integer(self):
        data_u = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
        data_i = np.array([10, 20, 30], dtype=np.int32)
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                _hds.write_array(f, "Q", data_u)
                _hds.write_array(f, "I", data_i)
            with h5py.File(tmp.name, "r") as f:
                self.assertEqual(f["Q"].attrs["HDSTYPE"], "_UBYTE")
                self.assertEqual(f["I"].attrs["HDSTYPE"], "_INTEGER")
                np.testing.assert_array_equal(_hds.read_array(f["Q"]), data_u)
                np.testing.assert_array_equal(_hds.read_array(f["I"]), data_i)

    def test_unsupported_dtype_raises(self):
        data = np.array([1.0], dtype=np.complex128)
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp, h5py.File(tmp.name, "w") as f:
            with self.assertRaises(NotImplementedError):
                _hds.write_array(f, "X", data)

    def test_char_array_round_trip(self):
        lines = ["Begin FrameSet", "Nframe = 5", "End FrameSet"]
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                _hds.write_char_array(f, "DATA", lines, width=80)
            with h5py.File(tmp.name, "r") as f:
                ds = f["DATA"]
                self.assertEqual(ds.attrs["HDSTYPE"], "_CHAR*80")
                self.assertEqual(ds.attrs["HDSNDIMS"], 1)
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


class HdsStructureTestCase(unittest.TestCase):
    def test_create_open_structure(self):
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                ndf = _hds.create_structure(f, "ROOT", "NDF")
                _hds.create_structure(ndf, "DATA_ARRAY", "ARRAY")
            with h5py.File(tmp.name, "r") as f:
                root, root_type = _hds.open_structure(f, "ROOT")
                self.assertEqual(root_type, "NDF")
                child_names = sorted(name for name, _ in _hds.iter_children(root))
                self.assertEqual(child_names, ["DATA_ARRAY"])
                _, child_type = _hds.open_structure(root, "DATA_ARRAY")
                self.assertEqual(child_type, "ARRAY")

    def test_open_structure_missing_hdstype_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                f.create_group("BAD")
            with h5py.File(tmp.name, "r") as f:
                with self.assertRaises(ValueError):
                    _hds.open_structure(f, "BAD")

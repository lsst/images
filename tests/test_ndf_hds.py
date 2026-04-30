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


class HdsExampleNdfTestCase(unittest.TestCase):
    """Validate _hds against a Starlink-generated NDF.

    The example file was produced by CCDPACK and contains a single
    top-level NDF structure named BIAS1 with DATA_ARRAY, WCS, and MORE.FITS
    components.
    """

    EXAMPLE = os.path.join(os.path.dirname(__file__), "data", "example-ndf.sdf")

    def test_top_level_structure(self):
        with h5py.File(self.EXAMPLE, "r") as f:
            # The example wraps a single NDF in a top-level container; verify
            # we can iterate the root and find the BIAS1 NDF.
            children = dict(_hds.iter_children(f))
            self.assertIn("BIAS1", children)
            bias1, hdstype = _hds.open_structure(f, "BIAS1")
            self.assertEqual(hdstype, "NDF")

    def test_data_array_present(self):
        with h5py.File(self.EXAMPLE, "r") as f:
            bias1, _ = _hds.open_structure(f, "BIAS1")
            # DATA_ARRAY is a dataset, not a structure
            data = bias1["DATA_ARRAY"]
            self.assertEqual(data.shape, (128, 128))
            self.assertEqual(data.dtype, np.float32)
            # Verify we can read the data directly
            array_data = data[:]
            self.assertEqual(array_data.shape, (128, 128))

    def test_wcs_present(self):
        with h5py.File(self.EXAMPLE, "r") as f:
            bias1, _ = _hds.open_structure(f, "BIAS1")
            wcs, hdstype = _hds.open_structure(bias1, "WCS")
            self.assertEqual(hdstype, "WCS")
            # WCS/DATA is a dataset with string data; read directly since
            # the file doesn't have HDSTYPE attributes on leaf datasets
            wcs_data = wcs["DATA"]
            lines = [s.decode() if isinstance(s, bytes) else s for s in wcs_data[:]]
            self.assertTrue(any("Begin FrameSet" in line for line in lines))
            self.assertTrue(any("End FrameSet" in line for line in lines))

    def test_more_fits_present(self):
        with h5py.File(self.EXAMPLE, "r") as f:
            bias1, _ = _hds.open_structure(f, "BIAS1")
            more, _ = _hds.open_structure(bias1, "MORE")
            fits_data = more["FITS"]
            cards = [s.decode() if isinstance(s, bytes) else s for s in fits_data[:]]
            # Sample a few cards we know are in the example.
            self.assertTrue(any(c.startswith("NAXIS") for c in cards))
            self.assertTrue(len(cards) > 0)

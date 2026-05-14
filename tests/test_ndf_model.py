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

import numpy as np

try:
    import h5py

    from lsst.images.ndf import _hds
    from lsst.images.ndf._model import HdsPrimitive, Ndf, NdfArray, NdfDocument, NdfQuality, NdfWcs

    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False


def _attr_str(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("ascii")
    return str(value)


@unittest.skipUnless(HAVE_H5PY, "h5py is not installed")
class NdfModelTestCase(unittest.TestCase):
    """Tests for the Python NDF intermediate representation."""

    def test_ndf_document_writes_standard_components(self) -> None:
        image = np.arange(6, dtype=np.float32).reshape(2, 3)
        quality = np.array([[0, 1, 0], [1, 0, 0]], dtype=np.uint8)
        wcs_lines = [" Begin FrameSet", " End FrameSet"]
        document = NdfDocument(root=Ndf(), root_name="TEST")
        document.root.set_array_component("DATA_ARRAY", image, origin=(20, 10))
        document.root.set_quality(
            NdfQuality(
                NdfArray(
                    quality,
                    origin=np.array([20, 10], dtype=np.int32),
                    bad_pixel=False,
                )
            )
        )
        document.root.set_wcs(NdfWcs(wcs_lines))
        document.root.set_units("adu")
        lsst = document.root.ensure_lsst_extension()
        lsst.children["JSON"] = HdsPrimitive.char_array(['{"kind":"image"}'], width=80)

        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                document.write_to_hdf5(f)
            with h5py.File(tmp.name, "r") as f:
                self.assertEqual(_attr_str(f["/"].attrs[_hds.ATTR_CLASS]), "NDF")
                self.assertEqual(_attr_str(f["/"].attrs[_hds.ATTR_ROOT_NAME]), "TEST")
                self.assertEqual(_attr_str(f["/DATA_ARRAY"].attrs[_hds.ATTR_CLASS]), "ARRAY")
                np.testing.assert_array_equal(f["/DATA_ARRAY/DATA"][()], image)
                np.testing.assert_array_equal(f["/DATA_ARRAY/ORIGIN"][()], np.array([20, 10]))
                self.assertEqual(_attr_str(f["/QUALITY"].attrs[_hds.ATTR_CLASS]), "QUALITY")
                self.assertEqual(f["/QUALITY/BADBITS"][()], 255)
                np.testing.assert_array_equal(f["/QUALITY/QUALITY/DATA"][()], quality)
                self.assertEqual(f["/QUALITY/QUALITY/BAD_PIXEL"].id.get_type().get_class(), h5py.h5t.BITFIELD)
                self.assertEqual(_attr_str(f["/WCS"].attrs[_hds.ATTR_CLASS]), "WCS")
                self.assertEqual(_hds.read_char_array(f["/WCS/DATA"]), wcs_lines)
                self.assertEqual(f["/UNITS"].shape, ())
                self.assertEqual(f["/UNITS"][()].decode("ascii").rstrip(" "), "adu")
                self.assertEqual(_hds.read_char_array(f["/MORE/LSST/JSON"]), ['{"kind":"image"}'])

    def test_document_read_preserves_typed_ndf_components(self) -> None:
        image = np.arange(4, dtype=np.int16).reshape(2, 2)
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                document = NdfDocument(root=Ndf(), root_name="READ")
                document.root.set_array_component("DATA_ARRAY", image, origin=(5, 6))
                document.root.set_units("count")
                document.write_to_hdf5(f)
            with h5py.File(tmp.name, "r") as f:
                recovered = NdfDocument.from_hdf5(f)
                self.assertIsInstance(recovered.root, Ndf)
                self.assertEqual(recovered.root_name, "READ")
                self.assertEqual(recovered.root.get_units(), "count")
                data = recovered.get("/DATA_ARRAY/DATA")
                self.assertIsInstance(data, HdsPrimitive)
                np.testing.assert_array_equal(data.read_array(), image)


if __name__ == "__main__":
    unittest.main()

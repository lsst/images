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

import numpy as np
import pytest

try:
    import h5py

    from lsst.images.ndf import _hds
    from lsst.images.ndf._model import HdsPrimitive, Ndf, NdfArray, NdfDocument, NdfQuality, NdfWcs

    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False

skip_no_h5py = pytest.mark.skipif(not HAVE_H5PY, reason="h5py is not installed")


def _attr_str(value: object) -> str:
    """Return value decoded to str, handling bytes from HDF5 attributes."""
    if isinstance(value, bytes):
        return value.decode("ascii")
    return str(value)


@skip_no_h5py
def test_ndf_document_writes_standard_components(tmp_path: Path) -> None:
    """Verify NdfDocument writes all standard NDF components to HDF5
    correctly.
    """
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

    path = str(tmp_path / "test.sdf")
    with h5py.File(path, "w") as f:
        document.write_to_hdf5(f)
    with h5py.File(path, "r") as f:
        assert _attr_str(f["/"].attrs[_hds.ATTR_CLASS]) == "NDF"
        assert _attr_str(f["/"].attrs[_hds.ATTR_ROOT_NAME]) == "TEST"
        assert _attr_str(f["/DATA_ARRAY"].attrs[_hds.ATTR_CLASS]) == "ARRAY"
        np.testing.assert_array_equal(f["/DATA_ARRAY/DATA"][()], image)
        np.testing.assert_array_equal(f["/DATA_ARRAY/ORIGIN"][()], np.array([20, 10]))
        assert _attr_str(f["/QUALITY"].attrs[_hds.ATTR_CLASS]) == "QUALITY"
        assert f["/QUALITY/BADBITS"][()] == 255
        np.testing.assert_array_equal(f["/QUALITY/QUALITY/DATA"][()], quality)
        assert f["/QUALITY/QUALITY/BAD_PIXEL"].id.get_type().get_class() == h5py.h5t.BITFIELD
        assert _attr_str(f["/WCS"].attrs[_hds.ATTR_CLASS]) == "WCS"
        assert _hds.read_char_array(f["/WCS/DATA"]) == wcs_lines
        assert f["/UNITS"].shape == ()
        assert f["/UNITS"][()].decode("ascii").rstrip(" ") == "adu"
        assert _hds.read_char_array(f["/MORE/LSST/JSON"]) == ['{"kind":"image"}']


@skip_no_h5py
def test_document_read_preserves_typed_ndf_components(tmp_path: Path) -> None:
    """Verify NdfDocument.from_hdf5 recovers typed NDF components after
    a round-trip.
    """
    image = np.arange(4, dtype=np.int16).reshape(2, 2)
    path = str(tmp_path / "test.sdf")
    with h5py.File(path, "w") as f:
        document = NdfDocument(root=Ndf(), root_name="READ")
        document.root.set_array_component("DATA_ARRAY", image, origin=(5, 6))
        document.root.set_units("count")
        document.write_to_hdf5(f)
    with h5py.File(path, "r") as f:
        recovered = NdfDocument.from_hdf5(f)
        assert isinstance(recovered.root, Ndf)
        assert recovered.root_name == "READ"
        assert recovered.root.get_units() == "count"
        data = recovered.get("/DATA_ARRAY/DATA")
        assert isinstance(data, HdsPrimitive)
        np.testing.assert_array_equal(data.read_array(), image)

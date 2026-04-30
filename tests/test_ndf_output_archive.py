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
import pydantic

from lsst.images.ndf._output_archive import NdfOutputArchive


class TinyTree(pydantic.BaseModel):
    """A trivial Pydantic model used as a serialization stand-in."""

    name: str


class NdfOutputArchiveBasicsTestCase(unittest.TestCase):
    """Tests for `NdfOutputArchive` constructor and `serialize_direct`."""

    def test_serialize_direct_calls_serializer_with_nested_archive(self):
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                arch = NdfOutputArchive(f)
                tree = arch.serialize_direct("top", lambda nested: TinyTree(name="hello"))
                self.assertEqual(tree.name, "hello")

    def test_constructor_marks_root_as_ndf(self):
        """The constructor should set CLASS=NDF on the root group so that
        Starlink tools recognise the file as an NDF.
        """
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                NdfOutputArchive(f)
            with h5py.File(tmp.name, "r") as f:
                self.assertEqual(f["/"].attrs["CLASS"], "NDF")


class NdfOutputArchiveAddArrayTestCase(unittest.TestCase):
    """Tests for `NdfOutputArchive.add_array` routing."""

    def test_top_level_image_routes_to_data_array(self):
        data = np.arange(20, dtype=np.float32).reshape(4, 5)
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                arch = NdfOutputArchive(f)
                ref = arch.add_array(data, name="image")
                self.assertEqual(ref.source, "ndf:/DATA_ARRAY/DATA")
            with h5py.File(tmp.name, "r") as f:
                ds = f["/DATA_ARRAY/DATA"]
                self.assertEqual(ds.dtype, np.float32)
                np.testing.assert_array_equal(ds[()], data)
                self.assertEqual(f["/DATA_ARRAY"].attrs["CLASS"], "ARRAY")
                origin = f["/DATA_ARRAY/ORIGIN"]
                self.assertEqual(origin.dtype, np.int64)
                self.assertEqual(origin.shape, (2,))

    def test_top_level_variance_routes_to_variance(self):
        data = np.full((3, 3), 0.5, dtype=np.float64)
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                arch = NdfOutputArchive(f)
                ref = arch.add_array(data, name="variance")
                self.assertEqual(ref.source, "ndf:/VARIANCE/DATA")
            with h5py.File(tmp.name, "r") as f:
                self.assertEqual(f["/VARIANCE"].attrs["CLASS"], "ARRAY")
                self.assertEqual(f["/VARIANCE/DATA"].dtype, np.float64)

    def test_top_level_compatible_mask_routes_to_quality(self):
        data = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                arch = NdfOutputArchive(f)
                ref = arch.add_array(data, name="mask")
                self.assertEqual(ref.source, "ndf:/QUALITY/QUALITY")
            with h5py.File(tmp.name, "r") as f:
                self.assertEqual(f["/QUALITY"].attrs["CLASS"], "QUALITY")
                self.assertEqual(f["/QUALITY/QUALITY"].dtype, np.uint8)
                self.assertEqual(f["/QUALITY/BADBITS"][()], 0xFF)

    def test_top_level_incompatible_mask_routes_to_more_lsst(self):
        # 3D mask array (multi-plane uint8) doesn't fit NDF QUALITY.
        data = np.zeros((3, 4, 2), dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                arch = NdfOutputArchive(f)
                ref = arch.add_array(data, name="mask")
                self.assertEqual(ref.source, "ndf:/MORE/LSST/MASK/DATA")
            with h5py.File(tmp.name, "r") as f:
                self.assertEqual(f["/MORE/LSST/MASK"].attrs["CLASS"], "STRUCT")
                self.assertEqual(f["/MORE/LSST/MASK/DATA"].shape, (3, 4, 2))

    def test_nested_array_hoists(self):
        data = np.array([[1.0, 2.0]], dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                arch = NdfOutputArchive(f)
                ref = arch.add_array(data, name="psf/coefficients")
                self.assertEqual(ref.source, "ndf:/MORE/LSST/PSF_COEFFICIENTS")
            with h5py.File(tmp.name, "r") as f:
                self.assertIn("MORE", f)
                self.assertIn("LSST", f["/MORE"])
                self.assertIn("PSF_COEFFICIENTS", f["/MORE/LSST"])


class NdfOutputArchivePointerTestCase(unittest.TestCase):
    """Tests for `NdfOutputArchive.serialize_pointer` and
    `serialize_frame_set`.
    """

    def test_serialize_pointer_writes_subtree_and_returns_pointer(self):
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                arch = NdfOutputArchive(f)
                ptr = arch.serialize_pointer(
                    "psf",
                    lambda nested: TinyTree(name="gaussian"),
                    key=("psf", 1),
                )
                self.assertEqual(ptr.ref, "/MORE/LSST/PSF")
            with h5py.File(tmp.name, "r") as f:
                # The hoisted sub-tree is stored as a _CHAR*N dataset
                # (1D byte-string array). Read it back and parse.
                raw = f["/MORE/LSST/PSF"][()]
                # Concatenate and decode (it may be a single padded line).
                joined = b"".join(raw).decode("ascii").rstrip(" ")
                self.assertIn('"name":"gaussian"', joined.replace(" ", ""))

    def test_serialize_pointer_caches_by_key(self):
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                arch = NdfOutputArchive(f)
                ptr1 = arch.serialize_pointer(
                    "psf",
                    lambda nested: TinyTree(name="first"),
                    key=("psf", 1),
                )
                # Same key -> returns cached pointer; serializer not re-run
                # (we'd otherwise overwrite the file content with "second").
                ptr2 = arch.serialize_pointer(
                    "psf",
                    lambda nested: TinyTree(name="second"),
                    key=("psf", 1),
                )
                self.assertEqual(ptr1, ptr2)
            with h5py.File(tmp.name, "r") as f:
                raw = f["/MORE/LSST/PSF"][()]
                joined = b"".join(raw).decode("ascii").rstrip(" ")
                self.assertIn("first", joined)
                self.assertNotIn("second", joined)

    def test_serialize_frame_set_records_for_iter(self):
        # serialize_frame_set is delegated to serialize_pointer plus
        # recording the (FrameSet, pointer) pair for iter_frame_sets,
        # mirroring how FITS and JSON archives behave. The frame_set
        # itself is opaque here -- we just check it round-trips through
        # the recording.
        sentinel = object()
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                arch = NdfOutputArchive(f)
                ptr = arch.serialize_frame_set(
                    "wcs/pixel_to_sky",
                    sentinel,
                    lambda nested: TinyTree(name="proj"),
                    key=("frame_set", 1),
                )
                self.assertEqual(ptr.ref, "/MORE/LSST/WCS_PIXEL_TO_SKY")
                recorded = list(arch.iter_frame_sets())
                self.assertEqual(len(recorded), 1)
                self.assertIs(recorded[0][0], sentinel)
                self.assertEqual(recorded[0][1].ref, "/MORE/LSST/WCS_PIXEL_TO_SKY")

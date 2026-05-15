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
import tempfile
import unittest

import astropy.io.fits
import astropy.table
import astropy.units as u
import numpy as np
import pydantic

from lsst.images import Box, Image, MaskedImage, MaskPlane, MaskSchema
from lsst.images._transforms import FrameLookupError, FrameSet, Transform
from lsst.images._transforms._frames import DetectorFrame, Frame
from lsst.images.fits import ExtensionKey, FitsOpaqueMetadata
from lsst.images.serialization import ArrayReferenceModel, InlineArrayModel
from lsst.images.tests import make_random_projection

try:
    import h5py

    from lsst.images.ndf import (
        NdfInputArchive,
        NdfOutputArchive,
        _hds,
        read,
        write,
    )

    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False


class TinyFrameSet(FrameSet):
    """Minimal concrete frame-set for archive bookkeeping tests."""

    def __contains__(self, frame: Frame) -> bool:
        return False

    def __getitem__[I: Frame, O: Frame](self, key: tuple[I, O]) -> Transform[I, O]:
        raise FrameLookupError(key)


class TinyTree(pydantic.BaseModel):
    """A trivial Pydantic model used as a serialization stand-in."""

    name: str


@unittest.skipUnless(HAVE_H5PY, "h5py is not installed")
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
                self.assertEqual(f["/"].attrs["CLASS"], b"NDF")


@unittest.skipUnless(HAVE_H5PY, "h5py is not installed")
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
                self.assertEqual(f["/DATA_ARRAY"].attrs["CLASS"], b"ARRAY")
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
                self.assertEqual(f["/VARIANCE"].attrs["CLASS"], b"ARRAY")
                self.assertEqual(f["/VARIANCE/DATA"].dtype, np.float64)

    def test_top_level_compatible_mask_routes_to_quality(self):
        data = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                arch = NdfOutputArchive(f)
                ref = arch.add_array(data, name="mask")
                self.assertEqual(ref.source, "ndf:/QUALITY/QUALITY/DATA")
            with h5py.File(tmp.name, "r") as f:
                self.assertEqual(f["/QUALITY"].attrs["CLASS"], b"QUALITY")
                self.assertEqual(f["/QUALITY/QUALITY"].attrs["CLASS"], b"ARRAY")
                self.assertEqual(f["/QUALITY/QUALITY/DATA"].dtype, np.uint8)
                np.testing.assert_array_equal(f["/QUALITY/QUALITY/DATA"][()], data)
                self.assertEqual(f["/QUALITY/QUALITY/ORIGIN"].dtype, np.int32)
                self.assertEqual(f["/QUALITY/QUALITY/ORIGIN"].shape, (2,))
                self.assertEqual(f["/QUALITY/QUALITY/BAD_PIXEL"].id.get_type().get_class(), h5py.h5t.BITFIELD)
                self.assertFalse(_hds.read_array(f["/QUALITY/QUALITY/BAD_PIXEL"]))
                self.assertEqual(f["/QUALITY/BADBITS"][()], 255)

    def test_top_level_incompatible_mask_routes_to_more_lsst(self):
        # 3D mask array in NDF storage order (mask-byte, y, x) is hoisted
        # as a sub-NDF inside /MORE/LSST/MASK, with a compressed 2D view
        # exposed as /QUALITY/QUALITY for standard NDF applications.
        data = np.zeros((2, 3, 4), dtype=np.uint8)
        data[0, 1, 2] = 4
        data[1, 2, 3] = 8
        expected_quality = np.any(data != 0, axis=0).astype(np.uint8)
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                arch = NdfOutputArchive(f)
                ref = arch.add_array(data, name="mask")
                self.assertEqual(ref.source, "ndf:/MORE/LSST/MASK/DATA_ARRAY/DATA")
            with h5py.File(tmp.name, "r") as f:
                # /MORE/LSST/MASK is a real NDF: top-level CLASS="NDF"
                # containing a DATA_ARRAY structure with DATA + ORIGIN.
                self.assertEqual(f["/MORE/LSST/MASK"].attrs["CLASS"], b"NDF")
                self.assertEqual(f["/MORE/LSST/MASK/DATA_ARRAY"].attrs["CLASS"], b"ARRAY")
                self.assertEqual(f["/MORE/LSST/MASK/DATA_ARRAY/DATA"].shape, data.shape)
                self.assertEqual(f["/QUALITY/QUALITY"].attrs["CLASS"], b"ARRAY")
                np.testing.assert_array_equal(f["/QUALITY/QUALITY/DATA"][()], expected_quality)
                self.assertEqual(f["/QUALITY/BADBITS"][()], 255)
                origin = f["/MORE/LSST/MASK/DATA_ARRAY/ORIGIN"]
                self.assertEqual(origin.dtype, np.int64)
                self.assertEqual(origin.shape, (3,))

    def test_nested_array_hoists_as_sub_ndf(self):
        # Hoisted numeric arrays land under /MORE/LSST as hierarchical
        # sub-NDFs (CLASS="NDF" with DATA_ARRAY/DATA + ORIGIN inside) so
        # Starlink tools can inspect them as ordinary NDFs while each HDS
        # component stays short.
        data = np.array([[1.0, 2.0]], dtype=np.float32)
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                arch = NdfOutputArchive(f)
                ref = arch.add_array(data, name="psf/coefficients")
                self.assertEqual(ref.source, "ndf:/MORE/LSST/PSF/COEFFICIENTS/DATA_ARRAY/DATA")
            with h5py.File(tmp.name, "r") as f:
                self.assertIn("MORE", f)
                self.assertIn("LSST", f["/MORE"])
                self.assertIn("PSF", f["/MORE/LSST"])
                self.assertIn("COEFFICIENTS", f["/MORE/LSST/PSF"])
                sub = f["/MORE/LSST/PSF/COEFFICIENTS"]
                self.assertEqual(sub.attrs["CLASS"], b"NDF")
                self.assertEqual(sub["DATA_ARRAY"].attrs["CLASS"], b"ARRAY")
                np.testing.assert_array_equal(sub["DATA_ARRAY/DATA"][()], data)
                origin = sub["DATA_ARRAY/ORIGIN"]
                self.assertEqual(origin.dtype, np.int64)
                self.assertEqual(origin.shape, (data.ndim,))


@unittest.skipUnless(HAVE_H5PY, "h5py is not installed")
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
                self.assertEqual(ptr.path, "/MORE/LSST/PSF/JSON")
            with h5py.File(tmp.name, "r") as f:
                # The hoisted sub-tree is stored as a "JSON" _CHAR*N
                # child of the target structure.
                raw = f["/MORE/LSST/PSF/JSON"][()]
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
                raw = f["/MORE/LSST/PSF/JSON"][()]
                joined = b"".join(raw).decode("ascii").rstrip(" ")
                self.assertIn("first", joined)
                self.assertNotIn("second", joined)

    def test_serialize_pointer_preserves_nested_arrays(self):
        # Regression test: a pointer target that writes a nested array via
        # the nested archive must round-trip with that array still in the
        # file. Previously the pointer JSON was written at the target path
        # itself, clobbering any nested data the serializer produced.
        class TreeWithArray(pydantic.BaseModel):
            name: str
            data: ArrayReferenceModel

        payload = np.arange(6, dtype=np.float32).reshape(2, 3)
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                arch = NdfOutputArchive(f)
                ptr = arch.serialize_pointer(
                    "psf",
                    lambda nested: TreeWithArray(
                        name="gaussian",
                        data=nested.add_array(payload, name="parameters"),
                    ),
                    key=("psf", 1),
                )
                self.assertEqual(ptr.path, "/MORE/LSST/PSF/JSON")
            with h5py.File(tmp.name, "r") as f:
                # JSON stayed at <path>/JSON; the nested array is still
                # accessible at the sub-NDF path the serializer wrote.
                self.assertIn("/MORE/LSST/PSF/JSON", f)
                self.assertIn("/MORE/LSST/PSF/PARAMETERS/DATA_ARRAY/DATA", f)
                np.testing.assert_array_equal(f["/MORE/LSST/PSF/PARAMETERS/DATA_ARRAY/DATA"][()], payload)
                # The pointer-target structure is typed after its leaf
                # name, not the generic EXT.
                self.assertEqual(f["/MORE/LSST/PSF"].attrs["CLASS"], b"PSF")
                self.assertEqual(f["/MORE/LSST"].attrs["CLASS"], b"LSST")

    def test_serialize_frame_set_records_for_iter(self):
        # serialize_frame_set is delegated to serialize_pointer plus
        # recording the (FrameSet, pointer) pair for iter_frame_sets,
        # mirroring how FITS and JSON archives behave.
        frame_set = TinyFrameSet()
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                arch = NdfOutputArchive(f)
                ptr = arch.serialize_frame_set(
                    "wcs/pixel_to_sky",
                    frame_set,
                    lambda nested: TinyTree(name="proj"),
                    key=("frame_set", 1),
                )
                self.assertEqual(ptr.path, "/MORE/LSST/WCS/PIXEL_TO_SKY/JSON")
                recorded = list(arch.iter_frame_sets())
                self.assertEqual(len(recorded), 1)
                self.assertIs(recorded[0][0], frame_set)
                self.assertEqual(recorded[0][1].path, "/MORE/LSST/WCS/PIXEL_TO_SKY/JSON")


@unittest.skipUnless(HAVE_H5PY, "h5py is not installed")
class NdfOutputArchiveAddTableTestCase(unittest.TestCase):
    """Tests for `NdfOutputArchive.add_table` and `add_structured_array`."""

    def test_add_table_returns_inline_table_model(self):
        t = astropy.table.Table({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                arch = NdfOutputArchive(f)
                model = arch.add_table(t, name="some_table")
                self.assertEqual(len(model.columns), 2)
                # v1 stores tables inline in the JSON tree.
                self.assertIsInstance(model.columns[0].data, InlineArrayModel)

    def test_add_structured_array_writes_column_ndfs_with_units(self):
        rec = np.zeros(3, dtype=[("x", np.float64), ("y", np.int32)])
        rec["x"] = [1.0, 2.0, 3.0]
        rec["y"] = [10, 20, 30]
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                arch = NdfOutputArchive(f)
                model = arch.add_structured_array(
                    rec,
                    name="rec",
                    units={"x": u.m},
                    descriptions={"y": "the y values"},
                )
                self.assertEqual(len(model.columns), 2)
                self.assertIsInstance(model.columns[0].data, ArrayReferenceModel)
                # Confirm units/descriptions were applied.
                col_x = next(c for c in model.columns if c.name == "x")
                col_y = next(c for c in model.columns if c.name == "y")
                self.assertEqual(col_x.unit, u.m)
                self.assertEqual(col_y.description, "the y values")
                self.assertEqual(col_x.data.source, "ndf:/MORE/LSST/REC/X/DATA_ARRAY/DATA")
                self.assertEqual(col_y.data.source, "ndf:/MORE/LSST/REC/Y/DATA_ARRAY/DATA")
            with h5py.File(tmp.name, "r") as f:
                self.assertEqual(f["/MORE/LSST/REC/X"].attrs["CLASS"], b"NDF")
                np.testing.assert_array_equal(f["/MORE/LSST/REC/X/DATA_ARRAY/DATA"][()], rec["x"])
                self.assertEqual(f["/MORE/LSST/REC/Y"].attrs["CLASS"], b"NDF")
                np.testing.assert_array_equal(f["/MORE/LSST/REC/Y/DATA_ARRAY/DATA"][()], rec["y"])
            with NdfInputArchive.open(tmp.name) as archive:
                recovered = archive.get_structured_array(model)
                np.testing.assert_array_equal(recovered, rec)

    def test_add_single_column_structured_array_uses_table_name(self):
        rec = np.zeros(1, dtype=[("solution", np.float64, (4,))])
        rec["solution"] = [[1.0, 2.0, 3.0, 4.0]]
        with tempfile.NamedTemporaryFile(suffix=".sdf") as tmp:
            with h5py.File(tmp.name, "w") as f:
                arch = NdfOutputArchive(f)
                model = arch.add_structured_array(rec, name="psf/piff/interp/solution")
                self.assertEqual(len(model.columns), 1)
                column = model.columns[0]
                self.assertIsInstance(column.data, ArrayReferenceModel)
                self.assertEqual(
                    column.data.source,
                    "ndf:/MORE/LSST/PSF/PIFF/INTERP/SOLUTION/DATA_ARRAY/DATA",
                )
                self.assertEqual(column.data.shape, [4])
            with h5py.File(tmp.name, "r") as f:
                self.assertIn("PSF", f["/MORE/LSST"])
                self.assertIn("PIFF", f["/MORE/LSST/PSF"])
                self.assertIn("INTERP", f["/MORE/LSST/PSF/PIFF"])
                self.assertIn("SOLUTION", f["/MORE/LSST/PSF/PIFF/INTERP"])
                np.testing.assert_array_equal(
                    f["/MORE/LSST/PSF/PIFF/INTERP/SOLUTION/DATA_ARRAY/DATA"][()],
                    rec["solution"],
                )


@unittest.skipUnless(HAVE_H5PY, "h5py is not installed")
class NdfWriteWcsTestCase(unittest.TestCase):
    """Tests for /WCS/DATA serialization in ndf.write()."""

    def test_write_with_projection_creates_wcs_component(self):
        rng = np.random.default_rng(42)
        det_frame = DetectorFrame(instrument="TestInst", detector=4, bbox=Box.factory[1:4096, 1:4096])
        bbox = Box.factory[10:14, 20:25]
        projection = make_random_projection(rng, det_frame, Box.factory[1:4096, 1:4096])
        image = Image(
            np.arange(20, dtype=np.float32).reshape(4, 5),
            bbox=bbox,
            projection=projection,
        )
        with tempfile.NamedTemporaryFile(suffix=".sdf", delete_on_close=False) as tmp:
            tmp.close()
            write(image, tmp.name)
            with h5py.File(tmp.name, "r") as f:
                self.assertIn("WCS", f)
                self.assertEqual(f["/WCS"].attrs["CLASS"], b"WCS")
                wcs_data = f["/WCS/DATA"]
                self.assertEqual(wcs_data.dtype, np.dtype("|S32"))
                records = [s.decode("ascii").rstrip(" ") for s in wcs_data[()]]
                self.assertTrue(all(record[0] in {" ", "+"} for record in records))
                self.assertFalse(any(record.startswith("#") for record in records))
                text = _hds.decode_ndf_ast_data(records)
                stripped = [line.lstrip() for line in text.splitlines()]
                self.assertTrue(any(s.startswith("Begin FrameSet") for s in stripped))
                self.assertTrue(any(s.startswith("End FrameSet") for s in stripped))
                self.assertIn('Domain = "GRID"', stripped)
                self.assertIn('Domain = "PIXEL"', stripped)
                self.assertIn("Sft1 = -19", stripped)
                self.assertIn("Sft2 = -9", stripped)

    def test_write_without_projection_omits_wcs_component(self):
        # Image with no projection -> no /WCS in the file.
        image = Image(np.zeros((2, 2), dtype=np.float32))
        with tempfile.NamedTemporaryFile(suffix=".sdf", delete_on_close=False) as tmp:
            tmp.close()
            write(image, tmp.name)
            with h5py.File(tmp.name, "r") as f:
                self.assertNotIn("WCS", f)

    def test_mask_sub_ndf_gets_3d_wcs(self):
        # When an incompatible mask is hoisted to /MORE/LSST/MASK as a
        # sub-NDF, it should carry its own 3D /WCS.  The first two axes
        # retain the parent image sky projection while the third axis is
        # a generic mask-byte coordinate.
        rng = np.random.default_rng(42)
        det_frame = DetectorFrame(instrument="TestInst", detector=4, bbox=Box.factory[1:4096, 1:4096])
        bbox = Box.factory[10:14, 20:25]
        projection = make_random_projection(rng, det_frame, Box.factory[1:4096, 1:4096])
        # 12-plane schema -> native 3D uint8 mask, hoisted to /MORE/LSST/MASK.
        planes = [MaskPlane(f"P{i}", f"Plane {i}") for i in range(12)]
        image = Image(
            np.arange(20, dtype=np.float32).reshape(4, 5),
            bbox=bbox,
            projection=projection,
        )
        masked = MaskedImage(image, mask_schema=MaskSchema(planes))
        with tempfile.NamedTemporaryFile(suffix=".sdf", delete_on_close=False) as tmp:
            tmp.close()
            write(masked, tmp.name)
            with h5py.File(tmp.name, "r") as f:
                # Top-level WCS is present (existing behaviour).
                self.assertIn("WCS", f)
                top_lines = [s.decode("ascii") for s in f["/WCS/DATA"][()]]
                self.assertIn("MASK", f["/MORE/LSST"])
                self.assertIn("WCS", f["/MORE/LSST/MASK"])
                self.assertEqual(f["/MORE/LSST/MASK/WCS"].attrs["CLASS"], b"WCS")
                mask_lines = [s.decode("ascii") for s in f["/MORE/LSST/MASK/WCS/DATA"][()]]
                self.assertNotEqual(top_lines, mask_lines)
                mask_text = _hds.decode_ndf_ast_data(mask_lines)
                stripped = [line.lstrip() for line in mask_text.splitlines()]
                self.assertIn("Naxes = 3", stripped)
                self.assertIn('Domain = "GRID"', stripped)
                self.assertIn('Domain = "PIXEL"', stripped)
                self.assertIn("Sft1 = -19", stripped)
                self.assertIn("Sft2 = -9", stripped)
                self.assertIn("Sft3 = 1", stripped)
                self.assertIn("Begin CmpFrame", stripped)
                self.assertIn("Begin SkyFrame", stripped)
                self.assertIn('Domain = "MASK"', stripped)
                self.assertIn("Begin CmpMap", stripped)
                self.assertIn("Series = 0", stripped)

    def test_mask_sub_ndf_no_wcs_when_image_has_no_projection(self):
        planes = [MaskPlane(f"P{i}", f"Plane {i}") for i in range(12)]
        masked = MaskedImage(
            Image(np.zeros((4, 5), dtype=np.float32)),
            mask_schema=MaskSchema(planes),
        )
        with tempfile.NamedTemporaryFile(suffix=".sdf", delete_on_close=False) as tmp:
            tmp.close()
            write(masked, tmp.name)
            with h5py.File(tmp.name, "r") as f:
                self.assertNotIn("WCS", f)
                self.assertIn("MASK", f["/MORE/LSST"])
                self.assertNotIn("WCS", f["/MORE/LSST/MASK"])


@unittest.skipUnless(HAVE_H5PY, "h5py is not installed")
class NdfWriteFunctionTestCase(unittest.TestCase):
    """End-to-end tests for the module-level `write()` function."""

    def test_write_image_produces_valid_layout(self):
        image = Image(
            np.arange(20, dtype=np.float32).reshape(4, 5),
            bbox=Box.factory[10:14, 20:25],
        )
        with tempfile.NamedTemporaryFile(suffix=".sdf", delete_on_close=False) as tmp:
            tmp.close()
            tree = write(image, tmp.name)
            self.assertIsNotNone(tree)
            with h5py.File(tmp.name, "r") as f:
                # Root is an NDF with a name.
                self.assertEqual(f["/"].attrs["CLASS"], b"NDF")
                self.assertIn("HDS_ROOT_NAME", f["/"].attrs)
                # DATA_ARRAY uses the complex form (DATA + ORIGIN).
                self.assertEqual(f["/DATA_ARRAY"].attrs["CLASS"], b"ARRAY")
                np.testing.assert_array_equal(f["/DATA_ARRAY/DATA"][()], image.array)
                origin = f["/DATA_ARRAY/ORIGIN"][()]
                self.assertEqual(origin.dtype, np.int64)
                self.assertEqual(len(origin), 2)
                # ORIGIN encodes bbox lower bounds in Fortran order. The exact
                # values depend on Box's API; just verify it isn't the
                # all-zeros placeholder when the bbox is non-trivial.
                self.assertFalse((origin == 0).all())
                # Main JSON tree at /MORE/LSST/JSON.
                self.assertIn("MORE", f)
                self.assertIn("LSST", f["/MORE"])
                self.assertIn("JSON", f["/MORE/LSST"])

    def test_write_image_preserves_opaque_fits_metadata(self):
        image = Image(np.zeros((2, 2), dtype=np.float32))
        # Attach an opaque-metadata primary header to the image.
        primary = astropy.io.fits.Header()
        primary["FOO"] = ("bar", "test card")
        long_value = "x" * 100
        primary["LONGSTR"] = (long_value, "long string value")
        opaque = FitsOpaqueMetadata()
        opaque.add_header(primary, name="", ver=1)
        image._opaque_metadata = opaque
        with tempfile.NamedTemporaryFile(suffix=".sdf", delete_on_close=False) as tmp:
            tmp.close()
            write(image, tmp.name)
            with h5py.File(tmp.name, "r") as f:
                self.assertIn("FITS", f["/MORE"])
                cards = [c.decode("ascii").rstrip(" ") for c in f["/MORE/FITS"][()]]
                self.assertTrue(any(c.startswith("FOO") for c in cards))
                self.assertTrue(any(c.startswith("CONTINUE") for c in cards))
                self.assertTrue(all(len(c.encode("ascii")) <= 80 for c in cards))
            result = read(Image, tmp.name)
            recovered = result.deserialized._opaque_metadata.headers[ExtensionKey()]
            self.assertEqual(recovered["LONGSTR"], long_value)

    def test_write_image_main_json_round_trips_back(self):
        # Sanity: the main JSON tree at /MORE/LSST/JSON should parse as the
        # in-memory ArchiveTree and contain the array reference for DATA_ARRAY.
        image = Image(np.arange(6, dtype=np.float32).reshape(2, 3))
        with tempfile.NamedTemporaryFile(suffix=".sdf", delete_on_close=False) as tmp:
            tmp.close()
            tree = write(image, tmp.name)
            with h5py.File(tmp.name, "r") as f:
                raw = f["/MORE/LSST/JSON"][()]
            joined = b"".join(raw).decode("ascii").rstrip(" ")
            recovered = json.loads(joined)
            # The exact structure depends on Image's serialization model; we
            # just check the JSON is parseable and the ArchiveTree object the
            # write() function returned dumps to the same JSON.
            self.assertEqual(json.loads(tree.model_dump_json()), recovered)

    def test_write_image_with_unit_creates_units_component(self):
        image = Image(np.arange(6, dtype=np.float32).reshape(2, 3), unit=u.ct)
        with tempfile.NamedTemporaryFile(suffix=".sdf", delete_on_close=False) as tmp:
            tmp.close()
            write(image, tmp.name)
            with h5py.File(tmp.name, "r") as f:
                self.assertIn("UNITS", f)
                self.assertEqual(f["/UNITS"].shape, ())
                self.assertEqual(f["/UNITS"][()].decode("ascii").rstrip(" "), "count")
            result = read(Image, tmp.name)
            self.assertEqual(result.deserialized.unit, u.ct)

    def test_write_propagates_metadata(self):
        image = Image(np.arange(6, dtype=np.float32).reshape(2, 3))
        extra = {"test_key": 42, "another": "hello"}
        with tempfile.NamedTemporaryFile(suffix=".sdf", delete_on_close=False) as tmp:
            tmp.close()
            tree = write(image, tmp.name, metadata=extra)
            self.assertEqual(tree.metadata["test_key"], 42)
            self.assertEqual(tree.metadata["another"], "hello")
            result = read(Image, tmp.name)
            self.assertEqual(result.metadata["test_key"], 42)
            self.assertEqual(result.metadata["another"], "hello")

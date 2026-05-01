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

from lsst.images import Box, Image
from lsst.images.ndf._input_archive import NdfInputArchive
from lsst.images.ndf._output_archive import write


class NdfInputArchiveOpenTestCase(unittest.TestCase):
    """Tests for `NdfInputArchive.open` and `get_tree`."""

    def test_open_round_trips_image_tree(self):
        image = Image(
            np.arange(20, dtype=np.float32).reshape(4, 5),
            bbox=Box.factory[10:14, 20:25],
        )
        with tempfile.NamedTemporaryFile(suffix=".sdf", delete_on_close=False) as tmp:
            tmp.close()
            written_tree = write(image, tmp.name)
            with NdfInputArchive.open(tmp.name) as archive:
                tree = archive.get_tree(type(written_tree))
                self.assertEqual(tree.model_dump_json(), written_tree.model_dump_json())

    def test_get_tree_raises_when_main_json_missing(self):
        # A file with no /MORE/LSST/JSON should raise ArchiveReadError.
        import h5py

        from lsst.images.serialization import ArchiveReadError

        with tempfile.NamedTemporaryFile(suffix=".sdf", delete_on_close=False) as tmp:
            tmp.close()
            with h5py.File(tmp.name, "w") as f:
                f["/"].attrs["CLASS"] = "NDF"
            with NdfInputArchive.open(tmp.name) as archive:
                # type(written_tree) doesn't exist in scope here, but get_tree
                # raises before the type matters.
                from lsst.images._image import ImageSerializationModel
                from lsst.images.ndf._common import NdfPointerModel

                model_type = ImageSerializationModel[NdfPointerModel]
                with self.assertRaises(ArchiveReadError):
                    archive.get_tree(model_type)


class NdfInputArchiveDataTestCase(unittest.TestCase):
    """Tests for `get_array`, `deserialize_pointer`, and `get_frame_set`."""

    def test_get_array_reads_image_array(self):
        image = Image(np.arange(20, dtype=np.float32).reshape(4, 5))
        with tempfile.NamedTemporaryFile(suffix=".sdf", delete_on_close=False) as tmp:
            tmp.close()
            tree = write(image, tmp.name)
            with NdfInputArchive.open(tmp.name) as archive:
                # The Image tree's `data` attribute is an
                # ArrayReferenceModel pointing at /DATA_ARRAY/DATA.
                arr = archive.get_array(tree.data)
                np.testing.assert_array_equal(arr, image.array)

    def test_get_array_supports_slicing(self):
        image = Image(np.arange(20, dtype=np.float32).reshape(4, 5))
        with tempfile.NamedTemporaryFile(suffix=".sdf", delete_on_close=False) as tmp:
            tmp.close()
            tree = write(image, tmp.name)
            with NdfInputArchive.open(tmp.name) as archive:
                arr = archive.get_array(tree.data, slices=(slice(0, 2), slice(1, 4)))
                np.testing.assert_array_equal(arr, image.array[:2, 1:4])

    def test_get_array_handles_inline_array(self):
        from lsst.images.serialization import InlineArrayModel, NumberType

        inline = InlineArrayModel(data=[1.0, 2.0, 3.0], datatype=NumberType.float64)
        image = Image(np.zeros((2, 2), dtype=np.float32))
        with tempfile.NamedTemporaryFile(suffix=".sdf", delete_on_close=False) as tmp:
            tmp.close()
            write(image, tmp.name)
            with NdfInputArchive.open(tmp.name) as archive:
                arr = archive.get_array(inline)
                np.testing.assert_array_equal(arr, np.array([1.0, 2.0, 3.0]))

    def test_get_array_unrecognised_source_raises(self):
        from lsst.images.serialization import (
            ArchiveReadError,
            ArrayReferenceModel,
            NumberType,
        )

        image = Image(np.zeros((2, 2), dtype=np.float32))
        bogus = ArrayReferenceModel(source="fits:NOTUS", datatype=NumberType.float32)
        with tempfile.NamedTemporaryFile(suffix=".sdf", delete_on_close=False) as tmp:
            tmp.close()
            write(image, tmp.name)
            with NdfInputArchive.open(tmp.name) as archive:
                with self.assertRaises(ArchiveReadError):
                    archive.get_array(bogus)

    def test_deserialize_pointer_round_trips_subtree(self):
        # Build a file with a hoisted sub-tree we can read back. Use the
        # output archive directly to avoid pulling in the full Image stack.
        import h5py
        import pydantic

        from lsst.images.ndf._output_archive import NdfOutputArchive

        class TinyTree(pydantic.BaseModel):
            name: str

        with tempfile.NamedTemporaryFile(suffix=".sdf", delete_on_close=False) as tmp:
            tmp.close()
            with h5py.File(tmp.name, "w") as f:
                arch = NdfOutputArchive(f)
                ptr = arch.serialize_pointer("psf", lambda nested: TinyTree(name="hello"), key=("psf", 1))
            with NdfInputArchive.open(tmp.name) as archive:
                # Deserializer just returns the model unchanged.
                result = archive.deserialize_pointer(ptr, TinyTree, lambda m, _a: m)
                self.assertEqual(result.name, "hello")

    def test_deserialize_pointer_caches_by_ref(self):
        import h5py
        import pydantic

        from lsst.images.ndf._output_archive import NdfOutputArchive

        class TinyTree(pydantic.BaseModel):
            name: str

        calls = []

        def deserializer(model, _archive):
            calls.append(model)
            return model

        with tempfile.NamedTemporaryFile(suffix=".sdf", delete_on_close=False) as tmp:
            tmp.close()
            with h5py.File(tmp.name, "w") as f:
                arch = NdfOutputArchive(f)
                ptr = arch.serialize_pointer("psf", lambda nested: TinyTree(name="x"), key=("psf", 1))
            with NdfInputArchive.open(tmp.name) as archive:
                first = archive.deserialize_pointer(ptr, TinyTree, deserializer)
                second = archive.deserialize_pointer(ptr, TinyTree, deserializer)
                self.assertIs(first, second)
                self.assertEqual(len(calls), 1)

    def test_get_frame_set_returns_cached_value(self):
        # Exercise the cache mechanism with a sentinel object pretending
        # to be a FrameSet. Real FrameSet plumbing comes when the AST
        # text dump for /WCS/DATA lands in a follow-up task.
        from lsst.images.ndf._common import NdfPointerModel

        sentinel = object()
        with tempfile.NamedTemporaryFile(suffix=".sdf", delete_on_close=False) as tmp:
            tmp.close()
            write(Image(np.zeros((2, 2), dtype=np.float32)), tmp.name)
            with NdfInputArchive.open(tmp.name) as archive:
                # Manually populate the cache as deserialize_pointer would
                # if a FrameSet deserializer ran.
                archive._frame_set_cache["/MORE/LSST/PIXEL_TO_SKY"] = sentinel
                ref = NdfPointerModel(ref="/MORE/LSST/PIXEL_TO_SKY")
                self.assertIs(archive.get_frame_set(ref), sentinel)

    def test_get_frame_set_raises_if_not_cached(self):
        from lsst.images.ndf._common import NdfPointerModel

        with tempfile.NamedTemporaryFile(suffix=".sdf", delete_on_close=False) as tmp:
            tmp.close()
            write(Image(np.zeros((2, 2), dtype=np.float32)), tmp.name)
            with NdfInputArchive.open(tmp.name) as archive:
                ref = NdfPointerModel(ref="/MORE/LSST/UNKNOWN")
                with self.assertRaises(AssertionError):
                    archive.get_frame_set(ref)


class NdfInputArchiveOpaqueMetadataTestCase(unittest.TestCase):
    """Tests for `NdfInputArchive.get_opaque_metadata`."""

    def test_more_fits_round_trips_via_opaque_metadata(self):
        import astropy.io.fits

        from lsst.images.fits._common import ExtensionKey, FitsOpaqueMetadata

        image = Image(np.zeros((2, 2), dtype=np.float32))
        primary = astropy.io.fits.Header()
        primary["FOO"] = ("bar", "test card")
        opaque = FitsOpaqueMetadata()
        opaque.add_header(primary, name="", ver=1)
        image._opaque_metadata = opaque
        with tempfile.NamedTemporaryFile(suffix=".sdf", delete_on_close=False) as tmp:
            tmp.close()
            write(image, tmp.name)
            with NdfInputArchive.open(tmp.name) as archive:
                recovered = archive.get_opaque_metadata()
                self.assertIn(ExtensionKey(), recovered.headers)
                self.assertEqual(recovered.headers[ExtensionKey()]["FOO"], "bar")

    def test_get_opaque_metadata_empty_when_no_more_fits(self):
        from lsst.images.fits._common import FitsOpaqueMetadata

        # Image with no opaque metadata -> /MORE/FITS is absent in the file.
        image = Image(np.zeros((2, 2), dtype=np.float32))
        with tempfile.NamedTemporaryFile(suffix=".sdf", delete_on_close=False) as tmp:
            tmp.close()
            write(image, tmp.name)
            with NdfInputArchive.open(tmp.name) as archive:
                recovered = archive.get_opaque_metadata()
                self.assertIsInstance(recovered, FitsOpaqueMetadata)
                # No primary header should be populated since /MORE/FITS
                # was never written.
                self.assertFalse(recovered.headers)

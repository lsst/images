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

import astropy.io.fits
import astropy.units as u
import numpy as np
import numpy.typing as npt

from lsst.images import Box, Mask, MaskPlane, MaskSchema

DATA_DIR = os.environ.get("TESTDATA_IMAGES_DIR", None)


class MaskTestCase(unittest.TestCase):
    """Tests for Mask and its helper classes."""

    def setUp(self) -> None:
        self.maxDiff = None
        self.rng = np.random.default_rng(500)

    def make_mask_planes(self, n_planes: int, n_placeholders: int) -> list[MaskPlane | None]:
        planes: list[MaskPlane | None] = []
        for i in range(n_planes):
            planes.append(MaskPlane(f"M{i}", f"D{i}"))
        planes.extend([None] * n_placeholders)
        self.rng.shuffle(planes)
        return planes

    def test_schema(self) -> None:
        """Test MaskSchema."""
        self.assertEqual(MaskSchema.bits_per_element(np.uint8), 8)
        planes = self.make_mask_planes(17, 5)
        with self.assertRaises(TypeError):
            MaskSchema.bits_per_element(np.float32)
        schema = MaskSchema(planes, dtype=np.uint8)
        self.assertEqual(list(schema), planes)
        self.assertEqual(len(schema), len(planes))
        self.assertEqual(schema[5], planes[5])
        self.assertEqual(
            eval(repr(schema), {"dtype": np.dtype, "MaskSchema": MaskSchema, "MaskPlane": MaskPlane}), schema
        )
        string = str(schema)
        self.assertEqual(len(string.split("\n")), 17)
        bit5 = schema.bit("M5")
        self.assertIn(f"M5 [{bit5.index}@{hex(bit5.mask)}]: D5", string)
        self.assertEqual(schema, MaskSchema(planes, np.uint8))
        self.assertNotEqual(schema, MaskSchema(planes, np.int16))
        self.assertNotEqual(schema, MaskSchema(planes[:-1], np.uint8))
        self.assertEqual(schema.dtype, np.dtype(np.uint8))
        self.assertEqual(schema.mask_size, 3)
        self.assertEqual(schema.names, {f"M{i}" for i in range(17)})
        self.assertEqual(schema.descriptions, {f"M{i}": f"D{i}" for i in range(17)})
        bit7 = schema.bit("M7")
        bitmask57 = schema.bitmask("M5", "M7")
        self.assertTrue(bitmask57[bit5.index] & bit5.mask)
        self.assertTrue(bitmask57[bit7.index] & bit7.mask)
        bitmask57[bit5.index] &= ~bit5.mask
        bitmask57[bit7.index] &= ~bit7.mask
        self.assertFalse(bitmask57.any())
        splits = schema.split(np.int16)
        self.assertEqual(len(splits), 2)
        self.assertEqual(splits[0].mask_size, 1)
        self.assertEqual(splits[1].mask_size, 1)
        self.assertEqual(list(splits[0]) + list(splits[1]), [p for p in planes if p is not None])
        self.assertEqual(len(splits[0]), 15)
        self.assertEqual(len(splits[1]), 2)

    def test_serialize_multi(self) -> None:
        """Test serializing a mask with more than 31 mask planes, requiring
        more than one HDU and EXTVER.

        Note that serialization for simpler cases is covered by
        test_masked_image.py.
        """
        planes = self.make_mask_planes(35, n_placeholders=5)
        schema = MaskSchema(planes, dtype=np.uint8)
        bbox = Box.factory[5:50, 6:60]
        mask = Mask(0, schema=schema, bbox=bbox)
        shape = bbox.shape
        for plane in schema:
            if plane is not None:
                mask.set(plane.name, self.rng.random(shape) > 0.5)
        with tempfile.NamedTemporaryFile(suffix=".fits", delete_on_close=False, delete=True) as tmp:
            tmp.close()
            mask.write_fits(tmp.name)
            roundtripped = Mask.read_fits(tmp.name)
            with astropy.io.fits.open(tmp.name, disable_image_compression=True) as fits:
                self.assertEqual(fits[1].header["EXTNAME"], "MASK")
                self.assertEqual(fits[1].header.get("EXTVER", 1), 1)
                self.assertEqual(fits[1].header["ZCMPTYPE"], "GZIP_2")
                self.assertEqual(fits[2].header["EXTNAME"], "MASK")
                self.assertEqual(fits[2].header["EXTVER"], 2)
                self.assertEqual(fits[2].header["ZCMPTYPE"], "GZIP_2")
        self.assertEqual(roundtripped.bbox, mask.bbox)
        self.assertEqual(roundtripped.schema, mask.schema)
        np.testing.assert_array_equal(roundtripped.array, mask.array)


if __name__ == "__main__":
    unittest.main()

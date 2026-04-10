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
import unittest

import numpy as np
from astro_metadata_translator import ObservationInfo

import lsst.utils.tests
from lsst.images import Box, Mask, MaskPlane, MaskSchema, get_legacy_visit_image_mask_planes
from lsst.images.tests import RoundtripFits, assert_masks_equal, compare_mask_to_legacy

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

    def test_basics(self) -> None:
        """Test some basic Mask functionality."""
        planes = self.make_mask_planes(35, n_placeholders=5)
        schema = MaskSchema(planes, dtype=np.uint8)
        bbox = Box.factory[5:50, 6:60]
        mask = Mask(
            0,
            schema=schema,
            bbox=bbox,
            metadata={"four_and_a_half": 4.5},
            obs_info=ObservationInfo(instrument="LSSTCam"),
        )

        self.assertIs(mask[...], mask)
        self.assertEqual(mask.__eq__(42), NotImplemented)
        self.assertEqual(mask, mask)
        self.assertEqual(mask.obs_info.instrument, "LSSTCam")
        self.maxDiff = None
        self.assertEqual(
            str(mask),
            "Mask([y=5:50, x=6:60], ['M34', 'M15', 'M29', 'M1', 'M20', 'M11', 'M13', 'M7', 'M17', 'M12', "
            "'M31', 'M16', 'M2', 'M3', 'M8', 'M26', 'M22', 'M5', 'M18', 'M19', 'M24', 'M21', 'M27', 'M6', "
            "'M28', 'M10', 'M4', 'M23', 'M0', 'M25', 'M9', 'M14', 'M33', 'M32', 'M30'])",
        )
        self.assertTrue(
            repr(mask).startswith(
                "Mask(..., bbox=Box(y=Interval(start=5, stop=50), x=Interval(start=6, stop=60)), "
                "schema=MaskSchema([MaskPlane(name='M34', description='D34')"
            ),
            f"Repr: {mask!r}",
        )

        with self.assertRaises(TypeError):
            # No bbox, size or array.
            Mask(0, schema=schema)

        with self.assertRaises(ValueError):
            # Box mismatch.
            Mask(mask.array, schema=schema, bbox=Box.factory[0:20, -5:45])

        with self.assertRaises(ValueError):
            # Shape mismatch.
            Mask(mask.array, schema=schema, shape=(5, 10, 5))

        with self.assertRaises(ValueError):
            # Cannot be 2-D.
            Mask(mask.array.reshape((2430, 5)), schema=schema, bbox=Box.factory[0:20, -5:45])

    def test_read_write(self) -> None:
        """Explicit calls to read and write fits."""
        planes = self.make_mask_planes(35, n_placeholders=5)
        schema = MaskSchema(planes, dtype=np.uint8)
        bbox = Box.factory[5:50, 6:60]
        mask = Mask(
            0,
            schema=schema,
            bbox=bbox,
            metadata={"four_and_a_half": 4.5},
            obs_info=ObservationInfo(instrument="LSSTCam"),
        )
        with lsst.utils.tests.getTempFilePath(".fits") as tmpFile:
            mask.write_fits(tmpFile)

            new = Mask.read_fits(tmpFile)
            self.assertEqual(new, mask)
            # __eq__ ignores metadata.
            self.assertEqual(new.metadata["four_and_a_half"], 4.5)
            self.assertEqual(new.obs_info.instrument, "LSSTCam")
            self.assertEqual(new.obs_info, mask.obs_info)
            self.assertEqual(new.metadata, mask.metadata)

    def test_serialize_multi(self) -> None:
        """Test serializing a mask with more than 31 mask planes, requiring
        more than one HDU and EXTVER.

        Note that serialization for simpler cases is covered by
        test_masked_image.py.
        """
        planes = self.make_mask_planes(35, n_placeholders=5)
        schema = MaskSchema(planes, dtype=np.uint8)
        bbox = Box.factory[5:50, 6:60]
        mask = Mask(0, schema=schema, bbox=bbox, metadata={"four_and_a_half": 4.5})
        shape = bbox.shape
        for plane in schema:
            if plane is not None:
                mask.set(plane.name, self.rng.random(shape) > 0.5)
        with RoundtripFits(self, mask) as roundtrip:
            fits = roundtrip.inspect()
            self.assertEqual(fits[1].header["EXTNAME"], "MASK")
            self.assertEqual(fits[1].header.get("EXTVER", 1), 1)
            self.assertEqual(fits[1].header["ZCMPTYPE"], "GZIP_2")
            self.assertEqual(fits[2].header["EXTNAME"], "MASK")
            self.assertEqual(fits[2].header["EXTVER"], 2)
            self.assertEqual(fits[2].header["ZCMPTYPE"], "GZIP_2")
            n = 0
            for plane in planes:
                if plane is not None:
                    hdu = fits[1] if n < 31 else fits[2]
                    self.assertEqual(hdu.header[f"MSKN{(n % 31):04d}"], plane.name)
                    self.assertEqual(hdu.header[f"MSKM{(n % 31):04d}"], 1 << (n % 31))
                    self.assertEqual(hdu.header[f"MSKD{(n % 31):04d}"], plane.description)
                    n += 1
        assert_masks_equal(self, mask, roundtrip.result)

    @unittest.skipUnless(DATA_DIR is not None, "TESTDATA_IMAGES_DIR is not in the environment.")
    def test_legacy(self) -> None:
        """Test Mask.read_legacy, Mask.to_legacy, and Mask.from_legacy."""
        assert DATA_DIR is not None, "Guaranteed by decorator."
        filename = os.path.join(DATA_DIR, "dp2", "legacy", "visit_image.fits")
        plane_map = get_legacy_visit_image_mask_planes()
        mask = Mask.read_legacy(filename, ext=2, plane_map=plane_map)
        try:
            from lsst.afw.image import MaskedImageFitsReader
        except ImportError:
            raise unittest.SkipTest("'lsst.afw.image' could not be imported.") from None
        reader = MaskedImageFitsReader(filename)
        self.assertEqual(mask.bbox, Box.from_legacy(reader.readBBox()))
        legacy_mask = reader.readMask()
        compare_mask_to_legacy(self, mask, legacy_mask, plane_map)
        compare_mask_to_legacy(self, mask, mask.to_legacy(plane_map), plane_map)
        assert_masks_equal(self, mask, Mask.from_legacy(legacy_mask, plane_map=plane_map))


if __name__ == "__main__":
    unittest.main()

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

import astropy.io.fits
import numpy as np

import lsst.utils.tests
from lsst.images import (
    Box,
    Mask,
    MaskPlane,
    MaskSchema,
    get_legacy_non_cell_coadd_mask_planes,
    get_legacy_visit_image_mask_planes,
)
from lsst.images._mask import _guess_legacy_plane_map
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

    def test_schema_from_fits_header(self) -> None:
        """MaskSchema.from_fits_header inverts update_header, assuming the
        default uint8 dtype.
        """
        planes = [
            MaskPlane("NO_DATA", "No data was available for this pixel."),
            MaskPlane("COSMIC_RAY", "A cosmic ray affected this pixel."),
            MaskPlane("DETECTED", "Pixel was part of a detected source."),
        ]
        schema = MaskSchema(planes, dtype=np.uint8)
        header = astropy.io.fits.Header()
        schema.update_header(header)
        result = MaskSchema.from_fits_header(header)
        self.assertEqual(result.dtype, np.dtype(np.uint8))
        self.assertEqual(list(result), planes)
        self.assertEqual(result, schema)

    def test_schema_from_fits_header_preserves_gaps(self) -> None:
        """A None placeholder between planes is reconstructed from the gap in
        the MSKN card numbering.
        """
        planes: list[MaskPlane | None] = [MaskPlane("A", "a"), None, MaskPlane("B", "b")]
        header = astropy.io.fits.Header()
        MaskSchema(planes, dtype=np.uint8).update_header(header)
        self.assertEqual(list(MaskSchema.from_fits_header(header)), planes)

    def test_schema_from_fits_header_requires_cards(self) -> None:
        """A header with no MSKN cards cannot describe a schema."""
        with self.assertRaises(ValueError):
            MaskSchema.from_fits_header(astropy.io.fits.Header())

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
        )

        self.assertIs(mask[...], mask)
        self.assertEqual(mask.__eq__(42), NotImplemented)
        self.assertEqual(mask, mask)
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
        )
        with lsst.utils.tests.getTempFilePath(".fits") as tmpFile:
            mask.write(tmpFile)

            new = Mask.read(tmpFile)
            self.assertEqual(new, mask)
            # __eq__ ignores metadata.
            self.assertEqual(new.metadata["four_and_a_half"], 4.5)
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

    def test_add_plane_returns_new_mask(self) -> None:
        """Adding a plane returns a new mask, leaves the original (and any
        views of it) untouched, and always reallocates the backing array.
        """
        planes = self.make_mask_planes(3, n_placeholders=0)
        schema = MaskSchema(planes, dtype=np.uint8)
        bbox = Box.factory[5:50, 6:60]
        mask = Mask(0, schema=schema, bbox=bbox)
        m0 = self.rng.random(bbox.shape) > 0.5
        mask.set("M0", m0)
        view = mask[bbox]  # shares the array and old schema with mask
        original_array = mask.array

        new_mask = mask.add_plane("OUTSIDE_STENCIL", "Pixel lies outside the stencil.")

        # The original mask and any views keep the old schema and array.
        self.assertNotIn("OUTSIDE_STENCIL", mask.schema.names)
        self.assertNotIn("OUTSIDE_STENCIL", view.schema.names)
        self.assertIs(mask.array, original_array)
        # The new mask reallocated a fresh array and carries the new plane.
        self.assertIsNot(new_mask.array, original_array)
        self.assertIn("OUTSIDE_STENCIL", new_mask.schema.names)
        self.assertEqual(new_mask.schema.descriptions["OUTSIDE_STENCIL"], "Pixel lies outside the stencil.")
        # The new plane is the fourth (overall index 3) so it lives in byte 0.
        bit = new_mask.schema.bit("OUTSIDE_STENCIL")
        self.assertEqual(bit.index, 0)
        self.assertEqual(bit.mask, 1 << 3)
        self.assertEqual(new_mask.schema.mask_size, 1)
        # Existing plane data is preserved and the new plane starts all-False.
        np.testing.assert_array_equal(new_mask.get("M0"), m0)
        self.assertFalse(new_mask.get("OUTSIDE_STENCIL").any())

    def test_add_plane_grows_byte(self) -> None:
        """Adding a ninth plane (crossing the 8-plane boundary) gives the new
        mask an extra byte while preserving existing plane data.
        """
        planes = self.make_mask_planes(8, n_placeholders=0)
        schema = MaskSchema(planes, dtype=np.uint8)
        bbox = Box.factory[5:50, 6:60]
        mask = Mask(0, schema=schema, bbox=bbox)
        set_planes = {}
        for plane in planes:
            assert plane is not None
            boolean_mask = self.rng.random(bbox.shape) > 0.5
            mask.set(plane.name, boolean_mask)
            set_planes[plane.name] = boolean_mask

        new_mask = mask.add_plane("OUTSIDE_STENCIL", "Pixel lies outside the stencil.")

        # The original is unchanged; the new mask spills into a second byte.
        self.assertEqual(mask.schema.mask_size, 1)
        bit = new_mask.schema.bit("OUTSIDE_STENCIL")
        self.assertEqual(bit.index, 1)
        self.assertEqual(bit.mask, 1 << 0)
        self.assertEqual(new_mask.schema.mask_size, 2)
        self.assertEqual(new_mask.array.shape, bbox.shape + (2,))
        self.assertFalse(new_mask.get("OUTSIDE_STENCIL").any())
        # Every pre-existing plane keeps its data.
        for name, boolean_mask in set_planes.items():
            np.testing.assert_array_equal(new_mask.get(name), boolean_mask)

    def test_add_planes_multiple(self) -> None:
        """Several planes can be added in a single call."""
        planes = self.make_mask_planes(3, n_placeholders=0)
        bbox = Box.factory[0:4, 0:5]
        mask = Mask(0, schema=MaskSchema(planes, dtype=np.uint8), bbox=bbox)
        m0 = self.rng.random(bbox.shape) > 0.5
        mask.set("M0", m0)

        new_mask = mask.add_planes([MaskPlane("A", "plane a"), MaskPlane("B", "plane b")])

        self.assertEqual(set(mask.schema.names), {"M0", "M1", "M2"})  # original unchanged
        self.assertEqual(set(new_mask.schema.names), {"M0", "M1", "M2", "A", "B"})
        np.testing.assert_array_equal(new_mask.get("M0"), m0)
        self.assertFalse(new_mask.get("A").any())
        self.assertFalse(new_mask.get("B").any())

    def test_add_planes_drop_reassigns_bits(self) -> None:
        """Dropping a plane compacts the schema, reassigns the planes after it
        to lower bits, and repacks the retained pixel values by name.
        """
        bbox = Box.factory[0:4, 0:5]
        schema = MaskSchema([MaskPlane("A", "a"), MaskPlane("B", "b"), MaskPlane("C", "c")], dtype=np.uint8)
        mask = Mask(0, schema=schema, bbox=bbox)
        a = self.rng.random(bbox.shape) > 0.5
        c = self.rng.random(bbox.shape) > 0.5
        mask.set("A", a)
        mask.set("B", self.rng.random(bbox.shape) > 0.5)
        mask.set("C", c)

        new_mask = mask.add_planes([MaskPlane("D", "d")], drop=["B"])

        # B is gone; D is appended after the retained planes.
        self.assertEqual(list(new_mask.schema.names), ["A", "C", "D"])
        self.assertNotIn("B", new_mask.schema.names)
        # C moved down from bit 2 to bit 1; D takes bit 2.
        self.assertEqual(new_mask.schema.bit("A").mask, 1 << 0)
        self.assertEqual(new_mask.schema.bit("C").mask, 1 << 1)
        self.assertEqual(new_mask.schema.bit("D").mask, 1 << 2)
        # Retained pixel values follow their planes; the new plane is cleared.
        np.testing.assert_array_equal(new_mask.get("A"), a)
        np.testing.assert_array_equal(new_mask.get("C"), c)
        self.assertFalse(new_mask.get("D").any())

    def test_add_planes_with_placeholder(self) -> None:
        """``None`` placeholders reserve bits.  A pre-existing placeholder
        keeps its position, and a ``None`` interleaved in the added planes
        stays where it is placed rather than moving to the end; both survive
        a round-trip.
        """
        bbox = Box.factory[0:4, 0:5]
        # Schema with a pre-existing placeholder reserving bit 1.
        schema = MaskSchema([MaskPlane("A", "a"), None, MaskPlane("B", "b")], dtype=np.uint8)
        mask = Mask(0, schema=schema, bbox=bbox)
        a = self.rng.random(bbox.shape) > 0.5
        b = self.rng.random(bbox.shape) > 0.5
        mask.set("A", a)
        mask.set("B", b)

        # Append a block that itself contains an interior placeholder.
        new_mask = mask.add_planes([MaskPlane("C", "c"), None, MaskPlane("D", "d")])

        # The pre-existing placeholder stays at bit 1; the added placeholder
        # stays between C and D (bit 4), not at the end.
        self.assertEqual(
            list(new_mask.schema),
            [MaskPlane("A", "a"), None, MaskPlane("B", "b"), MaskPlane("C", "c"), None, MaskPlane("D", "d")],
        )
        self.assertEqual(new_mask.schema.bit("A").mask, 1 << 0)
        self.assertEqual(new_mask.schema.bit("B").mask, 1 << 2)
        self.assertEqual(new_mask.schema.bit("C").mask, 1 << 3)
        self.assertEqual(new_mask.schema.bit("D").mask, 1 << 5)
        # Retained pixel values follow their planes; new planes start cleared.
        np.testing.assert_array_equal(new_mask.get("A"), a)
        np.testing.assert_array_equal(new_mask.get("B"), b)
        self.assertFalse(new_mask.get("C").any())
        self.assertFalse(new_mask.get("D").any())

        with RoundtripFits(self, new_mask) as roundtrip:
            assert_masks_equal(self, new_mask, roundtrip.result)

    def test_add_planes_drop_unknown_raises(self) -> None:
        """Dropping a plane that does not exist is an error."""
        mask = Mask(0, schema=MaskSchema([MaskPlane("A", "a")], dtype=np.uint8), bbox=Box.factory[0:2, 0:2])
        with self.assertRaises(ValueError):
            mask.add_planes([], drop=["NOPE"])

    def test_add_plane_duplicate_raises(self) -> None:
        """Adding a plane whose name already exists is an error."""
        planes = self.make_mask_planes(3, n_placeholders=0)
        schema = MaskSchema(planes, dtype=np.uint8)
        mask = Mask(0, schema=schema, bbox=Box.factory[0:4, 0:4])
        with self.assertRaises(ValueError):
            mask.add_plane("M0", "Duplicate of an existing plane.")

    def test_add_plane_roundtrip(self) -> None:
        """A runtime-added plane and its data survive a FITS round trip."""
        planes = self.make_mask_planes(8, n_placeholders=0)
        schema = MaskSchema(planes, dtype=np.uint8)
        bbox = Box.factory[5:50, 6:60]
        mask = Mask(0, schema=schema, bbox=bbox)
        mask = mask.add_plane("OUTSIDE_STENCIL", "Pixel lies outside the stencil.")
        mask.set("OUTSIDE_STENCIL", self.rng.random(bbox.shape) > 0.5)
        with lsst.utils.tests.getTempFilePath(".fits") as tmpFile:
            mask.write(tmpFile)
            new = Mask.read(tmpFile)
        self.assertEqual(new, mask)
        self.assertEqual(new.schema.descriptions["OUTSIDE_STENCIL"], "Pixel lies outside the stencil.")
        assert_masks_equal(self, new, mask)

    def test_legacy_non_cell_coadd_plane_map(self) -> None:
        """The non-cell coadd map defines a distinct ``SENSOR_EDGE`` plane."""
        plane_map = get_legacy_non_cell_coadd_mask_planes()
        self.assertIn("SENSOR_EDGE", plane_map)
        self.assertEqual(plane_map["SENSOR_EDGE"].name, "SENSOR_EDGE")

    def test_guess_legacy_plane_map_coadd_discriminator(self) -> None:
        """``INEXACT_PSF`` routes to a coadd map; ``SENSOR_EDGE`` distinguishes
        non-cell coadds (which use it) from cell coadds (which use CELL_EDGE).
        """
        non_cell = _guess_legacy_plane_map({"INEXACT_PSF": 11, "SENSOR_EDGE": 14})
        self.assertIn("SENSOR_EDGE", non_cell)
        cell = _guess_legacy_plane_map({"INEXACT_PSF": 11})
        self.assertNotIn("SENSOR_EDGE", cell)

    @unittest.skipUnless(DATA_DIR is not None, "TESTDATA_IMAGES_DIR is not in the environment.")
    def test_legacy(self) -> None:
        """Test Mask.read_legacy, Mask.to_legacy, and Mask.from_legacy."""
        assert DATA_DIR is not None, "Guaranteed by decorator."
        filename = os.path.join(DATA_DIR, "dp2", "legacy", "visit_image.fits")
        plane_map = get_legacy_visit_image_mask_planes()
        mask = Mask.read_legacy(filename, ext=2, plane_map=plane_map)
        self.assertEqual(mask.schema.names, {p.name for p in plane_map.values()})
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
        # Write the mask out in the new format, and test that we can read it
        # back either way.
        with RoundtripFits(self, mask, storage_class="MaskV2") as roundtrip:
            with self.subTest():
                try:
                    import lsst.afw.image
                except ImportError:
                    raise unittest.SkipTest("afw could not be imported") from None
                legacy_mask = roundtrip.get(storageClass="Mask")
                self.assertIsInstance(legacy_mask, lsst.afw.image.Mask)
                compare_mask_to_legacy(self, mask, legacy_mask)
        assert_masks_equal(self, roundtrip.result, mask)


if __name__ == "__main__":
    unittest.main()

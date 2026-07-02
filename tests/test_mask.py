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

import dataclasses
import os
from typing import Any

import astropy.io.fits
import numpy as np
import pytest

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

try:
    from lsst.afw.image import MaskedImageReader as LegacyMaskedImageReader

except ImportError:
    type LegacyMaskedImageReader = Any  # type: ignore[no-redef]

EXTERNAL_DATA_DIR = os.environ.get("TESTDATA_IMAGES_DIR", None)


@dataclasses.dataclass
class _LegacyTestData:
    mask: Mask
    reader: LegacyMaskedImageReader
    plane_map: dict[str, MaskPlane]


@pytest.fixture(scope="session")
def legacy_test_data() -> _LegacyTestData:
    """Return a Mask read directly from the legacy test dataset and a legacy
    reader for that image.

    Skips if TESTDATA_IMAGES_DIR is unset or lsst.afw.image is unavailable.
    """
    if EXTERNAL_DATA_DIR is None:
        pytest.skip("TESTDATA_IMAGES_DIR is not in the environment.")
    try:
        from lsst.afw.image import MaskedImageFitsReader
    except ImportError:
        pytest.skip("'lsst.afw.image' could not be imported.")
    filename = os.path.join(EXTERNAL_DATA_DIR, "dp2", "legacy", "visit_image.fits")
    plane_map = get_legacy_visit_image_mask_planes()
    mask = Mask.read_legacy(filename, ext=2, plane_map=plane_map)
    reader = MaskedImageFitsReader(filename)
    return _LegacyTestData(mask=mask, reader=reader, plane_map=plane_map)


def make_mask_planes(rng: np.random.Generator, n_planes: int, n_placeholders: int) -> list[MaskPlane | None]:
    """Return a shuffled list of MaskPlane objects with placeholder Nones."""
    planes: list[MaskPlane | None] = []
    for i in range(n_planes):
        planes.append(MaskPlane(f"M{i}", f"D{i}"))
    planes.extend([None] * n_placeholders)
    rng.shuffle(planes)
    return planes


def test_schema() -> None:
    """Test MaskSchema construction, accessors, and basic operations."""
    rng = np.random.default_rng(500)
    planes = make_mask_planes(rng, 17, 5)
    with pytest.raises(TypeError):
        MaskSchema.bits_per_element(np.float32)
    assert MaskSchema.bits_per_element(np.uint8) == 8
    schema = MaskSchema(planes, dtype=np.uint8)
    assert list(schema) == planes
    assert len(schema) == len(planes)
    assert schema[5] == planes[5]
    assert eval(repr(schema), {"dtype": np.dtype, "MaskSchema": MaskSchema, "MaskPlane": MaskPlane}) == schema
    string = str(schema)
    assert len(string.split("\n")) == 17
    bit5 = schema.bit("M5")
    assert f"M5 [{bit5.index}@{hex(bit5.mask)}]: D5" in string
    assert schema == MaskSchema(planes, np.uint8)
    assert schema != MaskSchema(planes, np.int16)
    assert schema != MaskSchema(planes[:-1], np.uint8)
    assert schema.dtype == np.dtype(np.uint8)
    assert schema.mask_size == 3
    assert schema.names == {f"M{i}" for i in range(17)}
    assert schema.descriptions == {f"M{i}": f"D{i}" for i in range(17)}
    bit7 = schema.bit("M7")
    bitmask57 = schema.bitmask("M5", "M7")
    assert bitmask57[bit5.index] & bit5.mask
    assert bitmask57[bit7.index] & bit7.mask
    bitmask57[bit5.index] &= ~bit5.mask
    bitmask57[bit7.index] &= ~bit7.mask
    assert not bitmask57.any()
    splits = schema.split(np.int16)
    assert len(splits) == 2
    assert splits[0].mask_size == 1
    assert splits[1].mask_size == 1
    assert list(splits[0]) + list(splits[1]) == [p for p in planes if p is not None]
    assert len(splits[0]) == 15
    assert len(splits[1]) == 2


def test_schema_from_fits_header() -> None:
    """Verify MaskSchema.from_fits_header inverts update_header."""
    planes = [
        MaskPlane("NO_DATA", "No data was available for this pixel."),
        MaskPlane("COSMIC_RAY", "A cosmic ray affected this pixel."),
        MaskPlane("DETECTED", "Pixel was part of a detected source."),
    ]
    schema = MaskSchema(planes, dtype=np.uint8)
    header = astropy.io.fits.Header()
    schema.update_header(header)
    result = MaskSchema.from_fits_header(header)
    assert result.dtype == np.dtype(np.uint8)
    assert list(result) == planes
    assert result == schema


def test_schema_from_fits_header_preserves_gaps() -> None:
    """Verify None placeholders are reconstructed from gaps in MSKN card
    numbering.
    """
    planes: list[MaskPlane | None] = [MaskPlane("A", "a"), None, MaskPlane("B", "b")]
    header = astropy.io.fits.Header()
    MaskSchema(planes, dtype=np.uint8).update_header(header)
    assert list(MaskSchema.from_fits_header(header)) == planes


def test_schema_from_fits_header_requires_cards() -> None:
    """Verify MaskSchema.from_fits_header raises ValueError on a header with
    no MSKN cards.
    """
    with pytest.raises(ValueError):
        MaskSchema.from_fits_header(astropy.io.fits.Header())


def test_basics() -> None:
    """Test basic Mask construction, string representation, and error
    conditions.
    """
    rng = np.random.default_rng(500)
    planes = make_mask_planes(rng, 35, n_placeholders=5)
    schema = MaskSchema(planes, dtype=np.uint8)
    bbox = Box.factory[5:50, 6:60]
    mask = Mask(
        0,
        schema=schema,
        bbox=bbox,
        metadata={"four_and_a_half": 4.5},
    )

    assert mask[...] is mask
    assert mask.__eq__(42) == NotImplemented
    assert mask == mask
    assert (
        str(mask)
        == "Mask([y=5:50, x=6:60], ['M34', 'M15', 'M29', 'M1', 'M20', 'M11', 'M13', 'M7', 'M17', 'M12', "
        "'M31', 'M16', 'M2', 'M3', 'M8', 'M26', 'M22', 'M5', 'M18', 'M19', 'M24', 'M21', 'M27', 'M6', "
        "'M28', 'M10', 'M4', 'M23', 'M0', 'M25', 'M9', 'M14', 'M33', 'M32', 'M30'])"
    )
    assert repr(mask).startswith(
        "Mask(..., bbox=Box(y=Interval(start=5, stop=50), x=Interval(start=6, stop=60)), "
        "schema=MaskSchema([MaskPlane(name='M34', description='D34')"
    ), f"Repr: {mask!r}"

    with pytest.raises(TypeError):
        # No bbox, size or array.
        Mask(0, schema=schema)

    with pytest.raises(ValueError):
        # Box mismatch.
        Mask(mask.array, schema=schema, bbox=Box.factory[0:20, -5:45])

    with pytest.raises(ValueError):
        # Shape mismatch.
        Mask(mask.array, schema=schema, shape=(5, 10, 5))

    with pytest.raises(ValueError):
        # Cannot be 2-D.
        Mask(mask.array.reshape((2430, 5)), schema=schema, bbox=Box.factory[0:20, -5:45])


def test_read_write() -> None:
    """Test explicit calls to Mask.read and Mask.write through FITS."""
    rng = np.random.default_rng(500)
    planes = make_mask_planes(rng, 35, n_placeholders=5)
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
        assert new == mask
        # __eq__ ignores metadata.
        assert new.metadata["four_and_a_half"] == 4.5
        assert new.metadata == mask.metadata


def test_serialize_multi() -> None:
    """Test serializing a mask with more than 31 mask planes (multiple
    HDUs).
    """
    rng = np.random.default_rng(500)
    planes = make_mask_planes(rng, 35, n_placeholders=5)
    schema = MaskSchema(planes, dtype=np.uint8)
    bbox = Box.factory[5:50, 6:60]
    mask = Mask(0, schema=schema, bbox=bbox, metadata={"four_and_a_half": 4.5})
    shape = bbox.shape
    for plane in schema:
        if plane is not None:
            mask.set(plane.name, rng.random(shape) > 0.5)
    with RoundtripFits(mask) as roundtrip:
        fits = roundtrip.inspect()
        assert fits[1].header["EXTNAME"] == "MASK"
        assert fits[1].header.get("EXTVER", 1) == 1
        assert fits[1].header["ZCMPTYPE"] == "GZIP_2"
        assert fits[2].header["EXTNAME"] == "MASK"
        assert fits[2].header["EXTVER"] == 2
        assert fits[2].header["ZCMPTYPE"] == "GZIP_2"
        n = 0
        for plane in planes:
            if plane is not None:
                hdu = fits[1] if n < 31 else fits[2]
                assert hdu.header[f"MSKN{(n % 31):04d}"] == plane.name
                assert hdu.header[f"MSKM{(n % 31):04d}"] == 1 << (n % 31)
                assert hdu.header[f"MSKD{(n % 31):04d}"] == plane.description
                n += 1
    assert_masks_equal(mask, roundtrip.result)


def test_add_plane_returns_new_mask() -> None:
    """Verify add_plane returns a new mask without modifying the original or
    its views.
    """
    rng = np.random.default_rng(500)
    planes = make_mask_planes(rng, 3, n_placeholders=0)
    schema = MaskSchema(planes, dtype=np.uint8)
    bbox = Box.factory[5:50, 6:60]
    mask = Mask(0, schema=schema, bbox=bbox)
    m0 = rng.random(bbox.shape) > 0.5
    mask.set("M0", m0)
    view = mask[bbox]  # shares the array and old schema with mask
    original_array = mask.array

    new_mask = mask.add_plane("OUTSIDE_STENCIL", "Pixel lies outside the stencil.")

    # The original mask and any views keep the old schema and array.
    assert "OUTSIDE_STENCIL" not in mask.schema.names
    assert "OUTSIDE_STENCIL" not in view.schema.names
    assert mask.array is original_array
    # The new mask reallocated a fresh array and carries the new plane.
    assert new_mask.array is not original_array
    assert "OUTSIDE_STENCIL" in new_mask.schema.names
    assert new_mask.schema.descriptions["OUTSIDE_STENCIL"] == "Pixel lies outside the stencil."
    # The new plane is the fourth (overall index 3) so it lives in byte 0.
    bit = new_mask.schema.bit("OUTSIDE_STENCIL")
    assert bit.index == 0
    assert bit.mask == 1 << 3
    assert new_mask.schema.mask_size == 1
    # Existing plane data is preserved and the new plane starts all-False.
    np.testing.assert_array_equal(new_mask.get("M0"), m0)
    assert not new_mask.get("OUTSIDE_STENCIL").any()


def test_add_plane_grows_byte() -> None:
    """Verify adding a ninth plane crosses the 8-plane boundary into a
    second byte.
    """
    rng = np.random.default_rng(500)
    planes = make_mask_planes(rng, 8, n_placeholders=0)
    schema = MaskSchema(planes, dtype=np.uint8)
    bbox = Box.factory[5:50, 6:60]
    mask = Mask(0, schema=schema, bbox=bbox)
    set_planes = {}
    for plane in planes:
        assert plane is not None
        boolean_mask = rng.random(bbox.shape) > 0.5
        mask.set(plane.name, boolean_mask)
        set_planes[plane.name] = boolean_mask

    new_mask = mask.add_plane("OUTSIDE_STENCIL", "Pixel lies outside the stencil.")

    # The original is unchanged; the new mask spills into a second byte.
    assert mask.schema.mask_size == 1
    bit = new_mask.schema.bit("OUTSIDE_STENCIL")
    assert bit.index == 1
    assert bit.mask == 1 << 0
    assert new_mask.schema.mask_size == 2
    assert new_mask.array.shape == bbox.shape + (2,)
    assert not new_mask.get("OUTSIDE_STENCIL").any()
    # Every pre-existing plane keeps its data.
    for name, boolean_mask in set_planes.items():
        np.testing.assert_array_equal(new_mask.get(name), boolean_mask)


def test_add_planes_multiple() -> None:
    """Verify add_planes adds several planes in a single call."""
    rng = np.random.default_rng(500)
    planes = make_mask_planes(rng, 3, n_placeholders=0)
    bbox = Box.factory[0:4, 0:5]
    mask = Mask(0, schema=MaskSchema(planes, dtype=np.uint8), bbox=bbox)
    m0 = rng.random(bbox.shape) > 0.5
    mask.set("M0", m0)

    new_mask = mask.add_planes([MaskPlane("A", "plane a"), MaskPlane("B", "plane b")])

    assert set(mask.schema.names) == {"M0", "M1", "M2"}  # original unchanged
    assert set(new_mask.schema.names) == {"M0", "M1", "M2", "A", "B"}
    np.testing.assert_array_equal(new_mask.get("M0"), m0)
    assert not new_mask.get("A").any()
    assert not new_mask.get("B").any()


def test_add_planes_drop_reassigns_bits() -> None:
    """Verify dropping a plane compacts the schema and repacks pixel values."""
    rng = np.random.default_rng(500)
    bbox = Box.factory[0:4, 0:5]
    schema = MaskSchema([MaskPlane("A", "a"), MaskPlane("B", "b"), MaskPlane("C", "c")], dtype=np.uint8)
    mask = Mask(0, schema=schema, bbox=bbox)
    a = rng.random(bbox.shape) > 0.5
    c = rng.random(bbox.shape) > 0.5
    mask.set("A", a)
    mask.set("B", rng.random(bbox.shape) > 0.5)
    mask.set("C", c)

    new_mask = mask.add_planes([MaskPlane("D", "d")], drop=["B"])

    # B is gone; D is appended after the retained planes.
    assert list(new_mask.schema.names) == ["A", "C", "D"]
    assert "B" not in new_mask.schema.names
    # C moved down from bit 2 to bit 1; D takes bit 2.
    assert new_mask.schema.bit("A").mask == 1 << 0
    assert new_mask.schema.bit("C").mask == 1 << 1
    assert new_mask.schema.bit("D").mask == 1 << 2
    # Retained pixel values follow their planes; the new plane is cleared.
    np.testing.assert_array_equal(new_mask.get("A"), a)
    np.testing.assert_array_equal(new_mask.get("C"), c)
    assert not new_mask.get("D").any()


def test_add_planes_with_placeholder() -> None:
    """Verify None placeholders reserve bits and survive add_planes and a
    FITS round-trip.
    """
    rng = np.random.default_rng(500)
    bbox = Box.factory[0:4, 0:5]
    # Schema with a pre-existing placeholder reserving bit 1.
    schema = MaskSchema([MaskPlane("A", "a"), None, MaskPlane("B", "b")], dtype=np.uint8)
    mask = Mask(0, schema=schema, bbox=bbox)
    a = rng.random(bbox.shape) > 0.5
    b = rng.random(bbox.shape) > 0.5
    mask.set("A", a)
    mask.set("B", b)

    # Append a block that itself contains an interior placeholder.
    new_mask = mask.add_planes([MaskPlane("C", "c"), None, MaskPlane("D", "d")])

    # The pre-existing placeholder stays at bit 1; the added placeholder
    # stays between C and D (bit 4), not at the end.
    assert list(new_mask.schema) == [
        MaskPlane("A", "a"),
        None,
        MaskPlane("B", "b"),
        MaskPlane("C", "c"),
        None,
        MaskPlane("D", "d"),
    ]
    assert new_mask.schema.bit("A").mask == 1 << 0
    assert new_mask.schema.bit("B").mask == 1 << 2
    assert new_mask.schema.bit("C").mask == 1 << 3
    assert new_mask.schema.bit("D").mask == 1 << 5
    # Retained pixel values follow their planes; new planes start cleared.
    np.testing.assert_array_equal(new_mask.get("A"), a)
    np.testing.assert_array_equal(new_mask.get("B"), b)
    assert not new_mask.get("C").any()
    assert not new_mask.get("D").any()

    with RoundtripFits(new_mask) as roundtrip:
        assert_masks_equal(new_mask, roundtrip.result)


def test_add_planes_drop_unknown_raises() -> None:
    """Verify dropping a non-existent plane raises ValueError."""
    mask = Mask(0, schema=MaskSchema([MaskPlane("A", "a")], dtype=np.uint8), bbox=Box.factory[0:2, 0:2])
    with pytest.raises(ValueError):
        mask.add_planes([], drop=["NOPE"])


def test_add_plane_duplicate_raises() -> None:
    """Verify adding a plane whose name already exists raises ValueError."""
    rng = np.random.default_rng(500)
    planes = make_mask_planes(rng, 3, n_placeholders=0)
    schema = MaskSchema(planes, dtype=np.uint8)
    mask = Mask(0, schema=schema, bbox=Box.factory[0:4, 0:4])
    with pytest.raises(ValueError):
        mask.add_plane("M0", "Duplicate of an existing plane.")


def test_add_plane_roundtrip() -> None:
    """Verify a runtime-added plane and its data survive a FITS round-trip."""
    rng = np.random.default_rng(500)
    planes = make_mask_planes(rng, 8, n_placeholders=0)
    schema = MaskSchema(planes, dtype=np.uint8)
    bbox = Box.factory[5:50, 6:60]
    mask = Mask(0, schema=schema, bbox=bbox)
    mask = mask.add_plane("OUTSIDE_STENCIL", "Pixel lies outside the stencil.")
    mask.set("OUTSIDE_STENCIL", rng.random(bbox.shape) > 0.5)
    with lsst.utils.tests.getTempFilePath(".fits") as tmpFile:
        mask.write(tmpFile)
        new = Mask.read(tmpFile)
    assert new == mask
    assert new.schema.descriptions["OUTSIDE_STENCIL"] == "Pixel lies outside the stencil."
    assert_masks_equal(new, mask)


def test_legacy_non_cell_coadd_plane_map() -> None:
    """Verify the non-cell coadd map defines a distinct SENSOR_EDGE plane."""
    plane_map = get_legacy_non_cell_coadd_mask_planes()
    assert "SENSOR_EDGE" in plane_map
    assert plane_map["SENSOR_EDGE"].name == "SENSOR_EDGE"


def test_guess_legacy_plane_map_coadd_discriminator() -> None:
    """Verify INEXACT_PSF routes to a coadd map and SENSOR_EDGE discriminates
    non-cell from cell.
    """
    non_cell = _guess_legacy_plane_map({"INEXACT_PSF": 11, "SENSOR_EDGE": 14})
    assert "SENSOR_EDGE" in non_cell
    cell = _guess_legacy_plane_map({"INEXACT_PSF": 11})
    assert "SENSOR_EDGE" not in cell


def test_legacy(legacy_test_data: _LegacyTestData) -> None:
    """Test Mask.read_legacy, Mask.to_legacy, and Mask.from_legacy."""
    assert legacy_test_data.mask.schema.names == {p.name for p in legacy_test_data.plane_map.values()}
    assert legacy_test_data.mask.bbox == Box.from_legacy(legacy_test_data.reader.readBBox())
    legacy_mask = legacy_test_data.reader.readMask()
    compare_mask_to_legacy(legacy_test_data.mask, legacy_mask, legacy_test_data.plane_map)
    compare_mask_to_legacy(
        legacy_test_data.mask,
        legacy_test_data.mask.to_legacy(legacy_test_data.plane_map),
        legacy_test_data.plane_map,
    )
    assert_masks_equal(
        legacy_test_data.mask, Mask.from_legacy(legacy_mask, plane_map=legacy_test_data.plane_map)
    )
    # Write the mask out in the new format, and test that we can read it back.
    with RoundtripFits(legacy_test_data.mask, storage_class="MaskV2") as roundtrip:
        pass
    assert_masks_equal(roundtrip.result, legacy_test_data.mask)


def test_legacy_butler_read(legacy_test_data: _LegacyTestData) -> None:
    """Test that a round-tripped MaskV2 can be read back as a legacy afw
    Mask via Butler.
    """
    with RoundtripFits(legacy_test_data.mask, storage_class="MaskV2") as roundtrip:
        legacy_mask = roundtrip.get(storageClass="Mask")
        assert isinstance(legacy_mask, lsst.afw.image.Mask)
        compare_mask_to_legacy(legacy_test_data.mask, legacy_mask)

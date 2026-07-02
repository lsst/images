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

"""Tests for reconstructing Image/MaskedImage from cut-down HDU lists such as
those written by ``dax_images_cutout``.
"""

from __future__ import annotations

import os
import tempfile
import unittest

import astropy.io.fits
import astropy.units as u
import numpy as np

from lsst.images import Box, GeneralFrame, Image, Mask, MaskedImage, MaskPlane, MaskSchema
from lsst.images import fits as images_fits
from lsst.images.fits import ExtensionKey, FitsOpaqueMetadata
from lsst.images.tests import (
    assert_images_equal,
    assert_masked_images_equal,
    make_random_sky_projection,
)


class ReadOffsetWcsTestCase(unittest.TestCase):
    """Tests for the inverse of `lsst.images.fits.add_offset_wcs`."""

    def test_round_trip(self) -> None:
        header = astropy.io.fits.Header()
        images_fits.add_offset_wcs(header, x=19190, y=22580, key="A")
        self.assertEqual(images_fits.read_offset_wcs(header, key="A"), (19190, 22580))

    def test_absent_returns_none(self) -> None:
        self.assertIsNone(images_fits.read_offset_wcs(astropy.io.fits.Header(), key="A"))

    def test_other_key_ignored(self) -> None:
        header = astropy.io.fits.Header()
        images_fits.add_offset_wcs(header, x=3, y=4, key="A")
        self.assertIsNone(images_fits.read_offset_wcs(header, key="B"))


class ReadYx0TestCase(unittest.TestCase):
    """Tests for recovering yx0 from either offset convention."""

    def test_from_offset_wcs(self) -> None:
        header = astropy.io.fits.Header()
        images_fits.add_offset_wcs(header, x=19190, y=22580)
        yx0 = images_fits.read_yx0(header)
        self.assertEqual((yx0.y, yx0.x), (22580, 19190))

    def test_from_ltv(self) -> None:
        header = astropy.io.fits.Header()
        header["LTV1"] = -19190
        header["LTV2"] = -22580
        yx0 = images_fits.read_yx0(header)
        self.assertEqual((yx0.y, yx0.x), (22580, 19190))

    def test_missing_raises(self) -> None:
        with self.assertRaises(ValueError):
            images_fits.read_yx0(astropy.io.fits.Header())


class FromHduListTestCase(unittest.TestCase):
    """Tests for reconstructing Image/MaskedImage from cut-down HDU lists."""

    def setUp(self) -> None:
        self.maxDiff = None
        self.rng = np.random.default_rng(7)

    def _build_masked_image(self, *, projection: bool = False, planes: int = 3) -> MaskedImage:
        shape = (20, 25)
        yx0 = (5, 8)
        bbox = Box.from_shape(shape, start=yx0)
        proj = make_random_sky_projection(self.rng, GeneralFrame(unit=u.pix), bbox) if projection else None
        image = Image(
            self.rng.normal(100.0, 8.0, shape).astype("float32"), unit=u.nJy, yx0=yx0, sky_projection=proj
        )
        schema = MaskSchema([MaskPlane(f"P{i}", f"description {i}") for i in range(planes)])
        masked_image = MaskedImage(image, mask_schema=schema)
        for i in range(planes):
            masked_image.mask.set(f"P{i}", self.rng.random(shape) > 0.5)
        masked_image.variance.array = self.rng.normal(64.0, 0.5, shape)
        return masked_image

    def _cutdown(self, obj: object, names: list[str], **write_kwargs: object) -> astropy.io.fits.HDUList:
        """Serialize ``obj`` and return an in-memory cut-down HDU list: the
        primary HDU plus uncompressed ImageHDUs for ``names`` (JSON and INDEX
        dropped), mimicking a ``dax_images_cutout`` file.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "x.fits")
            obj.write(path, **write_kwargs)  # type: ignore[attr-defined]
            with astropy.io.fits.open(path) as hdul:
                hdus: list[astropy.io.fits.hdu.base._BaseHDU] = [
                    astropy.io.fits.PrimaryHDU(header=hdul[0].header.copy())
                ]
                for name in names:
                    src = hdul[name]
                    hdus.append(
                        astropy.io.fits.ImageHDU(
                            data=np.asarray(src.data), header=src.header.copy(), name=name
                        )
                    )
                return astropy.io.fits.HDUList(hdus)

    def _legacy_masked_image_hdu_list(
        self,
        planes: dict[str, int],
        set_pixels: dict[str, tuple[int, int]],
        *,
        shape: tuple[int, int] = (6, 7),
        yx0: tuple[int, int] = (5, 8),
    ) -> astropy.io.fits.HDUList:
        """Build an afw-style legacy cut-down HDU list (PRIMARY + IMAGE + MASK
        + VARIANCE) whose MASK HDU carries ``MP_`` cards instead of ``MSKN``,
        mimicking a ``dax_images_cutout`` file made from an afw-written image.

        ``planes`` maps legacy plane name to bit index; ``set_pixels`` maps
        legacy plane name to a single ``(y, x)`` pixel to set for that plane.
        """
        mask_data = np.zeros(shape, dtype=np.int32)
        for name, (y, x) in set_pixels.items():
            mask_data[y, x] |= 1 << planes[name]
        hdus: list[astropy.io.fits.hdu.base._BaseHDU] = [astropy.io.fits.PrimaryHDU()]
        for name, data in [
            ("IMAGE", self.rng.normal(0.0, 1.0, shape).astype("float32")),
            ("MASK", mask_data),
            ("VARIANCE", self.rng.normal(1.0, 0.1, shape).astype("float32")),
        ]:
            hdu = astropy.io.fits.ImageHDU(data=data, name=name)
            hdu.header["LTV1"] = -yx0[1]
            hdu.header["LTV2"] = -yx0[0]
            hdus.append(hdu)
        with images_fits.suppress_fits_card_warnings():
            for name, bit in planes.items():
                hdus[2].header[f"MP_{name}"] = bit
        return astropy.io.fits.HDUList(hdus)

    def test_masked_image_round_trip(self) -> None:
        """A cut-down MaskedImage reconstructs to an equal MaskedImage."""
        masked_image = self._build_masked_image()
        cutdown = self._cutdown(masked_image, ["IMAGE", "MASK", "VARIANCE"])
        result = MaskedImage.from_hdu_list(cutdown)
        assert_masked_images_equal(result, masked_image)

    def test_legacy_non_cell_coadd_from_hdu_list(self) -> None:
        """A cut-down afw-style non-cell coadd (``MP_`` cards, ``SENSOR_EDGE``
        set) is reconstructed, mapping ``SENSOR_EDGE`` to its own plane.
        """
        planes = {"BAD": 0, "DETECTED": 5, "INEXACT_PSF": 11, "SENSOR_EDGE": 14}
        set_pixels = {"DETECTED": (1, 2), "INEXACT_PSF": (3, 4), "SENSOR_EDGE": (5, 6)}
        hdul = self._legacy_masked_image_hdu_list(planes, set_pixels)
        result = MaskedImage.from_hdu_list(hdul)
        self.assertIn("SENSOR_EDGE", result.mask.schema.names)
        self.assertTrue(result.mask.get("SENSOR_EDGE")[5, 6])
        self.assertTrue(result.mask.get("INEXACT_PSF")[3, 4])
        self.assertEqual(result.mask.bbox.y.start, 5)
        self.assertEqual(result.mask.bbox.x.start, 8)

    def test_masked_image_round_trip_with_projection(self) -> None:
        """The sky projection is recovered from the FITS WCS."""
        masked_image = self._build_masked_image(projection=True)
        cutdown = self._cutdown(masked_image, ["IMAGE", "MASK", "VARIANCE"])
        result = MaskedImage.from_hdu_list(cutdown)
        self.assertIsNotNone(result.sky_projection)
        center = (masked_image.bbox.x.size / 2, masked_image.bbox.y.size / 2)
        expected_wcs = masked_image.fits_wcs
        actual_wcs = result.fits_wcs
        assert expected_wcs is not None and actual_wcs is not None
        expected = expected_wcs.pixel_to_world(*center)
        actual = actual_wcs.pixel_to_world(*center)
        self.assertLess(expected.separation(actual).arcsec, 1e-3)

    def test_mask_planes_repacked_across_byte_boundary(self) -> None:
        """Nine planes stored as one on-disk int32 HDU are repacked into the
        two-byte uint8 in-memory layout, preserving every plane.
        """
        masked_image = self._build_masked_image(planes=9)
        self.assertEqual(masked_image.mask.schema.mask_size, 2)
        cutdown = self._cutdown(masked_image, ["IMAGE", "MASK", "VARIANCE"])
        self.assertEqual(np.asarray(cutdown["MASK"].data).dtype.kind, "i")
        result = MaskedImage.from_hdu_list(cutdown)
        self.assertEqual(result.mask.schema.mask_size, 2)
        assert_masked_images_equal(result, masked_image)

    def test_image_from_hdu_list_reads_first_two_hdus(self) -> None:
        """Image.from_hdu_list reads PRIMARY+IMAGE and ignores later HDUs."""
        masked_image = self._build_masked_image()
        # A full four-HDU list: only PRIMARY and IMAGE are consulted.
        result = Image.from_hdu_list(self._cutdown(masked_image, ["IMAGE", "MASK", "VARIANCE"]))
        assert_images_equal(result, masked_image.image)
        # A bare two-HDU list works too.
        two = self._cutdown(masked_image, ["IMAGE"])
        assert_images_equal(Image.from_hdu_list(two), masked_image.image)

    def test_missing_mask_schema_raises(self) -> None:
        """A MASK HDU without MSK* cards is rejected (for now)."""
        masked_image = self._build_masked_image()
        cutdown = self._cutdown(masked_image, ["IMAGE", "MASK", "VARIANCE"])
        for key in [k for k in cutdown["MASK"].header if k.startswith(("MSKN", "MSKM", "MSKD"))]:
            del cutdown["MASK"].header[key]
        with self.assertRaises(ValueError):
            MaskedImage.from_hdu_list(cutdown)

    def test_multiple_mask_hdus_raises(self) -> None:
        """Two MASK HDUs (e.g. EXTVER 1 and 2) are rejected rather than
        silently dropping the mask information from all but the first.
        """
        masked_image = self._build_masked_image()
        cutdown = self._cutdown(masked_image, ["IMAGE", "MASK", "VARIANCE"])
        extra_mask = astropy.io.fits.ImageHDU(
            data=np.asarray(cutdown["MASK"].data), header=cutdown["MASK"].header.copy(), name="MASK"
        )
        extra_mask.header["EXTVER"] = 2
        cutdown.append(extra_mask)
        with self.assertRaises(ValueError):
            MaskedImage.from_hdu_list(cutdown)

    def test_primary_header_preserved(self) -> None:
        """Confusing container cards are dropped; other primary cards survive
        as opaque metadata.
        """
        masked_image = self._build_masked_image()

        def add_card(header: astropy.io.fits.Header) -> None:
            header["MYCARD"] = "hello"

        cutdown = self._cutdown(masked_image, ["IMAGE", "MASK", "VARIANCE"], update_header=add_card)
        result = MaskedImage.from_hdu_list(cutdown)
        assert isinstance(result._opaque_metadata, FitsOpaqueMetadata)
        primary = result._opaque_metadata.headers[ExtensionKey()]
        self.assertEqual(primary["MYCARD"], "hello")
        self.assertNotIn("DATAMODL", primary)
        self.assertNotIn("INDXADDR", primary)
        self.assertNotIn("JSONADDR", primary)


class LegacyMaskBranchTestCase(unittest.TestCase):
    """Characterize the legacy MP_/LTV branch of the generalized mask reader.

    The full ``read_legacy`` paths require ``lsst.afw.image`` and external test
    data, so this exercises the legacy MP_/LTV branch directly with a synthetic
    afw-style HDU instead.
    """

    def test_legacy_mp_ltv_path(self) -> None:
        data = np.zeros((6, 7), dtype=np.int32)
        data[1, 2] = 0b01  # BAD
        data[3, 4] = 0b10  # SAT
        hdu = astropy.io.fits.ImageHDU(data=data, name="MASK")
        hdu.header["LTV1"] = -8
        hdu.header["LTV2"] = -5
        hdu.header["MP_BAD"] = 0
        hdu.header["MP_SAT"] = 1
        plane_map = {"BAD": MaskPlane("BAD", "bad"), "SAT": MaskPlane("SATURATED", "saturated")}
        mask = Mask._read_legacy_hdu(hdu, FitsOpaqueMetadata(), plane_map=plane_map)
        self.assertEqual(mask.bbox.y.start, 5)
        self.assertEqual(mask.bbox.x.start, 8)
        self.assertEqual(set(mask.schema.names), {"BAD", "SATURATED"})
        self.assertTrue(mask.get("BAD")[1, 2])
        self.assertTrue(mask.get("SATURATED")[3, 4])
        self.assertFalse(mask.get("BAD")[3, 4])


class LegacyPlaneRetentionTestCase(unittest.TestCase):
    """The legacy-compatibility reconstruction path keeps the ``MP_``
    mask-plane cards, re-indexed to the reshuffled schema, instead of stripping
    them.
    """

    # A sample set of ``MP_`` bit assignments used purely as a test fixture.
    # This is not an externally stable layout: the bits afw actually writes
    # depend on every Mask created in the process, not just an image's own
    # planes.  ``DETECTED_NEGATIVE`` is not part of the visit-image schema, so
    # it is dropped on reconstruction and every plane after it (``SUSPECT``,
    # ``NO_DATA``) shifts down by one.
    AFW_VISIT_BITS = {
        "BAD": 0,
        "SAT": 1,
        "INTRP": 2,
        "CR": 3,
        "EDGE": 4,
        "DETECTED": 5,
        "DETECTED_NEGATIVE": 6,
        "SUSPECT": 7,
        "NO_DATA": 8,
    }

    def test_read_legacy_strip_false_keeps_cards(self) -> None:
        """``MaskPlane.read_legacy(strip=False)`` reads the plane bits but
        leaves the ``MP_`` cards in the header.
        """
        header = astropy.io.fits.Header()
        header["MP_BAD"] = 0
        header["MP_SAT"] = 1
        planes = MaskPlane.read_legacy(header, strip=False)
        self.assertEqual(planes, {"BAD": 0, "SAT": 1})
        self.assertIn("MP_BAD", header)
        self.assertIn("MP_SAT", header)

    def test_read_legacy_default_strips_cards(self) -> None:
        """The default still strips the ``MP_`` cards (behavior for code that
        declares a new-schema-only mask).
        """
        header = astropy.io.fits.Header()
        header["MP_BAD"] = 0
        header["MP_SAT"] = 1
        planes = MaskPlane.read_legacy(header)
        self.assertEqual(planes, {"BAD": 0, "SAT": 1})
        self.assertNotIn("MP_BAD", header)
        self.assertNotIn("MP_SAT", header)

    def _legacy_mask_hdu(
        self, set_pixels: dict[str, tuple[int, int]], *, shape: tuple[int, int] = (6, 7)
    ) -> astropy.io.fits.ImageHDU:
        """Build a standalone afw-style ``MASK`` HDU."""
        data = np.zeros(shape, dtype=np.int32)
        for name, (y, x) in set_pixels.items():
            data[y, x] |= 1 << self.AFW_VISIT_BITS[name]
        hdu = astropy.io.fits.ImageHDU(data=data, name="MASK")
        hdu.header["LTV1"] = -8
        hdu.header["LTV2"] = -5
        with images_fits.suppress_fits_card_warnings():
            for name, bit in self.AFW_VISIT_BITS.items():
                hdu.header[f"MP_{name}"] = bit
        return hdu

    def _schema_index(self, mask: Mask, name: str) -> int:
        return next(n for n, plane in enumerate(mask.schema) if plane is not None and plane.name == name)

    def test_read_legacy_hdu_reindexes_retained_cards(self) -> None:
        """With ``strip_legacy_planes=False`` the surviving ``MP_`` cards are
        rewritten to the bit positions of the reshuffled schema.
        """
        hdu = self._legacy_mask_hdu({"SUSPECT": (1, 2), "NO_DATA": (3, 4), "BAD": (0, 0)})
        mask = Mask._read_legacy_hdu(hdu, FitsOpaqueMetadata(), strip_legacy_planes=False)
        # The plane pixels are repacked into the canonical visit-image schema.
        self.assertTrue(mask.get("SUSPECT")[1, 2])
        self.assertTrue(mask.get("NO_DATA")[3, 4])
        # SUSPECT/NO_DATA shifted down by one relative to the input layout.
        self.assertEqual(self._schema_index(mask, "SUSPECT"), 6)
        self.assertEqual(self._schema_index(mask, "NO_DATA"), 7)
        # Retained MP_ cards now record the new positions, not the input's.
        self.assertEqual(hdu.header["MP_SUSPECT"], 6)
        self.assertEqual(hdu.header["MP_NO_DATA"], 7)
        # Planes whose positions did not move are unchanged.
        self.assertEqual(hdu.header["MP_BAD"], 0)
        self.assertEqual(hdu.header["MP_DETECTED"], 5)
        # The plane that is not in the new schema is dropped entirely.
        self.assertNotIn("MP_DETECTED_NEGATIVE", hdu.header)

    def test_read_legacy_hdu_default_strips(self) -> None:
        """The default ``_read_legacy_hdu`` behavior still strips ``MP_``."""
        hdu = self._legacy_mask_hdu({"BAD": (0, 0)})
        Mask._read_legacy_hdu(hdu, FitsOpaqueMetadata())
        self.assertFalse([k for k in hdu.header if k.startswith("MP_")])

    def _legacy_full_hdu_list(
        self, set_pixels: dict[str, tuple[int, int]], *, shape: tuple[int, int] = (6, 7)
    ) -> astropy.io.fits.HDUList:
        hdus: list[astropy.io.fits.hdu.base._BaseHDU] = [astropy.io.fits.PrimaryHDU()]
        for name, data in [
            ("IMAGE", np.zeros(shape, dtype=np.float32)),
            ("MASK", None),
            ("VARIANCE", np.ones(shape, dtype=np.float32)),
        ]:
            if name == "MASK":
                hdus.append(self._legacy_mask_hdu(set_pixels, shape=shape))
            else:
                hdu = astropy.io.fits.ImageHDU(data=data, name=name)
                hdu.header["LTV1"] = -8
                hdu.header["LTV2"] = -5
                hdus.append(hdu)
        return astropy.io.fits.HDUList(hdus)

    def test_from_hdu_list_round_trips_reindexed_mp_cards(self) -> None:
        """A reconstructed legacy MaskedImage re-serializes with ``MP_`` cards
        whose bit indices match the written ``MSKN`` layout and select the
        correct pixels, so a legacy reader stays correct.
        """
        hdul = self._legacy_full_hdu_list({"SUSPECT": (1, 2), "NO_DATA": (3, 4), "BAD": (0, 0)})
        masked_image = MaskedImage.from_hdu_list(hdul)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.fits")
            masked_image.write(path)
            with astropy.io.fits.open(path) as out:
                header = out["MASK"].header
                array = np.asarray(out["MASK"].data)
        # New-schema MSKN index for each plane name.
        mskn = {header[k]: int(k.removeprefix("MSKN")) for k in header if k.startswith("MSKN")}
        # Retained MP_ cards agree with the MSKN layout.
        self.assertEqual(header["MP_SUSPECT"], mskn["SUSPECT"])
        self.assertEqual(header["MP_NO_DATA"], mskn["NO_DATA"])
        self.assertEqual(header["MP_SUSPECT"], 6)
        self.assertNotIn("MP_DETECTED_NEGATIVE", header)
        # The MP_ bit index selects the right pixel in the on-disk array.
        self.assertTrue(array[1, 2] & (1 << header["MP_SUSPECT"]))
        self.assertTrue(array[3, 4] & (1 << header["MP_NO_DATA"]))
        self.assertTrue(array[0, 0] & (1 << header["MP_BAD"]))

    def test_normal_read_strips_mp_cards(self) -> None:
        """The normal ``lsst.images`` reader drops any ``MP_`` cards, so they
        cannot drift out of sync with the authoritative ``MSKN`` schema or be
        re-propagated on rewrite.  They are written only for the legacy-cutout
        afw-compatibility scenario.
        """
        hdul = self._legacy_full_hdu_list({"SUSPECT": (1, 2), "NO_DATA": (3, 4), "BAD": (0, 0)})
        masked_image = MaskedImage.from_hdu_list(hdul)
        with tempfile.TemporaryDirectory() as tmp:
            legacy_cutout = os.path.join(tmp, "legacy_cutout.fits")
            masked_image.write(legacy_cutout)
            # The legacy-cutout file carries MP_ cards for afw.
            with astropy.io.fits.open(legacy_cutout) as out:
                self.assertTrue([k for k in out["MASK"].header if k.startswith("MP_")])
            # A normal read + rewrite must not retain them.
            rewritten = os.path.join(tmp, "rewritten.fits")
            MaskedImage.read(legacy_cutout).write(rewritten)
            with astropy.io.fits.open(rewritten) as out:
                self.assertFalse([k for k in out["MASK"].header if k.startswith("MP_")])


if __name__ == "__main__":
    unittest.main()

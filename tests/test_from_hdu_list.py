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

    def test_masked_image_round_trip(self) -> None:
        """A cut-down MaskedImage reconstructs to an equal MaskedImage."""
        masked_image = self._build_masked_image()
        cutdown = self._cutdown(masked_image, ["IMAGE", "MASK", "VARIANCE"])
        result = MaskedImage.from_hdu_list(cutdown)
        assert_masked_images_equal(self, result, masked_image)

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
        assert_masked_images_equal(self, result, masked_image)

    def test_image_from_hdu_list_reads_first_two_hdus(self) -> None:
        """Image.from_hdu_list reads PRIMARY+IMAGE and ignores later HDUs."""
        masked_image = self._build_masked_image()
        # A full four-HDU list: only PRIMARY and IMAGE are consulted.
        result = Image.from_hdu_list(self._cutdown(masked_image, ["IMAGE", "MASK", "VARIANCE"]))
        assert_images_equal(self, result, masked_image.image)
        # A bare two-HDU list works too.
        two = self._cutdown(masked_image, ["IMAGE"])
        assert_images_equal(self, Image.from_hdu_list(two), masked_image.image)

    def test_missing_mask_schema_raises(self) -> None:
        """A MASK HDU without MSK* cards is rejected (for now)."""
        masked_image = self._build_masked_image()
        cutdown = self._cutdown(masked_image, ["IMAGE", "MASK", "VARIANCE"])
        for key in [k for k in cutdown["MASK"].header if k.startswith(("MSKN", "MSKM", "MSKD"))]:
            del cutdown["MASK"].header[key]
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


if __name__ == "__main__":
    unittest.main()

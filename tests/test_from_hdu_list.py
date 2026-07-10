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

from pathlib import Path

import astropy.io.fits
import astropy.units as u
import numpy as np
import pytest

from lsst.images import Box, GeneralFrame, Image, Mask, MaskedImage, MaskPlane, MaskSchema
from lsst.images import fits as images_fits
from lsst.images.fits import ExtensionKey, FitsOpaqueMetadata
from lsst.images.tests import (
    assert_images_equal,
    assert_masked_images_equal,
    make_random_sky_projection,
)


def make_hdu_list(
    tmp_path: Path,
    *,
    projection: bool = False,
    planes: int = 3,
    rng: np.random.Generator | None = None,
) -> tuple[MaskedImage, astropy.io.fits.HDUList]:
    """Return a (MaskedImage, cut-down HDUList) pair for use in tests."""
    if rng is None:
        rng = np.random.default_rng(7)
    shape = (20, 25)
    yx0 = (5, 8)
    bbox = Box.from_shape(shape, start=yx0)
    proj = make_random_sky_projection(rng, GeneralFrame(unit=u.pix), bbox) if projection else None
    image = Image(rng.normal(100.0, 8.0, shape).astype("float32"), unit=u.nJy, yx0=yx0, sky_projection=proj)
    schema = MaskSchema([MaskPlane(f"P{i}", f"description {i}") for i in range(planes)])
    masked_image = MaskedImage(image, mask_schema=schema)
    for i in range(planes):
        masked_image.mask.set(f"P{i}", rng.random(shape) > 0.5)
    masked_image.variance.array = rng.normal(64.0, 0.5, shape)

    path = str(tmp_path / "x.fits")
    masked_image.write(path)
    with astropy.io.fits.open(path) as hdul:
        hdus: list[astropy.io.fits.hdu.base._BaseHDU] = [
            astropy.io.fits.PrimaryHDU(header=hdul[0].header.copy())
        ]
        for name in ["IMAGE", "MASK", "VARIANCE"]:
            src = hdul[name]
            hdus.append(
                astropy.io.fits.ImageHDU(data=np.asarray(src.data), header=src.header.copy(), name=name)
            )
        hdu_list = astropy.io.fits.HDUList(hdus)
    return masked_image, hdu_list


def _cutdown(
    obj: object, names: list[str], tmp_path: Path, **write_kwargs: object
) -> astropy.io.fits.HDUList:
    """Return an in-memory cut-down HDUList with the requested extension
    names.
    """
    path = str(tmp_path / "cutdown.fits")
    obj.write(path, **write_kwargs)  # type: ignore[attr-defined]
    with astropy.io.fits.open(path) as hdul:
        hdus: list[astropy.io.fits.hdu.base._BaseHDU] = [
            astropy.io.fits.PrimaryHDU(header=hdul[0].header.copy())
        ]
        for name in names:
            src = hdul[name]
            hdus.append(
                astropy.io.fits.ImageHDU(data=np.asarray(src.data), header=src.header.copy(), name=name)
            )
        return astropy.io.fits.HDUList(hdus)


def make_legacy_masked_image_hdu_list(
    planes: dict[str, int],
    set_pixels: dict[str, tuple[int, int]],
    *,
    shape: tuple[int, int] = (6, 7),
    yx0: tuple[int, int] = (5, 8),
    rng: np.random.Generator | None = None,
) -> astropy.io.fits.HDUList:
    """Return an afw-style legacy cut-down HDUList with MP_ cards in the MASK
    HDU.
    """
    if rng is None:
        rng = np.random.default_rng(7)
    mask_data = np.zeros(shape, dtype=np.int32)
    for name, (y, x) in set_pixels.items():
        mask_data[y, x] |= 1 << planes[name]
    hdus: list[astropy.io.fits.hdu.base._BaseHDU] = [astropy.io.fits.PrimaryHDU()]
    for name, data in [
        ("IMAGE", rng.normal(0.0, 1.0, shape).astype("float32")),
        ("MASK", mask_data),
        ("VARIANCE", rng.normal(1.0, 0.1, shape).astype("float32")),
    ]:
        hdu = astropy.io.fits.ImageHDU(data=data, name=name)
        hdu.header["LTV1"] = -yx0[1]
        hdu.header["LTV2"] = -yx0[0]
        hdus.append(hdu)
    with images_fits.suppress_fits_card_warnings():
        for name, bit in planes.items():
            hdus[2].header[f"MP_{name}"] = bit
    return astropy.io.fits.HDUList(hdus)


def _schema_index(mask: Mask, name: str) -> int:
    """Return the zero-based plane index of the named plane in the mask
    schema.
    """
    return next(n for n, plane in enumerate(mask.schema) if plane is not None and plane.name == name)


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


def _legacy_mask_hdu(
    set_pixels: dict[str, tuple[int, int]], *, shape: tuple[int, int] = (6, 7)
) -> astropy.io.fits.ImageHDU:
    """Return a standalone afw-style MASK ImageHDU with MP_ cards."""
    data = np.zeros(shape, dtype=np.int32)
    for name, (y, x) in set_pixels.items():
        data[y, x] |= 1 << AFW_VISIT_BITS[name]
    hdu = astropy.io.fits.ImageHDU(data=data, name="MASK")
    hdu.header["LTV1"] = -8
    hdu.header["LTV2"] = -5
    with images_fits.suppress_fits_card_warnings():
        for name, bit in AFW_VISIT_BITS.items():
            hdu.header[f"MP_{name}"] = bit
    return hdu


def _legacy_full_hdu_list(
    set_pixels: dict[str, tuple[int, int]], *, shape: tuple[int, int] = (6, 7)
) -> astropy.io.fits.HDUList:
    """Return an afw-style full HDUList (PRIMARY+IMAGE+MASK+VARIANCE) with MP_
    cards.
    """
    hdus: list[astropy.io.fits.hdu.base._BaseHDU] = [astropy.io.fits.PrimaryHDU()]
    for name, data in [
        ("IMAGE", np.zeros(shape, dtype=np.float32)),
        ("MASK", None),
        ("VARIANCE", np.ones(shape, dtype=np.float32)),
    ]:
        if name == "MASK":
            hdus.append(_legacy_mask_hdu(set_pixels, shape=shape))
        else:
            hdu = astropy.io.fits.ImageHDU(data=data, name=name)
            hdu.header["LTV1"] = -8
            hdu.header["LTV2"] = -5
            hdus.append(hdu)
    return astropy.io.fits.HDUList(hdus)


def test_offset_wcs_round_trip() -> None:
    """Verify add_offset_wcs and read_offset_wcs are inverses of each other."""
    header = astropy.io.fits.Header()
    images_fits.add_offset_wcs(header, x=19190, y=22580, key="A")
    assert images_fits.read_offset_wcs(header, key="A") == (19190, 22580)


def test_offset_wcs_absent_returns_none() -> None:
    """Verify read_offset_wcs returns None when no offset WCS key is
    present.
    """
    assert images_fits.read_offset_wcs(astropy.io.fits.Header(), key="A") is None


def test_offset_wcs_other_key_ignored() -> None:
    """Verify read_offset_wcs ignores WCS keys written under a different
    letter.
    """
    header = astropy.io.fits.Header()
    images_fits.add_offset_wcs(header, x=3, y=4, key="A")
    assert images_fits.read_offset_wcs(header, key="B") is None


def test_read_yx0_from_offset_wcs() -> None:
    """Verify read_yx0 recovers yx0 from an offset WCS header."""
    header = astropy.io.fits.Header()
    images_fits.add_offset_wcs(header, x=19190, y=22580)
    yx0 = images_fits.read_yx0(header)
    assert (yx0.y, yx0.x) == (22580, 19190)


def test_read_yx0_from_ltv() -> None:
    """Verify read_yx0 recovers yx0 from LTV1/LTV2 header cards."""
    header = astropy.io.fits.Header()
    header["LTV1"] = -19190
    header["LTV2"] = -22580
    yx0 = images_fits.read_yx0(header)
    assert (yx0.y, yx0.x) == (22580, 19190)


def test_read_yx0_missing_raises() -> None:
    """Verify read_yx0 raises ValueError when no offset information is
    present.
    """
    with pytest.raises(ValueError):
        images_fits.read_yx0(astropy.io.fits.Header())


def test_masked_image_round_trip(tmp_path: Path) -> None:
    """Verify a cut-down MaskedImage reconstructs to an equal MaskedImage."""
    masked_image, cutdown = make_hdu_list(tmp_path)
    result = MaskedImage.from_hdu_list(cutdown)
    assert_masked_images_equal(result, masked_image)


def test_legacy_non_cell_coadd_from_hdu_list() -> None:
    """Verify a legacy afw-style cut-down HDUList reconstructs a MaskedImage
    with MP_ planes.
    """
    planes = {"BAD": 0, "DETECTED": 5, "INEXACT_PSF": 11, "SENSOR_EDGE": 14}
    set_pixels = {"DETECTED": (1, 2), "INEXACT_PSF": (3, 4), "SENSOR_EDGE": (5, 6)}
    hdul = make_legacy_masked_image_hdu_list(planes, set_pixels)
    result = MaskedImage.from_hdu_list(hdul)
    assert "SENSOR_EDGE" in result.mask.schema.names
    assert result.mask.get("SENSOR_EDGE")[5, 6]
    assert result.mask.get("INEXACT_PSF")[3, 4]
    assert result.mask.bbox.y.start == 5
    assert result.mask.bbox.x.start == 8


def test_masked_image_round_trip_with_projection(tmp_path: Path) -> None:
    """Verify the sky projection is recovered from the FITS WCS in the cut-down
    HDUList.
    """
    masked_image, cutdown = make_hdu_list(tmp_path, projection=True)
    result = MaskedImage.from_hdu_list(cutdown)
    assert result.sky_projection is not None
    center = (masked_image.bbox.x.size / 2, masked_image.bbox.y.size / 2)
    expected_wcs = masked_image.fits_wcs
    actual_wcs = result.fits_wcs
    assert expected_wcs is not None and actual_wcs is not None
    expected = expected_wcs.pixel_to_world(*center)
    actual = actual_wcs.pixel_to_world(*center)
    assert expected.separation(actual).arcsec < 1e-3


def test_mask_planes_repacked_across_byte_boundary(tmp_path: Path) -> None:
    """Verify nine planes stored in an int32 HDU are repacked into the two-byte
    uint8 layout.
    """
    rng = np.random.default_rng(7)
    masked_image, _ = make_hdu_list(tmp_path, planes=9, rng=rng)
    assert masked_image.mask.schema.mask_size == 2
    cutdown = _cutdown(masked_image, ["IMAGE", "MASK", "VARIANCE"], tmp_path)
    assert np.asarray(cutdown["MASK"].data).dtype.kind == "i"
    result = MaskedImage.from_hdu_list(cutdown)
    assert result.mask.schema.mask_size == 2
    assert_masked_images_equal(result, masked_image)


def test_image_from_hdu_list_reads_first_two_hdus(tmp_path: Path) -> None:
    """Verify Image.from_hdu_list reads PRIMARY+IMAGE and ignores later
    HDUs.
    """
    masked_image, cutdown = make_hdu_list(tmp_path)
    result = Image.from_hdu_list(cutdown)
    assert_images_equal(result, masked_image.image)
    two = _cutdown(masked_image, ["IMAGE"], tmp_path)
    assert_images_equal(Image.from_hdu_list(two), masked_image.image)


def test_missing_mask_schema_raises(tmp_path: Path) -> None:
    """Verify a MASK HDU without MSK* cards raises ValueError."""
    _, cutdown = make_hdu_list(tmp_path)
    for key in [k for k in cutdown["MASK"].header if k.startswith(("MSKN", "MSKM", "MSKD"))]:
        del cutdown["MASK"].header[key]
    with pytest.raises(ValueError):
        MaskedImage.from_hdu_list(cutdown)


def test_multiple_mask_hdus_raises(tmp_path: Path) -> None:
    """Verify two MASK HDUs raise ValueError rather than silently dropping
    one.
    """
    _, cutdown = make_hdu_list(tmp_path)
    extra_mask = astropy.io.fits.ImageHDU(
        data=np.asarray(cutdown["MASK"].data), header=cutdown["MASK"].header.copy(), name="MASK"
    )
    extra_mask.header["EXTVER"] = 2
    cutdown.append(extra_mask)
    with pytest.raises(ValueError):
        MaskedImage.from_hdu_list(cutdown)


def test_primary_header_preserved(tmp_path: Path) -> None:
    """Verify confusing container cards are dropped and other primary cards
    survive as opaque metadata.
    """
    masked_image, _ = make_hdu_list(tmp_path)

    def add_card(header: astropy.io.fits.Header) -> None:
        header["MYCARD"] = "hello"

    cutdown = _cutdown(masked_image, ["IMAGE", "MASK", "VARIANCE"], tmp_path, update_header=add_card)
    result = MaskedImage.from_hdu_list(cutdown)
    assert isinstance(result._opaque_metadata, FitsOpaqueMetadata)
    primary = result._opaque_metadata.headers[ExtensionKey()]
    assert primary["MYCARD"] == "hello"
    assert "DATAMODL" not in primary
    assert "INDXADDR" not in primary
    assert "JSONADDR" not in primary


def test_legacy_mp_ltv_path() -> None:
    """Verify the legacy MP_/LTV branch reads a synthetic afw-style MASK HDU
    correctly.
    """
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
    assert mask.bbox.y.start == 5
    assert mask.bbox.x.start == 8
    assert set(mask.schema.names) == {"BAD", "SATURATED"}
    assert mask.get("BAD")[1, 2]
    assert mask.get("SATURATED")[3, 4]
    assert not mask.get("BAD")[3, 4]


def test_read_legacy_strip_false_keeps_cards() -> None:
    """Verify MaskPlane.read_legacy with strip=False reads planes but leaves
    MP_ cards in the header.
    """
    header = astropy.io.fits.Header()
    header["MP_BAD"] = 0
    header["MP_SAT"] = 1
    planes = MaskPlane.read_legacy(header, strip=False)
    assert planes == {"BAD": 0, "SAT": 1}
    assert "MP_BAD" in header
    assert "MP_SAT" in header


def test_read_legacy_default_strips_cards() -> None:
    """Verify MaskPlane.read_legacy default strips MP_ cards from the
    header.
    """
    header = astropy.io.fits.Header()
    header["MP_BAD"] = 0
    header["MP_SAT"] = 1
    planes = MaskPlane.read_legacy(header)
    assert planes == {"BAD": 0, "SAT": 1}
    assert "MP_BAD" not in header
    assert "MP_SAT" not in header


def test_read_legacy_hdu_reindexes_retained_cards() -> None:
    """Verify _read_legacy_hdu with strip_legacy_planes=False rewrites MP_
    cards to new bit positions.
    """
    hdu = _legacy_mask_hdu({"SUSPECT": (1, 2), "NO_DATA": (3, 4), "BAD": (0, 0)})
    mask = Mask._read_legacy_hdu(hdu, FitsOpaqueMetadata(), strip_legacy_planes=False)
    assert mask.get("SUSPECT")[1, 2]
    assert mask.get("NO_DATA")[3, 4]
    assert _schema_index(mask, "SUSPECT") == 6
    assert _schema_index(mask, "NO_DATA") == 7
    assert hdu.header["MP_SUSPECT"] == 6
    assert hdu.header["MP_NO_DATA"] == 7
    assert hdu.header["MP_BAD"] == 0
    assert hdu.header["MP_DETECTED"] == 5
    assert "MP_DETECTED_NEGATIVE" not in hdu.header


def test_read_legacy_hdu_default_strips() -> None:
    """Verify _read_legacy_hdu default behavior strips all MP_ cards."""
    hdu = _legacy_mask_hdu({"BAD": (0, 0)})
    Mask._read_legacy_hdu(hdu, FitsOpaqueMetadata())
    assert not [k for k in hdu.header if k.startswith("MP_")]


def test_from_hdu_list_round_trips_reindexed_mp_cards(tmp_path: Path) -> None:
    """Verify a reconstructed legacy MaskedImage re-serializes with correct MP_
    bit indices.
    """
    hdul = _legacy_full_hdu_list({"SUSPECT": (1, 2), "NO_DATA": (3, 4), "BAD": (0, 0)})
    masked_image = MaskedImage.from_hdu_list(hdul)
    path = str(tmp_path / "out.fits")
    masked_image.write(path)
    with astropy.io.fits.open(path) as out:
        header = out["MASK"].header
        array = np.asarray(out["MASK"].data)
    mskn = {header[k]: int(k.removeprefix("MSKN")) for k in header if k.startswith("MSKN")}
    assert header["MP_SUSPECT"] == mskn["SUSPECT"]
    assert header["MP_NO_DATA"] == mskn["NO_DATA"]
    assert header["MP_SUSPECT"] == 6
    assert "MP_DETECTED_NEGATIVE" not in header
    assert array[1, 2] & (1 << header["MP_SUSPECT"])
    assert array[3, 4] & (1 << header["MP_NO_DATA"])
    assert array[0, 0] & (1 << header["MP_BAD"])


def test_normal_read_strips_mp_cards(tmp_path: Path) -> None:
    """Verify a normal read + rewrite of a legacy-cutout file drops the MP_
    cards.
    """
    hdul = _legacy_full_hdu_list({"SUSPECT": (1, 2), "NO_DATA": (3, 4), "BAD": (0, 0)})
    masked_image = MaskedImage.from_hdu_list(hdul)
    legacy_cutout = str(tmp_path / "legacy_cutout.fits")
    masked_image.write(legacy_cutout)
    with astropy.io.fits.open(legacy_cutout) as out:
        assert [k for k in out["MASK"].header if k.startswith("MP_")]
    rewritten = str(tmp_path / "rewritten.fits")
    MaskedImage.read(legacy_cutout).write(rewritten)
    with astropy.io.fits.open(rewritten) as out:
        assert not [k for k in out["MASK"].header if k.startswith("MP_")]

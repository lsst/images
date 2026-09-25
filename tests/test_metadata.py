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

import copy
import warnings
from collections.abc import Mapping
from pathlib import Path

import astropy.io.fits
import numpy as np
import pytest

from lsst.images import Box, Image, Mask, MaskedImage, MaskPlane, MaskSchema, MetadataView, NativeMetadata
from lsst.images.fits import ExtensionKey, FitsExternalMetadata, FitsOpaqueMetadata
from lsst.images.serialization import EmptyExternalMetadata
from lsst.images.tests import RoundtripFits, RoundtripNdf, reset_afw_mask_planes  # noqa: F401

try:
    import h5py  # noqa: F401

    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False

EXTERNAL_KEYS = ["EXPTIME", "BGMEAN", "LSST ISR UNITS", "LOWER KEY", "NOVAL", "CPLX"]


def _make_header() -> astropy.io.fits.Header:
    """Return a primary header exercising repeated, HIERARCH, commentary,
    valueless, and complex cards.
    """
    header = astropy.io.fits.Header()
    header.append(("EXPTIME", 30.0), end=True)
    header.append(("BGMEAN", 1.5), end=True)
    header.append(("BGMEAN", 2.5), end=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", astropy.io.fits.verify.VerifyWarning)
        header.append(("HIERARCH LSST ISR UNITS", "adu"), end=True)
        # astropy preserves the case of HIERARCH keywords.
        header.append(("HIERARCH lower key", 3), end=True)
    header.append(("COMMENT", "a comment"), end=True)
    header.append(("HISTORY", "a history"), end=True)
    header.append(("", "blank"), end=True)
    header.append(astropy.io.fits.Card.fromstring("NOVAL   =".ljust(80)), end=True)
    header.append(("CPLX", 1 + 2j), end=True)
    return header


def test_empty_external_metadata() -> None:
    """Test that EmptyExternalMetadata behaves as an empty read-only
    mapping.
    """
    external = EmptyExternalMetadata()
    assert isinstance(external, Mapping)
    assert len(external) == 0
    assert list(external) == []
    assert "EXPTIME" not in external
    assert external.get("EXPTIME") is None
    with pytest.raises(KeyError):
        external["EXPTIME"]
    with pytest.raises(KeyError):
        external.get_all("EXPTIME")
    assert repr(external) == "EmptyExternalMetadata({})"


def test_fits_external_metadata_lookup() -> None:
    """Test case-insensitive lookup, repeated keywords, HIERARCH keywords,
    and value conversion.
    """
    external = FitsExternalMetadata(_make_header())
    assert external["EXPTIME"] == 30.0
    assert external["exptime"] == 30.0
    assert external["lsst isr units"] == "adu"
    assert external["HIERARCH LSST ISR UNITS"] == "adu"
    assert external["LOWER KEY"] == 3
    assert external["lower key"] == 3
    assert external["NOVAL"] is None
    assert external["CPLX"] == 1 + 2j
    assert external["BGMEAN"] in (1.5, 2.5)
    assert external.get_all("bgmean") == (1.5, 2.5)
    assert external.get_all("exptime") == (30.0,)
    assert external.get_all("NOVAL") == (None,)
    assert external.get_all("Lower Key") == (3,)
    assert "Exptime" in external
    assert "lower key" in external


def test_fits_external_metadata_hides_commentary_cards() -> None:
    """Test that COMMENT, HISTORY, and blank cards are not visible."""
    external = FitsExternalMetadata(_make_header())
    for key in ("COMMENT", "comment", "HISTORY", ""):
        assert key not in external
        with pytest.raises(KeyError):
            external[key]
        with pytest.raises(KeyError):
            external.get_all(key)


def test_fits_external_metadata_odd_keys() -> None:
    """Test that keys that cannot be FITS keywords are simply absent."""
    external = FitsExternalMetadata(_make_header())
    for key in ("roundtrip_test_1", "a=b", "é", "x" * 100):
        assert key not in external
        with pytest.raises(KeyError):
            external[key]
    assert 1 not in external


def test_fits_external_metadata_iteration() -> None:
    """Test that iteration yields each upper-case keyword once."""
    external = FitsExternalMetadata(_make_header())
    assert list(external) == EXTERNAL_KEYS
    assert len(external) == len(EXTERNAL_KEYS)
    assert repr(external).startswith("FitsExternalMetadata({'EXPTIME': 30.0")


def test_fits_external_metadata_lowercase_dotted_hierarch() -> None:
    """Test that a lower-case HIERARCH keyword that looks like a
    record-valued keyword is reported and found consistently.
    """
    header = astropy.io.fits.Header.fromstring("HIERARCH x.y = 3".ljust(80))
    header.append(("EXPTIME", 30.0), end=True)
    external = FitsExternalMetadata(header)
    assert list(external) == ["X.Y", "EXPTIME"]
    assert external["X.Y"] == 3
    assert external["x.y"] == 3
    assert external.get_all("X.Y") == (3,)
    assert dict(external) == {"X.Y": 3, "EXPTIME": 30.0}


def test_fits_opaque_metadata_external_metadata_tracks_primary_header() -> None:
    """Test that external metadata reflects a primary header added after it
    was first requested.
    """
    opaque_metadata = FitsOpaqueMetadata()
    assert len(opaque_metadata.external_metadata()) == 0
    opaque_metadata.add_header(_make_header(), name="", ver=1)
    assert list(opaque_metadata.external_metadata()) == EXTERNAL_KEYS


def test_fits_external_metadata_empty() -> None:
    """Test that a missing header behaves as an empty one."""
    external = FitsExternalMetadata(None)
    assert len(external) == 0
    assert "EXPTIME" not in external


def test_fits_opaque_metadata_external_metadata() -> None:
    """Test that only the primary header is exposed."""
    opaque_metadata = FitsOpaqueMetadata()
    assert len(opaque_metadata.external_metadata()) == 0
    opaque_metadata.add_header(_make_header(), name="", ver=1)
    extension_header = astropy.io.fits.Header()
    extension_header["EXTRA"] = 1
    opaque_metadata.add_header(extension_header, name="IMAGE", ver=1)
    external = opaque_metadata.external_metadata()
    assert list(external) == EXTERNAL_KEYS
    assert "EXTRA" not in external
    assert opaque_metadata.headers[ExtensionKey("IMAGE")]["EXTRA"] == 1


def test_extract_legacy_primary_header_strips_native_cards() -> None:
    """Test that the cards holding native metadata in a legacy file are
    returned as native metadata and not kept in the opaque header.
    """
    header = astropy.io.fits.Header()
    header.append(("PLATFORM", "lsstcam"), end=True)
    header.append(("HIERARCH LSST IMAGES KEY 1", "native_key"), end=True)
    header.append(("HIERARCH LSST IMAGES VALUE 1", 7), end=True)
    header.append(("HIERARCH LSST IMAGES KEY 2", "MixedCase"), end=True)
    header.append(("HIERARCH LSST IMAGES VALUE 2", "yes"), end=True)
    opaque_metadata = FitsOpaqueMetadata()
    assert opaque_metadata.extract_legacy_primary_header(header) == {"native_key": 7, "MixedCase": "yes"}
    stored = opaque_metadata.headers[ExtensionKey()]
    assert stored["PLATFORM"] == "lsstcam"
    assert not [keyword for keyword in stored if keyword.startswith("LSST IMAGES")]
    assert list(opaque_metadata.external_metadata()) == ["PLATFORM"]


def test_legacy_readers_restore_native_metadata(
    tmp_path: Path,
    reset_afw_mask_planes: None,  # noqa: F811
) -> None:
    """Test that every legacy reader restores native metadata from the
    ``LSST IMAGES`` cards and does not keep those cards as opaque metadata.
    """
    from lsst.daf.base import PropertyList

    masked_image = MaskedImage(
        Image(1.0, shape=(4, 5), dtype=np.float32),
        mask_schema=MaskSchema([MaskPlane("BAD", "Pixel is bad.")]),
        metadata={"native_key": 7, "MixedCase": "yes"},
    )
    legacy_metadata = PropertyList()
    masked_image._fill_legacy_metadata(legacy_metadata)
    path = tmp_path / "legacy_masked_image.fits"
    masked_image.to_legacy().writeFits(str(path), metadata=legacy_metadata)
    with astropy.io.fits.open(path) as hdu_list:
        assert hdu_list[0].header["LSST IMAGES KEY 1"] == "native_key"
    results = {
        "MaskedImage": MaskedImage.read_legacy(path),
        "MaskedImage image component": MaskedImage.read_legacy(path, component="image"),
        "MaskedImage mask component": MaskedImage.read_legacy(path, component="mask"),
        "Image": Image.read_legacy(path),
        "Mask": Mask.read_legacy(path, ext=2),
    }
    for label, result in results.items():
        assert result._metadata == {"native_key": 7, "MixedCase": "yes"}, label
        opaque_header = result._opaque_metadata.headers[ExtensionKey()]
        assert not [keyword for keyword in opaque_header if keyword.startswith("LSST IMAGES")], label


def _make_view(data: dict) -> MetadataView:
    external = FitsExternalMetadata(_make_header())
    return MetadataView(NativeMetadata(data, external), external)


def test_native_metadata_shadowing() -> None:
    """Test that new native keys may not shadow external keys, while keys
    already present may be updated.
    """
    external = FitsExternalMetadata(_make_header())
    data: dict = {"exptime": 1.0}
    native = NativeMetadata(data, external)
    native["exptime"] = 2.0
    assert data == {"exptime": 2.0}
    for key in ("EXPTIME", "Bgmean", "lsst isr units", "LOWER KEY", "noval"):
        with pytest.raises(KeyError):
            native[key] = 3.0
    with pytest.raises(KeyError):
        native.update({"cplx": 1})
    with pytest.raises(KeyError):
        native.setdefault("NOVAL", 1)
    native["fresh"] = 1
    del native["fresh"]
    assert data == {"exptime": 2.0}
    assert dict(native) == {"exptime": 2.0}
    assert len(native) == 1


def test_native_metadata_accepts_non_fits_keys() -> None:
    """Test that keys that cannot be FITS keywords are accepted."""
    external = FitsExternalMetadata(_make_header())
    data: dict = {}
    native = NativeMetadata(data, external)
    for key in ("roundtrip_test_1", "a=b", "é", "MixedCaseKey"):
        native[key] = 1
    assert set(data) == {"roundtrip_test_1", "a=b", "é", "MixedCaseKey"}


def test_metadata_view_lookup() -> None:
    """Test that native keys are found first with exact case, then external
    keys case-insensitively.
    """
    view = _make_view({"native_key": 7, "exptime": 1.0})
    assert view["native_key"] == 7
    assert view["exptime"] == 1.0
    assert view["EXPTIME"] == 30.0
    assert view["ExpTime"] == 30.0
    assert view["BGMEAN"] in (1.5, 2.5)
    assert view["noval"] is None
    assert "lsst isr units" in view
    assert view.get("missing") is None
    assert view.get_all("BGMEAN") == (1.5, 2.5)
    assert view.get_all("exptime") == (1.0,)
    assert view.get_all("EXPTIME") == (30.0,)
    for key in ("COMMENT", "HISTORY", ""):
        assert key not in view
    with pytest.raises(KeyError):
        view["missing"]
    with pytest.raises(KeyError):
        view.get_all("missing")


def test_metadata_view_iteration() -> None:
    """Test that iteration is the exact-case union of both sources and
    agrees with lookup.
    """
    view = _make_view({"native_key": 7, "exptime": 1.0})
    keys = list(view)
    assert sorted(keys) == sorted(["native_key", "exptime", *EXTERNAL_KEYS])
    assert len(view) == len(keys) == len(set(keys))
    as_dict = dict(view)
    assert as_dict["exptime"] == 1.0
    assert as_dict["EXPTIME"] == 30.0
    for key in keys:
        assert as_dict[key] == view[key]
    assert view == as_dict


def test_metadata_view_writes() -> None:
    """Test that writes and deletes go to native and respect shadowing."""
    data: dict = {"native_key": 7}
    view = _make_view(data)
    view["new"] = 1
    assert data == {"native_key": 7, "new": 1}
    with pytest.raises(KeyError):
        view["cplx"] = 1
    with pytest.raises(KeyError):
        del view["EXPTIME"]
    with pytest.raises(KeyError):
        view.pop("EXPTIME")
    assert view.pop("missing", None) is None
    assert view.pop("new") == 1
    view |= {"another": 2}
    assert data == {"native_key": 7, "another": 2}
    view.clear()
    assert data == {}
    assert view["EXPTIME"] == 30.0


def test_metadata_view_chainmap_operations() -> None:
    """Test the ChainMap operations that the view overrides."""
    data: dict = {"native_key": 7}
    view = _make_view(data)
    copied = view.copy()
    assert type(copied) is dict
    assert copied == data
    assert copied is not data
    shallow = copy.copy(view)
    assert type(shallow) is dict
    assert shallow == data
    merged = view | {"z": 1}
    assert type(merged) is dict
    assert merged == {**dict(view), "z": 1}
    reverse_merged = {"z": 1} | view
    assert type(reverse_merged) is dict
    assert reverse_merged == {"z": 1, **dict(view)}
    with pytest.raises(TypeError):
        view.new_child()
    with pytest.raises(TypeError):
        view.parents
    assert view.native is not None
    assert list(view.external) == EXTERNAL_KEYS


def _make_image(metadata: dict | None = None) -> Image:
    """Return a small image whose opaque metadata holds `_make_header`."""
    image = Image(0.0, shape=(4, 5), dtype=np.float32, metadata=metadata)
    opaque_metadata = FitsOpaqueMetadata()
    opaque_metadata.add_header(_make_header(), name="", ver=1)
    image._opaque_metadata = opaque_metadata
    return image


def test_image_metadata_view() -> None:
    """Test the metadata view on an image with external metadata."""
    image = _make_image({"native_key": 7})
    assert isinstance(image.metadata, MetadataView)
    assert image.metadata["native_key"] == 7
    assert image.metadata["exptime"] == 30.0
    assert image.metadata.native == {"native_key": 7}
    assert list(image.metadata.external) == EXTERNAL_KEYS
    image.metadata["other"] = "x"
    assert image.metadata.native == {"native_key": 7, "other": "x"}
    with pytest.raises(KeyError):
        image.metadata["ExpTime"] = 1.0
    with pytest.raises(KeyError):
        image.metadata.native["ExpTime"] = 1.0
    with pytest.raises(KeyError):
        image.metadata.update({"bgmean": 1.0})


def test_image_metadata_without_opaque_metadata() -> None:
    """Test that an in-memory image has empty external metadata."""
    image = Image(0.0, shape=(4, 5), dtype=np.float32, metadata={"a": 1})
    assert len(image.metadata.external) == 0
    assert image.metadata == {"a": 1}
    image.metadata["EXPTIME"] = 2.0
    assert image.metadata.native == {"a": 1, "EXPTIME": 2.0}


def test_image_metadata_view_reflects_later_opaque_metadata() -> None:
    """Test that the view sees opaque metadata attached after
    construction.
    """
    image = Image(0.0, shape=(4, 5), dtype=np.float32, metadata={"exptime": 1.0})
    opaque_metadata = FitsOpaqueMetadata()
    opaque_metadata.add_header(_make_header(), name="", ver=1)
    image._opaque_metadata = opaque_metadata
    assert image.metadata["exptime"] == 1.0
    assert image.metadata["EXPTIME"] == 30.0
    image.metadata["exptime"] = 2.0
    assert image.metadata.native == {"exptime": 2.0}


def test_subimage_metadata() -> None:
    """Test that subimages share native metadata and see the same external
    metadata.
    """
    image = _make_image({"native_key": 7})
    subimage = image[Box.factory[0:2, 0:3]]
    subimage.metadata["new"] = 1
    assert image.metadata["new"] == 1
    assert subimage.metadata["EXPTIME"] == 30.0
    copied = image.copy()
    copied.metadata["copied_only"] = 1
    assert "copied_only" not in image.metadata
    assert copied.metadata["EXPTIME"] == 30.0


def test_constructor_with_metadata_view() -> None:
    """Test that a metadata view passed to a constructor shares the native
    dict rather than being nested.
    """
    source = _make_image({"a": 1})
    # A view is accepted at runtime even though the annotation asks for a
    # dict.
    image = Image(0.0, shape=(4, 5), dtype=np.float32, metadata=source.metadata)  # type: ignore[arg-type]
    assert type(image._metadata) is dict
    assert image._metadata is source._metadata
    assert image.metadata.native == {"a": 1}
    assert "EXPTIME" not in image.metadata


def test_metadata_setter() -> None:
    """Test that the setter replaces native metadata without a shadowing
    check and shares the assigned dict.
    """
    image = _make_image()
    image.metadata = {"exptime": 1.0}
    assert image.metadata["exptime"] == 1.0
    assert image.metadata["EXPTIME"] == 30.0
    image.metadata["exptime"] = 2.0
    shared: dict = {"s": 1}
    image.metadata = shared
    shared["t"] = 2
    assert image.metadata["t"] == 2
    other = Image(0.0, shape=(4, 5), dtype=np.float32)
    other.metadata = image.metadata
    other.metadata["u"] = 3
    assert image.metadata["u"] == 3
    assert other.metadata.native == {"s": 1, "t": 2, "u": 3}
    image.metadata = image.metadata.copy()
    image.metadata["v"] = 4
    assert "v" not in other.metadata


def _make_masked_image() -> MaskedImage:
    """Return a small masked image with native metadata and opaque metadata
    holding `_make_header`.
    """
    masked_image = MaskedImage(
        Image(1.0, shape=(4, 5), dtype=np.float32),
        mask_schema=MaskSchema([MaskPlane("BAD", "Pixel is bad.")]),
        metadata={"native_key": 7, "MixedCase": "yes"},
    )
    opaque_metadata = FitsOpaqueMetadata()
    opaque_metadata.add_header(_make_header(), name="", ver=1)
    masked_image._opaque_metadata = opaque_metadata
    return masked_image


def _check_round_trip(result: MaskedImage) -> None:
    assert result.metadata.native == {"native_key": 7, "MixedCase": "yes"}
    assert list(result.metadata.external) == EXTERNAL_KEYS
    assert result.metadata["exptime"] == 30.0
    assert result.metadata.get_all("BGMEAN") == (1.5, 2.5)
    assert result.metadata["NOVAL"] is None
    assert result.metadata["CPLX"] == 1 + 2j
    assert result.metadata["lower key"] == 3
    assert "COMMENT" not in result.metadata


def test_external_metadata_fits_round_trip() -> None:
    """Test that native and external metadata survive a FITS round trip."""
    with RoundtripFits(_make_masked_image()) as roundtrip:
        pass
    _check_round_trip(roundtrip.result)


@pytest.mark.skipif(not HAVE_H5PY, reason="h5py is not installed")
def test_external_metadata_ndf_round_trip() -> None:
    """Test that native and external metadata survive an NDF round trip."""
    with RoundtripNdf(_make_masked_image()) as roundtrip:
        pass
    _check_round_trip(roundtrip.result)

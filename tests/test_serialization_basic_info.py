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
from pathlib import Path

import numpy as np
import pydantic
import pytest

from lsst.images import Box, Image
from lsst.images import fits as images_fits
from lsst.images import json as images_json
from lsst.images.fits import FitsInputArchive
from lsst.images.json import JsonInputArchive
from lsst.images.serialization import ArchiveInfo, InputArchive

try:
    import h5py

    from lsst.images import ndf as images_ndf
    from lsst.images.ndf import NdfInputArchive

    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False

skip_no_h5py = pytest.mark.skipif(not HAVE_H5PY, reason="h5py is not installed")


def _make_image() -> Image:
    """Return a small float32 Image for serialization tests."""
    return Image(np.zeros((4, 4), dtype=np.float32), bbox=Box.factory[0:4, 0:4])


def test_archive_info_from_schema_url() -> None:
    """Verify ArchiveInfo.from_schema_url parses schema name, version, and
    format correctly.
    """
    info = ArchiveInfo.from_schema_url("https://images.lsst.io/schemas/visit_image-1.2.3", format_version=1)
    assert info.schema_name == "visit_image"
    assert info.schema_version == "1.2.3"
    assert info.schema_url == "https://images.lsst.io/schemas/visit_image-1.2.3"
    assert info.format_version == 1


def test_archive_info_from_schema_url_none_format() -> None:
    """Verify ArchiveInfo.from_schema_url accepts None for format_version."""
    info = ArchiveInfo.from_schema_url("https://images.lsst.io/schemas/image-1.0.0", format_version=None)
    assert info.schema_name == "image"
    assert info.format_version is None


def test_archive_info_frozen() -> None:
    """Verify ArchiveInfo is frozen (schema_name cannot be reassigned)."""
    info = ArchiveInfo.from_schema_url("https://images.lsst.io/schemas/image-1.0.0", format_version=None)
    with pytest.raises(pydantic.ValidationError):
        info.schema_name = "other"  # type: ignore[misc]


def test_archive_info_from_schema_url_invalid() -> None:
    """Verify ArchiveInfo.from_schema_url raises ValueError for URLs without
    a version.
    """
    with pytest.raises(ValueError):
        ArchiveInfo.from_schema_url("https://images.lsst.io/schemas/noversion", format_version=None)


def test_archive_info_from_schema_url_foreign_host() -> None:
    """Verify ArchiveInfo.from_schema_url accepts third-party hosts whose
    URLs follow the ``.../schemas/{name}-{version}`` shape, since external
    packages mint schema URLs under their own documentation sites.
    """
    info = ArchiveInfo.from_schema_url(
        "https://example.org/products/schemas/extended_psf-1.2.0", format_version=None
    )
    assert info.schema_name == "extended_psf"
    assert info.schema_version == "1.2.0"


def test_archive_info_from_schema_url_not_schema_shaped() -> None:
    """Verify ArchiveInfo.from_schema_url rejects values that do not have
    the schema URL shape, such as a DATAMODL header written by an unrelated
    tool.
    """
    with pytest.raises(ValueError):
        # No "schemas" parent path segment.
        ArchiveInfo.from_schema_url("https://example.org/image-1.0.0", format_version=None)
    with pytest.raises(ValueError):
        # Not an http(s) URL at all.
        ArchiveInfo.from_schema_url("IMAGE", format_version=None)
    with pytest.raises(ValueError):
        # No host.
        ArchiveInfo.from_schema_url("https:///schemas/image-1.0.0", format_version=None)


def test_input_archive_get_basic_info_base_raises() -> None:
    """Verify InputArchive.get_basic_info raises NotImplementedError on the
    base class.
    """
    with pytest.raises(NotImplementedError):
        InputArchive.get_basic_info("x.fits")


def test_json_basic_info(tmp_path: Path) -> None:
    """Verify JsonInputArchive.get_basic_info returns correct schema
    metadata.
    """
    path = tmp_path / "x.json"
    images_json.write(_make_image(), path)
    info = JsonInputArchive.get_basic_info(path)
    assert info.schema_name == "image"
    assert info.schema_version == "1.0.0"
    assert info.schema_url == "https://images.lsst.io/schemas/image-1.0.0"
    assert info.format_version is None


def test_fits_basic_info(tmp_path: Path) -> None:
    """Verify FitsInputArchive.get_basic_info returns correct schema
    metadata.
    """
    path = tmp_path / "x.fits"
    images_fits.write(_make_image(), path)
    info = FitsInputArchive.get_basic_info(path)
    assert info.schema_name == "image"
    assert info.schema_version == "1.0.0"
    assert info.schema_url == "https://images.lsst.io/schemas/image-1.0.0"
    assert info.format_version == 1


@skip_no_h5py
def test_ndf_basic_info(tmp_path: Path) -> None:
    """Verify NdfInputArchive.get_basic_info returns correct schema
    metadata.
    """
    path = tmp_path / "x.sdf"
    images_ndf.write(_make_image(), path)
    info = NdfInputArchive.get_basic_info(path)
    assert info.schema_name == "image"
    assert info.schema_version == "1.0.0"
    assert info.schema_url == "https://images.lsst.io/schemas/image-1.0.0"
    assert info.format_version == 1


@skip_no_h5py
def test_ndf_basic_info_ignores_nested_json(tmp_path: Path) -> None:
    """Verify NdfInputArchive.get_basic_info reads the top-level schema, not
    nested pointer trees.
    """
    path = tmp_path / "nested.sdf"
    images_ndf.write(_make_image(), path)
    # Inject a nested pointer-tree JSON alongside the top-level one; it
    # sorts before "JSON" so a depth-first scan would hit it first.
    payload = json.dumps({"schema_url": "https://images.lsst.io/schemas/aaa_child-9.9.9"})
    with h5py.File(path, "a") as handle:
        child = handle.require_group("MORE/LSST/AAA")
        child.create_dataset("JSON", data=np.frombuffer(payload.encode("utf-8"), dtype=np.uint8))
    info = NdfInputArchive.get_basic_info(path)
    assert info.schema_name == "image"
    assert info.schema_url == "https://images.lsst.io/schemas/image-1.0.0"

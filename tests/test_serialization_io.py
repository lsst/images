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
from pathlib import Path

import numpy as np
import pytest

from lsst.images import Box, Image, VisitImage
from lsst.images.serialization import ArchiveReadError, read_archive, write_archive
from lsst.utils.introspection import get_full_type_name

try:
    import h5py  # noqa: F401  -- detect availability for NDF round-trip skip

    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

try:
    import piff  # noqa: F401  -- detect availability for piff_psf fixture skip

    PIFF_AVAILABLE = True
except ImportError:
    PIFF_AVAILABLE = False

skip_no_h5py = pytest.mark.skipif(not H5PY_AVAILABLE, reason="h5py is not installed")

LOCAL_DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "schema_v1")

# Full Python type produced when each fixture is read through the generic
# read_archive() API, keyed by the fixture's file name.  These are pinned
# here rather than derived from the schema registry so the test asserts
# the externally-observable type instead of re-running read_archive()'s
# own lookup against itself.
EXPECTED_TYPES = {
    "aperture_correction_map.json": "dict",
    "background_map.json": "lsst.images.BackgroundMap",
    "cell_psf.json": "lsst.images.cells.CellPointSpreadFunction",
    "cell_aperture_correction_map.json": "dict",
    "chebyshev_field.json": "lsst.images.fields.ChebyshevField",
    "coadd_provenance.json": "lsst.images.cells.CoaddProvenance",
    "color_image.json": "lsst.images.ColorImage",
    "detector.json": "lsst.images.cameras.Detector",
    "gaussian_psf.json": "lsst.images.psfs.GaussianPointSpreadFunction",
    "image.json": "lsst.images.Image",
    "mask.json": "lsst.images.Mask",
    "masked_image.json": "lsst.images.MaskedImage",
    "piff_psf.json": "lsst.images.psfs.PiffWrapper",
    "product_field.json": "lsst.images.fields.ProductField",
    "sky_projection.json": "lsst.images.SkyProjection",
    "sum_field.json": "lsst.images.fields.SumField",
    "transform.json": "lsst.images.Transform",
    "visit_image.json": "lsst.images.VisitImage",
    "cell_coadd.json": "lsst.images.cells.CellCoadd",
    "visit_image_dp1.json": "lsst.images.VisitImage",
    "visit_image_dp2.json": "lsst.images.VisitImage",
    "difference_image_dp2.json": "lsst.images.DifferenceImage",
}


def test_generic_read_visit_image_json() -> None:
    """Verify read_archive() on a visit_image JSON fixture returns a
    VisitImage.
    """
    path = os.path.join(LOCAL_DATA_DIR, "visit_image.json")
    result = read_archive(path)
    assert isinstance(result, VisitImage)


def test_generic_read_image_json() -> None:
    """Verify read_archive() on an image JSON fixture returns an Image."""
    path = os.path.join(LOCAL_DATA_DIR, "image.json")
    result = read_archive(path)
    assert isinstance(result, Image)


def test_read_unsupported_extension(tmp_path: Path) -> None:
    """Verify read_archive() raises ValueError for an unrecognized file
    extension.
    """
    path = tmp_path / "bogus.txt"
    with open(path, "w") as f:
        f.write("nope")
    with pytest.raises(ValueError):
        read_archive(path)


def test_read_unregistered_schema(tmp_path: Path) -> None:
    """Verify read_archive() raises ArchiveReadError for a JSON with an unknown
    schema.
    """
    path = tmp_path / "fake.json"
    with open(path, "w") as f:
        f.write(
            '{"schema_url": "https://images.lsst.io/schemas/no-such-schema-99.0.0",'
            ' "schema_version": "99.0.0", "min_read_version": 1, "indirect": []}'
        )
    with pytest.raises(ArchiveReadError) as exc_info:
        read_archive(path)
    assert "no-such-schema" in str(exc_info.value)


@pytest.mark.parametrize("entry", sorted(EXPECTED_TYPES))
def test_fixture_sweep(entry: str) -> None:
    """Verify every schema_v1 JSON fixture reads to the pinned Python type."""
    if entry == "piff_psf.json" and not PIFF_AVAILABLE:
        pytest.skip("piff not available")
    roots = [LOCAL_DATA_DIR, os.path.join(LOCAL_DATA_DIR, "legacy")]
    for root in roots:
        path = os.path.join(root, entry)
        if os.path.exists(path):
            result = read_archive(path)
            assert get_full_type_name(type(result)) == EXPECTED_TYPES[entry], entry
            return
    pytest.skip(f"fixture {entry!r} not found on disk")


def _make_image() -> Image:
    """Return a small float32 Image for round-trip tests."""
    return Image(np.arange(16, dtype=np.float32).reshape(4, 4), bbox=Box.factory[0:4, 0:4])


def test_generic_write_round_trip_fits(tmp_path: Path) -> None:
    """Verify write_archive() + read_archive() round-trips an Image
    through FITS.
    """
    image = _make_image()
    path = tmp_path / "x.fits"
    write_archive(image, path)
    result = read_archive(path)
    assert isinstance(result, Image)
    np.testing.assert_array_equal(result.array, image.array)


def test_generic_write_round_trip_json(tmp_path: Path) -> None:
    """Verify write_archive() + read_archive() round-trips an Image
    through JSON.
    """
    image = _make_image()
    path = tmp_path / "x.json"
    write_archive(image, path)
    result = read_archive(path)
    assert isinstance(result, Image)
    np.testing.assert_array_equal(result.array, image.array)


@skip_no_h5py
def test_generic_write_round_trip_ndf(tmp_path: Path) -> None:
    """Verify write_archive() + read_archive() round-trips an Image
    through NDF.
    """
    image = _make_image()
    path = tmp_path / "x.sdf"
    write_archive(image, path)
    result = read_archive(path)
    assert isinstance(result, Image)
    np.testing.assert_array_equal(result.array, image.array)


def test_read_bbox_subset_fits(tmp_path: Path) -> None:
    """Verify read_archive() forwards bbox kwarg to the FITS backend for subset
    reads.
    """
    img = Image(np.arange(64, dtype=np.float32).reshape(8, 8), bbox=Box.factory[0:8, 0:8])
    path = tmp_path / "x.fits"
    write_archive(img, path)
    sub = read_archive(path, bbox=Box.factory[2:6, 2:6])
    assert sub.array.shape == (4, 4)
    np.testing.assert_array_equal(sub.array, img.array[2:6, 2:6])


def test_read_cls_match() -> None:
    """Verify read_archive() with cls= returns the expected type when it
    matches.
    """
    path = os.path.join(LOCAL_DATA_DIR, "image.json")
    result = read_archive(path, cls=Image)
    assert isinstance(result, Image)


def test_read_cls_mismatch_raises() -> None:
    """Verify read_archive() raises TypeError when the deserialized type
    does not match cls.
    """
    from lsst.images import Mask

    path = os.path.join(LOCAL_DATA_DIR, "image.json")
    with pytest.raises(TypeError) as exc_info:
        read_archive(path, cls=Mask)
    msg = str(exc_info.value)
    assert "image" in msg  # path / schema name
    assert "Image" in msg  # actual deserialized type
    assert "Mask" in msg  # requested cls

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

from pathlib import Path

import numpy as np
import pytest

from lsst.images import Box, Image, VisitImage
from lsst.images.serialization import ArchiveReadError, read_archive, write_archive
from lsst.images.tests import current_fixture_path, iter_schema_fixtures
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

FIXTURE_DIR = Path(__file__).parent / "data" / "schemas"

# Full Python type produced when each fixture is read through the generic
# read_archive() API, keyed by (schema name, variant).  These are pinned here
# rather than derived from the schema registry so the test asserts the
# externally-observable type instead of re-running read_archive()'s own
# lookup against itself.  Keying by name and variant rather than by file name
# keeps the table stable across version bumps.
EXPECTED_TYPES: dict[tuple[str, str | None], str] = {
    ("aperture_correction_map", None): "dict",
    ("background_map", None): "lsst.images.BackgroundMap",
    ("camera_frame_set", None): "lsst.images.cameras.CameraFrameSet",
    ("cell_aperture_correction_map", None): "dict",
    ("cell_coadd", "as_shipped"): "lsst.images.cells.CellCoadd",
    ("cell_coadd", "canonical"): "lsst.images.cells.CellCoadd",
    ("cell_psf", None): "lsst.images.cells.CellPointSpreadFunction",
    ("chebyshev_field", None): "lsst.images.fields.ChebyshevField",
    ("coadd_provenance", None): "lsst.images.cells.CoaddProvenance",
    ("color_image", None): "lsst.images.ColorImage",
    ("detector", None): "lsst.images.cameras.Detector",
    ("difference_image", "dp2"): "lsst.images.DifferenceImage",
    ("gaussian_psf", None): "lsst.images.psfs.GaussianPointSpreadFunction",
    ("image", None): "lsst.images.Image",
    (
        "image_basis_convolution_kernel",
        None,
    ): "lsst.images.convolution_kernels.ImageBasisConvolutionKernel",
    ("mask", None): "lsst.images.Mask",
    ("masked_image", None): "lsst.images.MaskedImage",
    ("observation_summary_stats", None): "lsst.images.ObservationSummaryStats",
    ("piff_psf", None): "lsst.images.psfs.PiffWrapper",
    ("product_field", None): "lsst.images.fields.ProductField",
    ("sky_projection", None): "lsst.images.SkyProjection",
    ("spline_field", None): "lsst.images.fields.SplineField",
    ("sum_field", None): "lsst.images.fields.SumField",
    ("transform", None): "lsst.images.Transform",
    ("visit_image", None): "lsst.images.VisitImage",
    ("visit_image", "dp1"): "lsst.images.VisitImage",
    ("visit_image", "dp2"): "lsst.images.VisitImage",
}


def test_generic_read_visit_image_json() -> None:
    """Verify read_archive() on a visit_image JSON fixture returns a
    VisitImage.
    """
    path = current_fixture_path(FIXTURE_DIR, "visit_image")
    result = read_archive(path)
    assert isinstance(result, VisitImage)


def test_generic_read_image_json() -> None:
    """Verify read_archive() on an image JSON fixture returns an Image."""
    path = current_fixture_path(FIXTURE_DIR, "image")
    result = read_archive(path)
    assert isinstance(result, Image)


def test_read_unsupported_extension(tmp_path: Path) -> None:
    """Verify read_archive() raises ValueError for an unrecognized file
    extension.
    """
    path = tmp_path / "bogus.txt"
    with open(path, "w") as f:
        f.write("nope")
    with pytest.raises(ValueError, match="Unrecognized file extension"):
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


def _sweep_sort_key(entry: tuple[str, str | None]) -> tuple[str, str]:
    """Return a sort key for an ``EXPECTED_TYPES`` entry, ``None`` sorting
    first within a schema name.
    """
    name, variant = entry
    return (name, variant or "")


# Assigned to an explicitly annotated variable rather than sorted() inline in
# the decorator below: parametrize()'s stub types its arguments as `object`,
# and that context leaks into bidirectional generic inference for a bare
# sorted() call, defeating the key function's type.
_SWEEP_ENTRIES: list[tuple[str, str | None]] = sorted(EXPECTED_TYPES, key=_sweep_sort_key)


def test_expected_types_enumerates_every_committed_fixture_variant() -> None:
    """Verify EXPECTED_TYPES matches the fixture tree's (name, variant) pairs
    exactly, so a committed fixture cannot go missing unnoticed.

    Each rung below may legitimately skip an individual case when an optional
    dependency (piff) is unavailable, but that must never be confused with a
    committed fixture file itself going missing: this closes the presence
    gap for every variant at once, rather than relying on each consumer's own
    guard against a missing path (which a stray ``git rm`` or an interrupted
    freeze would otherwise defeat with a silent skip instead of a failure).

    The two directions fail separately and name their own remedy, because
    adding a fixture variant and losing one are unrelated mistakes with
    opposite fixes, and this test is the only thing that catches either.
    """
    present = {(f.name, f.variant) for f in iter_schema_fixtures(FIXTURE_DIR) if not f.retired}
    unlisted = sorted(present - set(EXPECTED_TYPES))
    absent = sorted(set(EXPECTED_TYPES) - present)
    assert not unlisted, (
        f"committed fixture variants missing from EXPECTED_TYPES: {unlisted}; add each with the "
        "full type name its fixture deserializes to, so test_fixture_sweep covers it"
    )
    assert not absent, (
        f"EXPECTED_TYPES entries with no committed fixture: {absent}; delete each entry, or "
        "restore the fixture file if it went missing by accident"
    )


@pytest.mark.parametrize("entry", _SWEEP_ENTRIES)
def test_fixture_sweep(entry: tuple[str, str | None]) -> None:
    """Verify every schema fixture reads to its pinned Python type."""
    name, variant = entry
    if name == "piff_psf" and not PIFF_AVAILABLE:
        pytest.skip("piff not available")
    path = current_fixture_path(FIXTURE_DIR, name, variant=variant)
    assert path.exists(), f"{path} is a committed fixture and must not go missing"
    result = read_archive(path)
    assert get_full_type_name(type(result)) == EXPECTED_TYPES[entry], entry


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


def test_write_metadata_override_does_not_mutate_source(tmp_path: Path) -> None:
    """Write-only metadata is detached from the source object's mapping."""
    image = _make_image()
    image.metadata["shared"] = "source"
    path = tmp_path / "metadata.json"
    tree = write_archive(image, path, metadata={"shared": "file", "extra": 5})

    assert image.metadata == {"shared": "source"}
    assert tree.metadata == {"shared": "file", "extra": 5}
    assert tree.metadata is not image.metadata
    result = read_archive(path, Image)
    assert result.metadata == tree.metadata


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
    path = current_fixture_path(FIXTURE_DIR, "image")
    result = read_archive(path, cls=Image)
    assert isinstance(result, Image)


def test_read_cls_mismatch_raises() -> None:
    """Verify read_archive() raises TypeError when the deserialized type
    does not match cls.
    """
    from lsst.images import Mask

    path = current_fixture_path(FIXTURE_DIR, "image")
    with pytest.raises(TypeError) as exc_info:
        read_archive(path, cls=Mask)
    msg = str(exc_info.value)
    assert "image" in msg  # path / schema name
    assert "Image" in msg  # actual deserialized type
    assert "Mask" in msg  # requested cls

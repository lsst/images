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

import pytest

try:
    from lsst.images.ndf import (
        HdsNameShrinker,
        NdfPointerModel,
        archive_path_to_hdf5_path,
    )
    from lsst.images.ndf._hds import DAT__SZNAM

    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False

skip_no_h5py = pytest.mark.skipif(not HAVE_H5PY, reason="h5py is not installed")


@skip_no_h5py
def test_round_trips_through_json() -> None:
    """Verify NdfPointerModel round-trips through JSON serialization."""
    original = NdfPointerModel(path="/MORE/LSST/PSF")
    json_bytes = original.model_dump_json().encode()
    recovered = NdfPointerModel.model_validate_json(json_bytes)
    assert recovered == original


@skip_no_h5py
def test_archive_path_to_hdf5_path() -> None:
    """Verify archive_path_to_hdf5_path maps archive paths to HDF5 paths."""
    shrinker = HdsNameShrinker()
    assert archive_path_to_hdf5_path("", shrinker) == "/MORE/LSST/JSON"
    assert archive_path_to_hdf5_path("/psf", shrinker) == "/MORE/LSST/PSF"
    assert archive_path_to_hdf5_path("/psf/coefficients", shrinker) == "/MORE/LSST/PSF/COEFFICIENTS"


@skip_no_h5py
def test_archive_path_shrinks_long_components() -> None:
    """Verify long path components are shrunk to the HDS name limit."""
    shrinker = HdsNameShrinker()
    result = archive_path_to_hdf5_path("/psf/this_component_is_too_long", shrinker)
    assert result.startswith("/MORE/LSST/PSF/")
    leaf = result.rsplit("/", 1)[-1]
    assert len(leaf) <= DAT__SZNAM
    # The short parent component is untouched; only the long leaf shrinks.
    assert result.split("/")[3] == "PSF"


@skip_no_h5py
def test_archive_path_shrink_round_trips_to_same_value() -> None:
    """Verify shrinking the same path twice returns the same HDF5 path."""
    shrinker = HdsNameShrinker()
    assert archive_path_to_hdf5_path("/noise_realizations/0", shrinker) == archive_path_to_hdf5_path(
        "/noise_realizations/0", shrinker
    )


@skip_no_h5py
def test_short_names_pass_through_uppercased() -> None:
    """Verify short names are uppercased and passed through unchanged."""
    shrinker = HdsNameShrinker()
    assert shrinker.shrink("psf") == "PSF"
    # A name exactly at the limit passes through unchanged (uppercased).
    assert shrinker.shrink("a" * DAT__SZNAM) == "A" * DAT__SZNAM
    # One character over the limit is shrunk to the limit.
    assert len(shrinker.shrink("a" * (DAT__SZNAM + 1))) == DAT__SZNAM


@skip_no_h5py
def test_long_names_keep_prefix_and_get_counter_token() -> None:
    """Verify long names are truncated to the HDS limit with a counter."""
    shrinker = HdsNameShrinker()
    shrunk = shrinker.shrink("noise_realizations")
    assert len(shrunk) == DAT__SZNAM
    assert shrunk == "NOISE_REALI_001"


@skip_no_h5py
def test_shrink_is_deterministic_per_instance() -> None:
    """Verify shrinking the same name twice returns the same result."""
    shrinker = HdsNameShrinker()
    assert shrinker.shrink("noise_realizations") == shrinker.shrink("noise_realizations")


@skip_no_h5py
def test_distinct_long_names_get_distinct_tokens() -> None:
    """Verify distinct long names with the same prefix get distinct tokens."""
    shrinker = HdsNameShrinker()
    # Identical truncated prefixes cannot collide because the counter
    # increments for each newly assigned name.
    assert shrinker.shrink("noise_realization_field") == "NOISE_REALI_001"
    assert shrinker.shrink("noise_realization_other") == "NOISE_REALI_002"


@skip_no_h5py
def test_reserve_shortens_the_budget() -> None:
    """Verify the reserve parameter reduces the available name length."""
    shrinker = HdsNameShrinker()
    shrunk = shrinker.shrink("noise_realizations", reserve=2)
    assert len(shrunk) == DAT__SZNAM - 2


@skip_no_h5py
def test_version_one_matches_plain_shrink() -> None:
    """Verify shrink_versioned(name, 1) equals plain shrink(name)."""
    shrinker = HdsNameShrinker()
    assert shrinker.shrink_versioned("noise_realizations", 1) == shrinker.shrink("noise_realizations")


@skip_no_h5py
def test_short_versioned_name_keeps_visible_suffix() -> None:
    """Verify short names with a version suffix remain human-readable."""
    shrinker = HdsNameShrinker()
    assert shrinker.shrink_versioned("data", 2) == "DATA_2"


@skip_no_h5py
def test_long_versioned_name_preserves_suffix_within_limit() -> None:
    """Verify long versioned names stay within the HDS limit."""
    shrinker = HdsNameShrinker()
    shrunk = shrinker.shrink_versioned("noise_realizations", 99)
    assert len(shrunk) == DAT__SZNAM
    assert shrunk.endswith("_99")


@skip_no_h5py
def test_same_base_different_versions_are_distinct() -> None:
    """Verify the same base name with differing versions gives distinct
    results.
    """
    shrinker = HdsNameShrinker()
    assert shrinker.shrink_versioned("noise_realizations", 2) != shrinker.shrink_versioned(
        "noise_realizations", 3
    )

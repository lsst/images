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

import unittest

try:
    from lsst.images.ndf._common import (
        NdfPointerModel,
        _shrink_hds_name,
        archive_path_to_hdf5_path,
        shrink_versioned_component,
    )
    from lsst.images.ndf._hds import DAT__SZNAM

    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False


@unittest.skipUnless(HAVE_H5PY, "h5py is not installed")
class NdfPointerModelTestCase(unittest.TestCase):
    """Tests for `NdfPointerModel` and `archive_path_to_hdf5_path`."""

    def test_round_trips_through_json(self):
        original = NdfPointerModel(path="/MORE/LSST/PSF")
        json_bytes = original.model_dump_json().encode()
        recovered = NdfPointerModel.model_validate_json(json_bytes)
        self.assertEqual(recovered, original)

    def test_archive_path_to_hdf5_path(self):
        self.assertEqual(archive_path_to_hdf5_path(""), "/MORE/LSST/JSON")
        self.assertEqual(archive_path_to_hdf5_path("/psf"), "/MORE/LSST/PSF")
        self.assertEqual(archive_path_to_hdf5_path("/psf/coefficients"), "/MORE/LSST/PSF/COEFFICIENTS")

    def test_archive_path_shrinks_long_components(self):
        result = archive_path_to_hdf5_path("/psf/this_component_is_too_long")
        self.assertTrue(result.startswith("/MORE/LSST/PSF/"))
        leaf = result.rsplit("/", 1)[-1]
        self.assertLessEqual(len(leaf), DAT__SZNAM)
        # The short parent component is untouched; only the long leaf shrinks.
        self.assertEqual(result.split("/")[3], "PSF")

    def test_archive_path_shrink_round_trips_to_same_value(self):
        self.assertEqual(
            archive_path_to_hdf5_path("/noise_realizations/0"),
            archive_path_to_hdf5_path("/noise_realizations/0"),
        )


@unittest.skipUnless(HAVE_H5PY, "h5py is not installed")
class ShrinkHdsNameTestCase(unittest.TestCase):
    """Tests for the pure HDS component shrinker."""

    def test_short_names_pass_through_uppercased(self):
        self.assertEqual(_shrink_hds_name("psf"), "PSF")
        # A name exactly at the limit passes through unchanged (uppercased).
        self.assertEqual(_shrink_hds_name("a" * DAT__SZNAM), "A" * DAT__SZNAM)
        # One character over the limit is shrunk to the limit.
        self.assertEqual(len(_shrink_hds_name("a" * (DAT__SZNAM + 1))), DAT__SZNAM)

    def test_long_names_are_shrunk_to_the_limit(self):
        shrunk = _shrink_hds_name("noise_realizations")
        self.assertEqual(len(shrunk), DAT__SZNAM)
        self.assertTrue(shrunk.startswith("NOISE"))
        self.assertEqual(shrunk, shrunk.upper())

    def test_shrink_is_deterministic(self):
        self.assertEqual(
            _shrink_hds_name("noise_realizations"),
            _shrink_hds_name("noise_realizations"),
        )

    def test_distinct_long_names_get_distinct_tokens(self):
        self.assertNotEqual(
            _shrink_hds_name("noise_realization_field"),
            _shrink_hds_name("noise_realization_other"),
        )


@unittest.skipUnless(HAVE_H5PY, "h5py is not installed")
class ShrinkVersionedComponentTestCase(unittest.TestCase):
    """Tests for version-aware HDS component shrinking."""

    def test_version_zero_matches_plain_shrink(self):
        self.assertEqual(
            shrink_versioned_component("noise_realizations", 0),
            _shrink_hds_name("noise_realizations"),
        )

    def test_short_versioned_name_keeps_visible_suffix(self):
        # The second occurrence (0-based version 1) gets a visible _2 suffix.
        self.assertEqual(shrink_versioned_component("data", 1), "DATA_2")

    def test_long_versioned_name_preserves_suffix_within_limit(self):
        shrunk = shrink_versioned_component("noise_realizations", 98)
        self.assertEqual(len(shrunk), DAT__SZNAM)
        self.assertTrue(shrunk.endswith("_99"))

    def test_same_base_different_versions_are_distinct(self):
        self.assertNotEqual(
            shrink_versioned_component("noise_realizations", 1),
            shrink_versioned_component("noise_realizations", 2),
        )

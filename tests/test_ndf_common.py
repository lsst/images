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

    def test_archive_path_to_hdf5_path_rejects_long_components(self):
        with self.assertRaisesRegex(ValueError, "16-character HDS limit"):
            archive_path_to_hdf5_path("/psf/this_component_is_too_long")


@unittest.skipUnless(HAVE_H5PY, "h5py is not installed")
class ShrinkHdsNameTestCase(unittest.TestCase):
    """Tests for the pure HDS component shrinker."""

    def test_short_names_pass_through_uppercased(self):
        self.assertEqual(_shrink_hds_name("psf"), "PSF")
        self.assertEqual(_shrink_hds_name("a" * 16), "A" * 16)

    def test_long_names_are_shrunk_to_the_limit(self):
        shrunk = _shrink_hds_name("noise_realizations")
        self.assertEqual(len(shrunk), 16)
        self.assertTrue(shrunk.startswith("NOISE_R"))
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

    def test_version_one_matches_plain_shrink(self):
        self.assertEqual(
            shrink_versioned_component("noise_realizations", 1),
            _shrink_hds_name("noise_realizations"),
        )

    def test_short_versioned_name_keeps_visible_suffix(self):
        self.assertEqual(shrink_versioned_component("data", 2), "DATA_2")

    def test_long_versioned_name_preserves_suffix_within_limit(self):
        shrunk = shrink_versioned_component("noise_realizations", 99)
        self.assertEqual(len(shrunk), 16)
        self.assertTrue(shrunk.endswith("_99"))

    def test_same_base_different_versions_are_distinct(self):
        self.assertNotEqual(
            shrink_versioned_component("noise_realizations", 2),
            shrink_versioned_component("noise_realizations", 3),
        )

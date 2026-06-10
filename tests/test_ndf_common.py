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
        HdsNameShrinker,
        NdfPointerModel,
        archive_path_to_hdf5_path,
    )
    from lsst.images.ndf._hds import DAT__SZNAM

    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False


@unittest.skipUnless(HAVE_H5PY, "h5py is not installed")
class NdfPointerModelTestCase(unittest.TestCase):
    """Tests for `NdfPointerModel` and `archive_path_to_hdf5_path`."""

    def setUp(self):
        self.shrinker = HdsNameShrinker()

    def test_round_trips_through_json(self):
        original = NdfPointerModel(path="/MORE/LSST/PSF")
        json_bytes = original.model_dump_json().encode()
        recovered = NdfPointerModel.model_validate_json(json_bytes)
        self.assertEqual(recovered, original)

    def test_archive_path_to_hdf5_path(self):
        self.assertEqual(archive_path_to_hdf5_path("", self.shrinker), "/MORE/LSST/JSON")
        self.assertEqual(archive_path_to_hdf5_path("/psf", self.shrinker), "/MORE/LSST/PSF")
        self.assertEqual(
            archive_path_to_hdf5_path("/psf/coefficients", self.shrinker), "/MORE/LSST/PSF/COEFFICIENTS"
        )

    def test_archive_path_shrinks_long_components(self):
        result = archive_path_to_hdf5_path("/psf/this_component_is_too_long", self.shrinker)
        self.assertTrue(result.startswith("/MORE/LSST/PSF/"))
        leaf = result.rsplit("/", 1)[-1]
        self.assertLessEqual(len(leaf), DAT__SZNAM)
        # The short parent component is untouched; only the long leaf shrinks.
        self.assertEqual(result.split("/")[3], "PSF")

    def test_archive_path_shrink_round_trips_to_same_value(self):
        self.assertEqual(
            archive_path_to_hdf5_path("/noise_realizations/0", self.shrinker),
            archive_path_to_hdf5_path("/noise_realizations/0", self.shrinker),
        )


@unittest.skipUnless(HAVE_H5PY, "h5py is not installed")
class HdsNameShrinkerTestCase(unittest.TestCase):
    """Tests for the stateful HDS component shrinker."""

    def setUp(self):
        self.shrinker = HdsNameShrinker()

    def test_short_names_pass_through_uppercased(self):
        self.assertEqual(self.shrinker.shrink("psf"), "PSF")
        # A name exactly at the limit passes through unchanged (uppercased).
        self.assertEqual(self.shrinker.shrink("a" * DAT__SZNAM), "A" * DAT__SZNAM)
        # One character over the limit is shrunk to the limit.
        self.assertEqual(len(self.shrinker.shrink("a" * (DAT__SZNAM + 1))), DAT__SZNAM)

    def test_long_names_keep_prefix_and_get_counter_token(self):
        shrunk = self.shrinker.shrink("noise_realizations")
        self.assertEqual(len(shrunk), DAT__SZNAM)
        self.assertEqual(shrunk, "NOISE_REALI_001")

    def test_shrink_is_deterministic_per_instance(self):
        self.assertEqual(
            self.shrinker.shrink("noise_realizations"),
            self.shrinker.shrink("noise_realizations"),
        )

    def test_distinct_long_names_get_distinct_tokens(self):
        # Identical truncated prefixes cannot collide because the counter
        # increments for each newly assigned name.
        self.assertEqual(self.shrinker.shrink("noise_realization_field"), "NOISE_REALI_001")
        self.assertEqual(self.shrinker.shrink("noise_realization_other"), "NOISE_REALI_002")

    def test_reserve_shortens_the_budget(self):
        shrunk = self.shrinker.shrink("noise_realizations", reserve=2)
        self.assertEqual(len(shrunk), DAT__SZNAM - 2)

    def test_version_one_matches_plain_shrink(self):
        self.assertEqual(
            self.shrinker.shrink_versioned("noise_realizations", 1),
            self.shrinker.shrink("noise_realizations"),
        )

    def test_short_versioned_name_keeps_visible_suffix(self):
        self.assertEqual(self.shrinker.shrink_versioned("data", 2), "DATA_2")

    def test_long_versioned_name_preserves_suffix_within_limit(self):
        shrunk = self.shrinker.shrink_versioned("noise_realizations", 99)
        self.assertEqual(len(shrunk), DAT__SZNAM)
        self.assertTrue(shrunk.endswith("_99"))

    def test_same_base_different_versions_are_distinct(self):
        self.assertNotEqual(
            self.shrinker.shrink_versioned("noise_realizations", 2),
            self.shrinker.shrink_versioned("noise_realizations", 3),
        )

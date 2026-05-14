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
    from lsst.images.ndf._common import NdfPointerModel, archive_path_to_hdf5_path

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

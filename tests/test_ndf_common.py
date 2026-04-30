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

from lsst.images.ndf._common import NdfPointerModel


class NdfPointerModelTestCase(unittest.TestCase):
    def test_round_trips_through_json(self):
        original = NdfPointerModel(ref="/MORE/LSST/PSF")
        json_bytes = original.model_dump_json().encode()
        recovered = NdfPointerModel.model_validate_json(json_bytes)
        self.assertEqual(recovered, original)

    def test_join_path_to_hdf5(self):
        # Helper for routing JSON Pointer paths into MORE/LSST/<UPPER_PATH>.
        from lsst.images.ndf._common import json_pointer_to_hdf5_path

        self.assertEqual(json_pointer_to_hdf5_path(""), "/MORE/LSST/JSON")
        self.assertEqual(json_pointer_to_hdf5_path("/psf"), "/MORE/LSST/PSF")
        self.assertEqual(
            json_pointer_to_hdf5_path("/psf/coefficients"), "/MORE/LSST/PSF_COEFFICIENTS"
        )

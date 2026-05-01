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
import unittest

import numpy as np

from lsst.images import Image
from lsst.images.fits._common import ExtensionKey
from lsst.images.ndf import read

# Starlink-generated NDF fixture (M57 image, hds-v5 HDF5).
EXAMPLE = os.path.join(os.path.dirname(__file__), "data", "example-ndf.sdf")


class StarlinkIngestTestCase(unittest.TestCase):
    """Integration tests that read a real Starlink-produced NDF via the
    auto-detect ``ndf.read()`` path.
    """

    def test_round_trips_to_image(self):
        """Auto-detect path returns an Image with the correct array shape."""
        result = read(Image, EXAMPLE)
        self.assertIsInstance(result.deserialized, Image)
        self.assertEqual(result.deserialized.array.shape, (611, 609))

    def test_wcs_produces_projection(self):
        """Auto-detect path builds a Projection from the NDF WCS component."""
        result = read(Image, EXAMPLE)
        image = result.deserialized
        self.assertIsNotNone(image.projection)
        # M57 (Ring Nebula) is near RA~283.4 deg, Dec~33.0 deg.
        sky = image.projection.pixel_to_sky_transform.apply_forward(
            x=np.array([300.0]), y=np.array([300.0]),
        )
        ra_deg = float(np.degrees(sky.x[0]))
        dec_deg = float(np.degrees(sky.y[0]))
        self.assertAlmostEqual(ra_deg, 283.4, delta=0.5)
        self.assertAlmostEqual(dec_deg, 33.0, delta=0.5)

    def test_opaque_fits_metadata_recovered(self):
        """MORE/FITS cards are surfaced in ``_opaque_metadata``."""
        result = read(Image, EXAMPLE)
        opaque = result.deserialized._opaque_metadata
        self.assertIn(ExtensionKey(), opaque.headers)
        primary = opaque.headers[ExtensionKey()]
        # The fixture is a real Starlink M57 NDF; MORE/FITS carries standard
        # FITS dimension keywords regardless of any later processing.
        self.assertIn("NAXIS", primary)
        self.assertIn("NAXIS1", primary)
        self.assertIn("NAXIS2", primary)


if __name__ == "__main__":
    unittest.main()

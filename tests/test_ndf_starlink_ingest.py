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

import numpy as np
import pytest

from lsst.images import Image
from lsst.images.fits._common import ExtensionKey

try:
    from lsst.images.ndf import read_starlink

    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False

skip_np_h5py = pytest.mark.skipif(not HAVE_H5PY, reason="h5py is not installed")


# Starlink-generated NDF fixture (M57 image, hds-v5 HDF5).
EXAMPLE = os.path.join(os.path.dirname(__file__), "data", "example-ndf.sdf")


@skip_np_h5py
def test_round_trips_to_image() -> None:
    """Verify the auto-detect path returns an Image with the correct
    array shape.
    """
    result = read_starlink(Image, EXAMPLE)
    assert isinstance(result, Image)
    assert result.array.shape == (611, 609)


@skip_np_h5py
def test_wcs_produces_projection() -> None:
    """Verify the auto-detect path builds a Projection from the NDF WCS
    component.
    """
    image = read_starlink(Image, EXAMPLE)
    assert image.sky_projection is not None
    # M57 (Ring Nebula) is near RA~283.4 deg, Dec~33.0 deg.
    sky = image.sky_projection.pixel_to_sky_transform.apply_forward(
        x=np.array([300.0]),
        y=np.array([300.0]),
    )
    ra_deg = float(np.degrees(sky.x[0]))
    dec_deg = float(np.degrees(sky.y[0]))
    assert abs(ra_deg - 283.4) < 0.5
    assert abs(dec_deg - 33.0) < 0.5


@skip_np_h5py
def test_opaque_fits_metadata_recovered() -> None:
    """Verify MORE/FITS cards are surfaced in ``_opaque_metadata``."""
    image = read_starlink(Image, EXAMPLE)
    opaque = image._opaque_metadata
    assert ExtensionKey() in opaque.headers
    primary = opaque.headers[ExtensionKey()]
    # The fixture is a real Starlink M57 NDF; MORE/FITS carries standard
    # FITS dimension keywords regardless of any later processing.
    assert "NAXIS" in primary
    assert "NAXIS1" in primary
    assert "NAXIS2" in primary

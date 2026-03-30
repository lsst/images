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

__all__ = ("make_random_projection",)


import astropy.units as u
import astropy.wcs.wcsapi
import numpy as np

from .._geom import Box
from .._transforms import Frame, Projection


def make_random_projection[F: Frame](rng: np.random.Generator, pixel_frame: F, bbox: Box) -> Projection[F]:
    """Create a test projection with random parameters.

    Parameters
    ----------
    rng
        Random number generator.
    pixel_frame
        Coordinate frame for the pixel grid.
    bbox
        Bounding box for the pixel grid.

    Returns
    -------
    `.Projection`
        A projection.  Guaranteed to be FITS-representable and have no FITS
        approximation attached.
    """
    header = {
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": rng.uniform(low=1, high=bbox.x.size),
        "CRPIX2": rng.uniform(low=1, high=bbox.y.size),
        "CRVAL1": rng.uniform(low=0.0, high=2 * np.pi),
        "CRVAL2": rng.uniform(low=-np.pi, high=np.pi),
        "CDELT1": (rng.uniform(low=0.18, high=0.22) * u.arcsec).to_value(u.deg),
        "CDELT2": (rng.uniform(low=0.18, high=0.22) * u.arcsec).to_value(u.deg),
        "CROTA1": rng.uniform(low=0.0, high=2 * np.pi),
    }
    fits_wcs = astropy.wcs.WCS(header)
    return Projection.from_fits_wcs(
        fits_wcs, pixel_frame, pixel_bounds=bbox, x0=bbox.x.start, y0=bbox.y.start
    )

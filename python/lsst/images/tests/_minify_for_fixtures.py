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

"""Minify a real on-disk archive into a small JSON test fixture.

Reads a FITS or NDF file via the appropriate input archive, takes a
small subset of the in-memory object, and writes JSON via
``JsonOutputArchive``. Used to populate ``tests/data/schema_v1/legacy/``
with derived-from-real test data that exercises the full read path
including the absence-of-stamp legacy default.

Per top-level type the subset rule is:

  Image, Mask, MaskedImage     crop to ~16x16 px
  VisitImage                   crop image, drop secondary metadata
  ColorImage                   crop all bands
  CellCoadd                    TODO: subset cells (need >=4 cells; the
                               outer-ring problem of inputs/PSFs that
                               overlap kept cells is unsolved). A
                               candidate fallback is to morph cells in
                               place — not an accurate subset but
                               sufficient for testing.
  Detector, CameraFrameSet     keep one detector / one frame-set
  BackgroundMap, ApertureCorrectionMap
                               keep a single field/region
  *PSF, *Field, *Transform     already small; copy through

Run interactively:

    .pyenv/bin/python -c "
    from lsst.images.tests._minify_for_fixtures import minify
    minify('/path/to/real.fits', 'tests/data/schema_v1/legacy/foo.json')
    "

This module is intentionally a stub: filling it out requires real
on-disk legacy files (which generally live outside the repo). Each
function below raises ``NotImplementedError`` until pointed at a real
file.
"""

from __future__ import annotations

__all__ = ("minify",)


def minify(in_path: str, out_path: str) -> None:
    """Read a real archive at ``in_path``, take a small subset, and write JSON.

    Parameters
    ----------
    in_path
        Path to a FITS (``.fits`` / ``.fits.gz``) or NDF (``.sdf`` / ``.ndf``)
        file to read.
    out_path
        Path to the JSON fixture to write. The parent directory is
        created if it does not exist.

    Raises
    ------
    ValueError
        If the file extension is not recognised.
    NotImplementedError
        Always raised for now — the per-type minify rules need real
        on-disk files to anchor against. See the module docstring for
        the per-type subset rules and the open CellCoadd issue.
    """
    if in_path.endswith(".fits") or in_path.endswith(".fits.gz"):
        backend = "fits"
    elif in_path.endswith(".sdf") or in_path.endswith(".ndf"):
        backend = "ndf"
    else:
        raise ValueError(f"Unrecognised file extension: {in_path}")
    raise NotImplementedError(
        f"_minify_for_fixtures is a stub. To minify a {backend} file at "
        f"{in_path!r}, fill in the per-type subset rules in this module. "
        "See the module docstring for the rules and the CellCoadd open "
        "question."
    )

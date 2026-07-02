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
from pathlib import Path

import numpy as np
import pytest

from lsst.images import Box, Image
from lsst.images import fits as images_fits
from lsst.images.serialization import Backend, backend_for_path


def test_backend_for_path_fits() -> None:
    """Verify that a .fits path resolves to the FITS backend."""
    from lsst.images.fits import FitsInputArchive

    b = backend_for_path("a/b/c.fits")
    assert isinstance(b, Backend)
    assert b.name == "fits"
    assert b.input_archive is FitsInputArchive
    assert callable(b.write)


def test_backend_for_path_fits_gz() -> None:
    """Verify that .fits.gz and URL-form .fits.gz both resolve to the
    FITS backend.
    """
    assert backend_for_path("c.fits.gz").name == "fits"
    assert backend_for_path("file://a/b/c.fits.gz?param=2").name == "fits"


def test_backend_for_path_json() -> None:
    """Verify that a .json path resolves to the JSON backend."""
    from lsst.images.json import JsonInputArchive

    b = backend_for_path("c.json")
    assert b.name == "json"
    assert b.input_archive is JsonInputArchive


def test_backend_for_path_ndf() -> None:
    """Verify that .sdf and .h5 paths both resolve to the NDF backend."""
    assert backend_for_path("c.sdf").name == "ndf"
    assert backend_for_path("c.h5").name == "ndf"


def test_backend_for_path_unknown_raises() -> None:
    """Verify that an unrecognised suffix raises ValueError naming known
    formats.
    """
    with pytest.raises(ValueError) as exc_info:
        backend_for_path("c.txt")
    assert ".fits" in str(exc_info.value)


def test_minify_unsupported_schema_uses_shared_dispatch(tmp_path: Path) -> None:
    """Verify minify resolves backend and schema via the shared APIs.

    Reaching the 'no subsetter' error proves backend_for_path and
    get_basic_info ran and detected schema_name 'image'.
    """
    from lsst.images.tests._minify_for_fixtures import minify

    src = os.path.join(tmp_path, "plain.fits")
    out = os.path.join(tmp_path, "plain.json")
    images_fits.write(Image(np.zeros((4, 4), dtype=np.float32), bbox=Box.factory[0:4, 0:4]), src)
    with pytest.raises(NotImplementedError) as exc_info:
        minify(src, out)
    assert "image" in str(exc_info.value)

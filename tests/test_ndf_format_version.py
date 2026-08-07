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

from pathlib import Path

import numpy as np
import pytest

from lsst.images import Image
from lsst.images.serialization import ArchiveReadError

try:
    import h5py

    from lsst.images.ndf import write as ndf_write

    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False


def _write_simple_image_ndf(path: Path | str) -> None:
    """Write a tiny Image to ``path`` as an NDF."""
    image = Image(0.0, shape=(4, 4), dtype="float32")
    ndf_write(image, path)


skip_no_h5py = pytest.mark.skipif(not HAVE_H5PY, reason="NDF backend requires h5py")


@skip_no_h5py
def test_write_emits_data_model_and_format_version(tmp_path: Path) -> None:
    """Verify a freshly-written NDF carries DATA_MODEL and FORMAT_VERSION."""
    path = tmp_path / "x.sdf"
    _write_simple_image_ndf(path)
    with h5py.File(path, "r") as f:
        assert "FORMAT_VERSION" in f["/MORE/LSST"]
        assert "DATA_MODEL" in f["/MORE/LSST"]


@skip_no_h5py
def test_read_succeeds_when_format_version_matches(tmp_path: Path) -> None:
    """Verify a freshly-written NDF reads successfully."""
    from lsst.images.ndf import NdfInputArchive

    path = tmp_path / "x.sdf"
    _write_simple_image_ndf(path)
    with NdfInputArchive.open(path):
        pass


@skip_no_h5py
def test_read_fails_when_format_version_too_high(tmp_path: Path) -> None:
    """Verify a file with a newer FORMAT_VERSION raises ArchiveReadError."""
    from lsst.images.ndf import NdfInputArchive

    path = tmp_path / "x.sdf"
    _write_simple_image_ndf(path)
    with h5py.File(path, "r+") as f:
        if "FORMAT_VERSION" in f["/MORE/LSST"]:
            del f["/MORE/LSST/FORMAT_VERSION"]
        f["/MORE/LSST"].create_dataset("FORMAT_VERSION", data=np.int32(2))
    with pytest.raises(ArchiveReadError):
        with NdfInputArchive.open(path):
            pass


@skip_no_h5py
def test_read_fails_when_format_version_absent(tmp_path: Path) -> None:
    """Verify an LSST-written file lacking FORMAT_VERSION is rejected.

    Every file this package writes stamps the container layout, so its
    absence alongside an LSST JSON tree is a damaged file rather than a
    legacy one, and guessing version 1 would guess at the layout.
    """
    from lsst.images.ndf import NdfInputArchive

    path = tmp_path / "x.sdf"
    _write_simple_image_ndf(path)
    with h5py.File(path, "r+") as f:
        del f["/MORE/LSST/FORMAT_VERSION"]
    with pytest.raises(ArchiveReadError, match="FORMAT_VERSION"):
        with NdfInputArchive.open(path):
            pass


@skip_no_h5py
def test_get_basic_info_fails_when_format_version_absent(tmp_path: Path) -> None:
    """Verify the info-only read requires the stamp as well.

    ``get_basic_info`` already refuses a file with no ``DATA_MODEL``, so by
    the time it reads the container version the LSST extension is known to be
    present and the stamp must be there too.
    """
    from lsst.images.ndf import NdfInputArchive

    path = tmp_path / "x.sdf"
    _write_simple_image_ndf(path)
    with h5py.File(path, "r+") as f:
        del f["/MORE/LSST/FORMAT_VERSION"]
    with pytest.raises(ArchiveReadError, match="FORMAT_VERSION"):
        NdfInputArchive.get_basic_info(path)


@skip_no_h5py
def test_read_starlink_needs_no_format_version(tmp_path: Path) -> None:
    """Verify a Starlink NDF still reads without any LSST extension.

    A file with no LSST extension has no container layout of ours to
    version, so the required stamp applies only to our own files; these go
    through ``read_starlink`` rather than the generic archive read.
    """
    from lsst.images.ndf import read_starlink

    path = tmp_path / "x.sdf"
    _write_simple_image_ndf(path)
    with h5py.File(path, "r+") as f:
        del f["/MORE/LSST"]
    assert read_starlink(Image, path).bbox.area == 16

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

from lsst.images import Box, Image

try:
    import zarr  # noqa: F401

    from lsst.images.zarr import write

    HAVE_ZARR = True
except ImportError:
    HAVE_ZARR = False

try:
    from ome_zarr.io import parse_url
    from ome_zarr.reader import Reader

    HAVE_OME_ZARR = True
except ImportError:
    HAVE_OME_ZARR = False

skip_no_ome_zarr = pytest.mark.skipif(not (HAVE_ZARR and HAVE_OME_ZARR), reason="ome-zarr is not installed")


@skip_no_ome_zarr
def test_ome_zarr_can_open_image(tmp_path: Path) -> None:
    """``ome-zarr-py`` can open archives written by ``lsst.images.zarr``."""
    original = Image(
        np.arange(20, dtype=np.float32).reshape(4, 5),
        bbox=Box.factory[10:14, 20:25],
    )
    target = str(tmp_path / "out.zarr")
    write(original, target)
    location = parse_url(target)
    assert location is not None
    reader = Reader(location)
    nodes = list(reader())
    assert len(nodes) >= 1
    data = nodes[0].data[0]  # level 0
    assert tuple(data.shape) == (4, 5)

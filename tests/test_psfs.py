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
import warnings
from typing import Any

import numpy as np
import pytest

from lsst.images import Box
from lsst.images.psfs import GaussianPointSpreadFunction, PiffWrapper, PointSpreadFunction, PSFExWrapper
from lsst.images.psfs._piff import _ArchivePiffWriter
from lsst.images.tests import (
    RoundtripFits,
    RoundtripJson,
    RoundtripNdf,
    RoundtripZarr,
    compare_psf_to_legacy,
)

try:
    import h5py  # noqa: F401

    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False

try:
    import zarr  # noqa: F401

    HAVE_ZARR = True
except ImportError:
    HAVE_ZARR = False

try:
    from lsst.afw.detection import Psf as LegacyPsf
except ImportError:
    type LegacyPsf = Any  # type: ignore[no-redef]

EXTERNAL_DATA_DIR = os.environ.get("TESTDATA_IMAGES_DIR", None)

skip_no_h5py = pytest.mark.skipif(not HAVE_H5PY, reason="h5py is not installed")
skip_no_zarr = pytest.mark.skipif(not HAVE_ZARR, reason="zarr is not installed")


@pytest.fixture(scope="session")
def legacy_piff_psf_and_bbox() -> tuple[LegacyPsf, Box]:
    """Return a legacy-wrapped Piff PSF and its bounding box.

    Skips if TESTDATA_IMAGES_DIR is unset, piff is unavailable, or afw is
    unavailable.
    """
    if EXTERNAL_DATA_DIR is None:
        pytest.skip("TESTDATA_IMAGES_DIR is not in the environment.")
    try:
        import piff  # noqa: F401

        from lsst.afw.image import ExposureFitsReader
    except ImportError:
        pytest.skip("'piff' or 'lsst.afw.image' could not be imported.")
    filename = os.path.join(EXTERNAL_DATA_DIR, "dp2", "legacy", "visit_image.fits")
    reader = ExposureFitsReader(filename)
    legacy_psf = reader.readPsf()
    bounds = Box.from_legacy(reader.readBBox())
    return legacy_psf, bounds


@pytest.fixture(scope="session")
def legacy_psfex_psf_and_bbox() -> tuple[LegacyPsf, Box]:
    """Return a legacy PSFEx PSF and its bounding box

    Skips if TESTDATA_IMAGES_DIR is unset, afw is unavailable, or psfex is
    unavailable.
    """
    if EXTERNAL_DATA_DIR is None:
        pytest.skip("TESTDATA_IMAGES_DIR is not in the environment.")
    try:
        from lsst.afw.image import ExposureFitsReader
        from lsst.meas.extensions.psfex import PsfexPsf  # noqa: F401
    except ImportError:
        pytest.skip("'lsst.afw.image' or 'lsst.meas.extensions.psfex' could not be imported.")
    filename = os.path.join(EXTERNAL_DATA_DIR, "dp2", "legacy", "preliminary_visit_image.fits")
    reader = ExposureFitsReader(filename)
    legacy_psf = reader.readPsf()
    bounds = Box.from_legacy(reader.readBBox())
    return legacy_psf, bounds


def test_gaussian() -> None:
    """Test the built-in Gaussian PSF implementation."""
    bounds = Box.factory[-1024:1024, -2048:2048]
    psf = GaussianPointSpreadFunction(2.5, bounds=bounds, stamp_size=33)
    assert psf.bounds == bounds

    kernel = psf.compute_kernel_image(x=5.0, y=3.0)
    assert kernel.bbox == psf.kernel_bbox
    assert abs(float(kernel.array.sum()) - 1.0) < 1e-6
    center = kernel.array.shape[0] // 2
    assert np.unravel_index(np.argmax(kernel.array), kernel.array.shape) == (center, center)

    stellar = psf.compute_stellar_image(x=5.25, y=3.75)
    assert stellar.bbox == psf.compute_stellar_bbox(x=5.25, y=3.75)
    assert abs(float(stellar.array.sum()) - 1.0) < 1e-6
    assert stellar.array[center - 1, center] > stellar.array[center + 1, center]
    assert stellar.array[center, center] > stellar.array[center, center - 1]
    assert stellar.array[center, center] > stellar.array[center - 1, center]

    with RoundtripFits(psf) as roundtrip:
        assert roundtrip.result == psf, f"{roundtrip.result} != {psf}"

    with pytest.raises(ValueError):
        # Even stamp size.
        GaussianPointSpreadFunction(2.5, bounds=bounds, stamp_size=32)

    with pytest.raises(ValueError):
        # Negative stamp size.
        GaussianPointSpreadFunction(2.5, bounds=bounds, stamp_size=-33)

    with pytest.raises(ValueError):
        # Negative sigma.
        GaussianPointSpreadFunction(-2.5, bounds=bounds, stamp_size=33)


def test_piff_writer_normalizes_tuple_metadata():  # intentionally untyped
    """Test that Piff metadata is normalized to JSON-like values."""
    writer = _ArchivePiffWriter()
    writer.write_struct(
        "interp",
        {
            "keys": ("u", "v"),
            "scale": np.float64(1.5),
            "flags": [np.bool_(True), np.int64(3)],
        },
    )
    model = writer.serialize(None)  # type: ignore[arg-type]
    assert model.structs["interp"]["keys"] == ["u", "v"]
    assert model.structs["interp"]["scale"] == 1.5
    assert model.structs["interp"]["flags"] == [True, 3]
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        model.model_dump_json()


def test_piff(legacy_piff_psf_and_bbox: tuple[LegacyPsf, Box]) -> None:
    """Test round-tripping a legacy Piff PSF through FITS and JSON archives,
    and converting it back to a legacy PSF.
    """
    from piff import PSF

    legacy_psf, bounds = legacy_piff_psf_and_bbox
    psf = PointSpreadFunction.from_legacy(legacy_psf, bounds)
    assert isinstance(psf, PiffWrapper)
    assert psf.bounds == bounds
    assert isinstance(psf.piff_psf, PSF)
    compare_psf_to_legacy(psf, legacy_psf)
    with RoundtripFits(psf) as roundtrip1:
        pass
    compare_psf_to_legacy(roundtrip1.result, legacy_psf)
    with RoundtripJson(psf) as roundtrip2:
        pass
    compare_psf_to_legacy(roundtrip2.result, legacy_psf)
    legacy_psf_2 = roundtrip1.result.to_legacy()
    compare_psf_to_legacy(psf, legacy_psf_2)
    assert legacy_psf.getAveragePosition() == legacy_psf_2.getAveragePosition()


@skip_no_h5py
def test_piff_ndf_roundtrip(legacy_piff_psf_and_bbox: tuple[LegacyPsf, Box]) -> None:
    """Test round-tripping a legacy Piff PSF through an NDF archive."""
    legacy_psf, bounds = legacy_piff_psf_and_bbox
    psf = PointSpreadFunction.from_legacy(legacy_psf, bounds)
    with RoundtripNdf(psf) as roundtrip:
        pass
    compare_psf_to_legacy(roundtrip.result, legacy_psf)


@skip_no_zarr
def test_piff_zarr_roundtrip(legacy_piff_psf_and_bbox: tuple[LegacyPsf, Box]) -> None:
    """Test round-tripping a legacy Piff PSF through a zarr archive."""
    legacy_psf, bounds = legacy_piff_psf_and_bbox
    psf = PointSpreadFunction.from_legacy(legacy_psf, bounds)
    with RoundtripZarr(psf) as roundtrip:
        pass
    compare_psf_to_legacy(roundtrip.result, legacy_psf)


def test_psfex(legacy_psfex_psf_and_bbox: tuple[LegacyPsf, Box]) -> None:
    """Test wrapping a legacy PSFEx PSF and round-tripping through FITS and
    JSON.
    """
    from lsst.meas.extensions.psfex import PsfexPsf

    legacy_psf, bounds = legacy_psfex_psf_and_bbox
    psf = PointSpreadFunction.from_legacy(legacy_psf, bounds)
    assert isinstance(psf, PSFExWrapper)
    assert psf.bounds == bounds
    assert isinstance(psf.legacy_psf, PsfexPsf)
    compare_psf_to_legacy(psf, legacy_psf)
    with RoundtripFits(psf) as roundtrip1:
        pass
    compare_psf_to_legacy(roundtrip1.result, legacy_psf)
    with RoundtripJson(psf) as roundtrip2:
        pass
    compare_psf_to_legacy(roundtrip2.result, legacy_psf)

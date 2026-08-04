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

from lsst.images import Box, Ellipse, Image
from lsst.images.describe import DescribableMixin, Report
from lsst.images.psfs import GaussianPointSpreadFunction, PiffWrapper, PointSpreadFunction, PSFExWrapper
from lsst.images.psfs._piff import PiffObjectModel, _ArchivePiffReader, _ArchivePiffWriter
from lsst.images.tests import (
    RoundtripFits,
    RoundtripJson,
    RoundtripNdf,
    assert_values_equal,
    compare_psf_to_legacy,
    reset_afw_mask_planes,  # noqa: F401
)

try:
    import h5py  # noqa: F401

    HAVE_H5PY = True
except ImportError:
    HAVE_H5PY = False

try:
    import galsim  # noqa: F401

    HAVE_GALSIM = True
except ImportError:
    HAVE_GALSIM = False

try:
    from lsst.afw.detection import Psf as LegacyPsf
except ImportError:
    type LegacyPsf = Any  # type: ignore[no-redef]

EXTERNAL_DATA_DIR = os.environ.get("TESTDATA_IMAGES_DIR", None)

skip_no_h5py = pytest.mark.skipif(not HAVE_H5PY, reason="h5py is not installed")
skip_no_galsim = pytest.mark.skipif(not HAVE_GALSIM, reason="galsim is not installed")


@pytest.fixture
def legacy_piff_psf_and_bbox(reset_afw_mask_planes: None) -> tuple[LegacyPsf, Box]:  # noqa: F811
    """Return a legacy-wrapped Piff PSF and its bounding box.

    Skips if TESTDATA_IMAGES_DIR is unset, piff is unavailable, or afw is
    unavailable.
    """
    # reset_afw_mask_planes will have already skipped if afw is not available.
    from lsst.afw.image import ExposureFitsReader

    if EXTERNAL_DATA_DIR is None:
        pytest.skip("TESTDATA_IMAGES_DIR is not in the environment.")
    try:
        import piff  # noqa: F401
    except ImportError:
        pytest.skip("'piff' could not be imported.")
    filename = os.path.join(EXTERNAL_DATA_DIR, "dp2", "legacy", "visit_image.fits")
    reader = ExposureFitsReader(filename)
    legacy_psf = reader.readPsf()
    bounds = Box.from_legacy(reader.readBBox())
    return legacy_psf, bounds


@pytest.fixture
def legacy_psfex_psf_and_bbox(reset_afw_mask_planes: None) -> tuple[LegacyPsf, Box]:  # noqa: F811
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


def test_gaussian_psf_repr_pinned() -> None:
    """GaussianPointSpreadFunction repr matches its documented form."""
    psf = GaussianPointSpreadFunction(2.5, bounds=Box.factory[0:10, 0:10], stamp_size=21)
    assert repr(psf) == f"GaussianPointSpreadFunction(2.5, stamp_size=21, bounds={psf.bounds!r})"


def test_gaussian_psf_describe() -> None:
    """GaussianPointSpreadFunction._describe returns a Report."""
    psf = GaussianPointSpreadFunction(2.5, bounds=Box.factory[0:10, 0:10], stamp_size=21)
    assert isinstance(psf, DescribableMixin)
    report = psf._describe()
    assert isinstance(report, Report)
    assert report.type_name == "GaussianPointSpreadFunction"
    labels = {f.label for f in report.fields}
    assert "sigma" in labels
    assert "stamp_size" in labels
    assert "bounds" in labels


def test_psf_base_describe() -> None:
    """Base _describe uses type(self).__name__ and includes bounds/kernel_bbox.

    GaussianPointSpreadFunction overrides _describe, so a minimal concrete
    subclass is used to exercise the actual base implementation.
    """

    class _MinimalPSF(PointSpreadFunction):
        """Minimal concrete subclass that exercises the base _describe path."""

        @property
        def bounds(self) -> Box:
            return Box.factory[-5:5, -5:5]

        @property
        def kernel_bbox(self) -> Box:
            return Box.factory[-2:3, -2:3]

        def compute_kernel_image(self, *, x: float, y: float) -> Image:
            arr = np.zeros((5, 5))
            arr[2, 2] = 1.0
            return Image(arr, bbox=self.kernel_bbox)

        def compute_stellar_image(self, *, x: float, y: float) -> Image:
            return self.compute_kernel_image(x=x, y=y)

        def compute_stellar_bbox(self, *, x: float, y: float) -> Box:
            return self.kernel_bbox

    psf = _MinimalPSF()
    report = psf._describe()
    assert report.type_name == "_MinimalPSF"
    labels = {f.label for f in report.fields}
    assert "bounds" in labels
    assert "kernel_bbox" in labels


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

    with pytest.raises(ValueError, match="stamp_size must be odd"):
        # Even stamp size.
        GaussianPointSpreadFunction(2.5, bounds=bounds, stamp_size=32)

    with pytest.raises(ValueError, match="stamp_size must be positive"):
        # Negative stamp size.
        GaussianPointSpreadFunction(2.5, bounds=bounds, stamp_size=-33)

    with pytest.raises(ValueError, match="sigma must be positive"):
        # Negative sigma.
        GaussianPointSpreadFunction(-2.5, bounds=bounds, stamp_size=33)


def test_gaussian_from_legacy(reset_afw_mask_planes: None) -> None:  # noqa: F811
    """Test converting a legacy GaussianPsf to and from
    GaussianPointSpreadFunction.
    """
    # reset_afw_mask_planes will have already skipped if afw is not available.
    from lsst.afw.detection import GaussianPsf
    from lsst.geom import Point2D

    bounds = Box.factory[0:64, 0:48]
    legacy_psf = GaussianPsf(33, 33, 2.5)
    psf = PointSpreadFunction.from_legacy(legacy_psf, bounds)
    assert isinstance(psf, GaussianPointSpreadFunction)
    assert psf == GaussianPointSpreadFunction(2.5, bounds=bounds, stamp_size=33)
    # The kernels match to round-off, so compare_psf_to_legacy (which checks
    # exact equality) cannot be used.
    for x, y in [(10.0, 10.0), (10.3, 20.7)]:
        legacy_kernel = Image.from_legacy(legacy_psf.computeKernelImage(Point2D(x, y)))
        kernel = psf.compute_kernel_image(x=x, y=y)
        assert kernel.bbox == legacy_kernel.bbox
        np.testing.assert_allclose(kernel.array, legacy_kernel.array, rtol=0, atol=1e-15)
    legacy_psf_2 = psf.to_legacy()
    assert isinstance(legacy_psf_2, GaussianPsf)
    assert legacy_psf_2.getSigma() == legacy_psf.getSigma()
    assert legacy_psf_2.getDimensions() == legacy_psf.getDimensions()

    with pytest.raises(ValueError, match="must be square"):
        PointSpreadFunction.from_legacy(GaussianPsf(33, 31, 2.5), bounds)


def _make_moments_psf() -> GaussianPointSpreadFunction:
    """Return a Gaussian PSF with a large, odd size for moments measurement."""
    return GaussianPointSpreadFunction(2.5, bounds=Box.factory[0:10, 0:10], stamp_size=33)


def test_compute_effective_area_gaussian() -> None:
    """Test that the effective area of a Gaussian PSF equals 4*pi*sigma^2."""
    psf = _make_moments_psf()
    expected = 4.0 * np.pi * 2.5**2
    for x, y in [(0.0, 0.0), (5.25, 3.75)]:
        assert_values_equal(psf.compute_effective_area(x=x, y=y), expected, rtol=1e-7)
    assert_values_equal(
        psf.compute_effective_area(x=5.25, y=3.75, use_stellar_image=True), expected, rtol=1e-7
    )


def test_compute_moments_kernel_unweighted() -> None:
    """Test compute_moments with use_stellar_image=False, adaptive=False."""
    psf = _make_moments_psf()
    ell = psf.compute_moments(x=5.0, y=3.0, adaptive=False)
    assert isinstance(ell, Ellipse)
    assert_values_equal(ell.center.x, 0.0, atol=1e-7)
    assert_values_equal(ell.center.y, 0.0, atol=1e-7)
    assert_values_equal(ell.xx, 2.5**2, rtol=1e-7)
    assert_values_equal(ell.yy, 2.5**2, rtol=1e-7)
    assert_values_equal(ell.xy, 0.0, atol=1e-7)
    assert_values_equal(ell.r_tr, 2.5, rtol=1e-7)


def test_compute_moments_stellar_unweighted() -> None:
    """Test compute_moments with use_stellar_image=True, adaptive=False."""
    psf = _make_moments_psf()
    x, y = 5.25, 3.75
    ell = psf.compute_moments(x=x, y=y, adaptive=False, use_stellar_image=True)
    assert isinstance(ell, Ellipse)
    assert_values_equal(ell.center.x, x, atol=1e-7)
    assert_values_equal(ell.center.y, y, atol=1e-7)
    assert_values_equal(ell.xx, 2.5**2, rtol=1e-7)
    assert_values_equal(ell.yy, 2.5**2, rtol=1e-7)
    assert_values_equal(ell.xy, 0.0, atol=1e-7)
    assert_values_equal(ell.r_tr, 2.5, rtol=1e-7)


@skip_no_galsim
def test_compute_moments_kernel_adaptive() -> None:
    """Test compute_moments with use_stellar_image=False, adaptive=True."""
    psf = _make_moments_psf()
    ell = psf.compute_moments(x=5.0, y=3.0, adaptive=True)
    assert isinstance(ell, Ellipse)
    assert_values_equal(ell.center.x, 0.0, atol=1e-7)
    assert_values_equal(ell.center.y, 0.0, atol=1e-7)
    assert_values_equal(ell.xx, 2.5**2, rtol=1e-7)
    assert_values_equal(ell.yy, 2.5**2, rtol=1e-7)
    assert_values_equal(ell.xy, 0.0, atol=1e-7)
    assert_values_equal(ell.r_tr, 2.5, rtol=1e-7)


@skip_no_galsim
def test_compute_moments_stellar_adaptive() -> None:
    """Test compute_moments with use_stellar_image=True, adaptive=True."""
    psf = _make_moments_psf()
    x, y = 5.25, 3.75
    ell = psf.compute_moments(x=x, y=y, adaptive=True, use_stellar_image=True)
    assert isinstance(ell, Ellipse)
    assert_values_equal(ell.center.x, x, atol=1e-7)
    assert_values_equal(ell.center.y, y, atol=1e-7)
    assert_values_equal(ell.xx, 2.5**2, rtol=1e-7)
    assert_values_equal(ell.yy, 2.5**2, rtol=1e-7)
    assert_values_equal(ell.xy, 0.0, atol=1e-7)
    assert_values_equal(ell.r_tr, 2.5, rtol=1e-7)


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


def test_piff_writer_rejects_bandpass() -> None:
    """Test that a Piff PSF with a bandpass is rejected on write."""
    galsim = pytest.importorskip("galsim")
    writer = _ArchivePiffWriter()
    bandpass = galsim.Bandpass("wave/1000", wave_type="nm", blue_limit=400.0, red_limit=800.0)
    with pytest.raises(NotImplementedError, match="bandpass"):
        writer.write_bandpass("bandpass", bandpass)


def test_piff_reader_reports_no_bandpass() -> None:
    """Test that we can read serialized Piff PSFs with a Piff library
    that expects a bandpass.
    """
    reader = _ArchivePiffReader(PiffObjectModel(), None)  # type: ignore[arg-type]
    assert reader.read_bandpass("bandpass") is None


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

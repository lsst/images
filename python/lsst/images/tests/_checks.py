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

__all__ = (
    "arrays_to_legacy_points",
    "assert_close",
    "check_astropy_wcs_interface",
    "check_projection",
    "check_transform",
    "compare_projection_to_legacy_wcs",
    "compare_psf_to_legacy",
    "legacy_coords_to_astropy",
    "legacy_points_to_xy_array",
)

import unittest
from typing import Any, cast

import astropy.units as u
import astropy.wcs.wcsapi
import numpy as np
from astropy.coordinates import SkyCoord

from .._geom import XY, Box
from .._image import Image
from .._transforms import Frame, Projection, SkyFrame, Transform
from ..psfs import PointSpreadFunction


def assert_close(
    tc: unittest.TestCase,
    a: np.ndarray | u.Quantity | float,
    b: np.ndarray | u.Quantity | float,
    **kwargs: Any,
) -> None:
    """Test that two arrays, floats, or quantities are almost equal.

    Parameters
    ----------
    tc
        Test case object with assert methods to use.
    a
        Array, float, or quantity to compare.
    b
        Array, float, or quantity to compare.
    **kwargs
        Forwarded to `astropy.units.allclose`.
    """
    tc.assertTrue(u.allclose(a, b, **kwargs), msg=f"{a} != {b}")


def compare_psf_to_legacy(tc: unittest.TestCase, psf: PointSpreadFunction, legacy_psf: Any) -> None:
    """Compare a PSF model object to its legacy interface.

    Parameters
    ----------
    tc
        Test case object with assert methods to use.
    """
    from lsst.geom import Point2D

    for p in [Point2D(50.0, 60.0), Point2D(801.2, 322.8), Point2D(33.5, 22.1)]:
        tc.assertEqual(psf.kernel_bbox, Box.from_legacy(legacy_psf.computeKernelBBox(p)))
        tc.assertEqual(
            psf.compute_kernel_image(x=p.x, y=p.y), Image.from_legacy(legacy_psf.computeKernelImage(p))
        )
        tc.assertEqual(
            psf.compute_stellar_bbox(x=p.x, y=p.y), Box.from_legacy(legacy_psf.computeImageBBox(p))
        )
        tc.assertEqual(psf.compute_stellar_image(x=p.x, y=p.y), Image.from_legacy(legacy_psf.computeImage(p)))


def compare_projection_to_legacy_wcs[F: Frame](
    tc: unittest.TestCase,
    projection: Projection[F],
    legacy_wcs: Any,
    pixel_frame: F,
    subimage_bbox: Box,
    is_fits: bool = False,
) -> None:
    """Test a Projection object by comparing it to an equivalent
    `lsst.afw.geom.SkyWcs`.

    Parameters
    ----------
    tc
        Test case object with assert methods to use.
    projection
        Projection to test.
    legacy_wcs : ``lsst.afw.geom.SkyWcs``
        Equivalent legacy WCS.
    pixel_frame
        Expected pixel frame for the projection.
    subimage_bbox
        Bounding box of points to generate for tests.
    is_fits
        Whether this projection is expected to be exactly representable as a
        FIT WCS. If `False` it is assumed to have a FITS approximation
        attached instead.
    """
    # Pixel coordinates to test on over the subimage region of interest:
    pixel_xy = subimage_bbox.meshgrid(step=50).map(np.ravel)
    # Array indices of those pixel values (subtract off bbox starts):
    subimage_array_xy = XY(x=pixel_xy.x - subimage_bbox.x.start, y=pixel_xy.y - subimage_bbox.y.start)
    sky_coords = legacy_coords_to_astropy(
        legacy_wcs.pixelToSky(arrays_to_legacy_points(pixel_xy.x, pixel_xy.y))
    )
    # Test transforming with the Projection itself, which also tests its
    # nested Transform and an Astropy High-Level WCS view with no origin
    # change.
    check_projection(tc, projection, pixel_xy, sky_coords, pixel_frame)
    # Also test the Astropy High-Level WCS view with an origin change to
    # array indices.
    check_astropy_wcs_interface(
        tc, projection.as_astropy(subimage_bbox), subimage_array_xy, sky_coords, pixel_atol=1e-5
    )
    if is_fits:
        fits_wcs = projection.as_fits_wcs(subimage_bbox, allow_approximation=True)
        assert fits_wcs is not None
        check_astropy_wcs_interface(tc, fits_wcs, subimage_array_xy, sky_coords, pixel_atol=1e-5)
        # Use that FITS approximation to check that we can make a
        # Projection from a FITS WCs, too.
        fits_projection = Projection.from_fits_wcs(fits_wcs, pixel_frame)
        check_projection(
            tc,
            fits_projection,
            subimage_array_xy,
            sky_coords,
            pixel_frame,
            pixel_atol=1e-5,
        )
    else:
        tc.assertIsNone(projection.as_fits_wcs(subimage_bbox, allow_approximation=False))
        # The legacy SkyWcs should instead have a FITS approximation
        # attached; run the same tests on that.
        assert projection.fits_approximation is not None
        compare_projection_to_legacy_wcs(
            tc,
            projection.fits_approximation,
            legacy_wcs.getFitsApproximation(),
            pixel_frame,
            subimage_bbox,
            is_fits=True,
        )


def check_transform[I: Frame, O: Frame](
    tc: unittest.TestCase,
    transform: Transform[I, O],
    input_xy: XY[np.ndarray],
    output_xy: XY[np.ndarray],
    in_frame: Frame,
    out_frame: Frame,
    *,
    check_inverted: bool = True,
    in_atol: u.Quantity | None = None,
    out_atol: u.Quantity | None = None,
) -> None:
    """Test Transform against known arrays of input and output points.

    Parameters
    ----------
    tc
        Test case object with assert methods to use.
    transform
        Transform to test.
    input_xy
        Arrays of input points.
    outout_xy
        Arrays of output points.
    in_frame
        Expected input frame.
    out_frame
        Expected output frame.
    check_inverted
        If `True`, recurse (once) to test the inverse transform.
    in_atol
        Expected absolute precision of input points.
    out_atol
        Expected absolute precision of output points.
    """
    tc.assertEqual(transform.in_frame, in_frame)
    tc.assertEqual(transform.out_frame, out_frame)
    in_atol_v = in_atol.to_value(in_frame.unit) if in_atol is not None else None
    out_atol_v = out_atol.to_value(out_frame.unit) if out_atol is not None else None
    # Test array interfaces.
    test_output_xy = transform.apply_forward(x=input_xy.x, y=input_xy.y)
    assert_close(tc, test_output_xy.x, output_xy.x, atol=out_atol_v)
    assert_close(tc, test_output_xy.y, output_xy.y, atol=out_atol_v)
    test_input_xy = transform.apply_inverse(x=output_xy.x, y=output_xy.y)
    assert_close(tc, test_input_xy.x, input_xy.x, atol=in_atol_v)
    assert_close(tc, test_input_xy.y, input_xy.y, atol=in_atol_v)
    # Test scalar interfaces with numpy scalars.
    for input_x, input_y, output_x, output_y in zip(input_xy.x, input_xy.y, output_xy.x, output_xy.y):
        assert_close(tc, transform.apply_forward(x=input_x, y=input_y).x, output_x, atol=out_atol_v)
        assert_close(tc, transform.apply_forward(x=input_x, y=input_y).y, output_y, atol=out_atol_v)
        assert_close(tc, transform.apply_inverse(x=output_x, y=output_y).x, input_x, atol=in_atol_v)
        assert_close(tc, transform.apply_inverse(x=output_x, y=output_y).y, input_y, atol=in_atol_v)
    # Test quantity array interfaces.
    input_q_xy = XY(x=input_xy.x * transform.in_frame.unit, y=input_xy.y * transform.in_frame.unit)
    output_q_xy = XY(x=output_xy.x * transform.out_frame.unit, y=output_xy.y * transform.out_frame.unit)
    test_output_q_xy = transform.apply_forward_q(x=input_q_xy.x, y=input_q_xy.y)
    assert_close(tc, test_output_q_xy.x, output_q_xy.x, atol=out_atol)
    assert_close(tc, test_output_q_xy.y, output_q_xy.y, atol=out_atol)
    test_input_q_xy = transform.apply_inverse_q(x=output_q_xy.x, y=output_q_xy.y)
    assert_close(tc, test_input_q_xy.x, input_q_xy.x, atol=in_atol)
    assert_close(tc, test_input_q_xy.y, input_q_xy.y, atol=in_atol)
    # Test quantity scalar interfaces.
    for input_q_x, input_q_y, output_q_x, output_q_y in zip(
        input_q_xy.x, input_q_xy.y, output_q_xy.x, output_q_xy.y
    ):
        assert_close(tc, transform.apply_forward_q(x=input_q_x, y=input_q_y).x, output_q_x, atol=out_atol)
        assert_close(tc, transform.apply_forward_q(x=input_q_x, y=input_q_y).y, output_q_y, atol=out_atol)
        assert_close(tc, transform.apply_inverse_q(x=output_q_x, y=output_q_y).x, input_q_x, atol=in_atol)
        assert_close(tc, transform.apply_inverse_q(x=output_q_x, y=output_q_y).y, input_q_y, atol=in_atol)
    if check_inverted:
        # Test the inverse transform.
        check_transform(
            tc,
            transform.inverted(),
            output_xy,
            input_xy,
            out_frame,
            in_frame,
            check_inverted=False,
            out_atol=in_atol,
            in_atol=out_atol,
        )


def check_projection[P: Frame](
    tc: unittest.TestCase,
    projection: Projection[P],
    pixel_xy: XY[np.ndarray],
    sky_coords: SkyCoord,
    pixel_frame: Frame,
    *,
    pixel_atol: float | None = None,
    sky_atol: u.Quantity | None = None,
) -> None:
    """Test a Projection instance against known arrays of pixel and sky
    coordinates.

    Parameters
    ----------
    tc
        Test case object with assert methods to use.
    projection
        Projection to test.
    pixel_xy
        Arrays of pixel coordinates.
    sky_coords
        Corresponding sky coordinates.
    pixel_frame
        Expected pixel frame.
    pixel_atol
        Expected absolute precision of pixel points.
    sky_atol
        Expected absolute precision of sky coordinates.
    """
    tc.assertEqual(projection.pixel_frame, pixel_frame)
    tc.assertEqual(projection.sky_frame, SkyFrame.ICRS)
    sky_atol_v = sky_atol.to_value(SkyFrame.ICRS.unit) if sky_atol is not None else None
    pixel_atol_q = pixel_atol * u.pix if pixel_atol is not None else None
    # Test array interfaces.
    test_pixel_xy = cast(XY[np.ndarray], projection.sky_to_pixel(sky_coords))
    assert_close(tc, test_pixel_xy.x, pixel_xy.x, atol=pixel_atol)
    assert_close(tc, test_pixel_xy.y, pixel_xy.y, atol=pixel_atol)
    test_sky_astropy = projection.pixel_to_sky(x=pixel_xy.x, y=pixel_xy.y)
    assert_close(tc, test_sky_astropy.ra, sky_coords.ra, atol=sky_atol_v)
    assert_close(tc, test_sky_astropy.dec, sky_coords.dec, atol=sky_atol_v)
    # Test scalar interfaces.
    for pixel_x, pixel_y, sky_single in zip(pixel_xy.x, pixel_xy.y, sky_coords):
        assert_close(tc, projection.sky_to_pixel(sky_single).x, pixel_x, atol=pixel_atol)
        assert_close(tc, projection.sky_to_pixel(sky_single).y, pixel_y, atol=pixel_atol)
        assert_close(tc, projection.pixel_to_sky(x=pixel_x, y=pixel_y).ra, sky_single.ra, atol=sky_atol_v)
        assert_close(tc, projection.pixel_to_sky(x=pixel_x, y=pixel_y).dec, sky_single.dec, atol=sky_atol_v)
    # Test the underlying Transform object.
    sky_xy = XY(x=sky_coords.ra.to_value(u.rad), y=sky_coords.dec.to_value(u.rad))
    check_transform(
        tc,
        projection.pixel_to_sky_transform,
        pixel_xy,
        sky_xy,
        pixel_frame,
        SkyFrame.ICRS,
        check_inverted=False,
        in_atol=pixel_atol_q,
        out_atol=sky_atol,
    )
    check_transform(
        tc,
        projection.sky_to_pixel_transform,
        sky_xy,
        pixel_xy,
        SkyFrame.ICRS,
        pixel_frame,
        check_inverted=False,
        in_atol=sky_atol,
        out_atol=pixel_atol_q,
    )
    # Test the Astropy interface adapter.
    check_astropy_wcs_interface(
        tc, projection.as_astropy(), pixel_xy, sky_coords, pixel_atol=pixel_atol, sky_atol=sky_atol
    )


def check_astropy_wcs_interface(
    tc: unittest.TestCase,
    wcs: astropy.wcs.wcsapi.BaseHighLevelWCS,
    pixel_xy: XY[np.ndarray],
    sky_coords: SkyCoord,
    *,
    pixel_atol: float | None = None,
    sky_atol: u.Quantity | None = None,
) -> None:
    """Test an Astropy WCS instance against known arrays of pixel and
    sky coordinates.

    Parameters
    ----------
    tc
        Test case object with assert methods to use.
    wcs
        Astropy WCS object to test.
    pixel_xy
        Arrays of pixel coordinates.
    sky_coords
        Corresponding sky coordinates.
    pixel_atol
        Expected absolute precision of pixel points.
    sky_atol
        Expected absolute precision of sky coordinates.
    """
    test_x, test_y = wcs.world_to_pixel(sky_coords)
    assert_close(tc, test_x, pixel_xy.x, atol=pixel_atol)
    assert_close(tc, test_y, pixel_xy.y, atol=pixel_atol)
    test_sky_coords = wcs.pixel_to_world(pixel_xy.x, pixel_xy.y)
    assert_close(tc, test_sky_coords.ra, sky_coords.ra, atol=sky_atol)
    assert_close(tc, test_sky_coords.dec, sky_coords.dec, atol=sky_atol)


def legacy_points_to_xy_array(legacy_points: list[Any]) -> XY[np.ndarray]:
    """Convert a list of ``lsst.geom.Point2D`` objects to an `.XY` array."""
    return XY(x=np.array([p.x for p in legacy_points]), y=np.array([p.y for p in legacy_points]))


def legacy_coords_to_astropy(legacy_coords: list[Any]) -> SkyCoord:
    """Convert a list of ``lsst.geom.SpherePoint`` objects to an Astropy
    coordinate object.
    """
    return SkyCoord(
        ra=np.array([p.getRa().asRadians() for p in legacy_coords]) * u.rad,
        dec=np.array([p.getDec().asRadians() for p in legacy_coords]) * u.rad,
    )


def arrays_to_legacy_points(x: np.ndarray, y: np.ndarray) -> list[Any]:
    """Convert arrays of ``x`` and ``y`` to a list of ``lsst.geom.Point2D``."""
    from lsst.geom import Point2D

    return [Point2D(x=xv, y=yv) for xv, yv in zip(x, y)]

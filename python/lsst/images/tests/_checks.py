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
    "assert_equal_allow_nan",
    "assert_images_equal",
    "assert_masked_images_equal",
    "assert_masks_equal",
    "assert_projections_equal",
    "assert_psfs_equal",
    "check_astropy_wcs_interface",
    "check_projection",
    "check_transform",
    "compare_amplifier_to_legacy",
    "compare_aperture_corrections_to_legacy",
    "compare_cell_coadd_to_legacy",
    "compare_detector_to_legacy",
    "compare_field_to_legacy",
    "compare_image_to_legacy",
    "compare_mask_to_legacy",
    "compare_masked_image_to_legacy",
    "compare_observation_summary_stats_to_legacy",
    "compare_projection_to_legacy_wcs",
    "compare_psf_to_legacy",
    "compare_visit_image_to_legacy",
    "legacy_coords_to_astropy",
    "legacy_points_to_xy_array",
)

import dataclasses
import math
import unittest
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, cast

import astropy.units as u
import astropy.wcs.wcsapi
import numpy as np
from astropy.coordinates import SkyCoord

from .._geom import XY, YX, BoundsError, Box
from .._image import Image
from .._mask import Mask, MaskPlane
from .._masked_image import MaskedImage
from .._observation_summary_stats import ObservationSummaryStats
from .._transforms import DetectorFrame, Frame, Projection, SkyFrame, TractFrame, Transform
from .._visit_image import VisitImage
from ..aperture_corrections import ApertureCorrectionMap
from ..cameras import Amplifier, Detector, DetectorType, ReadoutCorner
from ..cells import CellCoadd, CellIJ, CoaddProvenance
from ..fields import BaseField
from ..psfs import PointSpreadFunction

if TYPE_CHECKING:
    try:
        from lsst.cell_coadds import MultipleCellCoadd
    except ImportError:
        type MultipleCellCoadd = Any  # type: ignore[no-redef]


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


def assert_equal_allow_nan(tc: unittest.TestCase, a: float, b: float) -> None:
    """Test that two floating point values are equal, with nan == nan."""
    try:
        tc.assertEqual(a, b)
    except AssertionError:
        if not (math.isnan(a) and math.isnan(b)):
            raise


def assert_images_equal(
    tc: unittest.TestCase,
    a: Image,
    b: Image,
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
    expect_view: bool | Literal["array"] | None = None,
) -> None:
    """Assert that two images are equal or nearly equal."""
    tc.assertEqual(a.bbox, b.bbox)
    tc.assertEqual(a.unit, b.unit)
    assert_projections_equal(tc, a.projection, b.projection)
    if expect_view is not None:
        tc.assertEqual(np.may_share_memory(a.array, b.array), bool(expect_view))
        if expect_view == "array":
            tc.assertEqual(a.metadata, b.metadata)
        else:
            tc.assertEqual(a.metadata is b.metadata, expect_view)
    if not expect_view:
        assert_close(tc, a.array, b.array, atol=atol, rtol=rtol)
        tc.assertEqual(a.metadata, b.metadata)


def assert_masks_equal(tc: unittest.TestCase, a: Mask, b: Mask) -> None:
    """Assert that two masks are equal or nearly equal."""
    tc.assertEqual(a.bbox, b.bbox)
    tc.assertEqual(a.schema, b.schema)
    tc.assertEqual(a.metadata, b.metadata)
    assert_projections_equal(tc, a.projection, b.projection)
    np.testing.assert_array_equal(a.array, b.array)


def assert_masked_images_equal(
    tc: unittest.TestCase,
    a: MaskedImage,
    b: MaskedImage,
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
    expect_view: bool | None = None,
) -> None:
    """Assert that two masked images are equal or nearly equal."""
    tc.assertEqual(a.metadata, b.metadata)
    assert_projections_equal(tc, a.projection, b.projection)
    assert_images_equal(tc, a.image, b.image, rtol=rtol, atol=atol, expect_view=expect_view)
    assert_masks_equal(tc, a.mask, b.mask)
    assert_images_equal(tc, a.variance, b.variance, rtol=rtol, atol=atol, expect_view=expect_view)


def assert_psfs_equal(
    tc: unittest.TestCase,
    psf1: PointSpreadFunction,
    psf2: PointSpreadFunction,
    points: YX[np.ndarray] | XY[np.ndarray] | None = None,
) -> int:
    """Compare two PSF objets.

    Parameters
    ----------
    tc
        Test case object with assert methods to use.
    psf1
        Point-spread function to test.
    psf2
        The other point-spread function to test.
    points
        Points to evaluate the PSFs at. If not provided, the intersection of
        the PSF bounding boxes are used to generate points on a grid.

    Returns
    -------
    `int`
        The number of points actually tested.
    """
    if points is None:
        points = psf1.bounds.bbox.intersection(psf1.bounds.bbox).meshgrid(3).map(np.ravel)

    tc.assertEqual(psf1.kernel_bbox, psf2.kernel_bbox)

    n_points_tested: int = 0
    for x, y in zip(points.x, points.y):
        if not psf1.bounds.contains(x=x, y=y):
            with tc.assertRaises(BoundsError):
                psf2.compute_kernel_image(x=x, y=y)
            continue
        tc.assertEqual(psf1.compute_kernel_image(x=x, y=y), psf2.compute_kernel_image(x=x, y=y))
        tc.assertEqual(psf1.compute_stellar_bbox(x=x, y=y), psf2.compute_stellar_bbox(x=x, y=y))
        tc.assertEqual(psf1.compute_stellar_image(x=x, y=y), psf2.compute_stellar_image(x=x, y=y))
        n_points_tested += 1
    return n_points_tested


def compare_image_to_legacy(
    tc: unittest.TestCase, image: Image, legacy_image: Any, expect_view: bool | None = None
) -> None:
    """Compare an `.Image` object to a legacy `lsst.afw.image.Image` object."""
    tc.assertEqual(image.bbox, Box.from_legacy(legacy_image.getBBox()))
    if expect_view is not None:
        tc.assertEqual(np.may_share_memory(image.array, legacy_image.array), expect_view)
    if not expect_view:
        np.testing.assert_array_equal(image.array, legacy_image.array)


def compare_mask_to_legacy(
    tc: unittest.TestCase, mask: Mask, legacy_mask: Any, plane_map: Mapping[str, MaskPlane] | None = None
) -> None:
    """Compare a `.Mask` object to a legacy `lsst.afw.image.Mask` object."""
    tc.assertEqual(mask.bbox, Box.from_legacy(legacy_mask.getBBox()))
    if plane_map is None:
        plane_map = {plane.name: plane for plane in mask.schema if plane is not None}
    for old_name, new_plane in plane_map.items():
        np.testing.assert_array_equal(
            (legacy_mask.array & legacy_mask.getPlaneBitMask(old_name)).astype(bool),
            mask.get(new_plane.name),
        )


def compare_masked_image_to_legacy(
    tc: unittest.TestCase,
    masked_image: MaskedImage,
    legacy_masked_image: Any,
    *,
    plane_map: Mapping[str, MaskPlane] | None = None,
    expect_view: bool | None = None,
    alternates: Mapping[str, Any] | None = None,
) -> None:
    """Compare a `.MaskedImage` object to a legacy `lsst.afw.image.MaskedImage`
    object.

    Parameters
    ----------
    tc
        Test case to use for asserts.
    masked_image
        New image to test.
    legacy_masked_image
        Legacy image to test against.
    plane_map
        Mapping between new and legacy mask planes.
    expect_view
        Whether to test that the image and variance arrays do or do not share
        memory.
    alternates
        A mapping of other versions of one or more (new) components to also
        check against the legacy versions of those components.
    """
    compare_image_to_legacy(tc, masked_image.image, legacy_masked_image.getImage(), expect_view=expect_view)
    compare_mask_to_legacy(tc, masked_image.mask, legacy_masked_image.getMask(), plane_map=plane_map)
    compare_image_to_legacy(
        tc, masked_image.variance, legacy_masked_image.getVariance(), expect_view=expect_view
    )
    if alternates:
        if image := alternates.get("image"):
            compare_image_to_legacy(tc, image, legacy_masked_image.getImage(), expect_view=expect_view)
        if mask := alternates.get("mask"):
            compare_mask_to_legacy(tc, mask, legacy_masked_image.getMask(), plane_map=plane_map)
        if variance := alternates.get("variance"):
            compare_image_to_legacy(tc, variance, legacy_masked_image.getVariance(), expect_view=expect_view)


def compare_visit_image_to_legacy(
    tc: unittest.TestCase,
    visit_image: VisitImage,
    legacy_exposure: Any,
    *,
    plane_map: Mapping[str, MaskPlane] | None = None,
    expect_view: bool | None = None,
    instrument: str,
    visit: int,
    detector: int,
    alternates: Mapping[str, Any] | None = None,
) -> None:
    """Compare a `.VisitImage` object to a legacy `lsst.afw.image.Exposure`
    object.

    Parameters
    ----------
    tc
        Test case to use for asserts.
    visit_image
        New image to test.
    legacy_exposure
        Legacy image to test against.
    plane_map
        Mapping between new and legacy mask planes.
    expect_view
        Whether to test that the image and variance arrays do or do not share
        memory.
    instrument
        Expected instrument name.
    visit
        Expected visit ID.
    detector
        Expected detector ID.
    alternates
        A mapping of other versions of one or more (new) components to also
        check against the legacy versions of those components.
    """
    compare_masked_image_to_legacy(
        tc,
        visit_image,
        legacy_exposure.getMaskedImage(),
        plane_map=plane_map,
        expect_view=expect_view,
        alternates=alternates,
    )
    detector_bbox = Box.from_legacy(legacy_exposure.getDetector().getBBox())
    compare_projection_to_legacy_wcs(
        tc,
        visit_image.projection,
        legacy_exposure.getWcs(),
        DetectorFrame(instrument=instrument, visit=visit, detector=detector, bbox=detector_bbox),
        visit_image.bbox,
    )
    tc.assertIs(visit_image.projection, visit_image.mask.projection)
    tc.assertIs(visit_image.projection, visit_image.variance.projection)
    compare_psf_to_legacy(tc, visit_image.psf, legacy_exposure.getPsf())
    compare_observation_summary_stats_to_legacy(
        tc, visit_image.summary_stats, legacy_exposure.info.getSummaryStats()
    )
    # Make a tiny box for Field comparisons that need to make arrays; that can
    # get expensive otherwisre.
    tiny_bbox = detector_bbox.local[2:4, 3:6]
    compare_aperture_corrections_to_legacy(
        tc, visit_image.aperture_corrections, legacy_exposure.info.getApCorrMap(), tiny_bbox
    )
    if alternates:
        if projection := alternates.get("projection"):
            compare_projection_to_legacy_wcs(
                tc,
                projection,
                legacy_exposure.getWcs(),
                DetectorFrame(instrument=instrument, visit=visit, detector=detector, bbox=detector_bbox),
                visit_image.bbox,
            )
        if psf := alternates.get("psf"):
            compare_psf_to_legacy(tc, psf, legacy_exposure.getPsf())
        if summary_stats := alternates.get("summary_stats"):
            compare_observation_summary_stats_to_legacy(
                tc, summary_stats, legacy_exposure.info.getSummaryStats()
            )
        if obs_info := alternates.get("obs_info"):
            visitInfo = legacy_exposure.visitInfo
            tc.assertEqual(obs_info.instrument, visitInfo.getInstrumentLabel())
        if aperture_corrections := alternates.get("aperture_corrections"):
            compare_aperture_corrections_to_legacy(
                tc, aperture_corrections, legacy_exposure.info.getApCorrMap(), tiny_bbox
            )


def compare_cell_coadd_to_legacy(
    tc: unittest.TestCase,
    cell_coadd: CellCoadd,
    legacy_cell_coadd: MultipleCellCoadd,
    *,
    tract_bbox: Box,
    plane_map: Mapping[str, MaskPlane] | None = None,
    alternates: Mapping[str, Any] | None = None,
    psf_points: XY[np.ndarray] | YX[np.ndarray] | None = None,
) -> None:
    """Compare a `.cells.CellCoadd` object to a legacy
    `lsst.cell_coadds.MultipleCellCoadd` object.

    Parameters
    ----------
    tc
        Test case to use for asserts.
    cell_coadd
        New coadd to test.
    legacy_cell_coadd
        Legacy coadd to test against.
    tract_bbox
        Bounding box of the full tract.
    psf_points
        Points to use to compare the PSFs.
    plane_map
        Mapping between new and legacy mask planes.
    alternates
        A mapping of other versions of one or more (new) components to also
        check against the legacy versions of those components.
    """
    legacy_stitched = legacy_cell_coadd.stitch(cell_coadd.bbox.to_legacy())
    compare_image_to_legacy(tc, cell_coadd.image, legacy_stitched.image, expect_view=False)
    compare_mask_to_legacy(tc, cell_coadd.mask, legacy_stitched.mask, plane_map=plane_map)
    compare_image_to_legacy(tc, cell_coadd.variance, legacy_stitched.variance, expect_view=False)
    if legacy_stitched.mask_fractions is not None:
        compare_image_to_legacy(
            tc, cell_coadd.mask_fractions["rejected"], legacy_stitched.mask_fractions, expect_view=False
        )
    for n in range(legacy_stitched.n_noise_realizations):
        compare_image_to_legacy(
            tc, cell_coadd.noise_realizations[n], legacy_stitched.noise_realizations[n], expect_view=False
        )
    tc.assertEqual(cell_coadd.skymap, legacy_stitched.identifiers.skymap)
    tc.assertEqual(cell_coadd.tract, legacy_stitched.identifiers.tract)
    tc.assertEqual(cell_coadd.patch.index.x, legacy_stitched.identifiers.patch.x)
    tc.assertEqual(cell_coadd.patch.index.y, legacy_stitched.identifiers.patch.y)
    tc.assertEqual(cell_coadd.band, legacy_stitched.identifiers.band)
    tc.assertTrue(tract_bbox.contains(cell_coadd.patch.outer_bbox))
    tc.assertTrue(cell_coadd.patch.outer_bbox.contains(cell_coadd.patch.inner_bbox))
    tc.assertTrue(cell_coadd.patch.outer_bbox.contains(cell_coadd.bbox))
    tc.assertEqual(cell_coadd.unit, u.Unit(legacy_cell_coadd.common.units.value))
    tc.assertTrue(cell_coadd.bounds.bbox.contains(cell_coadd.bbox))
    tc.assertTrue(cell_coadd.grid.bbox.contains(cell_coadd.bbox))
    compare_projection_to_legacy_wcs(
        tc,
        cell_coadd.projection,
        legacy_cell_coadd.wcs,
        TractFrame(
            skymap=legacy_cell_coadd.identifiers.skymap,
            tract=legacy_cell_coadd.identifiers.tract,
            bbox=tract_bbox,
        ),
        cell_coadd.bbox,
        is_fits=True,
    )
    tc.assertIs(cell_coadd.projection, cell_coadd.mask.projection)
    tc.assertIs(cell_coadd.projection, cell_coadd.variance.projection)
    compare_psf_to_legacy(
        tc, cell_coadd.psf, legacy_stitched.psf, expect_legacy_raise_on_out_of_bounds=True, points=psf_points
    )
    compare_cell_coadd_provenance_to_legacy(tc, cell_coadd.provenance, legacy_cell_coadd)
    if alternates:
        if projection := alternates.get("projection"):
            compare_projection_to_legacy_wcs(
                tc,
                projection,
                legacy_stitched.wcs,
                TractFrame(
                    skymap=legacy_cell_coadd.identifiers.skymap,
                    tract=legacy_cell_coadd.identifiers.tract,
                    bbox=tract_bbox,
                ),
                cell_coadd.bbox,
                is_fits=True,
            )
        if psf := alternates.get("psf"):
            compare_psf_to_legacy(tc, psf, legacy_stitched.psf, points=psf_points)
        if provenance := alternates.get("provenance"):
            compare_cell_coadd_provenance_to_legacy(tc, provenance, legacy_cell_coadd)


def compare_cell_coadd_provenance_to_legacy(
    tc: unittest.TestCase, provenance: CoaddProvenance, legacy_cell_coadd: MultipleCellCoadd
) -> None:
    """Compare a `.cells.CoaddProvenance` object to a legacy
    `lsst.cell_coadds.MultipleCellCoadd` object.

    Parameters
    ----------
    tc
        Test case to use for asserts.
    provenance
        New provenance object to test.
    legacy_cell_coadd
        Legacy coadd to test against.
    """
    from lsst.cell_coadds import ObservationIdentifiers

    for legacy_cell in legacy_cell_coadd.cells.values():
        cell_index = CellIJ.from_legacy(legacy_cell.identifiers.cell)
        prov = provenance[cell_index]
        legacy_table = astropy.table.Table(
            rows=[
                [
                    ids.instrument,
                    ids.visit,
                    ids.detector,
                    ids.day_obs,
                    ids.physical_filter,
                    legacy_input.overlaps_center,
                    legacy_input.overlap_fraction,
                    legacy_input.weight,
                    legacy_input.psf_shape.getIxx(),
                    legacy_input.psf_shape.getIyy(),
                    legacy_input.psf_shape.getIxy(),
                    legacy_input.psf_shape_flag,
                ]
                for ids, legacy_input in legacy_cell.inputs.items()
            ],
            dtype=[
                np.object_,
                np.uint64,
                np.uint16,
                np.uint32,
                np.object_,
                np.bool_,
                np.float64,
                np.float64,
                np.float64,
                np.float64,
                np.float64,
                np.bool_,
            ],
            names=[
                "instrument",
                "visit",
                "detector",
                "day_obs",
                "physical_filter",
                "overlaps_center",
                "overlap_fraction",
                "weight",
                "psf_shape_xx",
                "psf_shape_yy",
                "psf_shape_xy",
                "psf_shape_flag",
            ],
        )
        # For a single cell all 'inputs' are also 'contributions'.
        tc.assertEqual(len(legacy_cell.inputs), len(prov.inputs))
        tc.assertEqual(len(legacy_cell.inputs), len(prov.contributions))
        prov.inputs.sort(["instrument", "visit", "detector"])
        prov.contributions.sort(["instrument", "visit", "detector"])
        legacy_table.sort(["instrument", "visit", "detector"])
        np.testing.assert_array_equal(prov.inputs["instrument"], prov.contributions["instrument"])
        np.testing.assert_array_equal(prov.inputs["visit"], prov.contributions["visit"])
        np.testing.assert_array_equal(prov.inputs["detector"], prov.contributions["detector"])
        np.testing.assert_array_equal(prov.inputs["instrument"], legacy_table["instrument"])
        np.testing.assert_array_equal(prov.inputs["visit"], legacy_table["visit"])
        np.testing.assert_array_equal(prov.inputs["detector"], legacy_table["detector"])
        np.testing.assert_array_equal(prov.inputs["physical_filter"], legacy_table["physical_filter"])
        np.testing.assert_array_equal(prov.inputs["day_obs"], legacy_table["day_obs"])
        np.testing.assert_array_equal(prov.contributions["overlaps_center"], legacy_table["overlaps_center"])
        np.testing.assert_array_equal(
            prov.contributions["overlap_fraction"], legacy_table["overlap_fraction"]
        )
        np.testing.assert_array_equal(prov.contributions["weight"], legacy_table["weight"])
        np.testing.assert_array_equal(prov.contributions["psf_shape_xx"], legacy_table["psf_shape_xx"])
        np.testing.assert_array_equal(prov.contributions["psf_shape_yy"], legacy_table["psf_shape_yy"])
        np.testing.assert_array_equal(prov.contributions["psf_shape_xy"], legacy_table["psf_shape_xy"])
        np.testing.assert_array_equal(prov.contributions["psf_shape_flag"], legacy_table["psf_shape_flag"])
        for row in prov.inputs:
            polygon_key = ObservationIdentifiers(**{k: row[k] for k in row.keys() if k != "polygon"})
            legacy_polygon = legacy_cell_coadd.common.visit_polygons[polygon_key]
            tc.assertEqual(legacy_polygon, row["polygon"].to_legacy())


def compare_psf_to_legacy(
    tc: unittest.TestCase,
    psf: PointSpreadFunction,
    legacy_psf: Any,
    points: YX[np.ndarray] | XY[np.ndarray] | None = None,
    expect_legacy_raise_on_out_of_bounds: bool = False,
) -> int:
    """Compare a PSF model object to its legacy interface.

    Parameters
    ----------
    tc
        Test case object with assert methods to use.
    psf
        Point-spread function to test.
    legacy_psf
        Legacy `lsst.afw.detection.Psf` instance to compare with.
    points
        Points to evaluate the PSFs at. If not provided, the intersection of
        the PSF bounding boxes are used to generate points on a grid.
    expect_legacy_raise_on_out_of_bounds
        If `True`, expect ``legacy_psf`` to raise
        `lsst.afw.detection.InvalidPsfError` when evaluated at a position
        considered out-of-bounds by ``psf``.

    Returns
    -------
    `int`
        The number of points actually tested.
    """
    from lsst.afw.detection import InvalidPsfError

    if points is None:
        points = psf.bounds.bbox.meshgrid(n=3).map(np.ravel)
    legacy_points = arrays_to_legacy_points(points.x, points.y)
    n_points_tested: int = 0
    for p in legacy_points:
        if not psf.bounds.contains(x=p.x, y=p.y):
            if expect_legacy_raise_on_out_of_bounds:
                with tc.assertRaises(InvalidPsfError):
                    legacy_psf.computeKernelImage(p)
            continue
        tc.assertEqual(psf.kernel_bbox, Box.from_legacy(legacy_psf.computeKernelBBox(p)))
        tc.assertEqual(
            psf.compute_kernel_image(x=p.x, y=p.y), Image.from_legacy(legacy_psf.computeKernelImage(p))
        )
        tc.assertEqual(
            psf.compute_stellar_bbox(x=p.x, y=p.y), Box.from_legacy(legacy_psf.computeImageBBox(p))
        )
        tc.assertEqual(psf.compute_stellar_image(x=p.x, y=p.y), Image.from_legacy(legacy_psf.computeImage(p)))
        n_points_tested += 1
    return n_points_tested


def compare_field_to_legacy(
    tc: unittest.TestCase,
    field: BaseField,
    legacy_field: Any,
    subimage_bbox: Box,
) -> None:
    """Test a Field object by comparing it to an equivalent
    `lsst.afw.math.BoundedField`.

    Parameters
    ----------
    tc
        Test case object with assert methods to use.
    field
        Field to test.
    legacy_field : ``lsst.afw.math.BoundedField``
        Equivalent legacy bounded field.
    subimage_bbox
        Bounding box for full-image tests.
    """
    tc.assertEqual(field.bounds.bbox, Box.from_legacy(legacy_field.getBBox()))
    # Pixel coordinates to test the numpy array interface with.
    pixel_xy = field.bounds.bbox.meshgrid(n=5).map(np.ravel)
    assert_close(tc, field(x=pixel_xy.x, y=pixel_xy.y), legacy_field.evaluate(pixel_xy.x, pixel_xy.y))
    legacy_image_1 = Image(0, bbox=subimage_bbox, dtype=np.float64).to_legacy()
    legacy_field.addToImage(legacy_image_1, overlapOnly=True)
    assert_images_equal(tc, field.render(subimage_bbox), Image.from_legacy(legacy_image_1), rtol=1e-13)


def compare_aperture_corrections_to_legacy(
    tc: unittest.TestCase,
    aperture_corrections: ApertureCorrectionMap,
    legacy_ap_corr_map: Any,
    subimage_bbox: Box,
) -> None:
    """Test an aperture correction `dict` by comparing it to an equivalent
    `lsst.afw.image.ApCorrMap`.

    Parameters
    ----------
    tc
        Test case object with assert methods to use.
    aperture_corrections
        Dictionary to test.
    legacy_ap_corr_map : ``lsst.afw.image.ApCorrMap``
        Equivalent legacy aperture correction map.
    subimage_bbox
        Bounding box for full-image tests.
    """
    tc.assertEqual(aperture_corrections.keys(), set(legacy_ap_corr_map.keys()))
    for name, field in aperture_corrections.items():
        compare_field_to_legacy(tc, field, legacy_ap_corr_map[name], subimage_bbox)


def compare_observation_summary_stats_to_legacy(
    tc: unittest.TestCase,
    summary_stats: ObservationSummaryStats,
    legacy_summary_stats: Any,
) -> None:
    """Test an ObservationSummaryStats object by comparing it to an equivalent
    `lsst.afw.image.ExposureSummaryStats`.

    Parameters
    ----------
    tc
        Test case object with assert methods to use.
    summary_stats
        Struct to test.
    legacy : ``lsst.afw.image.ExposureSummaryStats``
        Equivalent legacy struct.
    """
    for field in dataclasses.fields(legacy_summary_stats):
        a = getattr(legacy_summary_stats, field.name)
        b = getattr(summary_stats, field.name)
        if isinstance(b, tuple):
            for ai, bi in zip(a, b):
                tc.assertTrue(ai == bi or (math.isnan(ai) and math.isnan(bi)), f"{field.name}: {a} != {b}")
        else:
            tc.assertTrue(a == b or (math.isnan(a) and math.isnan(b)), f"{field.name}: {a} != {b}")


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
        # Projection from a FITS WCS, too.
        fits_projection = Projection.from_fits_wcs(fits_wcs, pixel_frame)
        check_projection(
            tc,
            fits_projection,
            subimage_array_xy,
            sky_coords,
            pixel_frame,
            pixel_atol=1e-5,
        )
        # We want Projections we create from a FITS WCS to be backed by an
        # AST FrameSet so we can convert them into legacy
        # `lsst.afw.geom.SkyWcs` objects if desired.
        tc.assertIn("Begin FrameSet", fits_projection.show())
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
    output_xy
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
    """Test a `.Projection` instance against known arrays of pixel and sky
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


def assert_projections_equal(
    tc: unittest.TestCase,
    a: Projection[Any] | None,
    b: Projection[Any] | None,
    expect_identity: bool | None = None,
) -> None:
    """Test that two `.Projection` instances are equivalent."""
    if a is None and b is None:
        return
    assert a is not None and b is not None
    match expect_identity:
        case True:
            tc.assertIs(a, b)
            return
        case False:
            tc.assertIsNot(a, b)
        case None if a is b:
            return
    tc.assertEqual(a.pixel_frame, b.pixel_frame)
    tc.assertEqual(a.show(simplified=True), b.show(simplified=True))
    assert_projections_equal(
        tc, a.fits_approximation, cast(Projection[Any], b.fits_approximation), expect_identity=False
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


def compare_amplifier_to_legacy(
    tc: unittest.TestCase,
    amplifier: Amplifier,
    legacy_amplifier: Any,
    *,
    is_raw_assembled: bool,
    expect_nominal_calibrations: bool = True,
) -> None:
    """Compare an `~.cameras.Amplifier` to a legacy
    `lsst.afw.cameraGeom.Amplifier`.
    """
    tc.assertEqual(legacy_amplifier.getName(), amplifier.name)
    tc.assertEqual(Box.from_legacy(legacy_amplifier.getBBox()), amplifier.bbox)
    if is_raw_assembled:
        raw_geom = amplifier.assembled_raw_geometry
    else:
        raw_geom = amplifier.unassembled_raw_geometry
    assert raw_geom is not None
    tc.assertEqual(ReadoutCorner.from_legacy(legacy_amplifier.getReadoutCorner()), raw_geom.readout_corner)
    tc.assertEqual(Box.from_legacy(legacy_amplifier.getRawBBox()), raw_geom.bbox)
    tc.assertEqual(Box.from_legacy(legacy_amplifier.getRawDataBBox()), raw_geom.data_bbox)
    tc.assertEqual(legacy_amplifier.getRawFlipX(), raw_geom.flip_x)
    tc.assertEqual(legacy_amplifier.getRawFlipY(), raw_geom.flip_y)
    tc.assertEqual(legacy_amplifier.getRawXYOffset().getX(), raw_geom.x_offset)
    tc.assertEqual(legacy_amplifier.getRawXYOffset().getY(), raw_geom.y_offset)
    tc.assertEqual(
        Box.from_legacy(legacy_amplifier.getRawHorizontalOverscanBBox()), raw_geom.horizontal_overscan_bbox
    )
    tc.assertEqual(
        Box.from_legacy(legacy_amplifier.getRawVerticalOverscanBBox()), raw_geom.vertical_overscan_bbox
    )
    tc.assertEqual(Box.from_legacy(legacy_amplifier.getRawPrescanBBox()), raw_geom.horizontal_prescan_bbox)
    if expect_nominal_calibrations:
        assert amplifier.nominal_calibrations is not None
        assert_equal_allow_nan(tc, legacy_amplifier.getGain(), amplifier.nominal_calibrations.gain)
        assert_equal_allow_nan(tc, legacy_amplifier.getReadNoise(), amplifier.nominal_calibrations.read_noise)
        assert_equal_allow_nan(
            tc, legacy_amplifier.getSaturation(), amplifier.nominal_calibrations.saturation
        )
        assert_equal_allow_nan(
            tc, legacy_amplifier.getSuspectLevel(), amplifier.nominal_calibrations.suspect_level
        )
        np.testing.assert_array_equal(
            legacy_amplifier.getLinearityCoeffs(), amplifier.nominal_calibrations.linearity_coefficients
        )
        tc.assertEqual(legacy_amplifier.getLinearityType(), amplifier.nominal_calibrations.linearity_type)


def compare_detector_to_legacy(
    tc: unittest.TestCase,
    detector: Detector,
    legacy_detector: Any,
    *,
    is_raw_assembled: bool,
    expect_nominal_calibrations: bool = True,
) -> None:
    """Compare a `~.cameras.Detector` to a `lsst.afw.cameraGeom.Detector`."""
    from lsst.afw.cameraGeom import FIELD_ANGLE, FOCAL_PLANE, PIXELS

    tc.assertEqual(legacy_detector.getName(), detector.name)
    tc.assertEqual(legacy_detector.getId(), detector.id)
    tc.assertEqual(DetectorType.from_legacy(legacy_detector.getType()), detector.type)
    tc.assertEqual(Box.from_legacy(legacy_detector.getBBox()), detector.bbox)
    tc.assertEqual(legacy_detector.getSerial(), detector.serial)
    legacy_orientation = legacy_detector.getOrientation()
    tc.assertEqual(legacy_orientation.getFpPosition3().getX(), detector.orientation.focal_plane_x)
    tc.assertEqual(legacy_orientation.getFpPosition3().getY(), detector.orientation.focal_plane_y)
    tc.assertEqual(legacy_orientation.getFpPosition3().getZ(), detector.orientation.focal_plane_z)
    tc.assertEqual(legacy_orientation.getReferencePoint().getX(), detector.orientation.pixel_reference_x)
    tc.assertEqual(legacy_orientation.getReferencePoint().getY(), detector.orientation.pixel_reference_y)
    tc.assertEqual(legacy_orientation.getYaw().asRadians(), detector.orientation.yaw.to_value(u.rad))
    tc.assertEqual(legacy_orientation.getPitch().asRadians(), detector.orientation.pitch.to_value(u.rad))
    tc.assertEqual(legacy_orientation.getRoll().asRadians(), detector.orientation.roll.to_value(u.rad))
    tc.assertEqual(legacy_detector.getPixelSize().getX(), detector.pixel_size)
    tc.assertEqual(legacy_detector.getPhysicalType(), detector.physical_type)
    for amplifier, legacy_amplifier in zip(detector.amplifiers, legacy_detector.getAmplifiers(), strict=True):
        compare_amplifier_to_legacy(
            tc,
            amplifier,
            legacy_amplifier,
            is_raw_assembled=is_raw_assembled,
            expect_nominal_calibrations=expect_nominal_calibrations,
        )
    pixel_xy = detector.bbox.meshgrid(n=3).map(lambda z: z.ravel().astype(np.float64))
    pixel_legacy_points = arrays_to_legacy_points(y=pixel_xy.y, x=pixel_xy.x)
    fp_legacy_points = legacy_detector.transform(pixel_legacy_points, PIXELS, FOCAL_PLANE)
    check_transform(
        tc,
        detector.to_focal_plane,
        pixel_xy,
        legacy_points_to_xy_array(fp_legacy_points),
        detector.frame,
        detector.to_focal_plane.out_frame,
        in_atol=1e-9 * u.pix,
    )
    fa_legacy_points = legacy_detector.transform(pixel_legacy_points, PIXELS, FIELD_ANGLE)
    check_transform(
        tc,
        detector.to_field_angle,
        pixel_xy,
        legacy_points_to_xy_array(fa_legacy_points),
        detector.frame,
        detector.to_field_angle.out_frame,
        in_atol=1e-9 * u.pix,
    )

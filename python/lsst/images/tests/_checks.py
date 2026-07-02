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
    "assert_cell_coadds_equal",
    "assert_close",
    "assert_equal_allow_nan",
    "assert_images_equal",
    "assert_masked_images_equal",
    "assert_masks_equal",
    "assert_psfs_equal",
    "assert_sky_projections_equal",
    "assert_visit_images_equal",
    "check_archive_tree_class_invariants",
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
    "compare_photo_calib_to_legacy",
    "compare_psf_to_legacy",
    "compare_sky_projection_to_legacy_wcs",
    "compare_visit_image_to_legacy",
    "iter_concrete_archive_tree_subclasses",
    "legacy_coords_to_astropy",
    "legacy_points_to_xy_array",
)

import dataclasses
import math
import re
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any, Literal, cast

import astropy.units as u
import astropy.wcs.wcsapi
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from .._geom import XY, YX, Box
from .._image import Image
from .._mask import Mask, MaskPlane
from .._masked_image import MaskedImage
from .._observation_summary_stats import ObservationSummaryStats
from .._transforms import DetectorFrame, Frame, SkyFrame, SkyProjection, TractFrame, Transform
from .._visit_image import VisitImage
from ..cameras import Amplifier, Detector, DetectorType, ReadoutCorner
from ..cells import CellCoadd, CellIJ, CoaddProvenance
from ..fields import BaseField, ChebyshevField
from ..psfs import PointSpreadFunction
from ..serialization import ArchiveTree

if TYPE_CHECKING:
    try:
        from lsst.cell_coadds import MultipleCellCoadd
    except ImportError:
        type MultipleCellCoadd = Any  # type: ignore[no-redef]
    try:
        from lsst.afw.image import PhotoCalib as LegacyPhotoCalib
    except ImportError:
        type LegacyPhotoCalib = Any  # type: ignore[no-redef]


def assert_close(
    a: np.ndarray | u.Quantity | float,
    b: np.ndarray | u.Quantity | float,
    **kwargs: Any,
) -> None:
    """Test that two arrays, floats, or quantities are almost equal.

    Parameters
    ----------
    a
        Array, float, or quantity to compare.
    b
        Array, float, or quantity to compare.
    **kwargs
        Forwarded to `astropy.units.allclose`.
    """
    assert u.allclose(a, b, **kwargs), f"{a} != {b}"


def assert_equal_allow_nan(a: float, b: float) -> None:
    """Test that two floating point values are equal, with nan == nan.

    Parameters
    ----------
    a
        First value to compare.
    b
        Second value to compare.
    """
    if not (a == b or (math.isnan(a) and math.isnan(b))):
        raise AssertionError(f"{a!r} != {b!r}")


def assert_images_equal(
    a: Image,
    b: Image,
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
    expect_view: bool | Literal["array"] | None = None,
) -> None:
    """Assert that two images are equal or nearly equal.

    Parameters
    ----------
    a
        First image to compare.
    b
        Second image to compare.
    rtol
        Relative tolerance for the pixel comparison.
    atol
        Absolute tolerance for the pixel comparison.
    expect_view
        If not `None`, also assert whether ``b`` shares memory with ``a``
        (i.e. is a view); ``"array"`` checks only the pixel arrays.
    """
    assert a.bbox == b.bbox
    assert a.unit == b.unit
    assert_sky_projections_equal(a.sky_projection, b.sky_projection)
    if expect_view is not None:
        assert np.may_share_memory(a.array, b.array) == bool(expect_view)
        if expect_view == "array":
            assert a.metadata == b.metadata
        else:
            assert (a.metadata is b.metadata) == expect_view
    if not expect_view:
        assert_close(a.array, b.array, atol=atol, rtol=rtol)
        assert a.metadata == b.metadata


def assert_masks_equal(a: Mask, b: Mask) -> None:
    """Assert that two masks are equal or nearly equal.

    Parameters
    ----------
    a
        First mask to compare.
    b
        Second mask to compare.
    """
    assert a.bbox == b.bbox
    assert a.schema == b.schema
    assert a.metadata == b.metadata
    assert_sky_projections_equal(a.sky_projection, b.sky_projection)
    np.testing.assert_array_equal(a.array, b.array)


def assert_masked_images_equal(
    a: MaskedImage,
    b: MaskedImage,
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
    expect_view: bool | None = None,
) -> None:
    """Assert that two masked images are equal or nearly equal.

    Parameters
    ----------
    a
        First masked image to compare.
    b
        Second masked image to compare.
    rtol
        Relative tolerance for the pixel comparison.
    atol
        Absolute tolerance for the pixel comparison.
    expect_view
        If not `None`, also assert whether ``b`` shares memory with ``a``
        (i.e. is a view).
    """
    assert a.metadata == b.metadata
    assert_sky_projections_equal(a.sky_projection, b.sky_projection)
    assert_images_equal(a.image, b.image, rtol=rtol, atol=atol, expect_view=expect_view)
    assert_masks_equal(a.mask, b.mask)
    assert_images_equal(a.variance, b.variance, rtol=rtol, atol=atol, expect_view=expect_view)


def assert_psfs_equal(
    psf1: PointSpreadFunction,
    psf2: PointSpreadFunction,
    points: YX[np.ndarray] | XY[np.ndarray] | None = None,
) -> int:
    """Compare two PSF objets.

    Parameters
    ----------
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
        points = psf1.bounds.bbox.intersection(psf2.bounds.bbox).meshgrid(3).map(np.ravel)

    assert psf1.kernel_bbox == psf2.kernel_bbox

    n_points_tested: int = 0
    for x, y in zip(points.x, points.y):
        # The two PSFs must agree on which points fall inside their input
        # domain.  Querying ``.contains`` directly (rather than relying on
        # ``compute_kernel_image`` to raise) makes this test tolerant of
        # implementations that do not raise on out-of-domain points -- in
        # particular ``CellPointSpreadFunction``, where evaluating in a
        # missing cell does not always raise ``BoundsError``.
        contains1 = psf1.bounds.contains(x=x, y=y)
        contains2 = psf2.bounds.contains(x=x, y=y)
        assert contains1 == contains2, (
            f"PSFs disagree on whether ({x}, {y}) is in-bounds: psf1={contains1}, psf2={contains2}"
        )
        if not contains1:
            continue
        assert psf1.compute_kernel_image(x=x, y=y) == psf2.compute_kernel_image(x=x, y=y)
        assert psf1.compute_stellar_bbox(x=x, y=y) == psf2.compute_stellar_bbox(x=x, y=y)
        assert psf1.compute_stellar_image(x=x, y=y) == psf2.compute_stellar_image(x=x, y=y)
        n_points_tested += 1
    return n_points_tested


def assert_visit_images_equal(
    a: VisitImage,
    b: VisitImage,
    *,
    expect_view: bool | None = None,
) -> None:
    """Assert that two `.VisitImage` instances carry the same persistent state.

    Extends `assert_masked_images_equal` with the VisitImage-specific
    attributes (PSF, filter, observation info, detector, aperture
    corrections, photometric scaling, backgrounds, polygon bounds,
    summary stats) so a round-trip check on a `.VisitImage` does not
    silently miss differences in any of them.

    Parameters
    ----------
    a
        First visit image to compare.
    b
        Second visit image to compare.
    expect_view
        If not `None`, also assert whether ``b`` shares memory with ``a``
        (i.e. is a view).
    """
    assert_masked_images_equal(a, b, expect_view=expect_view)
    assert a.summary_stats == b.summary_stats
    assert a.physical_filter == b.physical_filter
    assert a.band == b.band
    assert a.obs_info == b.obs_info
    assert a.detector == b.detector
    assert dict(a.aperture_corrections) == dict(b.aperture_corrections)
    assert a.photometric_scaling == b.photometric_scaling
    assert dict(a.backgrounds) == dict(b.backgrounds)
    assert a.backgrounds.subtracted == b.backgrounds.subtracted
    assert a.bounds == b.bounds
    assert_psfs_equal(a.psf, b.psf)


def assert_cell_coadds_equal(
    a: CellCoadd,
    b: CellCoadd,
    *,
    expect_view: bool | None = None,
) -> None:
    """Assert that two `.CellCoadd` instances carry the same persistent state.

    Extends the masked-image-style equality check with the
    CellCoadd-specific attributes (PSF, cell grid, missing cells,
    backgrounds, patch/tract, band) so a round-trip check does not
    silently miss differences in any of them.

    Parameters
    ----------
    a
        First cell coadd to compare.
    b
        Second cell coadd to compare.
    expect_view
        If not `None`, also assert whether ``b`` shares memory with ``a``
        (i.e. is a view).
    """
    assert_masked_images_equal(a, b, expect_view=expect_view)
    assert a.band == b.band
    assert a.patch == b.patch
    assert a.tract == b.tract
    assert a.grid == b.grid
    assert a.bounds.missing == b.bounds.missing
    assert dict(a.backgrounds) == dict(b.backgrounds)
    assert a.backgrounds.subtracted == b.backgrounds.subtracted
    assert_psfs_equal(a.psf, b.psf)


def compare_image_to_legacy(image: Image, legacy_image: Any, expect_view: bool | None = None) -> None:
    """Compare an `.Image` object to a legacy `lsst.afw.image.Image` object.

    Parameters
    ----------
    image
        Image to compare.
    legacy_image
        Legacy `lsst.afw.image.Image` to compare against.
    expect_view
        If not `None`, also assert whether ``image`` shares memory with
        ``legacy_image`` (i.e. is a view).
    """
    assert image.bbox == Box.from_legacy(legacy_image.getBBox())
    if expect_view is not None:
        assert np.may_share_memory(image.array, legacy_image.array) == expect_view
    if not expect_view:
        np.testing.assert_array_equal(image.array, legacy_image.array)


def compare_mask_to_legacy(
    mask: Mask, legacy_mask: Any, plane_map: Mapping[str, MaskPlane] | None = None
) -> None:
    """Compare a `.Mask` object to a legacy `lsst.afw.image.Mask` object.

    Parameters
    ----------
    mask
        Mask to compare.
    legacy_mask
        Legacy `lsst.afw.image.Mask` to compare against.
    plane_map
        Mapping from legacy plane name to the new mask plane; defaults to
        the planes in ``mask.schema``.
    """
    assert mask.bbox == Box.from_legacy(legacy_mask.getBBox())
    if plane_map is None:
        plane_map = {plane.name: plane for plane in mask.schema if plane is not None}
    for old_name, new_plane in plane_map.items():
        np.testing.assert_array_equal(
            (legacy_mask.array & legacy_mask.getPlaneBitMask(old_name)).astype(bool),
            mask.get(new_plane.name),
        )


def compare_masked_image_to_legacy(
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
    compare_image_to_legacy(masked_image.image, legacy_masked_image.getImage(), expect_view=expect_view)
    compare_mask_to_legacy(masked_image.mask, legacy_masked_image.getMask(), plane_map=plane_map)
    compare_image_to_legacy(masked_image.variance, legacy_masked_image.getVariance(), expect_view=expect_view)
    if alternates:
        if image := alternates.get("image"):
            compare_image_to_legacy(image, legacy_masked_image.getImage(), expect_view=expect_view)
        if mask := alternates.get("mask"):
            compare_mask_to_legacy(mask, legacy_masked_image.getMask(), plane_map=plane_map)
        if variance := alternates.get("variance"):
            compare_image_to_legacy(variance, legacy_masked_image.getVariance(), expect_view=expect_view)


def compare_visit_image_to_legacy(
    visit_image: VisitImage,
    legacy_exposure: Any,
    *,
    plane_map: Mapping[str, MaskPlane] | None = None,
    expect_view: bool | None = None,
    instrument: str,
    visit: int,
    detector: int,
    applied_legacy_photo_calib: LegacyPhotoCalib | None = None,
    alternates: Mapping[str, Any] | None = None,
) -> None:
    """Compare a `.VisitImage` object to a legacy `lsst.afw.image.Exposure`
    object.

    Parameters
    ----------
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
    applied_legacy_photo_calib
        Legacy `lsst.afw.image.PhotoCalib` already applied to
        ``legacy_exposure``, used when comparing photometric scaling.
    alternates
        A mapping of other versions of one or more (new) components to also
        check against the legacy versions of those components.
    """
    compare_masked_image_to_legacy(
        visit_image,
        legacy_exposure.getMaskedImage(),
        plane_map=plane_map,
        expect_view=expect_view,
        alternates=alternates,
    )
    detector_bbox = Box.from_legacy(legacy_exposure.getDetector().getBBox())
    compare_sky_projection_to_legacy_wcs(
        visit_image.sky_projection,
        legacy_exposure.getWcs(),
        DetectorFrame(instrument=instrument, visit=visit, detector=detector, bbox=detector_bbox),
        visit_image.bbox,
    )
    assert visit_image.sky_projection is visit_image.mask.sky_projection
    assert visit_image.sky_projection is visit_image.variance.sky_projection
    compare_psf_to_legacy(visit_image.psf, legacy_exposure.getPsf())
    compare_observation_summary_stats_to_legacy(
        visit_image.summary_stats, legacy_exposure.info.getSummaryStats()
    )
    compare_detector_to_legacy(visit_image.detector, legacy_exposure.getDetector(), is_raw_assembled=True)
    # Make a tiny box for Field comparisons that need to make arrays; that can
    # get expensive otherwisre.
    tiny_bbox = detector_bbox.local[2:4, 3:6]
    compare_aperture_corrections_to_legacy(
        visit_image.aperture_corrections, legacy_exposure.info.getApCorrMap(), tiny_bbox
    )
    compare_photo_calib_to_legacy(
        visit_image.photometric_scaling,
        legacy_exposure.info.getPhotoCalib(),
        applied_legacy_photo_calib=applied_legacy_photo_calib,
        subimage_bbox=tiny_bbox,
    )
    if alternates:
        if (bbox := alternates.get("bbox")) is not None:
            assert bbox == visit_image.bbox
        if sky_projection := alternates.get("sky_projection"):
            compare_sky_projection_to_legacy_wcs(
                sky_projection,
                legacy_exposure.getWcs(),
                DetectorFrame(instrument=instrument, visit=visit, detector=detector, bbox=detector_bbox),
                visit_image.bbox,
            )
        if psf := alternates.get("psf"):
            compare_psf_to_legacy(psf, legacy_exposure.getPsf())
        if summary_stats := alternates.get("summary_stats"):
            compare_observation_summary_stats_to_legacy(summary_stats, legacy_exposure.info.getSummaryStats())
        if detector_obj := alternates.get("detector"):
            compare_detector_to_legacy(detector_obj, legacy_exposure.getDetector(), is_raw_assembled=True)
        if obs_info := alternates.get("obs_info"):
            visitInfo = legacy_exposure.visitInfo
            assert obs_info.instrument == visitInfo.getInstrumentLabel()
        if aperture_corrections := alternates.get("aperture_corrections"):
            compare_aperture_corrections_to_legacy(
                aperture_corrections, legacy_exposure.info.getApCorrMap(), tiny_bbox
            )
        if (photometric_scaling := alternates.get("photometic_scaling", ...)) is not ...:
            compare_photo_calib_to_legacy(
                photometric_scaling,
                legacy_exposure.info.getPhotoCalib(),
                applied_legacy_photo_calib=applied_legacy_photo_calib,
                subimage_bbox=tiny_bbox,
            )


def compare_photo_calib_to_legacy(
    photometric_scaling: BaseField | None,
    legacy_photo_calib: LegacyPhotoCalib,
    *,
    applied_legacy_photo_calib: LegacyPhotoCalib | None = None,
    subimage_bbox: Box,
) -> None:
    if legacy_photo_calib._isConstant:
        if legacy_photo_calib.getCalibrationMean() == 1.0:
            if applied_legacy_photo_calib is None:
                assert photometric_scaling is None
                return
            else:
                legacy_photo_calib = applied_legacy_photo_calib
    if legacy_photo_calib._isConstant:
        assert isinstance(photometric_scaling, ChebyshevField)
        assert_close(photometric_scaling.coefficients, np.array([[legacy_photo_calib.getCalibrationMean()]]))
    else:
        assert photometric_scaling is not None
        compare_field_to_legacy(
            photometric_scaling / legacy_photo_calib.getCalibrationMean(),
            legacy_photo_calib.computeScaledCalibration(),
            subimage_bbox,
        )


def compare_cell_coadd_to_legacy(
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
    cell_coadd
        New coadd to test.
    legacy_cell_coadd
        Legacy coadd to test against.
    tract_bbox
        Bounding box of the full tract.
    plane_map
        Mapping between new and legacy mask planes.
    alternates
        A mapping of other versions of one or more (new) components to also
        check against the legacy versions of those components.
    psf_points
        Points to use to compare the PSFs.
    """
    legacy_stitched = legacy_cell_coadd.stitch(cell_coadd.bbox.to_legacy())
    compare_image_to_legacy(cell_coadd.image, legacy_stitched.image, expect_view=False)
    compare_mask_to_legacy(cell_coadd.mask, legacy_stitched.mask, plane_map=plane_map)
    compare_image_to_legacy(cell_coadd.variance, legacy_stitched.variance, expect_view=False)
    if legacy_stitched.mask_fractions is not None:
        compare_image_to_legacy(
            cell_coadd.mask_fractions["rejected"], legacy_stitched.mask_fractions, expect_view=False
        )
    for n in range(legacy_stitched.n_noise_realizations):
        compare_image_to_legacy(
            cell_coadd.noise_realizations[n], legacy_stitched.noise_realizations[n], expect_view=False
        )
    assert cell_coadd.skymap == legacy_stitched.identifiers.skymap
    assert cell_coadd.tract == legacy_stitched.identifiers.tract
    assert cell_coadd.patch.index.x == legacy_stitched.identifiers.patch.x
    assert cell_coadd.patch.index.y == legacy_stitched.identifiers.patch.y
    assert cell_coadd.band == legacy_stitched.identifiers.band
    assert tract_bbox.contains(cell_coadd.patch.outer_bbox)
    assert cell_coadd.patch.outer_bbox.contains(cell_coadd.patch.inner_bbox)
    assert cell_coadd.patch.outer_bbox.contains(cell_coadd.bbox)
    assert cell_coadd.unit == u.Unit(legacy_cell_coadd.common.units.value)
    assert cell_coadd.bounds.bbox.contains(cell_coadd.bbox)
    assert cell_coadd.grid.bbox.contains(cell_coadd.bbox)
    compare_sky_projection_to_legacy_wcs(
        cell_coadd.sky_projection,
        legacy_cell_coadd.wcs,
        TractFrame(
            skymap=legacy_cell_coadd.identifiers.skymap,
            tract=legacy_cell_coadd.identifiers.tract,
            bbox=tract_bbox,
        ),
        cell_coadd.bbox,
        is_fits=True,
    )
    assert cell_coadd.sky_projection is cell_coadd.mask.sky_projection
    assert cell_coadd.sky_projection is cell_coadd.variance.sky_projection
    compare_psf_to_legacy(
        cell_coadd.psf, legacy_stitched.psf, expect_legacy_raise_on_out_of_bounds=True, points=psf_points
    )
    compare_aperture_corrections_to_legacy(
        cell_coadd.aperture_corrections, legacy_stitched.ap_corr_map, cell_coadd.bbox
    )
    compare_cell_coadd_provenance_to_legacy(cell_coadd.provenance, legacy_cell_coadd)
    if alternates:
        if sky_projection := alternates.get("sky_projection"):
            compare_sky_projection_to_legacy_wcs(
                sky_projection,
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
            compare_psf_to_legacy(psf, legacy_stitched.psf, points=psf_points)
        if aperture_corrections := alternates.get("aperture_corrections"):
            compare_aperture_corrections_to_legacy(
                aperture_corrections, legacy_stitched.ap_corr_map, cell_coadd.bbox
            )
        if provenance := alternates.get("provenance"):
            compare_cell_coadd_provenance_to_legacy(provenance, legacy_cell_coadd)


def compare_cell_coadd_provenance_to_legacy(
    provenance: CoaddProvenance, legacy_cell_coadd: MultipleCellCoadd
) -> None:
    """Compare a `.cells.CoaddProvenance` object to a legacy
    `lsst.cell_coadds.MultipleCellCoadd` object.

    Parameters
    ----------
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
        assert len(legacy_cell.inputs) == len(prov.inputs)
        assert len(legacy_cell.inputs) == len(prov.contributions)
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
            assert legacy_polygon == row["polygon"].to_legacy()


def compare_psf_to_legacy(
    psf: PointSpreadFunction,
    legacy_psf: Any,
    points: YX[np.ndarray] | XY[np.ndarray] | None = None,
    expect_legacy_raise_on_out_of_bounds: bool = False,
) -> int:
    """Compare a PSF model object to its legacy interface.

    Parameters
    ----------
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
                with pytest.raises(InvalidPsfError):
                    legacy_psf.computeKernelImage(p)
            continue
        assert psf.kernel_bbox == Box.from_legacy(legacy_psf.computeKernelBBox(p))
        assert psf.compute_kernel_image(x=p.x, y=p.y) == Image.from_legacy(legacy_psf.computeKernelImage(p))
        assert psf.compute_stellar_bbox(x=p.x, y=p.y) == Box.from_legacy(legacy_psf.computeImageBBox(p))
        assert psf.compute_stellar_image(x=p.x, y=p.y) == Image.from_legacy(legacy_psf.computeImage(p))
        n_points_tested += 1
    return n_points_tested


def compare_field_to_legacy(
    field: BaseField,
    legacy_field: Any,
    subimage_bbox: Box,
) -> None:
    """Test a Field object by comparing it to an equivalent
    `lsst.afw.math.BoundedField`.

    Parameters
    ----------
    field
        Field to test.
    legacy_field : ``lsst.afw.math.BoundedField``
        Equivalent legacy bounded field.
    subimage_bbox
        Bounding box for full-image tests.
    """
    from lsst.afw.math import BoundedField as LegacyBoundedField

    assert field.bounds.bbox == Box.from_legacy(legacy_field.getBBox())
    # Pixel coordinates to test the numpy array interface with.
    pixel_xy = field.bounds.bbox.meshgrid(n=5).map(np.ravel)
    if not isinstance(field.bounds, Box):
        mask = field.bounds.contains(x=pixel_xy.x, y=pixel_xy.y)
        pixel_xy = pixel_xy.map(lambda v: v[mask])
    try:
        assert_close(
            field(x=pixel_xy.x, y=pixel_xy.y),
            legacy_field.evaluate(pixel_xy.x, pixel_xy.y),
            equal_nan=True,
        )
    except AssertionError as err:
        err.add_note(f"evaluated at {pixel_xy}")
        raise
    if not isinstance(legacy_field, LegacyBoundedField):
        # Legacy StitchedApertureCorrection objects are not true BoundedFields
        # and don't have addToImage.
        return
    legacy_image_1 = Image(0, bbox=subimage_bbox, dtype=np.float64).to_legacy()
    legacy_field.addToImage(legacy_image_1, overlapOnly=True)
    assert_images_equal(
        field.render(subimage_bbox), Image.from_legacy(legacy_image_1, unit=field.unit), rtol=1e-13
    )


def compare_aperture_corrections_to_legacy(
    aperture_corrections: Mapping[str, BaseField],
    legacy_ap_corr_map: Any,
    subimage_bbox: Box,
) -> None:
    """Test an aperture correction `dict` by comparing it to an equivalent
    `lsst.afw.image.ApCorrMap`.

    Parameters
    ----------
    aperture_corrections
        Dictionary to test.
    legacy_ap_corr_map : ``lsst.afw.image.ApCorrMap``
        Equivalent legacy aperture correction map.
    subimage_bbox
        Bounding box for full-image tests.
    """
    assert aperture_corrections.keys() == set(legacy_ap_corr_map.keys())
    for name, field in aperture_corrections.items():
        compare_field_to_legacy(field, legacy_ap_corr_map[name], subimage_bbox)


def compare_observation_summary_stats_to_legacy(
    summary_stats: ObservationSummaryStats,
    legacy_summary_stats: Any,
) -> None:
    """Test an ObservationSummaryStats object by comparing it to an equivalent
    `lsst.afw.image.ExposureSummaryStats`.

    Parameters
    ----------
    summary_stats
        Struct to test.
    legacy_summary_stats : ``lsst.afw.image.ExposureSummaryStats``
        Equivalent legacy struct.
    """
    for field in dataclasses.fields(legacy_summary_stats):
        a = getattr(legacy_summary_stats, field.name)
        b = getattr(summary_stats, field.name)
        if isinstance(b, tuple):
            for ai, bi in zip(a, b):
                assert ai == bi or (math.isnan(ai) and math.isnan(bi)), f"{field.name}: {a} != {b}"
        else:
            assert a == b or (math.isnan(a) and math.isnan(b)), f"{field.name}: {a} != {b}"


def compare_sky_projection_to_legacy_wcs[F: Frame](
    sky_projection: SkyProjection[F],
    legacy_wcs: Any,
    pixel_frame: F,
    subimage_bbox: Box,
    is_fits: bool = False,
) -> None:
    """Test a Projection object by comparing it to an equivalent
    `lsst.afw.geom.SkyWcs`.

    Parameters
    ----------
    sky_projection
        Projection to test.
    legacy_wcs : ``lsst.afw.geom.SkyWcs``
        Equivalent legacy WCS.
    pixel_frame
        Expected pixel frame for the sky_projection.
    subimage_bbox
        Bounding box of points to generate for tests.
    is_fits
        Whether this sky_projection is expected to be exactly representable as
        a FITS WCS. If `False` it is assumed to have a FITS approximation
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
    check_projection(sky_projection, pixel_xy, sky_coords, pixel_frame)
    # Also test the Astropy High-Level WCS view with an origin change to
    # array indices.
    check_astropy_wcs_interface(
        sky_projection.as_astropy(subimage_bbox), subimage_array_xy, sky_coords, pixel_atol=1e-5
    )
    if is_fits:
        fits_wcs = sky_projection.as_fits_wcs(subimage_bbox, allow_approximation=True)
        assert fits_wcs is not None
        check_astropy_wcs_interface(fits_wcs, subimage_array_xy, sky_coords, pixel_atol=1e-5)
        # Use that FITS approximation to check that we can make a
        # Projection from a FITS WCS, too.
        fits_projection = SkyProjection.from_fits_wcs(fits_wcs, pixel_frame)
        check_projection(
            fits_projection,
            subimage_array_xy,
            sky_coords,
            pixel_frame,
            pixel_atol=1e-5,
        )
        # We want Projections we create from a FITS WCS to be backed by an
        # AST FrameSet so we can convert them into legacy
        # `lsst.afw.geom.SkyWcs` objects if desired.
        assert "Begin FrameSet" in fits_projection.show()
    else:
        assert sky_projection.as_fits_wcs(subimage_bbox, allow_approximation=False) is None
        # The legacy SkyWcs should instead have a FITS approximation
        # attached; run the same tests on that.
        assert sky_projection.fits_approximation is not None
        compare_sky_projection_to_legacy_wcs(
            sky_projection.fits_approximation,
            legacy_wcs.getFitsApproximation(),
            pixel_frame,
            subimage_bbox,
            is_fits=True,
        )


def check_transform[I: Frame, O: Frame](
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
    assert transform.in_frame == in_frame
    assert transform.out_frame == out_frame
    in_atol_v = in_atol.to_value(in_frame.unit) if in_atol is not None else None
    out_atol_v = out_atol.to_value(out_frame.unit) if out_atol is not None else None
    # Test array interfaces.
    test_output_xy = transform.apply_forward(x=input_xy.x, y=input_xy.y)
    assert_close(test_output_xy.x, output_xy.x, atol=out_atol_v)
    assert_close(test_output_xy.y, output_xy.y, atol=out_atol_v)
    test_input_xy = transform.apply_inverse(x=output_xy.x, y=output_xy.y)
    assert_close(test_input_xy.x, input_xy.x, atol=in_atol_v)
    assert_close(test_input_xy.y, input_xy.y, atol=in_atol_v)
    # Test scalar interfaces with numpy scalars.
    for input_x, input_y, output_x, output_y in zip(input_xy.x, input_xy.y, output_xy.x, output_xy.y):
        assert_close(transform.apply_forward(x=input_x, y=input_y).x, output_x, atol=out_atol_v)
        assert_close(transform.apply_forward(x=input_x, y=input_y).y, output_y, atol=out_atol_v)
        assert_close(transform.apply_inverse(x=output_x, y=output_y).x, input_x, atol=in_atol_v)
        assert_close(transform.apply_inverse(x=output_x, y=output_y).y, input_y, atol=in_atol_v)
    # Test quantity array interfaces.
    input_q_xy = XY(x=input_xy.x * transform.in_frame.unit, y=input_xy.y * transform.in_frame.unit)
    output_q_xy = XY(x=output_xy.x * transform.out_frame.unit, y=output_xy.y * transform.out_frame.unit)
    test_output_q_xy = transform.apply_forward_q(x=input_q_xy.x, y=input_q_xy.y)
    assert_close(test_output_q_xy.x, output_q_xy.x, atol=out_atol)
    assert_close(test_output_q_xy.y, output_q_xy.y, atol=out_atol)
    test_input_q_xy = transform.apply_inverse_q(x=output_q_xy.x, y=output_q_xy.y)
    assert_close(test_input_q_xy.x, input_q_xy.x, atol=in_atol)
    assert_close(test_input_q_xy.y, input_q_xy.y, atol=in_atol)
    # Test quantity scalar interfaces.
    for input_q_x, input_q_y, output_q_x, output_q_y in zip(
        input_q_xy.x, input_q_xy.y, output_q_xy.x, output_q_xy.y
    ):
        assert_close(transform.apply_forward_q(x=input_q_x, y=input_q_y).x, output_q_x, atol=out_atol)
        assert_close(transform.apply_forward_q(x=input_q_x, y=input_q_y).y, output_q_y, atol=out_atol)
        assert_close(transform.apply_inverse_q(x=output_q_x, y=output_q_y).x, input_q_x, atol=in_atol)
        assert_close(transform.apply_inverse_q(x=output_q_x, y=output_q_y).y, input_q_y, atol=in_atol)
    if check_inverted:
        # Test the inverse transform.
        check_transform(
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
    sky_projection: SkyProjection[P],
    pixel_xy: XY[np.ndarray],
    sky_coords: SkyCoord,
    pixel_frame: Frame,
    *,
    pixel_atol: float | None = None,
    sky_atol: u.Quantity | None = None,
) -> None:
    """Test a `.SkyProjection` instance against known arrays of pixel and sky
    coordinates.

    Parameters
    ----------
    sky_projection
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
    assert sky_projection.pixel_frame == pixel_frame
    assert sky_projection.sky_frame == SkyFrame.ICRS
    sky_atol_v = sky_atol.to_value(SkyFrame.ICRS.unit) if sky_atol is not None else None
    pixel_atol_q = pixel_atol * u.pix if pixel_atol is not None else None
    # Test array interfaces.
    test_pixel_xy = cast(XY[np.ndarray], sky_projection.sky_to_pixel(sky_coords))
    assert_close(test_pixel_xy.x, pixel_xy.x, atol=pixel_atol)
    assert_close(test_pixel_xy.y, pixel_xy.y, atol=pixel_atol)
    test_sky_astropy = sky_projection.pixel_to_sky(x=pixel_xy.x, y=pixel_xy.y)
    assert_close(test_sky_astropy.ra, sky_coords.ra, atol=sky_atol_v)
    assert_close(test_sky_astropy.dec, sky_coords.dec, atol=sky_atol_v)
    # Test scalar interfaces.
    for pixel_x, pixel_y, sky_single in zip(pixel_xy.x, pixel_xy.y, sky_coords):
        assert_close(sky_projection.sky_to_pixel(sky_single).x, pixel_x, atol=pixel_atol)
        assert_close(sky_projection.sky_to_pixel(sky_single).y, pixel_y, atol=pixel_atol)
        assert_close(sky_projection.pixel_to_sky(x=pixel_x, y=pixel_y).ra, sky_single.ra, atol=sky_atol_v)
        assert_close(sky_projection.pixel_to_sky(x=pixel_x, y=pixel_y).dec, sky_single.dec, atol=sky_atol_v)
    # Test the underlying Transform object.
    sky_xy = XY(x=sky_coords.ra.to_value(u.rad), y=sky_coords.dec.to_value(u.rad))
    check_transform(
        sky_projection.pixel_to_sky_transform,
        pixel_xy,
        sky_xy,
        pixel_frame,
        SkyFrame.ICRS,
        check_inverted=False,
        in_atol=pixel_atol_q,
        out_atol=sky_atol,
    )
    check_transform(
        sky_projection.sky_to_pixel_transform,
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
        sky_projection.as_astropy(), pixel_xy, sky_coords, pixel_atol=pixel_atol, sky_atol=sky_atol
    )


def assert_sky_projections_equal(
    a: SkyProjection[Any] | None,
    b: SkyProjection[Any] | None,
    expect_identity: bool | None = None,
) -> None:
    """Test that two `.SkyProjection` instances are equivalent.

    Parameters
    ----------
    a
        First sky projection to compare.
    b
        Second sky projection to compare.
    expect_identity
        If not `None`, assert whether ``a`` and ``b`` are the same object.
    """
    if a is None and b is None:
        return
    assert a is not None and b is not None
    match expect_identity:
        case True:
            assert a is b
            return
        case False:
            assert a is not b
        case None if a is b:
            return
    assert a == b


def check_astropy_wcs_interface(
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
    assert_close(test_x, pixel_xy.x, atol=pixel_atol)
    assert_close(test_y, pixel_xy.y, atol=pixel_atol)
    test_sky_coords = wcs.pixel_to_world(pixel_xy.x, pixel_xy.y)
    assert_close(test_sky_coords.ra, sky_coords.ra, atol=sky_atol)
    assert_close(test_sky_coords.dec, sky_coords.dec, atol=sky_atol)


def legacy_points_to_xy_array(legacy_points: list[Any]) -> XY[np.ndarray]:
    """Convert a list of ``lsst.geom.Point2D`` objects to an `.XY` array.

    Parameters
    ----------
    legacy_points
        Legacy ``lsst.geom.Point2D`` objects to convert.
    """
    return XY(x=np.array([p.x for p in legacy_points]), y=np.array([p.y for p in legacy_points]))


def legacy_coords_to_astropy(legacy_coords: list[Any]) -> SkyCoord:
    """Convert a list of ``lsst.geom.SpherePoint`` objects to an Astropy
    coordinate object.

    Parameters
    ----------
    legacy_coords
        Legacy ``lsst.geom.SpherePoint`` objects to convert.
    """
    return SkyCoord(
        ra=np.array([p.getRa().asRadians() for p in legacy_coords]) * u.rad,
        dec=np.array([p.getDec().asRadians() for p in legacy_coords]) * u.rad,
    )


def arrays_to_legacy_points(x: np.ndarray, y: np.ndarray) -> list[Any]:
    """Convert arrays of ``x`` and ``y`` to a list of ``lsst.geom.Point2D``.

    Parameters
    ----------
    x
        X coordinates of the points.
    y
        Y coordinates of the points.
    """
    from lsst.geom import Point2D

    return [Point2D(x=xv, y=yv) for xv, yv in zip(x, y)]


def compare_amplifier_to_legacy(
    amplifier: Amplifier,
    legacy_amplifier: Any,
    *,
    is_raw_assembled: bool,
    expect_nominal_calibrations: bool = True,
) -> None:
    """Compare an `~.cameras.Amplifier` to a legacy
    `lsst.afw.cameraGeom.Amplifier`.

    Parameters
    ----------
    amplifier
        Amplifier to compare.
    legacy_amplifier
        Legacy `lsst.afw.cameraGeom.Amplifier` to compare against.
    is_raw_assembled
        Whether the raw geometry is expected to be the assembled-raw
        geometry (`True`) or the unassembled-raw geometry (`False`).
    expect_nominal_calibrations
        Whether the amplifier is expected to carry nominal calibrations.
    """
    assert legacy_amplifier.getName() == amplifier.name
    assert Box.from_legacy(legacy_amplifier.getBBox()) == amplifier.bbox
    if is_raw_assembled:
        raw_geom = amplifier.assembled_raw_geometry
    else:
        raw_geom = amplifier.unassembled_raw_geometry
    assert raw_geom is not None
    assert ReadoutCorner.from_legacy(legacy_amplifier.getReadoutCorner()) == raw_geom.readout_corner
    assert Box.from_legacy(legacy_amplifier.getRawBBox()) == raw_geom.bbox
    assert Box.from_legacy(legacy_amplifier.getRawDataBBox()) == raw_geom.data_bbox
    assert legacy_amplifier.getRawFlipX() == raw_geom.flip_x
    assert legacy_amplifier.getRawFlipY() == raw_geom.flip_y
    assert legacy_amplifier.getRawXYOffset().getX() == raw_geom.x_offset
    assert legacy_amplifier.getRawXYOffset().getY() == raw_geom.y_offset
    assert (
        Box.from_legacy(legacy_amplifier.getRawHorizontalOverscanBBox()) == raw_geom.horizontal_overscan_bbox
    )
    assert Box.from_legacy(legacy_amplifier.getRawVerticalOverscanBBox()) == raw_geom.vertical_overscan_bbox
    assert Box.from_legacy(legacy_amplifier.getRawPrescanBBox()) == raw_geom.horizontal_prescan_bbox
    if expect_nominal_calibrations:
        assert amplifier.nominal_calibrations is not None
        assert_equal_allow_nan(legacy_amplifier.getGain(), amplifier.nominal_calibrations.gain)
        assert_equal_allow_nan(legacy_amplifier.getReadNoise(), amplifier.nominal_calibrations.read_noise)
        assert_equal_allow_nan(legacy_amplifier.getSaturation(), amplifier.nominal_calibrations.saturation)
        assert_equal_allow_nan(
            legacy_amplifier.getSuspectLevel(), amplifier.nominal_calibrations.suspect_level
        )
        np.testing.assert_array_equal(
            legacy_amplifier.getLinearityCoeffs(), amplifier.nominal_calibrations.linearity_coefficients
        )
        assert legacy_amplifier.getLinearityType() == amplifier.nominal_calibrations.linearity_type


def compare_detector_to_legacy(
    detector: Detector,
    legacy_detector: Any,
    *,
    is_raw_assembled: bool,
    expect_nominal_calibrations: bool = True,
) -> None:
    """Compare a `~.cameras.Detector` to a `lsst.afw.cameraGeom.Detector`.

    Parameters
    ----------
    detector
        Detector to compare.
    legacy_detector
        Legacy `lsst.afw.cameraGeom.Detector` to compare against.
    is_raw_assembled
        Whether the raw geometry is expected to be the assembled-raw
        geometry (`True`) or the unassembled-raw geometry (`False`).
    expect_nominal_calibrations
        Whether the detector's amplifiers are expected to carry nominal
        calibrations.
    """
    from lsst.afw.cameraGeom import FIELD_ANGLE, FOCAL_PLANE, PIXELS

    assert legacy_detector.getName() == detector.name
    assert legacy_detector.getId() == detector.id
    assert DetectorType.from_legacy(legacy_detector.getType()) == detector.type
    assert Box.from_legacy(legacy_detector.getBBox()) == detector.bbox
    assert legacy_detector.getSerial() == detector.serial
    legacy_orientation = legacy_detector.getOrientation()
    assert legacy_orientation.getFpPosition3().getX() == detector.orientation.focal_plane_x
    assert legacy_orientation.getFpPosition3().getY() == detector.orientation.focal_plane_y
    assert legacy_orientation.getFpPosition3().getZ() == detector.orientation.focal_plane_z
    assert legacy_orientation.getReferencePoint().getX() == detector.orientation.pixel_reference_x
    assert legacy_orientation.getReferencePoint().getY() == detector.orientation.pixel_reference_y
    assert legacy_orientation.getYaw().asRadians() == detector.orientation.yaw.to_value(u.rad)
    assert legacy_orientation.getPitch().asRadians() == detector.orientation.pitch.to_value(u.rad)
    assert legacy_orientation.getRoll().asRadians() == detector.orientation.roll.to_value(u.rad)
    assert legacy_detector.getPixelSize().getX() == detector.pixel_size
    assert legacy_detector.getPhysicalType() == detector.physical_type
    for amplifier, legacy_amplifier in zip(detector.amplifiers, legacy_detector.getAmplifiers(), strict=True):
        compare_amplifier_to_legacy(
            amplifier,
            legacy_amplifier,
            is_raw_assembled=is_raw_assembled,
            expect_nominal_calibrations=expect_nominal_calibrations,
        )
    pixel_xy = detector.bbox.meshgrid(n=3).map(lambda z: z.ravel().astype(np.float64))
    pixel_legacy_points = arrays_to_legacy_points(y=pixel_xy.y, x=pixel_xy.x)
    fp_legacy_points = legacy_detector.transform(pixel_legacy_points, PIXELS, FOCAL_PLANE)
    check_transform(
        detector.to_focal_plane,
        pixel_xy,
        legacy_points_to_xy_array(fp_legacy_points),
        detector.frame,
        detector.to_focal_plane.out_frame,
        in_atol=1e-9 * u.pix,
        out_atol=1e-7 * detector.to_focal_plane.out_frame.unit,
    )
    fa_legacy_points = legacy_detector.transform(pixel_legacy_points, PIXELS, FIELD_ANGLE)
    check_transform(
        detector.to_field_angle,
        pixel_xy,
        legacy_points_to_xy_array(fa_legacy_points),
        detector.frame,
        detector.to_field_angle.out_frame,
        in_atol=1e-9 * u.pix,
        out_atol=1e-7 * u.arcsec,
    )


def iter_concrete_archive_tree_subclasses() -> Iterator[type[ArchiveTree]]:
    """Yield every importable concrete `.serialization.ArchiveTree` subclass.

    Walks the ``ArchiveTree.__subclasses__()`` tree, skipping abstract
    classes.  Importing this module already imports every ``lsst.images``
    module that defines a subclass, so the tree is fully populated by the time
    this is called.

    This discovery is deliberately separate from
    `check_archive_tree_class_invariants` so that the per-class check stays
    usable on a single class even if this metaprogramming is removed later.
    """
    seen: set[type] = set()
    stack: list[type] = [ArchiveTree]
    while stack:
        kls = stack.pop()
        for sub in kls.__subclasses__():
            if sub in seen:
                continue
            seen.add(sub)
            stack.append(sub)
            if not getattr(sub, "__abstractmethods__", None):
                yield sub


def check_archive_tree_class_invariants(cls: type[ArchiveTree]) -> None:
    """Assert that one concrete `.serialization.ArchiveTree` subclass declares
    well-formed schema-version constants and an in-memory type.

    Checks that ``SCHEMA_NAME``, ``SCHEMA_VERSION``, ``MIN_READ_VERSION`` and
    ``PUBLIC_TYPE`` are present and well-typed, that the version is
    ``major.minor.patch``, and that ``MIN_READ_VERSION`` does not exceed the
    schema major.

    Parameters
    ----------
    cls
        The concrete `.serialization.ArchiveTree` subclass to check.
    """
    assert hasattr(cls, "SCHEMA_NAME"), f"{cls.__name__} lacks SCHEMA_NAME"
    assert hasattr(cls, "SCHEMA_VERSION"), f"{cls.__name__} lacks SCHEMA_VERSION"
    assert hasattr(cls, "MIN_READ_VERSION"), f"{cls.__name__} lacks MIN_READ_VERSION"
    assert hasattr(cls, "PUBLIC_TYPE"), f"{cls.__name__} lacks PUBLIC_TYPE"
    assert isinstance(cls.SCHEMA_NAME, str)
    assert len(cls.SCHEMA_NAME) > 0
    assert re.fullmatch(r"^\d+\.\d+\.\d+$", cls.SCHEMA_VERSION)
    assert isinstance(cls.MIN_READ_VERSION, int)
    assert cls.MIN_READ_VERSION >= 1
    assert isinstance(cls.PUBLIC_TYPE, type)
    major = int(cls.SCHEMA_VERSION.split(".")[0])
    assert cls.MIN_READ_VERSION <= major

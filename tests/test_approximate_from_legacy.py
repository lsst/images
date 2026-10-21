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

"""Unit tests for the missing/partial-cell sampling rules of conversions
from legacy (non-cell) coadd components.
"""

from __future__ import annotations

import dataclasses
import os
import pickle
import types
from collections.abc import Iterable
from typing import Any

import numpy as np
import pytest

from lsst.images import XY, YX, BoundsError, Box, Image, SkyProjection, TractFrame
from lsst.images.cells import (
    CellCoadd,
    CellField,
    CellGridBounds,
    CellIJ,
    CellPointSpreadFunction,
    CoaddProvenance,
    PatchDefinition,
)
from lsst.images.tests import (
    DP2_COADD_DATA_ID,
    assert_values_equal,
    compare_cell_coadd_provenance_to_legacy_lazy,
    reset_afw_mask_planes,  # noqa: F401
)

try:
    from lsst.afw.detection import InvalidPsfError
    from lsst.afw.geom import makeCdMatrix, makeSkyWcs
    from lsst.afw.image import Exposure as LegacyExposure
    from lsst.afw.image import ExposureFitsReader
    from lsst.afw.math import ChebyshevBoundedField as LegacyChebyshevBoundedField
    from lsst.afw.table import ExposureCatalog as LegacyExposureCatalog
    from lsst.geom import Box2D as LegacyBox2D
    from lsst.geom import Box2I as LegacyBox2I
    from lsst.geom import Point2I as LegacyPoint2I
    from lsst.meas.algorithms import CoaddBoundedField as LegacyCoaddBoundedField
    from lsst.meas.algorithms import CoaddBoundedFieldElement as LegacyCoaddBoundedFieldElement
    from lsst.meas.algorithms import CoaddPsf as LegacyCoaddPsf
    from lsst.pex.exceptions import DomainError
    from lsst.skymap import TractInfo as LegacyTractInfo

    HAVE_LEGACY = True
except ImportError:
    type LegacyChebyshevBoundedField = Any  # type: ignore[no-redef]
    type LegacyExposure = Any  # type: ignore[no-redef]
    type LegacyExposureCatalog = Any  # type: ignore[no-redef]
    type LegacyBox2I = Any  # type: ignore[no-redef]
    type LegacyPoint2I = Any  # type: ignore[no-redef]
    type LegacyCoaddBoundedFieldElement = Any  # type: ignore[no-redef]
    type LegacyCoaddBoundedField = Any  # type: ignore[no-redef]
    type LegacyCoaddPsf = Any  # type: ignore[no-redef]
    type LegacyTractInfo = Any  # type: ignore[no-redef]
    HAVE_LEGACY = False


EXTERNAL_DATA_DIR = os.environ.get("TESTDATA_IMAGES_DIR", None)

# Index into the sparse cell's ``Box.boundary()`` corners whose overlapping
# inputs are also (in addition to the center's inputs).
PARTIAL_CORNER_INDEX = 0
# Number of the sparse cell's corners that stay valid under ``trim_partial`` (4
# corners minus the dropped one).
PARTIAL_N_VALID_CORNERS = 3


@dataclasses.dataclass(frozen=True)
class _TrimmedLegacyCoaddComponents:
    """Legacy coadd components rebuilt without the inputs covering a point."""

    ccds: LegacyExposureCatalog
    psf: LegacyCoaddPsf
    ap_corrections: dict[str, LegacyCoaddBoundedField]
    dropped: frozenset[tuple[int, int]]  # dropped (visit, detector) pairs


@dataclasses.dataclass(frozen=True)
class _LegacyTestCoadd:
    """Test data loaded from EXTERNAL_DATA_DIR."""

    coadd: LegacyExposure
    tract_info: LegacyTractInfo
    patch: PatchDefinition
    bounds: CellGridBounds

    @property
    def sky_projection(self) -> SkyProjection[TractFrame]:
        """The coadd's sky projection in its tract frame."""
        tract_bbox = Box.from_legacy(self.tract_info.getBBox())
        return SkyProjection.from_legacy(
            self.coadd.getWcs(),
            TractFrame(
                skymap=DP2_COADD_DATA_ID["skymap"],
                tract=DP2_COADD_DATA_ID["tract"],
                bbox=tract_bbox,
            ),
            pixel_bounds=tract_bbox,
        )

    @property
    def sparse_cell_index(self) -> CellIJ:
        """Index of the sparse cell that has an invalid corner or center,
        depending on the test.
        """
        return next(iter(self.bounds.cell_indices()))

    @property
    def sparse_cell_box(self) -> Box:
        """Pixel bounding box of the sparse cell."""
        return self.patch.cells.bbox_of(self.sparse_cell_index)

    @property
    def control_cell_index(self) -> CellIJ:
        """Index of a control cell: a regular cell fully covered by inputs."""
        return self.sparse_cell_index + CellIJ(i=0, j=1)

    @property
    def control_cell_box(self) -> Box:
        """Pixel bounding box of the control cell."""
        return self.patch.cells.bbox_of(self.control_cell_index)

    def trim(
        self,
        *points: XY[float] | YX[int],
        ap_corr_names: Iterable[str] = (
            "base_PsfFlux_instFlux",
            "base_PsfFlux_instFluxErr",
            "base_GaussianFlux_instFluxErr",
            "ext_convolved_ConvolvedFlux_0_3_3_instFlux",
        ),
    ) -> _TrimmedLegacyCoaddComponents:
        """Trim the original coadd's inputs to leave some of the sparse
        cell uncovered.

        Parameters
        ----------
        *points
            Points that should be left uncovered by inputs.
        ap_corr_names
            Aperture correction fields to propagate (we don't need them all
            for testing).
        """
        coadd_wcs = self.coadd.getWcs()

        def _element_covers(element: LegacyCoaddBoundedFieldElement, point: XY[float] | YX[int]) -> bool:
            """Test whether one visit-level ApCorr covers a coadd point."""
            projected = element.wcs.skyToPixel(coadd_wcs.pixelToSky(point.to_legacy_float_point()))
            # Box2D.contains rejects NaN projections (distant elements outside
            # the TAN domain project here as NaN).
            return LegacyBox2D(element.field.getBBox()).contains(projected) and element.validPolygon.contains(
                projected
            )

        ccds = self.coadd.info.getCoaddInputs().ccds
        dropped: set[tuple[int, int]] = set()
        for point in points:
            for record in ccds.subsetContaining(
                point.to_legacy_float_point(), coadd_wcs, includeValidPolygon=True
            ):
                dropped.add((record.get("visit"), record.get("ccd")))
        trimmed = ccds.copy()
        # 'del catalog[i]' only supports one row at a time; drop from the back.
        drop_rows = sorted(
            i for i, record in enumerate(ccds) if (record.get("visit"), record.get("ccd")) in dropped
        )
        for i in reversed(drop_rows):
            del trimmed[i]
        psf = LegacyCoaddPsf(trimmed, coadd_wcs)
        fields: dict[str, Any] = {}
        for name in ap_corr_names:
            stored = self.coadd.info.getApCorrMap()[name]
            keep = [
                element
                for element in stored.getElements()
                if not any(_element_covers(element, point) for point in points)
            ]
            assert keep, f"every element of {name} covers the trimmed points"
            field = LegacyCoaddBoundedField(stored.getBBox(), coadd_wcs, keep)
            # Check that the default throw behavior is what other tests expect
            # so we don't miss out on coverage of both modes.
            assert field.getThrowOnMissing(), f"{name} does not throw at uncovered points"
            fields[name] = field
        return _TrimmedLegacyCoaddComponents(
            ccds=trimmed, psf=psf, ap_corrections=fields, dropped=frozenset(dropped)
        )


@pytest.fixture
def legacy_test_coadd(reset_afw_mask_planes: None) -> _LegacyTestCoadd:  # noqa: F811
    """Return the DP2 template_coadd and its derived cell bounds."""
    if not HAVE_LEGACY:
        pytest.skip("Legacy packages (lsst.afw, etc) could not be imported.")
    if EXTERNAL_DATA_DIR is None:
        pytest.skip("TESTDATA_IMAGES_DIR is not in the environment.")
    filename = os.path.join(EXTERNAL_DATA_DIR, "dp2", "legacy", "template_coadd.fits")
    coadd = ExposureFitsReader(filename).read()
    filename = os.path.join(EXTERNAL_DATA_DIR, "dp2", "legacy", "skyMap.pickle")
    with open(filename, "rb") as stream:
        skymap = pickle.load(stream)
    tract_info = skymap[DP2_COADD_DATA_ID["tract"]]
    patch = PatchDefinition.from_legacy_patch_info(tract_info[DP2_COADD_DATA_ID["patch"]])
    # Cell bounds over the four sparsest cells of the patch (2x2 region).
    # Tract-frame origin of the sparsest cell of patch 43: its center is
    # covered by only 20 of the fixture's 103 inputs, while all four outermost
    # pixel centers keep coverage from at least one other input.
    region = Box.from_shape(patch.cells.cell_shape.map(lambda s: 2 * s), start=YX(y=12450, x=9600))
    bounds = CellGridBounds(grid=patch.cells, bbox=patch.cells.bbox)[region]
    return _LegacyTestCoadd(coadd=coadd, tract_info=tract_info, patch=patch, bounds=bounds)


@pytest.fixture
def trim_missing(legacy_test_coadd: _LegacyTestCoadd) -> _TrimmedLegacyCoaddComponents:
    """Components with all inputs covering the sparse cell dropped."""
    return legacy_test_coadd.trim(
        legacy_test_coadd.sparse_cell_box.center, *legacy_test_coadd.sparse_cell_box.boundary()
    )


@pytest.fixture
def trim_partial(legacy_test_coadd: _LegacyTestCoadd) -> _TrimmedLegacyCoaddComponents:
    """Components with the inputs covering the sparse cell's center and one
    corner dropped, leaving the other three corners valid.
    """
    return legacy_test_coadd.trim(
        legacy_test_coadd.sparse_cell_box.center,
        list(legacy_test_coadd.sparse_cell_box.boundary())[PARTIAL_CORNER_INDEX],
    )


def test_trim_missing_fixture(
    legacy_test_coadd: _LegacyTestCoadd, trim_missing: _TrimmedLegacyCoaddComponents
) -> None:
    """Test that the `trim_missing` test fixture does what it claims."""
    sparse_box = legacy_test_coadd.sparse_cell_box
    center = sparse_box.center.to_legacy_float_point()
    corners = [corner.to_legacy_float_point() for corner in sparse_box.boundary()]
    for point in (center, *corners):
        with pytest.raises(InvalidPsfError):
            trim_missing.psf.computeKernelImage(point)
        for field in trim_missing.ap_corrections.values():
            with pytest.raises(DomainError):
                field.evaluate(point)
    other_center = legacy_test_coadd.control_cell_box.center.to_legacy_float_point()
    trim_missing.psf.computeKernelImage(other_center)  # must not raise
    for field in trim_missing.ap_corrections.values():
        assert np.isfinite(field.evaluate(other_center))


def test_trim_partial_fixture(
    legacy_test_coadd: _LegacyTestCoadd, trim_partial: _TrimmedLegacyCoaddComponents
) -> None:
    """Test that the `trim_partial` fixture does what it claims."""
    sparse_box = legacy_test_coadd.sparse_cell_box
    corners = [corner.to_legacy_float_point() for corner in sparse_box.boundary()]
    for point in (sparse_box.center.to_legacy_float_point(), corners[PARTIAL_CORNER_INDEX]):
        with pytest.raises(InvalidPsfError):
            trim_partial.psf.computeKernelImage(point)
        for field in trim_partial.ap_corrections.values():
            with pytest.raises(DomainError):
                field.evaluate(point)
    valid_others = []
    for k, corner in enumerate(corners):
        if k == PARTIAL_CORNER_INDEX:
            continue
        try:
            trim_partial.psf.computeKernelImage(corner)
            finite = all(
                np.isfinite(field.evaluate(corner)) for field in trim_partial.ap_corrections.values()
            )
        except (InvalidPsfError, DomainError):
            finite = False
        if finite:
            valid_others.append(k)
    # Exact count and identity: exactly the three corners other than the
    # dropped one stay valid (the partial-corner regime the averaging tests
    # below rely on).
    assert valid_others == [k for k in range(len(corners)) if k != PARTIAL_CORNER_INDEX]
    assert len(valid_others) == PARTIAL_N_VALID_CORNERS
    other_center = legacy_test_coadd.control_cell_box.center.to_legacy_float_point()
    trim_partial.psf.computeKernelImage(other_center)  # must not raise
    for field in trim_partial.ap_corrections.values():
        assert np.isfinite(field.evaluate(other_center))


def test_psf_valid_cells_store_center_stamps(
    legacy_test_coadd: _LegacyTestCoadd, trim_missing: _TrimmedLegacyCoaddComponents
) -> None:
    """Test that where the center is valid, the stored stamp is exactly the
    legacy kernel image at the center (zero-padded into the shared kernel
    bbox).
    """
    psf = CellPointSpreadFunction.approximate_from_legacy(trim_missing.psf, legacy_test_coadd.bounds)
    stamp = Image.from_legacy(
        trim_missing.psf.computeKernelImage(legacy_test_coadd.control_cell_box.center.to_legacy_float_point())
    )
    expected = np.zeros(psf.kernel_bbox.shape, dtype=np.float64)
    slices = stamp.bbox.slice_within(psf.kernel_bbox)
    expected[slices.y, slices.x] = stamp.array
    assert psf[legacy_test_coadd.control_cell_index] == Image(expected, bbox=psf.kernel_bbox)


def test_psf_partial_corner_fallback_averages_valid_corners(
    legacy_test_coadd: _LegacyTestCoadd, trim_partial: _TrimmedLegacyCoaddComponents
) -> None:
    """Test that the cell whose center and one corner are invalid stores the
    mean of only the valid corner stamps.
    """
    psf = CellPointSpreadFunction.approximate_from_legacy(trim_partial.psf, legacy_test_coadd.bounds)
    assert legacy_test_coadd.sparse_cell_index not in psf.bounds.missing
    corners = [corner.to_legacy_float_point() for corner in legacy_test_coadd.sparse_cell_box.boundary()]
    total = np.zeros(psf.kernel_bbox.shape, dtype=np.float64)
    n_valid = 0
    for k, corner in enumerate(corners):
        if k == PARTIAL_CORNER_INDEX:
            continue
        stamp = Image.from_legacy(trim_partial.psf.computeKernelImage(corner))
        slices = stamp.bbox.slice_within(psf.kernel_bbox)
        total[slices.y, slices.x] += stamp.array
        n_valid += 1
    assert n_valid == PARTIAL_N_VALID_CORNERS
    expected = total / n_valid
    stored = psf[legacy_test_coadd.sparse_cell_index]
    assert stored.bbox == psf.kernel_bbox
    # Bitwise: the conversion sums the same valid stamps in the same
    # boundary() order, then divides in place by the number of valid stamps.
    assert np.array_equal(stored.array, expected)


def test_psf_missing_cells_marked_and_nan(
    legacy_test_coadd: _LegacyTestCoadd, trim_missing: _TrimmedLegacyCoaddComponents
) -> None:
    """Test that a fully-invalid cell is added to bounds.missing and its
    array slice is NaN (the documented convention), and that pre-existing
    missing cells survive the bounds reconstruction.
    """
    pre_existing = CellGridBounds(
        grid=legacy_test_coadd.bounds.grid,
        bbox=legacy_test_coadd.bounds.bbox,
        missing=frozenset({legacy_test_coadd.control_cell_index}),
    )
    psf = CellPointSpreadFunction.approximate_from_legacy(trim_missing.psf, pre_existing)
    assert legacy_test_coadd.sparse_cell_index in psf.bounds.missing
    assert legacy_test_coadd.control_cell_index in psf.bounds.missing
    # We have to access the private array to see that it's NaN (public
    # interface
    # raises).
    index = legacy_test_coadd.sparse_cell_index - psf.bounds.subgrid_start
    assert np.isnan(psf._array[index.i, index.j]).all()


def test_psf_raises_when_no_cell_is_valid(
    legacy_test_coadd: _LegacyTestCoadd, trim_missing: _TrimmedLegacyCoaddComponents
) -> None:
    """Test that conversion of a region with no valid sample anywhere is an
    error, not an all-missing result.
    """
    single = legacy_test_coadd.bounds[
        legacy_test_coadd.bounds.grid.bbox_of(legacy_test_coadd.sparse_cell_index)
    ]
    with pytest.raises(ValueError, match="could not be evaluated"):
        CellPointSpreadFunction.approximate_from_legacy(trim_missing.psf, single)


def test_field_valid_cells_store_center_values(
    legacy_test_coadd: _LegacyTestCoadd, trim_missing: _TrimmedLegacyCoaddComponents
) -> None:
    """Test that where the center is valid, the stored value is exactly the
    legacy evaluation at the center.
    """
    for name, legacy_field in trim_missing.ap_corrections.items():
        field = CellField.approximate_from_legacy(legacy_field, legacy_test_coadd.bounds)
        assert field.value_in_cell(legacy_test_coadd.control_cell_index) == legacy_field.evaluate(
            legacy_test_coadd.control_cell_box.center.to_legacy_float_point()
        ), name


def test_field_partial_corner_fallback_averages_valid_only(
    legacy_test_coadd: _LegacyTestCoadd, trim_partial: _TrimmedLegacyCoaddComponents
) -> None:
    """Test that the cell whose center and one corner are invalid stores the
    mean of only the valid corner evaluations.  A mean over the valid samples
    differs from a sum over them divided by four, so this pins the divisor
    against a regression to the corner count.
    """
    cell_a = legacy_test_coadd.sparse_cell_index
    corners = [corner.to_legacy_float_point() for corner in legacy_test_coadd.sparse_cell_box.boundary()]
    for name, legacy_field in trim_partial.ap_corrections.items():
        field = CellField.approximate_from_legacy(legacy_field, legacy_test_coadd.bounds)
        assert cell_a not in field.bounds.missing, name
        values = [
            legacy_field.evaluate(corner) for k, corner in enumerate(corners) if k != PARTIAL_CORNER_INDEX
        ]
        assert len(values) == PARTIAL_N_VALID_CORNERS
        assert all(np.isfinite(values)), name
        assert_values_equal(
            field.value_in_cell(cell_a),
            float(np.mean(values)),
            rtol=1e-13,
            label=f"partial fallback cell value [{name}]",
        )


def test_field_missing_cells_marked_and_nan(
    legacy_test_coadd: _LegacyTestCoadd, trim_missing: _TrimmedLegacyCoaddComponents
) -> None:
    """Test that a cell where center and all corners raise is marked missing,
    its array entry is NaN (the documented convention), and the accessor
    refuses missing cells.
    """
    for name, legacy_field in trim_missing.ap_corrections.items():
        field = CellField.approximate_from_legacy(legacy_field, legacy_test_coadd.bounds)
        assert legacy_test_coadd.sparse_cell_index in field.bounds.missing, name
        # We have to access the private array to see that it's NaN (public
        # interface raises).
        index = legacy_test_coadd.sparse_cell_index - field.bounds.subgrid_start
        assert np.isnan(field._array[index.i, index.j]), name
        with pytest.raises(BoundsError):
            field.value_in_cell(legacy_test_coadd.sparse_cell_index)


def test_provenance_skips_missing_cells_and_matches_trimmed_inputs(
    legacy_test_coadd: _LegacyTestCoadd,
    trim_missing: _TrimmedLegacyCoaddComponents,
) -> None:
    """Test that provenance built from the trimmed catalog contributes
    nothing to the missing cell and is consistent with the trimmed catalog's
    own geometry.
    """
    psf = CellPointSpreadFunction.approximate_from_legacy(trim_missing.psf, legacy_test_coadd.bounds)
    bounds = psf.bounds
    cell_a = legacy_test_coadd.sparse_cell_index
    assert cell_a in bounds.missing

    provenance = CoaddProvenance.from_legacy_lazy(
        # from_legacy_lazy takes a CoaddInputs-like object and reads only
        # its .ccds attribute (lsst/images/cells/_provenance.py).
        types.SimpleNamespace(ccds=trim_missing.ccds),
        bounds=bounds,
        sky_projection=legacy_test_coadd.sky_projection,
        instrument="LSSTCam",
    )
    assert provenance is not None

    contribution_cells = {
        CellIJ(i=int(row["cell_i"]), j=int(row["cell_j"])) for row in provenance.contributions
    }
    assert cell_a not in contribution_cells
    # Every remaining cell here is covered by surviving inputs, so
    # contributions cover exactly the non-missing cells of the region.
    assert contribution_cells == set(bounds.cell_indices())
    # The provenance inputs table carries (visit, detector) identities;
    # from_legacy_lazy legitimately skips inputs that do not overlap the
    # region bounds (the NoOverlapError path in _provenance.py), so the
    # shipped helper asserts the identities are a subset of the trimmed
    # catalog rather than equal to it.

    trimmed_legacy = types.SimpleNamespace(
        info=types.SimpleNamespace(getCoaddInputs=lambda: types.SimpleNamespace(ccds=trim_missing.ccds)),
        getWcs=lambda: legacy_test_coadd.coadd.getWcs(),
    )
    compare_cell_coadd_provenance_to_legacy_lazy(provenance, trimmed_legacy, bounds)


def _move_record_far_away(record: Any, coadd_wcs: Any) -> None:
    """Give a coadd-input record a WCS centered on the far side of the sky
    from the coadd's tangent point, outside the coadd projection's domain.
    """
    from lsst.geom import Point2D, SpherePoint, degrees

    origin = coadd_wcs.getSkyOrigin()
    antipode = SpherePoint(origin.getLongitude() + 180.0 * degrees, -origin.getLatitude())
    cd_matrix = makeCdMatrix(scale=record.getWcs().getPixelScale())
    record.setWcs(makeSkyWcs(crpix=Point2D(0.0, 0.0), crval=antipode, cdMatrix=cd_matrix))


def _provenance_keys(provenance: CoaddProvenance) -> set[tuple[int, int]]:
    return {(int(row["visit"]), int(row["detector"])) for row in provenance.inputs}


def test_provenance_skips_inputs_outside_projection_domain(legacy_test_coadd: _LegacyTestCoadd) -> None:
    """Test that an input whose footprint cannot be projected onto the coadd
    is dropped rather than aborting the conversion.
    """
    ccds = legacy_test_coadd.coadd.info.getCoaddInputs().ccds
    expected = CoaddProvenance.from_legacy_lazy(
        types.SimpleNamespace(ccds=ccds),
        bounds=legacy_test_coadd.bounds,
        sky_projection=legacy_test_coadd.sky_projection,
        instrument="LSSTCam",
    )
    assert expected is not None
    moved = ccds.copy(deep=True)
    # Move an input that does overlap the test region, so the drop is visible.
    moved_key = next(iter(sorted(_provenance_keys(expected))))
    for record in moved:
        if (int(record.get("visit")), int(record.get("ccd"))) == moved_key:
            _move_record_far_away(record, legacy_test_coadd.coadd.getWcs())
    provenance = CoaddProvenance.from_legacy_lazy(
        types.SimpleNamespace(ccds=moved),
        bounds=legacy_test_coadd.bounds,
        sky_projection=legacy_test_coadd.sky_projection,
        instrument="LSSTCam",
    )
    assert provenance is not None
    assert _provenance_keys(provenance) == _provenance_keys(expected) - {moved_key}


def test_provenance_is_none_without_overlapping_inputs(legacy_test_coadd: _LegacyTestCoadd) -> None:
    """Test that provenance conversion yields `None`, not an error, when no
    input overlaps the coadd.
    """
    moved = legacy_test_coadd.coadd.info.getCoaddInputs().ccds.copy(deep=True)
    for record in moved:
        _move_record_far_away(record, legacy_test_coadd.coadd.getWcs())
    provenance = CoaddProvenance.from_legacy_lazy(
        types.SimpleNamespace(ccds=moved),
        bounds=legacy_test_coadd.bounds,
        sky_projection=legacy_test_coadd.sky_projection,
        instrument="LSSTCam",
    )
    assert provenance is None


def test_from_legacy_lazy_coadd_without_coadd_inputs(legacy_test_coadd: _LegacyTestCoadd) -> None:
    """Test that a legacy coadd with no coadd-inputs component converts to a
    coadd with no provenance.
    """
    coadd = legacy_test_coadd.coadd.clone()
    coadd.info.setCoaddInputs(None)
    assert coadd.info.getCoaddInputs() is None
    cell_coadd = CellCoadd.from_legacy_lazy_coadd(
        coadd,
        instrument="LSSTCam",
        tract_info=legacy_test_coadd.tract_info,
        patch=DP2_COADD_DATA_ID["patch"],
    )
    with pytest.raises(AttributeError):
        cell_coadd.provenance

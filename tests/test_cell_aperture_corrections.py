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

from typing import Any

import numpy as np
import pytest

from lsst.images import YX, BoundsError, Box
from lsst.images.cells import (
    CellApertureCorrectionMapSerializationModel,
    CellField,
    CellGrid,
    CellGridBounds,
    CellIJ,
)
from lsst.images.json import JsonInputArchive, JsonOutputArchive

try:
    from lsst.cell_coadds import StitchedApertureCorrection as LegacyStitchedApertureCorrection
    from lsst.cell_coadds import UniformGrid as LegacyUniformGrid
    from lsst.skymap import Index2D as LegacyIndex2D

    HAVE_LEGACY = True
except ImportError:
    HAVE_LEGACY = False
    type LegacyStitchedApertureCorrection = Any  # type: ignore[no-redef]
    type LegacyUniformGrid = Any  # type: ignore[no-redef]
    type LegacyIndex2D = Any  # type: ignore[no-redef]

skip_no_legacy = pytest.mark.skipif(not HAVE_LEGACY, reason="lsst.cell_coadds could not be imported.")

# A 3x3 grid with 10x10-pixel cells.
GRID = CellGrid(bbox=Box.from_shape((30, 30)), cell_shape=YX(10, 10))
BASE_BOUNDS = CellGridBounds(grid=GRID, bbox=Box.factory[0:30, 0:30])
CELL_X = CellIJ(i=0, j=0)
CELL_Y = CellIJ(i=1, j=1)


def _make_field(missing: frozenset[CellIJ]) -> CellField:
    """Build a CellField over the shared 3x3 grid with ``missing`` cells set
    to NaN and all other cells set to 0.5.
    """
    bounds = CellGridBounds(grid=GRID, bbox=Box.factory[0:30, 0:30], missing=missing)
    array = np.full((3, 3), 0.5)
    for cell in missing:
        index = cell - bounds.subgrid_start
        array[index.i, index.j] = np.nan
    return CellField(bounds, array)


def _make_legacy(gc: dict[LegacyIndex2D, float]) -> LegacyStitchedApertureCorrection:
    """Construct a legacy `StitchedApertureCorrection` on the shared 3x3 grid
    from the given per-cell values.
    """
    ugrid = LegacyUniformGrid(cell_size=YX(10, 10).to_legacy_int_extent(), shape=LegacyIndex2D(x=3, y=3))
    return LegacyStitchedApertureCorrection(ugrid, gc)


def _roundtrip(
    aperture_corrections: dict[CellIJ, CellField],
) -> tuple[CellApertureCorrectionMapSerializationModel, dict[str, CellField]]:
    """Serialize a CellField map, persist it to JSON, and read it back.

    Returns the serialized model and the deserialized map, so tests can assert
    both the shared serialized bounds and the per-field result.

    We can't use RoundtripJson for this because the in-memory type is just
    `dict` and hence doesn't have a ``serialize`` method.
    """
    output_archive = JsonOutputArchive()
    model = CellApertureCorrectionMapSerializationModel.serialize(aperture_corrections, output_archive)
    output_tree = output_archive.finish(model)
    input_tree = CellApertureCorrectionMapSerializationModel.model_validate_json(
        output_tree.model_dump_json()
    )
    input_archive = JsonInputArchive()
    return input_tree, input_tree.deserialize(input_archive)


@skip_no_legacy
def test_from_legacy_records_missing_cells() -> None:
    """Test that cells absent from the legacy map are recorded as missing."""
    missing = {CELL_X, CELL_Y}
    present = [c for c in BASE_BOUNDS.cell_indices() if c not in missing]
    gc = {c.to_legacy(): 0.5 + 0.1 * c.i + 0.01 * c.j for c in present}
    legacy = _make_legacy(gc)
    field = CellField.from_legacy_aperture_correction(legacy, BASE_BOUNDS)
    assert field.bounds.missing == missing
    for cell in present:
        np.testing.assert_allclose(field.value_in_cell(cell), gc[cell.to_legacy()])
    for cell in missing:
        with pytest.raises(BoundsError):
            field.value_in_cell(cell)


@skip_no_legacy
def test_from_legacy_all_cells_present() -> None:
    """Test that when every legacy cell has an entry, bounds are unchanged."""
    gc = {c.to_legacy(): 0.5 for c in BASE_BOUNDS.cell_indices()}
    legacy = _make_legacy(gc)
    field = CellField.from_legacy_aperture_correction(legacy, BASE_BOUNDS)
    assert field.bounds.missing == frozenset()


def test_serialize_deserialize_differing_missing_cells() -> None:
    """Test that a map whose fields differ in missing cells round-trips,
    restoring each field's per-cell missing set.
    """
    ap = {"A": _make_field(frozenset({CELL_X})), "B": _make_field(frozenset({CELL_X, CELL_Y}))}
    model, result = _roundtrip(ap)
    # Shared serialized bounds keep only cells missing from every field.
    assert model.bounds.missing == frozenset({CELL_X})
    assert result["A"].bounds.missing == frozenset({CELL_X})
    assert result["B"].bounds.missing == frozenset({CELL_X, CELL_Y})
    for name, field in ap.items():
        for cell in BASE_BOUNDS.cell_indices():
            if cell not in field.bounds.missing:
                np.testing.assert_allclose(result[name].value_in_cell(cell), field.value_in_cell(cell))
            else:
                with pytest.raises(BoundsError):
                    result[name].value_in_cell(cell)


def test_serialize_raises_on_inconsistent_grid() -> None:
    """Test that fields on different grids cannot be serialized together."""
    other_grid = CellGrid(bbox=Box.from_shape((40, 40)), cell_shape=YX(10, 10))
    other_bounds = CellGridBounds(grid=other_grid, bbox=Box.factory[0:30, 0:30])
    other_field = CellField(other_bounds, np.full((3, 3), 0.25))
    with pytest.raises(ValueError, match="do not have a consistent grid"):
        CellApertureCorrectionMapSerializationModel.serialize(
            {"A": _make_field(frozenset()), "B": other_field}, JsonOutputArchive()
        )


def test_deserialize_nan_column_becomes_missing() -> None:
    """Test that a NaN in one field column makes that cell missing for that
    field.
    """
    ap = {"A": _make_field(frozenset()), "B": _make_field(frozenset({CELL_Y}))}
    _, result = _roundtrip(ap)
    assert result["A"].bounds.missing == frozenset()
    assert result["B"].bounds.missing == frozenset({CELL_Y})
    np.testing.assert_allclose(result["A"].value_in_cell(CELL_Y), ap["A"].value_in_cell(CELL_Y))
    with pytest.raises(BoundsError):
        result["B"].value_in_cell(CELL_Y)

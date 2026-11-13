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

import json
from collections.abc import Mapping
from pathlib import Path
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
from lsst.images.serialization import read_archive
from lsst.images.tests import current_fixture_path

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

FIXTURE_DIR = Path(__file__).parent / "data" / "schemas"
COMPAT_DIR = Path(__file__).parent / "data" / "compat"

# One name of the pre-standardization map has no value for this cell, while
# the others do.
_BORDER_MISSING_NAME = "ext_convolved_ConvolvedFlux_2_3_3_instFlux"
_BORDER_MISSING_CELL = CellIJ(i=10, j=7)


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
    aperture_corrections: Mapping[str, CellField | None],
) -> tuple[CellApertureCorrectionMapSerializationModel, dict[str, CellField | None]]:
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
    """Test that cells the legacy map has no entry for, and cells whose entry
    is non-finite, are both recorded as missing.
    """
    missing = {CELL_X, CELL_Y}
    present = [c for c in BASE_BOUNDS.cell_indices() if c not in missing]
    gc = {c.to_legacy(): 0.5 + 0.1 * c.i + 0.01 * c.j for c in present}
    gc[CELL_Y.to_legacy()] = np.nan
    legacy = _make_legacy(gc)
    field = CellField.from_legacy_aperture_correction(legacy, BASE_BOUNDS)
    assert field is not None
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


def test_value_in_cell_rejects_index_below_bounds() -> None:
    """An index below the bounds must not wrap around the NumPy array."""
    field = _make_field(frozenset())
    with pytest.raises(BoundsError, match="out of bounds"):
        field.value_in_cell(CellIJ(i=-1, j=0))


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


def test_serialize_merges_fields_with_different_bounds() -> None:
    """Test that a field with missing cells on a full boundary gets a smaller
    bbox in its bounds.
    """
    wide = _make_field(frozenset({CELL_X}))
    narrow_bounds = CellGridBounds(grid=GRID, bbox=Box.factory[10:30, 10:30])
    narrow = CellField(narrow_bounds, np.full((2, 2), 0.25))
    model, result = _roundtrip({"A": wide, "B": narrow})
    assert model.bounds == wide.bounds
    assert result["A"] is not None
    assert result["B"] is not None
    assert result["A"].bounds == wide.bounds
    assert result["B"].bounds == narrow.bounds
    assert result["B"].value_in_cell(CellIJ(i=1, j=1)) == 0.25


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


@skip_no_legacy
def test_from_legacy_returns_none_without_usable_values() -> None:
    """A name with no usable value in any cell converts to `None` instead of a
    field with no cells.
    """
    all_nan = {c.to_legacy(): np.nan for c in BASE_BOUNDS.cell_indices()}
    assert CellField.from_legacy_aperture_correction(_make_legacy(all_nan), BASE_BOUNDS) is None
    # Cells the legacy container has no entry for are unusable in the same way.
    assert CellField.from_legacy_aperture_correction(_make_legacy({}), BASE_BOUNDS) is None


def test_serialize_without_any_usable_name_is_none() -> None:
    """A map with no usable name — empty, or with only `None`-valued names —
    records no aperture corrections at all, since there is nothing left to
    define the shared bounds.
    """
    assert CellApertureCorrectionMapSerializationModel.serialize({}, JsonOutputArchive()) is None
    assert CellApertureCorrectionMapSerializationModel.serialize({"A": None}, JsonOutputArchive()) is None


def test_read_pre_standardization_missing_border_column() -> None:
    """Test that an aperture-correction map written before `CellGridBounds`
    became canonical reads as the map it describes.

    The fixture under ``tests/data/compat`` was written by the code just
    before the standardization (``eb8d302^``) from the fixture of its own
    schema.  Its region reaches one cell further in +x than that fixture's, as
    a matched aperture correction covers the coadd it was stitched from rather
    than the smaller coadd stored with it, and the column at its low-x edge has
    no value for any name.  The old code stored that region as given; the
    current code shrinks a bbox to the cells that have values.  The one name
    that lacks a value the others have keeps that difference in its NaN cells
    rather than in the shared bounds.

    Because the table holds one row per cell with a value, shrinking the bbox
    moves nothing: an aperture-correction map needs no special handling on read
    the way a PSF, whose array is indexed by its bbox, does.
    """
    path = COMPAT_DIR / "cell_aperture_correction_map-border-missing-column.json"
    stored = json.loads(path.read_text())
    assert stored["bounds"]["bbox"] == {
        "y": {"start": 42, "stop": 66},
        "x": {"start": 30, "stop": 54},
    }, "the fixture must still name the wider region the old code wrote"
    stored_missing = {CellIJ(i=cell["i"], j=cell["j"]) for cell in stored["bounds"]["missing"]}
    assert {CellIJ(i=i, j=5) for i in range(7, 11)} <= stored_missing, (
        "the fixture must lack a whole column, which is what the shrink removes"
    )

    corrections = read_archive(path)
    truth = read_archive(current_fixture_path(FIXTURE_DIR, "cell_aperture_correction_map"))
    assert corrections.keys() == truth.keys()
    for name, field in corrections.items():
        was = truth[name]
        assert field.bounds.bbox == Box.factory[42:66, 36:54]
        expected_missing = set(was.bounds.missing)
        if name == _BORDER_MISSING_NAME:
            expected_missing.add(_BORDER_MISSING_CELL)
        assert field.bounds.missing == frozenset(expected_missing)
        assert all(np.isfinite(field.value_in_cell(cell)) for cell in field.bounds.cell_indices())
        shared = set(was.bounds.cell_indices()) & set(field.bounds.cell_indices())
        assert shared
        assert all(field.value_in_cell(cell) == was.value_in_cell(cell) for cell in shared)
        for cell in field.bounds.missing:
            with pytest.raises(BoundsError):
                field.value_in_cell(cell)

    # The shared bounds do not know about the one name's missing cell, so the
    # value the other names have for it survives in the table.
    others = corrections["modelfit_CModel_initial_instFlux"]
    assert _BORDER_MISSING_CELL not in others.bounds.missing
    assert np.isfinite(others.value_in_cell(_BORDER_MISSING_CELL))

    # Reading the file lands on the form the current code writes, so reading
    # what we just wrote back changes nothing.
    _, again = _roundtrip(corrections)
    for name, field in corrections.items():
        assert again[name] is not None
        assert again[name].bounds == field.bounds
        assert all(
            again[name].value_in_cell(cell) == field.value_in_cell(cell)
            for cell in field.bounds.cell_indices()
        )

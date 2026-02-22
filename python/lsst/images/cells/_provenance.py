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

__all__ = ("CoaddProvenance", "CoaddProvenanceSerializationModel")

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, ClassVar

import astropy.table
import astropy.units as u
import numpy as np
import pydantic

from .._cell_grid import CellIJ
from .._polygon import Polygon
from ..serialization import ArchiveTree, InputArchive, OutputArchive, TableModel

if TYPE_CHECKING:
    try:
        from lsst.cell_coadds import CoaddInputs as LegacyCoaddInputs
        from lsst.cell_coadds import MultipleCellCoadd, ObservationIdentifiers
        from lsst.skymap import Index2D
    except ImportError:
        type Index2D = Any  # type: ignore[no-redef]
        type LegacyCoaddInputs = Any  # type: ignore[no-redef]
        type MultipleCellCoadd = Any  # type: ignore[no-redef]
        type ObservationIdentifiers = Any  # type: ignore[no-redef]


class CoaddProvenance:
    """A pair of tables that record the inputs to a cell-based coadd.

    Parameters
    ----------
    inputs
        A table of {visit, detector} combinations that contribute to any cell
        in the coadd.
    contributions
        A table of {visit, detector, cell} combinations that describe how an
        observation contributed to a cell.

    Notes
    -----
    This object can represent the provenance of a whole patch, a single cell,
    or anything in between.  In the single-cell case, the ``inputs`` and
    ``contributions`` tables have the same number of rows (but may not be
    ordered the same way!).
    """

    def __init__(self, inputs: astropy.table.Table, contributions: astropy.table.Table):
        self._inputs = inputs
        self._contributions = contributions

    _INPUT_TABLE_DESCRIPTIONS: ClassVar[Mapping[str, str]] = {
        "instrument": "Name of the instrument.",
        "visit": "ID of the visit.",
        "detector": "ID of the detector.",
        "physical_filter": "Full name of the bandpass filter.",
        "day_obs": "Observation night as a YYYYMMDD integer.",
        "polygon": (
            "Polygon that approximates the overlap of the observation and the coadd patch, "
            "in coadd coordinates."
        ),
    }

    _CONTRIBUTION_TABLE_DESCRIPTIONS: ClassVar[Mapping[str, str]] = {
        "cell_i": "Y-axis index of the cell within the patch.",
        "cell_j": "X-axis index of the cell within the patch.",
        "instrument": "Name of the instrument.",
        "visit": "ID of the visit.",
        "detector": "ID of the detector.",
        "overlaps_center": "Whether a this observation overlaps the center of the cell.",
        "overlap_fraction": "Fraction of the cell that is covered by the overlap region.",
        "weight": "Weight to be used for this input in this cell.",
        "psf_shape_xx": "Second order moments of the PSF.",
        "psf_shape_yy": "Second order moments of the PSF.",
        "psf_shape_xy": "Second order moments of the PSF.",
        "psf_shape_flag": "Flag indicating whether the PSF shape measurement was successful.",
    }

    @classmethod
    def make_empty_input_table(cls, n_rows: int) -> astropy.table.Table:
        """Make an empty `inputs` table with a set number of rows."""
        result = astropy.table.Table(
            [
                astropy.table.Column(name="instrument", length=n_rows, dtype=np.object_),
                astropy.table.Column(name="visit", length=n_rows, dtype=np.uint64),
                astropy.table.Column(name="detector", length=n_rows, dtype=np.uint16),
                astropy.table.Column(name="physical_filter", length=n_rows, dtype=np.object_),
                astropy.table.Column(name="day_obs", length=n_rows, dtype=np.uint32),
                astropy.table.Column(name="polygon", length=n_rows, dtype=np.object_),
            ]
        )
        for k, v in cls._INPUT_TABLE_DESCRIPTIONS.items():
            result.columns[k].description = v
        return result

    @classmethod
    def make_empty_contribution_table(cls, n_rows: int) -> astropy.table.Table:
        """Make an empty `contributions` table with a set number of rows."""
        result = astropy.table.Table(
            [
                astropy.table.Column(name="cell_i", length=n_rows, dtype=np.uint16),
                astropy.table.Column(name="cell_j", length=n_rows, dtype=np.uint16),
                astropy.table.Column(name="instrument", length=n_rows, dtype=np.object_),
                astropy.table.Column(name="visit", length=n_rows, dtype=np.uint64),
                astropy.table.Column(name="detector", length=n_rows, dtype=np.uint16),
                astropy.table.Column(name="overlaps_center", length=n_rows, dtype=np.bool_),
                astropy.table.Column(name="overlap_fraction", length=n_rows, dtype=np.float64),
                astropy.table.Column(name="weight", length=n_rows, dtype=np.float64),
                astropy.table.Column(name="psf_shape_xx", length=n_rows, dtype=np.float64, unit=u.pix**2),
                astropy.table.Column(name="psf_shape_yy", length=n_rows, dtype=np.float64, unit=u.pix**2),
                astropy.table.Column(name="psf_shape_xy", length=n_rows, dtype=np.float64, unit=u.pix**2),
                astropy.table.Column(name="psf_shape_flag", length=n_rows, dtype=np.bool_),
            ]
        )
        for k, v in cls._CONTRIBUTION_TABLE_DESCRIPTIONS.items():
            result.columns[k].description = v
        return result

    @property
    def inputs(self) -> astropy.table.Table:
        """A table of {visit, detector} combinations that contribute to any
        cell in the coadd.
        """
        return self._inputs

    @property
    def contributions(self) -> astropy.table.Table:
        """A table of {visit, detector, cell} combinations that describe how an
        observation contributed to a cell.
        """
        return self._contributions

    def __getitem__(self, cell: CellIJ) -> CoaddProvenance:
        return self.subset([cell])

    def subset(self, cells: Iterable[CellIJ]) -> CoaddProvenance:
        """Return a new provenance object with just the given cells."""
        cells_to_keep = astropy.table.Table(
            rows=[(index.i, index.j) for index in cells],
            names=["cell_i", "cell_j"],
            dtype=[np.uint16, np.uint16],
        )
        contributions = astropy.table.join(self._contributions, cells_to_keep)
        assert contributions.columns.keys() == self._CONTRIBUTION_TABLE_DESCRIPTIONS.keys()
        inputs = astropy.table.join(contributions["instrument", "visit", "detector"], self._inputs)
        assert inputs.columns.keys() == self._INPUT_TABLE_DESCRIPTIONS.keys()
        return CoaddProvenance(inputs=inputs, contributions=contributions)

    def serialize(self, archive: OutputArchive[Any]) -> CoaddProvenanceSerializationModel:
        """Serialize the provenance to an output archive.

        Parameters
        ----------
        archive
            Archive to write to.
        """
        inputs = self._inputs.copy(copy_data=False)
        contributions = self._contributions.copy(copy_data=False)
        instrument = CoaddProvenanceSerializationModel._fix_str_for_serialization(
            "instrument", inputs, contributions
        )
        physical_filter = CoaddProvenanceSerializationModel._fix_str_for_serialization(
            "physical_filter", inputs
        )
        CoaddProvenanceSerializationModel._fix_polygon_for_serialization(inputs)
        inputs_model = archive.add_table(inputs, name="inputs")
        contributions_model = archive.add_table(contributions, name="contributions")
        return CoaddProvenanceSerializationModel(
            instrument=instrument,
            physical_filter=physical_filter,
            inputs=inputs_model,
            contributions=contributions_model,
        )

    @classmethod
    def deserialize(
        cls,
        model: CoaddProvenanceSerializationModel,
        archive: InputArchive[Any],
    ) -> CoaddProvenance:
        """Deserialize a provenance from an input archive.

        Parameters
        ----------
        model
            A Pydantic model representation of the image, holding references
            to data stored in the archive.
        archive
            Archive to read from.

        Notes
        -----
        While `CoaddProvenance.subset` can be used to filter provenance
        information down to just certain cells, there is no advantage to be
        had from doing this during deserialization (the table data is not
        ordered by cell, and hence there's read-slicing we can do).
        """
        inputs = archive.get_table(model.inputs)
        contributions = archive.get_table(model.contributions)
        CoaddProvenanceSerializationModel._fix_str_for_deserialization(
            "instrument", model.instrument, inputs, contributions
        )
        CoaddProvenanceSerializationModel._fix_str_for_deserialization(
            "physical_filter", model.physical_filter, inputs
        )
        CoaddProvenanceSerializationModel._fix_polygon_for_deserialization(inputs)
        for k, v in cls._INPUT_TABLE_DESCRIPTIONS.items():
            inputs.columns[k].description = v
        for k, v in cls._CONTRIBUTION_TABLE_DESCRIPTIONS.items():
            contributions.columns[k].description = v
        return cls(inputs=inputs, contributions=contributions)

    @staticmethod
    def from_legacy(legacy_cell_coadd: MultipleCellCoadd) -> CoaddProvenance:
        """Extract provenance from a legacy
        `lsst.cell_coadds.MultipleCellCoadd` object.
        """
        inputs = CoaddProvenance.make_empty_input_table(len(legacy_cell_coadd.common.visit_polygons))
        for n, (legacy_identifiers, legacy_polygon) in enumerate(
            legacy_cell_coadd.common.visit_polygons.items()
        ):
            inputs["instrument"][n] = legacy_identifiers.instrument
            inputs["visit"][n] = legacy_identifiers.visit
            inputs["detector"][n] = legacy_identifiers.detector
            inputs["physical_filter"][n] = legacy_identifiers.physical_filter
            inputs["day_obs"][n] = legacy_identifiers.day_obs
            inputs["polygon"][n] = Polygon.from_legacy(legacy_polygon)
        n_contributions = 0
        for legacy_cell in legacy_cell_coadd.cells.values():
            n_contributions += len(legacy_cell.inputs)
        contributions = CoaddProvenance.make_empty_contribution_table(n_contributions)
        n = 0
        for legacy_cell in legacy_cell_coadd.cells.values():
            for legacy_identifiers, legacy_inputs in legacy_cell.inputs.items():
                contributions["cell_i"][n] = legacy_cell.identifiers.cell.y
                contributions["cell_j"][n] = legacy_cell.identifiers.cell.x
                contributions["instrument"][n] = legacy_identifiers.instrument
                contributions["visit"][n] = legacy_identifiers.visit
                contributions["detector"][n] = legacy_identifiers.detector
                contributions["overlaps_center"][n] = legacy_inputs.overlaps_center
                contributions["overlap_fraction"][n] = legacy_inputs.overlap_fraction
                contributions["weight"][n] = legacy_inputs.weight
                contributions["psf_shape_xx"][n] = legacy_inputs.psf_shape.getIxx()
                contributions["psf_shape_yy"][n] = legacy_inputs.psf_shape.getIyy()
                contributions["psf_shape_xy"][n] = legacy_inputs.psf_shape.getIxy()
                contributions["psf_shape_flag"][n] = legacy_inputs.psf_shape_flag
                n += 1
        return CoaddProvenance(inputs=inputs, contributions=contributions)


class CoaddProvenanceSerializationModel(ArchiveTree):
    """A Pydantic model used to represent a serialized `CoaddProvenance`.

    Notes
    -----
    We can't rewrite the Astropy tables directly into the archive (e.g. as
    FITS binary tables for a FITS archive), because:

    - `str` columns are a huge pain in both Numpy and FITS;
    - the polygon columns need to be rewritten as array-valued columns.

    To deal with the string columns (``instrument`` and ``physical_filter``)
    we do dictionary compression: we map each distinct value of those columns
    to an integer, and then we save that mapping to the model while saving
    an integer version of that column in the table.  But if there is actually
    only one value in that column (the most common case by far) we just drop
    the column and store that value directly in the model.
    """

    instrument: str | dict[str, int] = pydantic.Field(
        description=(
            "Instrument name for all inputs to this coadd, or a mapping from "
            "instrument name to the integer used in its place in the tables."
        )
    )
    physical_filter: str | dict[str, int] = pydantic.Field(
        description="Physical filter name for all inputs to this coadd."
    )
    inputs: TableModel = pydantic.Field(description="Table of all inputs to the coadd.")
    contributions: TableModel = pydantic.Field(description="Table of per-cell contributions to the coadd.")

    @staticmethod
    def _fix_str_for_serialization(column: str, *tables: astropy.table.Table) -> str | dict[str, int]:
        """Rewrite a string column as an integer column or drop it.

        Parameters
        ----------
        column
            Name of the column to rewrite.
        *tables
            One or more astropy tables to rewrite.  The first table is assumed
            to have all values for this column that might appear in any other
            tables.

        Returns
        -------
        `str` | `dict` [`str`, `int`]
            If there is only one unique value for this column in the first
            table, that value (and the column will have been dropped from
            all givne tables).  If the tables are empty, the column is
            dropped and an empty `dict` is returned.  In all other cases the
            given column is replaced with an integer column in all given
            tables and the mapping from strings to integers is returned.
        """
        result: str | dict[str, int] = {name: n for n, name in enumerate(sorted(set(tables[0][column])))}
        match len(result):
            case 0:
                pass
            case 1:
                (result,) = result.keys()  # type: ignore[union-attr]
            case _:
                for table in tables:
                    table.columns[column] = astropy.table.Column(
                        data=[result[k] for k in table.columns[column]],
                        name=column,
                        dtype=np.uint8,
                        description=f"Integer mapped to {column} name.",
                    )
                return result
        # If we didn't remap to an integer (case 0 and 1 above), delete the
        # column.
        for table in tables:
            del table.columns[column]
        return result

    @staticmethod
    def _fix_str_for_deserialization(
        column: str, value: str | dict[str, int], *tables: astropy.table.Table
    ) -> None:
        """Rewrite an integer column back to a string one.

        Parameters
        ----------
        column
            Name of the column to rewrite.
        value
            Value or mapping of values returned by
            `_fix_str_for_serialization`.
        tables
            Tables to rewrite this column in.
        """
        match value:
            case str():
                for table in tables:
                    table.columns[column] = astropy.table.Column([value] * len(table), dtype=object)
            case dict():
                mapping = {v: k for k, v in value.items()}
                for table in tables:
                    table.columns[column] = astropy.table.Column(
                        [mapping[k] for k in table[column]], dtype=object
                    )

    @staticmethod
    def _fix_polygon_for_serialization(inputs: astropy.table.Table) -> None:
        """Rewrite a polygon `object` column as a pair of array-valued columns
        and an array-size column.

        Parameters
        ----------
        inputs
            A copy of the in-memory coadd inputs table to modify in-place into
            its serialization form.
        """
        max_n_vertices = max(p.n_vertices for p in inputs["polygon"])
        inputs["n_vertices"] = astropy.table.Column(
            [p.n_vertices for p in inputs["polygon"]],
            name="n_vertices",
            dtype=np.uint8,
            description="Number of polygon vertices.",
        )
        inputs["x_vertices"] = astropy.table.Column(
            name="x_vertices",
            dtype=np.float64,
            length=len(inputs),
            shape=(max_n_vertices,),
            description="X coordinates of polygon vertices, in tract coordinates.",
        )
        inputs["x_vertices"][:, :] = np.nan
        inputs["y_vertices"] = astropy.table.Column(
            name="y_vertices",
            dtype=np.float64,
            length=len(inputs),
            shape=(max_n_vertices,),
            description="Y coordinates of polygon vertices, in tract coordinates.",
        )
        inputs["y_vertices"][:, :] = np.nan
        for i, polygon in enumerate(inputs["polygon"]):
            inputs["n_vertices"][i] = polygon.n_vertices
            inputs["x_vertices"][i][: polygon.n_vertices] = polygon.x_vertices
            inputs["y_vertices"][i][: polygon.n_vertices] = polygon.y_vertices
        del inputs["polygon"]

    @staticmethod
    def _fix_polygon_for_deserialization(inputs: astropy.table.Table) -> None:
        """Rewrite a a pair of array-valued columns and an array-size column
        into a polygon `object` column.

        Parameters
        ----------
        inputs
            The serialized version of the coadd inputs table, to be modified
            in-place into its in-memory form.
        """
        polygons = [
            Polygon(x_vertices=x_vertices[:n_vertices], y_vertices=y_vertices[:n_vertices])
            for n_vertices, x_vertices, y_vertices in zip(
                inputs["n_vertices"], inputs["x_vertices"], inputs["y_vertices"]
            )
        ]
        del inputs["n_vertices"]
        del inputs["x_vertices"]
        del inputs["y_vertices"]
        inputs["polygon"] = astropy.table.Column(polygons, name="polygon", dtype=np.object_)

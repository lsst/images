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

from collections.abc import Iterable
from typing import Annotated, Any

import astropy.table
import astropy.units as u
import numpy as np
import pydantic

from .._geom import YX, Box
from ..polygon import SimplePolygon
from ..serialization import ArchiveTree, InputArchive, OutputArchive, TableReferenceModel
from ..utils import TableRowModel


class CoaddInput(TableRowModel, arbitrary_types_allowed=True):
    instrument: Annotated[str, np.dtype(object)] = pydantic.Field(
        description="Name of the instrument this observation is from."
    )

    visit: Annotated[int, np.dtype(np.uint64)] = pydantic.Field(description="ID of the observation's visit.")

    detector: Annotated[int, np.dtype(np.uint8)] = pydantic.Field(description="ID of the detector.")

    physical_filter: Annotated[str, np.dtype(object)] = pydantic.Field(
        description="Full name of the bandpass filter."
    )

    day_obs: Annotated[int, np.dtype(np.uint32)] = pydantic.Field(
        description="Observation night as a YYYYMMDD integer."
    )

    polygon: Annotated[SimplePolygon, np.dtype(object)] = pydantic.Field(
        description="Polygon that approximates the overlap of the observation and the coadd patch."
    )


class CoaddContribution(TableRowModel):
    cell_x: Annotated[int, np.dtype(np.uint8)]
    """X index of the cell within its patch."""

    cell_y: Annotated[int, np.dtype(np.uint8)]
    """Y index of the cell within its patch."""

    instrument: Annotated[str, np.dtype(object)]
    """Name of the instrument this observation is from."""

    visit: Annotated[int, np.dtype(np.uint64)]
    """ID of the observation's visit."""

    detector: Annotated[int, np.dtype(np.uint8)]
    """ID of the detector."""

    overlaps_center: bool
    """Whether a this observation overlaps the center of the cell."""

    overlap_fraction: float
    """Fraction of the cell that is covered by the overlap region."""

    weight: float
    """Weight to be used for this input in this cell."""

    psf_shape_xx: Annotated[float, u.pix**2]
    """Second order moments of the PSF."""

    psf_shape_yy: Annotated[float, u.pix**2]
    """Second order moments of the PSF."""

    psf_shape_xy: Annotated[float, u.pix**2]
    """Second order moments of the PSF."""

    psf_shape_flag: bool
    """Flag indicating whether the PSF shape measurement was successful."""

    @property
    def cell(self) -> YX[int]:
        """2-d index of the cell within its patch (`YX` [`int`])."""
        return YX(y=self.cell_y, x=self.cell_x)


class CoaddProvenance:
    def __init__(
        self,
        inputs: Iterable[CoaddInput] | astropy.table.Table,
        contributions: Iterable[CoaddContribution] | astropy.table.Table,
        *,
        instrument: str | None = None,
        physical_filter: str | None = None,
    ):
        if isinstance(inputs, astropy.table.Table):
            self._inputs = inputs
        else:
            self._inputs = CoaddInput.model_make_table(inputs)
        if isinstance(contributions, astropy.table.Table):
            self._contributions = contributions
        else:
            self._contributions = CoaddContribution.model_make_table(contributions)

    @property
    def inputs(self) -> astropy.table.Table:
        return self._inputs

    @property
    def contributions(self) -> astropy.table.Table:
        return self._contributions

    def serialize(self, archive: OutputArchive[Any]) -> CoaddProvenanceSerializationModel:
        inputs = self._inputs.copy(copy_data=False)
        contributions = self._inputs.copy(copy_data=False)
        instrument = self._fix_str_for_serialization("instrument", inputs, contributions)
        physical_filter = self._fix_str_for_serialization("physical_filter", inputs)
        self._fix_polygon_for_serialization(inputs)
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
        *,
        bbox: Box | None = None,
    ) -> CoaddProvenance:
        inputs = archive.get_table(model.inputs)
        contributions = archive.get_table(model.contributions)
        cls._fix_str_for_deserialization("instrument", model.instrument, inputs, contributions)
        cls._fix_str_for_deserialization("physical_filter", model.physical_filter, inputs)
        cls._fix_polygon_for_deserialization(inputs)
        for name, field_info in CoaddInput.model_fields.items():
            inputs.columns[name].description = field_info.description
        for name, field_info in CoaddContribution.model_fields.items():
            contributions.columns[name].description = field_info.description
        return cls(inputs=inputs, contributions=contributions)

    @staticmethod
    def _fix_str_for_serialization(self, column: str, *tables: astropy.table.Table) -> str | dict[str, int]:
        result: str | dict[str, int] = {
            name: n for n, name in enumerate(sorted(set(tables[0]["instrument"])))
        }
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
        for table in tables:
            del table.columns[column]
        return result

    @staticmethod
    def _fix_str_for_deserialization(
        column: str, value: str | dict[str, int], *tables: astropy.table.Table
    ) -> None:
        match value:
            case str():
                for table in tables:
                    table.columns[column] = astropy.tables.Column([value] * len(table), dtype=object)
            case dict():
                mapping = {v: k for k, v in value.items()}
                for table in tables:
                    table.columns[column] = astropy.tables.Column(
                        [mapping[k] for k in table[column]], dtype=object
                    )

    @staticmethod
    def _fix_polygon_for_serialization(inputs: astropy.table.Table) -> None:
        max_n_vertices = max(p.n_vertices for p in inputs["polygon"])
        inputs["n_vertices"] = astropy.table.Column(
            [p.n_vertices for p in inputs["polygon"]],
            name="n_vertices",
            dtype=np.uint8,
            description="Number of polygon vertices.",
        )
        inputs["x_vertices"] = astropy.table.Column(
            name="x_vertices",
            dtype=np.float32,
            shape=(max_n_vertices,),
            description="X coordinates of polygon vertices, in tract coordinates.",
        )
        inputs["x_vertices"][:, :] = np.nan
        inputs["y_vertices"] = astropy.table.Column(
            name="y_vertices",
            dtype=np.float32,
            shape=(max_n_vertices,),
            description="Y coordinates of polygon vertices, in tract coordinates.",
        )
        inputs["y_vertices"][:, :] = np.nan
        for i, polygon in enumerate(inputs["polygon"]):
            inputs["n_vertices"][i] = polygon.n_vertices
            inputs["x_vertices"][i][:] = [p.x for p in polygon]
        del inputs["polygon"]

    @staticmethod
    def _fix_polygon_for_deserialization(inputs: astropy.table.Table) -> None:
        polygons = [
            SimplePolygon(x_vertices=x_vertices[:n_vertices], y_vertices=y_vertices[:n_vertices])
            for n_vertices, x_vertices, y_vertices in zip(
                inputs["n_vertices"], inputs["x_vertices"], inputs["y_vertices"]
            )
        ]
        del inputs["n_vertices"]
        del inputs["x_vertices"]
        del inputs["y_vertices"]
        inputs["polygon"] = astropy.table.Column(
            polygons,
            name="polygon",
            dtype=object,
            description=CoaddInput.model_fields["polygon"].description,
        )


class CoaddProvenanceSerializationModel(ArchiveTree):
    instrument: str | dict[str, int] = pydantic.Field(
        description=(
            "Instrument name for all inputs to this coadd, or a mapping from "
            "instrument name to the integer used in its place in the tables."
        )
    )
    physical_filter: str | dict[str, int] = pydantic.Field(
        description="Physical filter name for all inputs to this coadd."
    )
    inputs: TableReferenceModel = pydantic.Field(description="Table of all inputs to the coadd.")
    contributions: TableReferenceModel = pydantic.Field(
        description="Table of per-cell contributions to the coadd."
    )

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

__all__ = ("PSFExWrapper",)

from typing import Any

import numpy as np
import pydantic

from .._geom import Domain, SerializableDomain
from ..archive import InputArchive, OutputArchive
from ..asdf_utils import InlineArray
from ..tables import TableModel
from .legacy import LegacyPointSpreadFunction


class PSFExWrapper(LegacyPointSpreadFunction):
    """A specialization of LegacyPointSpreadFunction for the PSFEx backend."""

    def __init__(self, impl: Any, domain: Domain):
        from lsst.meas.extensions.psfex import PsfexPsf

        if not isinstance(impl, PsfexPsf):
            raise TypeError(f"{impl!r} is not a PSFEx object.")
        super().__init__(impl, domain)

    def serialize(self, archive: OutputArchive[Any]) -> PSFExSerializationModel:
        """Serialize the PSF to an archive.

        This method is intended to be usable as the callback function passed to
        `..archives.OutputArchive.serialize_direct` or
        `..archives.OutputArchive.serialize_pointer`.
        """
        data = self._impl.getSerializationData()
        shape = tuple(reversed(data.size))
        dtype = np.dtype([("parameters", data.comp.dtype, shape[1:])])
        structured_array = np.empty(shape[:1], dtype=dtype)
        structured_array["parameters"] = data.comp.reshape(*shape)
        table_ref = archive.add_structured_array("parameters", structured_array)
        return PSFExSerializationModel(
            average_x=data.average_x,
            average_y=data.average_y,
            pixel_step=data.pixel_step,
            group=data.group,
            degree=data.degree,
            basis=data.basis,
            coeff=data.coeff,
            parameters=table_ref,
            context=data.context,
            domain=self.domain.serialize(),
        )

    @classmethod
    def deserialize(cls, model: PSFExSerializationModel, archive: InputArchive[Any]) -> PSFExWrapper:
        """Deserialize the PSF from an archive.

        This method is intended to be usable as the callback function passed to
        `..archives.InputArchive.deserialize_pointer`.
        """
        from lsst.meas.extensions.psfex import PsfexPsf, PsfexPsfSerializationData

        structured_array = archive.get_structured_array(model.parameters)
        parameters = structured_array["parameters"].astype(np.float32)
        data = PsfexPsfSerializationData()
        data.average_x = model.average_x
        data.average_y = model.average_y
        data.pixel_step = model.pixel_step
        data.group = model.group
        data.degree = model.degree
        data.basis = model.basis
        data.coeff = model.coeff
        data.size = list(reversed(parameters.shape))
        data.comp = parameters.flatten()
        data.context = model.context
        legacy_psf = PsfexPsf.fromSerializationData(data)
        return cls(legacy_psf, Domain.deserialize(model.domain))


class PSFExSerializationModel(pydantic.BaseModel):
    """Model used for serializing PSFEx PSF models."""

    average_x: float = pydantic.Field(
        description="Average X position of the stars used to build this PSF model."
    )

    average_y: float = pydantic.Field(
        description="Average Y position of the stars used to build this PSF model."
    )

    pixel_step: float = pydantic.Field(
        description="Size of a model pixel, as a fraction or multiple of the native pixel size."
    )

    group: list[int] = pydantic.Field(
        default_factory=lambda: [0, 0],
        exclude_if=lambda v: v == [0, 0],
        description="Number of model groups in each dimension.",
    )

    degree: list[int] = pydantic.Field(description="Polynomial degree for each model group.")

    basis: list[float] = pydantic.Field(description="Basis function values.")

    coeff: list[float] = pydantic.Field(description="Polynomial coefficients.")

    parameters: TableModel = pydantic.Field(
        description="Reference to a table with the complete model parameters."
    )

    context: InlineArray = pydantic.Field(description="Internal PSFEx context array.")

    domain: SerializableDomain = pydantic.Field(description="Validity range for this PSF model.")

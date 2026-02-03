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

__all__ = ("ImageModel",)

from collections.abc import Sequence

import pydantic

from .._geom import Box, Interval
from ._asdf_utils import ArrayReferenceModel, ArrayReferenceQuantityModel, Unit


class ImageModel(pydantic.BaseModel):
    """Pydantic model used to represent the serialized form of an `.Image`."""

    data: ArrayReferenceQuantityModel | ArrayReferenceModel
    start: list[int]

    @property
    def bbox(self) -> Box:
        """The bounding box of the image."""
        if isinstance(self.data, ArrayReferenceQuantityModel):
            shape = self.data.value.shape
        else:
            shape = self.data.shape
        return Box(*[Interval.factory[begin : begin + size] for begin, size in zip(self.start, shape)])

    @classmethod
    def pack(
        cls,
        array_model: ArrayReferenceModel,
        start: Sequence[int],
        unit: Unit | None,
    ) -> ImageModel:
        """Construct an `ImageModel` from the components of a serialized
        image.

        Parameters
        ----------
        array_model
            Serialized form of the underlying array.
        start
            Logical coordinates of the first pixel in the array.
        unit : `astropy.units.UnitBase` or `None`
            Units for the image's pixel values.
        """
        if unit is None:
            return cls.model_construct(data=array_model, start=list(start))
        return cls.model_construct(
            data=ArrayReferenceQuantityModel.model_construct(value=array_model, unit=unit),
            start=list(start),
        )

    def unpack(self) -> tuple[ArrayReferenceModel, Unit | None]:
        """Return the components of a serialized image from this model.

        Returns
        -------
        array_model
            Serialized form of the underlying array.
        unit
            Units for the image's pixel values.

        Notes
        -----
        The ``start`` attribute is not included in the results because it is
        directly accessible, rather than possibly nested under `data` as with
        the other attributes.
        """
        if isinstance(self.data, ArrayReferenceQuantityModel):
            return self.data.value, self.data.unit
        return self.data, None

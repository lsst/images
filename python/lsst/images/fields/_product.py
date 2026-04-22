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

__all__ = ("ProductField", "ProductFieldSerializationModel")

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal, final

import astropy.units
import numpy as np
import pydantic

from .._geom import Bounds, Box
from .._image import Image
from ..serialization import ArchiveTree, InputArchive, OutputArchive
from ._base import BaseField

if TYPE_CHECKING:
    try:
        from lsst.afw.math import ProductBoundedField as LegacyProductBoundedField
    except ImportError:
        type LegacyProductBoundedField = Any  # type: ignore[no-redef]

    from ._concrete import Field, FieldSerializationModel


@final
class ProductField(BaseField):
    """A field that multiplies other fields lazily.

    Parameters
    ----------
    operands : `~collections.abc.Iterable` [ `BaseField` ]
        The fields to multiply together.
    """

    def __init__(self, operands: Iterable[Field]):
        self._operands = tuple(operands)
        if not self._operands:
            raise ValueError("At least one operand must be provided.")
        iterator = iter(self._operands)
        first = next(iterator)
        self._bounds = first.bounds
        self._unit = first.unit
        for operand in iterator:
            self._bounds = self._bounds.intersection(operand.bounds)
            if operand.unit is not None:
                if self._unit is None:
                    self._unit = operand.unit
                else:
                    self._unit *= operand.unit

    @property
    def bounds(self) -> Bounds:
        return self._bounds

    @property
    def unit(self) -> astropy.units.UnitBase | None:
        return self._unit

    @property
    def operands(self) -> tuple[Field, ...]:
        """The fields that are multiplied together
        (`tuple` [`BaseField`, ...]).
        """
        return self._operands

    def evaluate(
        self, *, x: np.ndarray, y: np.ndarray, quantity: bool = False
    ) -> np.ndarray | astropy.units.Quantity:
        iterator = iter(self._operands)
        first = next(iterator)
        result = first(x=x, y=y, quantity=False)
        for operand in iterator:
            result *= operand(x=x, y=y, quantity=False)
        if quantity:
            return result * self.unit
        return result

    def render(self, bbox: Box | None = None, *, dtype: np.typing.DTypeLike | None = None) -> Image:
        if bbox is None:
            bbox = self.bounds.bbox
        result = Image(1.0, bbox=bbox, dtype=dtype, unit=self.unit)
        for operand in self._operands:
            result.array *= operand.render(bbox, dtype=dtype).array
        return result

    def multiply_constant(
        self, factor: float | astropy.units.Quantity | astropy.units.UnitBase
    ) -> ProductField:
        new_operands = list(self._operands[:-1])
        new_operands.append(self._operands[-1] * factor)
        return ProductField(new_operands)

    def serialize(self, archive: OutputArchive[Any]) -> ProductFieldSerializationModel:
        """Serialize the field to an output archive."""
        return ProductFieldSerializationModel(
            operands=[operand.serialize(archive) for operand in self._operands]
        )

    @staticmethod
    def deserialize(model: ProductFieldSerializationModel, archive: InputArchive[Any]) -> ProductField:
        """Deserialize the field from an input archive."""
        from ._concrete import deserialize_field

        return ProductField([deserialize_field(operand, archive) for operand in model.operands])

    @staticmethod
    def _get_archive_tree_type(
        pointer_type: type[Any],
    ) -> type[ProductFieldSerializationModel]:
        """Return the serialization model type for this object for an archive
        type that uses the given pointer type.
        """
        return ProductFieldSerializationModel

    @staticmethod
    def from_legacy(
        legacy: LegacyProductBoundedField, unit: astropy.units.UnitBase | None = None
    ) -> ProductField:
        """Convert from a legacy `lsst.afw.math.ProductBoundedField`."""
        from ._concrete import field_from_legacy

        legacy_factors = legacy.getFactors()
        operands = [field_from_legacy(f) for f in legacy_factors[:-1]]
        operands.append(field_from_legacy(legacy_factors[-1], unit=unit))
        return ProductField(operands)

    def to_legacy(self) -> LegacyProductBoundedField:
        """Convert to a legacy `lsst.afw.math.ProductBoundedField`."""
        from lsst.afw.math import ProductBoundedField

        # Not all Field types have a to_legacy, since they don't all have an
        # afw analog.  But we just let that "no method" exception propagate.
        return ProductBoundedField(
            [operand.to_legacy() for operand in self._operands]  # type: ignore[union-attr]
        )


class ProductFieldSerializationModel(ArchiveTree):
    """Serialization model for `ProductField`."""

    operands: list[FieldSerializationModel] = pydantic.Field(default_factory=list)

    field_type: Literal["PRODUCT"] = "PRODUCT"

    def finish_deserialize(self, archive: InputArchive) -> ProductField:
        return ProductField.deserialize(self, archive)

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

__all__ = ("SumField", "SumFieldSerializationModel")

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal, final

import astropy.units
import numpy as np
import pydantic

from .._geom import Bounds, Box
from .._image import Image
from ..serialization import ArchiveTree, InputArchive, InvalidParameterError, OutputArchive
from ._base import BaseField

if TYPE_CHECKING:
    try:
        from lsst.afw.math import BackgroundList as LegacyBackgroundList
    except ImportError:
        type LegacyBackgroundList = Any  # type: ignore[no-redef]

    from ._concrete import Field, FieldSerializationModel


@final
class SumField(BaseField):
    """A field that sums other fields lazily.

    Parameters
    ----------
    operands : `~collections.abc.Iterable` [ `BaseField` ]
        The fields to sum together.
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
            if operand.unit is None:
                if self._unit is not None:
                    raise astropy.units.UnitConversionError(
                        "Cannot add a field with no units to a field with units."
                    )
            elif self._unit is None:
                raise astropy.units.UnitConversionError(
                    "Cannot add a field with units to a field with no units."
                )
            else:
                # Raise if these units are not sum-compatible.
                self._unit.to(operand.unit)

    @property
    def bounds(self) -> Bounds:
        return self._bounds

    @property
    def unit(self) -> astropy.units.UnitBase | None:
        return self._unit

    @property
    def operands(self) -> tuple[Field, ...]:
        """The fields that are summed together (`tuple` [`BaseField`, ...])."""
        return self._operands

    @property
    def is_constant(self) -> bool:
        return all(operand.is_constant for operand in self._operands)

    def evaluate(
        self, *, x: np.ndarray, y: np.ndarray, quantity: bool = False
    ) -> np.ndarray | astropy.units.Quantity:
        iterator = iter(self._operands)
        first = next(iterator)
        # We have to add quantities if this is a unit-aware field, as the
        # terms in the sum might have different-but-compatible units.
        result = first(x=x, y=y, quantity=(self.unit is not None))
        for operand in iterator:
            result += operand(x=x, y=y, quantity=(self.unit is not None))
        if self.unit is not None and not quantity:
            # Caller doesn't want a Quantity back.
            assert isinstance(result, astropy.units.Quantity)
            return result.to_value(self.unit)
        if self.unit is None and quantity:
            # Caller wants a Quantity back even though there's no units.
            return astropy.units.Quantity(result)
        return result

    def render(self, bbox: Box | None = None, *, dtype: np.typing.DTypeLike | None = None) -> Image:
        if bbox is None:
            bbox = self.bounds.bbox
        result = Image(0.0, bbox=bbox, dtype=dtype, unit=self.unit)
        for operand in self._operands:
            result.quantity += operand.render(bbox, dtype=dtype).quantity
        return result

    def multiply_constant(self, factor: float | astropy.units.Quantity | astropy.units.UnitBase) -> SumField:
        return SumField([operand * factor for operand in self._operands])

    def serialize(self, archive: OutputArchive[Any]) -> SumFieldSerializationModel:
        """Serialize the field to an output archive."""
        return SumFieldSerializationModel(operands=[operand.serialize(archive) for operand in self._operands])

    @staticmethod
    def _get_archive_tree_type(
        pointer_type: type[Any],
    ) -> type[SumFieldSerializationModel]:
        """Return the serialization model type for this object for an archive
        type that uses the given pointer type.
        """
        return SumFieldSerializationModel

    @staticmethod
    def from_legacy_background(
        legacy_background: LegacyBackgroundList,
        bounds: Bounds | None = None,
        unit: astropy.units.UnitBase | None = None,
    ) -> SumField:
        """Convert from a legacy `lsst.afw.math.BackgroundList` instance.

        Parameters
        ----------
        legacy
            Legacy background object to convert.
        bounds
            The bounds of the returned field, if they should be different from
            the bounding box of ``legacy_background``.
        unit
            The units of the returned field (`lsst.afw.math.BackgroundList`
            objects do not know their units).
        """
        from ._concrete import field_from_legacy_background

        return SumField(
            [field_from_legacy_background(b, bounds=bounds, unit=unit) for b, *_ in legacy_background]
        )


class SumFieldSerializationModel(ArchiveTree):
    """Serialization model for `SumField`."""

    operands: list[FieldSerializationModel] = pydantic.Field(default_factory=list)

    field_type: Literal["SUM"] = "SUM"

    def deserialize(self, archive: InputArchive, **kwargs: Any) -> SumField:
        """Deserialize the field from an input archive."""
        if kwargs:
            raise InvalidParameterError(f"Unrecognized parameters for SumField: {set(kwargs.keys())}.")
        return SumField([operand.deserialize(archive) for operand in self.operands])

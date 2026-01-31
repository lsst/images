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
    "ArrayReferenceModel",
    "ArrayReferenceQuantityModel",
    "InlineArray",
    "InlineArrayModel",
    "InlineArrayQuantity",
    "InlineArrayQuantityModel",
    "Quantity",
    "QuantityModel",
    "Time",
    "TimeModel",
    "Unit",
)

from typing import Annotated, Any, ClassVar, Literal

import astropy.time
import astropy.units
import numpy as np
import pydantic
import pydantic_core.core_schema as pcs

from ._dtypes import NumberType


class _UnitSerialization:
    """Pydantic hooks for unit serialization.

    This class provides implementations for the `Unit` type alias for
    `astropy.unit.Unit` that adds Pydantic serialization and validation.
    """

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: pydantic.GetCoreSchemaHandler
    ) -> pcs.CoreSchema:
        from_str_schema = pcs.chain_schema(
            [
                pcs.str_schema(),
                pcs.no_info_plain_validator_function(cls.from_str),
            ]
        )
        return pcs.json_or_python_schema(
            json_schema=from_str_schema,
            python_schema=pcs.union_schema([pcs.is_instance_schema(astropy.units.UnitBase), from_str_schema]),
            serialization=pcs.plain_serializer_function_ser_schema(cls.to_str),
        )

    @classmethod
    def from_str(cls, value: str) -> astropy.units.UnitBase:
        return astropy.units.Unit(value, format="vounit")

    @staticmethod
    def to_str(unit: astropy.units.UnitBase) -> str:
        return unit.to_string("vounit")


type Unit = Annotated[
    astropy.units.UnitBase,
    _UnitSerialization,
    pydantic.WithJsonSchema(
        {
            "type": "string",
            "$schema": "http://stsci.edu/schemas/yaml-schema/draft-01",
            "id": "http://stsci.edu/schemas/asdf/unit/unit-1.0.0",
            "tag": "!unit/unit-1.0.0",
        }
    ),
]


class ArrayReferenceModel(pydantic.BaseModel):
    """Model for the subset of the ASDF 'ndarray' schema, in the case where the
    array data is stored elsewhere.
    """

    source: str | int
    shape: list[int]
    datatype: NumberType
    byteorder: Literal["big"] = "big"

    model_config = pydantic.ConfigDict(
        json_schema_extra={
            "$schema": "http://stsci.edu/schemas/yaml-schema/draft-01",
            "id": "http://stsci.edu/schemas/asdf/core/ndarray-1.1.0",
            "tag": "!core/ndarray-1.1.0",
        }
    )

    source_is_table: ClassVar[Literal[False]] = False


class InlineArrayModel(pydantic.BaseModel):
    """Model for the subset of the ASDF 'ndarray' schema, in the case where the
    array data is stored inline.
    """

    data: list[Any]
    datatype: NumberType

    model_config = pydantic.ConfigDict(
        json_schema_extra={
            "$schema": "http://stsci.edu/schemas/yaml-schema/draft-01",
            "id": "http://stsci.edu/schemas/asdf/core/ndarray-1.1.0",
            "tag": "!core/ndarray-1.1.0",
        }
    )


class _InlineArraySerialization:
    """Pydantic hooks for array serialization.

    This class provides implementations for the `Array` type alias for
    `numpy.ndarray` that adds Pydantic serialization and validation.
    """

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: pydantic.GetCoreSchemaHandler
    ) -> pcs.CoreSchema:
        from_model_schema = pcs.chain_schema(
            [
                handler(InlineArrayModel),
                pcs.no_info_plain_validator_function(cls.from_model),
            ]
        )
        return pcs.json_or_python_schema(
            json_schema=from_model_schema,
            python_schema=pcs.union_schema([pcs.is_instance_schema(np.ndarray), from_model_schema]),
            serialization=pcs.plain_serializer_function_ser_schema(cls.to_model),
        )

    @classmethod
    def from_model(cls, model: InlineArrayModel) -> np.ndarray:
        return np.array(model.data, dtype=model.datatype.to_numpy())

    @classmethod
    def to_model(cls, array: np.ndarray) -> InlineArrayModel:
        datatype = NumberType.from_numpy(array.dtype)
        return InlineArrayModel(data=array.tolist(), datatype=datatype)


type InlineArray = Annotated[np.ndarray, _InlineArraySerialization]


class _GenericQuantityModel[T: InlineArrayModel | ArrayReferenceModel | pydantic.StrictFloat](
    pydantic.BaseModel
):
    """Model for a subset of the ASDF 'quantity' schema."""

    value: T
    unit: Unit

    model_config = pydantic.ConfigDict(
        json_schema_extra={
            "$schema": "http://stsci.edu/schemas/yaml-schema/draft-01",
            "id": "http://stsci.edu/schemas/asdf/unit/quantity-1.2.0",
            "tag": "!unit/quantity-1.2.0",
        }
    )


class QuantityModel(_GenericQuantityModel[pydantic.StrictFloat]):
    """Model for a subset of the ASDF 'quantity' schema for scalars."""


class InlineArrayQuantityModel(_GenericQuantityModel[InlineArrayModel]):
    """Model for a subset of the ASDF 'quantity' schema for inline arrays."""


class ArrayReferenceQuantityModel(_GenericQuantityModel[ArrayReferenceModel]):
    """Model for a subset of the ASDF 'quantity' schema for external arrays."""


class _QuantitySerialization:
    """Pydantic hooks for scalar quantity serialization."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: pydantic.GetCoreSchemaHandler
    ) -> pcs.CoreSchema:
        from_model_schema = pcs.chain_schema(
            [
                handler(QuantityModel),
                pcs.no_info_plain_validator_function(cls.from_model),
            ]
        )
        return pcs.json_or_python_schema(
            json_schema=from_model_schema,
            python_schema=pcs.union_schema(
                [pcs.is_instance_schema(astropy.units.Quantity), from_model_schema]
            ),
            serialization=pcs.plain_serializer_function_ser_schema(cls.to_model),
        )

    @classmethod
    def from_model(cls, model: QuantityModel) -> astropy.units.Quantity:
        return astropy.units.Quantity(model.value, unit=model.unit)

    @classmethod
    def to_model(cls, quantity: astropy.units.Quantity) -> QuantityModel:
        assert quantity.isscalar
        return QuantityModel(value=quantity.to_value(), unit=_UnitSerialization.to_str(quantity.unit))


type Quantity = Annotated[astropy.units.Quantity, _QuantitySerialization]


class _InlineArrayQuantitySerialization:
    """Pydantic hooks for inline array quantity serialization."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: pydantic.GetCoreSchemaHandler
    ) -> pcs.CoreSchema:
        from_model_schema = pcs.chain_schema(
            [
                handler(InlineArrayQuantityModel),
                pcs.no_info_plain_validator_function(cls.from_model),
            ]
        )
        return pcs.json_or_python_schema(
            json_schema=from_model_schema,
            python_schema=pcs.union_schema(
                [pcs.is_instance_schema(astropy.units.Quantity), from_model_schema]
            ),
            serialization=pcs.plain_serializer_function_ser_schema(cls.to_model),
        )

    @classmethod
    def from_model(cls, model: InlineArrayQuantityModel) -> astropy.units.Quantity:
        return astropy.units.Quantity(_InlineArraySerialization.from_model(model.value), unit=model.unit)

    @classmethod
    def to_model(cls, quantity: astropy.units.Quantity) -> InlineArrayQuantityModel:
        assert quantity.isscalar
        return InlineArrayQuantityModel(
            value=_InlineArraySerialization.to_model(quantity.to_value()),
            unit=_UnitSerialization.to_str(quantity.unit),
        )


type InlineArrayQuantity = Annotated[astropy.units.Quantity, _InlineArrayQuantitySerialization]


class TimeModel(pydantic.BaseModel):
    """Model for a subset of the ASDF 'time' schema."""

    value: str
    scale: Literal["utc", "tai"]
    format: Literal["iso"] = "iso"

    model_config = pydantic.ConfigDict(
        json_schema_extra={
            "$schema": "http://stsci.edu/schemas/yaml-schema/draft-01",
            "id": "http://stsci.edu/schemas/asdf/time/time-1.2.0",
            "tag": "!time/time-1.2.0",
        }
    )


class _TimeSerialization:
    """Pydantic hooks for time serialization.

    This class provides implementations for the `Time` type alias for
    `astropy.time.Time` that adds Pydantic serialization and validation.
    """

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: pydantic.GetCoreSchemaHandler
    ) -> pcs.CoreSchema:
        from_model_schema = pcs.chain_schema(
            [
                TimeModel.__pydantic_core_schema__,
                pcs.no_info_plain_validator_function(cls.from_model),
            ]
        )
        return pcs.json_or_python_schema(
            json_schema=from_model_schema,
            python_schema=pcs.union_schema([pcs.is_instance_schema(astropy.time.Time), from_model_schema]),
            serialization=pcs.plain_serializer_function_ser_schema(cls.to_model, info_arg=False),
        )

    @classmethod
    def from_model(cls, model: TimeModel) -> astropy.time.Time:
        return astropy.time.Time(model.value, scale=model.scale, format=model.format)

    @classmethod
    def to_model(cls, time: astropy.time.Time) -> TimeModel:
        if time.scale != "utc" and time.scale != "tai":
            time = time.tai
        return TimeModel(value=time.to_value("iso"), scale=time.scale, format="iso")


type Time = Annotated[astropy.time.Time, _TimeSerialization]

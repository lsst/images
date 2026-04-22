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
    "Field",
    "FieldSerializationModel",
    "deserialize_field",
    "field_from_legacy",
    "field_from_legacy_background",
)

from typing import TYPE_CHECKING, Annotated, Any

import astropy.units
import pydantic

from ..serialization import InputArchive
from ._chebyshev import ChebyshevField, ChebyshevFieldSerializationModel
from ._product import ProductField, ProductFieldSerializationModel
from ._spline import SplineField, SplineFieldSerializationModel
from ._sum import SumField, SumFieldSerializationModel

if TYPE_CHECKING:
    try:
        from lsst.afw.math import BackgroundList as LegacyBackgroundList
        from lsst.afw.math import BackgroundMI as LegacyBackground
        from lsst.afw.math import BoundedField as LegacyBoundedField
    except ImportError:
        type LegacyBoundedField = Any  # type: ignore[no-redef]
        type LegacyBackground = Any  # type: ignore[no-redef]
        type LegacyBackgroundList = Any  # type: ignore[no-redef]


# Since Sphinx can't handle doc links to type aliases, whenever we annotate
# a type as `Field`, we override the docs to say `BaseField`, since
# `BaseField` is a base class that serves as a much more useful doc link, and
# because the hierarchy is closed they're equivalent.  But we have to use
# `Field` in the type annotations because there's no way to declare to MyPy
# et all that the hierarchy is closed.

type Field = ChebyshevField | ProductField | SplineField | SumField
type FieldSerializationModel = Annotated[
    ChebyshevFieldSerializationModel
    | ProductFieldSerializationModel
    | SplineFieldSerializationModel
    | SumFieldSerializationModel,
    pydantic.Field(discriminator="field_type"),
]


def deserialize_field(model: FieldSerializationModel, archive: InputArchive[Any]) -> Field:
    """Deserialize a field from a serialization model of unknown type."""
    return model.finish_deserialize(archive)


ProductFieldSerializationModel.model_rebuild()
SumFieldSerializationModel.model_rebuild()


def field_from_legacy(
    legacy_bounded_field: LegacyBoundedField, unit: astropy.units.UnitBase | None = None
) -> Field:
    """Convert a legacy `lsst.afw.math.BoundedField` subclass to a `BaseField`
    object.
    """
    from lsst.afw.math import ChebyshevBoundedField, ProductBoundedField

    match legacy_bounded_field:
        case ChebyshevBoundedField():
            return ChebyshevField.from_legacy(legacy_bounded_field, unit=unit)
        case ProductBoundedField():
            return ProductField.from_legacy(legacy_bounded_field, unit=unit)
        case _:
            raise NotImplementedError(
                f"Conversion from {type(legacy_bounded_field).__name__} is not supported."
            )


def field_from_legacy_background(
    legacy_background: LegacyBackground | LegacyBackgroundList, unit: astropy.units.UnitBase | None = None
) -> Field:
    """Convert a legacy `lsst.afw.math.Background` or
    `lsst.afw.math.BackgroundList` instance to a `BaseField` object.
    """
    from lsst.afw.math import ApproximateControl, BackgroundList

    if isinstance(legacy_background, BackgroundList):
        return SumField.from_legacy_background(legacy_background)

    approx_control = legacy_background.getBackgroundControl().getApproximateControl()
    if approx_control.getStyle() == ApproximateControl.UNKNOWN:
        return SplineField.from_legacy_background(legacy_background, unit=unit)
    else:
        return ChebyshevField.from_legacy_background(legacy_background, unit=unit)

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
    "field_from_legacy",
    "field_from_legacy_background",
    "field_from_legacy_photo_calib",
)

from typing import TYPE_CHECKING, Annotated, Any

import astropy.units
import numpy as np
import pydantic

from .._geom import Bounds
from ._chebyshev import ChebyshevField, ChebyshevFieldSerializationModel
from ._product import ProductField, ProductFieldSerializationModel
from ._spline import SplineField, SplineFieldSerializationModel
from ._sum import SumField, SumFieldSerializationModel

if TYPE_CHECKING:
    try:
        from lsst.afw.image import PhotoCalib as LegacyPhotoCalib
        from lsst.afw.math import BackgroundList as LegacyBackgroundList
        from lsst.afw.math import BackgroundMI as LegacyBackground
        from lsst.afw.math import BoundedField as LegacyBoundedField
    except ImportError:
        type LegacyBoundedField = Any  # type: ignore[no-redef]
        type LegacyBackground = Any  # type: ignore[no-redef]
        type LegacyBackgroundList = Any  # type: ignore[no-redef]
        type LegacyPhotoCalib = Any  # type: ignore[no-redef]


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


ProductFieldSerializationModel.model_rebuild()
SumFieldSerializationModel.model_rebuild()


def field_from_legacy(
    legacy_bounded_field: LegacyBoundedField,
    bounds: Bounds | None = None,
    unit: astropy.units.UnitBase | None = None,
) -> Field:
    """Convert a legacy `lsst.afw.math.BoundedField` subclass to a `BaseField`
    object.

    Parameters
    ----------
    legacy_bounded_field
        Legacy field to convert.
    bounds
        The bounds of the returned field, if they should be different from
        the bounding box of ``legacy``.
    unit
        The units of the returned field (`lsst.afw.math.BoundedField`
        objects do not know their units).
    """
    from lsst.afw.math import ChebyshevBoundedField, ProductBoundedField

    match legacy_bounded_field:
        case ChebyshevBoundedField():
            return ChebyshevField.from_legacy(legacy_bounded_field, unit=unit, bounds=bounds)
        case ProductBoundedField():
            return ProductField.from_legacy(legacy_bounded_field, unit=unit, bounds=bounds)
        case _:
            raise NotImplementedError(
                f"Conversion from {type(legacy_bounded_field).__name__} is not supported."
            )


def field_from_legacy_background(
    legacy_background: LegacyBackground | LegacyBackgroundList,
    bounds: Bounds | None = None,
    unit: astropy.units.UnitBase | None = None,
) -> Field:
    """Convert a legacy `lsst.afw.math.Background` or
    `lsst.afw.math.BackgroundList` instance to a `BaseField` object.

    Parameters
    ----------
    legacy_background
        Legacy background object to convert.
    bounds
        The bounds of the returned field, if they should be different from
        the bounding box of ``legacy_background``.
    unit
        The units of the returned field (`lsst.afw.math.Background`
        objects do not know their units).
    """
    from lsst.afw.math import ApproximateControl, BackgroundList

    if isinstance(legacy_background, BackgroundList):
        return SumField.from_legacy_background(legacy_background)

    approx_control = legacy_background.getBackgroundControl().getApproximateControl()
    if approx_control.getStyle() == ApproximateControl.UNKNOWN:
        return SplineField.from_legacy_background(legacy_background, unit=unit)
    else:
        return ChebyshevField.from_legacy_background(legacy_background, unit=unit)


def field_from_legacy_photo_calib(
    legacy_photo_calib: LegacyPhotoCalib,
    bounds: Bounds,
    post_isr_unit: astropy.units.UnitBase = astropy.units.electron,
) -> Field | None:
    """Convert a legacy `lsst.afw.image.PhotoCalib` into a `BaseField` object.

    Parameters
    ----------
    legacy_photo_calib
        Calibration object to convert.
    bounds
        Bounds of the returned field.
    post_isr_unit
        The instrumental units the legacy calibration transforms from.  These
        will be used as the denominator of the units of the returned field,
        with ``astropy.units.nJy`` as the numerator.

    Returns
    -------
    `BaseField` | `None`
        A field that transforms instrumental units to ``nJy``, or `None` if
        the given calibration object was an identity mapping for a legacy
        image that already had ``nJy`` pixels.
    """
    calibration_mean = legacy_photo_calib.getCalibrationMean()
    if legacy_photo_calib._isConstant:
        if calibration_mean == 1.0:
            # This image's pixels have been calibrated to nJy
            # already, which means the calibration *from* post-ISR
            # electrons that we want is stored elsewhere.
            return None
        else:
            return ChebyshevField(
                bounds,
                np.array([[calibration_mean]], dtype=np.float64),
                unit=astropy.units.nJy / post_isr_unit,
            )
    else:
        normalized_field = field_from_legacy(
            legacy_photo_calib.computeScaledCalibration(),
            unit=astropy.units.nJy / post_isr_unit,
            bounds=bounds,
        )
        return normalized_field * calibration_mean

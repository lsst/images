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
    "ApertureCorrectionMap",
    "ApertureCorrectionMapSerializationModel",
    "aperture_corrections_from_legacy",
    "aperture_corrections_to_legacy",
)

from typing import TYPE_CHECKING, Any, final

import pydantic

from .fields import Field, FieldSerializationModel, field_from_legacy
from .serialization import ArchiveTree, InputArchive, OutputArchive

if TYPE_CHECKING:
    try:
        from lsst.afw.image import ApCorrMap as LegacyApCorrMap
    except ImportError:
        type LegacyApCorrMap = Any  # type: ignore[no-redef]

type ApertureCorrectionMap = dict[str, Field]


def aperture_corrections_from_legacy(legacy_ap_corr_map: LegacyApCorrMap) -> ApertureCorrectionMap:
    """Convert a `lsst.afw.image.ApCorrMap` instance to a `dict` mapping
    `str` algorithm name to `~.fields.BaseField`.
    """
    return {name: field_from_legacy(legacy_field) for name, legacy_field in legacy_ap_corr_map.items()}


def aperture_corrections_to_legacy(aperture_corrections: ApertureCorrectionMap) -> LegacyApCorrMap:
    """Convert from a `dict` (mapping `str` algorithm name to
    `~.fields.BaseField`) to a `lsst.afw.image.ApCorrMap` instance.
    """
    from lsst.afw.image import ApCorrMap

    result = ApCorrMap()
    for name, field in aperture_corrections.items():
        # Not all Field types have a to_legacy, but the ones we care about do;
        # if we're wrong about that, the AttributeError is probably the best
        # we can do.
        result[name] = field.to_legacy()  # type: ignore[union-attr]
    return result


@final
class ApertureCorrectionMapSerializationModel(ArchiveTree):
    """Serialization model for aperture correction maps.

    Notes
    -----
    The in-memory aperture correction map type is just a `dict` from `str`
    to `~.fields.BaseField`, so the `serialize` and `deserialize` methods are
    defined here.
    """

    fields: dict[str, FieldSerializationModel] = pydantic.Field(
        default_factory=dict,
        description="Mapping from flux algorithm name to the aperture correction field for that algorithm.",
    )

    @staticmethod
    def serialize(
        map: ApertureCorrectionMap, archive: OutputArchive[Any]
    ) -> ApertureCorrectionMapSerializationModel:
        """Write an aperture correction map to an archive."""
        result = ApertureCorrectionMapSerializationModel()
        for name, field in map.items():
            result.fields[name] = field.serialize(archive)
        return result

    def deserialize(self, archive: InputArchive[Any]) -> ApertureCorrectionMap:
        """Read an aperture correction map from an archive."""
        from .fields import deserialize_field

        return {name: deserialize_field(field, archive) for name, field in self.fields.items()}

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
    "ArchiveReadError",
    "ArchiveTree",
    "ButlerInfo",
    "MetadataValue",
    "OpaqueArchiveMetadata",
    "ReadResult",
    "no_header_updates",
)

import operator
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, Self

import astropy.table
import astropy.units
import pydantic

from .._geom import Box
from ..utils import is_none

try:
    from lsst.daf.butler import DatasetProvenance, SerializedDatasetRef
except ImportError:
    type DatasetProvenance = Any  # type: ignore[no-redef]
    type SerializedDatasetRef = Any  # type: ignore[no-redef]

if TYPE_CHECKING:
    import astropy.io.fits


type MetadataValue = (
    pydantic.StrictInt | pydantic.StrictFloat | pydantic.StrictStr | pydantic.StrictBool | None
)


class ButlerInfo(pydantic.BaseModel):
    """Information about a butler dataset."""

    dataset: SerializedDatasetRef
    provenance: DatasetProvenance = pydantic.Field(default_factory=DatasetProvenance)


class ArchiveTree(
    pydantic.BaseModel, ser_json_inf_nan="constants", ser_json_bytes="base64", val_json_bytes="base64"
):
    """An intermediate base class of `pydantic.BaseModel` that should be used
    for all objects that may be used as the top-level tree models written to
    archives.
    """

    metadata: dict[str, MetadataValue] = pydantic.Field(
        default_factory=dict, description="Additional unstructured metadata.", exclude_if=operator.not_
    )
    butler_info: ButlerInfo | None = pydantic.Field(
        default=None,
        description="Information about the butler dataset backed by this file.",
        exclude_if=is_none,
    )


class ReadResult[T: Any](NamedTuple):
    """A struct that can be used to return both a deserialized object and
    metadata associated with it, even when the in-memory type cannot hold
    metadata.
    """

    deserialized: T
    """The deserialized object itself."""

    metadata: dict[str, MetadataValue]
    """Additional flexible metadata stored with the object."""

    butler_info: ButlerInfo | None
    """Butler provenance information for the dataset this file backs."""


class ArchiveReadError(RuntimeError):
    """Exception raised when the contents of an archive cannot be read."""


class OpaqueArchiveMetadata(Protocol):
    """Interface for opaque archive metadata.

    In addition to implementing the methods defined here, all implementations
    must be pickleable.
    """

    def copy(self) -> Self | None:
        """Copy, reference, or discard metadata when its holding object is
        copied.
        """
        ...

    def subset(self, bbox: Box) -> Self | None:
        """Copy, reference, or discard metadata when a subset of its its
        holding object is extracted.
        """
        ...


def no_header_updates(header: astropy.io.fits.Header) -> None:
    """Do not make any modifications to the given FITS header."""

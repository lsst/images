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
    "InvalidComponentError",
    "InvalidParameterError",
    "JsonRef",
    "MetadataValue",
    "OpaqueArchiveMetadata",
    "ReadResult",
    "no_header_updates",
)

import operator
from abc import ABC, abstractmethod
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

    from ._input_archive import InputArchive


type MetadataValue = (
    pydantic.StrictInt | pydantic.StrictFloat | pydantic.StrictStr | pydantic.StrictBool | None
)


class ButlerInfo(pydantic.BaseModel):
    """Information about a butler dataset."""

    dataset: SerializedDatasetRef
    provenance: DatasetProvenance = pydantic.Field(default_factory=DatasetProvenance)


class JsonRef(pydantic.BaseModel, serialize_by_alias=True):
    """Pydantic model for JSON Reference / Pointer (IETF RFC 6901).

    Notes
    -----
    This model does not do any of the escaping or special-character
    interpretation required by the spec; it assumes that's already been done,
    so its job is *just* putting a ``$ref`` field inside another model.
    """

    ref: str = pydantic.Field(alias="$ref")


class ArchiveTree(
    pydantic.BaseModel, ABC, ser_json_inf_nan="constants", ser_json_bytes="base64", val_json_bytes="base64"
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
    indirect: list[Any] = pydantic.Field(
        default_factory=list,
        description="Serialized nested objects that may be saved or read more than once.",
        exclude_if=operator.not_,
    )

    @abstractmethod
    def deserialize(self, archive: InputArchive[Any], **kwargs: Any) -> Any:
        """Return the in-memory object that was serialized to this tree.

        Parameters
        ----------
        archive
            The input archive to read from.
        **kwargs
            Additional keyword arguments specific to this type.

        Raises
        ------
        ~lsst.images.serialization.InvalidParameterError
            Raised for unsupported ``**kwargs``.

        Notes
        -----
        Subclass implementations may take additional keyword-only arguments.
        Callers that invoke this method without knowing what those might be
        should catch `TypeError` and re-raise as
        `~lsst.images.serialization.InvalidParameterError` if they pass
        additional keyword arguments.
        """
        raise NotImplementedError()

    def deserialize_component(self, component: str, archive: InputArchive[Any], **kwargs: Any) -> Any:
        """Return a component in-memory object that was serialized to this
        tree.

        Parameters
        ----------
        component
            Name of the component to read.
        archive
            The input archive to read from.
        **kwargs
            Additional keyword arguments specific to this type.

        Raises
        ------
        ~lsst.images.serialization.InvalidComponentError
            Raise if ``component`` is not recognized.
        ~lsst.images.serialization.InvalidParameterError
            Raised for unsupported ``**kwargs``.

        Notes
        -----
        The default implementation for this method tries to get an attribute
        with the component's name from ``self``, and then:

         - returns `None` if it is `None`;
         - calls `deserialize` on that object if it is also an
           `~lsst.images.serialization.ArchiveTree`;
         - returns it directly otherwise.

        If there is no such attribute, it raises
        `~lsst.images.serialization.InvalidComponentError`.

        ``**kwargs`` are forwarded to component `deserialize` methods, but
        are otherwise not checked.  Subclasses are generally expected to
        implement this method to do that checking and handle any components
        for which the other will not work, and then delegate to `super` at
        the end.
        """
        try:
            component_model = getattr(self, component)
        except AttributeError:
            raise InvalidComponentError(
                f"Component {component!r} is not recognized by {type(self).__name__}."
            ) from None
        if component_model is None:
            return None
        if isinstance(component_model, ArchiveTree):
            return component_model.deserialize(archive, **kwargs)
        return component_model


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


class InvalidParameterError(ArchiveReadError):
    """Exception raised by `ArchiveTree.deserialize` or
    `ArchiveTree.deserialize_component` when passed an invalid keyword
    argument.
    """


class InvalidComponentError(ArchiveReadError):
    """Exception `ArchiveTree.deserialize_component` when passed an invalid
    component name.
    """


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


def _parse_major(version: str) -> int:
    """Return the integer major component of a major.minor.patch string.

    Raises
    ------
    ArchiveReadError
        If ``version`` is not a non-empty string of the form
        ``major.minor.patch`` with integer components.
    """
    if not isinstance(version, str) or not version:
        raise ArchiveReadError(f"Schema version {version!r} is not a non-empty string.")
    head = version.split(".", 1)[0]
    try:
        return int(head)
    except ValueError as exc:
        raise ArchiveReadError(f"Schema version {version!r} has non-integer major.") from exc


def _check_compat(
    name: str,
    on_disk_version: str,
    on_disk_min_read: int,
    in_code_version: str,
) -> None:
    """Raise `ArchiveReadError` if a tree written with the given
    schema_version/min_read_version cannot be read by the current code.

    See ``docs/superpowers/specs/2026-05-15-schema-versioning-design.md``
    §4.2 for the rule.
    """
    in_code_major = _parse_major(in_code_version)
    if on_disk_min_read > in_code_major:
        raise ArchiveReadError(
            f"{name}: tree requires reader major >= {on_disk_min_read}; this release is {in_code_version}."
        )


def _check_format_version(name: str, on_disk: int, in_code: int) -> None:
    """Raise `ArchiveReadError` if a backend file's container layout
    version is newer than this release knows how to read.
    """
    if on_disk > in_code:
        raise ArchiveReadError(
            f"{name}: on-disk container format version {on_disk} is "
            f"newer than this release ({in_code}); cannot read."
        )

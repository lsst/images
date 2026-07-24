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

__all__ = ("Background", "BackgroundMap", "BackgroundMapSerializationModel")

import dataclasses
import sys
from collections.abc import Iterable, Iterator, Mapping
from typing import Any, ClassVar, cast, final

import pydantic

from .describe import DescribableMixin, FieldRole, Report, ReportField, ReportTable
from .fields import Field, FieldSerializationModel
from .serialization import ArchiveTree, InputArchive, InvalidParameterError, OutputArchive


@dataclasses.dataclass(frozen=True)
class Background:
    """A named background model and optional description."""

    name: str
    """A unique name for this background."""

    field: Field
    """The actual background model itself."""

    description: str = ""
    """A description of how the background model was produced and/or how it
    should be used.
    """


class BackgroundMap(DescribableMixin, Mapping[str, Background]):
    """A mapping of background models associated with an image.

    Unlike most image characterization objects, the best background model
    often depends on the science case, and hence we may want to associate more
    than one with an image.

    Parameters
    ----------
    backgrounds
        Background models to include in the map, keyed by their name.
    subtracted
        Name of the background that has been subtracted from the image, or
        `None` if no background has been subtracted.
    """

    def __init__(self, backgrounds: Iterable[Background] = (), subtracted: str | None = None) -> None:
        self._backgrounds = {b.name: b for b in backgrounds}
        self._subtracted = subtracted
        if isinstance(self._subtracted, str) and self._subtracted not in self._backgrounds:
            raise KeyError(f"Subtracted background {self._subtracted!r} not present in map.")

    @property
    def subtracted(self) -> Background | None:
        """The background subtracted from this image (`Background` | `None`).

        Notes
        -----
        If `None`, none of the backgrounds in this map were subtracted from
        the image.  This does not necessarily mean no background at all was
        subtracted (e.g. in a coadd, backgrounds are generally subtracted from
        the input images before they are combined, and the sum of those
        backgrounds may not be available in a coadd background map.)
        """
        if self._subtracted is None:
            return None
        return self._backgrounds[self._subtracted]

    def __iter__(self) -> Iterator[str]:
        return iter(self._backgrounds.keys())

    def __getitem__(self, key: str) -> Background:
        return self._backgrounds[key]

    def __len__(self) -> int:
        return len(self._backgrounds)

    if "sphinx" in sys.modules:
        # The Python standard library docstring is not valid reStructuredText,
        # but the true signature (with involves overloads) is complicated.
        def get[V](self, key: str, default: V | None = None) -> Background | V | None:  # type: ignore
            """Return the background with the given key or the given default
            value.
            """
            return super().get(key, default)

    def copy(self) -> BackgroundMap:
        """Return a copy of the background map."""
        return BackgroundMap(self.values(), self._subtracted)

    def add(self, name: str, field: Field, description: str = "", *, is_subtracted: bool = False) -> None:
        """Add a new background to the map.

        Parameters
        ----------
        name
            Unique name for this background model.
        field
            The background field itself.
        description
            A description of how this background model was produced and/or how
            it should be used.
        is_subtracted
            Whether this background is the one that was subtracted from the
            image this background map is attached to.

        Notes
        -----
        There are no guards against ``is_subtracted=True`` being passed for
        multiple different backgrounds; correctness is up to the caller.  Note
        that we only allow one background to be subtracted at once
        (incremental backgrounds should be modeled via `.fields.SumField`, not
        multiple named entries in this map).
        """
        if name in self._backgrounds:
            raise KeyError(f"A background with name {name!r} already exists.")
        self._backgrounds[name] = Background(name, field, description)
        if is_subtracted:
            self._subtracted = name

    def _describe(self, **kwargs: Any) -> Report:
        """Return a `Report` describing this background map.

        Parameters
        ----------
        **kwargs
            Unused; accepted for interface compatibility.
        """
        subtracted_name = self._subtracted
        rows = [[name, name == subtracted_name] for name in self._backgrounds]
        return Report(
            type_name="BackgroundMap",
            summary=f"BackgroundMap({len(self)} background{'s' if len(self) != 1 else ''})",
            fields=[
                ReportField(label="subtracted", value=subtracted_name, role=FieldRole.DERIVED),
            ],
            tables=[
                ReportTable(
                    title="Backgrounds",
                    columns=["Name", "Subtracted"],
                    rows=rows,
                )
            ],
        )

    def serialize(self, archive: OutputArchive[Any]) -> BackgroundMapSerializationModel:
        """Write a background map to an archive.

        Parameters
        ----------
        archive
            Archive to write to.
        """
        result = BackgroundMapSerializationModel(subtracted=self._subtracted)
        for name, background in self.items():
            result.fields[name] = cast(
                FieldSerializationModel,
                archive.serialize_direct(f"fields/{name}", background.field.serialize),
            )
            result.descriptions[name] = background.description
        return result


@final
class BackgroundMapSerializationModel(ArchiveTree):
    """Serialization model for background maps."""

    SCHEMA_NAME: ClassVar[str] = "background_map"
    SCHEMA_VERSION: ClassVar[str] = "1.0.0"
    MIN_READ_VERSION: ClassVar[int] = 1
    PUBLIC_TYPE: ClassVar[type] = BackgroundMap

    fields: dict[str, FieldSerializationModel] = pydantic.Field(
        default_factory=dict,
        description="Mapping from background model name to the model field itself.",
    )

    descriptions: dict[str, str] = pydantic.Field(
        default_factory=dict,
        description="Mapping from background model name to its description.",
    )

    subtracted: str | None = pydantic.Field(
        default=None,
        description="Name of the background that was subtracted, or None if no background was subtracted.",
    )

    def deserialize(self, archive: InputArchive[Any], **kwargs: Any) -> BackgroundMap:
        if kwargs:
            raise InvalidParameterError(f"Unrecognized parameters for BackgroundMap: {set(kwargs.keys())}.")
        return BackgroundMap(
            [
                Background(
                    name=name, field=field.deserialize(archive), description=self.descriptions.get(name, "")
                )
                for name, field in self.fields.items()
            ],
            subtracted=self.subtracted,
        )

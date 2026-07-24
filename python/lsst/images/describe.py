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
    "Describable",
    "DescribableMixin",
    "FieldRole",
    "Report",
    "ReportField",
    "ReportTable",
)

import dataclasses
import enum
from typing import Any, Protocol, runtime_checkable


class FieldRole(enum.Enum):
    """Whether a report field reconstructs the object or merely informs."""

    ARG = "arg"
    """A constructor argument; part of the object's identity and reproduced
    in ``repr``.
    """

    DERIVED = "derived"
    """An informational or computed value; never reproduced in ``repr``."""


@dataclasses.dataclass(frozen=True)
class ReportField:
    """A single labeled value in a `Report`."""

    label: str
    """Human-readable label for the value."""

    value: Any
    """Display value for the field."""

    unit: str | None = None
    """Unit rendered after the value, if any."""

    repr_value: str | None = None
    """Eval-ish fragment used in ``repr``; defaults to ``repr(value)``."""

    role: FieldRole = FieldRole.ARG
    """Whether this field feeds ``repr`` (`FieldRole.ARG`) or not."""

    positional: bool = False
    """If `True`, ``repr`` emits the value positionally (no ``label=``)."""


@dataclasses.dataclass(frozen=True)
class ReportTable:
    """Homogeneous columnar data rendered as an aligned table."""

    title: str | None
    """Title shown above the table, if any."""

    columns: list[str]
    """Header row labels."""

    rows: list[list[Any]]
    """One list of cell values per row, aligned to ``columns``."""

    role: FieldRole = FieldRole.DERIVED
    """Tables never feed ``repr``; always `FieldRole.DERIVED`."""


@dataclasses.dataclass
class Report:
    """A renderer-agnostic description of an object."""

    type_name: str
    """Name of the described type."""

    title: str | None = None
    """Optional headline shown above the fields."""

    summary: str | None = None
    """Optional one-line hint used by ``__str__``."""

    fields: list[ReportField] = dataclasses.field(default_factory=list)
    """Ordered labeled values."""

    tables: list[ReportTable] = dataclasses.field(default_factory=list)
    """Ordered tables of columnar data."""

    children: dict[str, Report] = dataclasses.field(default_factory=dict)
    """Named nested sub-reports."""

    def to_repr(self) -> str:
        """Return an eval-ish ``repr`` string built from ``ARG`` fields."""
        parts: list[str] = []
        for field in self.fields:
            if field.role is not FieldRole.ARG:
                continue
            value = field.repr_value if field.repr_value is not None else repr(field.value)
            parts.append(value if field.positional else f"{field.label}={value}")
        return f"{self.type_name}({', '.join(parts)})"

    def to_str(self) -> str:
        """Return a compact one-line summary."""
        if self.summary is not None:
            return self.summary
        args = [str(field.value) for field in self.fields if field.role is FieldRole.ARG]
        inner = ", ".join(args[:3])
        return f"{self.type_name}({inner})"


@runtime_checkable
class Describable(Protocol):
    """Protocol for objects that can produce a structured `Report`."""

    def describe(self) -> Report:
        """Return a structured description of this object."""
        ...


class DescribableMixin:
    """Base class for objects that implement the `Describable` protocol."""

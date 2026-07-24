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
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from rich.console import Console
from rich.table import Table
from rich.tree import Tree

if TYPE_CHECKING:
    from rich.console import RenderableType


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

    def _field_line(self, field: ReportField) -> str:
        """Return a ``label: value unit`` string for a field."""
        text = f"{field.label}: {field.value}"
        if field.unit is not None:
            text = f"{text} {field.unit}"
        return text

    def _as_table(self, table: ReportTable) -> Table:
        """Convert a `ReportTable` to a `rich.table.Table`."""
        rich_table = Table(title=table.title, title_justify="left")
        for column in table.columns:
            rich_table.add_column(column)
        for row in table.rows:
            rich_table.add_row(*(str(cell) for cell in row))
        return rich_table

    def __rich__(self) -> Tree:
        """Return a `rich.tree.Tree` describing this report."""
        tree = Tree(self.title if self.title is not None else self.type_name)
        for field in self.fields:
            tree.add(self._field_line(field))
        for table in self.tables:
            tree.add(self._as_table(table))
        for key, child in self.children.items():
            branch = tree.add(key)
            branch.add(child.__rich__())
        return tree

    def _repr_html_(self) -> str:
        """Return an HTML rendering produced by rich."""
        console = Console(record=True, width=100)
        console.print(self)
        return console.export_html(inline_styles=True)


@runtime_checkable
class Describable(Protocol):
    """An object that can produce a `Report` describing itself."""

    def _describe(self, **kwargs: Any) -> Report:
        """Return a `Report` describing this object.

        Parameters
        ----------
        **kwargs
            Optional rendering parameters (e.g. a ``bbox`` to compute derived
            sky-coordinate fields).
        """
        ...


class DescribableMixin:
    """Mixin that wires repr, str, rich, and HTML rendering to `_describe`."""

    def _describe(self, **kwargs: Any) -> Report:
        """Return a `Report` describing this object.

        Parameters
        ----------
        **kwargs
            Optional rendering parameters.
        """
        raise NotImplementedError()

    def describe(self, **kwargs: Any) -> Report:
        """Return a `Report` describing this object.

        Parameters
        ----------
        **kwargs
            Optional rendering parameters passed through to `_describe`.
        """
        return self._describe(**kwargs)

    def __repr__(self) -> str:
        return self._describe().to_repr()

    def __str__(self) -> str:
        return self._describe().to_str()

    def _repr_html_(self) -> str:
        return self._describe()._repr_html_()

    def __rich__(self) -> RenderableType:
        return self._describe().__rich__()

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
    "DescribeOptions",
    "FieldRole",
    "Report",
    "ReportField",
    "ReportTable",
    "ReportValueGroup",
)

import dataclasses
import enum
import io
from collections.abc import Collection
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

if TYPE_CHECKING:
    from rich.console import RenderableType


class FieldRole(enum.Enum):
    """Where a report field appears: in ``repr``, in the expanded report, or
    both.
    """

    ARG = "arg"
    """A constructor argument that also informs; reproduced in ``repr`` and
    shown in the expanded report.
    """

    REPR_ONLY = "repr_only"
    """A constructor argument that ``repr`` needs to round-trip but that would
    duplicate a child or a more readable field in the expanded report.
    """

    DERIVED = "derived"
    """An informational or computed value; shown in the expanded report but
    never reproduced in ``repr``.
    """

    @property
    def in_repr(self) -> bool:
        """Whether fields with this role feed ``repr`` and ``str`` (`bool`)."""
        return self is not FieldRole.DERIVED

    @property
    def in_display(self) -> bool:
        """Whether fields with this role appear in the expanded report
        (`bool`).
        """
        return self is not FieldRole.REPR_ONLY


@dataclasses.dataclass(frozen=True)
class DescribeOptions:
    """Rendering options threaded through a tree of `Report` objects.

    Options that apply to a single type only (such as the ``bbox`` understood
    by `~lsst.images.SkyProjection`) are explicit keyword arguments on that
    type's ``_describe`` instead of members here, so that this class stays
    meaningful at every level of a report tree.
    """

    brief: bool = False
    """Whether to build only what ``repr`` and ``str`` read.

    When `True`, implementations skip children, tables, and any derived value
    that is expensive to compute.  Which *fields* feed ``repr`` is governed by
    `FieldRole`, not by this flag.
    """

    detail: bool = False
    """Whether to include extras that are too expensive for a default report,
    such as per-plane set-pixel counts that must scan the mask.
    """

    exclude: frozenset[str] = frozenset()
    """Names of report elements a composite has already shown once at the top
    level, and which its children should therefore omit.
    """

    def for_child(self, *exclude: str) -> DescribeOptions:
        """Return the options a child report should be built with.

        Parameters
        ----------
        *exclude
            Names of report elements the child should omit because the parent
            displays them once at the top level.  Replaces, rather than adds
            to, any `exclude` set on ``self``.

        Returns
        -------
        options : `DescribeOptions`
            Options carrying this object's `brief` and `detail` settings.
        """
        return dataclasses.replace(self, exclude=frozenset(exclude))


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
    """Where this field appears (`FieldRole`)."""

    positional: bool = False
    """If `True`, ``repr`` emits the value positionally (no ``label=``)."""


@dataclasses.dataclass(frozen=True)
class ReportValueGroup:
    """Short labeled values packed several to a rendered line.

    Use this where a report has many small scalars that would each be
    uninformative on a line of their own.  The renderer joins them and lets
    them wrap to the available width, so long values (a list, say) belong in
    a `ReportField` instead, where they get a line to themselves.
    """

    values: list[tuple[str, Any]]
    """Ordered ``(label, value)`` pairs."""

    role: FieldRole = FieldRole.DERIVED
    """Value groups never feed ``repr``; always `FieldRole.DERIVED`."""


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

    value_groups: list[ReportValueGroup] = dataclasses.field(default_factory=list)
    """Ordered groups of short values packed several to a line."""

    children: dict[str, Report] = dataclasses.field(default_factory=dict)
    """Named nested sub-reports."""

    inline: bool = False
    """If `True`, render as a single ``key: summary`` line when embedded as a
    child of another report, instead of a nested branch.
    """

    def to_repr(self) -> str:
        """Return a ``repr`` string built from the fields whose role feeds
        ``repr``.

        Notes
        -----
        Reports with no such fields describe objects that cannot be rebuilt
        from a string, such as those wrapping an AST mapping.  Those get the
        angle-bracket form Python uses for objects whose ``repr`` is
        descriptive, rather than an empty call that would claim an
        eval-ability they do not have.
        """
        parts: list[str] = []
        for field in self.fields:
            if not field.role.in_repr:
                continue
            value = field.repr_value if field.repr_value is not None else repr(field.value)
            parts.append(value if field.positional else f"{field.label}={value}")
        if not parts:
            if self.summary is None:
                return f"<{self.type_name}>"
            # Some summaries already open with the type name, which reads
            # naturally on its own; do not state it twice.
            if self.summary.startswith(self.type_name):
                return f"<{self.summary}>"
            return f"<{self.type_name}: {self.summary}>"
        return f"{self.type_name}({', '.join(parts)})"

    def to_str(self) -> str:
        """Return a compact one-line summary."""
        if self.summary is not None:
            return self.summary
        args = [str(field.value) for field in self.fields if field.role.in_repr]
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
        rich_table = Table(
            title=Text(table.title) if table.title is not None else None,
            title_justify="left",
        )
        for column in table.columns:
            rich_table.add_column(Text(column))
        for row in table.rows:
            rich_table.add_row(*(Text(str(cell)) for cell in row))
        return rich_table

    def __rich__(self) -> Tree:
        """Return a `rich.tree.Tree` describing this report."""
        tree = Tree(Text(self.title if self.title is not None else self.type_name))
        for field in self.fields:
            if not field.role.in_display:
                continue
            tree.add(Text(self._field_line(field)))
        for group in self.value_groups:
            # A plain Text wraps to the available width, packing as many
            # values onto each line as will fit.
            tree.add(Text("  ".join(f"{label}={value}" for label, value in group.values)))
        for table in self.tables:
            tree.add(self._as_table(table))
        for key, child in self.children.items():
            if child.inline:
                tree.add(Text(f"{key}: {child.to_str()}"))
            else:
                branch = tree.add(Text(key))
                branch.add(child.__rich__())
        return tree

    def _repr_html_(self) -> str:
        """Return an HTML rendering produced by rich."""
        # force_jupyter=False keeps console.print writing to the recording
        # buffer; inside a notebook a Jupyter-aware console would instead
        # publish the render itself, doubling the displayed output.
        console = Console(record=True, width=100, file=io.StringIO(), force_jupyter=False)
        console.print(self)
        return console.export_html(inline_styles=True)


# runtime_checkable supports the isinstance check in the "describe" command
# line subcommand, which must decide whether an arbitrary deserialized object
# can be described at all.
@runtime_checkable
class Describable(Protocol):
    """An object that can produce a `Report` describing itself.

    `describe` is the entry point callers use; `_describe` is the recursive
    contract composites use to build child reports.  They differ in signature:
    `describe` spells its options out as keyword arguments for convenience,
    while `_describe` takes the single `DescribeOptions` value that composites
    thread down the tree.
    """

    def _describe(self, options: DescribeOptions = DescribeOptions(), /) -> Report:
        """Return a `Report` describing this object.

        Parameters
        ----------
        options : `DescribeOptions`, optional
            Rendering options.  Implementations ignore the members they have
            no use for, so new members can be added without breaking them.

        Returns
        -------
        report : `Report`
            Report describing this object.
        """
        ...

    def describe(self, *, brief: bool = False, detail: bool = False, exclude: Collection[str] = ()) -> Report:
        """Return a `Report` describing this object.

        Parameters
        ----------
        brief : `bool`, optional
            Whether to build only what ``repr`` and ``str`` read.
        detail : `bool`, optional
            Whether to include extras that are too expensive for a default
            report.
        exclude : `~collections.abc.Collection` [`str`], optional
            Names of report elements to omit.

        Returns
        -------
        report : `Report`
            Report describing this object.
        """
        ...


class DescribableMixin:
    """Mixin that wires repr, str, rich, and HTML rendering to `_describe`."""

    def _describe(self, options: DescribeOptions = DescribeOptions(), /) -> Report:
        """Return a `Report` describing this object.

        Parameters
        ----------
        options : `DescribeOptions`, optional
            Rendering options.

        Returns
        -------
        report : `Report`
            Report describing this object.
        """
        raise NotImplementedError()

    def describe(self, *, brief: bool = False, detail: bool = False, exclude: Collection[str] = ()) -> Report:
        """Return a `~lsst.images.Report` describing this object.

        Parameters
        ----------
        brief : `bool`, optional
            Whether to build only what ``repr`` and ``str`` read.
        detail : `bool`, optional
            Whether to include extras that are too expensive for a default
            report, such as per-plane set-pixel counts.
        exclude : `~collections.abc.Collection` [`str`], optional
            Names of report elements to omit.

        Returns
        -------
        report : `~lsst.images.Report`
            Report describing this object.
        """
        return self._describe(DescribeOptions(brief=brief, detail=detail, exclude=frozenset(exclude)))

    def __repr__(self) -> str:
        return self._describe(DescribeOptions(brief=True)).to_repr()

    def __str__(self) -> str:
        return self._describe(DescribeOptions(brief=True)).to_str()

    def _repr_html_(self) -> str:
        return self._describe(DescribeOptions(detail=True))._repr_html_()

    def __rich__(self) -> RenderableType:
        return self._describe(DescribeOptions(detail=True)).__rich__()

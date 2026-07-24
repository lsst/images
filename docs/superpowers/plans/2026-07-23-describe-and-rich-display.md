# Describe Reports and Rich Display Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every user-facing `lsst.images` class a structured `describe()` report that drives plain-text, rich-terminal, notebook-HTML, and `repr` output from one source, plus a CLI `describe` subcommand.

**Architecture:** A renderer-agnostic `Report` tree (fields, tables, nested children) is produced by each class's `_describe()` method. A `DescribableMixin` derives `__repr__`, `__str__`, `_repr_html_`, and `__rich__` from that report. `rich` renders both terminal and notebook output; `repr` is reconstructed from the report's constructor-argument fields.

**Tech Stack:** Python 3.12+, pydantic 2.12+, astropy 7+, `rich` (new hard dependency), click, pytest, numpy 2+, starlink-pyast.

## Global Constraints

- All changes must be ruff-clean and mypy-clean (repo is configured for both).
- Use top-level imports; function-scoped imports only to avoid circular imports or expensive optional deps.
- One sentence per line in prose docs; American English spelling.
- Docstrings/comments describe the code as it is today — no references to plans, history, or prior behavior.
- Numpydoc validation runs in pre-commit; public methods need conforming docstrings.
- End every commit message with a blank line, then `Generated with AI`, blank line, `Co-Authored-By: SLAC AI`.
- Work on branch `tickets/DM-55612` (already checked out). Never push; never act as the user on GitHub.
- Tests are module-level `pytest` functions (`def test_*() -> None:`), one file per subject under `tests/`, each starting with the standard LSST license header.
- Every new source file starts with the standard LSST license header (copy from any existing file in the same directory).

## File Structure

- Create `python/lsst/images/describe.py` — the `Report`, `ReportField`, `ReportTable`, `FieldRole`, `Describable` protocol, `DescribableMixin`, and the renderers. Exported from `lsst.images`.
- Create `python/lsst/images/cli/_describe.py` — the `describe` CLI subcommand.
- Create `tests/test_describe.py` — tests for the report model, renderers, and mixin.
- Modify `python/lsst/images/__init__.py` — export the describe public API.
- Modify `pyproject.toml` — add `rich` to core dependencies.
- Modify each class's source file to add `_describe()` and adopt the mixin, group by group.
- Modify `python/lsst/images/cli/_main.py` — register the `describe` command.

---

### Task 1: Add `rich` dependency and the report data model

**Files:**
- Modify: `pyproject.toml:24-36` (core `dependencies`)
- Create: `python/lsst/images/describe.py`
- Test: `tests/test_describe.py`

**Interfaces:**
- Consumes: nothing (foundation task).
- Produces:
  - `class FieldRole(enum.Enum)` with members `ARG` and `DERIVED`.
  - `@dataclasses.dataclass(frozen=True) class ReportField` with fields `label: str`, `value: Any`, `unit: str | None = None`, `repr_value: str | None = None`, `role: FieldRole = FieldRole.ARG`, `positional: bool = False`.
  - `@dataclasses.dataclass(frozen=True) class ReportTable` with fields `title: str | None`, `columns: list[str]`, `rows: list[list[Any]]`, `role: FieldRole = FieldRole.DERIVED`.
  - `@dataclasses.dataclass class Report` with fields `type_name: str`, `title: str | None = None`, `summary: str | None = None`, `fields: list[ReportField]` (default empty), `tables: list[ReportTable]` (default empty), `children: dict[str, Report]` (default empty).

- [ ] **Step 1: Install `rich` into the project environment**

Run: `.pyenv/bin/pip install 'rich>=13'`
Expected: rich installs successfully (already resolvable; version 15.x available).

- [ ] **Step 2: Add `rich` to core dependencies**

In `pyproject.toml`, add `"rich >= 13",` to the `dependencies` list (alphabetically after `"pydantic >= 2.12",`):

```toml
dependencies = [
    "astropy >= 7.0",
    "click >= 8",
    "fsspec",
    "numpy >= 2.0",
    "packaging",
    "pydantic >= 2.12",
    "rich >= 13",
    "scipy >= 1.13",
    "lsst-resources",
    "astro-metadata-translator >=30.2026.900",
    "starlink-pyast >=4.0.0",
    "shapely >= 2.1",
]
```

- [ ] **Step 3: Write the failing test for the data model**

Create `tests/test_describe.py` with the standard license header, then:

```python
from __future__ import annotations

from lsst.images.describe import FieldRole, Report, ReportField, ReportTable


def test_report_model_defaults() -> None:
    """Report and its components have the expected fields and defaults."""
    field = ReportField(label="bbox", value="[y=0:4, x=0:4]")
    assert field.unit is None
    assert field.repr_value is None
    assert field.role is FieldRole.ARG
    assert field.positional is False

    table = ReportTable(title="Axes", columns=["Axis", "Label"], rows=[[1, "RA"]])
    assert table.role is FieldRole.DERIVED

    report = Report(type_name="Image")
    assert report.title is None
    assert report.summary is None
    assert report.fields == []
    assert report.tables == []
    assert report.children == {}
```

- [ ] **Step 4: Run test to verify it fails**

Run: `.pyenv/bin/pytest tests/test_describe.py::test_report_model_defaults -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lsst.images.describe'`.

- [ ] **Step 5: Implement the data model**

Create `python/lsst/images/describe.py` with the standard license header, then:

```python
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
```

- [ ] **Step 6: Run test to verify it passes**

Run: `.pyenv/bin/pytest tests/test_describe.py::test_report_model_defaults -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml python/lsst/images/describe.py tests/test_describe.py
git commit -m "Add rich dependency and describe report data model

Generated with AI

Co-Authored-By: SLAC AI"
```

---

### Task 2: Text renderers `to_repr()` and `to_str()`

**Files:**
- Modify: `python/lsst/images/describe.py`
- Test: `tests/test_describe.py`

**Interfaces:**
- Consumes: `Report`, `ReportField`, `ReportTable`, `FieldRole` from Task 1.
- Produces:
  - `Report.to_repr(self) -> str` — `TypeName(label=repr_value, ...)` over `ARG` fields only; `repr_value` defaults to `repr(value)`. Tables and `DERIVED` fields are omitted.
  - `Report.to_str(self) -> str` — one line: `summary` if set, else `type_name` followed by `title` and the first up-to-three `ARG` field `value`s.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_describe.py`:

```python
from lsst.images.describe import FieldRole  # (already imported at top)


def test_to_repr_uses_arg_fields_only() -> None:
    """to_repr emits Type(label=repr_value) for ARG fields, skipping others."""
    report = Report(
        type_name="Image",
        fields=[
            ReportField(label="bbox", value="[y=0:4, x=0:4]", repr_value="Box(...)"),
            ReportField(label="array", value="<huge>", repr_value="...", role=FieldRole.ARG),
            ReportField(label="corner", value="10:04:21", role=FieldRole.DERIVED),
        ],
        tables=[ReportTable(title="T", columns=["a"], rows=[[1]])],
    )
    assert report.to_repr() == "Image(bbox=Box(...), array=...)"


def test_to_repr_defaults_repr_value_to_repr_of_value() -> None:
    """A field with no repr_value falls back to repr(value)."""
    report = Report(type_name="Interval", fields=[ReportField(label="start", value=3)])
    assert report.to_repr() == "Interval(start=3)"


def test_to_repr_supports_positional_fields() -> None:
    """Positional ARG fields emit their value without a label=."""
    report = Report(
        type_name="MaskSchema",
        fields=[
            ReportField(label="planes", value="[...]", repr_value="[...]", positional=True),
            ReportField(label="dtype", value="uint8", repr_value="dtype('uint8')"),
        ],
    )
    assert report.to_repr() == "MaskSchema([...], dtype=dtype('uint8'))"


def test_to_str_prefers_summary() -> None:
    """to_str returns the summary verbatim when present."""
    report = Report(type_name="Image", summary="Image([y=0:4, x=0:4], float32)")
    assert report.to_str() == "Image([y=0:4, x=0:4], float32)"


def test_to_str_without_summary_lists_fields() -> None:
    """to_str falls back to type name plus the first few ARG values."""
    report = Report(
        type_name="Interval",
        fields=[ReportField(label="start", value=0), ReportField(label="stop", value=4)],
    )
    assert report.to_str() == "Interval(0, 4)"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.pyenv/bin/pytest tests/test_describe.py -k "to_repr or to_str" -v`
Expected: FAIL with `AttributeError: 'Report' object has no attribute 'to_repr'`.

- [ ] **Step 3: Implement the text renderers**

In `python/lsst/images/describe.py`, add these methods to the `Report` class:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.pyenv/bin/pytest tests/test_describe.py -k "to_repr or to_str" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/lsst/images/describe.py tests/test_describe.py
git commit -m "Add text renderers for describe reports

Generated with AI

Co-Authored-By: SLAC AI"
```

---

### Task 3: Rich renderers and the `DescribableMixin`

**Files:**
- Modify: `python/lsst/images/describe.py`
- Test: `tests/test_describe.py`

**Interfaces:**
- Consumes: `Report` (with `to_repr`/`to_str`) from Tasks 1-2.
- Produces:
  - `Report.__rich__(self) -> rich.tree.Tree` — a tree whose root label is the `title` (or `type_name`), with one node per field (`label : value unit`), each `ReportTable` rendered as a `rich.table.Table`, and each child rendered as a nested subtree (branch label is the child key).
  - `Report._repr_html_(self) -> str` — rich HTML export of the `__rich__` renderable.
  - `class Describable(Protocol)` with `_describe(self, **kwargs: Any) -> Report`.
  - `class DescribableMixin` providing `describe`, `__repr__`, `__str__`, `_repr_html_`, `__rich__`, all delegating to `_describe()`. `_describe` itself raises `NotImplementedError`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_describe.py` (extend the top imports with `import re`):

```python
def test_rich_renders_fields_tables_and_children() -> None:
    """__rich__ produces text containing labels, table headers, and child keys."""
    from rich.console import Console

    report = Report(
        type_name="SkyProjection",
        title="ICRS coordinates",
        fields=[ReportField(label="Domain", value="SKY")],
        tables=[ReportTable(title="Axes", columns=["Axis", "Label"], rows=[[1, "RA"], [2, "Dec"]])],
        children={"pixel": Report(type_name="GeneralFrame", fields=[ReportField(label="unit", value="pix")])},
    )
    console = Console(record=True, width=100)
    console.print(report)
    text = console.export_text()
    assert "ICRS coordinates" in text
    assert "Domain" in text and "SKY" in text
    assert "Axis" in text and "Label" in text and "RA" in text
    assert "pixel" in text and "GeneralFrame" in text


def test_repr_html_produces_html() -> None:
    """_repr_html_ returns an HTML fragment mentioning the content."""
    report = Report(type_name="Interval", fields=[ReportField(label="start", value=3)])
    html = report._repr_html_()
    assert "<" in html and ">" in html
    assert "Interval" in html


def test_mixin_derives_dunders_from_describe() -> None:
    """DescribableMixin wires repr/str/html to _describe."""

    class Widget(DescribableMixin):
        def _describe(self, **kwargs: object) -> Report:
            return Report(
                type_name="Widget",
                summary="Widget(size=5)",
                fields=[ReportField(label="size", value=5)],
            )

    widget = Widget()
    assert repr(widget) == "Widget(size=5)"
    assert str(widget) == "Widget(size=5)"
    assert "Widget" in widget._repr_html_()
    assert isinstance(widget.describe(), Report)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.pyenv/bin/pytest tests/test_describe.py -k "rich or html or mixin" -v`
Expected: FAIL (no `__rich__`, `_repr_html_`, or `DescribableMixin`).

- [ ] **Step 3: Implement rich rendering and the mixin**

In `python/lsst/images/describe.py`, extend the top-level imports:

```python
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from rich.console import Console
from rich.table import Table
from rich.tree import Tree

if TYPE_CHECKING:
    from rich.console import RenderableType
```

Add these methods to `Report` (a private helper builds the tree so both renderers share it):

```python
    def _field_line(self, field: ReportField) -> str:
        """Return a ``label : value unit`` line for a field."""
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
```

Then add the protocol and mixin at module scope:

```python
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
    """Mixin that derives ``repr``/``str``/rich/HTML output from `_describe`."""

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
```

Note: add `from rich.console import RenderableType` under the `TYPE_CHECKING` guard (already added in the import step) and quote the annotation as `"RenderableType"` in `__rich__` since it is only imported for typing.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.pyenv/bin/pytest tests/test_describe.py -v`
Expected: PASS (all describe tests).

- [ ] **Step 5: Run ruff and mypy**

Run: `.pyenv/bin/ruff check python/lsst/images/describe.py && .pyenv/bin/mypy python/lsst/images/describe.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add python/lsst/images/describe.py tests/test_describe.py
git commit -m "Add rich renderers and DescribableMixin

Generated with AI

Co-Authored-By: SLAC AI"
```

---

### Task 4: Export the describe public API

**Files:**
- Modify: `python/lsst/images/__init__.py`
- Test: `tests/test_describe.py`

**Interfaces:**
- Consumes: the public names from `describe.py` (`Describable`, `DescribableMixin`, `FieldRole`, `Report`, `ReportField`, `ReportTable`).
- Produces: those names importable directly from `lsst.images`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_describe.py`:

```python
def test_public_api_importable_from_package() -> None:
    """The describe public API is re-exported from lsst.images."""
    import lsst.images as images

    for name in ("Describable", "DescribableMixin", "FieldRole", "Report", "ReportField", "ReportTable"):
        assert hasattr(images, name), name
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.pyenv/bin/pytest tests/test_describe.py::test_public_api_importable_from_package -v`
Expected: FAIL (names not on `lsst.images`).

- [ ] **Step 3: Add the import**

In `python/lsst/images/__init__.py`, add a line in the existing block of `from ._module import *` imports, placed alphabetically (after `from ._difference_image import *`):

```python
from .describe import *
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.pyenv/bin/pytest tests/test_describe.py::test_public_api_importable_from_package -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/lsst/images/__init__.py tests/test_describe.py
git commit -m "Export describe public API from lsst.images

Generated with AI

Co-Authored-By: SLAC AI"
```

---

### Task 5: `MaskSchema._describe()` (tabular)

**Files:**
- Modify: `python/lsst/images/_mask.py` (`MaskSchema`, lines ~152-232)
- Test: `tests/test_mask.py`

**Interfaces:**
- Consumes: `Report`, `ReportField`, `ReportTable`, `DescribableMixin` from Tasks 1-4.
- Produces: `MaskSchema._describe(self, **kwargs) -> Report` with one `ReportTable` titled `"Mask planes"` (columns `["Bit", "Index", "Mask", "Name", "Description"]`), a positional `ARG` field for `planes` (`repr_value=repr(list(self._planes))`), and a `dtype` `ARG` field (`repr_value=repr(self._dtype)`). `MaskSchema` gains `DescribableMixin` as a base; its hand-written `__str__`/`__repr__` are removed.

- [ ] **Step 1: Pin current repr/str behavior (characterization test)**

The existing `tests/test_mask.py::test_schema` already asserts `eval(repr(schema), ...) == schema` and that `str(schema)` has one line per plane containing `M5 [index@mask]: D5`. Run it now to confirm the baseline passes:

Run: `.pyenv/bin/pytest tests/test_mask.py::test_schema -v`
Expected: PASS (baseline before changes).

- [ ] **Step 2: Write the failing test for `_describe`**

Add to `tests/test_mask.py` (imports at top already include `numpy as np`, `MaskSchema`, `MaskPlane`; add `from lsst.images.describe import Report, ReportTable` and reuse `make_mask_planes`):

```python
def test_schema_describe() -> None:
    """MaskSchema._describe yields a mask-plane table and reconstructable repr."""
    rng = np.random.default_rng(500)
    planes = make_mask_planes(rng, 3, 0)
    schema = MaskSchema(planes, dtype=np.uint8)

    report = schema.describe()
    assert isinstance(report, Report)
    assert report.type_name == "MaskSchema"

    tables = [t for t in report.tables if t.title == "Mask planes"]
    assert len(tables) == 1
    table = tables[0]
    assert table.columns == ["Bit", "Index", "Mask", "Name", "Description"]
    assert len(table.rows) == 3
    # Names appear in the Name column (index 3).
    assert {row[3] for row in table.rows} == {"M0", "M1", "M2"}

    # repr is still eval-able and round-trips.
    reconstructed = eval(
        repr(schema), {"dtype": np.dtype, "MaskSchema": MaskSchema, "MaskPlane": MaskPlane}
    )
    assert reconstructed == schema
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.pyenv/bin/pytest tests/test_mask.py::test_schema_describe -v`
Expected: FAIL with `AttributeError: 'MaskSchema' object has no attribute 'describe'`.

- [ ] **Step 4: Adopt the mixin and implement `_describe`**

In `python/lsst/images/_mask.py`, add the import near the other package imports:

```python
from .describe import DescribableMixin, Report, ReportField, ReportTable
```

Change the class declaration from `class MaskSchema:` to `class MaskSchema(DescribableMixin):`.

Delete the existing `__repr__` (lines ~223-224) and `__str__` (lines ~226-232) methods and replace them with:

```python
    def _describe(self, **kwargs: Any) -> Report:
        """Return a `Report` describing this mask schema.

        Parameters
        ----------
        **kwargs
            Unused; accepted for interface compatibility.
        """
        rows = [
            [
                n,
                self._bits[plane.name].index,
                hex(self._bits[plane.name].mask),
                plane.name,
                plane.description,
            ]
            for n, plane in enumerate(self._planes)
            if plane is not None
        ]
        return Report(
            type_name="MaskSchema",
            fields=[
                ReportField(
                    label="planes",
                    value=f"<{len(self._planes)} planes>",
                    repr_value=repr(list(self._planes)),
                    positional=True,
                ),
                ReportField(label="dtype", value=str(self._dtype), repr_value=repr(self._dtype)),
            ],
            tables=[
                ReportTable(
                    title="Mask planes",
                    columns=["Bit", "Index", "Mask", "Name", "Description"],
                    rows=rows,
                )
            ],
        )
```

Confirm `Any` is imported in `_mask.py` (it is used widely; if not, add it to the `typing` import).

- [ ] **Step 5: Run the new and baseline tests**

Run: `.pyenv/bin/pytest tests/test_mask.py::test_schema_describe tests/test_mask.py::test_schema -v`
Expected: `test_schema_describe` PASS. `test_schema` may FAIL on the `str(schema)` line-count assertion, because `__str__` is now the compact one-liner. Proceed to Step 6.

- [ ] **Step 6: Update the baseline `str` assertions in `test_schema`**

In `tests/test_mask.py::test_schema`, the block that asserts the old multi-line `str` output (the `string = str(schema)` line, the `len(string.split("\n")) == 17` assertion, and the `f"M5 [...]: D5" in string` assertion) no longer matches the new compact `__str__`. Replace those three lines with an assertion against the report table instead:

```python
    report = schema.describe()
    plane_table = next(t for t in report.tables if t.title == "Mask planes")
    assert len(plane_table.rows) == 17
    assert ["M5" == row[3] for row in plane_table.rows].count(True) == 1
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `.pyenv/bin/pytest tests/test_mask.py -v`
Expected: PASS.

- [ ] **Step 8: Run ruff and mypy**

Run: `.pyenv/bin/ruff check python/lsst/images/_mask.py && .pyenv/bin/mypy python/lsst/images/_mask.py`
Expected: no errors.

- [ ] **Step 9: Commit**

```bash
git add python/lsst/images/_mask.py tests/test_mask.py
git commit -m "Add tabular describe report to MaskSchema

Generated with AI

Co-Authored-By: SLAC AI"
```

---

### Task 6: `SkyProjection._describe(bbox=None)` (tabular, KAPPA-style)

**Files:**
- Modify: `python/lsst/images/_transforms/_sky_projection.py` (`SkyProjection`, lines ~55-97)
- Test: `tests/test_transforms.py`

**Interfaces:**
- Consumes: `Report`, `ReportField`, `ReportTable`, `FieldRole`, `DescribableMixin` from Tasks 1-4; `Box` from `.._geom`; `make_random_sky_projection` from `lsst.images.tests`.
- Produces:
  - `SkyProjection._pixel_axis_report(self, bbox: Box) -> list[tuple[float, str, str, bool]]`: per **pixel** axis (x then y), a `(scale_arcsec, label, units, diagonal)` tuple where `label`/`units` name the sky direction that pixel axis predominantly tracks (`"Right ascension"`/`"hh:mm:ss.s"` or `"Declination"`/`"dd:mm:ss"`) and `diagonal` flags an axis running near 45° to both sky directions (label ambiguous). Computed with the Starlink KAPPA *technique* (great-circle `AST_DISTANCE` analogue via `SkyCoord.separation`, median over a 3×3 grid of test points), so it is correct near the poles and under coordinate rotation. **Reporting per pixel axis is what makes a ~90° rotation come out right:** the scale stays attached to its pixel axis while the label follows the sky direction, so a 90° rotation reads Axis `x` → "Declination", Axis `y` → "Right ascension". Private for now; a candidate for promotion later.
  - `SkyProjection._describe(self, *, bbox: Box | None = None, **kwargs) -> Report`. `SkyProjection` gains `DescribableMixin` as a base. The report has:
    - `title="ICRS coordinates"`, `summary` naming the pixel and sky frames.
    - `ARG` field `pixel_to_sky` (`repr_value="..."`, lossy) and `DERIVED` fields `domain` (`"ICRS"`), `center` (sky coord of the bbox center, only when a bbox is available), `fits_wcs` (`"available"` / `"approximate"` / `"none"`).
    - A `DERIVED` `ReportTable` titled `"Axes"` (columns `["Axis", "Label", "Units", "Nominal pixel scale"]`), **one row per pixel axis** (`"x"`, `"y"`) with the label/units/scale from `_pixel_axis_report`; a near-diagonal axis appends `" (diagonal)"` to its label. Scales are populated only when a bbox is available (otherwise the label/units default to the unrotated `x=RA, y=Dec` convention and the scale is `"-"`).
    - A `DERIVED` `ReportTable` titled `"Corners"` (columns `["Corner", "RA", "Dec"]`), only when a bbox is available.

#### The pixel-scale algorithm (Starlink KAPPA technique, per pixel axis)

The Fortran original is `KPG1_DSFRM` → `KPG1_SCALE` → `KPG1_PXSCL` in Starlink KAPLIBS. It attributes motion to *sky* axes by searching all unit-offset neighbours for the one that moves farthest along each sky axis. Here the "Axes" table reports per **pixel** axis, so that farthest-neighbour search is unnecessary — we perturb directly along each pixel axis. What we keep from KAPPA is the robust *technique*:

- **Great-circle distance** (`SkyCoord.separation`, the astropy analogue of `AST_DISTANCE` on a sky frame) for the sky step — pole- and rotation-safe, unlike a naive dRA/dpix.
- **Median over a 3×3 test grid** (bbox center ± `0.3 × axis extent` on each pixel axis) so one degenerate sample cannot skew the result — this is the `KPG1_SCALE` wrapper.

Per pixel axis `a` (unit step `(1,0)` for x, `(0,1)` for y) at each test point:
- `scale = center.separation(step_a).to_value(arcsec)` — the great-circle scale, kept with pixel axis `a`.
- Direction: compare the RA component `|Δra·cos(dec)|` against the Dec component `|Δdec|`; the larger names the axis. When the two are comparable (`min/max > 0.8`), flag the axis `diagonal` (near 45°, label ambiguous).
Take the **median** scale per pixel axis and the median-based direction.

Do not simplify to a single adjacent-pixel `separation` with fixed `x=RA, y=Dec` labels; that mislabels rotated WCS and is the heuristic this task deliberately replaces.

- [ ] **Step 1: Write the failing test for the pixel-axis report**

Add to `tests/test_transforms.py` (add `import astropy.wcs` and `from lsst.images._transforms._sky_projection import SkyProjection` to the top-of-file imports if not already present):

```python
def _rotated_tan(rot_deg: float, *, crval2: float = 30.0, scale_y: float = 0.2) -> SkyProjection:
    """A TAN SkyProjection: 0.2 arcsec/pixel on x, ``scale_y`` on y, rotated."""
    cx = (0.2 * u.arcsec).to_value(u.deg)
    cy = (scale_y * u.arcsec).to_value(u.deg)
    t = np.deg2rad(rot_deg)
    header = {
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 50,
        "CRPIX2": 100,
        "CRVAL1": 45.0,
        "CRVAL2": crval2,
        "CD1_1": -cx * np.cos(t),
        "CD1_2": cy * np.sin(t),
        "CD2_1": -cx * np.sin(t),
        "CD2_2": -cy * np.cos(t),
    }
    return SkyProjection.from_fits_wcs(astropy.wcs.WCS(header), GeneralFrame(unit=u.pix))


def test_sky_projection_pixel_axis_report() -> None:
    """_pixel_axis_report keeps scale with the pixel axis, label with the sky.

    Uses great-circle distances (pole- and rotation-safe) and reports per
    pixel axis so a ~90 deg rotation swaps the RA/Dec labels while the scale
    stays attached to its pixel axis.
    """
    bbox = Box.factory[0:200, 0:100]

    # Unrotated, anisotropic: x tracks RA at 0.2, y tracks Dec at 0.3.
    report = _rotated_tan(0.0, scale_y=0.3)._pixel_axis_report(bbox)
    assert len(report) == 2
    (sx, lx, ux, dx), (sy, ly, uy, dy) = report
    np.testing.assert_allclose([sx, sy], [0.2, 0.3], rtol=1e-3)
    assert (lx, ly) == ("Right ascension", "Declination")
    assert (ux, uy) == ("hh:mm:ss.s", "dd:mm:ss")
    assert not dx and not dy

    # Rotated 90 deg: the labels swap but the scale stays with the pixel axis.
    (sx, lx, _, _), (sy, ly, _, _) = _rotated_tan(90.0, scale_y=0.3)._pixel_axis_report(bbox)
    np.testing.assert_allclose([sx, sy], [0.2, 0.3], rtol=1e-3)
    assert (lx, ly) == ("Declination", "Right ascension")

    # Rotated 45 deg: both axes run diagonally, so both are flagged ambiguous.
    (_, _, _, dx), (_, _, _, dy) = _rotated_tan(45.0)._pixel_axis_report(bbox)
    assert dx and dy

    # Reference pixel ~2 arcsec from the north pole: great-circle scale holds.
    (sx, _, _, _), (sy, _, _, _) = _rotated_tan(30.0, crval2=89.9995)._pixel_axis_report(bbox)
    np.testing.assert_allclose([sx, sy], [0.2, 0.2], rtol=1e-3)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.pyenv/bin/pytest tests/test_transforms.py::test_sky_projection_pixel_axis_report -v`
Expected: FAIL with `AttributeError: 'SkyProjection' object has no attribute '_pixel_axis_report'`.

- [ ] **Step 3: Implement the pixel-axis report method**

In `python/lsst/images/_transforms/_sky_projection.py`, add these imports near the other package/stdlib imports (skip any already present — `numpy as np`, `astropy.units as u`, and `SkyCoord` are very likely already imported; `itertools` and `statistics` are stdlib):

```python
import itertools
import statistics

from .._geom import Box
```

Add this method to the class (for example, just after `as_fits_wcs`):

```python
    def _pixel_axis_report(self, bbox: Box) -> list[tuple[float, str, str, bool]]:
        """Return per-pixel-axis scale and dominant sky direction.

        Parameters
        ----------
        bbox : `Box`
            Pixel bounding box over which the axes are characterized.

        Returns
        -------
        `list` [`tuple`]
            One ``(scale_arcsec, label, units, diagonal)`` entry per pixel
            axis (``x`` then ``y``).  ``scale_arcsec`` is the nominal pixel
            scale in arcsec/pixel along that pixel axis, ``label``/``units``
            name the sky direction the axis predominantly tracks
            (``"Right ascension"``/``"hh:mm:ss.s"`` or
            ``"Declination"``/``"dd:mm:ss"``), and ``diagonal`` is `True` when
            the axis runs near 45 deg to both sky directions (label
            ambiguous).

        Notes
        -----
        This adapts the Starlink KAPPA ``KPG1_SCALE``/``KPG1_PXSCL`` technique
        to per-pixel-axis reporting.  The scale is the great-circle sky
        distance for a unit step along the pixel axis (the astropy analogue of
        AST's ``AST_DISTANCE``), taken as the median over a 3x3 grid of test
        points (bbox center plus/minus 0.3 times the axis extent).  Great-circle
        distances keep the result correct near the poles and under coordinate
        rotation; reporting per pixel axis keeps each scale attached to its
        pixel axis while the label follows the sky direction, so a ~90 deg
        rotation swaps the RA/Dec labels correctly.
        """
        step_x = 0.3 * bbox.x.size
        step_y = 0.3 * bbox.y.size
        unit_steps = ((1.0, 0.0), (0.0, 1.0))
        scales: tuple[list[float], list[float]] = ([], [])
        ra_components: tuple[list[float], list[float]] = ([], [])
        dec_components: tuple[list[float], list[float]] = ([], [])
        for dx, dy in itertools.product((-step_x, 0.0, step_x), (-step_y, 0.0, step_y)):
            cx = bbox.x.center + dx
            cy = bbox.y.center + dy
            center = self.pixel_to_sky(x=cx, y=cy)
            for axis, (ox, oy) in enumerate(unit_steps):
                step = self.pixel_to_sky(x=cx + ox, y=cy + oy)
                scales[axis].append(center.separation(step).to_value(u.arcsec))
                dra = (step.ra - center.ra).wrap_at(180 * u.deg).rad * np.cos(center.dec.rad)
                ddec = (step.dec - center.dec).rad
                ra_components[axis].append(abs(dra))
                dec_components[axis].append(abs(ddec))
        report: list[tuple[float, str, str, bool]] = []
        for axis in (0, 1):
            scale = statistics.median(scales[axis])
            dra = statistics.median(ra_components[axis])
            ddec = statistics.median(dec_components[axis])
            hi = max(dra, ddec)
            diagonal = hi > 0.0 and min(dra, ddec) / hi > 0.8
            if dra > ddec:
                label, units = "Right ascension", "hh:mm:ss.s"
            else:
                label, units = "Declination", "dd:mm:ss"
            report.append((scale, label, units, diagonal))
        return report
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.pyenv/bin/pytest tests/test_transforms.py::test_sky_projection_pixel_axis_report -v`
Expected: PASS (labels swap under 90 deg rotation, both axes flagged at 45 deg, scale holds near the pole).

- [ ] **Step 5: Write the failing test for `_describe`**

Add to `tests/test_transforms.py` (top-of-file imports already include `numpy as np`, `astropy.units as u`, `Box`, `GeneralFrame`, and `make_random_sky_projection`; add `from lsst.images.describe import FieldRole, Report`):

```python
def test_sky_projection_describe() -> None:
    """SkyProjection._describe yields KAPPA-style axes and corners tables."""
    rng = np.random.default_rng(43)
    bbox = Box.factory[0:200, 0:100]
    pixel_frame = GeneralFrame(unit=u.pix)
    sky_projection = make_random_sky_projection(rng, pixel_frame, bbox)

    # Without a bbox: Axes table present, Corners absent, no center field.
    # Rows are per pixel axis; without a bbox the labels default to the
    # unrotated x=RA, y=Dec convention and the scales are "-".
    report = sky_projection.describe()
    assert isinstance(report, Report)
    assert report.type_name == "SkyProjection"
    assert report.title == "ICRS coordinates"
    axes = next(t for t in report.tables if t.title == "Axes")
    assert axes.columns == ["Axis", "Label", "Units", "Nominal pixel scale"]
    assert len(axes.rows) == 2
    assert [row[0] for row in axes.rows] == ["x", "y"]
    assert [row[1] for row in axes.rows] == ["Right ascension", "Declination"]
    assert all(row[3] == "-" for row in axes.rows)  # no scale without a bbox
    assert not any(t.title == "Corners" for t in report.tables)
    assert not any(f.label == "center" for f in report.fields)

    # With a bbox: Corners table plus per-pixel-axis scales and a center field.
    # This projection has a random rotation, so the labels are whichever sky
    # direction each pixel axis predominantly tracks; assert they are valid.
    report = sky_projection.describe(bbox=bbox)
    axes = next(t for t in report.tables if t.title == "Axes")
    assert [row[0] for row in axes.rows] == ["x", "y"]
    assert all(row[3] != "-" for row in axes.rows)
    valid = {"Right ascension", "Declination"}
    assert all(row[1].removesuffix(" (diagonal)") in valid for row in axes.rows)
    corners = next(t for t in report.tables if t.title == "Corners")
    assert corners.columns == ["Corner", "RA", "Dec"]
    assert len(corners.rows) == 4
    center = next(f for f in report.fields if f.label == "center")
    assert center.role is FieldRole.DERIVED

    # FITS-WCS availability is reported (this projection is FITS-representable).
    fits_field = next(f for f in report.fields if f.label == "fits_wcs")
    assert fits_field.value == "available"

    # repr does not depend on a bbox and does not evaluate the mapping.
    assert repr(sky_projection).startswith("SkyProjection(")
```

- [ ] **Step 6: Run test to verify it fails**

Run: `.pyenv/bin/pytest tests/test_transforms.py::test_sky_projection_describe -v`
Expected: FAIL with `AttributeError: 'SkyProjection' object has no attribute 'describe'`.

- [ ] **Step 7: Adopt the mixin and implement `_describe`**

In `python/lsst/images/_transforms/_sky_projection.py`, add imports near the other package imports:

```python
from ..describe import DescribableMixin, FieldRole, Report, ReportField, ReportTable
```

(`Box`, `SkyCoord`, and `astropy.units as u` were added or confirmed present in Step 3; do not duplicate them.)

Change the class declaration from `class SkyProjection[F: Frame]:` to `class SkyProjection[F: Frame](DescribableMixin):`.

Add this method to the class (for example, just after `fits_approximation`):

```python
    def _describe(self, *, bbox: Box | None = None, **kwargs: Any) -> Report:
        """Return a `Report` describing this sky projection.

        Parameters
        ----------
        bbox : `Box`, optional
            Pixel bounding box.  When provided, the report gains the sky
            coordinates of the box center and corners and the nominal pixel
            scale along each axis.
        **kwargs
            Unused; accepted for interface compatibility.
        """
        fields = [
            ReportField(label="pixel_to_sky", value="<transform>", repr_value="...", positional=True),
            ReportField(label="domain", value=self.sky_frame.value, role=FieldRole.DERIVED),
        ]
        # Default (no bbox): unrotated x=RA, y=Dec convention with no scale.
        axis_rows: list[list[Any]] = [
            ["x", "Right ascension", "hh:mm:ss.s", "-"],
            ["y", "Declination", "dd:mm:ss", "-"],
        ]
        corners_table: list[ReportTable] = []
        if bbox is not None:
            center = self.pixel_to_sky(x=bbox.x.center, y=bbox.y.center)
            fields.append(
                ReportField(
                    label="center",
                    value=center.to_string("hmsdms"),
                    role=FieldRole.DERIVED,
                )
            )
            # One row per pixel axis; label follows the sky direction the axis
            # tracks (so a rotation swaps RA/Dec), scale stays with the axis.
            axis_rows = []
            for name, (scale, label, units, diagonal) in zip(
                ("x", "y"), self._pixel_axis_report(bbox), strict=True
            ):
                if diagonal:
                    label = f"{label} (diagonal)"
                axis_rows.append([name, label, units, f"{scale:.6g}"])
            mn, mx = bbox.min, bbox.max
            corner_defs = [
                ("(min x, min y)", mn.x, mn.y),
                ("(max x, min y)", mx.x, mn.y),
                ("(max x, max y)", mx.x, mx.y),
                ("(min x, max y)", mn.x, mx.y),
            ]
            rows = []
            for label, x, y in corner_defs:
                sky = self.pixel_to_sky(x=x, y=y)
                rows.append(
                    [
                        label,
                        sky.ra.to_string(unit=u.hour, sep=":", pad=True),
                        sky.dec.to_string(sep=":", pad=True, alwayssign=True),
                    ]
                )
            corners_table.append(ReportTable(title="Corners", columns=["Corner", "RA", "Dec"], rows=rows))

        if self._fits_approximation is not None:
            fits_wcs = "approximate"
        elif bbox is not None:
            fits_wcs = "available" if self.as_fits_wcs(bbox) is not None else "none"
        else:
            fits_wcs = "available"
        fields.append(ReportField(label="fits_wcs", value=fits_wcs, role=FieldRole.DERIVED))

        axes = ReportTable(
            title="Axes",
            columns=["Axis", "Label", "Units", "Nominal pixel scale"],
            rows=axis_rows,
        )
        return Report(
            type_name="SkyProjection",
            title="ICRS coordinates",
            summary=f"{type(self.pixel_frame).__name__} → {self.sky_frame.value}",
            fields=fields,
            tables=[axes, *corners_table],
        )
```

Note: `as_fits_wcs` requires a bbox, so FITS-WCS availability is only tested against the real mapping when a bbox is supplied; with no bbox it is reported as `"available"` optimistically unless a FITS approximation is attached (in which case `"approximate"`).

- [ ] **Step 8: Run tests to verify they pass**

Run: `.pyenv/bin/pytest tests/test_transforms.py::test_sky_projection_describe tests/test_transforms.py::test_sky_projection_pixel_axis_report -v`
Expected: PASS.

- [ ] **Step 9: Confirm no existing SkyProjection tests regressed**

Run: `.pyenv/bin/pytest tests/test_transforms.py -v`
Expected: PASS (SkyProjection previously had no `__repr__`/`__str__`, so nothing to pin; the mixin only adds behavior).

- [ ] **Step 10: Run ruff and mypy**

Run: `.pyenv/bin/ruff check python/lsst/images/_transforms/_sky_projection.py && .pyenv/bin/mypy python/lsst/images/_transforms/_sky_projection.py`
Expected: no errors.

- [ ] **Step 11: Commit**

```bash
git add python/lsst/images/_transforms/_sky_projection.py tests/test_transforms.py
git commit -m "Add KAPPA-style describe report and pixel-axis report to SkyProjection

Generated with AI

Co-Authored-By: SLAC AI"
```

---

### Task 7: Geometry primitives (`Box`, `Interval`, `Region`, `Polygon`)

These classes already have `__str__`/`__repr__`.
Migrate them to `_describe()` and pin the current output first, so the mixin's derived `repr`/`str` reproduce today's behavior exactly.

`XY` and `YX` are `NamedTuple`s with adequate auto-generated `repr`s (`XY(x=1, y=2)`); they are intentionally left unchanged (the spec lists them, but a `NamedTuple` cannot cleanly adopt the mixin and gains nothing here).

**Files:**
- Modify: `python/lsst/images/_geom.py` (`Interval` ~336-534, `Box` ~773-1085)
- Modify: `python/lsst/images/_polygon.py` (`Region` ~38-95, `Polygon` ~295-361)
- Test: `tests/test_geom.py`, `tests/test_polygon.py` (create the latter if absent; otherwise add to the existing polygon test file — confirm with `ls tests/ | grep -i polygon`)

**Interfaces:**
- Consumes: `Report`, `ReportField`, `DescribableMixin` from Tasks 1-4.
- Produces: `_describe()` on each of `Interval`, `Box`, `Region`, `Polygon`, each adopting `DescribableMixin`, with these exact `repr`/`str` results preserved:
  - `Interval`: `str` = `"{start}:{stop}"`; `repr` = `"Interval(start={start}, stop={stop})"`.
  - `Box`: `str` = `"[y={self.y}, x={self.x}]"`; `repr` = `"Box(y={self.y!r}, x={self.x!r})"`.
  - `Region`: `str` = `self._impl.wkt`; `repr` = `"Region.from_wkt({wkt!r})"`.
  - `Polygon`: `repr` = `"Polygon(x_vertices={x!r}, y_vertices={y!r})"` (inherits `Region.__str__`).

- [ ] **Step 1: Pin current geometry str/repr (characterization test)**

Add to `tests/test_geom.py`:

```python
def test_interval_box_repr_str_pinned() -> None:
    """Interval and Box str/repr match their documented forms."""
    interval = Interval(start=3, stop=10)
    assert str(interval) == "3:10"
    assert repr(interval) == "Interval(start=3, stop=10)"

    box = Box.factory[0:5, 0:4]
    assert str(box) == f"[y={box.y}, x={box.x}]"
    assert repr(box) == f"Box(y={box.y!r}, x={box.x!r})"
    # repr round-trips through eval.
    assert eval(repr(box), {"Box": Box, "Interval": Interval}) == box
```

Run: `.pyenv/bin/pytest tests/test_geom.py::test_interval_box_repr_str_pinned -v`
Expected: PASS (baseline; these are the current behaviors).

- [ ] **Step 2: Migrate `Interval` to `_describe`**

In `python/lsst/images/_geom.py`, add the import:

```python
from .describe import DescribableMixin, Report, ReportField
```

Change `class Interval:` to `class Interval(DescribableMixin):`.
Delete its `__str__` and `__repr__` (lines ~530-534) and add:

```python
    def _describe(self, **kwargs: Any) -> Report:
        """Return a `Report` describing this interval.

        Parameters
        ----------
        **kwargs
            Unused; accepted for interface compatibility.
        """
        return Report(
            type_name="Interval",
            summary=f"{self.start}:{self.stop}",
            fields=[
                ReportField(label="start", value=self.start),
                ReportField(label="stop", value=self.stop),
            ],
        )
```

Because `to_str()` prefers `summary` when present (Task 2), `str(interval)` becomes `"3:10"` as required, and `to_repr()` emits `Interval(start=3, stop=10)`.

Confirm `Any` is imported at the top of `_geom.py`.

- [ ] **Step 3: Migrate `Box` to `_describe`**

Change `class Box:` to `class Box(DescribableMixin):`.
Delete its `__str__`/`__repr__` (lines ~1081-1085) and add:

```python
    def _describe(self, **kwargs: Any) -> Report:
        """Return a `Report` describing this box.

        Parameters
        ----------
        **kwargs
            Unused; accepted for interface compatibility.
        """
        return Report(
            type_name="Box",
            summary=f"[y={self.y}, x={self.x}]",
            fields=[
                ReportField(label="y", value=self.y, repr_value=repr(self.y)),
                ReportField(label="x", value=self.x, repr_value=repr(self.x)),
            ],
        )
```

`to_repr()` emits `Box(y=Interval(start=0, stop=5), x=Interval(start=0, stop=4))`, matching `Box(y={self.y!r}, x={self.x!r})`; `str(box)` becomes `[y=0:5, x=0:4]` via `summary`.

- [ ] **Step 4: Run geometry tests**

Run: `.pyenv/bin/pytest tests/test_geom.py -v`
Expected: PASS (including the pinned test).

- [ ] **Step 5: Pin and migrate `Region` and `Polygon`**

Confirm the polygon test file name: `ls tests/ | grep -i polygon`.
Add a pinning test to that file (or `tests/test_polygon.py`):

```python
def test_region_polygon_repr_str_pinned() -> None:
    """Region and Polygon str/repr match their documented forms."""
    poly = Polygon(x_vertices=[0.0, 4.0, 4.0, 0.0], y_vertices=[0.0, 0.0, 5.0, 5.0])
    assert repr(poly) == f"Polygon(x_vertices={poly.x_vertices!r}, y_vertices={poly.y_vertices!r})"
    region = Region.from_wkt(poly.wkt)
    assert str(region) == region.wkt
    assert repr(region) == f"Region.from_wkt({region.wkt!r})"
```

Run it first to confirm the baseline passes, then migrate.

In `python/lsst/images/_polygon.py`, add the import `from .describe import DescribableMixin, Report, ReportField`, change `class Region:` to `class Region(DescribableMixin):`, delete `Region.__str__`/`__repr__` and `Polygon.__repr__`, and add:

```python
    # In Region:
    def _describe(self, **kwargs: Any) -> Report:
        """Return a `Report` describing this region.

        Parameters
        ----------
        **kwargs
            Unused; accepted for interface compatibility.
        """
        return Report(
            type_name="Region",
            summary=self._impl.wkt,
            fields=[
                ReportField(
                    label="wkt",
                    value=self._impl.wkt,
                    repr_value=repr(self._impl.wkt),
                    positional=True,
                )
            ],
        )
```

Because `Region.__repr__` must be `Region.from_wkt(...)` rather than `Region(...)`, do NOT rely on the mixin's default `to_repr` for `Region`; instead override `__repr__` on `Region` explicitly to keep the factory form:

```python
    def __repr__(self) -> str:
        return f"Region.from_wkt({self._impl.wkt!r})"
```

For `Polygon`, override `_describe` to carry its own `ARG` fields and keep its factory-free `Polygon(...)` repr via the mixin:

```python
    # In Polygon:
    def _describe(self, **kwargs: Any) -> Report:
        """Return a `Report` describing this polygon.

        Parameters
        ----------
        **kwargs
            Unused; accepted for interface compatibility.
        """
        return Report(
            type_name="Polygon",
            summary=self._impl.wkt,
            fields=[
                ReportField(label="x_vertices", value=self.x_vertices, repr_value=repr(self.x_vertices)),
                ReportField(label="y_vertices", value=self.y_vertices, repr_value=repr(self.y_vertices)),
            ],
        )
```

Note: `Polygon` inherits `Region.__repr__` (the factory form) unless overridden.
Add an explicit `Polygon.__repr__` restoring the current behavior:

```python
    def __repr__(self) -> str:
        return f"Polygon(x_vertices={self.x_vertices!r}, y_vertices={self.y_vertices!r})"
```

(The mixin still supplies `__str__`, `_repr_html_`, and `__rich__` for both.)

- [ ] **Step 6: Run tests, ruff, mypy**

Run: `.pyenv/bin/pytest tests/test_geom.py tests/test_polygon.py -v && .pyenv/bin/ruff check python/lsst/images/_geom.py python/lsst/images/_polygon.py && .pyenv/bin/mypy python/lsst/images/_geom.py python/lsst/images/_polygon.py`
Expected: PASS, no lint/type errors.

- [ ] **Step 7: Commit**

```bash
git add python/lsst/images/_geom.py python/lsst/images/_polygon.py tests/test_geom.py tests/test_polygon.py
git commit -m "Migrate geometry primitives to describe reports

Generated with AI

Co-Authored-By: SLAC AI"
```

---

### Task 8: Container images — `Image`, `Mask`, `ColorImage`

These three leaf containers have hand-written `str`/`repr`.
Pin their output, then migrate to `_describe()`.
`Image` and `ColorImage` use a lossy `repr` (`ClassName(..., bbox=..., dtype=...)`), reproduced via a positional field whose `repr_value="..."`.
`Mask` nests its `MaskSchema` report (from Task 5) as a child.

**Files:**
- Modify: `python/lsst/images/_image.py` (`Image`, `__str__`/`__repr__` ~213-217)
- Modify: `python/lsst/images/_mask.py` (`Mask`, `__str__`/`__repr__` ~551-555)
- Modify: `python/lsst/images/_color_image.py` (`ColorImage`, `__str__`/`__repr__` ~156-160)
- Test: `tests/test_image.py`, `tests/test_mask.py`, `tests/test_color_image.py` (confirm names with `ls tests/`)

**Interfaces:**
- Consumes: `Report`, `ReportField`, `DescribableMixin` (Tasks 1-4); `MaskSchema._describe` (Task 5); `SkyProjection._describe` (Task 6).
- Produces: `_describe()` on `Image`, `Mask`, `ColorImage`, each adopting `DescribableMixin` (note: they extend `GeneralizedImage` — add the mixin to that base or to each class; see Step 2). Preserved outputs:
  - `Image`: `str` = `"Image({bbox!s}, {dtype.type.__name__})"`; `repr` = `"Image(..., bbox={bbox!r}, dtype={dtype!r})"`.
  - `Mask`: `str` = `"Mask({bbox!s}, {list(schema.names)})"`; `repr` = `"Mask(..., bbox={bbox!r}, schema={schema!r})"`.
  - `ColorImage`: `str` = `"ColorImage({bbox!s}, {dtype.type.__name__})"`; `repr` = `"ColorImage(..., bbox={bbox!r}, dtype={dtype!r})"`.

- [ ] **Step 1: Pin current output (characterization tests)**

Add to `tests/test_image.py`:

```python
def test_image_repr_str_pinned() -> None:
    """Image str/repr match their documented forms."""
    image = Image(np.zeros((5, 4), dtype=np.float32), bbox=Box.factory[0:5, 0:4])
    assert str(image) == f"Image({image.bbox!s}, float32)"
    assert repr(image) == f"Image(..., bbox={image.bbox!r}, dtype={image.array.dtype!r})"
```

Add to `tests/test_color_image.py` an analogous `test_color_image_repr_str_pinned`, and to `tests/test_mask.py`:

```python
def test_mask_repr_str_pinned() -> None:
    """Mask str/repr match their documented forms."""
    rng = np.random.default_rng(7)
    schema = MaskSchema(make_mask_planes(rng, 3, 0), dtype=np.uint8)
    mask = Mask(schema, bbox=Box.factory[0:5, 0:4])
    assert str(mask) == f"Mask({mask.bbox!s}, {list(mask.schema.names)})"
    assert repr(mask) == f"Mask(..., bbox={mask.bbox!r}, schema={mask.schema!r})"
```

(Adjust `Mask(...)` construction to match the real signature — check `grep -n "def __init__" python/lsst/images/_mask.py` around the `Mask` class.)

Run these three pinning tests; expect PASS (baseline).

- [ ] **Step 2: Decide where the mixin attaches**

`Image`, `Mask`, `ColorImage`, `MaskedImage` all extend `GeneralizedImage` (`python/lsst/images/_generalized_image.py`).
Adopt `DescribableMixin` on `GeneralizedImage` so every image type inherits the derived `__str__`/`_repr_html_`/`__rich__`/`describe`, then implement `_describe()` per concrete class.
In `_generalized_image.py`, add `from .describe import DescribableMixin` and change the class base to include `DescribableMixin`.
Confirm `GeneralizedImage` does not itself define `__str__`/`__repr__` that would shadow the mixin (grep it); if it does, remove them and rely on subclasses' `_describe`.

- [ ] **Step 3: Implement `Image._describe`**

In `_image.py`, add `from .describe import Report, ReportField` (the mixin comes via `GeneralizedImage`).
Delete `Image.__str__`/`__repr__` and add:

```python
    def _describe(self, **kwargs: Any) -> Report:
        """Return a `Report` describing this image.

        Parameters
        ----------
        **kwargs
            Unused; accepted for interface compatibility.
        """
        children = {}
        if self._sky_projection is not None:
            children["sky_projection"] = self._sky_projection._describe(bbox=self._bbox)
        return Report(
            type_name="Image",
            summary=f"Image({self.bbox!s}, {self.array.dtype.type.__name__})",
            fields=[
                ReportField(label="array", value="<array>", repr_value="...", positional=True),
                ReportField(label="bbox", value=self.bbox, repr_value=repr(self.bbox)),
                ReportField(label="dtype", value=str(self.array.dtype), repr_value=repr(self.array.dtype)),
            ],
            children=children,
        )
```

`to_repr()` yields `Image(..., bbox=Box(...), dtype=dtype('float32'))`, matching today.
`to_str()` uses `summary` → `Image([y=0:5, x=0:4], float32)`, matching today.

- [ ] **Step 4: Implement `Mask._describe` (nests MaskSchema)**

In `_mask.py`, delete `Mask.__str__`/`__repr__` and add:

```python
    def _describe(self, **kwargs: Any) -> Report:
        """Return a `Report` describing this mask.

        Parameters
        ----------
        **kwargs
            Unused; accepted for interface compatibility.
        """
        children = {"schema": self.schema._describe()}
        if self._sky_projection is not None:
            children["sky_projection"] = self._sky_projection._describe(bbox=self._bbox)
        return Report(
            type_name="Mask",
            summary=f"Mask({self.bbox!s}, {list(self.schema.names)})",
            fields=[
                ReportField(label="schema", value="<schema>", repr_value="...", positional=True),
                ReportField(label="bbox", value=self.bbox, repr_value=repr(self.bbox)),
                ReportField(label="schema", value=self.schema, repr_value=repr(self.schema)),
            ],
            children=children,
        )
```

Note the current `Mask.__repr__` is `Mask(..., bbox=..., schema=...)` — a lossy positional `...` plus `bbox=` and `schema=` keyword args.
Use ONE positional `...` field, then `bbox` and `schema` keyword fields (drop the duplicate first `schema` label shown above; the positional field represents the pixel array, so label it `array`):

```python
            fields=[
                ReportField(label="array", value="<array>", repr_value="...", positional=True),
                ReportField(label="bbox", value=self.bbox, repr_value=repr(self.bbox)),
                ReportField(label="schema", value=self.schema, repr_value=repr(self.schema)),
            ],
```

This yields `Mask(..., bbox=Box(...), schema=MaskSchema([...], dtype=...))`, matching today.
Verify the sky-projection attribute name on `Mask` (`grep -n "_sky_projection\|sky_projection" python/lsst/images/_mask.py`); use whatever the accessor is.

- [ ] **Step 5: Implement `ColorImage._describe`**

In `_color_image.py`, delete `ColorImage.__str__`/`__repr__` and add the analogue of `Image._describe` with `type_name="ColorImage"`, the `ColorImage(...)` summary, the same positional `...`/`bbox`/`dtype` fields, and a `sky_projection` child when present.

- [ ] **Step 6: Run tests, ruff, mypy**

Run: `.pyenv/bin/pytest tests/test_image.py tests/test_mask.py tests/test_color_image.py -v && .pyenv/bin/ruff check python/lsst/images/_image.py python/lsst/images/_mask.py python/lsst/images/_color_image.py python/lsst/images/_generalized_image.py && .pyenv/bin/mypy python/lsst/images/_image.py python/lsst/images/_mask.py python/lsst/images/_color_image.py python/lsst/images/_generalized_image.py`
Expected: PASS, clean.

- [ ] **Step 7: Commit**

```bash
git add python/lsst/images/_image.py python/lsst/images/_mask.py python/lsst/images/_color_image.py python/lsst/images/_generalized_image.py tests/test_image.py tests/test_mask.py tests/test_color_image.py
git commit -m "Migrate leaf image containers to describe reports

Generated with AI

Co-Authored-By: SLAC AI"
```

---

### Task 9: Composite containers — `MaskedImage`, `VisitImage`, `DifferenceImage`, `CellCoadd`

These nest their components as children and thread their own `bbox` into the `sky_projection` child so the WCS sub-report shows corner sky coordinates automatically.
All four have hand-written `str`/`repr` to preserve.

**Files:**
- Modify: `python/lsst/images/_masked_image.py` (`MaskedImage`, ~196-200)
- Modify: `python/lsst/images/_visit_image.py` (`VisitImage`, ~315-319)
- Modify: `python/lsst/images/_difference_image.py` (`DifferenceImage`, ~253-256)
- Modify: `python/lsst/images/cells/_coadd.py` (`CellCoadd`, ~266-270)
- Test: `tests/test_masked_image.py`, `tests/test_visit_image.py`, `tests/test_difference_image.py`, `tests/test_cell_coadd.py` (confirm names with `ls tests/`)

**Interfaces:**
- Consumes: `Report`, `ReportField`, `DescribableMixin` (Tasks 1-4); child `_describe` from `Image`/`Mask`/`SkyProjection` (Tasks 6-8).
- Produces: `_describe()` on each, adopting `DescribableMixin` (inherited via `GeneralizedImage` for `MaskedImage`/`VisitImage`/`DifferenceImage`/`CellCoadd`, which all descend from it). Preserved outputs:
  - `MaskedImage`: `str` = `"MaskedImage({image!s}, {list(mask.schema.names)})"`; `repr` = `"MaskedImage({image!r}, mask_schema={mask.schema!r})"`.
  - `VisitImage`: `str` = `"VisitImage({image!s}, {list(mask.schema.names)})"`; `repr` = `"VisitImage({image!r}, mask_schema={mask.schema!r})"`.
  - `DifferenceImage`: check its exact current `str`/`repr` (`sed -n '253,256p' python/lsst/images/_difference_image.py`) and reproduce verbatim.
  - `CellCoadd`: `str` = `"CellCoadd({bbox!s}, tract={tract})"`; `repr` = check `sed -n '269,271p' python/lsst/images/cells/_coadd.py` and reproduce verbatim.

- [ ] **Step 1: Pin current output for all four**

For each of the four test files, add a `test_<name>_repr_str_pinned` asserting the exact current `str(obj)` and `repr(obj)`.
Use whatever fixtures the existing tests use to build these objects (search each test file for how it constructs the object, e.g. `grep -n "= VisitImage\|make_visit_image\|read_archive" tests/test_visit_image.py`).
Run all four; expect PASS (baseline).

- [ ] **Step 2: Implement `MaskedImage._describe`**

In `_masked_image.py`, add `from .describe import Report, ReportField`, delete `MaskedImage.__str__`/`__repr__`, and add:

```python
    def _describe(self, **kwargs: Any) -> Report:
        """Return a `Report` describing this masked image.

        Parameters
        ----------
        **kwargs
            Unused; accepted for interface compatibility.
        """
        children = {
            "image": self._image._describe(),
            "mask": self._mask._describe(),
            "variance": self._variance._describe(),
        }
        if self.sky_projection is not None:
            children["sky_projection"] = self.sky_projection._describe(bbox=self.bbox)
        return Report(
            type_name="MaskedImage",
            summary=f"MaskedImage({self.image!s}, {list(self.mask.schema.names)})",
            fields=[
                ReportField(label="image", value=self.image, repr_value=repr(self.image), positional=True),
                ReportField(label="mask_schema", value=self.mask.schema, repr_value=repr(self.mask.schema)),
            ],
            children=children,
        )
```

`to_repr()` yields `MaskedImage(Image(...), mask_schema=MaskSchema(...))`, matching today.

- [ ] **Step 3: Implement `VisitImage._describe`**

In `_visit_image.py`, add `from .describe import Report, ReportField`, delete `VisitImage.__str__`/`__repr__`, and add a `_describe` mirroring `MaskedImage` with `type_name="VisitImage"` and the `VisitImage(...)` summary, but with these additional children when the corresponding attribute is present (all are non-optional accessors on `VisitImage` — see the property list; guard optional ones like `photometric_scaling`):

```python
        children = {
            "image": self.image._describe(),
            "mask": self.mask._describe(),
            "variance": self.variance._describe(),
            "sky_projection": self.sky_projection._describe(bbox=self.bbox),
            "psf": self.psf._describe(),
            "detector": self.detector._describe(),
            "summary_stats": self.summary_stats._describe(),
            "aperture_corrections": self.aperture_corrections._describe(),
            "backgrounds": self.backgrounds._describe(),
        }
        if self.photometric_scaling is not None:
            children["photometric_scaling"] = self.photometric_scaling._describe()
```

Add `ARG` fields matching the current `repr` (`image` positional + `mask_schema=`), plus `DERIVED` fields for `band` and `physical_filter` (these appear in the rich view only, not `repr`):

```python
        fields=[
            ReportField(label="image", value=self.image, repr_value=repr(self.image), positional=True),
            ReportField(label="mask_schema", value=self.mask.schema, repr_value=repr(self.mask.schema)),
            ReportField(label="band", value=self.band, role=FieldRole.DERIVED),
            ReportField(label="physical_filter", value=self.physical_filter, role=FieldRole.DERIVED),
        ],
```

(Import `FieldRole` too.)
The `_describe` calls on `psf`, `detector`, `summary_stats`, `aperture_corrections`, `backgrounds`, and `photometric_scaling` depend on Task 10 giving those classes a `_describe`; until then, they will `AttributeError`.
To keep this task independently green, guard each child with `getattr`-style capability checks:

```python
        children = {"image": self.image._describe(), "mask": self.mask._describe(),
                    "variance": self.variance._describe(),
                    "sky_projection": self.sky_projection._describe(bbox=self.bbox)}
        for name, comp in [
            ("psf", self.psf), ("detector", self.detector),
            ("summary_stats", self.summary_stats),
            ("aperture_corrections", self.aperture_corrections),
            ("backgrounds", self.backgrounds),
        ]:
            if hasattr(comp, "_describe"):
                children[name] = comp._describe()
        if self.photometric_scaling is not None and hasattr(self.photometric_scaling, "_describe"):
            children["photometric_scaling"] = self.photometric_scaling._describe()
```

Once Task 10 lands, every component has `_describe`, so all children populate.

- [ ] **Step 4: Implement `DifferenceImage._describe` and `CellCoadd._describe`**

`DifferenceImage` extends `VisitImage`.
Its current output is `str` = `"DifferenceImage({image!s}, {list(mask.schema.names)})"` and `repr` = `"DifferenceImage({image!r}, mask_schema={mask.schema!r})"` — the same layout as `VisitImage` with a different type name.
Reuse the parent report and retarget it:

```python
    def _describe(self, **kwargs: Any) -> Report:
        """Return a `Report` describing this difference image.

        Parameters
        ----------
        **kwargs
            Unused; accepted for interface compatibility.
        """
        report = super()._describe(**kwargs)
        report.type_name = "DifferenceImage"
        report.summary = f"DifferenceImage({self.image!s}, {list(self.mask.schema.names)})"
        if hasattr(self.kernel, "_describe"):
            report.children["kernel"] = self.kernel._describe()
        return report
```

Because the parent's `ARG` fields are `image` (positional) and `mask_schema`, `to_repr()` emits `DifferenceImage(Image(...), mask_schema=MaskSchema(...))`, matching today.

`CellCoadd` extends `MaskedImage`, and its current `repr` is defined as `return str(self)` — i.e. `repr(cell_coadd) == str(cell_coadd) == "CellCoadd({bbox!s}, tract={tract})"`, a non-eval-ish form.
This is the one class whose `repr` is deliberately identical to `str`.
To preserve it exactly, do NOT rely on the mixin's default `to_repr`; override `__repr__` on `CellCoadd` to return `str(self)`, and let `_describe` drive `str`/rich/HTML:

```python
    def _describe(self, **kwargs: Any) -> Report:
        """Return a `Report` describing this cell coadd.

        Parameters
        ----------
        **kwargs
            Unused; accepted for interface compatibility.
        """
        children = {
            "image": self.image._describe(),
            "mask": self.mask._describe(),
            "variance": self.variance._describe(),
            "sky_projection": self.sky_projection._describe(bbox=self.bbox),
        }
        for name, comp in [("psf", self.psf), ("backgrounds", self.backgrounds)]:
            if hasattr(comp, "_describe"):
                children[name] = comp._describe()
        return Report(
            type_name="CellCoadd",
            summary=f"CellCoadd({self.bbox!s}, tract={self.tract})",
            fields=[
                ReportField(label="skymap", value=self.skymap, role=FieldRole.DERIVED),
                ReportField(label="tract", value=self.tract, role=FieldRole.DERIVED),
                ReportField(label="patch", value=self.patch, role=FieldRole.DERIVED),
                ReportField(label="band", value=self.band, role=FieldRole.DERIVED),
            ],
            children=children,
        )

    def __repr__(self) -> str:
        return str(self)
```

(Import `FieldRole` in `_coadd.py`.)
Since all `CellCoadd` report fields are `DERIVED`, the mixin's `to_repr` would emit `CellCoadd()`; the explicit `__repr__` override above keeps the current `str`-equals-`repr` behavior.

- [ ] **Step 5: Run tests, ruff, mypy for all four**

Run: `.pyenv/bin/pytest tests/test_masked_image.py tests/test_visit_image.py tests/test_difference_image.py tests/test_cell_coadd.py -v && .pyenv/bin/ruff check python/lsst/images/_masked_image.py python/lsst/images/_visit_image.py python/lsst/images/_difference_image.py python/lsst/images/cells/_coadd.py && .pyenv/bin/mypy python/lsst/images/_masked_image.py python/lsst/images/_visit_image.py python/lsst/images/_difference_image.py python/lsst/images/cells/_coadd.py`
Expected: PASS, clean.

- [ ] **Step 6: Commit**

```bash
git add python/lsst/images/_masked_image.py python/lsst/images/_visit_image.py python/lsst/images/_difference_image.py python/lsst/images/cells/_coadd.py tests/test_masked_image.py tests/test_visit_image.py tests/test_difference_image.py tests/test_cell_coadd.py
git commit -m "Migrate composite image containers to describe reports

Generated with AI

Co-Authored-By: SLAC AI"
```

---

### Task 10: Characterization models — PSFs, fields, `BackgroundMap`, `ObservationSummaryStats`, `Detector`

These classes are mostly new to `repr` (except `GaussianPointSpreadFunction`, which has one to preserve).
`ObservationSummaryStats` is a pydantic model — keep pydantic's `repr`, but add `_describe` and the mixin's rich views by adopting the mixin without overriding `__repr__` (see Step 5).
`ApertureCorrectionMap` is a `dict` type alias, not a class, so it is skipped.
`Amplifier` and `AmplifierCalibrations` are pydantic models with adequate default `repr`s; they are out of scope here (the spec lists `Amplifier`, but it is nested plumbing under `Detector` and gains little from a report — leave it to a follow-up if a user asks).

**Files:**
- Modify: `python/lsst/images/psfs/_base.py` (`PointSpreadFunction` — add mixin + default `_describe`), `_gaussian.py`, `_piff.py`, `_legacy.py`
- Modify: `python/lsst/images/fields/_base.py` (`BaseField` — add mixin + default `_describe`), and each subclass that adds distinctive parameters
- Modify: `python/lsst/images/_backgrounds.py` (`BackgroundMap`)
- Modify: `python/lsst/images/_observation_summary_stats.py` (`ObservationSummaryStats`)
- Modify: `python/lsst/images/cameras.py` (`Detector`)
- Test: `tests/test_psfs.py`, `tests/test_fields.py`, `tests/test_backgrounds.py`, `tests/test_observation_summary_stats.py`, `tests/test_cameras.py` (confirm names with `ls tests/`)

**Interfaces:**
- Consumes: `Report`, `ReportField`, `FieldRole`, `DescribableMixin` (Tasks 1-4).
- Produces: `_describe()` on `PointSpreadFunction` (base default, overridden by `GaussianPointSpreadFunction`, `PiffWrapper`, `PSFExWrapper`, `LegacyPointSpreadFunction`), `BaseField` (base default, overridden by `ChebyshevField`/`SplineField`/`ProductField`/`SumField`), `BackgroundMap`, `ObservationSummaryStats`, `Detector`.
  - `GaussianPointSpreadFunction.repr` must remain `"GaussianPointSpreadFunction({sigma}, stamp_size={stamp_size}, bounds={bounds!r})"`.

- [ ] **Step 1: Pin `GaussianPointSpreadFunction` repr**

Add to `tests/test_psfs.py`:

```python
def test_gaussian_psf_repr_pinned() -> None:
    """GaussianPointSpreadFunction repr matches its documented form."""
    psf = GaussianPointSpreadFunction(2.5, bounds=Box.factory[0:10, 0:10], stamp_size=21)
    assert repr(psf) == f"GaussianPointSpreadFunction(2.5, stamp_size=21, bounds={psf.bounds!r})"
```

Run it; expect PASS (baseline).

- [ ] **Step 2: PSF base mixin and default `_describe`**

In `python/lsst/images/psfs/_base.py`, add `from ..describe import DescribableMixin, Report, ReportField, FieldRole`, change `class PointSpreadFunction(ABC):` to `class PointSpreadFunction(DescribableMixin, ABC):`, and add a default `_describe`:

```python
    def _describe(self, **kwargs: Any) -> Report:
        """Return a `Report` describing this PSF.

        Parameters
        ----------
        **kwargs
            Unused; accepted for interface compatibility.
        """
        return Report(
            type_name=type(self).__name__,
            summary=f"{type(self).__name__} over {self.bounds}",
            fields=[
                ReportField(label="bounds", value=self.bounds, role=FieldRole.DERIVED),
                ReportField(label="kernel_bbox", value=self.kernel_bbox, role=FieldRole.DERIVED),
            ],
        )
```

- [ ] **Step 3: `GaussianPointSpreadFunction._describe` (preserves repr)**

In `_gaussian.py`, delete `__repr__` and add:

```python
    def _describe(self, **kwargs: Any) -> Report:
        """Return a `Report` describing this Gaussian PSF.

        Parameters
        ----------
        **kwargs
            Unused; accepted for interface compatibility.
        """
        return Report(
            type_name="GaussianPointSpreadFunction",
            summary=f"Gaussian PSF sigma={self.sigma}",
            fields=[
                ReportField(label="sigma", value=self.sigma, positional=True),
                ReportField(label="stamp_size", value=self._stamp_size),
                ReportField(label="bounds", value=self._bounds, repr_value=repr(self._bounds)),
            ],
        )
```

`to_repr()` yields `GaussianPointSpreadFunction(2.5, stamp_size=21, bounds=Box(...))`, matching today.

- [ ] **Step 4: PSF wrapper overrides (`PiffWrapper`, `PSFExWrapper`, `LegacyPointSpreadFunction`)**

Add a `_describe` to each wrapper listing its distinctive attributes as `DERIVED` fields (e.g. `bounds`, `kernel_bbox`, and any degree/order/model parameters exposed as public properties — inspect each class's `@property` list).
Where a wrapper adds no distinctive public parameters beyond the base, rely on the base default and skip an override.

- [ ] **Step 5: `ObservationSummaryStats` (pydantic — keep repr, add rich views)**

In `_observation_summary_stats.py`, add `from .describe import DescribableMixin, Report, ReportField, FieldRole`.
`ObservationSummaryStats` extends `ArchiveTree` (a pydantic model); add `DescribableMixin` to its bases AFTER the pydantic base so pydantic's `__repr__` still wins (pydantic defines `__repr__` on `BaseModel`, which sits earlier in the MRO than the mixin only if listed first — verify with a quick `repr()` check in Step 7).
If the mixin's `__repr__` shadows pydantic's, do NOT add the mixin to the bases; instead define only `_repr_html_` and `__rich__` methods on the class that delegate to `self._describe()`, leaving pydantic's `__repr__`/`__str__` intact.
Add `_describe` grouping the stat fields:

```python
    def _describe(self, **kwargs: Any) -> Report:
        """Return a `Report` describing these summary statistics.

        Parameters
        ----------
        **kwargs
            Unused; accepted for interface compatibility.
        """
        fields = [
            ReportField(label=name, value=getattr(self, name), role=FieldRole.DERIVED)
            for name in ("psfSigma", "psfArea", "ra", "dec", "pixelScale")
        ]
        return Report(type_name="ObservationSummaryStats", fields=fields)
```

(Since these are DERIVED, they never feed `to_repr`, so pydantic's repr is unaffected regardless.)

- [ ] **Step 6: `BackgroundMap`, `BaseField`/subclasses, `Detector`**

- `BackgroundMap` (extends `Mapping`): add mixin + `_describe` with a `DERIVED` field for `subtracted` and one `DERIVED` field per background key (or a small `ReportTable` titled "Backgrounds" with columns `["Name", "Subtracted"]`).
- `BaseField`: add mixin + default `_describe` (`bounds`, `unit`, `is_constant` as DERIVED). Override in `ChebyshevField`/`SplineField` to add `order`/`degree`/basis info; `ProductField`/`SumField` list their operand fields as children.
- `Detector`: add mixin + `_describe` with DERIVED fields `instrument`, `name`, `id`, `type`, `serial`, `bbox`.

Each override follows the same shape as Steps 2-3; use only public accessors confirmed by grepping each class.

- [ ] **Step 7: Run tests, ruff, mypy**

Run: `.pyenv/bin/pytest tests/test_psfs.py tests/test_fields.py tests/test_backgrounds.py tests/test_observation_summary_stats.py tests/test_cameras.py -v`
Then verify pydantic repr is intact:
`.pyenv/bin/python -c "from lsst.images import ObservationSummaryStats; print(repr(ObservationSummaryStats()))"`
Expected: the pydantic-style `ObservationSummaryStats(psfSigma=nan, ...)` repr, not a mixin-derived one.
Then ruff + mypy on all modified files.

- [ ] **Step 8: Commit**

```bash
git add python/lsst/images/psfs python/lsst/images/fields python/lsst/images/_backgrounds.py python/lsst/images/_observation_summary_stats.py python/lsst/images/cameras.py tests/
git commit -m "Add describe reports to characterization models

Generated with AI

Co-Authored-By: SLAC AI"
```

---

### Task 11: Transforms and frames — `Transform`, `FrameSet`, `CameraFrameSet`, `*Frame`

`Transform`, `FrameSet`, and `CameraFrameSet` are plain classes new to `repr`.
The `*Frame` classes (`DetectorFrame`, `FocalPlaneFrame`, `FieldAngleFrame`, `TractFrame`, `GeneralFrame`) are pydantic models; keep pydantic's `repr` and only add `_describe` + rich views (same policy as `ObservationSummaryStats` in Task 10).
`SkyFrame` is a `StrEnum` and needs nothing.

**Files:**
- Modify: `python/lsst/images/_transforms/_transform.py` (`Transform`)
- Modify: `python/lsst/images/_transforms/_frame_set.py` (`FrameSet`)
- Modify: `python/lsst/images/_transforms/_camera_frame_set.py` (`CameraFrameSet`)
- Modify: `python/lsst/images/_transforms/_frames.py` (the five `*Frame` classes)
- Test: `tests/test_transforms.py`

**Interfaces:**
- Consumes: `Report`, `ReportField`, `FieldRole`, `DescribableMixin` (Tasks 1-4).
- Produces: `_describe()` on `Transform`, `FrameSet`, `CameraFrameSet`, and each `*Frame`, plus rich views. `Transform`/`FrameSet`/`CameraFrameSet` gain mixin-derived `__repr__`/`__str__`; the `*Frame` pydantic models retain pydantic's `__repr__`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_transforms.py`:

```python
def test_transform_describe() -> None:
    """Transform._describe reports its frames and bounds."""
    from lsst.images.describe import Report

    pixel_frame = DetectorFrame(**DP2_VISIT_DETECTOR_DATA_ID, bbox=Box.factory[:5, :4])
    transform = Transform(pixel_frame, ICRS, astshim.UnitMap(2))
    report = transform.describe()
    assert isinstance(report, Report)
    assert report.type_name == "Transform"
    labels = {f.label for f in report.fields}
    assert {"in_frame", "out_frame"} <= labels
    # The AST dump is available as a DERIVED field.
    assert any(f.label == "mapping" for f in report.fields)
```

(Reuse `DP2_VISIT_DETECTOR_DATA_ID` and `astshim`/`ICRS`/`DetectorFrame` already imported in the test module.)

Run: `.pyenv/bin/pytest tests/test_transforms.py::test_transform_describe -v`
Expected: FAIL with `AttributeError: 'Transform' object has no attribute 'describe'`.

- [ ] **Step 2: `Transform._describe`**

In `_transform.py`, add `from ..describe import DescribableMixin, FieldRole, Report, ReportField`, change `class Transform[I: Frame, O: Frame]:` to `class Transform[I: Frame, O: Frame](DescribableMixin):`, and add:

```python
    def _describe(self, **kwargs: Any) -> Report:
        """Return a `Report` describing this transform.

        Parameters
        ----------
        **kwargs
            Unused; accepted for interface compatibility.
        """
        return Report(
            type_name="Transform",
            summary=f"{self.in_frame!s} → {self.out_frame!s}",
            fields=[
                ReportField(label="in_frame", value=self.in_frame, role=FieldRole.DERIVED),
                ReportField(label="out_frame", value=self.out_frame, role=FieldRole.DERIVED),
                ReportField(label="in_bounds", value=self.in_bounds, role=FieldRole.DERIVED),
                ReportField(label="out_bounds", value=self.out_bounds, role=FieldRole.DERIVED),
                ReportField(
                    label="mapping",
                    value=self.show(simplified=True),
                    role=FieldRole.DERIVED,
                ),
            ],
        )
```

Because all fields are `DERIVED`, `to_repr()` emits `Transform()`; that is acceptable for a class that previously had no `repr`.
If a more useful `repr` is wanted, add one `ARG` field carrying `repr_value="..."`; keep it minimal for now.

- [ ] **Step 3: `FrameSet` and `CameraFrameSet`**

`FrameSet` is abstract; add `DescribableMixin` to its bases (`class FrameSet(DescribableMixin, ABC):`) and give it no concrete `_describe` (subclasses provide it), OR a minimal default reporting `type_name` only.
`CameraFrameSet._describe`: report `instrument` as a `DERIVED` field and list the available frame kinds.

```python
    # In CameraFrameSet:
    def _describe(self, **kwargs: Any) -> Report:
        """Return a `Report` describing this camera frame set.

        Parameters
        ----------
        **kwargs
            Unused; accepted for interface compatibility.
        """
        return Report(
            type_name="CameraFrameSet",
            summary=f"CameraFrameSet({self.instrument!r})",
            fields=[ReportField(label="instrument", value=self.instrument, role=FieldRole.DERIVED)],
        )
```

Add the imports to both modules.

- [ ] **Step 4: `*Frame` pydantic models**

For `DetectorFrame`, `FocalPlaneFrame`, `FieldAngleFrame`, `TractFrame`, `GeneralFrame`, follow the `ObservationSummaryStats` policy: add `_describe` plus `_repr_html_`/`__rich__` delegating to it, WITHOUT letting the mixin's `__repr__` shadow pydantic's.
The simplest safe approach: define three small methods on each frame class (or on a shared base if they have one) rather than inheriting the full mixin:

```python
    def _describe(self, **kwargs: Any) -> Report:
        """Return a `Report` describing this frame.

        Parameters
        ----------
        **kwargs
            Unused; accepted for interface compatibility.
        """
        fields = [
            ReportField(label=name, value=getattr(self, name), role=FieldRole.DERIVED)
            for name in type(self).model_fields
        ]
        return Report(type_name=type(self).__name__, fields=fields)

    def _repr_html_(self) -> str:
        return self._describe()._repr_html_()

    def __rich__(self) -> Any:
        return self._describe().__rich__()
```

Factor these into a small helper to avoid repetition across the five classes (e.g. a module-level function `_frame_report(frame)` returning the `Report`, called by each `_describe`).
Do NOT add `describe()` if you skip the mixin; add it explicitly if a public entry point is desired:

```python
    def describe(self, **kwargs: Any) -> Report:
        return self._describe(**kwargs)
```

- [ ] **Step 5: Run tests, verify pydantic frame repr intact, ruff, mypy**

Run: `.pyenv/bin/pytest tests/test_transforms.py -v`
Then: `.pyenv/bin/python -c "import astropy.units as u; from lsst.images import GeneralFrame; print(repr(GeneralFrame(unit=u.pix)))"`
Expected: pydantic-style `GeneralFrame(unit=...)` repr.
Then ruff + mypy on all four modified transform modules.

- [ ] **Step 6: Commit**

```bash
git add python/lsst/images/_transforms tests/test_transforms.py
git commit -m "Add describe reports to transforms and frames

Generated with AI

Co-Authored-By: SLAC AI"
```

---

### Task 12: CLI `describe` subcommand, package exports, and integration test

Wire the report system into the package's public API and add the CLI `describe` command that deserializes a file and renders the top-level object's report via the rich terminal renderer.
Leave `inspect` unchanged.

**Files:**
- Modify: `python/lsst/images/__init__.py` (export describe API)
- Create: `python/lsst/images/cli/_describe.py`
- Modify: `python/lsst/images/cli/_main.py` (register `describe`)
- Test: `tests/test_describe.py` (integration), `tests/test_cli.py` (confirm name with `ls tests/ | grep -i cli`; otherwise create `tests/test_cli_describe.py`)

**Interfaces:**
- Consumes: `read_archive` from `lsst.images.serialization`; `Describable`/`Report` from `lsst.images.describe`; `rich.console.Console`.
- Produces: `describe` click command; `from .describe import *` in the package `__init__`.

- [ ] **Step 1: Confirm the describe API is exported**

Task 4 already added `from .describe import *` to `python/lsst/images/__init__.py`.
Verify it is present and working: `.pyenv/bin/python -c "from lsst.images import Report, ReportField, ReportTable, FieldRole, Describable, DescribableMixin; print('ok')"`
Expected: `ok`.
If missing (Task 4 skipped), add `from .describe import *` after `from ._difference_image import *`.

- [ ] **Step 2: Write the failing integration test for a nested container**

Add to `tests/test_describe.py`:

```python
def test_visit_image_describe_nested() -> None:
    """A deserialized VisitImage produces a nested report with WCS corners."""
    import os

    from lsst.images.serialization import read_archive

    path = os.path.join(os.path.dirname(__file__), "data", "schema_v1", "visit_image.json")
    visit_image = read_archive(path)
    report = visit_image.describe()
    assert report.type_name == "VisitImage"
    # Components appear as children.
    assert "image" in report.children
    assert "mask" in report.children
    assert "sky_projection" in report.children
    # The sky_projection child received the container bbox, so it has corners.
    sky = report.children["sky_projection"]
    assert any(t.title == "Corners" for t in sky.tables)
    # Rich and HTML renderers run without error.
    assert isinstance(report._repr_html_(), str)
    report.__rich__()
```

Run: `.pyenv/bin/pytest tests/test_describe.py::test_visit_image_describe_nested -v`
Expected: FAIL until Tasks 6-9 are complete (the child reports and bbox threading must exist).
If running this task before Tasks 6-9, mark it xfail temporarily; otherwise expect PASS once those tasks land.

- [ ] **Step 3: Write the CLI `describe` command**

Create `python/lsst/images/cli/_describe.py` with the standard license header (copy from `_inspect.py`), then:

```python
from __future__ import annotations

__all__ = ("describe",)

import click
from rich.console import Console

from ..describe import Describable
from ..serialization import ArchiveReadError, read_archive


@click.command(name="describe")
@click.argument("file", type=click.Path(exists=True, dir_okay=False))
def describe(file: str) -> None:  # numpydoc ignore=PR01
    """Deserialize an lsst.images file and print its data-model report.

    Unlike ``inspect`` (which reports only the file layout), this reads the
    full object and renders its `~lsst.images.Report` via the rich terminal
    renderer, including nested components and, where available, WCS corner
    sky coordinates.
    """
    try:
        obj = read_archive(file)
    except (ArchiveReadError, ValueError, TypeError) as err:
        raise click.ClickException(f"Could not read {file}: {err}") from None
    if not isinstance(obj, Describable):
        raise click.ClickException(
            f"{type(obj).__name__} does not support 'describe'."
        )
    Console().print(obj.describe())
```

- [ ] **Step 4: Register the command**

In `python/lsst/images/cli/_main.py`, add `from ._describe import describe` (alphabetically after `from ._convert import convert`) and `main.add_command(describe)` (after `main.add_command(convert)`).

- [ ] **Step 5: Write and run the CLI test**

Add to the CLI test file (using click's `CliRunner`, following the existing CLI test pattern — inspect `tests/test_cli.py` or the closest existing CLI test for the import and invocation style):

```python
def test_cli_describe_visit_image() -> None:
    """The describe command renders a deserialized VisitImage."""
    import os

    from click.testing import CliRunner

    from lsst.images.cli._main import main

    path = os.path.join(os.path.dirname(__file__), "data", "schema_v1", "visit_image.json")
    result = CliRunner().invoke(main, ["describe", path])
    assert result.exit_code == 0, result.output
    assert "VisitImage" in result.output
```

Run: `.pyenv/bin/pytest tests/test_describe.py tests/test_cli*.py -v`
Expected: PASS.

- [ ] **Step 6: Full test suite, ruff, mypy**

Run: `.pyenv/bin/pytest -q && .pyenv/bin/ruff check python/lsst/images && .pyenv/bin/mypy python/lsst/images`
Expected: all pass, no lint/type errors.

- [ ] **Step 7: Commit**

```bash
git add python/lsst/images/__init__.py python/lsst/images/cli/_describe.py python/lsst/images/cli/_main.py tests/
git commit -m "Add describe CLI subcommand and export describe API

Generated with AI

Co-Authored-By: SLAC AI"
```

---

## Final verification

After all tasks:

- [ ] Run the full suite once more: `.pyenv/bin/pytest -q`.
- [ ] Confirm every migrated class's pinned `str`/`repr` test still passes (behavioral compatibility).
- [ ] Confirm `.pyenv/bin/python -c "import lsst.images"` emits no warnings and the `rich` dependency resolves.
- [ ] Spot-check notebook HTML for one nested container by writing `VisitImage.describe()._repr_html_()` to a file and opening it, verifying the tree, "Axes"/"Corners"/"Mask planes" tables, and nested children render.

# Structured `describe()` reports and rich display for `lsst.images`

## Problem

Many `lsst.images` types have no useful string representation.
In a notebook, holding a `SkyProjection`, `Transform`, PSF, or field prints `<lsst.images...object at 0x...>` because these are plain Python classes with no `__repr__`.
Others (the pydantic serialization models, camera models, frames) inherit a default repr that lists fields but offers no rich display and no domain-aware summary.

No class in the package defines `_repr_html_`, `_repr_pretty_`, `__rich__`, or a data-model description method.
Rich Jupyter/terminal display is entirely absent.

Separately, the CLI `inspect` command reports only the file layer (schema name/version/URL, container format version) and never describes the deserialized data model.
There is no way, from the command line or a notebook, to see a human-readable summary of a `SkyProjection`'s sky coverage, a PSF's parameters, or how a `VisitImage`'s components fit together.

The motivating comparison is Starlink KAPPA `ndftrace`, which reports WCS like:

```
        Frame title         : "ICRS coordinates"
        Domain              : SKY
        First pixel centre  : 10:04:21.6, 2:13:22

           Axis 1:
              Label              : Right ascension
              Units              : hh:mm:ss.s
              Nominal Pixel scale: 0.199966 arc-sec
```

Astropy's `WCS` repr, by contrast, prints a FITS header, which is not generally applicable to the generalized (non-FITS) mappings this package supports.

## Goals

- A single, renderer-agnostic description of each user-facing object that drives plain-text, rich-terminal, notebook-HTML, and `repr` output from one source, so the four never drift.
- Domain-aware WCS reporting (frame/domain metadata, sky coordinates of pixel-box corners, nominal pixel scale, FITS-WCS availability) in the spirit of KAPPA `ndftrace` rather than a FITS header dump.
- First-class notebook and terminal display for the classes a user actually holds and discovers from `VisitImage`, `CellCoadd`, `DifferenceImage`, and friends.
- A CLI `describe` subcommand that deserializes a file and prints its data-model report, kept separate from the existing file-layer `inspect` command.
- Genuine tabular rendering (aligned columns, header row) for the two classes that most benefit from it: `SkyProjection` (per-axis and corner tables) and `MaskSchema` (mask-plane table).

## Non-goals

- Custom display for the `*SerializationModel` / `ArchiveTree` classes.
  These keep pydantic's default repr for now; they are serialization plumbing, not what users hold.
- Changing `__repr__` semantics away from an eval-ish `TypeName(...)` form.
  The form is retained; only its *source* changes (derived from the report rather than hand-written).
- A configurable/pluggable rendering theme system.
  One rich style, one HTML style.

## Design overview

Three pieces:

1. A **`Report`** data structure: a renderer-agnostic tree of a type name, an optional title/summary, an ordered list of fields, and named child sub-reports.
2. A **`Describable`** mixin (backed by a `Describable` protocol) that each user-facing class adopts.
   The class implements `_describe(self, **kwargs) -> Report`; the mixin derives `__repr__`, `__str__`, `_repr_html_`, and `__rich__` from it.
3. A CLI **`describe`** subcommand that deserializes a file and renders the top-level object's report.

`rich` becomes a hard runtime dependency and drives both terminal output and notebook HTML (via rich's console HTML export), so the two stay visually consistent.

### The `Report` model

A new module `python/lsst/images/describe.py` (exported from `lsst.images`) defines:

```python
class FieldRole(enum.Enum):
    ARG = "arg"          # a constructor argument; part of the object's identity
    DERIVED = "derived"  # informational/computed; not needed to reconstruct

@dataclasses.dataclass(frozen=True)
class ReportField:
    label: str                       # "Nominal Pixel scale", "bbox"
    value: Any                       # display value (str/number/Quantity/SkyCoord/...)
    unit: str | None = None          # "arcsec", "pix"; rendered after the value
    repr_value: str | None = None    # eval-ish fragment; defaults to repr(value)
    role: FieldRole = FieldRole.ARG

@dataclasses.dataclass(frozen=True)
class ReportTable:
    title: str | None                # "Mask planes", "Axes", "Corners"
    columns: list[str]               # header row: ["Bit", "Index", "Mask", "Name", "Description"]
    rows: list[list[Any]]            # one list of cell values per row
    role: FieldRole = FieldRole.DERIVED

@dataclasses.dataclass(frozen=True)
class Report:
    type_name: str                   # "SkyProjection", "VisitImage"
    title: str | None = None         # "ICRS coordinates" (KAPPA-style)
    summary: str | None = None       # one-line hint used by __str__
    fields: list[ReportField] = dataclasses.field(default_factory=list)
    tables: list[ReportTable] = dataclasses.field(default_factory=list)
    children: dict[str, Report] = dataclasses.field(default_factory=dict)
```

The design intent:

- One `_describe()` implementation per class is the single source of truth.
  Text, rich-terminal, notebook-HTML, and repr all derive from it.
- `role` separates *what reconstructs the object* (`ARG`) from *what is useful to look at* (`DERIVED`, e.g. corner sky coordinates).
  Only `ARG` fields feed `__repr__`.
- `repr_value` lets a field's eval-ish form differ from its display form, and supports deliberately lossy reprs: a constructor-arg field may set `repr_value="..."` when the real value is huge or unprintable (e.g. an `Image`'s pixel array), preserving today's `Image(..., bbox=..., dtype=...)` behavior.
- `tables` carry homogeneous, columnar data (mask planes, WCS axes, box corners) that renders as a genuine aligned table with a header row, rather than as a long list of near-identical fields or child sub-reports.
  Tables are `DERIVED` by default and never feed `__repr__`; a class that reconstructs from tabular data (e.g. `MaskSchema` from its planes) keeps a separate `ARG` field carrying the eval-ish `repr_value`.

### The `Describable` protocol and mixin

```python
@runtime_checkable
class Describable(Protocol):
    def _describe(self, **kwargs: Any) -> Report: ...

class DescribableMixin:
    def _describe(self, **kwargs: Any) -> Report: ...   # each class overrides

    def describe(self, **kwargs: Any) -> Report:
        return self._describe(**kwargs)

    def __repr__(self) -> str:
        return self._describe().to_repr()

    def __str__(self) -> str:
        return self._describe().to_str()

    def _repr_html_(self) -> str:
        return self._describe()._repr_html_()

    def __rich__(self) -> Any:
        return self._describe().__rich__()
```

`obj.describe(**kwargs)` is the public entry point that returns a `Report`; users can pass parameters (such as a `bbox`) and can call renderers on the returned report directly.

Contract for adopters: `_describe()` must list constructor arguments as `ARG` fields with faithful `repr_value`s.
This is what guarantees `repr` and the rich views never drift.
`_describe()` **with no kwargs must be cheap** — no pixel loads.
Fields that require data (corner sky coordinates) are computed only when the relevant kwargs are supplied or the data is already cheaply available.

### Renderers

The renderers live with the `Report` model and consume a `Report`:

- **`Report.to_repr()`** — walks `fields` where `role == ARG`, emits `TypeName(label=repr_value, ...)`.
  `repr_value` defaults to `repr(value)`.
  Drives `__repr__`.
- **`Report.to_str()`** — a single line: `title`/`summary` plus a few key fields.
  Drives `__str__`.
- **`Report.__rich__()`** — returns a `rich.tree.Tree` whose nodes are aligned `label : value unit` rows (rendered via `rich.table`), with each `ReportTable` rendered as a `rich.table.Table` (header row plus data rows) and `children` as nested branches.
  KAPPA-style layout.
  Drives terminal output in the CLI `describe` command and rich's own display.
- **`Report._repr_html_()`** — produced via `rich`'s console HTML export of the same `__rich__` renderable, so notebook and terminal output stay visually consistent.

`DERIVED` fields and all `tables` appear in `__rich__` and `_repr_html_` (and `tables` optionally in `to_str`), but never in `to_repr`.

## Per-class content

### Container images

`VisitImage`, `DifferenceImage`, `CellCoadd`, `MaskedImage`, `ColorImage`, `Image`, `Mask`.

Each `_describe()` lists its own `ARG` fields (bbox, dtype, unit, band, ...) and nests **children** for its components (`image`, `mask`, `variance`, `sky_projection`, `psf`, `summary_stats`, `backgrounds`, `detector`, ...).
A container threads its **own bbox down** when building the `sky_projection` child, so the WCS sub-report shows corner sky coordinates automatically:

```python
def _describe(self, **kwargs: Any) -> Report:
    children = {"image": self.image._describe(), "mask": self.mask._describe()}
    if self.sky_projection is not None:
        children["sky_projection"] = self.sky_projection._describe(bbox=self.bbox)
    ...
```

### WCS / transforms

`SkyProjection` is one of the two primary tabular cases (see below).
`SkyProjection._describe(bbox=None)` produces the KAPPA-style report, covering all of:

- Frame/domain metadata as `fields` — `title="ICRS coordinates"`, domain, center sky coordinate.
  Works with no bbox.
- A per-axis `ReportTable` titled "Axes" with columns `["Axis", "Label", "Units", "Nominal pixel scale"]`, one row per axis (the KAPPA "Axis 1 / Axis 2" block).
  Works with no bbox.
- A corners `ReportTable` titled "Corners" with columns `["Corner", "RA", "Dec"]`, one row per box corner with RA/Dec in sexagesimal, only when a bbox is available.
- FITS-WCS availability as a `field` — whether a `fits_approximation` exists and whether `as_fits_wcs(bbox)` would succeed (generalized mappings are not always FITS-representable).

`Transform` reports in/out frames and bounds, with the AST `show()` dump available as a `DERIVED` field.
`FrameSet`/`CameraFrameSet` and the five `*Frame` classes (`DetectorFrame`, `FocalPlaneFrame`, `FieldAngleFrame`, `TractFrame`, `GeneralFrame`, plus the `SkyFrame` enum) report frame identity, units, and domain.

### Characterization models

- PSFs (`GaussianPointSpreadFunction`, `PiffWrapper`, `PSFExWrapper`, `LegacyPointSpreadFunction`, `CellPointSpreadFunction`): kind, dimensions, key parameters.
- Fields (`ChebyshevField`, `SplineField`, `ProductField`, `SumField`, `CellField`): basis type, degree/order, bbox.
- `BackgroundMap`: entries and which background is subtracted.
- `ObservationSummaryStats`: grouped stat fields.
- `ApertureCorrectionMap`, `Detector`, `Amplifier`.

### MaskSchema

`MaskSchema` is the other primary tabular case.
`_describe()` emits a single `ReportTable` titled "Mask planes" with columns `["Bit", "Index", "Mask", "Name", "Description"]`, one row per defined plane (reserved `None` bits may be shown or skipped).
`Mask` and `MaskedImage`/`VisitImage` nest this schema report as a child.
Because a `MaskSchema` reconstructs from its planes, it keeps a single `ARG` field whose `repr_value` reproduces today's `MaskSchema([...], dtype=...)` repr; the table itself is `DERIVED`.

### Geometry primitives

`Box`, `Interval`, `Region`, `Polygon`, `XY`, `YX`.
Reproduce existing `__str__`/`__repr__` output via `_describe()`, adding the structured and HTML views.

### Classes gaining a repr for the first time

The plain classes that currently show `<...object at 0x...>` gain real `__repr__`/`__str__` via the mixin: `SkyProjection`, `Transform`, `FrameSet`, `CameraFrameSet`, the PSF wrappers, the `BaseField` subclasses, `Detector`, `BackgroundMap`, and `GeneralizedImage`.

## CLI

Leave `inspect` unchanged: fast, file-layer only, no deserialization.
Add a new `describe` subcommand that deserializes the file and prints the top-level object's report via the rich terminal renderer.
The method verb (`describe`) and the command verb (`describe`) match, and both are clearly distinct from file-layout `inspect`.

Registration follows the existing pattern in `python/lsst/images/cli/_main.py` (`main.add_command(describe)`), with the command defined in a new `python/lsst/images/cli/_describe.py`.

## Dependency change

Add `rich` to the core `dependencies` in `pyproject.toml`.
It drives both terminal output and notebook HTML.

## Behavioral compatibility

For classes with hand-written `__str__`/`__repr__` today (`Image`, `Mask`, `MaskSchema`, `MaskedImage`, `VisitImage`, `DifferenceImage`, `ColorImage`, `Interval`, `Box`, `Region`, `Polygon`, `CellCoadd`, `GaussianPointSpreadFunction`), the derived output must reproduce the current output.
Pin the current `str`/`repr` results in tests **before** migrating each class to `_describe()`, so any change is intentional and reviewed.

## Testing strategy

- Characterization tests capturing current `str`/`repr` for every class that has them, asserted before migration.
- Per-class `_describe()` tests: field presence, roles, and that `to_repr()` output is eval-ish (or carries the documented `...` placeholder).
- `SkyProjection._describe(bbox=...)` tests against a known WCS: the "Axes" and "Corners" tables (column headers, per-row RA/Dec in sexagesimal, pixel scale), center field, and FITS-WCS availability flag; and `bbox=None` omitting the "Corners" table while keeping "Axes".
- `MaskSchema._describe()` tests: the "Mask planes" table rows match the schema, and the retained `ARG` field's `repr_value` still reproduces the current `MaskSchema(...)` repr.
- Cheapness test: `_describe()` with no kwargs does not trigger pixel loads.
- Renderer tests: text, HTML, and rich-tree output for a representative nested container (`VisitImage`), including automatic bbox threading into the `sky_projection` child.
- CLI `describe` test against sample files for the main container types.

## Open questions

None outstanding; naming, scope, rich integration, str/repr policy, lossy-repr handling, CLI shape, and WCS report content are all resolved.

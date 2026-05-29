# Schema versioning design (DM-54557)

## 1. Summary

Add schema URLs and version stamps to the serialized output of `lsst.images`,
on two distinct axes:

- **Per-model (data-model) versioning** — every concrete `ArchiveTree`
  subclass declares a schema URL and a `major.minor.patch` version. Both are
  written into every serialized JSON tree. The reader applies a single
  major-only compatibility check using a per-model `min_read_version`
  integer.
- **Container (file-format) versioning** — the FITS and NDF backends each
  carry a single integer version that bumps when the backend's *layout*
  changes (HDU placement, `NdfDocument` shape), independent of any
  Pydantic model. The JSON backend has no container distinct from the root
  tree, so its data-model version covers everything.

The two axes evolve independently: changing the FITS HDU layout bumps
`FMTVER` without touching any data-model version, and bumping
`CellCoaddSerializationModel` from `1.0.0` to `1.1.0` doesn't touch any
container version.

The compatibility model uses **two version fields** on each model rather
than a single symmetric major comparison, addressing the asymmetric-bump
concern raised in review (see
[`2026-05-28-schema-compat-tradeoffs.md`](./2026-05-28-schema-compat-tradeoffs.md)
for the full trade-off analysis):

- `schema_version` — full `major.minor.patch`, identifies the on-disk shape.
- `min_read_version` — integer (major only); the smallest reader major
  that can safely interpret this file. Acts as the sole gate for
  rejecting "old reader vs new file" combinations. The opposite
  direction ("new reader vs old file") is gated only by whether the
  in-code Pydantic model can validate the older shape; this design adds
  no separate gate there.

These let new code keep reading older files whenever Pydantic and any
backfill we write can interpret them, while still rejecting old code
reading newer files when the writer declares the new shape unsafe for
older readers.

## 2. Scope

**In scope (this ticket).**

- A new mechanism on `ArchiveTree` for declaring per-subclass schema URL,
  version, and minimum reader version, with `schema_version` and
  `min_read_version` written as Pydantic fields and `schema_url` as a
  computed field.
- Read-side major-only compatibility check.
- A container-version stamp in the FITS primary HDU header (`FMTVER`),
  alongside a `DATAMODL` keyword carrying the root tree's full `schema_url`
  for visibility from FITS tooling.
- A container-version stamp in the top-level NDF metadata
  (`.MORE.LSST.FORMAT_VERSION`), alongside `.MORE.LSST.DATA_MODEL` carrying
  the root tree's full `schema_url`.
- Initial `SCHEMA_VERSION = "1.0.0"`, `MIN_READ_VERSION = 1` for every
  existing `ArchiveTree` subclass; initial `FMTVER = 1` for FITS,
  `FORMAT_VERSION = 1` for NDF.
- Absence-of-stamp policy: files written before this change have no stamp;
  the reader treats absence as `1.0.0` / `min_read_version=1` /
  `FMTVER=1` so existing files continue to read.
- Reference v1 JSON fixtures for every top-level `ArchiveTree` subclass,
  plus a minify helper that produces small fixtures from real legacy files.
- Tests covering write+read round-trip, version-mismatch errors, and the
  absence-of-stamp legacy path.

**Out of scope.**

- Deferred-fail substitution: substituting an `ArchiveReadError`-raising
  sentinel for an incompatible *submodel* so the rest of the tree still
  reads (analogous to how PSF deserialization defers failure today). See
  §7 for the design; v1 hard-fails the entire read on any version
  mismatch.
- Migration framework: transforming an older-major tree into the current
  shape so we can keep reading it. When this design says "new code reads
  old file" works, it means in cases where Pydantic itself can validate
  the older shape under the current model (i.e. only optional fields were
  added since). Bumping `min_read_version` is what we do when Pydantic
  *can't* do that — and at that point old files become unreadable until
  someone writes a migration.
- Schema hosting: the URLs follow the
  `https://images.lsst.io/schemas/<name>-<version>` pattern but do not
  have to actually resolve to a fetchable schema document at the time
  this lands.
- Folding `ObservationSummaryStats.version: int` into the new scheme.
  Mechanical follow-up.
- Schema-snapshot tests that diff `model_json_schema()` to catch
  forgotten version bumps.

## 3. Design overview

### 3.1 The two version axes

| Axis | Where it lives | Granularity | When it bumps |
|---|---|---|---|
| Data-model version | `schema_version` + `min_read_version` fields on every `ArchiveTree` subclass; round-trips through JSON. | Per `ArchiveTree` subclass (~20 today). | When the Pydantic shape of *that* subclass changes. |
| Container version | A single integer keyword in the FITS primary HDU header (`FMTVER`); a single integer component in the NDF top-level structure (`.MORE.LSST.FORMAT_VERSION`). Not present in JSON output. | Per backend (one for FITS, one for NDF). | When the container layout itself changes (HDU placement, `NdfDocument` shape, etc.). |

Data-model versions follow `major.minor.patch` with the ASDF interpretation
(see [ASDF versioning conventions][asdf-versioning]):

- **major**: backward-incompatible — the on-disk shape no longer validates
  under the current Pydantic model without help.
- **minor**: backward-compatible additions (new optional field).
- **patch**: changes that don't affect file-format interpretation
  (description tweaks, doc fixes).

Container versions are integer-only; the major-vs-minor distinction
doesn't apply to backend layout changes (a layout change either is or
isn't readable by old code).

[asdf-versioning]: https://www.asdf-format.org/projects/asdf-standard/en/latest/versioning.html

### 3.2 URL scheme

`https://images.lsst.io/schemas/<schema-name>-<major>.<minor>.<patch>`

`<schema-name>` is the lowercase, hyphen-separated form of the
`ArchiveTree` subclass's public name minus the `SerializationModel`
suffix (e.g. `cell_coadd`, `masked_image`, `image`, `mask`,
`gaussian_psf`, `piff_psf`, `chebyshev_field`, …). The exact mapping is
owned by each subclass via its `SCHEMA_NAME` `ClassVar`.

The URL is not required to resolve to a hosted document at landing time.

### 3.3 Why two version fields per model

Recap of [`2026-05-28-schema-compat-tradeoffs.md`](./2026-05-28-schema-compat-tradeoffs.md):
the symmetric ASDF rule (different major → reject in either direction) is
wrong-by-default for our most common breaking changes — adding a required
field, adding a discriminated-union variant. Both want **old code to
reject new files** but **new code to keep reading old files**.

Splitting the version into a `(schema_version, min_read_version)` pair
lets the writer say "here is the shape; here is the smallest reader
major that can safely interpret it". Often these aren't equal: bumping
`schema_version` from `1.0.0` to `2.0.0` because we added a required
field doesn't necessarily bump `min_read_version` — old readers can't
read 2.0.0 files (they'd lose the new field), but new readers can still
read 1.0.0 files (default the new field on input).

`min_read_version` is an integer because, by convention, only the major
component drives compatibility. Encoding minor/patch in it would be
noise.

### 3.4 External (non-`ArchiveTree`) embedded models

Some `ArchiveTree` subclasses embed Pydantic models from outside this
package — currently `astro_metadata_translator.ObservationInfo` (in
`VisitImageSerializationModel.obs_info`) and `ObservationSummaryStats`
(in `VisitImageSerializationModel.summary_stats`, with its own ad-hoc
`version: int` field that predates this design).

These models do not get their own `schema_version`/`min_read_version`
stamp in the JSON. The version-stamp mechanism in this design lives on
`ArchiveTree` and we don't control upstream models. Instead, the
**effective version of an embedded external model is implicitly tied to
the containing tree's `SCHEMA_VERSION`**:

- If an upstream model (e.g. `ObservationInfo`) changes shape in a way
  that breaks validation of older files, the containing tree
  (`VisitImageSerializationModel`) must bump its `SCHEMA_VERSION` to
  express that. The bump may also require `MIN_READ_VERSION` to move,
  using the same criteria as for any other major change.
- If an upstream model adds a backward-compatible optional field, no
  action is required from us — Pydantic accepts both old and new shapes,
  and the existing `SCHEMA_VERSION` continues to apply. (A diligent
  release could bump our minor anyway, for traceability.)
- The on-read failure mode for an unanticipated upstream change is a
  Pydantic validation error inside the embedded model, not a clean
  `ArchiveReadError` from our compatibility check. Callers who care to
  distinguish the two should still treat both as "this file can't be
  read by this release."

This is honest about what version stamping can and can't do for embedded
external models, and it requires no new mechanism. The cost is a
discipline: every `ArchiveTree` that embeds an external model must be
checked when the upstream releases. The set of such embeddings is small
and listed above; expanding it should be a deliberate decision.

A future follow-up could record upstream package versions
(`astro_metadata_translator.__version__`) into the tree's `metadata`
dict at write time for forensics — useful if upstream changes turn out
to drift more often than we'd like, but not part of this ticket.

## 4. Per-model versioning mechanism

### 4.1 Base class additions on `ArchiveTree`

```python
class ArchiveTree(pydantic.BaseModel, ABC, ...):
    SCHEMA_NAME: ClassVar[str]            # e.g. "image"
    SCHEMA_VERSION: ClassVar[str]         # e.g. "1.0.0"
    MIN_READ_VERSION: ClassVar[int]       # e.g. 1

    schema_version: str = pydantic.Field(
        default="",  # filled in by _populate_and_check_schema_version
        description="Data-model schema version of this tree (major.minor.patch).",
    )
    min_read_version: int = pydantic.Field(
        default=0,  # filled in by _populate_and_check_schema_version
        description="Smallest reader major that can interpret this tree.",
    )

    @pydantic.computed_field(description="Canonical schema URL for this tree.")
    @property
    def schema_url(self) -> str:
        cls = type(self)
        return (
            f"https://images.lsst.io/schemas/"
            f"{cls.SCHEMA_NAME}-{cls.SCHEMA_VERSION}"
        )

    @pydantic.model_validator(mode="before")
    @classmethod
    def _populate_and_check_schema_version(cls, data):
        if not isinstance(data, dict):
            return data
        on_disk_version = data.pop("schema_version", "1.0.0")
        on_disk_min_read = data.pop("min_read_version", 1)
        # Strip the computed-field key if present (it's read-only).
        data.pop("schema_url", None)
        _check_compat(
            cls.SCHEMA_NAME, on_disk_version, on_disk_min_read, cls.SCHEMA_VERSION
        )
        # Normalise so re-serializing this instance carries the in-code values.
        data["schema_version"] = cls.SCHEMA_VERSION
        data["min_read_version"] = cls.MIN_READ_VERSION
        return data
```

`SCHEMA_NAME`, `SCHEMA_VERSION`, and `MIN_READ_VERSION` are `ClassVar`s,
so they don't become Pydantic fields. The base class has no value for
them — only concrete subclasses can be instantiated, and each must
declare all three.

The `model_validator(mode="before")` does three jobs in one place:

1. *Check compatibility* of the on-disk version against the in-code
   reader.
2. *Normalise* the field so the instance always carries the in-code
   values (so re-serializing immediately bumps a file from `1.0.0` to
   `1.1.0` if the in-code version has moved up).
3. *Strip* the computed `schema_url` from input data so Pydantic doesn't
   reject it as extra.

### 4.2 The compatibility check

```python
def _check_compat(
    name: str,
    on_disk_version: str,
    on_disk_min_read: int,
    in_code_version: str,
) -> None:
    in_code_major = int(in_code_version.split(".", 1)[0])
    if on_disk_min_read > in_code_major:
        raise ArchiveReadError(
            f"{name}: on-disk schema requires reader major >= "
            f"{on_disk_min_read}; this release is {in_code_version}."
        )
    # Otherwise: silent. min_read_version is the sole gate for
    # "old reader vs new file"; the writer is responsible for setting it
    # truthfully when it bumps schema_version's major.
```

The check looks at `min_read_version` and *not* at `schema_version`'s
major directly. This is deliberate: `min_read_version` is meant to be
the sole gate for "old code reading new file" rejection. A redundant
`on_disk_major > in_code_major` check would re-impose the symmetric
ASDF rule and defeat the asymmetric-bump escape that motivates the
two-field design (e.g. a 2.0.0 file with `min_read_version=1`, written
to be deliberately readable by major-1 code).

The "new code reading old file" direction is *not* gated here at all.
That direction is purely a Pydantic-shape question: if the in-code model
accepts the on-disk shape, the read works; otherwise Pydantic raises its
own validation error downstream. When we make a backward-incompatible
change that we still want new code to handle, we add backfill logic in
the model validator (or a dedicated migration in the future).

There is no warning band for newer-minor in v1 — the substitution
mechanism (§7) is the future home of nuance, and we choose not to
second-guess that here.

### 4.3 Each subclass declares three constants

```python
class ImageSerializationModel[P](ArchiveTree):
    SCHEMA_NAME = "image"
    SCHEMA_VERSION = "1.0.0"
    MIN_READ_VERSION = 1
    data: ArrayReferenceQuantityModel | ... = pydantic.Field(...)
    ...
```

That's the entire per-subclass cost. The ~20 existing `ArchiveTree`
subclasses each get three new lines, no other changes.

A unit test enforces that *every* concrete `ArchiveTree` subclass has all
three constants set, that all `SCHEMA_NAME` values are unique, and that
`MIN_READ_VERSION <= int(SCHEMA_VERSION.split('.')[0])`.

### 4.4 What appears in the JSON

A written tree carries both stamps as ordinary JSON keys plus the
computed URL:

```json
{
  "schema_url": "https://images.lsst.io/schemas/image-1.0.0",
  "schema_version": "1.0.0",
  "min_read_version": 1,
  "data": { ... },
  "start": [0, 0],
  ...
}
```

`schema_url` is emitted because `pydantic.computed_field` participates in
serialization by default. `schema_version` and `min_read_version` are
emitted because they're normal fields. On read, all three are consumed by
the `model_validator`; the values on the instance always reflect the
in-code constants, not the on-disk values.

## 5. Container (file-format) versioning

Container versions bump when the *backend layout* changes — independent
of any Pydantic model. Single integer (major-only) per backend.

### 5.1 FITS

Two new keywords in the primary HDU header, written by `FitsOutputArchive`
next to the existing `INDXADDR`/`INDXSIZE`/`JSONADDR`/`JSONSIZE` keywords
(file `python/lsst/images/fits/_output_archive.py`, current TODO at
line 128):

| Keyword | Type | Value (initial) | Comment |
|---|---|---|---|
| `DATAMODL` | str  | e.g. `"https://images.lsst.io/schemas/image-1.0.0"` | Top-level data model schema URL (= root tree's `schema_url`). |
| `FMTVER`   | int  | `1` | FITS container layout version. |

`DATAMODL` mirrors the root tree's `schema_url` — name and version in one
keyword. It exists for visibility from FITS tooling without parsing the
JSON HDU; the JSON HDU remains the source of truth on read. We chose not
to split `DATAMODL` and `DMVERS` because the URL form already encodes
both and matches the in-tree `schema_url` exactly. (FITS string values
fit ~68 characters, well over the URL length.)

`FitsInputArchive` (`_input_archive.py:114`, also currently a TODO) reads
`FMTVER`, runs `_check_format_version("fits", on_disk_fmtver, _FITS_FORMAT_VERSION)`,
and proceeds. `DATAMODL` is informational only on read; the JSON tree's
fields are what drive the data-model check.

A new module-level constant:

```python
# fits/_output_archive.py and fits/_input_archive.py
_FITS_FORMAT_VERSION = 1
```

Absence of `FMTVER` is treated as `1` so pre-stamp files continue to read.

### 5.2 NDF

Two new components in the top-level NDF structure, alongside the existing
`.MORE.LSST.*` package metadata:

| Component | Type | Value (initial) |
|---|---|---|
| `.MORE.LSST.DATA_MODEL`     | `_CHAR`    | e.g. `"https://images.lsst.io/schemas/image-1.0.0"` |
| `.MORE.LSST.FORMAT_VERSION` | `_INTEGER` | `1` |

`FORMAT_VERSION` is a single integer component, **not** a substructure.
The earlier `.MORE.LSST.FORMAT.{NAME,VERSION}` shape was rejected in
favor of this flatter form during design.

`NdfInputArchive` reads `.MORE.LSST.FORMAT_VERSION` and runs
`_check_format_version("ndf", on_disk, _NDF_FORMAT_VERSION)`. Absence → `1`.

A new module-level constant:

```python
# ndf/_output_archive.py and ndf/_input_archive.py
_NDF_FORMAT_VERSION = 1
```

The container version is owned by the output-archive module, not by
`NdfDocument` itself — it is a property of how the backend serializer
maps trees onto IR + bytes, not a property of the IR object.

### 5.3 JSON

No separate stamp. The root `ArchiveTree`'s `schema_version`,
`min_read_version`, and `schema_url` are sufficient: there is no
container layout distinct from the tree.

### 5.4 The container-version check

A small helper alongside `_check_compat`:

```python
def _check_format_version(name: str, on_disk: int, in_code: int) -> None:
    if on_disk > in_code:
        raise ArchiveReadError(
            f"{name}: on-disk container format version {on_disk} is "
            f"newer than this release ({in_code}); cannot read."
        )
```

Because the file-format version is integer-major-only, lower on-disk
values are always silently accepted (we promise to keep reading older
container layouts when feasible; bumping `FMTVER`/`FORMAT_VERSION` is the
escape hatch for "no, the layout is genuinely incompatible").

## 6. Read-side rule and absence policy

### 6.1 Order of checks

The container check happens *before* JSON parsing (the input archive
reads the FITS header / NDF metadata first). The data-model check happens
*during* JSON parsing inside the `ArchiveTree` model validator. So a
file with an incompatible `FMTVER` is rejected before any tree shape is
interpreted.

### 6.2 Absence is the v1 default

Files written before this change carry none of:

- `schema_version`, `min_read_version`, `schema_url` (per-model);
- `DATAMODL`, `FMTVER` (FITS);
- `.MORE.LSST.DATA_MODEL`, `.MORE.LSST.FORMAT_VERSION` (NDF).

The reader treats absence as the v1 defaults (`schema_version="1.0.0"`,
`min_read_version=1`, `FMTVER=1`, `FORMAT_VERSION=1`) so legacy files
round-trip cleanly. Once a file is re-written by post-stamp code, the
stamps appear.

## 7. Deferred-fail substitution (design only, not implemented in v1)

This section captures the design for letting an incompatible *submodel*
fail at point-of-use rather than reject the whole tree. **It is not
implemented in v1.** The default behavior in v1 is hard-fail: any
version mismatch in any submodel rejects the entire read. We document
the design here so that adding it later is a small change.

### 7.1 Motivation

When reading a `MaskedImage` whose `psf` submodel claims
`min_read_version: 2` and we're on major 1, hard-fail rejects the entire
file even though the image plane is fine. Today the PSF code already
throws `ArchiveReadError` from `deserialize()` when an optional
dependency (e.g. `piff`) is missing — that failure is deferred to
point-of-use. Schema incompatibility could behave the same way.

### 7.2 The `_ReadFailed` substitute

A single generic class:

```python
class _ReadFailed(ArchiveTree):
    """Substitute for an ArchiveTree subclass whose on-disk version is
    incompatible. Validates trivially; raises on use.
    """
    on_disk_data: dict[str, Any]
    reason: str
    original_class: str  # SCHEMA_NAME of the model we replaced

    def deserialize(self, archive, **kwargs):
        raise ArchiveReadError(self.reason)
```

A single class is enough because the only behavior we override is
`deserialize`/`deserialize_component`; we don't need a per-subclass
substitute.

### 7.3 Where substitution happens

Inside `_populate_and_check_schema_version`:

```python
@pydantic.model_validator(mode="before")
@classmethod
def _populate_and_check_schema_version(cls, data, info):
    ...
    try:
        _check_compat(...)
    except ArchiveReadError as exc:
        if _deferred_failures_enabled(info.context):
            return _ReadFailed.placeholder_dict(cls, on_disk_data=data, reason=str(exc))
        raise
    ...
```

The `info.context` is Pydantic's per-validation context dict, set by the
input-archive layer when the caller passes
`defer_schema_failures=True`.

### 7.4 Caller-facing API (future)

```python
def read(..., defer_schema_failures: bool = False) -> ArchiveTree: ...
```

The flag flows through the input archive into `info.context`. Default
remains `False` even after this is implemented.

### 7.5 Known limitation: unknown discriminated-union variants

Pydantic discriminator validation runs *before* per-subclass model
validators, so an *unknown* variant tag is detected before our
substitution path runs. Two cases:

1. *Known variant, version mismatch* — variant resolves, our validator
   runs, substitution applies. Works.
2. *Unknown variant* — Pydantic raises before we see the data;
   substitution requires intercepting at the union level.

Case 2 is out of scope even when deferred-fail lands. Documenting it
here so future implementers know not to be surprised.

### 7.6 Testing the design retroactively

If/when we add deferred-fail, we can test it without producing real
incompatible files: hand-craft fixtures whose `min_read_version` or
`schema_version` is artificially set to an incompatible value, then read
them with `defer_schema_failures=True` and assert the resulting tree
contains `_ReadFailed` instances at the right places.

## 8. JSON Schema interaction

The related spec
`docs/superpowers/specs/2026-05-15-schema-validation.md` describes the
existing `model_json_schema()` situation. The changes in this design
interact with it as follows:

- `schema_version` and `min_read_version` are real Pydantic fields, so
  they show up in `model_json_schema()` output as `string` and `integer`
  properties — visible to external tools.
- `schema_url` is a `pydantic.computed_field`, which Pydantic includes
  in `model_json_schema(mode="serialization")` but excludes from
  `mode="validation"`. That's the right behavior: the URL is
  informational on output, never accepted on input.
- The `$id` and `title` of the generated JSON Schema gain natural
  values: `$id` is set from `schema_url`, `title` from `SCHEMA_NAME`.
  This is injected automatically by overriding
  `__pydantic_init_subclass__` on `ArchiveTree`, so each concrete
  subclass's `model_config.json_schema_extra` is populated from its
  `SCHEMA_NAME`/`SCHEMA_VERSION` `ClassVar`s at class-creation time. No
  per-subclass `model_config` boilerplate is needed.

This makes a generated JSON Schema document fully self-identifying.

## 9. Initial version assignments

Every existing `ArchiveTree` subclass and every existing backend ships
with the v1 defaults when this lands:

- `SCHEMA_VERSION = "1.0.0"`
- `MIN_READ_VERSION = 1`
- `_FITS_FORMAT_VERSION = 1`, `_NDF_FORMAT_VERSION = 1`

There is no attempt to retroactively distinguish between "version of the
shape that existed before stamping" and "version of the shape now" —
the absence policy means pre-existing files are *defined* to be 1.0.0
and post-landing files explicitly carry 1.0.0.

The first time we make a backward-compatible addition to any subclass
afterwards we bump that subclass to `1.1.0` (and `MIN_READ_VERSION`
stays at 1). The first backward-incompatible change bumps `SCHEMA_VERSION`
to `2.0.0`. Whether to also bump `MIN_READ_VERSION` is a separate
decision, driven *only* by "does the new shape mislead an old reader?":

- Adding a required field with no sensible default → bump
  `SCHEMA_VERSION` to `2.0.0`. Old readers (which use Pydantic's default
  `extra="ignore"`) silently drop the new field. Whether that's
  acceptable depends on the field. If silent loss is dangerous, bump
  `MIN_READ_VERSION = 2`. If old readers can safely ignore it, leave
  `MIN_READ_VERSION = 1`. Independently, the new reader needs backfill
  logic to make sense of old files that lack the field.
- Adding a new discriminated-union variant → bump `SCHEMA_VERSION` to
  `2.0.0`. Old readers don't recognise the new tag and will fail loudly
  on files that contain it — but most files won't. Per-class
  `MIN_READ_VERSION = 1` plus per-instance `min_read_version = 2` on
  files that actually use the new variant gives us "old readers reject
  only the affected files." The mechanism for setting per-instance
  values is in place (it's a normal field); the convention for *when*
  is an open question (§13).
- Renaming a field, retyping a field, changing semantics → bump both
  `SCHEMA_VERSION` and `MIN_READ_VERSION` to `2`. Old files become
  unreadable by new code until someone writes a migration; old code
  rejects new files via the `min_read_version` gate.

## 10. Implementation surface

Files touched:

**`python/lsst/images/serialization/_common.py`**
- Add to `ArchiveTree`:
  - `SCHEMA_NAME`, `SCHEMA_VERSION`, `MIN_READ_VERSION` `ClassVar`s.
  - `schema_version`, `min_read_version` fields.
  - `schema_url` computed field.
  - `_populate_and_check_schema_version` model-validator.
  - `__pydantic_init_subclass__` that injects `$id`/`title` into
    `model_config.json_schema_extra`.
- Add `_check_compat` and `_check_format_version` helpers.
- `ArchiveReadError` already exists; reused for hard rejections.

**Each `ArchiveTree` subclass** (~20 files): three `ClassVar` lines
(`SCHEMA_NAME`, `SCHEMA_VERSION = "1.0.0"`, `MIN_READ_VERSION = 1`):

- `python/lsst/images/_image.py`, `_masked_image.py`, `_mask.py`,
  `_visit_image.py`, `_color_image.py`, `_backgrounds.py`,
  `_generalized_image.py`
- `python/lsst/images/cells/_provenance.py`, `cells/_psf.py`
- `python/lsst/images/psfs/_gaussian.py`, `psfs/_legacy.py`,
  `psfs/_piff.py`
- `python/lsst/images/_transforms/_camera_frame_set.py`,
  `_transforms/_projection.py`, `_transforms/_transform.py`
- `python/lsst/images/fields/_chebyshev.py`, `fields/_spline.py`,
  `fields/_sum.py`, `fields/_product.py`
- `python/lsst/images/cameras.py` (`DetectorSerializationModel`)
- `python/lsst/images/aperture_corrections.py`

(The CellCoadd serialization model in `cells/_coadd.py` is included
implicitly — confirmed during implementation.)

**FITS container stamp:**
- `python/lsst/images/fits/_output_archive.py`: write `DATAMODL`,
  `FMTVER`; replace TODO at line 128. New module-level
  `_FITS_FORMAT_VERSION = 1`.
- `python/lsst/images/fits/_input_archive.py`: read `FMTVER`, run check;
  replace TODO at line 114.

**NDF container stamp:**
- `python/lsst/images/ndf/_output_archive.py`: write
  `.MORE.LSST.DATA_MODEL` and `.MORE.LSST.FORMAT_VERSION`. New
  module-level `_NDF_FORMAT_VERSION = 1`.
- `python/lsst/images/ndf/_input_archive.py`: read
  `.MORE.LSST.FORMAT_VERSION`, run check.

**JSON backend:** no changes. The root tree carries everything.

**Helpers:**
- `python/lsst/images/tests/_make_schema_fixtures.py` — generates
  synthetic v1 JSON fixtures from in-memory test factories.
- `python/lsst/images/tests/_minify_for_fixtures.py` — minifies real
  legacy files to small JSON fixtures. CellCoadd subsetting flagged as
  TODO (see §11).

**Tests:** see §11.

**Changelog:** new fragment under `doc/changes/`.

## 11. Testing

### 11.1 Reference v1 JSON fixtures

A new directory `tests/data/schema_v1/` holds reference JSON fixtures
for every concrete `ArchiveTree` subclass:

```
tests/data/schema_v1/
    image.json
    masked_image.json
    visit_image.json
    cell_coadd.json
    coadd_provenance.json
    cell_psf.json
    gaussian_psf.json
    piff_psf.json
    psfex_psf.json
    chebyshev_field.json
    spline_field.json
    sum_field.json
    product_field.json
    camera_frame_set.json
    projection.json
    transform.json
    detector.json
    aperture_correction_map.json
    background_map.json
    color_image.json
    mask.json
    legacy/
        <legacy-derived fixtures>
    README.md
```

Files in `tests/data/schema_v1/` are produced by the
`_make_schema_fixtures.py` helper from in-memory test factories;
files in `tests/data/schema_v1/legacy/` are produced by the
`_minify_for_fixtures.py` helper from real on-disk files (see §11.3).

Both kinds are pretty-printed (`json.dumps(..., indent=2,
sort_keys=False)`) so diffs in git review are readable. The
README records source paths and helper rev for each legacy fixture so
they can be regenerated.

### 11.2 Test modules

- `tests/test_schema_v1_fixtures.py` — for every fixture file:
  - Loads via `JsonInputArchive`; asserts no warnings, no errors.
  - Round-trips: re-serialises, asserts JSON equality using existing
    helpers.
  - Asserts root `schema_url`, `schema_version`, `min_read_version` are
    present and `schema_url` matches `https://images.lsst.io/schemas/<name>-<version>`.
  - For nested submodels, asserts the same keys appear there too.
- `tests/test_schema_versioning.py`:
  - Class invariants: every concrete `ArchiveTree` subclass declares
    `SCHEMA_NAME`, `SCHEMA_VERSION`, `MIN_READ_VERSION`; all
    `SCHEMA_NAME`s unique; `MIN_READ_VERSION <= int(SCHEMA_VERSION.split('.')[0])`.
  - Mutation tests: take a fixture, set `min_read_version` to a value
    above the reader's major → expect `ArchiveReadError`. Set
    `schema_version` to a higher major while keeping `min_read_version`
    at the reader's major → expect *successful* read (the asymmetric
    escape; documents that the writer's `min_read_version` declaration
    is what gates rejection, not the major comparison itself). Strip
    `schema_version`/`min_read_version` entirely → expect successful
    read (legacy default).
  - Negative test: confirm `schema_url` cannot be set from input JSON
    (it's a computed field; supplying it must not affect the parsed
    instance).
- `tests/test_fits_format_version.py`:
  - Write+read with `FMTVER = 1`, `DATAMODL = <root url>`.
  - Manually-corrupted file with `FMTVER = 2` → `ArchiveReadError`.
  - File with `FMTVER` keyword absent → reads (legacy default).
- `tests/test_ndf_format_version.py`:
  - Same matrix for NDF: write+read, mismatched `FORMAT_VERSION`,
    absent component.

### 11.3 The minify helper

`python/lsst/images/tests/_minify_for_fixtures.py` exposes:

```python
def minify(in_path: str, out_path: str, **kwargs) -> None: ...
```

It reads a real on-disk file via the appropriate input archive, picks a
small subset of the in-memory object, and writes JSON via
`JsonOutputArchive`. Per top-level type:

| Type | Minify rule |
|---|---|
| `Image`, `Mask`, `MaskedImage` | Crop to ~16×16 pixels (existing test bbox). |
| `VisitImage` | Crop image, keep PSF stamp_size unchanged but drop secondary metadata. |
| `ColorImage` | Crop all bands. |
| `CellCoadd` | **TODO**: pick a 2×2 contiguous block of inner cells (≥4 cells; otherwise it isn't really a CellCoadd). The outer-ring problem (PSFs/inputs that overlap the kept inner cells must also be retained) is not yet solved. A candidate fallback approach is to *morph* the cells into smaller versions in place — not an accurate subset but sufficient for testing. The implementer will pick one. |
| `Detector`, `CameraFrameSet` | Keep one detector / one frame-set, drop siblings. |
| `BackgroundMap`, `ApertureCorrectionMap` | Keep a single field/region. |
| `*PSF`, `*Field`, `*Transform` | Already small; copy through. |

Existing JSON round-trip tests (`tests/test_*.py`) need their fixtures
updated to include the new stamps where they compare bytes; round-trip
equality tests are unaffected.

## 12. Risks and considerations

- **Forgetting to bump.** If a `SerializationModel` shape changes but no
  one bumps `SCHEMA_VERSION`, files written by the new code will claim
  to be the old version and old readers will silently accept them —
  possibly failing on a now-required field. Mitigation: the unit test
  in §11.2 enforces that constants exist, but not that they were
  bumped. A changelog discipline plus the existing review process is
  the practical guard. (A schema-snapshot test that dumps
  `model_json_schema()` for each tree and fails on diff would be a
  stronger guard; out of scope here.)
- **`pydantic.computed_field` and `schema_url`.** The base class strips
  `schema_url` from input dicts (it's a computed field, can't be set).
  The `model_validator(mode="before")` shown above does this.
- **Field ordering in JSON output.** Pydantic by default writes fields
  in declaration order. We want `schema_url`, `schema_version`,
  `min_read_version` near the top of every JSON tree so a human reader
  sees them first. Declaring them on `ArchiveTree` (the base) puts
  them ahead of every subclass's fields automatically.
- **Container check happens before data-model check.** A file with an
  incompatible `FMTVER` is rejected before any JSON parsing happens, so
  we never try to interpret bytes whose layout we don't understand.
- **`min_read_version` discipline.** Bumping `SCHEMA_VERSION` is
  mechanical; deciding the right `MIN_READ_VERSION` requires thought.
  The convention is: if you wrote a backfill in the new reader so it
  can still load older shapes, leave `MIN_READ_VERSION` alone. If you
  didn't, bump it. A future ticket may add a CI check that flags
  `SCHEMA_VERSION` bumps without an explicit `MIN_READ_VERSION`
  decision.
- **CellCoadd minify.** Flagged TODO — decide between subsetting and
  morphing during implementation.

## 13. Open questions deferred to implementation

- **Per-instance `min_read_version` for union variants.** Today
  `min_read_version` is a `ClassVar`-derived default. For the
  union-variant case, we'd want some files to have a higher
  `min_read_version` than others (only files actually containing the
  new variant). The field is already a normal Pydantic field, so the
  *mechanism* is in place; we just need a convention for setting it
  per-instance during write. The first time we add a union variant
  this becomes concrete.
- **Stamping `lsst.images.__version__` in the FITS primary header for
  forensics.** Useful but not strictly part of schema versioning; can
  be added independently.
- **Folding `ObservationSummaryStats.version: int` into the new
  scheme.** Mechanical follow-up.

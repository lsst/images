# Schema versioning design (DM-54557)

## 1. Summary

Add schema URLs and version stamps to the serialized output of `lsst.images`, on
two distinct axes:

- **Per-model (data-model) versioning** — every concrete `ArchiveTree`
  subclass declares a schema URL and a `major.minor.patch` version. Both are
  written into every serialized JSON tree and validated on read using an
  ASDF-style compatibility rule.
- **Container (file-format) versioning** — the FITS and NDF backends additionally
  carry a *container-layout* version, separate from any data model. The JSON
  backend has no container distinct from the root tree, so its data-model
  version covers everything.

The two axes evolve independently: changing the JSON tree layout of the FITS
backend (e.g. moving the index HDU) bumps the FITS container version without
touching any data-model version, and bumping `CellCoaddSerializationModel`
from 1.0.0 to 1.1.0 doesn't touch any container version.

## 2. Scope

**In scope.**

- A new mechanism on `ArchiveTree` for declaring per-subclass schema URL and
  version, with the version written as a Pydantic field and the URL as a
  computed field.
- ASDF-style read-side compatibility rule (same major → read silently; newer
  minor on disk → warn; newer/older major → raise).
- A container-version stamp in the FITS primary HDU header.
- A container-version stamp in the top-level NDF metadata, in a location
  consistent with how the NDF backend already carries `.MORE.LSST` metadata.
- Initial `SCHEMA_VERSION = "1.0.0"` for every existing `ArchiveTree`
  subclass; initial container versions `1.0.0` for FITS and NDF.
- A policy for *absence* of the new fields/keywords: files written before this
  change has no stamp, and the reader must treat absence as `1.0.0` so existing
  files continue to read.
- Tests covering write+read round-trip, version-mismatch errors, and
  the absence-of-stamp legacy path.

**Out of scope.**

- Any *migration* framework that transforms an older-version tree into the
  current shape. When we have to break compatibility, we bump the major and
  pre-existing files become read-only with older releases. Migration is a
  separate future ticket.
- Schema hosting: the URLs follow the `https://pipelines.lsst.io/schemas/images/`
  pattern but do not have to actually resolve to a fetchable schema document
  at the time this lands. (We can host them later by dumping
  `model_json_schema(mode="serialization")` per the related spec
  `2026-05-15-schema-validation.md`.)
- Changes to `ObservationSummaryStats.version`, which is its own ad-hoc
  per-model `int` and predates this ticket. It can be folded into the new
  scheme as a follow-up.

## 3. Design overview

### 3.1 The two version axes

| Axis | Where it lives | Granularity | When it bumps |
|---|---|---|---|
| Data-model version | A `schema_version` field on every `ArchiveTree` subclass; round-trips through JSON. | Per `ArchiveTree` subclass (~19 today). | When the Pydantic shape of *that* subclass changes. |
| Container version | A keyword in the FITS primary HDU header; a metadata component in the NDF top-level structure. Not present in JSON output. | Per backend (one for FITS, one for NDF). | When the container layout itself changes (index/JSON HDU placement, NDF component layout, etc.). |

Both axes follow `major.minor.patch` with the ASDF interpretation
(see [ASDF versioning conventions][asdf-versioning]):

- **major**: backward-incompatible — older readers cannot interpret the file.
- **minor**: backward-compatible additions (new optional field, new keyword).
- **patch**: changes that don't affect file-format interpretation (doc fixes,
  description tweaks).

[asdf-versioning]: https://www.asdf-format.org/projects/asdf-standard/en/latest/versioning.html

### 3.2 URL scheme

`https://pipelines.lsst.io/schemas/images/<schema-name>-<major>.<minor>.<patch>`

`<schema-name>` is the lowercase, hyphen-separated form of the `ArchiveTree`
subclass's public name minus the `SerializationModel` suffix
(e.g. `cell_coadd`, `masked_image`, `image`, `mask`, `gaussian_psf`,
`piff_psf`, `chebyshev_field`, ...). The exact mapping is owned by each
subclass via its `SCHEMA_NAME` `ClassVar`.

Container versions are identified by a short string pair (a `FMTNAME` like
`lsst.images.fits` and an `FMTVER` like `1.0.0`) rather than a URL — FITS
primary-header keywords have an 8-char name and a ~68-char string-value
limit, and a short identifier is more natural in that space than a URL.

The per-model URL is not required to resolve to a hosted document at
landing time.

## 4. Per-model versioning mechanism

### 4.1 Base class additions on `ArchiveTree`

```python
class ArchiveTree(pydantic.BaseModel, ABC, ...):
    SCHEMA_NAME: ClassVar[str]      # e.g. "image"
    SCHEMA_VERSION: ClassVar[str]   # e.g. "1.0.0"

    schema_version: str = pydantic.Field(
        default="",  # filled in by _populate_schema_version below
        description="Data-model schema version of this tree (major.minor.patch).",
    )

    @pydantic.computed_field(description="Canonical schema URL for this tree.")
    @property
    def schema_url(self) -> str:
        cls = type(self)
        return (
            f"https://pipelines.lsst.io/schemas/images/"
            f"{cls.SCHEMA_NAME}-{cls.SCHEMA_VERSION}"
        )

    @pydantic.model_validator(mode="before")
    @classmethod
    def _populate_and_check_schema_version(cls, data):
        # Accept dicts (JSON) and existing model instances (re-validation).
        if not isinstance(data, dict):
            return data
        on_disk = data.pop("schema_version", None)
        # Strip the computed-field key if present (it's read-only).
        data.pop("schema_url", None)
        if on_disk is None:
            on_disk = "1.0.0"   # absence policy; pre-versioning files
        _check_compat(cls.SCHEMA_NAME, on_disk, cls.SCHEMA_VERSION)
        data["schema_version"] = cls.SCHEMA_VERSION
        return data
```

`SCHEMA_NAME` and `SCHEMA_VERSION` are `ClassVar`s, so they don't become
Pydantic fields. The base class has no value for them — only concrete
subclasses can be instantiated, and each must declare both.

The `model_validator(mode="before")` does three jobs in one place:

1. *Check compatibility* of the on-disk version against the in-code version.
2. *Normalise* the field so the instance always carries the in-code version
   (so re-serializing immediately bumps a file from `1.0.0` to `1.1.0` if the
   in-code version has moved up).
3. *Strip* the computed `schema_url` from input data so Pydantic doesn't
   reject it as extra.

### 4.2 Each subclass declares two constants

```python
class ImageSerializationModel[P](ArchiveTree):
    SCHEMA_NAME = "image"
    SCHEMA_VERSION = "1.0.0"
    data: ArrayReferenceQuantityModel | ... = pydantic.Field(...)
    ...
```

That's the entire per-subclass cost. The 19 existing `ArchiveTree` subclasses
each get two new lines, no other changes.

A unit test enforces that *every* concrete `ArchiveTree` subclass has both
constants set and that all `SCHEMA_NAME` values are unique.

### 4.3 What appears in the JSON

A written tree carries both stamps as ordinary JSON keys:

```json
{
  "schema_url": "https://pipelines.lsst.io/schemas/images/image-1.0.0",
  "schema_version": "1.0.0",
  "data": { ... },
  "start": [0, 0],
  ...
}
```

`schema_url` is emitted because `pydantic.computed_field` participates in
serialization by default. `schema_version` is emitted because it's a normal
field. On read, both are consumed by the `model_validator`; the
`schema_version` field on the instance always reflects the in-code value,
not the on-disk value.

## 5. Container (file-format) versioning

### 5.1 FITS

Two new keywords in the primary HDU header, written by `FitsOutputArchive`
next to the existing `INDXADDR`/`INDXSIZE`/`JSONADDR`/`JSONSIZE` keywords
(file `python/lsst/images/fits/_output_archive.py`, current TODO at line 128):

| Keyword | Value | Comment |
|---|---|---|
| `FMTNAME` | `lsst.images.fits` | Container format identifier. |
| `FMTVER`  | `1.0.0` | Container format version (major.minor.patch). |

The FITS primary header is the natural home: FITS readers see it
immediately, it precedes the JSON tree HDU, and the user-controlled opaque
primary header is layered on top of it.

`FitsInputArchive` (`_input_archive.py:114`, also currently a TODO) reads
`FMTNAME`/`FMTVER`, runs `_check_compat("fits", on_disk, "1.0.0")`, strips
both keywords, and proceeds.

Absence of `FMTNAME`/`FMTVER` is treated as `lsst.images.fits` /  `1.0.0` so
pre-stamp files continue to read.

### 5.2 NDF

Add a top-level structure `.MORE.LSST.FORMAT` containing two `_CHAR`
components, `NAME` and `VERSION`, with values `lsst.images.ndf` and `1.0.0`.
The NDF backend already writes an `.MORE.LSST` extension with
package-specific metadata; `.FORMAT` sits alongside that data, at the top
of the NDF tree, not buried inside any data model.

`NdfInputArchive` reads `.MORE.LSST.FORMAT.NAME` and `.FORMAT.VERSION` on
open and runs `_check_compat("ndf", on_disk, "1.0.0")`. Absence → `1.0.0`.

### 5.3 JSON

No separate stamp. The root `ArchiveTree`'s `schema_version` and
`schema_url` are sufficient: there is no container layout distinct from the
tree.

## 6. Read-side compatibility rule

A single helper `_check_compat(name: str, on_disk: str, in_code: str)`
implements the ASDF rule and is called by both the per-model `model_validator`
and the per-backend container check:

```python
def _check_compat(name: str, on_disk: str, in_code: str) -> None:
    d_maj, d_min, d_pat = _parse(on_disk)
    c_maj, c_min, c_pat = _parse(in_code)
    if d_maj != c_maj:
        raise ArchiveReadError(
            f"{name}: on-disk schema version {on_disk} is incompatible with "
            f"this release ({in_code}); major version differs."
        )
    if (d_min, d_pat) > (c_min, c_pat):
        warnings.warn(
            f"{name}: on-disk schema version {on_disk} is newer than this "
            f"release ({in_code}); reading on a best-effort basis.",
            ArchiveVersionWarning,
            stacklevel=2,
        )
    # same major, on-disk minor/patch <= in-code: silent.
```

`ArchiveVersionWarning` is a new subclass of `UserWarning` exported from
`lsst.images.serialization`, so callers can promote it to an error with
`warnings.filterwarnings("error", category=ArchiveVersionWarning)`.

`ArchiveReadError` is the existing exception in `serialization/_common.py`;
no new exception type is needed for hard rejections.

### 6.1 Older-major files

Same code path as newer-major: `ArchiveReadError`. Since this design has no
migration framework, "I can't read this" is the only honest response.

### 6.2 Absence is 1.0.0

Files written before this change carry neither `schema_version` nor
`schema_url` (per-model), and neither `FMTNAME`/`FMTVER` (FITS) nor
`.MORE.LSST.FORMAT` (NDF). The reader treats absence as `1.0.0` everywhere
so legacy files round-trip cleanly. Once a file is re-written by post-stamp
code, the stamps appear.

## 7. JSON Schema interaction

The related spec `docs/superpowers/specs/2026-05-15-schema-validation.md`
describes the existing `model_json_schema()` situation. The changes in this
design interact with it as follows:

- `schema_version` is a real Pydantic field, so it shows up in
  `model_json_schema()` output as a `string` property — visible to external
  tools.
- `schema_url` is a `pydantic.computed_field`, which Pydantic includes in
  `model_json_schema(mode="serialization")` but excludes from
  `mode="validation"`. That's the right behavior: the URL is informational
  on output, never accepted on input.
- The `$id` and `title` of the generated JSON Schema gain natural values:
  `$id` is set from `schema_url`, `title` from `SCHEMA_NAME`. This is
  injected automatically by overriding `__pydantic_init_subclass__` on
  `ArchiveTree`, so each concrete subclass's `model_config.json_schema_extra`
  is populated from its `SCHEMA_NAME`/`SCHEMA_VERSION` `ClassVar`s at class
  creation time. No per-subclass `model_config` boilerplate is needed.

This makes a generated JSON Schema document fully self-identifying.

## 8. Initial version assignments

Every existing `ArchiveTree` subclass and every existing backend ships
with `SCHEMA_VERSION = "1.0.0"` (or `FMTVER = "1.0.0"`) when this lands.
There is no attempt to retroactively distinguish between "version of the
shape that existed before stamping" and "version of the shape now" — the
"absence is 1.0.0" rule means pre-existing files are *defined* to be
1.0.0, and post-landing files explicitly carry 1.0.0.

The first time we make a backward-compatible addition to any subclass
afterwards we bump that subclass to `1.1.0`. The first backward-incompatible
change bumps to `2.0.0` and the corresponding pre-existing files become
unreadable with that release of `lsst.images`.

## 9. Implementation surface

Files touched, by area:

- `python/lsst/images/serialization/_common.py`
  - Add `SCHEMA_NAME`/`SCHEMA_VERSION` `ClassVar`s, `schema_version` field,
    `schema_url` computed field, `_populate_and_check_schema_version`
    model-validator, `ArchiveVersionWarning`, `_check_compat`, `_parse` to
    `ArchiveTree`.
- Every `ArchiveTree` subclass (~19 files): add two `ClassVar` lines:
  - `python/lsst/images/_image.py`, `_masked_image.py`, `_mask.py`,
    `_visit_image.py`, `_color_image.py`
  - `python/lsst/images/cells/_coadd.py`, `cells/_provenance.py`,
    `cells/_psf.py`
  - `python/lsst/images/psfs/_gaussian.py`, `psfs/_legacy.py`, `psfs/_piff.py`
  - `python/lsst/images/_transforms/_camera_frame_set.py`,
    `_transforms/_projection.py`, `_transforms/_transform.py`
  - `python/lsst/images/fields/_chebyshev.py`, `fields/_spline.py`,
    `fields/_sum.py`, `fields/_product.py`
  - `python/lsst/images/cameras.py`
  - `python/lsst/images/aperture_corrections.py`
- FITS container stamp:
  - `python/lsst/images/fits/_output_archive.py` (write `FMTNAME`/`FMTVER`,
    remove TODO at line 128)
  - `python/lsst/images/fits/_input_archive.py` (read+strip+check, remove
    TODO at line 114)
- NDF container stamp:
  - `python/lsst/images/ndf/_output_archive.py` (write
    `.MORE.LSST.FORMAT.{NAME,VERSION}`)
  - `python/lsst/images/ndf/_input_archive.py` (read+check)
- Changelog fragment in `doc/changes/`.

## 10. Testing

- A new test module `tests/test_schema_versioning.py` that, for every
  concrete `ArchiveTree` subclass:
  - asserts `SCHEMA_NAME` and `SCHEMA_VERSION` are set and well-formed;
  - asserts `SCHEMA_NAME` values are unique across the package;
  - round-trips a minimal instance via JSON and checks the on-disk JSON
    contains the expected `schema_url` and `schema_version`;
  - mutates the on-disk JSON to set a newer minor (warns), a newer major
    (raises), an older major (raises), and an absent key (silently
    `1.0.0`), and checks each behavior.
- An additional test in `tests/test_fits_input_archive.py` (or a new
  `test_fits_format_version.py`) covering the FITS container-version stamp:
  write+read, mismatched `FMTVER`, absent keywords.
- The same set of cases for NDF in `tests/test_ndf_input_archive.py` (or a
  new module).
- A negative test that confirms `schema_url` cannot be set from input JSON
  (it's a computed field; supplying it must not affect the parsed instance).
- The existing `tests/test_*.py` JSON round-trip tests need their fixtures
  updated to include the new stamps in expected output where they compare
  bytes. Where they only check round-trip equality this is automatic.

## 11. Migration to the new scheme

A separate ticket (not this one) will fold `ObservationSummaryStats.version`
into the new mechanism. The existing `int` field is incompatible with the
`major.minor.patch` string and predates this design; consolidating it is
mechanical but touches more files than belong in this PR.

A future migration framework — read older-major files by transforming the
tree before validation — is also a separate ticket. The design above is
deliberately compatible with adding such a framework later: the
`_check_compat` helper is the single chokepoint, and the version field is
already exposed on every tree.

## 12. Risks and considerations

- **Forgetting to bump.** If a `SerializationModel` shape changes but no
  one bumps `SCHEMA_VERSION`, files written by the new code will claim to
  be the old version and older readers will silently accept them — possibly
  failing on a now-required field. Mitigation: the unit test above
  enforces that constants exist, but not that they were bumped. A
  changelog discipline plus the existing review process is the practical
  guard. (A schema-snapshot test that dumps `model_json_schema()` for each
  tree and fails on diff would be a stronger guard; out of scope here but
  feasible.)
- **`pydantic.computed_field` and `schema_url`.** The base class has to
  strip `schema_url` from input dicts (it's a computed field, can't be
  set). The `model_validator(mode="before")` shown above does this.
- **Field ordering in JSON output.** Pydantic by default writes fields in
  declaration order. We want `schema_url` and `schema_version` near the
  top of every JSON tree so a human reader sees them first. Declaring them
  on `ArchiveTree` (the base) puts them ahead of every subclass's fields
  automatically.
- **The container check happens before the data-model check.** A file with
  an incompatible `FMTVER` is rejected before any JSON parsing happens, so
  we never try to interpret bytes whose layout we don't understand.

## 13. Open questions deferred to implementation

- Whether `_check_compat` should accept a `strict_minor: bool` flag for
  testing or callers that want to error on newer-minor. Likely yes but
  trivial to add.
- Whether to also stamp a Python-package version (`lsst.images.__version__`)
  in the FITS primary header for forensics. Useful but not strictly part of
  schema versioning; can be added independently.

# ASDF I/O backend design

## 1. Summary

Add an ASDF file-format backend to `lsst.images`, alongside the existing FITS,
JSON, and NDF backends. The backend produces real `.asdf` files that any
installation of the standard `asdf` Python library can open natively, with
binary blocks for image arrays, full lazy / component reads, and standard
ASDF tags for as many primitives as possible (ndarray, quantity, unit, time,
table column, WCS). LSST-specific types (`Image`, `Mask`, `MaskedImage`)
get new `pipelines.lsst.io`-namespaced tags and schemas, plus thin
`asdf.extension.Converter` shims that materialize them into real Python
objects when `lsst-images[asdf]` is installed.

The MVP scope is `Image`, `Mask`, and `MaskedImage`; everything else
(`VisitImage`, `CellCoadd`, all PSF variants, fields, geometry, cameras)
follows as one schema + one converter per type in follow-up tickets.

## 2. Scope

**In scope.**

- New `lsst.images.asdf` subpackage with `AsdfOutputArchive` /
  `AsdfInputArchive` matching the `OutputArchive` / `InputArchive`
  interfaces in `lsst.images.serialization`, plus module-level
  `write()` / `read()` functions.
- Optional `asdf` (≥ 4.0) and `asdf-astropy` (≥ 0.7) dependencies, gated
  the same way `h5py` is gated for the NDF backend.
- A new `.asdf` row in the unified butler formatter's `_BACKENDS` table,
  including support for the `format="asdf"` write parameter and
  `.asdf`-suffixed URIs at read time.
- New asdf tag URIs and schema documents for `Image`, `Mask`, and
  `MaskedImage`, registered to the `asdf` library via `pyproject.toml`
  entry points (`asdf.extensions` and `asdf.resource_mappings`).
- `Converter` shims for the three in-scope tags so that
  `asdf.open("foo.asdf")` (without going through `lsst.images.asdf.read`)
  returns real `Image` / `Mask` / `MaskedImage` instances when
  `lsst-images[asdf]` is installed.
- FrameSet / WCS round-trip via `starlink.Ast.YamlChan`, which emits
  standard `asdf-wcs-schemas` tags — no LSST or astshim converter needed.
- Lazy reads via `asdf.open(lazy_load=True)` and component reads
  (`partial=True`) through the existing butler component path.
- Round-trip tests, schema/pydantic agreement tests, lazy-read tests,
  interop tests that open files via bare `asdf.open()`, and formatter
  dispatch tests, all parameterized like the existing FITS/JSON/NDF
  tests.
- Documentation under `doc/lsst.images/asdf.rst` describing the file
  layout, interop expectations, and current limitations.
- A towncrier `feature` fragment for the change.

**Out of scope.**

- ASDF support for any type other than `Image`, `Mask`, and `MaskedImage`.
  Each follow-up type (`VisitImage`, `CellCoadd`, PSF variants,
  `ChebyshevField` / `SplineField` / `SumField` / `ProductField`, `Polygon`,
  `Box`, `Interval`, `Projection`, `CameraFrameSet`, `Transform`,
  `Camera`, `ObservationSummaryStats`, `ApertureCorrections`) ships in its
  own ticket — see § 11.
- Per-block compression. The `recipe` write parameter remains FITS-only.
  ASDF block compression lands in a follow-up ticket.
- Native `asdf-astropy`-tagged tables. MVP routes table data through the
  existing `TableColumnModel` to keep the on-disk shape consistent with
  the JSON / FITS / NDF backends; a future ticket may emit
  `asdf-astropy` table tags instead.
- Hand-rolling any part of the ASDF format. We rely on the `asdf` Python
  library for binary blocks, YAML serialization, schema registration,
  and lazy loading.
- Compatibility with `asdf` < 4.0. The converter context APIs we depend
  on stabilized in asdf 4.0.

## 3. Architecture

### 3.1 Module layout

```
python/lsst/images/asdf/
├── __init__.py         # asdf-version-checked import gate + re-exports
├── _common.py          # AsdfRef pointer model, opaque metadata, constants
├── _output_archive.py  # AsdfOutputArchive + write()
├── _input_archive.py   # AsdfInputArchive + read()
├── _converters.py      # asdf.extension.Converter shims for LSST tags
├── _extension.py       # Extension + get_resource_mappings entry points
└── schemas/
    └── lsst.org/
        └── images/
            ├── image-1.0.0.yaml
            ├── mask-1.0.0.yaml
            └── masked-image-1.0.0.yaml
```

This mirrors the existing `lsst.images.ndf` / `lsst.images.fits` /
`lsst.images.json` packages, so a developer familiar with one backend
finds the same files in the same roles in the new one.

### 3.2 Optional dependency gating

`pyproject.toml` gains:

```toml
[project.optional-dependencies]
asdf = ["asdf >= 4.0", "asdf-astropy >= 0.7"]
```

`python/lsst/images/asdf/__init__.py` imports `asdf` at module top inside
a `try` block and re-raises `ImportError` with an install hint identical
in shape to the NDF gate:

```python
try:
    import asdf  # noqa: F401
except ImportError as e:
    raise ImportError(
        "lsst.images.asdf requires the optional 'asdf' package. "
        "Install it directly or via 'pip install lsst-images[asdf]'."
    ) from e
```

A version check on `asdf.__version__ >= "4.0"` raises a clearer
`ImportError` for users with a stale env rather than letting them hit
opaque converter-context errors at runtime.

`python/lsst/images/formatters.py` grows a `_HAVE_ASDF` flag mirroring
`_HAVE_NDF`. The `_BACKENDS` table gets a `.asdf` row only when
`_HAVE_ASDF` is true. `GenericFormatter.supported_extensions` grows
`.asdf`. `get_write_extension` accepts `format="asdf"` and raises the
existing "Requested format … is not supported" error if asdf isn't
installed and the user asks for it.

### 3.3 Entry-point registration

```toml
[project.entry-points."asdf.extensions"]
lsst_images = "lsst.images.asdf._extension:get_extensions"

[project.entry-points."asdf.resource_mappings"]
lsst_images = "lsst.images.asdf._extension:get_resource_mappings"
```

`get_extensions()` returns a single `_LsstImagesExtension` instance that
declares the in-MVP converters; `get_resource_mappings()` returns a
`DirectoryResourceMapping` rooted at `schemas/`. Both follow the
standard asdf plugin patterns documented in the asdf-standard project.

## 4. Write path

### 4.1 `write()` and `AsdfOutputArchive.open()`

`lsst.images.asdf.write(obj, path, *, metadata=None, butler_info=None)`
matches the signatures of `lsst.images.json.write` and
`lsst.images.fits.write`. The body is the same shape as the other
backends:

```python
with AsdfOutputArchive.open(path) as archive:
    tree = (archive.serialize_direct(name, obj.serialize)
            if name is not None else obj.serialize(archive))
    if metadata is not None:
        tree.metadata.update(metadata)
    if butler_info is not None:
        tree.butler_info = butler_info
    archive.add_tree(tree)
```

`AsdfOutputArchive` holds:

- `_asdf_file: asdf.AsdfFile` — created in `open()`, owns the binary blocks
  and the YAML tree until `add_tree` finalizes the write.
- `_pointers_by_key: dict[Hashable, AsdfRef]` — pointer-dedup map matching
  the JSON / FITS archives.
- `_indirect: list[Any]` — buffer of serialized-pointer-target models;
  spliced into the asdf tree at finalization (see § 6).
- `_blocks: list[np.ndarray]` — buffer of arrays in the order
  `add_array` was called, also spliced into the asdf tree at finalization
  (see § 6). The list index is the value returned in
  `ArrayReferenceModel.source`.
- `_frame_sets: list[tuple[FrameSet, AsdfRef]]` — same role as in the
  JSON / FITS archives, supports `iter_frame_sets`.

`open()` refuses to overwrite an existing file, raising `OSError` for
parity with `FitsOutputArchive.open()`.

### 4.2 `add_array` and `add_table`

`add_array(array, *, name=None, update_header=...)`:

- If called with `name=None` and the archive is not nested inside a
  `NestedOutputArchive`, raise `RuntimeError` (matches FITS / NDF;
  ASDF blocks need a stable identifier).
- Append `array` to `self._blocks` and let `index = len(self._blocks) - 1`.
  The array is *not* yet placed into the asdf tree; that happens in
  `add_tree` (see § 6). asdf promotes any `np.ndarray` it finds in the
  tree at write time into a binary block, so this list-driven approach
  guarantees deterministic block indices ordered by `add_array` call
  order.
- Return `ArrayReferenceModel(source=index, shape=list(array.shape),
  datatype=NumberType.from_numpy(array.dtype), byteorder="big")`.
- `update_header` is silently ignored — it's an FITS-specific callback,
  and the OutputArchive contract explicitly allows it to be dropped by
  non-FITS backends.

`add_table` and `add_structured_array` build the existing `TableModel` /
`TableColumnModel` shape and register each column's underlying ndarray
via `add_array`, so column data lands as ASDF blocks. The on-disk shape
is therefore the same as the FITS / NDF backends in terms of which data
is binary and which is YAML / JSON.

### 4.3 `serialize_direct` and `serialize_pointer`

`serialize_direct(name, serializer)` returns a `NestedOutputArchive`
exactly like `JsonOutputArchive`.

`serialize_pointer(name, serializer, key)`:

- If a pointer for `key` already exists, return it.
- Otherwise serialize the model, append to `self._indirect`, and return
  `AsdfRef(ref=f"#/lsst/indirect/{len(self._indirect) - 1}")`. We keep
  JSON-Pointer semantics rather than using YAML anchors so the
  `serialize_pointer` contract is uniform across backends and so the
  same `JsonRef` validator code path can be reused (with a backend-local
  `AsdfRef` alias for clarity).

### 4.4 `serialize_frame_set` and FrameSet via YamlChan

`AsdfOutputArchive.serialize_frame_set(name, frame_set, serializer, key)`:

1. Behaves as in JSON / FITS for the pointer mapping bookkeeping.
2. Asks the `FrameSet`'s underlying `astshim` object to emit itself via
   a new `YamlChan` wrapper alongside the existing `Channel` / `FitsChan`
   wrappers in `lsst.images._transforms._ast`.
3. Parses the YamlChan output into a Python dict (it's already standard
   ASDF YAML with `tag:stsci.edu/asdf/wcs/...` tags from the
   `asdf-wcs-schemas` package).
4. Embeds the parsed dict into the asdf tree at the appropriate location
   so external ASDF tools see a real, standard WCS object — no LSST or
   astshim shim required for round-trip with any client that has
   `asdf-wcs-schemas` installed.

### 4.5 `add_tree` (finalizer)

`add_tree(tree)`:

1. Sets `tree.indirect = self._indirect`.
2. Dumps the pydantic tree to a Python dict via
   `model_dump(mode="python")`.
3. Walks the dict and, for every dict-valued node whose corresponding
   pydantic model declares a `tag` in `model_config["json_schema_extra"]`,
   wraps the dict in asdf's tagged-mapping construct so the tag is
   emitted in the YAML output. This is the single piece of genuinely
   new code; it's tightly bounded (≈ 30 lines) and reusable.
4. Moves the `indirect` key and inserts the `_blocks` list under a new
   top-level `lsst:` namespace, producing `tree["lsst"]["indirect"]` and
   `tree["lsst"]["blocks"]`. The root of the asdf tree therefore only
   exposes the canonical fields (`metadata`, `butler_info`,
   `schema_version`, and the top-level type's content) plus the single
   `lsst:` mapping to casual ASDF browsing.
5. Splices the result into `self._asdf_file.tree`. The ndarrays in
   `tree["lsst"]["blocks"]` are seen by asdf at write time and promoted
   to binary blocks; YAML references to them appear inline at that
   location.
6. Calls `self._asdf_file.write_to(path)`.

## 5. Read path

### 5.1 `read()` and `AsdfInputArchive.open()`

`lsst.images.asdf.read(cls, target, **kwargs)` mirrors
`lsst.images.json.read`:

```python
with AsdfInputArchive.open(target) as archive:
    tree = archive.get_tree(cls._get_archive_tree_type(AsdfRef))
    obj = tree.deserialize(archive, **kwargs)
return ReadResult(obj, tree.metadata, tree.butler_info)
```

`AsdfInputArchive.open(uri, *, partial=False)` accepts the same `partial`
kwarg `FitsInputArchive` does, for formatter parity. The body opens the
file with `asdf.open(uri, lazy_load=True, copy_arrays=False)` regardless
of `partial`; asdf's lazy loading means the YAML tree is read but blocks
aren't until first attribute access. `partial=True` is a hint propagated
to `read_component` callers and otherwise behaves the same as
`partial=False` at the archive level.

### 5.2 `get_tree`, `get_array`, `get_table`, `deserialize_pointer`

- `get_tree(tree_type)` — pulls the top-level mapping from
  `self._asdf_file.tree`, moves `lsst.indirect` back to the top-level
  `indirect` key (inverse of write-side namespacing), strips YAML tags
  from any tagged values (so pydantic doesn't trip on asdf's
  tagged-mapping subclass), and returns `tree_type.model_validate(...)`.
  The `lsst.blocks` list stays where it is, since it's consulted
  directly by `get_array` via the archive (not via the pydantic tree).

- `get_array(model, *, slices=..., strip_header=...)` — if `model` is an
  `ArrayReferenceModel`, returns
  `self._asdf_file.tree["lsst"]["blocks"][model.source][slices]` —
  the same on-disk location that `add_array` populates. The `[slices]`
  access triggers asdf's lazy block load only for the requested slice.
  If `model` is an `InlineArrayModel`, materialize as in the JSON
  backend.

- `get_table` / `get_structured_array` — same shape as the JSON
  implementations, but with each column's `data.source` resolved via the
  block path above rather than `InlineArrayModel.data`.

- `deserialize_pointer(pointer, model_type, deserializer)` — indexes
  into the `indirect` list as in JSON, with the same per-archive cache
  of deserialized values keyed on the pointer index.

- `get_frame_set(ref)` — dereference the pointer, dump the WCS subtree
  back to a YAML string, feed it to `YamlChan.read()`, and wrap the
  resulting AST object as a `FrameSet`. Cached per archive so repeated
  lookups return the same instance.

- `get_opaque_metadata()` — returns a small pydantic model wrapping
  asdf's history-entries list so a read → write round-trip in the same
  format preserves it.

### 5.3 Component reads via the unified formatter

`formatters.py` already routes `partial=True` only to FITS; NDF uses a
different mechanism. The ASDF row follows the FITS branch (`partial=True`
keyword on `open()`), and `_read_component_from_uri` works unchanged
because `AsdfInputArchive` exposes the same `get_tree` + archive
interface as `FitsInputArchive`.

The `tag-stripping` step in `get_tree` is the only adjustment to
`_read_component_from_uri`'s tree handling: the component-pluck path
reads attributes off the pydantic tree, so once `get_tree` returns a
validated `ArchiveTree`, the rest of the formatter code is identical to
the FITS path.

## 6. The `lsst:` namespace

Two backend-specific buffers live under a top-level `lsst:` mapping in
the on-disk YAML, rather than at the root:

- `tree["lsst"]["indirect"]` — the pointer-target buffer (analogous to
  `ArchiveTree.indirect` on FITS / JSON).
- `tree["lsst"]["blocks"]` — the ordered list of binary-block-backed
  ndarrays referenced by `ArrayReferenceModel.source`.

Everything else (`metadata`, `butler_info`, `schema_version`, and the
top-level type's content) stays at the root, so external tools opening
the file see only the canonical fields plus a single `lsst:` mapping
for LSST-specific bookkeeping.

`AsdfRef.ref` values therefore look like `"#/lsst/indirect/3"`. The
URL fragment is a pointer into the *on-disk* YAML; the archive's
`deserialize_pointer` parses the index out and indexes into its
in-memory `_indirect` list (which `get_tree` populated from
`lsst.indirect` and then moved back to the top-level `indirect` slot
of the pydantic tree). Round-trip equality holds because the namespace
move is a pure tree rewrite both on write and on read.

The pydantic-side representation of the on-disk shape (used for
schema/pydantic agreement tests in § 7.3 and § 9) is generated against
the asdf-backend's `ArchiveTree` subclass, which serializes `indirect`
under `lsst.indirect` via either a custom serializer or per-backend
tree subclass — the implementation plan decides which.

## 7. Tags, schemas, and converters

### 7.1 Tag URI scheme

LSST tags use:

```
asdf://pipelines.lsst.io/images/tags/<name>-<major>.<minor>.<patch>
```

Schemas use:

```
asdf://pipelines.lsst.io/images/schemas/<name>-<major>.<minor>.<patch>
```

`<name>` follows the schema-naming convention from the existing
schema-versioning spec (lowercase, hyphen-separated form of the
`ArchiveTree` subclass's public name, minus the `SerializationModel`
suffix): `image`, `mask`, `masked-image`. The `asdf://` scheme prefix
is required by the asdf library for tag/schema URIs (vs. the `https://`
form the human-facing pipelines.lsst.io URL uses); the path component
matches.

### 7.2 Tags emitted in MVP

| In-memory type | Tag URI | Source of truth |
|---|---|---|
| `np.ndarray` (inline) | `tag:stsci.edu/asdf/core/ndarray-1.1.0` | existing `InlineArrayModel` |
| `np.ndarray` (block-backed) | `tag:stsci.edu/asdf/core/ndarray-1.1.0` | existing `ArrayReferenceModel` |
| `astropy.units.Unit / Quantity / Time` | existing `stsci.edu` tags | existing `_asdf_utils.py` |
| `astropy.table.Table` column | `tag:stsci.edu/asdf/core/column-1.1.0` | existing `TableColumnModel` |
| `Image` | `asdf://pipelines.lsst.io/images/tags/image-1.0.0` | new — `ImageSerializationModel` |
| `Mask` | `asdf://pipelines.lsst.io/images/tags/mask-1.0.0` | new — `MaskSerializationModel` |
| `MaskedImage` | `asdf://pipelines.lsst.io/images/tags/masked-image-1.0.0` | new — `MaskedImageSerializationModel` |
| WCS / `FrameSet` | standard `tag:stsci.edu/asdf/wcs/...` tags (via `asdf-wcs-schemas`) | starlink-pyast `YamlChan` |

Each in-MVP `ArchiveTree` subclass declares its tag in
`model_config["json_schema_extra"]` exactly as the existing types in
`_asdf_utils.py` and `_tables.py` already do. The tag-injection walker
in `AsdfOutputArchive.add_tree` reads from there directly — no separate
registry.

### 7.3 Schema documents

Each new tag ships a YAML schema document under
`python/lsst/images/asdf/schemas/`. Schemas are deliberately
**structural**: required fields, types, references to other tags. The
pydantic model remains the authority for write-side validation; the
schema is for read-side checks by external `asdf` tools and for
self-description of the file format.

To prevent drift, a CI test compares each on-disk schema YAML against
`Model.model_json_schema(mode="serialization")` after a JSON-schema → YAML-schema
translation step. The translation is small and ships as a private test
helper (location TBD by the implementation plan); it asserts equality of
the structural core (required fields, types, descriptions) and ignores
asdf-only metadata that pydantic doesn't surface (e.g. `$schema`, `tag`).

### 7.4 Converters

Each new tag gets an `asdf.extension.Converter` subclass in
`_converters.py`. Converters are thin: they hand the YAML tree to
pydantic and then to the model's existing `deserialize` method.

```python
class _MaskedImageConverter(Converter):
    tags = ["asdf://pipelines.lsst.io/images/tags/masked-image-*"]
    types = ["lsst.images._masked_image.MaskedImage"]

    def to_yaml_tree(self, obj, tag, ctx):
        archive = _AsdfWriterContext.for_ctx(ctx)
        return archive.write_object(obj)

    def from_yaml_tree(self, node, tag, ctx):
        archive = _AsdfReaderContext.for_ctx(ctx)
        model = MaskedImageSerializationModel.model_validate(_strip_tags(node))
        return model.deserialize(archive)
```

The `_AsdfWriterContext` / `_AsdfReaderContext` adapters wrap asdf's
serialization context (which exposes the active `AsdfFile`) so the
existing `add_array` / `get_array` paths work from inside a converter
the same as they do from inside the `read` / `write` module-level
functions. Adapters are created lazily on first converter call and
cached on the `AsdfFile`. This is the plumbing that lets a bare
`asdf.open("foo.asdf")` (no `lsst.images.asdf.read` involved) return a
real `MaskedImage` instance when `lsst-images[asdf]` is installed.

The extension class:

```python
class _LsstImagesExtension(Extension):
    extension_uri = "asdf://pipelines.lsst.io/images/extensions/lsst-images-1.0.0"
    converters = [_ImageConverter(), _MaskConverter(), _MaskedImageConverter()]
    tags = [...]  # tag-to-schema URI bindings
```

### 7.5 Interaction with schema versioning

The recently-merged schema-versioning spec
(`2026-05-15-schema-versioning-design.md`) adds `SCHEMA_NAME` and a
round-tripped `schema_version` field to every `ArchiveTree`. For ASDF:

- The tag's name and `major.minor.patch` come from `SCHEMA_NAME` and
  `schema_version` on the model. No second source of truth.
- The tag-injection walker emits
  `asdf://pipelines.lsst.io/images/tags/{SCHEMA_NAME}-{schema_version}`
  automatically.
- A new tag / schema pair is required whenever the major version bumps
  (ASDF semver semantics, which match the schema-versioning spec).
  Minor / patch bumps reuse the existing schema.

The schema-versioning spec also adds a container-version stamp for FITS
and NDF. For ASDF, the asdf library itself stamps an
`asdf_library` / `history` entry on every file, which serves as the
container-version indicator; we do not add a separate container
version.

## 8. Error handling

- `import lsst.images.asdf` without `asdf` installed → `ImportError`
  with an install hint matching the NDF gate.
- `asdf` < 4.0 detected at import → `ImportError` calling out the
  minimum version, before any converter-context code runs.
- Reading an `.asdf` file whose top-level tag is not an in-scope LSST
  type → `ArchiveReadError("Top-level type {tag} not supported by
  lsst.images")`.
- Reading a file whose schema-version major doesn't match the model's
  current major → `ArchiveReadError` with the version-mismatch
  message defined by the schema-versioning spec.
- `add_array(array, name=None)` outside a nested archive → `RuntimeError`,
  matching FITS / NDF. JSON allows this because it inlines arrays; ASDF
  cannot, because blocks need a stable identifier.
- Writing to an existing path → `OSError`, matching FITS.

## 9. Testing

New test modules:

- `tests/test_asdf_input_archive.py`
- `tests/test_asdf_output_archive.py`
- `tests/test_asdf_roundtrip.py`
- `tests/test_asdf_layout.py`

Coverage:

- **Round-trip parity tests** — for each in-scope type, write to
  `.asdf`, read back, assert deep equality. Reuses the existing
  `_roundtrip.py` helpers parameterized on backend.
- **Interop sanity tests** — open the file via bare `asdf.open()` (no
  archive wrapper) and assert:
  - The top-level tag is the expected LSST tag.
  - Each `np.ndarray` field comes back as a real ndarray (proves the
    binary block round-trip).
  - With the entry points registered (and `lsst-images[asdf]` installed
    in the test env), `asdf.open()` returns a real `MaskedImage` object
    (proves the converter shim path).
  - WCS round-trips as a standard ASDF WCS object when
    `asdf-wcs-schemas` is installed.
- **Lazy read tests** — assert that opening a `MaskedImage` file with
  `partial=True` and reading only the `mask` component does *not*
  materialize the `image` block (check via asdf's internal block-load
  state).
- **Schema/pydantic agreement tests** — for each new schema YAML, parse
  it and compare against the model's
  `model_json_schema(mode="serialization")` after the JSON→YAML schema
  translation. Guards against drift.
- **Formatter dispatch tests** — extend `tests/test_formatters.py` with
  `format="asdf"` and `.asdf` URI cases for each scope type, mirroring
  existing FITS / JSON / NDF test rows.
- **`asdf`-not-installed test** — skip behaviour mirrors the existing
  NDF `_HAVE_NDF` pattern in `tests/test_formatters.py`.

## 10. Documentation

- New `doc/lsst.images/asdf.rst` mirroring `ndf.rst` — purpose, file
  layout, interop expectations, current limitations.
- New `doc/changes/DM-XXXXX.feature.md` fragment for towncrier.
- Update `lsst.images.serialization` module docstring — remove the
  "ASDF (just hypothetical for now)" line and replace it with a
  pointer to `lsst.images.asdf`.

## 11. Deferred to follow-ups

Each row below is a follow-up ticket, ordered by approximate dependency:

| Type | Notes |
|---|---|
| `Box`, `Interval`, `Polygon` | Investigate whether existing `gwcs` / asdf-stdlib geometry schemas fit; otherwise propose new `pipelines.lsst.io` ones. |
| `Projection`, `CameraFrameSet`, `Transform` | Build on the `FrameSet` support already landed in MVP. |
| `GaussianPsf`, `PiffPsf`, `LegacyPsf` | Each gets its own tag / schema. Piff is the heavy one (binary blob in a block). |
| `ChebyshevField`, `SplineField`, `SumField`, `ProductField` | Bounded-field family — likely one base schema plus per-variant specializations. |
| `ObservationSummaryStats`, `ApertureCorrections` | Standalone, small. |
| `VisitImage` | Trivial once PSF + summary stats + aperture corrections land. |
| `CellCoadd` | Trivial once PSF + provenance + sub-image scaffolding land. |
| `Camera` family | Pulls in `Detector`, `Amplifier`, etc. — relatively large surface. |
| Block compression | A `write_parameters["recipe"]` analogue for ASDF blocks. |
| `asdf-astropy` native tables | Optionally surface columns as full `astropy.table.Table` tags instead of `TableColumnModel`. Trade-off: deeper interop vs. divergence from the JSON / FITS / NDF backend shape. |

# Zarr I/O Backend for `lsst.images` — Design (revised)

**Status:** Approved (design phase). Supersedes the v1 design at commit `a11db46` after collaborator review.
**Date:** 2026-05-22 (revised)
**Author:** Tim Jenness (with Claude collaborator)

## 1. Goals, Scope, Non-Goals

### Goals

Add a `lsst.images.zarr` subpackage providing:

- A `ZarrOutputArchive` and `ZarrInputArchive` implementing the existing
  `lsst.images.serialization` `OutputArchive` / `InputArchive` ABCs.
- Top-level `read()` and `write()` helpers consistent with the FITS,
  JSON, and NDF backends.
- A Python intermediate representation (IR) — `ZarrDocument`,
  `ZarrGroup`, `ZarrArray`, etc. — that describes the on-disk layout
  independently of `zarr-python`, mirroring the role `NdfDocument`
  plays for the NDF backend.

Because the backend builds on the abstract archive interface, every
image type that already serializes to FITS/JSON/NDF (`Image`, `Mask`,
`MaskedImage`, `VisitImage`, `ColorImage`, `CellCoadd`, plus any
`serialize()`-implementing object reachable through the archive) works
with no per-type code in the backend itself.

### Standards alignment (changed from v1)

The on-disk layout is **xarray-/CF-shaped at the root** with
**OME-NGFF v0.5 metadata as a discoverability layer on top**. The root
group is a sibling collection of arrays (`image/`, `variance/`,
`mask/`) so:

- `xr.open_zarr(path)` returns a `Dataset` with the masked-image
  components as data variables sharing the `(y, x)` dimensions.
- Geospatial / CF tooling (rasterio, GDAL's Zarr driver, QGIS) reads
  the `mask` array's `flag_masks` / `flag_meanings` /
  `flag_descriptions` attributes directly.
- OME-NGFF tooling (`napari`, `neuroglancer`, `ngff-validator`,
  `ome-zarr-py`) sees an OME multiscales block whose
  `dataset.path` points at the same `image` array — the OME view and
  the xarray view share bytes.

The pivot vs the v1 design: the root is no longer a multiscale image
with `lsst:` companions hanging off it; companion arrays are
first-class siblings, and OME's `multiscales.datasets[].path` references
them. This enables xarray / GDAL interop with no byte duplication.

### Cloud-first, local works too

- Default chunk geometry is tile-aligned (~1024×1024 for plain images,
  `cell_shape` for `CellCoadd`).
- Sharding (zarr v3 native) is enabled by default with a tunable shard
  size to keep object counts manageable on S3/GCS.
- Subset reads via `slices=` exploit zarr's chunk index, going
  straight to the lazy `zarr.Array` handle so only the touched chunks
  are fetched.
- Both `DirectoryStore` and `ZipStore` are supported; the choice is
  driven by URI shape (`*.zarr.zip` → `ZipStore`, otherwise directory).
  Remote URIs go through `lsst.resources.ResourcePath` and `fsspec`.

### Scope

Same image-type coverage as the FITS backend: `Image`, `Mask` (2-D in
v1), `MaskedImage`, `VisitImage`, `ColorImage`, `CellCoadd`, plus any
`serialize()`-implementing object reachable through the archive
interface.

`ColorImage` writes its three channels as **sibling sub-archives**
(`red/`, `green/`, `blue/`), not as a stacked `(3, Y, X)` array — see
[§3](#3-on-disk-layout). The previous design's stacking + JSON-pointer
rewrite is removed because it duplicates bytes for large images.

`CellCoadd`'s per-cell PSF is whatever shape `CellCoadd.serialize`
natively emits — typically a 4-D `(Cy, Cx, Py, Px)` array — with
cell-aligned chunks. No fixup pass.

### Non-Goals (initial release)

- No dask / lazy `read_lazy()` API — added later, tracked as
  follow-up.
- No multi-level OME multiscale pyramid (we only ever write one level
  pointed at by `path: image`).
- **No NGFF RFC-5 nonlinear coordinate transformations as
  authoritative.** v1 emits an OME-NGFF v0.5 affine
  `coordinateTransformations` as an external-tool affordance, with
  the AST `FrameSet` string as the authoritative round-trip source.
  RFC-5 transformations as authoritative is a follow-up — see
  [§6](#6-follow-up-work-out-of-scope) — **blocked on writing an AST
  JSON channel** that serializes a `FrameSet` to / from RFC-5
  transformation JSON.
- No 3-D mask layout for masks with more than 64 planes — v1 raises
  on write. 3-D fallback tracked as follow-up.
- No automatic OME `consolidated_metadata` extension. Tracked as
  follow-up.

### Dependency

Optional `[zarr]` extra requiring `zarr >= 3.0` and any required codec
packages. The top-level `lsst/images/zarr/__init__.py` does a guarded
`import zarr` and raises `ImportError` with installation guidance if
missing, mirroring the NDF backend.

## 2. Module Layout and Architecture

```
python/lsst/images/zarr/
├── __init__.py          guarded `import zarr`; re-exports
├── _common.py           ZarrPointerModel (analog of NdfPointerModel),
│                         attribute namespace constants ("lsst:", "ome:"),
│                         ZarrCompressionOptions dataclass,
│                         path/JSON-pointer helpers
├── _model.py            Python intermediate representation:
│                         ZarrDocument, ZarrGroup, ZarrArray, ZarrAttributes,
│                         OmeMultiscale, OmeOmero, from_zarr() / to_zarr()
│                         materialization methods
├── _layout.py           Layout rules: archive-class → axes mapping;
│                         CF flag-attrs construction for mask groups;
│                         affine extraction + residual validator;
│                         OME multiscale block construction
├── _output_archive.py   ZarrOutputArchive and write()
├── _input_archive.py    ZarrInputArchive and read()
└── _store.py            Wrapper that turns a ResourcePath / fsspec URI
                         into the right zarr.storage.Store
                         (LocalStore / ZipStore / FsspecStore)
```

### Fit with existing abstractions

- `ZarrOutputArchive[ZarrPointerModel]` implements the abstract
  methods (`serialize_direct`, `serialize_pointer`,
  `serialize_frame_set`, `add_array`, `add_table`,
  `add_structured_array`, `iter_frame_sets`).
- `ZarrPointerModel` is a small Pydantic model holding a zarr path
  (e.g. `"/lsst/psf/tree"`); when a model field carries a
  `ZarrPointerModel`, the consumer dereferences it through the input
  archive — same pattern as `NdfPointerModel`.
- `update_header` callbacks (intended for FITS) are accepted and
  ignored, identical to the JSON backend.
- The `serialization.ArchiveTree` JSON tree is stored verbatim as a
  UTF-8 zarr array at `tree` (root-level). Array references in the
  tree resolve to zarr paths under the same root.

### Two-pass write driven by the IR

During `obj.serialize(archive)`, the archive populates an in-memory
`ZarrDocument`. Only when the context manager exits does the IR
materialize to zarr-python via the configured store.

Benefits:

- Per-class layout decisions (CF flag attrs on mask, OME multiscale
  block, cell-grid metadata) are made once in `_layout.py` against
  the populated IR.
- Tests can assert on the IR without writing files.
- A future "validate-then-commit" step (e.g. `ngff-validator`
  integration) can run against the IR.

Compared to the v1 design, the IR's *write* side has **no fixup
pass** that rewrites or stacks staged arrays. Each `add_array(name)`
call lands at the zarr path equal to `name` (after stripping the
leading `/`). `name="image"` → `/image`; `name="mask"` → `/mask`;
the nested `name="red/image"` produced by
`serialize_direct("red", red.serialize)` → `/red/image`. There is
no special-case dictionary mapping JSON pointers to zarr paths.

### Lazy read invariant (unchanged from v1)

`ZarrArray.data` holds either a staged `numpy.ndarray` (write side)
or a lazy `zarr.Array` handle (read side). `from_zarr` never reads
chunk bytes; only `ZarrArray.read(slices=...)` does, and it forwards
`slices` straight to the lazy handle so only chunks intersecting
the slice are fetched. A `_CountingStore`-based regression test
asserts a single-chunk subset of a 16×16 / chunks=(4,4) array
touches strictly fewer chunk reads than a full read.

### Read mirrors write

`ZarrInputArchive.open()` opens the store, builds a `ZarrDocument`
view backed by lazy zarr-python objects, validates the
`lsst.archive_class` and `lsst.version` root attributes, locates the
`tree` JSON document, and parses it into the appropriate
`ArchiveTree` Pydantic model. `get_array(model, slices=...)`
translates the model's path into a chunk-aligned zarr read.

`ArrayReferenceModel.source` strings are plain `zarr:/<path>`. The
v1 design's `?c=N` and `?cell=Cy,Cx` query suffixes are removed —
no stacking means no compound source URLs.

### Backend write helper signature

```python
def write(
    obj: Any,
    path: ResourcePathExpression | None = None,
    *,
    chunks: Mapping[str, tuple[int, ...] | None] | None = None,
    shards: Mapping[str, tuple[int, ...] | None] | None = None,
    compression: Mapping[str, ZarrCompressionOptions | None] | None = None,
    metadata: dict[str, MetadataValue] | None = None,
    butler_info: ButlerInfo | None = None,
) -> ArchiveTree: ...
```

`chunks`, `shards`, and `compression` are per-array dicts keyed by
the JSON pointer of the attribute the array backs (or its zarr
path), mirroring the existing `compression_options` pattern from
the FITS backend. Different arrays have different ranks (2-D image,
2-D mask, 4-D per-cell PSF) so a single tuple value would not be
meaningful. Missing keys fall back to the per-class defaults from
[§3](#chunking-and-sharding-defaults). A value of `None` for a key
means "use the default for this array"; explicitly setting `shards`
to `{}` does *not* disable sharding — to disable, pass
`{"<key>": None}` per array.

`image`, `variance`, and `mask` are expected to share the spatial
chunk shape (CF / xarray / GDAL all assume aligned chunks). The
output archive derives `variance` and `mask` chunks from `image`'s
chunk shape when the user has not overridden them.

## 3. On-Disk Layout (the spec)

### Root layout per archive class

Every archive class lays out its data as **siblings under the root**.
Non-array metadata (the JSON round-trip tree, the AST WCS string)
also lives at the root so xarray and ome-zarr both see a clean
group.

For a `MaskedImage` / `VisitImage`:

```
visitimage.zarr/
├── zarr.json            ← group attrs (see below)
├── image/               ← (Y, X) zarr array, science pixels
├── variance/            ← (Y, X) zarr array
├── mask/                ← (Y, X) zarr array, packed mask integers
├── tree                 ← 1-D uint8 array, pydantic JSON round-trip
└── wcs_ast              ← 1-D uint8 array, AST FrameSet text
```

For an `Image` with no projection, `wcs_ast` is omitted; for an
`Image` with no mask/variance, those siblings are simply absent.

For `ColorImage`:

```
colorimage.zarr/
├── zarr.json            ← lsst.archive_class = "ColorImage"; no OME multiscales
├── red/                 ← itself a valid Image-shaped sub-archive
├── green/               ← (with its own image/, multiscales, etc.)
├── blue/
├── tree
└── wcs_ast
```

Each channel sub-archive is a valid `Image` archive in its own right
(its own `image/` array, its own `lsst.archive_class = "Image"`, its
own OME multiscales). The root group's `lsst.archive_class` is
`"ColorImage"` and it has **no OME multiscales of its own** — there
is no stacked multi-channel array, so there is nothing for OME to
render at root level. External tools reading the root see three
nested OME images, which is consistent with the recursive-composition
rule. (A future follow-up may add a stacked single-array view; v1
does not because of the no-byte-duplication rule.)

For `CellCoadd`:

```
cellcoadd.zarr/
├── zarr.json            ← lsst.archive_class + lsst.cell_grid
├── image/               ← (Y, X), chunks aligned to cell_shape
├── variance/
├── mask/
├── psf/                 ← (Cy, Cx, Py, Px) 4-D, chunks (1, 1, Py, Px)
├── tree
└── wcs_ast
```

`psf` is whatever shape `CellCoadd.serialize` natively emits — there
is no stacking fixup. Cell-grid metadata lives in the
`lsst.cell_grid` block of the root group's attributes.

### Top-level group attributes (`zarr.json` `attributes`)

```jsonc
{
  "data_model": "org.lsst.masked_image",  // or .image / .visit_image / etc.
  "version": 1,                           // org.lsst.* schema version

  "ome": {
    "version": "0.5",
    "multiscales": [{
      "name": "<archive_class lowercase>",
      "axes": [/* see per-class table below */],
      "datasets": [{
        "path": "image",
        "coordinateTransformations": [/* affine; see §4 below */]
      }]
    }],
    // Only present on archive classes whose top-level array has a
    // channel axis. Not used in v1 (no stacked ColorImage view).
    "omero": { "channels": [...] }
  },

  "lsst": {
    "version": 1,                       // schema version of lsst extension
    "archive_class": "VisitImage",      // dispatch for read-side construction
    "tree": "tree",                     // zarr path to JSON tree (relative)
    "wcs_ast": "wcs_ast",               // zarr path to AST string, optional
    "wcs_simplified_dropped": false,    // see §4 below
    "wcs_simplified_max_residual_pixels": 0.13,  // observed max; only when affine emitted
    "opaque_metadata_format": "fits",   // optional, only when present
    "cell_grid": { "bbox": ..., "cell_shape": [256, 256] }  // CellCoadd only
  }
}
```

For `ColorImage`, the root group has `lsst.archive_class = "ColorImage"`
and no `ome.multiscales`.

### Axis choice per archive class

| Archive class | Axes (root multiscale) | Top-level science array | Notes |
|---|---|---|---|
| `Image`, `MaskedImage`, `VisitImage`, `CellCoadd` | `[y, x]` | `image` | Standard 2-D image. |
| `Mask` (standalone, 2-D) | `[y, x]` | `mask` | When written outside a parent. |
| `ColorImage` | (none at root) | (none at root) | Each `red/`, `green/`, `blue/` sub-archive carries its own `[y, x]` multiscale. |

### Image / variance arrays — array attrs

`image/zarr.json` (and likewise `variance/zarr.json` and any other
2-D float sibling):

```jsonc
{
  "_ARRAY_DIMENSIONS": ["y", "x"],     // xarray
  "long_name": "science image",        // CF
  "units": "adu"                       // CF (when known)
}
```

### Mask array — 2-D packed integers with CF flag attrs

`mask` is a **2-D `(y, x)` unsigned-integer array**. The dtype is
chosen by the schema's plane count: `uint8` for ≤8 planes, `uint16`
for ≤16, `uint32` for ≤32, `uint64` for ≤64. Each pixel's bits encode
which planes apply at that pixel — the same logical representation
the FITS backend writes, so FITS↔Zarr mask round-trips need no bit-
repacking.

`mask/zarr.json`:

```jsonc
{
  "_ARRAY_DIMENSIONS": ["y", "x"],
  "flag_masks":      [1, 2, 4, 8, 16],
  "flag_meanings":   "BAD SAT CR INTRP NO_DATA",
  "flag_descriptions": [
    "Bad pixel.",
    "Saturated.",
    "Cosmic ray.",
    "Interpolated.",
    "No data."
  ]
}
```

`flag_masks` and `flag_meanings` are CF conventions:
`flag_meanings` is a **single space-separated string** (not a list)
per CF; `flag_descriptions` is the LSST extension carrying the
human-readable per-plane text from `MaskPlane.description`.

Schemas with **more than 64 planes** raise on write in v1. A 3-D
`(plane_byte, y, x)` fallback is tracked as a follow-up.

### The JSON round-trip tree (`tree`)

A 1-D `uint8` zarr array containing UTF-8 JSON. Same content the JSON
backend produces, but with `ArrayReferenceModel` references whose
source strings are zarr paths within the store: `"zarr:/image"`,
`"zarr:/mask"`, `"zarr:/red/image"` (for nested ColorImage channels),
`"zarr:/psf"` (for CellCoadd). These resolve into the zarr store, not
into the JSON document itself, so they do not use the JSON-Pointer
`#/` fragment prefix. There are **no compound source URLs** (no
`?c=N`, no `?cell=Cy,Cx`) because no arrays are stacked.

### AST WCS string (`wcs_ast`)

A 1-D `uint8` zarr array containing the AST `FrameSet` text produced
by an `astshim.Channel`. The full text is stored as bytes; this is
the **authoritative round-trip source** for the WCS. The OME affine
emitted in `multiscales.datasets[].coordinateTransformations` is an
approximation for external tools and is dropped when its residual
exceeds the [§4](#4-error-handling-edge-cases-round-trips) threshold.

For multi-frame-set archives (`serialize_frame_set` calls referencing
distinct WCS objects), each frame set is stored at
`/lsst/frame_sets/<key>` and referenced via `ZarrPointerModel` in
the JSON tree, mirroring the NDF / FITS pattern.

### Tables

A table named `<name>` lives at `/lsst/tables/<name>/<column>`: one
1-D zarr array per column, sibling to the others under a group whose
attributes carry the `lsst.table = {columns: [...], length: N,
meta: {...}}` block. Structured arrays use the same group form; the
deserialised type differs.

### Recursive composition

Any sub-archive that holds image-shaped data (e.g. `red/`, `green/`,
`blue/` for `ColorImage`; PSF model parameter images for archives
that nest them) creates a nested group at its archive path that is
itself a valid OME-NGFF / xarray group, with its own
`ome.multiscales` and `lsst.archive_class` attributes. The top-level
is not special; the same rules apply at every level.

### <a id="chunking-and-sharding-defaults"></a>Chunking and sharding defaults

- Default chunk for a 2-D image: `min(1024, dim)` per axis. For
  `CellCoadd`: `cell_shape`.
- Default shard: 4×4 chunks (i.e. 4096×4096 for plain images, 4×4
  cells for `CellCoadd`) if shard size would be ≥ 1 MiB; otherwise
  no sharding.
- Default codec stack: `bytes -> blosc(zstd, clevel=5,
  shuffle=byte)` for floats; `bytes -> blosc(zstd, clevel=5,
  shuffle=bit)` for integers and masks.
- All defaults are overridable via `ZarrCompressionOptions` per-array
  (keyed by JSON pointer / zarr path).
- `image`, `variance`, and `mask` share the spatial chunk shape;
  the output archive derives `variance` / `mask` chunks from
  `image`'s when not explicitly overridden.

## 4. Error Handling, Edge Cases, Round-Trips

### Round-trip rules

- A zarr file written from an object read from FITS preserves its
  primary-HDU `FitsOpaqueMetadata` at
  `/lsst/opaque_metadata/fits/primary` (1-D `uint8` array of
  JSON-encoded astropy `Header`). Reading the zarr back attaches an
  equivalent `FitsOpaqueMetadata` to the deserialized object so a
  subsequent FITS write preserves the original cards.
- Any `lsst.*` attributes the archive does not recognise are
  preserved verbatim and re-emitted on write of an unchanged tree
  (forward compatibility).

### WCS validation: simplified-affine residual check

When emitting OME `coordinateTransformations` for a multiscale
dataset, the layout layer:

1. Extracts the linear / affine portion of the AST `FrameSet`'s
   pixel-to-sky mapping as a 3×3 affine block.
2. Samples residuals on an **11×11 grid** spanning the image bbox.
   At each grid point, computes pixel→sky via both the full AST
   `FrameSet` and the simplified affine, takes the great-circle
   separation, and divides by the pixel scale to get a
   pixel-equivalent residual.
3. If `max_residual > 1.0 pixel`, **drops the
   `coordinateTransformations` block** for the dataset (emits the
   unit scale `[1.0, 1.0]` only) and sets
   `lsst.wcs_simplified_dropped: true` on the root group, recording
   the observed max residual under `lsst.wcs_simplified_max_residual_pixels`.

Readers always reconstruct the projection from `wcs_ast` regardless
of whether the affine block was emitted or dropped — the OME affine
is purely an external-tool affordance.

### Error taxonomy

Extends existing `serialization.ArchiveReadError`:

- `ArchiveReadError("File has no zarr.json")` for missing root
  metadata.
- `ArchiveReadError("File is not an LSST zarr archive")` when
  `lsst.archive_class` is missing.
- `ArchiveReadError(f"Unsupported lsst:version {N}")` for
  forward-incompatible schema versions.
- `ArchiveReadError(f"Mask has {N} planes; v1 supports up to 64. "
  f"3-D fallback is a follow-up.")` on write of a `>64`-plane Mask.
- `ArchiveReadError("On-disk mask schema does not match requested "
  "schema: ...")` for read-time schema mismatches; both schemas are
  attached, identical to NDF.
- `InvalidParameterError` for unknown `read()` kwargs.
- `InvalidComponentError` for `deserialize_component` on unknown
  component names.
- Validation failures from `model_validate_json` propagate as
  `ArchiveReadError`.

### Mode and atomicity

- Write opens the store in create-only mode (refuses to overwrite an
  existing zarr root, mirroring FITS/NDF).
- For `LocalStore`, a partial failure leaves a partial directory —
  same risk profile as NDF write failures. Document this and
  recommend writing to a temp `ResourcePath` then renaming.
- `ZipStore` writes are atomic (the file is not valid until the
  central directory is written), so failures leave no garbage.

### Chunk-aligned subset reads (lazy invariant)

- `get_array(model, slices=...)` passes `slices` straight to the
  backing `zarr.Array` handle. Zarr handles chunk boundary
  alignment internally; only chunks intersecting the slice are
  fetched.
- For 2-D mask reads (the v1 layout), spatial slices apply as on
  the image; there is no plane-axis to consider.
- A `_CountingStore`-based regression test asserts that a
  single-chunk subset of a 16×16 / chunks=(4,4) array touches
  strictly fewer chunk reads than a full read. This is the load-
  bearing test for cloud-friendly subsetting.

### Mask schema mismatches

If a `Mask` is read where the on-disk plane definitions differ from
the in-memory schema being requested, raise `ArchiveReadError`
with both schemas attached, identical to the NDF backend.

### Empty / minimal cases

- `Image` with no projection: omit `wcs_ast`; the OME multiscale's
  `coordinateTransformations` is the unit scale `[1.0, 1.0]`. The
  `tree` JSON document is just an `ImageSerializationModel` with
  no `projection` field.
- `Image` plus metadata only: as above; `metadata` lives in the
  JSON tree.

### Forward compatibility

- `lsst.version` is an integer; readers refuse versions newer than
  they understand.
- Unknown `lsst.*` keys at any level are preserved verbatim through
  the IR (`ZarrAttributes.load` keeps them; `dump` re-emits them).
  This buys partial-knowledge round-trips without losing extension
  data.

## 5. Testing Strategy and Rollout

### Test layout

Mirrors the NDF pattern (`tests/test_ndf_*.py`):

- `tests/test_zarr_common.py` — `_common.py` constants, path
  helpers, `ZarrCompressionOptions` dataclass.
- `tests/test_zarr_model.py` — IR types in isolation: `ZarrDocument`
  round-trip via `from_zarr` / `to_zarr` against an in-memory
  store, attribute schema validation. Lazy invariant on
  `ZarrArray.from_zarr`.
- `tests/test_zarr_layout.py` — `_layout.py` rules: which axes for
  which archive class, CF flag-attrs construction for masks,
  affine-residual validator (synthetic linear WCS passes; synthetic
  high-distortion WCS triggers the drop), chunk derivation
  (including `cell_shape` alignment).
- `tests/test_zarr_store.py` — URI dispatch (`LocalStore` /
  `ZipStore` / `FsspecStore`), create-only refusal.
- `tests/test_zarr_output_archive.py` — write paths for every
  supported archive class (`Image`, `Mask`, `MaskedImage`,
  `VisitImage`, `ColorImage`, `CellCoadd`), verifying the on-disk
  layout matches the spec by inspecting the IR.
- `tests/test_zarr_input_archive.py` — read paths and `slices=`
  subset reads, `_CountingStore` lazy-invariant assertion, error
  taxonomy tests, opaque-metadata round-trips.
- `tests/test_zarr_round_trip.py` — full write→read round-trips for
  every type, plus FITS↔Zarr cross-format round-trips for the
  types that already do FITS↔NDF round-trips.
- `tests/test_zarr_xarray_interop.py` — `xr.open_zarr(path)` returns
  a `Dataset` with `image` / `variance` / `mask` data variables
  sharing `(y, x)` dims; CF flag attributes survive on the mask
  variable. Skipped if `xarray` is not installed.
- `tests/test_zarr_ome_compliance.py` — *if* `ngff-validator` (or
  equivalent) can be installed in CI, run it against representative
  outputs to catch OME-Zarr spec drift. Skipped if the tool is
  unavailable.
- `tests/test_zarr_external_reader.py` — sanity-check that the
  `ome-zarr` Python tooling can open our files and read the science
  array (not LSST extensions). Skipped if `ome-zarr` is not
  installed.

### CI / dev requirements

Add `zarr >= 3.0` to the optional test dependency set so tests run
automatically. The package metadata adds `[zarr]` extra to the
user-facing extras.

### Rollout plan

Scoped into separate tickets/PRs to keep review tractable:

1. Skeleton + `_common.py` + `_model.py` IR + tests for the IR
   alone. No write/read yet.
2. `_store.py` + `_layout.py` (axes, chunks, affine validator) +
   `ZarrOutputArchive` + write helper. Cover `Image`,
   `MaskedImage`, `VisitImage` only. Output-side tests, including
   CF flag-attrs assertions on the mask group and the affine-
   residual validator behaviour.
3. `ZarrInputArchive` + read helper + `slices=` subset reads (with
   `_CountingStore` regression test) + error taxonomy. Input-side
   tests + round-trip for the types in step 2.
4. `ColorImage` (recursive composition of three `Image` sub-archives)
   + `CellCoadd` (cell-aligned chunks + 4-D PSF). Round-trip tests.
5. Cross-format round-trips (FITS ↔ Zarr opaque metadata
   round-trip). Optional `ome-zarr` external-reader sanity test.
   `xarray` interop test.
6. Documentation: module docstring (mirroring the FITS/NDF module
   docstrings) describing the layout, plus a changelog entry.

## 6. Follow-Up Work (Out of Scope)

Captured here so they are not lost; each is to be tracked as its
own ticket once the initial backend lands.

- **NGFF RFC-5 nonlinear coordinate transformations.** Replace the
  affine-only OME block with a real `sequence(affine, projection,
  ...)` block and treat it as authoritative; `wcs_ast` becomes an
  optional fallback rather than the source of truth. This is high-
  interest because tangent-plane pixel-to-sky transformations
  (CellCoadd) and polynomial corrections (VisitImage TAN-SIP)
  currently round-trip only through the AST string; richer OME
  support would expose them to external tools. **This work is
  blocked on writing an AST JSON channel** that serializes a
  `FrameSet` to and from RFC-5 transformation JSON — this is a
  non-trivial piece of work in its own right and is recorded as a
  tracked dependency with no v1 timeline.
- **3-D mask fallback for `>64`-plane masks.** Adds a per-class
  layout switch: 2-D packed for ≤64 planes (CF-compliant), 3-D
  `(plane_byte, y, x)` for `>64` (CF-extension annotations). v1
  raises on write for `>64`.
- **Lazy / dask-friendly read API** (`read_lazy()` returning open
  zarr arrays / `xr.Dataset` for downstream dask integration).
- **Multiscale pyramid generation** (level 1, 2, … coarsenings) for
  visualization tools.
- **`zarr.consolidated_metadata` extension** to reduce object-list
  calls on cloud stores.
- **Stacked OME view for `ColorImage`.** A future need for a single
  `(3, Y, X)` OME-readable array could be met by writing a stacked
  view alongside the per-channel sub-archives. v1 does not because
  of the no-byte-duplication rule; the per-channel sub-archives are
  themselves valid OME images.
- **NCZarr / NetCDF interop.** Unidata's NCZarr layers a NetCDF data
  model on top of Zarr, unlocking native reads via `libnetcdf` and
  the downstream R / Fortran / MATLAB / IDL ecosystems. v1 is
  already partially compatible because `_ARRAY_DIMENSIONS` (xarray)
  is the same dimension-naming convention NCZarr uses. Full
  compliance is **purely additive**: add `_NCZARR_GROUP` and
  `_NCZARR_ARRAY` attribute markers (no layout change, no extra
  bytes), and optionally write 1-D `y` / `x` coordinate variables
  so the file is self-describing as a NetCDF dataset. Held out of
  v1 because NCZarr's zarr-v3 mapping is still evolving and we'd
  rather pin against a stable revision; the upgrade requires no
  migration of existing files when we adopt it.

# NDF archive subpackage — design

**Ticket:** DM-54817
**Date:** 2026-04-29
**Status:** Draft, pending implementation plan.

## Goal

Add a third serialization backend to `lsst.images`, sibling to the existing
`fits/` and `json/` subpackages, that writes and reads HDF5 files conforming to
the **Starlink HDS data structure on HDF5** mapping
([arxiv:1502.04029](https://arxiv.org/pdf/1502.04029)) and the
**NDF data model** ([arxiv:1410.7513](https://arxiv.org/pdf/1410.7513)). Files
produced by the new backend must be readable by Starlink tools (KAPPA,
`hdstrace`, etc.) as ordinary NDF files, and must round-trip the
`lsst.images` data model.

The backend uses pure `h5py` plus the existing `starlink-pyast` dependency.
It does not depend on `pyndf`, `libndf`, or `libhds` (the `pyndf` wrapper is
not actively maintained).

## Scope (DM-54817)

- **Top-level types supported:** `Image`, `MaskedImage`, `VisitImage`.
- **Read scope:** files produced by this writer; plus best-effort ingest of
  Starlink-generated NDFs that contain only `_REAL`/`_DOUBLE` arrays in the
  recognised components (`DATA_ARRAY`, `VARIANCE`, `QUALITY`, `WCS`,
  `MORE.FITS`). `AXIS`, `HISTORY`, `_LOGICAL` primitives, and unfamiliar
  `MORE.*` extensions are warned-about-and-dropped.
- **Out of scope** (explicitly deferred): `ColorImage`, `GeneralizedImage`,
  multi-NDF HDS containers, compressed/scaled/delta NDF array variants on
  write, fsspec-direct h5py reads, preservation of unrecognised HDS
  components for round-trip, migration of `MORE/LSST/<NAME>` blobs from JSON
  to typed HDS structures.

## Module layout

```
python/lsst/images/ndf/
  __init__.py            # public re-exports + module docstring
  _hds.py                # HDS-on-HDF5 helpers (private; format-only)
  _common.py             # small shared types (no NdfOpaqueMetadata; the
                         # archive reuses FitsOpaqueMetadata for FITS headers)
  _output_archive.py     # NdfOutputArchive : implements OutputArchive ABC
  _input_archive.py      # NdfInputArchive  : implements InputArchive ABC
  formatters.py          # NdfFormatter : ResourcePath-based load/dump
```

Two layers:

- **`_hds.py`** knows only the HDS-on-HDF5 conventions (per the
  canonical `hds-v5` library): structures as h5py groups with `CLASS`
  attributes, primitives as bare h5py datasets (HDS type inferred from
  HDF5 dtype), dimension reversal between NDF (Fortran) and HDF5 (C),
  `|S<N>`-encoded `_CHAR*N` arrays, the `HDS_ROOT_NAME` marker on the
  file root. No knowledge of NDF semantics, `lsst.images`, or pydantic.
- **`_output_archive.py` / `_input_archive.py`** know NDF semantics and
  LSST-side conventions (which structure means `DATA_ARRAY`, where mask
  plane names go, the `MORE/LSST/*` hoisting policy). They call `_hds`
  exclusively for on-disk reads/writes.

`formatters.py` mirrors `fits/formatters.py` and `json/formatters.py`: a
`Formatter` subclass that opens a `lsst.resources.ResourcePath`,
materialises a local file (downloading if remote), and delegates to the
archive classes.

## `_hds.py` — HDS-on-HDF5 layer

### Data model summary

HDS has two object kinds: **structures** (containers; named, typed, with
zero-or-more named children) and **primitives** (typed N-dimensional
arrays of a fixed primitive type).

### HDF5 mapping

Conventions follow the canonical Starlink `hds-v5` library
(`reference/hds-v5/dat1.h:137-140` and `dat1New.c:202`):

- **Structure** → `h5py.Group` with attribute `CLASS` ∈
  `{"NDF", "WCS", "ARRAY", "EXT", "QUALITY", "STRUCT", ...}`. Children
  named by HDS component name in uppercase.
- **Arrays of structures** (deferred from v1; we only handle scalar
  structures): additionally carry `HDS_STRUCTURE_DIMS` and store
  per-element groups under the `ARRAY_OF_STRUCTURES_CELL` prefix.
- **File root**: when the root represents a top-level HDS structure, the
  root group carries an `HDS_ROOT_NAME` attribute (HDS object name) plus
  the same `CLASS` attribute as any other structure. We don't support
  the `HDS_ROOT_IS_PRIMITIVE` case (root group holding a primitive).
- **Primitive** → bare `h5py.Dataset`, no HDS-specific attributes. The
  HDS type is inferred from the HDF5 dtype:
  `H5T_NATIVE_FLOAT` → `_REAL`, `H5T_NATIVE_DOUBLE` → `_DOUBLE`,
  `H5T_NATIVE_UCHAR` → `_UBYTE`, `H5T_NATIVE_INT` → `_INTEGER`,
  `H5T_NATIVE_SHORT` → `_WORD`, `|S<N>` → `_CHAR*<N>`. (HDS `_LOGICAL`
  is not in the supported set; on read we warn-and-drop.)
- **Dimension order**: NDF/Fortran dims `(N1, N2, …, Nk)` map to HDF5
  dims `(Nk, …, N2, N1)`. Callers always pass and receive C-order numpy
  arrays; the byte stream on disk matches Fortran storage by virtue of
  the reversal.
- **Legacy compatibility on read**: `_hds.open_structure` accepts the
  older `HDSTYPE` attribute as a fallback when `CLASS` is absent, so
  files produced by pre-canonical HDS variants can still be inspected.
- **Top level**: file root group has `CLASS="NDF"` and
  `HDS_ROOT_NAME=<some-name>` for the simple
  `Image`/`MaskedImage`/`VisitImage` cases.

### Type system

| HDS type   | numpy dtype             | Read | Write |
|------------|-------------------------|------|-------|
| `_REAL`    | `float32`               | yes  | yes   |
| `_DOUBLE`  | `float64`               | yes  | yes   |
| `_UBYTE`   | `uint8`                 | yes  | yes   |
| `_INTEGER` | `int32`                 | yes  | yes   |
| `_WORD`    | `int16`                 | yes  | no    |
| `_CHAR*N`  | `|S{N}` (fixed width)   | yes  | yes   |
| `_LOGICAL` | n/a                     | warn-and-drop | no |

Anything outside this set on read raises `ArchiveReadError` from a
recognised component, or is logged-and-dropped if from an unrecognised
component.

### `_CHAR*N` convention

FITS cards and AST WCS text dumps are stored as 1D `_CHAR*N` (typically
`_CHAR*80`) arrays. `_hds.write_char_array(...)` pads/truncates each
string to fixed width with trailing spaces (HDS convention) and writes a
1D dataset with dtype `|S{width}`; no HDS-specific attributes are
written, since the HDF5 dtype encodes the type directly. The reader
strips trailing spaces.

### Public surface

```python
# Canonical attribute names (per hds-v5 dat1.h)
ATTR_CLASS = "CLASS"
ATTR_STRUCTURE_DIMS = "HDS_STRUCTURE_DIMS"
ATTR_ROOT_NAME = "HDS_ROOT_NAME"

# Structures
def create_structure(parent: h5py.Group, name: str, hds_type: str) -> h5py.Group: ...
def open_structure(parent: h5py.Group, name: str) -> tuple[h5py.Group, str]: ...
def set_root_name(file: h5py.File, hds_name: str, hds_type: str) -> None: ...
def iter_children(group: h5py.Group) -> Iterator[tuple[str, h5py.Group | h5py.Dataset]]: ...

# Primitives (no HDS-specific attributes; HDS type inferred from HDF5 dtype)
def write_array(parent: h5py.Group, name: str, data: np.ndarray, *,
                compression: str | None = None) -> h5py.Dataset: ...
def read_array(dataset: h5py.Dataset) -> np.ndarray: ...

def write_char_array(parent: h5py.Group, name: str,
                     lines: Sequence[str], *, width: int = 80) -> h5py.Dataset: ...
def read_char_array(dataset: h5py.Dataset) -> list[str]: ...

# Type helpers
def hds_type_for_dtype(dtype: np.dtype) -> str: ...
HDS_TO_NUMPY: Mapping[str, np.dtype]
NUMPY_TO_HDS: Mapping[np.dtype, str]
```

### Out of scope at the `_hds` layer

- Arrays of structures (HDS allows them; no NDF component we care about
  uses them for our supported types).
- `DELTA`/`SCALED`/`SPARSE` array variants on write. The reader accepts
  the bare-primitive and `PRIMITIVE`/`ARRAY` wrapper forms; other variants
  raise.

## NDF component layout

### `Image`

```
/                                  CLASS="NDF"
  DATA_ARRAY/                      CLASS="ARRAY"
    DATA                           _REAL|_DOUBLE, NDF shape (NX, NY)
    ORIGIN                         _INTEGER, shape (2,)   — bbox lower bounds
  WCS/                             CLASS="WCS"
    DATA                           _CHAR*N — AST FrameSet text dump
  MORE/                            CLASS="EXT"
    FITS                           _CHAR*80 — opaque FITS cards (if any)
    LSST/                          CLASS="EXT"
      JSON                         _CHAR*N — main Pydantic tree as JSON
      <NAME>...                    hoisted blobs (none for plain Image)
```

### `MaskedImage` — compatible mask (`uint8` dtype, ≤8 planes)

Add to the `Image` layout:

```
  VARIANCE/                        CLASS="ARRAY"
    DATA                           _REAL|_DOUBLE
    ORIGIN                         _INTEGER, (2,)
  QUALITY/                         CLASS="QUALITY"
    QUALITY                        _UBYTE, NDF shape (NX, NY)
    BADBITS                        _UBYTE, scalar — 0xFF means all defined
                                   plane bits flag bad pixels (tunable later)
```

Plane-name metadata (the `MaskSchema`) is small enough to ride inline in
the main JSON tree at `MORE/LSST/JSON`, the same way the FITS and JSON
archives store it. No separate Starlink-native `MORE/IRQ` extension is
written in v1; full IRQ-extension support (`MORE/IRQ/QUAL`
array-of-structures readable by KAPPA `setqual`/`showqual`) is a
deferred follow-up that requires extending `_hds.py` with
array-of-structures helpers.

### `MaskedImage` — incompatible mask (wider dtype OR >8 planes)

Drop `/QUALITY`; the full mask array goes into `MORE/LSST/MASK`:

```
  MORE/LSST/
    JSON                           main tree (with JSON-Pointer to MASK)
    MASK/                          CLASS="STRUCT"
      DATA                         raw mask array (possibly 3D, dtype as written)
```

The `MaskSchema` (plane names, bit indices, descriptions, dtype) rides
inline in the main JSON tree just as in the compatible case; only the
mask array is hoisted, because it's an array.

### `VisitImage`

`MaskedImage` layout (compatible or incompatible) plus hoisted PSF and
summary stats:

```
  MORE/LSST/
    JSON                           main tree (with JSON-Pointer references)
    PSF                            _CHAR*N JSON for parametric PSFs;
                                   structure with hoisted arrays for richer PSFs
    SUMMARY_STATS                  _CHAR*N JSON
    ...                            any other top-level component the type's
                                   serialize() registered
```

### Routing rules — writer

The NDF backend implements the abstract `OutputArchive` methods with
these rules:

| Call                                                | Top level (path empty)                                                                 | Nested (path non-empty)                            |
|-----------------------------------------------------|---------------------------------------------------------------------------------------|----------------------------------------------------|
| `add_array(arr, name="image")`                      | `/DATA_ARRAY/DATA` + `/DATA_ARRAY/ORIGIN`                                              | hoist                                              |
| `add_array(arr, name="variance")`                   | `/VARIANCE/DATA` + `ORIGIN`                                                            | hoist                                              |
| `add_array(arr, name="mask")`                       | `/QUALITY/QUALITY` (compatible) **or** `/MORE/LSST/MASK/DATA` (incompatible)           | hoist                                              |
| `serialize_frame_set(name="projection", ...)`       | `/WCS/DATA`                                                                            | hoist                                              |
| `serialize_direct(name=any, ...)`                   | inline into the main JSON tree                                                         | inline                                             |
| `serialize_pointer(name=any, ...)`                  | hoist: write subtree to `/MORE/LSST/<UPPER_PATH>`, return JSON Pointer                 | hoist                                              |
| `add_table(...)`                                    | hoist to `/MORE/LSST/<UPPER_PATH>` (JSON for v1)                                       | hoist                                              |
| Opaque FITS cards (from `FitsOpaqueMetadata`)       | `/MORE/FITS`; not part of the JSON tree                                                | n/a                                                |

Plain English:

- **Recognised top-level names go to native NDF slots** (`DATA_ARRAY`,
  `VARIANCE`, `QUALITY`, `WCS`). The `name` strings used by
  `Image`/`MaskedImage`/`VisitImage`'s `serialize()` methods determine
  the routing; if those methods pass different strings, the table
  substitutes whatever they actually use without architectural change.
- **Anything else hoists.** Hoisted destination =
  `/MORE/LSST/<UPPER_PATH>`, where `<UPPER_PATH>` is the JSON-Pointer
  path uppercased and `/` replaced with `_` (e.g.
  `/psf/coefficients` → `PSF_COEFFICIENTS`). Mirrors the FITS archive's
  `EXTNAME` convention.
- **Main JSON tree** dumped to `/MORE/LSST/JSON` as `_CHAR*N` line-
  wrapped JSON.
- **Opaque FITS metadata** reuses the existing `FitsOpaqueMetadata`
  type (no parallel `NdfOpaqueMetadata` for headers); the NDF archive
  reads `MORE/FITS` into its `headers[ExtensionKey()]` and writes them
  back.

### Routing rules — reader

1. Open the root group; verify `CLASS="NDF"` (or fall back to
   best-effort with a warning if the top is some other HDS type).
2. Read `DATA_ARRAY`, `VARIANCE` (if present), `QUALITY` (if present),
   `WCS`, `MORE/FITS` in that order.
3. If `MORE/LSST/JSON` present → parse main tree, materialise hoisted
   blobs by JSON Pointer path → instantiate the lsst-images type the
   Pydantic discriminator identifies.
4. If `MORE/LSST/JSON` absent → auto-detect: `Image` if no
   `QUALITY`/`VARIANCE`, else `MaskedImage` with a default mask schema
   (`BAD=bit 0`). If `VARIANCE` is present without `QUALITY`, the mask
   is an empty (all-zero) `uint8` array with the default schema.
   Reading Starlink-generated `MORE/IRQ`-named bits is a follow-up
   that arrives with full IRQ write support.
5. Unrecognised components (`HISTORY`, `AXIS`, `LABEL`, custom `MORE.*`
   not in `{FITS, IRQ, LSST}`, `_LOGICAL`-typed primitives) → log a
   warning once per unique kind per file; skip.

## Integration points

### WCS round-trip via `starlink-pyast`

- **Write**: serialize the `FrameSet` to a list of text lines using
  `starlink.Ast.Channel`; write via `_hds.write_char_array(wcs_group,
  "DATA", lines, width=max(80, max_line_length))`. Set `wcs_group`
  `CLASS="WCS"`.
- **Read**: `_hds.read_char_array(...)` → feed lines to `Channel` reader
  → `FrameSet` → wrap in `Projection` via `_transforms/_ast.py`.

### bbox ↔ ORIGIN

`Image.bbox = Box(x_min, y_min, width, height)`. In numpy C-order the
array is `(height, width)`. NDF stores:

- `DATA` primitive: HDF5 shape `(height, width)`, type
  `_REAL`/`_DOUBLE` matching the array dtype.
- `ORIGIN` primitive: 1D `_INTEGER`, length 2, values `[x_min, y_min]`
  in NDF/Fortran axis order.

On read: `bbox = Box(origin[0], origin[1], width=array.shape[1],
height=array.shape[0])`. Upper bounds are computed by NDF tools as
`LBND + dim - 1`; we don't store `UBND`.

### Mask plane name storage

For v1, plane names live in `MORE/LSST/MASK_PLANES` as a single
`_CHAR*N` JSON document holding the serialized `MaskSchema`. The
reader handles both this and the full Starlink IRQ form (so
Starlink-generated NDFs with proper `MORE/IRQ/QUAL` arrays can be
ingested as far as named bits are concerned).

Full IRQ-extension write support (`MORE/IRQ/QUAL` as an HDS
array-of-structures with per-plane `NAME`/`BIT`/`COMMENT`/`FIXED`/
`VALID`/`MASK` fields, plus IRQ-extension housekeeping) is a
follow-up. It requires extending `_hds.py` with helpers for arrays of
structures, which v1 deliberately doesn't include. Once added, KAPPA
`setqual`/`showqual` can interpret named bits without round-tripping
through us.

### File I/O & `ResourcePath`

`NdfFormatter` accepts a `ResourcePath`. Local path → `h5py.File(...)`
directly. Remote → materialise locally via `ResourcePath.as_local()`,
operate, and (on write) upload back. fsspec-direct h5py reads are a
follow-up.

### Top-level type discrimination

`MORE/LSST/JSON` carries the same Pydantic discriminator the FITS and
JSON archives produce (e.g. an `image_type` field). Reading it picks
the right in-memory type. Auto-detect (Image/MaskedImage) only fires
when `MORE/LSST/JSON` is absent.

### Constructor / `compression_options`

`NdfOutputArchive(file: h5py.File, compression_options: Any = None,
opaque_metadata: FitsOpaqueMetadata | None = None)` — argument shape
parallel to `FitsOutputArchive`. For v1, `compression_options` is a
dict like `{"compression": "gzip", "compression_opts": 4}` passed
through to `h5py.create_dataset`. Quantized lossy compression
(FITS `Q` analogue) is a follow-up.

## Error handling

### Write side

- Unsupported top-level type → `NotImplementedError` naming the type
  and pointing at FITS/JSON archives as alternatives.
- `add_array` with a dtype outside the supported HDS primitive set →
  `NotImplementedError` (current writer bug).
- HDF5 errors (locked file, permission, disk full) propagate from h5py.
- Existing-file behaviour follows the caller's mode flag, mirroring
  `FitsFormatter`.

### Read side

Uses the existing `ArchiveReadError` from `serialization/_common.py`.

- Not an HDF5 file → `ArchiveReadError("not an HDF5 file")`.
- Root group not a recognisable HDS structure → `ArchiveReadError`.
- Recognised component malformed (wrong `CLASS`, dtype, dimensionality) →
  `ArchiveReadError` with path and expectation.
- `MORE/LSST/JSON` present but not parseable / Pydantic discriminator
  unknown → `ArchiveReadError`.
- Unrecognised components → `logging.getLogger(__name__).warning(...)`
  once per unique kind per file; do not raise.
- Missing both `MORE/LSST/JSON` and core image components → 
  `ArchiveReadError("file is HDS but contains no image data")`.

The narrow read scope is enforced at `_hds.read_array`, so the archive
layer does not need defensive type checks at every call site.

## Testing strategy

Following `python/lsst/images/tests/_roundtrip.py` and existing
per-type test files.

### `_hds.py` unit tests (`tests/test_ndf_hds.py`, new)

- Round-trip primitives of each supported type, scalar and N-D, with
  dimension reversal verified.
- `_CHAR*N` round-trip with padding/stripping.
- Structure creation, child iteration, nested structures.
- `set_root_name` writes both `HDS_ROOT_NAME` and `CLASS` on the file
  root.
- Legacy-fallback `open_structure`: a group with the older `HDSTYPE`
  attribute (no `CLASS`) is still recognised as a structure.
- A canonical-format Starlink NDF fixture (added under `tests/data/`
  once available) read with `_hds`: traverse the tree and assert
  expected `CLASS` values, dtypes, and shapes for the primary
  components. Pins format conventions to a real Starlink-generated
  file independently of any NDF-archive logic.

### Round-trip tests

Extend `python/lsst/images/tests/_roundtrip.py`. The recently
factored-out non-FITS base class
(commit `56d4c10`) is reused: add `RoundtripNdf` as a sibling and
parametrise over `Image`, `MaskedImage` (compatible + incompatible
mask schemas), and `VisitImage`. Equality covers array data, variance,
mask (with plane name fidelity), WCS (`FrameSet` semantic equality via
AST), opaque FITS metadata, bbox/origin, top-level type identity.

### Layout sanity tests (`tests/test_ndf_layout.py`, new)

Open files written by `NdfOutputArchive` with raw h5py and assert the
on-disk structure matches the layout section exactly: group paths,
`CLASS` attributes, primitive HDF5 dtypes, dimension reversal.
Catches drift if the writer stops following HDS conventions.

### Read-only ingest test (`tests/test_ndf_starlink_ingest.py`, new)

Read `example.sdf` via `NdfInputArchive` → assert the result is an
`Image` (auto-detect path: no `MORE/LSST/JSON`, no `QUALITY`/
`VARIANCE`). Verify FITS opaque metadata recovered from `MORE/FITS`,
projection recovered from `WCS`, image shape and dtype.

### Cross-archive consistency

For each test object, write via FITS, read; write via NDF, read;
assert recovered objects are equal modulo backend-specific opaque
metadata. Catches divergence between archives.

### Optional Starlink interop check

`@pytest.mark.starlink` marker that, when `hdstrace`/`kappa` is on
`PATH`, runs `hdstrace` on a written file and asserts exit-code 0
with recognisable output. Not part of CI.

## Deferred to later tickets

- Compression options (gzip on h5py datasets first; quantized lossy
  follow-up).
- `ColorImage`, `GeneralizedImage`.
- Synthesised QUALITY for incompatible masks (Q5-C).
- Full IRQ-extension write support (`MORE/IRQ/QUAL` array-of-structures
  with named-bit metadata readable by KAPPA `setqual`/`showqual`).
  Requires array-of-structures helpers in `_hds.py`.
- Preservation of unrecognised HDS components for round-trip (Q7-Y).
- fsspec-direct h5py reads.
- Multi-NDF HDS containers (for future `ColorImage` etc.).
- Migration of `MORE/LSST/<NAME>` blobs from JSON to typed HDS
  structures (the eventual binary-tabular plan).

## New dependencies

- `h5py` — added to `requirements.txt` and `pyproject.toml`.

`starlink-pyast`, `numpy`, `pydantic`, and `lsst-resources` are
already present.
